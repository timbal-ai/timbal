"""How does a voice say an acronym, and does it say it the same way every take?

A neural voice decides in each request whether "DNI" is a word or three
letters, and how slowly to spell them. The caller hears an agent that says
"de-ene-i" briskly in one turn and "de... ene... i" in the next.

A transcript cannot tell the takes apart: a speech-to-text model writes "DNI"
however it was said. What differs is how long it takes. Each take puts the
term in a carrier sentence, and the time between the word before it and the
word after it is read from word timestamps (ElevenLabs Scribe). Deepgram
Nova-3 transcribes the same audio as a second opinion on what was said there.

    uv run python benchmarks/voice/tts_acronyms.py DNI deneí
    uv run python benchmarks/voice/tts_acronyms.py NIE níe --model eleven_v4_turbo --takes 8
    uv run python benchmarks/voice/tts_acronyms.py SMS esemese "ese eme ese" --voice <voice id>

The first argument is the term as a model writes it. The others are spoken
forms to compare with it, the candidates for ``VoiceConfig.pronunciations``.
``--out DIR`` writes one WAV per take, to listen to.

Needs ``ELEVENLABS_API_KEY`` (the transcription with word times, and the voice
when ``--provider`` is ``elevenlabs``), ``DEEPGRAM_API_KEY`` and the key of any
other ``--provider``. A missing key exits 2 and names it.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import os
import re
import statistics
import sys
import unicodedata
import wave
from pathlib import Path

import httpx
from dotenv import load_dotenv
from timbal.voice import AudioOutputConfig, VoiceConfig
from timbal.voice.providers import resolve_tts

RATE = 24_000
# A take is "drawn out" from here: twice the usual time of the term read briskly.
SLOW_SECS = 0.84

# (sentence, the word before the term, the word after it)
CARRIERS = {
    "es": [
        ("¿Me dices tu {} por favor?", "tu", "por"),
        ("Necesito el {} del titular.", "el", "del"),
        ("Tu {} acaba en zeta, ¿verdad?", "tu", "acaba"),
    ],
    "en": [
        ("Could you give me your {} please?", "your", "please"),
        ("I need the {} of the account holder.", "the", "of"),
        ("Your {} ends in a zed, right?", "your", "ends"),
    ],
}


def wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def bare(word: str) -> str:
    """Lower case, no accents, no punctuation: "Níe." and "NIE" are the same hearing."""
    folded = "".join(ch for ch in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(ch))
    return re.sub(r"[^\w]", "", folded)


async def synthesize(text: str, args: argparse.Namespace) -> bytes:
    tts = resolve_tts(args.provider)
    await tts.connect(AudioOutputConfig(model=args.model, voice=args.voice, sample_rate=RATE))
    try:
        return b"".join([chunk async for chunk in tts.synthesize(text)])
    finally:
        await tts.close()


async def timed_words(client: httpx.AsyncClient, pcm: bytes, language: str) -> list[dict]:
    for attempt in range(3):
        try:
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                data={
                    "model_id": "scribe_v1",
                    "language_code": language,
                    "tag_audio_events": "false",
                    "timestamps_granularity": "word",
                },
                files={"file": ("take.wav", wav(pcm), "audio/wav")},
                timeout=90,
            )
            if response.status_code == 200:
                return [word for word in response.json().get("words", []) if word.get("type") == "word"]
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.5 * (attempt + 1))
    return []


async def heard(client: httpx.AsyncClient, pcm: bytes, language: str) -> str:
    response = await client.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "language": language, "smart_format": "false", "punctuate": "true"},
        headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": "audio/wav"},
        content=wav(pcm),
        timeout=90,
    )
    if response.status_code != 200:
        return ""
    return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


def gap_secs(words: list[dict], before: str, after: str) -> float | None:
    """Seconds between the end of ``before`` and the start of ``after``."""
    texts = [bare(word["text"]) for word in words]
    try:
        first = texts.index(before)
        last = texts.index(after, first + 1)
    except ValueError:
        return None
    return round(words[last]["start"] - words[first]["end"], 2)


def between(transcript: str, before: str, after: str) -> str | None:
    words = transcript.split()
    texts = [bare(word) for word in words]
    try:
        first = texts.index(before)
        last = texts.index(after, first + 1)
    except ValueError:
        return None
    return " ".join(words[first + 1 : last]).strip(" ,.?¿")


async def take(client, gate, args, form: str, carrier: tuple[str, str, str], out: Path | None, name: str) -> dict:
    sentence, before, after = carrier
    async with gate:
        pcm = await synthesize(sentence.format(form), args)
    if out is not None:
        (out / f"{name}.wav").write_bytes(wav(pcm))
    words, transcript = await asyncio.gather(timed_words(client, pcm, args.language), heard(client, pcm, args.language))
    return {"gap": gap_secs(words, before, after), "heard": between(transcript, before, after)}


async def run(args: argparse.Namespace) -> dict:
    carriers = CARRIERS[args.language]
    out = Path(args.out) if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)
    wanted = bare(args.term)
    report: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        for form in [args.term, *args.forms]:
            takes = await asyncio.gather(
                *(
                    take(client, gate, args, form, carrier, out, f"{bare(form)}_{c}_{n}")
                    for c, carrier in enumerate(carriers)
                    for n in range(args.takes)
                )
            )
            gaps = sorted(t["gap"] for t in takes if t["gap"] is not None)
            report[form] = {
                "takes": len(takes),
                "heard_as_term": sum(1 for t in takes if bare(t["heard"] or "") == wanted),
                "heard": dict(collections.Counter(t["heard"] or "?" for t in takes)),
                "gaps": gaps,
                "slow": sum(1 for gap in gaps if gap >= SLOW_SECS),
            }
            row = report[form]
            spread = f"{statistics.median(gaps):.2f} s ({gaps[0]:.2f}-{gaps[-1]:.2f})" if gaps else "not found"
            print(  # noqa: T201
                f"{form!r:16} heard as {args.term} {row['heard_as_term']}/{row['takes']}   "
                f"time {spread}   {SLOW_SECS} s or more: {row['slow']}/{len(gaps)}",
                flush=True,
            )
    return report


def main() -> None:
    load_dotenv()
    defaults = VoiceConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("term", help="the term as a model writes it: DNI")
    parser.add_argument("forms", nargs="*", help="spoken forms to compare with it: deneí")
    parser.add_argument("--provider", default="elevenlabs")
    parser.add_argument("--model", default="eleven_v3_conversational")
    parser.add_argument("--voice", default=defaults.voice)
    parser.add_argument("--language", default="es", choices=sorted(CARRIERS))
    parser.add_argument("--takes", type=int, default=6, help="takes per carrier sentence (3 sentences)")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--out", help="directory for one WAV per take")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()
    for key in ("ELEVENLABS_API_KEY", "DEEPGRAM_API_KEY"):
        if not os.environ.get(key):
            print(f"{key} is not set", file=sys.stderr)  # noqa: T201
            sys.exit(2)
    report = asyncio.run(run(args))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
