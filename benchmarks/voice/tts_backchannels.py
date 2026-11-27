"""Does a voice say "Mm-hm." as a short listening sound?

``VoiceConfig.backchannel`` plays a short sound in the agent's voice while the
caller is still talking. The sound is synthesized once per call from a written
phrase, and a voice may read that phrase as something else: a question
("¿Ya?"), another word ("Val."), a laugh, or a long take. This measures, for
each candidate phrase and model, through the adapter a session uses:

* **said**: what ElevenLabs Scribe hears (audio events on) is the phrase,
  letters only, accents and case apart, allowing the spellings a transcript
  gives a hum ("Mhm" for "Mm-hm", "Aha" for "Ajá");
* **secs**: how long the sound is once the silence around it is trimmed, as a
  session trims it (``timbal.voice.backchannel.prepare_clip``);
* **kept**: takes a session would keep (not silent, not too long).

    uv run python benchmarks/voice/tts_backchannels.py
    uv run python benchmarks/voice/tts_backchannels.py --languages es --models eleven_v3_conversational --takes 6
    uv run python benchmarks/voice/tts_backchannels.py --provider gemini --models gemini-3.8-flash-tts --voice Fenrir

``--candidates`` adds the phrases that were tried and left out of the defaults.
``--out DIR`` writes one WAV per take, to listen to.

Needs ``ELEVENLABS_API_KEY`` (the transcription, and the voice when
``--provider`` is ``elevenlabs``) and the key of any other ``--provider``.
"""

from __future__ import annotations

import argparse
import asyncio
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
from timbal.voice.backchannel import DEFAULT_BACKCHANNELS, prepare_clip, trim_silence
from timbal.voice.providers import resolve_tts

RATE = 24_000

# Tried and left out of the defaults.
CANDIDATES = {
    "es": ["Ya.", "Claro.", "Mm.", "Sí, sí."],
    "en": ["I see."],
    "ca": ["D'acord.", "Sí, sí."],
}

# How a transcript writes a hum.
SPELLINGS = {
    "mmhm": {"mmhm", "mhm", "mhmm", "mmhmm", "mmm", "mm", "hmm", "mjm"},
    "aja": {"aja", "aha", "ajaa"},
    "aha": {"aha", "aja"},
    "uhhuh": {"uhhuh", "ahha", "aha", "uhuh"},
    "mm": {"mm", "mmm", "hmm", "hm"},
    "okay": {"okay", "ok"},
}
_EVENT = re.compile(r"[\[(][^\])]*[\])]")


def letters(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in folded if ch.isalpha() and not unicodedata.combining(ch))


def said(phrase: str, transcript: str) -> bool:
    """Whether ``transcript`` is ``phrase`` and nothing else: no other word, no question, no labelled sound."""
    if _EVENT.search(transcript) or "?" in transcript:
        return False
    want, got = letters(phrase), letters(transcript)
    return got == want or got in SPELLINGS.get(want, ())


def wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


async def synthesize(text: str, model: str, args: argparse.Namespace) -> bytes:
    tts = resolve_tts(args.provider)
    await tts.connect(AudioOutputConfig(model=model, voice=args.voice, sample_rate=RATE))
    try:
        return b"".join([chunk async for chunk in tts.synthesize(text)])
    finally:
        await tts.close()


async def transcribe(client: httpx.AsyncClient, pcm: bytes, language: str) -> str:
    for attempt in range(3):
        try:
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                data={"model_id": "scribe_v1", "language_code": language, "tag_audio_events": "true"},
                files={"file": ("take.wav", wav(pcm), "audio/wav")},
                timeout=90,
            )
            if response.status_code == 200:
                return response.json().get("text", "").strip()
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.5 * (attempt + 1))
    return ""


async def take(client, gate, args, model: str, language: str, phrase: str, out: Path | None, name: str) -> dict:
    async with gate:
        try:
            pcm = await synthesize(phrase, model, args)
        except Exception as error:  # noqa: BLE001 - a model this account cannot use is a result, not a crash
            return {"phrase": phrase, "error": type(error).__name__}
    if out is not None:
        (out / f"{name}.wav").write_bytes(wav(pcm))
    heard = await transcribe(client, pcm, language)
    return {
        "phrase": phrase,
        "heard": heard,
        "said": said(phrase, heard),
        "secs": round(len(trim_silence(pcm, RATE)) / 2 / RATE, 2),
        "kept": prepare_clip(pcm, RATE) is not None,
    }


async def run(args: argparse.Namespace) -> dict:
    out = Path(args.out) if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)
    report: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        for model in args.models.split(","):
            for language in args.languages.split(","):
                phrases = list(DEFAULT_BACKCHANNELS.get(language, ()))
                if args.candidates:
                    phrases += CANDIDATES.get(language, [])
                for index, phrase in enumerate(phrases):
                    takes = await asyncio.gather(
                        *(
                            take(client, gate, args, model, language, phrase, out, f"{model}_{language}_{index}_{n}")
                            for n in range(args.takes)
                        )
                    )
                    done = [t for t in takes if "error" not in t]
                    secs = [t["secs"] for t in done]
                    row = {
                        "takes": len(done),
                        "errors": len(takes) - len(done),
                        "said": sum(t["said"] for t in done),
                        "kept": sum(t["kept"] for t in done),
                        "secs": [statistics.median(secs), min(secs), max(secs)] if secs else None,
                        "other": sorted({t["heard"] for t in done if not t["said"]}),
                    }
                    report[f"{model}/{language}/{phrase}"] = {**row, "all": takes}
                    print(  # noqa: T201
                        f"{model:26} {language} {phrase!r:11} said {row['said']}/{row['takes']}"
                        f"  kept {row['kept']}/{row['takes']}"
                        + (f"  {row['secs'][0]:.2f} s ({row['secs'][1]:.2f}-{row['secs'][2]:.2f})" if secs else "")
                        + (f"  heard instead: {row['other']}" if row["other"] else "")
                        + (f"  errors {row['errors']}" if row["errors"] else ""),
                        flush=True,
                    )
    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="elevenlabs")
    parser.add_argument(
        "--models", default="eleven_v3_conversational,eleven_v4_turbo,eleven_flash_v2_5,eleven_multilingual_v2"
    )
    parser.add_argument("--voice", default=VoiceConfig().voice)
    parser.add_argument("--languages", default="es,ca,en")
    parser.add_argument("--takes", type=int, default=6, help="takes of each phrase")
    parser.add_argument("--candidates", action="store_true", help="also the phrases left out of the defaults")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--out", help="directory for one WAV per take")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ELEVENLABS_API_KEY is not set", file=sys.stderr)  # noqa: T201
        raise SystemExit(2)
    report = asyncio.run(run(args))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
