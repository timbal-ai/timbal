"""Does a voice perform ``[laughs]`` as a laugh, or read it aloud?

``VoiceConfig.audio_tags`` lets the model write a laugh or a sigh into its
reply. That is only safe with a voice that turns the mark into the sound: the
others say the word ("Chuckles. Claro que sí..."), which is worse than no
laugh at all. This measures which is which, through the adapter a session
uses.

Each take is a short sentence with one mark in it. Its audio is transcribed by
ElevenLabs Scribe with audio events on, which labels a sound it hears
(``[risa]``, ``[suspiro]``) apart from the words. A take counts as:

* **sound**: the transcript has a labelled sound;
* **words**: the transcript has words the sentence does not (the mark read
  aloud, or mangled: "Zais", "laps--");
* **neither**: the sentence alone, the mark silently dropped.

The same sentences without the mark are the control: a labelled sound there
is the transcriber's noise, not a laugh.

    uv run python benchmarks/voice/tts_audio_tags.py
    uv run python benchmarks/voice/tts_audio_tags.py --models eleven_flash_v2_5,eleven_multilingual_v2
    uv run python benchmarks/voice/tts_audio_tags.py --provider gemini --models gemini-3.8-flash-tts --voice Fenrir

``--out DIR`` writes one WAV per take, to listen to.

Needs ``ELEVENLABS_API_KEY`` (the transcription, and the voice when
``--provider`` is ``elevenlabs``) and the key of any other ``--provider``.
A missing key exits 2 and names it.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import io
import json
import os
import re
import sys
import wave
from pathlib import Path

import httpx
from dotenv import load_dotenv
from timbal.voice import AudioOutputConfig, VoiceConfig
from timbal.voice.audio_tags import strip_audio_tags
from timbal.voice.providers import resolve_tts

RATE = 24_000

SENTENCES = [
    "[laughs] Claro que sí, ahora mismo te lo miro.",
    "[chuckles] Claro que sí, ahora mismo te lo miro.",
    "[sighs] Vaya, siento mucho lo que ha pasado con tu factura.",
    "Uy, [laughs] perdona, me he liado yo sola.",
    "[clears throat] Perdona. Te decía que el cargo es del lunes.",
    "[exhales] A ver, déjame que lo compruebe otra vez.",
]

_EVENT = re.compile(r"[\[(][^\])]*[\])]")


def wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def words(text: str) -> list[str]:
    return [word for word in (re.sub(r"[^\w]", "", part.lower()) for part in text.split()) if word]


def added_words(sentence: str, transcript: str) -> list[str]:
    """Words of ``transcript`` that are not a hearing of a word of ``sentence``."""
    said, got = words(sentence), words(transcript)
    added: list[str] = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=said, b=got, autojunk=False).get_opcodes():
        if op == "insert":
            added.extend(got[b0:b1])
        elif op == "replace" and (b1 - b0) > (a1 - a0):
            # A word misheard is one for one. More words than were written were said.
            added.extend(got[b0 + (a1 - a0) : b1])
    return added


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


async def take(client, gate, args, model: str, text: str, out: Path | None, name: str) -> dict:
    async with gate:
        try:
            pcm = await synthesize(text, model, args)
        except Exception as error:  # noqa: BLE001 - a model this account cannot use is a result, not a crash
            return {"text": text, "error": type(error).__name__}
    if out is not None:
        (out / f"{name}.wav").write_bytes(wav(pcm))
    transcript = await transcribe(client, pcm, args.language)
    sentence = strip_audio_tags(text)
    return {
        "text": text,
        "heard": transcript,
        "sound": bool(_EVENT.search(transcript)),
        "added": added_words(sentence, _EVENT.sub(" ", transcript)),
    }


def tally(takes: list[dict]) -> dict:
    done = [t for t in takes if "error" not in t]
    return {
        "takes": len(done),
        "errors": len(takes) - len(done),
        "sound": sum(1 for t in done if t["sound"]),
        "words": sum(1 for t in done if t["added"]),
        "neither": sum(1 for t in done if not t["sound"] and not t["added"]),
    }


async def run(args: argparse.Namespace) -> dict:
    out = Path(args.out) if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)
    report: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        for model in args.models.split(","):
            marked = await asyncio.gather(
                *(
                    take(client, gate, args, model, text, out, f"{model}_{i}_{n}")
                    for i, text in enumerate(SENTENCES)
                    for n in range(args.takes)
                )
            )
            plain = await asyncio.gather(
                *(
                    take(client, gate, args, model, strip_audio_tags(text), out, f"{model}_{i}_plain")
                    for i, text in enumerate(SENTENCES)
                )
            )
            report[model] = {"marked": tally(marked), "control": tally(plain), "takes": marked + plain}
            row, control = report[model]["marked"], report[model]["control"]
            print(  # noqa: T201
                f"{model:28} with a mark ({row['takes']}): sound {row['sound']}, words {row['words']}, "
                f"neither {row['neither']}   control ({control['takes']}): sound {control['sound']}, "
                f"words {control['words']}"
                + (f"   errors {row['errors'] + control['errors']}" if row["errors"] else ""),
                flush=True,
            )
            for t in marked:
                if t.get("added"):
                    print(f"    {t['text']!r} -> {t['heard']!r}", flush=True)  # noqa: T201
    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="elevenlabs")
    parser.add_argument(
        "--models",
        default="eleven_v3_conversational,eleven_v3,eleven_v4_turbo,eleven_flash_v2_5,eleven_multilingual_v2",
    )
    parser.add_argument("--voice", default=VoiceConfig().voice)
    parser.add_argument("--language", default="es")
    parser.add_argument("--takes", type=int, default=3, help="takes of each sentence")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--out", help="directory for one WAV per take")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ELEVENLABS_API_KEY is not set", file=sys.stderr)  # noqa: T201
        sys.exit(2)
    report = asyncio.run(run(args))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
