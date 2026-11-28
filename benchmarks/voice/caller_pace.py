"""How fast does a caller talk, as ``adaptive_tempo`` measures it?

``VoiceConfig.adaptive_tempo`` divides the syllables of what the caller said
by the seconds they spent saying it, and slows the reply down for a slow
talker. Both halves are estimates: the syllables are counted from the
transcript's spelling (:func:`timbal.voice.adaptive_tempo.syllables`) and the
seconds come from a level meter on the mic
(:class:`timbal.voice.mic_activity.MicActivity`). This script says what the two
give together for speech of a known speed, which is where the default rates
(:data:`timbal.voice.adaptive_tempo.DEFAULT_RATES`) come from.

A voice says a set of caller lines. Each take is then slowed down and sped up
without changing its pitch (the same ``atempo`` filter the session uses), line
noise is added, and the audio is fed to the meter in 20 ms pieces as a mic
would send it. For every speed it prints the rate the session would measure
and the factor it would apply. As a check on the meter, the same audio is
transcribed with word times (ElevenLabs Scribe) and the time between the first
and the last word, pauses over 0.2 s left out, is compared with the meter's.

    uv run python benchmarks/voice/caller_pace.py
    uv run python benchmarks/voice/caller_pace.py --language en --noise-dbfs -40
    uv run python benchmarks/voice/caller_pace.py --provider deepgram --model aura-2-celeste-es

The speaker is a synthetic voice, not a person: its speed 1.0 is a neutral
reading, and people at ease talk a little faster and less evenly. The table is
about the instrument, not about callers.

Needs ``ELEVENLABS_API_KEY`` (Scribe, and the voice when ``--provider`` is
``elevenlabs``) and the key of any other ``--provider``. A missing key exits 2
and names it.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import statistics
import sys
import wave
from array import array

import httpx
from dotenv import load_dotenv
from timbal.voice import AudioOutputConfig, VoiceConfig
from timbal.voice.adaptive_tempo import CallerPace, syllables
from timbal.voice.mic_activity import MicActivity
from timbal.voice.providers import resolve_tts
from timbal.voice.tempo import change_tempo

RATE = 16_000
LINES = {
    "es": [
        "Hola, buenas tardes, quería saber cuánto tengo que pagar este mes de la luz.",
        "Pues mira, te llamo porque me ha llegado una factura que no entiendo.",
        "Quería cambiar la cuenta del banco donde me cobráis los recibos.",
        "Es que llevo toda la semana sin internet y nadie me dice nada.",
        "Me gustaría pedir cita para la semana que viene, si puede ser por la mañana.",
        "Sí, soy el titular del contrato, y el piso está a mi nombre.",
        "No, no, eso ya lo intenté ayer y me dio el mismo error.",
        "Vale, entonces espero el mensaje y os vuelvo a llamar si no llega.",
    ],
    "en": [
        "Hi, good afternoon, I wanted to know how much I have to pay this month.",
        "So I'm calling because I got a bill that I don't really understand.",
        "I'd like to change the bank account you take the payments from.",
        "I've been without internet all week and nobody has told me anything.",
        "I'd like to book an appointment for next week, in the morning if possible.",
        "Yes, I'm the account holder, and the flat is in my name.",
        "No, I already tried that yesterday and it gave me the same error.",
        "Okay, then I'll wait for the message and call you back if it doesn't come.",
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


async def synthesize(text: str, args: argparse.Namespace) -> bytes:
    tts = resolve_tts(args.provider)
    await tts.connect(AudioOutputConfig(model=args.model, voice=args.voice, sample_rate=RATE))
    try:
        return b"".join([chunk async for chunk in tts.synthesize(text)])
    finally:
        await tts.close()


async def at_tempo(pcm: bytes, tempo: float) -> bytes:
    async def chunks():
        for start in range(0, len(pcm), 3200):
            yield pcm[start : start + 3200]

    return b"".join([chunk async for chunk in change_tempo(chunks(), sample_rate=RATE, tempo=tempo)])


def on_a_line(pcm: bytes, noise_dbfs: float | None, seed: int) -> bytes:
    """Half a second of line before and after, and its noise over everything."""
    pad = bytes(RATE)  # 0.5 s
    samples = array("h")
    samples.frombytes(pad + pcm[: len(pcm) - len(pcm) % 2] + pad)
    if noise_dbfs is None:
        return samples.tobytes()
    rng = random.Random(seed)
    amp = 32768 * 10 ** (noise_dbfs / 20) * 3**0.5
    return array("h", (max(-32768, min(32767, s + int(rng.uniform(-amp, amp)))) for s in samples)).tobytes()


def metered_secs(pcm: bytes) -> float:
    meter = MicActivity(RATE)
    step = RATE // 50 * 2
    for start in range(0, len(pcm), step):
        meter.push(pcm[start : start + step])
    return meter.speech_secs


async def word_secs(client: httpx.AsyncClient, pcm: bytes, language: str) -> float | None:
    """Seconds from the first word to the last by Scribe's word times, pauses over 0.2 s left out."""
    for attempt in range(3):
        try:
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                data={"model_id": "scribe_v1", "language_code": language, "timestamps_granularity": "word"},
                files={"file": ("take.wav", wav(pcm), "audio/wav")},
                timeout=90,
            )
            if response.status_code == 200:
                words = [word for word in response.json().get("words", []) if word.get("type") == "word"]
                if len(words) < 2:
                    return None
                total = words[-1]["end"] - words[0]["start"]
                for before, after in zip(words, words[1:], strict=False):
                    gap = after["start"] - before["end"]
                    if gap > 0.2:
                        total -= gap
                return total
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.5 * (attempt + 1))
    return None


async def run(args: argparse.Namespace) -> list[dict]:
    lines = LINES[args.language]
    semaphore = asyncio.Semaphore(args.parallel)

    async def spoken(text: str) -> bytes:
        async with semaphore:
            return await synthesize(text, args)

    takes = await asyncio.gather(*(spoken(text) for text in lines))
    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        for tempo in args.tempos:

            async def one(index: int, text: str, pcm: bytes, tempo: float = tempo) -> dict:
                line = on_a_line(await at_tempo(pcm, tempo), args.noise_dbfs, seed=index)
                secs = metered_secs(line)
                async with semaphore:
                    by_words = None if args.no_check else await word_secs(client, line, args.language)
                rate = CallerPace.measure(text, secs)
                return {
                    "tempo": tempo,
                    "syllables": syllables(text),
                    "metered_secs": round(secs, 2),
                    "word_secs": None if by_words is None else round(by_words, 2),
                    "rate": None if rate is None else round(rate, 2),
                    "factor": CallerPace(upset_tempo=None).heard(text, secs, args.language),
                }

            rows.extend(
                await asyncio.gather(
                    *(one(i, text, pcm) for i, (text, pcm) in enumerate(zip(lines, takes, strict=True)))
                )
            )
    return rows


def report(rows: list[dict], args: argparse.Namespace) -> None:
    slow, normal = CallerPace().rates(args.language)
    noise = "no line noise" if args.noise_dbfs is None else f"line noise {args.noise_dbfs:g} dBFS"
    lines = len(LINES[args.language])
    print(f"{args.provider} {args.model} voice {args.voice}, {args.language}, {lines} lines, {noise}")  # noqa: T201
    print(f"factor 0.9 at {slow:g} syllables/s or less, 1 at {normal:g} or more")  # noqa: T201
    print("speed   syllables/s median (range)   factor median (range)   meter / word times")  # noqa: T201
    for tempo in args.tempos:
        of_tempo = [row for row in rows if row["tempo"] == tempo]
        rates = [row["rate"] for row in of_tempo if row["rate"] is not None]
        factors = [row["factor"] for row in of_tempo]
        ratios = [row["metered_secs"] / row["word_secs"] for row in of_tempo if row["word_secs"]]
        rate = f"{statistics.median(rates):.1f} ({min(rates):.1f}-{max(rates):.1f})" if rates else "none measured"
        factor = f"{statistics.median(factors):.2f} ({min(factors):.2f}-{max(factors):.2f})"
        ratio = f"{statistics.median(ratios):.2f} ({min(ratios):.2f}-{max(ratios):.2f})" if ratios else "not checked"
        print(f"{tempo:<7g} {rate:<28s} {factor:<23s} {ratio}")  # noqa: T201


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=sorted(LINES), default="es")
    parser.add_argument("--provider", default="elevenlabs")
    parser.add_argument("--model", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--tempos", type=float, nargs="+", default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    parser.add_argument("--noise-dbfs", type=float, default=-55.0, help="level of the line noise added (default -55)")
    parser.add_argument("--no-noise", dest="noise_dbfs", action="store_const", const=None)
    parser.add_argument("--no-check", action="store_true", help="skip the comparison with word times")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print the takes as JSON")
    args = parser.parse_args()
    if args.provider == "elevenlabs":
        defaults = VoiceConfig()
        args.model = args.model or defaults.tts_model
        args.voice = args.voice or defaults.voice
    needed = ["ELEVENLABS_API_KEY"] if not args.no_check or args.provider == "elevenlabs" else []
    missing = [key for key in needed if not os.environ.get(key)]
    if missing:
        print(f"missing {', '.join(missing)}", file=sys.stderr)  # noqa: T201
        sys.exit(2)
    rows = asyncio.run(run(args))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))  # noqa: T201
    else:
        report(rows, args)


if __name__ == "__main__":
    main()
