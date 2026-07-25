"""Audio synthesis, caching and PCM helpers for the voice replay harness.

ElevenLabs -> raw PCM16 16k mono, cached by a hash of ``(text, voice, model)``.
Synthesis is the only expensive part of the harness, so nothing is ever generated
twice for unchanged script text.

Everything here is deterministic once the cache is warm: the same script always
composes to the same bytes, which is what makes replay runs comparable.

Standalone use — hear what the harness will actually feed the session::

    uv run python benchmarks/voice/synth.py "I'd like to order a large"
    uv run python benchmarks/voice/synth.py --verify
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import wave
from collections.abc import Iterable, Iterator
from pathlib import Path

import structlog
from dotenv import load_dotenv
from timbal.voice import AudioOutputConfig
from timbal.voice.elevenlabs import ElevenLabsStreamTTS

SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # PCM16 mono
FRAME_SECS = 0.02
FRAME_BYTES = int(BYTES_PER_SECOND * FRAME_SECS)
SILENCE_FRAME = b"\x00" * FRAME_BYTES

TTS_MODEL = os.environ.get("TIMBAL_TTS_MODEL", "eleven_flash_v2_5")
# Distinct voices for the two sides: replaying the user in the assistant's voice
# would trip the session's echo heuristics on legitimate speech.
ASSISTANT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "1SM7GgM6IMuvQlz2BwM3")
USER_VOICE_ID = os.environ.get("TIMBAL_BENCH_USER_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------


def silence(secs: float) -> bytes:
    """Exactly ``secs`` of silence, rounded to whole 20ms frames."""
    return SILENCE_FRAME * max(0, round(secs / FRAME_SECS))


def duration_secs(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SECOND


def frame_pad(pcm: bytes) -> bytes:
    """Zero-pad to a whole number of frames.

    Clips rarely end on a frame boundary. Padding each part keeps the composed
    stream frame-aligned at the cost of <20ms of silence at a seam, which is
    below anything turn detection can resolve.
    """
    remainder = len(pcm) % FRAME_BYTES
    return pcm if remainder == 0 else pcm + b"\x00" * (FRAME_BYTES - remainder)


def frames(pcm: bytes) -> Iterator[bytes]:
    """Split into 20ms frames, padding the tail."""
    padded = frame_pad(pcm)
    for i in range(0, len(padded), FRAME_BYTES):
        yield padded[i : i + FRAME_BYTES]


def compose(parts: Iterable[bytes | float]) -> bytes:
    """Concatenate clips (``bytes``) and silences (``float`` seconds).

    The static half of a script: everything except reactive steps, which need the
    session's own audio timing and therefore live in the harness.
    """
    return b"".join(frame_pad(p) if isinstance(p, bytes) else silence(p) for p in parts)


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


# ---------------------------------------------------------------------------
# Synthesis + cache
# ---------------------------------------------------------------------------


def clip_path(text: str, voice_id: str = USER_VOICE_ID, model: str = TTS_MODEL) -> Path:
    digest = hashlib.sha256(f"{model}|{voice_id}|{text}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.pcm"


async def synthesize_clips(
    texts: Iterable[str],
    *,
    voice_id: str = USER_VOICE_ID,
    model: str = TTS_MODEL,
    quiet: bool = False,
) -> dict[str, bytes]:
    """Return ``{text: pcm}``, synthesizing only what the cache is missing."""
    wanted = list(dict.fromkeys(texts))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [t for t in wanted if not clip_path(t, voice_id, model).exists()]

    if missing:
        if not quiet:
            print(f"synthesizing {len(missing)} clip(s) with voice {voice_id}...")
        tts = ElevenLabsStreamTTS()
        await tts.connect(AudioOutputConfig(model=model, voice=voice_id, sample_rate=SAMPLE_RATE))
        try:
            for text in missing:
                buf = bytearray()
                async for chunk in tts.synthesize(text):
                    buf.extend(chunk)
                if not buf:
                    raise RuntimeError(f"ElevenLabs returned no audio for {text!r}")
                clip_path(text, voice_id, model).write_bytes(bytes(buf))
        finally:
            await tts.close()

    return {t: clip_path(t, voice_id, model).read_bytes() for t in wanted}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VERIFY_PARTS: list[str | float] = [
    "Verification clip one.",
    0.5,
    "Verification clip two.",
]


async def _verify() -> int:
    """Phase 1 exit criterion: clips and silences compose byte-exactly."""
    texts = [p for p in _VERIFY_PARTS if isinstance(p, str)]
    checks: list[tuple[str, bool, str]] = []

    first = await synthesize_clips(texts)
    second = await synthesize_clips(texts)  # must be a pure cache read
    checks.append(("cache read is stable", first == second, f"{len(first)} clip(s)"))

    def _build(clips: dict[str, bytes]) -> bytes:
        return compose([clips[p] if isinstance(p, str) else p for p in _VERIFY_PARTS])

    a, b = _build(first), _build(second)
    digest = hashlib.sha256(a).hexdigest()[:16]
    checks.append(("compose is byte-exact", a == b, f"sha256:{digest}"))
    checks.append(("frame-aligned", len(a) % FRAME_BYTES == 0, f"{len(a)} bytes"))
    checks.append(("non-empty", duration_secs(a) > 1.0, f"{duration_secs(a):.2f}s"))

    gap = silence(0.5)
    checks.append(("silence is exact", duration_secs(gap) == 0.5, f"{len(gap)} bytes"))
    checks.append(("frames round-trip", b"".join(frames(a)) == a, f"{len(a) // FRAME_BYTES} frames"))

    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<24} {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="text to synthesize")
    parser.add_argument("-o", "--out", type=Path, help="write a WAV here (default: results/synth/)")
    parser.add_argument("--voice", default=USER_VOICE_ID)
    parser.add_argument("--model", default=TTS_MODEL)
    parser.add_argument("--verify", action="store_true", help="check byte-exact reproducibility")
    parser.add_argument("--verbose", action="store_true", help="keep timbal debug logs")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not args.verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

    if args.verify:
        return await _verify()
    if not args.text:
        parser.print_help()
        return 2

    text = " ".join(args.text)
    clips = await synthesize_clips([text], voice_id=args.voice, model=args.model)
    pcm = clips[text]
    out = args.out or HERE / "results" / "synth" / f"{clip_path(text, args.voice, args.model).stem}.wav"
    write_wav(out, pcm)
    print(f"{duration_secs(pcm):.2f}s  {len(pcm)} bytes  ->  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
