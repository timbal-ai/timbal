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
import base64
import hashlib
import json
import logging
import os
import sys
import wave
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import httpx
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
# Fluent utterances
# ---------------------------------------------------------------------------
#
# A speaker who pauses mid-sentence keeps the intonation of a sentence still in
# progress. Synthesizing the fragments separately does not: ElevenLabs renders
# each one as its own sentence and gives it a falling, finished contour.
#
# Measured with Smart Turn v3 on identical words: "I want to return an item I
# bought" scores 0.986 (finished) rendered standalone and 0.019 (mid-thought)
# when cut out of the fluent whole-sentence render. That is the difference
# between a scenario testing turn-taking and a scenario testing TTS phrasing.
#
# So fluent utterances are rendered once, whole, and sliced at the character
# timestamps ElevenLabs returns alongside the audio.

ALIGNMENT_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"


def alignment_path(text: str, voice_id: str = USER_VOICE_ID, model: str = TTS_MODEL) -> Path:
    return clip_path(text, voice_id, model).with_suffix(".json")


def part_key(utterance: str, index: int) -> str:
    """Cache/clip key for one slice of a fluent utterance."""
    return f"{utterance}#{index}"


def _part_end_chars(parts: Sequence[str]) -> list[int]:
    """Exclusive character offset of each part within ``" ".join(parts)``."""
    ends: list[int] = []
    pos = 0
    for i, part in enumerate(parts):
        if i:
            pos += 1  # the joining space
        pos += len(part)
        ends.append(pos)
    return ends


def slice_fluent(pcm: bytes, char_end_times: Sequence[float], parts: Sequence[str]) -> list[bytes]:
    """Cut ``pcm`` at the boundary where each part's last character ends.

    The final part runs to the end of the audio so no trailing samples are lost.
    """
    slices: list[bytes] = []
    start = 0
    for i, end_char in enumerate(_part_end_chars(parts)):
        if i == len(parts) - 1:
            stop = len(pcm)
        else:
            # Byte offsets must stay sample-aligned or the PCM16 frames shear.
            stop = min(int(char_end_times[end_char - 1] * SAMPLE_RATE) * 2, len(pcm))
        slices.append(pcm[start:stop])
        start = stop
    return slices


async def _synthesize_aligned(
    utterance: str, voice_id: str, model: str
) -> tuple[bytes, list[float]]:
    """Whole-utterance PCM plus per-character end times, cached on disk.

    Uses the REST ``with-timestamps`` endpoint; the streaming websocket used for
    ordinary clips does not return alignment.
    """
    pcm_file = clip_path(utterance, voice_id, model)
    align_file = alignment_path(utterance, voice_id, model)
    if pcm_file.exists() and align_file.exists():
        return pcm_file.read_bytes(), json.loads(align_file.read_text())

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("Set ELEVENLABS_API_KEY to synthesize fluent utterances.")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            ALIGNMENT_ENDPOINT.format(voice_id=voice_id),
            params={"output_format": f"pcm_{SAMPLE_RATE}"},
            headers={"xi-api-key": api_key},
            json={"text": utterance, "model_id": model},
        )
        response.raise_for_status()
        payload = response.json()

    pcm = base64.b64decode(payload["audio_base64"])
    ends = payload["alignment"]["character_end_times_seconds"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pcm_file.write_bytes(pcm)
    align_file.write_text(json.dumps(ends))
    return pcm, ends


async def synthesize_fluent(
    groups: Iterable[tuple[str, Sequence[str]]],
    *,
    voice_id: str = USER_VOICE_ID,
    model: str = TTS_MODEL,
    quiet: bool = False,
) -> dict[str, bytes]:
    """Return ``{part_key: pcm}`` for each ``(utterance, parts)`` group."""
    wanted = {utterance: tuple(parts) for utterance, parts in groups}
    if not wanted:
        return {}
    missing = [u for u in wanted if not alignment_path(u, voice_id, model).exists()]
    if missing and not quiet:
        print(f"synthesizing {len(missing)} fluent utterance(s) with alignment...")

    clips: dict[str, bytes] = {}
    for utterance, parts in wanted.items():
        pcm, ends = await _synthesize_aligned(utterance, voice_id, model)
        for i, chunk in enumerate(slice_fluent(pcm, ends, parts)):
            clips[part_key(utterance, i)] = chunk
    return clips


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

    # Slicing a fluent utterance must lose no audio and stay on PCM16 sample
    # boundaries, or every downstream fragment shears by half a sample.
    parts = ("Verification clip one,", "sliced in two.")
    utterance = " ".join(parts)
    pcm, ends = await _synthesize_aligned(utterance, USER_VOICE_ID, TTS_MODEL)
    pieces = slice_fluent(pcm, ends, parts)
    checks.append(("fluent slice is lossless", b"".join(pieces) == pcm, f"{len(pcm)} bytes"))
    checks.append(
        (
            "fluent slice is sample-aligned",
            all(len(p) % 2 == 0 for p in pieces),
            " + ".join(f"{duration_secs(p):.2f}s" for p in pieces),
        )
    )

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
