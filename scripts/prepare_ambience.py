"""Turn a downloaded field recording into a CDN ambience preset.

Output contract (all presets): mono, 16 kHz, PCM16, −30 dBFS RMS, loop
crossfade baked into the asset so end→start is seamless. Needs
``timbal[voice]`` (av). CC0 sources only (attribution-free) — record every
asset in ``ambience/ATTRIBUTIONS.md``. Use ``--lowpass`` on anything with
voices so leaked speech can't be transcribed; ``--seamless`` for tracks that
already loop cleanly; ``--layer`` to mix a second file underneath.

Output goes to ``dist/ambience/{name}.wav``. To ship it: upload the file to
``https://timbalusercontent.com/assets/voice/ambience/{name}.wav`` and pin
the printed sha256 in ``timbal.voice.ambience.PRESETS``. CDN assets are
immutable — a changed track needs a new name.

Usage:
    uv run python scripts/prepare_ambience.py ~/Downloads/track.wav office --seamless
    uv run python scripts/prepare_ambience.py ~/Downloads/walla.wav cafe --lowpass 2500
"""

from __future__ import annotations

import argparse
import hashlib
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CROSSFADE_SECS = 0.25
TARGET_RMS_DBFS = -30.0
OUT_DIR = Path(__file__).parent.parent / "dist" / "ambience"


def _fft_lowpass(x: np.ndarray, cutoff_hz: float, *, soft_hz: float = 300.0) -> np.ndarray:
    """Low-pass with a soft (cosine-ramped) edge to avoid ringing."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)
    gain = np.ones_like(freqs)
    gain[freqs > cutoff_hz] = 0.0
    ramp = (freqs > cutoff_hz - soft_hz) & (freqs <= cutoff_hz)
    gain[ramp] = 0.5 - 0.5 * np.cos(np.pi * (cutoff_hz - freqs[ramp]) / soft_hz)
    return np.fft.irfft(spectrum * gain, n=len(x))


def _normalize_rms(x: np.ndarray, dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2)))
    return x * (10 ** (dbfs / 20.0) / rms)


def _bake_loop_crossfade(x: np.ndarray) -> np.ndarray:
    """Blend the tail into the head (equal power) and trim it, so end→start is seamless."""
    n = int(CROSSFADE_SECS * SAMPLE_RATE)
    t = np.linspace(0.0, np.pi / 2, n)
    head = x[:n] * np.sin(t) + x[-n:] * np.cos(t)
    return np.concatenate([head, x[n:-n]])


def _write_wav(path: Path, x: np.ndarray) -> None:
    pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _decode_mono_16k(path: Path) -> np.ndarray:
    """Decode any audio file to float mono at SAMPLE_RATE via PyAV."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1).astype(np.float64))
        for out in resampler.resample(None):  # drain
            chunks.append(out.to_ndarray().reshape(-1).astype(np.float64))
    if not chunks:
        raise SystemExit(f"no audio decoded from {path}")
    return np.concatenate(chunks) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="source audio file (any format PyAV can decode)")
    parser.add_argument("name", help="preset name — output is ambience/{name}.wav")
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip at the head (default 0)")
    parser.add_argument("--duration", type=float, default=60.0, help="max seconds to keep (default 60)")
    parser.add_argument("--lowpass", type=float, default=None,
                        help="low-pass cutoff Hz; use ~2500 on murmur with voices to smear intelligibility")
    parser.add_argument("--layer", default=None,
                        help="second audio file mixed under the main one (looped/trimmed to length)")
    parser.add_argument("--layer-db", type=float, default=-6.0,
                        help="layer level relative to the main track (default -6 dB)")
    parser.add_argument("--seamless", action="store_true",
                        help="source already loops cleanly — skip the crossfade bake")
    args = parser.parse_args()

    x = _decode_mono_16k(Path(args.input).expanduser())
    start = int(args.start * SAMPLE_RATE)
    if start >= len(x):
        raise SystemExit(f"--start {args.start}s is past the end of the file ({len(x) / SAMPLE_RATE:.1f}s)")
    x = x[start : start + int(args.duration * SAMPLE_RATE)]
    if len(x) < SAMPLE_RATE * 5:
        raise SystemExit("less than 5s of audio after trimming — not enough for a loop")
    if args.layer:
        y = _normalize_rms(_decode_mono_16k(Path(args.layer).expanduser()), TARGET_RMS_DBFS)
        y = np.tile(y, int(np.ceil(len(x) / len(y))))[: len(x)]
        x = _normalize_rms(x, TARGET_RMS_DBFS) + y * (10 ** (args.layer_db / 20.0))
    if args.lowpass:
        x = _fft_lowpass(x, float(args.lowpass))
    x = _normalize_rms(x, TARGET_RMS_DBFS)
    # Quiet beds normalized up can have huge crest factors (chair clicks,
    # typing transients) — cap the peak at -1 dBFS instead of hard-clipping.
    peak = float(np.abs(x).max())
    if peak > 0.89:
        x *= 0.89 / peak
    if not args.seamless:
        x = _bake_loop_crossfade(x)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.name}.wav"
    _write_wav(out, x)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB)")  # noqa: T201
    print(f'    "{args.name}": "{sha}",  # pin in timbal.voice.ambience.PRESETS, then upload to the CDN')  # noqa: T201


if __name__ == "__main__":
    main()
