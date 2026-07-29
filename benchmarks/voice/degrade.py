"""Microphone-path degradation for the voice replay harness.

Every number this suite has produced came from clean 16 kHz TTS speech handed
straight to the STT: no room noise, no telephone band, no companding. That makes
the measurements reproducible, and it is also the standing reason not to trust
them for real callers. Endpointing thresholds in particular are tuned against
synthetic prosody in silence — `vad_silence_threshold_secs` on ElevenLabs and
`eot_threshold` on Flux both decide when speech has *stopped*, which is precisely
the judgement a noise floor makes harder.

This module builds the missing path, in the order a real one applies:

    user speech (+ echo leak)  ->  + noise bed  ->  telephone band  ->  STT

Noise is continuous, including through silence. That matters more than the level:
noise that starts and stops with the speech is a *cue* — it tells the endpointer
exactly where the utterance ended — so gating it to the clips would test
something easier than clean audio rather than harder.

Deliberately not modelled, and worth knowing before reading any result from here:
packet loss and jitter. A dropped 20 ms frame mid-word is a different failure from
a noisy one and is likely the more dangerous of the two for turn-taking.

Everything is deterministic given a seed, because a degradation axis that cannot
be replayed cannot be bisected.

Standalone use::

    uv run python benchmarks/voice/degrade.py --verify
    uv run python benchmarks/voice/degrade.py --wav 15 --telephone "some text"
"""

from __future__ import annotations

import argparse
import array
import asyncio
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from synth import (
    BYTES_PER_SECOND,
    FRAME_BYTES,
    SAMPLE_RATE,
    duration_secs,
    synthesize_clips,
    write_wav,
)

# ---------------------------------------------------------------------------
# G.711 mu-law companding
# ---------------------------------------------------------------------------
#
# 8-bit logarithmic quantisation: ~38 dB SNR on speech, and coarse enough at low
# amplitude to matter for a VAD deciding whether near-silence is silence.

_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635
_ULAW_EXPONENT = [
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    *([4] * 16), *([5] * 32), *([6] * 64), *([7] * 128),
]


def _encode_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0x00
    magnitude = min(-sample if sample < 0 else sample, _ULAW_CLIP) + _ULAW_BIAS
    exponent = _ULAW_EXPONENT[(magnitude >> 7) & 0xFF]
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _decode_ulaw(byte: int) -> int:
    byte = ~byte & 0xFF
    magnitude = (((byte & 0x0F) << 3) + _ULAW_BIAS) << ((byte >> 4) & 0x07)
    magnitude -= _ULAW_BIAS
    return -magnitude if byte & 0x80 else magnitude


# One table lookup per sample beats two function calls: this runs on every frame
# of every run, inside the loop that also has to hit a 20ms deadline.
_COMPAND = [_decode_ulaw(_encode_ulaw(s)) for s in range(-32768, 32768)]


def compand(pcm: bytes) -> bytes:
    """Round-trip PCM16 through G.711 mu-law, as a phone network would."""
    samples = array.array("h", pcm)
    for i, s in enumerate(samples):
        samples[i] = _COMPAND[s + 32768]
    return samples.tobytes()


# ---------------------------------------------------------------------------
# Biquad filtering (RBJ cookbook, coefficients derived not tabulated)
# ---------------------------------------------------------------------------

Coeffs = tuple[float, float, float, float, float]
BiquadState = list[float]


def _lowpass(fc: float, fs: int = SAMPLE_RATE, q: float = 0.7071) -> Coeffs:
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b0, b1, b2 = (1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2
    a0, a1, a2 = 1 + alpha, -2 * cos_w0, 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _highpass(fc: float, fs: int = SAMPLE_RATE, q: float = 0.7071) -> Coeffs:
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b0, b1, b2 = (1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2
    a0, a1, a2 = 1 + alpha, -2 * cos_w0, 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def new_state() -> BiquadState:
    return [0.0, 0.0, 0.0, 0.0]


def _run_biquad(samples: list[float], c: Coeffs, state: BiquadState) -> None:
    """In-place Direct Form I. ``state`` carries across calls.

    Filter state must survive frame boundaries: resetting per frame would stamp a
    transient every 20ms, which is a periodic click the STT would hear and this
    module would not be measuring.
    """
    b0, b1, b2, a1, a2 = c
    x1, x2, y1, y2 = state
    for i, x0 in enumerate(samples):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        samples[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    state[0], state[1], state[2], state[3] = x1, x2, y1, y2


# The G.711 passband. Two low-pass stages, because one 12 dB/octave skirt still
# leaks enough 5-6 kHz energy that the result sounds like a muffled wideband
# recording rather than a phone call.
_TELEPHONE_STAGES: tuple[Coeffs, ...] = (
    _highpass(300.0),
    _lowpass(3400.0),
    _lowpass(3400.0),
)


def telephone_state() -> list[BiquadState]:
    return [new_state() for _ in _TELEPHONE_STAGES]


def band_limit(pcm: bytes, state: list[BiquadState] | None = None) -> bytes:
    """Restrict to the 300-3400 Hz telephone band."""
    if state is None:
        state = telephone_state()
    samples = [float(s) for s in array.array("h", pcm)]
    for stage, st in zip(_TELEPHONE_STAGES, state, strict=True):
        _run_biquad(samples, stage, st)
    out = array.array("h", bytes(len(pcm)))
    for i, s in enumerate(samples):
        out[i] = 32767 if s > 32767 else (-32768 if s < -32768 else int(s))
    return out.tobytes()


def resample_8k(pcm: bytes) -> bytes:
    """Decimate to 8 kHz and interpolate back, as an 8 kHz link would.

    Only meaningful after :func:`band_limit`; on its own it would alias. The band
    limit is what changes the sound, but the round trip is cheap and it is the
    part that is literally true of a phone call.
    """
    samples = array.array("h", pcm)
    if not samples:
        return pcm
    out = array.array("h", bytes(len(pcm)))
    for i in range(len(samples)):
        if i % 2 == 0:
            out[i] = samples[i]
        else:
            prev = samples[i - 1]
            nxt = samples[i + 1] if i + 1 < len(samples) else prev
            out[i] = (prev + nxt) // 2
    return out.tobytes()


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def rms(pcm: bytes) -> float:
    samples = array.array("h", pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(float(s) * s for s in samples) / len(samples))


def active_rms(pcm: bytes, floor_ratio: float = 0.1) -> float:
    """RMS of the speech, ignoring the silence around and inside it.

    SNR has to be stated against the level of the speech, not the level of the
    recording: a scenario that is 60% deliberate pause would otherwise get a much
    louder noise bed than one that is not, and the two would stop being
    comparable at the same nominal SNR.
    """
    frame_rms = [rms(pcm[i : i + FRAME_BYTES]) for i in range(0, len(pcm), FRAME_BYTES)]
    if not frame_rms:
        return 0.0
    threshold = max(frame_rms) * floor_ratio
    active = [r for r in frame_rms if r >= threshold]
    if not active:
        return 0.0
    return math.sqrt(sum(r * r for r in active) / len(active))


def noise_rms_for_snr(speech_rms: float, snr_db: float) -> float:
    return speech_rms / (10 ** (snr_db / 20))


def white_noise(n: int, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def pink_noise(n: int, rng: random.Random) -> list[float]:
    """1/f noise (Kellet's economy method).

    Pink is the default because room, traffic and HVAC noise is pink, so it is the
    honest stand-in for a caller who is not in a recording booth.

    Note it is not the harsher of the two in the telephone band: at matched
    broadband level, pink puts most of its energy below 300 Hz where the codec's
    high-pass removes it, so `white` delivers *more* in-band energy (measured in
    ``--verify``). Use `white` to stress the STT, `pink` to be realistic.
    """
    b = [0.0] * 7
    out: list[float] = []
    for _ in range(n):
        w = rng.gauss(0.0, 1.0)
        b[0] = 0.99886 * b[0] + w * 0.0555179
        b[1] = 0.99332 * b[1] + w * 0.0750759
        b[2] = 0.96900 * b[2] + w * 0.1538520
        b[3] = 0.86650 * b[3] + w * 0.3104856
        b[4] = 0.55000 * b[4] + w * 0.5329522
        b[5] = -0.7616 * b[5] - w * 0.0168980
        out.append(b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362)
        b[6] = w * 0.115926
    return out


NOISE_KINDS = ("pink", "white")


def noise_bed(secs: float, target_rms: float, *, kind: str = "pink", seed: int = 0) -> bytes:
    """A loopable bed of noise at ``target_rms``, deterministic for ``seed``."""
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}; expected one of {NOISE_KINDS}")
    n = max(1, int(secs * SAMPLE_RATE))
    rng = random.Random(seed)
    raw = pink_noise(n, rng) if kind == "pink" else white_noise(n, rng)
    scale = 0.0
    current = math.sqrt(sum(s * s for s in raw) / len(raw))
    if current > 0:
        scale = target_rms / current
    out = array.array("h", bytes(n * 2))
    for i, s in enumerate(raw):
        v = s * scale
        out[i] = 32767 if v > 32767 else (-32768 if v < -32768 else int(v))
    return out.tobytes()


# ---------------------------------------------------------------------------
# The axis
# ---------------------------------------------------------------------------

# One minute, looped. Long enough that no scenario hears a repeat, short enough
# that generating it costs a fraction of a second of the run's startup.
BED_SECS = 60.0


@dataclass(frozen=True)
class MicPath:
    """What happens to audio between the speaker's mouth and the STT.

    The default is the identity: every run before this axis existed.
    """

    snr_db: float | None = None
    telephone: bool = False
    noise: str = "pink"
    seed: int = 0

    @property
    def active(self) -> bool:
        return self.snr_db is not None or self.telephone

    @property
    def label(self) -> str:
        """Cell-label suffix, matching the `[leak=...]` axis convention."""
        parts = []
        if self.snr_db is not None:
            kind = "" if self.noise == "pink" else f",{self.noise}"
            parts.append(f"snr={self.snr_db:g}{kind}")
        if self.telephone:
            parts.append("phone")
        return f"[{','.join(parts)}]" if parts else ""

    def bed(self, speech_rms: float) -> bytes:
        """The noise to mix in, scaled against the measured speech level.

        Returned *clean*: the caller mixes it with the speech and then runs the
        codec over the sum, which is the order a real microphone and line apply.
        So ``snr_db`` is the acoustic SNR at the microphone, not the SNR the STT
        receives — speech and noise lose different amounts of energy to the
        telephone band, so the ratio downstream of it is not the same number.
        """
        if self.snr_db is None or speech_rms <= 0:
            return b""
        target = noise_rms_for_snr(speech_rms, self.snr_db)
        return noise_bed(BED_SECS, target, kind=self.noise, seed=self.seed)

    def apply(self, pcm: bytes, state: list[BiquadState] | None = None) -> bytes:
        """Run the codec over one frame or clip. Pass ``state`` to keep it continuous."""
        if not self.telephone:
            return pcm
        return compand(resample_8k(band_limit(pcm, state)))


def parse_mic_path(spec: str | None, *, seed: int = 0) -> MicPath:
    """Parse ``--mic-path`` : comma-separated ``snr=15``, ``white``, ``phone``.

    Examples: ``snr=15``, ``phone``, ``snr=10,white,phone``.
    """
    if not spec:
        return MicPath(seed=seed)
    snr_db: float | None = None
    telephone = False
    noise = "pink"
    for token in (t.strip().lower() for t in spec.split(",") if t.strip()):
        if token in ("phone", "telephone", "g711"):
            telephone = True
        elif token in NOISE_KINDS:
            noise = token
        elif token.startswith("snr="):
            snr_db = float(token[4:])
        else:
            raise ValueError(
                f"unknown --mic-path token {token!r}; expected snr=<db>, "
                f"one of {NOISE_KINDS}, or 'phone'"
            )
    return MicPath(snr_db=snr_db, telephone=telephone, noise=noise, seed=seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _tone(freq: float, secs: float, amplitude: int = 8000) -> bytes:
    n = int(secs * SAMPLE_RATE)
    out = array.array("h", bytes(n * 2))
    for i in range(n):
        out[i] = int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
    return out.tobytes()


def _db(ratio: float) -> float:
    return 20 * math.log10(ratio) if ratio > 0 else -math.inf


def _verify() -> int:
    """Offline checks: no API key, no network."""
    checks: list[tuple[str, bool, str]] = []

    # Companding must be idempotent, or repeated application in a chain would
    # keep degrading and the axis would not mean one pass through a codec.
    speech_like = _tone(220, 0.5, amplitude=12000)
    once = compand(speech_like)
    twice = compand(once)
    checks.append(("compand is idempotent", once == twice, f"{len(once)} bytes"))

    err = rms(bytes(array.array("h", [a - b for a, b in
              zip(array.array("h", speech_like), array.array("h", once), strict=True)]).tobytes()))
    snr = _db(rms(speech_like) / err) if err else math.inf
    checks.append(("compand SNR is G.711-like", 30 <= snr <= 45, f"{snr:.1f} dB"))

    # The band: 1kHz through, 6kHz gone, rumble gone.
    for freq, lo, hi, name in (
        (1000.0, -2.0, 1.0, "1kHz passes"),
        (6000.0, -math.inf, -25.0, "6kHz is rejected"),
        (100.0, -math.inf, -12.0, "100Hz is rejected"),
    ):
        tone = _tone(freq, 0.5)
        # Skip the filter's settling transient before measuring.
        got = _db(rms(band_limit(tone)[FRAME_BYTES * 5 :]) / rms(tone[FRAME_BYTES * 5 :]))
        checks.append((name, lo <= got <= hi, f"{got:+.1f} dB"))

    path = MicPath(snr_db=15.0, telephone=True)
    checks.append(("telephone is deterministic", path.apply(speech_like) == path.apply(speech_like), "byte-exact"))
    checks.append(("length is preserved", len(path.apply(speech_like)) == len(speech_like), f"{len(speech_like)} bytes"))

    # SNR must come out at the number that was asked for, or every result on this
    # axis is labelled with a level it does not have.
    speech = _tone(300, 1.0, amplitude=10000) + bytes(BYTES_PER_SECOND)  # half speech, half silence
    level = active_rms(speech)
    checks.append(("active_rms ignores silence", abs(_db(level / rms(speech)) - 3.0) < 0.6, f"{_db(level / rms(speech)):+.1f} dB vs whole"))
    bed = noise_bed(1.0, noise_rms_for_snr(level, 15.0), seed=1)
    got_snr = _db(level / rms(bed))
    checks.append(("bed hits the requested SNR", abs(got_snr - 15.0) < 0.5, f"{got_snr:.2f} dB"))

    checks.append(("bed is deterministic", noise_bed(0.2, 500, seed=7) == noise_bed(0.2, 500, seed=7), "seed 7"))
    checks.append(("seeds differ", noise_bed(0.2, 500, seed=7) != noise_bed(0.2, 500, seed=8), "7 vs 8"))
    # What makes pink pink: energy rising towards DC. Checked below 300 Hz, which
    # is also exactly the region the telephone high-pass throws away — so at
    # matched broadband level, white ends up the harsher of the two in-band.
    def _sub_300(pcm: bytes) -> float:
        samples = [float(s) for s in array.array("h", pcm)]
        state = new_state()
        stage = _lowpass(300.0)
        _run_biquad(samples, stage, state)
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    pink_bed = noise_bed(1.0, 2000, kind="pink", seed=3)
    white_bed = noise_bed(1.0, 2000, kind="white", seed=3)
    checks.append((
        "pink is low-frequency weighted",
        _sub_300(pink_bed) > _sub_300(white_bed) * 2,
        f"{_sub_300(pink_bed):.0f} vs {_sub_300(white_bed):.0f} below 300Hz",
    ))
    checks.append((
        "so white is harsher in the phone band",
        rms(band_limit(white_bed)) > rms(band_limit(pink_bed)),
        f"{rms(band_limit(white_bed)):.0f} vs {rms(band_limit(pink_bed)):.0f} in band",
    ))

    checks.append(("labels round-trip", parse_mic_path("snr=10,white,phone").label == "[snr=10,white,phone]", parse_mic_path("snr=10,white,phone").label))
    checks.append(("identity by default", not MicPath().active and MicPath().label == "", "no suffix"))

    # This runs inside the loop that must hand the session a frame every 20ms, so
    # the cost is part of the contract, not an implementation detail.
    frame = _tone(440, 0.02)
    state = telephone_state()
    import time

    t0 = time.perf_counter()
    reps = 200
    for _ in range(reps):
        MicPath(telephone=True).apply(frame, state)
    per_frame_ms = (time.perf_counter() - t0) / reps * 1000
    checks.append((
        "cost fits the frame budget",
        per_frame_ms < 4.0,
        f"{per_frame_ms:.2f} ms per 20 ms frame ({per_frame_ms / 20 * 100:.0f}%)",
    ))

    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<44} {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


async def _write_sample(text: str, spec: str, out: Path | None) -> int:
    path = parse_mic_path(spec)
    clips = await synthesize_clips([text])
    clean = clips[text]
    bed = path.bed(active_rms(clean))
    # Same order as ScriptFeeder.stream, or this writes a file that sounds like
    # something the harness never feeds: noise into the mic, then the line.
    degraded = clean
    if bed:
        from harness import mix_pcm16

        degraded = mix_pcm16(degraded, bed[: len(degraded)], 1.0)
    degraded = path.apply(degraded)
    out = out or Path(__file__).parent / "results" / "synth" / f"degraded{path.label}.wav"
    write_wav(out, degraded)
    print(f"{duration_secs(degraded):.2f}s  {path.label or '[clean]'}  ->  {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="text to synthesize and degrade")
    parser.add_argument("--verify", action="store_true", help="offline self-checks")
    parser.add_argument("--mic-path", default="snr=15,phone", help="e.g. 'snr=15,phone'")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    if args.verify:
        return _verify()
    if not args.text:
        parser.print_help()
        return 2

    from dotenv import load_dotenv

    load_dotenv(override=True)
    return asyncio.run(_write_sample(" ".join(args.text), args.mic_path, args.out))


if __name__ == "__main__":
    sys.exit(main())
