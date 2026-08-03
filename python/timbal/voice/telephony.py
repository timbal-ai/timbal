"""Provider-agnostic telephony primitives.

Phone carriers deliver audio as G.711 μ-law mono at 8kHz over provider media
WebSockets (Twilio Media Streams, Telnyx bidirectional streaming). This module
holds everything the server bridge needs that is *not* provider dialect:

- G.711 μ-law encode/decode (pure Python, table-driven)
- ``PcmResampler`` — stateful 8kHz ↔ session-rate PCM16 resampling (via av)
- ``TelephonyPlaybackTracker`` — mark-based playback tracking that also fires
  a ``clear`` callback on barge-in so the provider drops its buffered audio

The provider dialects (frame shapes, webhook formats, signatures) live in
``timbal.server.telephony``.
"""

from __future__ import annotations

import fractions
from array import array
from collections.abc import Callable

from .playback import BufferedPlaybackTracker

TELEPHONY_SAMPLE_RATE = 8_000
"""G.711 carrier audio: 8kHz mono, one μ-law byte per sample (8 bytes/ms)."""

_BIAS = 0x84
_CLIP = 32_635

# exponent = position of the highest set bit of (magnitude >> 7), i.e. the
# G.711 segment number. Classic Sun/CCITT exp_lut.
_EXP_LUT = bytes(max(0, i.bit_length() - 1) for i in range(256))


def _encode_sample(sample: int) -> int:
    """Encode one signed 16-bit sample to a μ-law byte (G.711)."""
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    if sample > _CLIP:
        sample = _CLIP
    sample += _BIAS
    exponent = _EXP_LUT[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _build_decode_table() -> array:
    table = array("h", bytes(512))
    for b in range(256):
        u = ~b & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
        table[b] = -magnitude if sign else magnitude
    return table


_DECODE_TABLE = _build_decode_table()
_ENCODE_TABLE: bytes | None = None


def _encode_table() -> bytes:
    # 64KB, built once on first downlink use (a few ms), keyed by the
    # sample's unsigned 16-bit representation.
    global _ENCODE_TABLE
    if _ENCODE_TABLE is None:
        table = bytearray(65_536)
        for s in range(-32_768, 32_768):
            table[s & 0xFFFF] = _encode_sample(s)
        _ENCODE_TABLE = bytes(table)
    return _ENCODE_TABLE


def ulaw_decode(data: bytes) -> bytes:
    """μ-law bytes → PCM16 little-endian mono (same sample count)."""
    out = array("h", bytes(2 * len(data)))
    dec = _DECODE_TABLE
    for i, b in enumerate(data):
        out[i] = dec[b]
    return out.tobytes()


def ulaw_encode(pcm: bytes) -> bytes:
    """PCM16 little-endian mono → μ-law bytes (drops a trailing odd byte)."""
    table = _encode_table()
    samples = memoryview(pcm[: len(pcm) & ~1]).cast("h")
    return bytes(table[s & 0xFFFF] for s in samples)


ULAW_SILENCE = b"\xff"
"""μ-law encoding of a zero sample — used to pad sub-minimum media chunks."""


class PcmResampler:
    """Stateful PCM16 mono resampler between two fixed rates.

    Thin wrapper over ``av.AudioResampler`` (ships with ``timbal[voice]``)
    that speaks raw bytes instead of frames. One instance per direction per
    call — the underlying FIR filter carries state across chunks.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        try:
            from av import AudioFrame, AudioResampler
        except ImportError as e:  # pragma: no cover — exercised only without the extra
            raise ImportError(
                "Telephony audio bridging requires the timbal[voice] extra: "
                "uv pip install 'timbal[voice]'"
            ) from e
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._src_rate = src_rate
        self._audio_frame_cls = AudioFrame
        self._resampler = AudioResampler(format="s16", layout="mono", rate=dst_rate)
        self._pts = 0

    def process(self, pcm: bytes) -> bytes:
        """Resample a chunk. May return b"" or a different length (filter delay)."""
        samples = len(pcm) // 2
        if samples <= 0:
            return b""
        frame = self._audio_frame_cls(format="s16", layout="mono", samples=samples)
        frame.planes[0].update(pcm[: samples * 2])
        frame.sample_rate = self._src_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self._src_rate)
        self._pts += samples
        # Plane buffers are alignment-padded; only samples*2 bytes are audio.
        return b"".join(bytes(out.planes[0])[: out.samples * 2] for out in self._resampler.resample(frame))


class TelephonyPlaybackTracker(BufferedPlaybackTracker):
    """Buffered tracker whose acks come from provider ``mark`` echoes.

    The provider buffers our media messages and plays them at line rate; we
    send a ``mark`` after each media message and treat its echo as a playback
    ack. On barge-in the session calls :meth:`on_interrupted` — beyond
    freezing the played axis, the provider must also drop its buffered audio,
    so the bridge's ``clear``-sender fires here.
    """

    def __init__(self, bytes_per_second: int, on_clear: Callable[[], None]) -> None:
        super().__init__(bytes_per_second)
        self._on_clear = on_clear

    def on_interrupted(self) -> None:
        super().on_interrupted()
        self._on_clear()
