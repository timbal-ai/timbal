"""Unit tests for the telephony primitives (G.711 μ-law, resampler, tracker)."""

from __future__ import annotations

import pytest
from timbal.voice.telephony import (
    TELEPHONY_SAMPLE_RATE,
    ULAW_SILENCE,
    PcmResampler,
    TelephonyPlaybackTracker,
    ulaw_decode,
    ulaw_encode,
)


def _pcm(samples: list[int]) -> bytes:
    out = bytearray()
    for s in samples:
        out += int(s).to_bytes(2, "little", signed=True)
    return bytes(out)


def _samples(pcm: bytes) -> list[int]:
    return [int.from_bytes(pcm[i : i + 2], "little", signed=True) for i in range(0, len(pcm), 2)]


class TestUlawCodec:
    def test_silence_is_0xff(self) -> None:
        assert ulaw_encode(_pcm([0])) == ULAW_SILENCE
        assert ulaw_decode(ULAW_SILENCE) == _pcm([0])

    def test_round_trip_within_quantization_error(self) -> None:
        values = [-32768, -32635, -10000, -1000, -137, -1, 0, 1, 137, 1000, 10000, 32635, 32767]
        decoded = _samples(ulaw_decode(ulaw_encode(_pcm(values))))
        for original, got in zip(values, decoded, strict=True):
            clipped = max(-32635, min(32635, original))
            # μ-law is logarithmic: error grows with amplitude (~4% of value).
            tolerance = max(33, abs(clipped) * 0.04)
            assert abs(got - clipped) <= tolerance, f"{original} -> {got}"

    def test_sign_symmetry(self) -> None:
        for v in (1, 100, 5000, 30000):
            pos = _samples(ulaw_decode(ulaw_encode(_pcm([v]))))[0]
            neg = _samples(ulaw_decode(ulaw_encode(_pcm([-v]))))[0]
            assert neg == -pos

    def test_decode_covers_all_bytes(self) -> None:
        pcm = ulaw_decode(bytes(range(256)))
        assert len(pcm) == 512
        values = _samples(pcm)
        assert min(values) < -30000
        assert max(values) > 30000

    def test_encode_drops_trailing_odd_byte(self) -> None:
        assert ulaw_encode(b"\x00\x00\x01") == ULAW_SILENCE


class TestPcmResampler:
    def test_downsample_halves_length(self) -> None:
        pytest.importorskip("av")
        resampler = PcmResampler(16_000, TELEPHONY_SAMPLE_RATE)
        out = b""
        # 100ms per chunk; multiple chunks so filter delay amortizes.
        for _ in range(5):
            out += resampler.process(b"\x00\x01" * 1600)
        # 500ms in → ~250ms out (4000 samples) minus a little filter delay.
        assert 3600 * 2 <= len(out) <= 4000 * 2

    def test_upsample_doubles_length(self) -> None:
        pytest.importorskip("av")
        resampler = PcmResampler(TELEPHONY_SAMPLE_RATE, 16_000)
        out = b""
        for _ in range(5):
            out += resampler.process(b"\x00\x01" * 800)
        assert 7200 * 2 <= len(out) <= 8000 * 2

    def test_empty_chunk_is_noop(self) -> None:
        pytest.importorskip("av")
        resampler = PcmResampler(TELEPHONY_SAMPLE_RATE, 16_000)
        assert resampler.process(b"") == b""

    def test_reset_drops_filter_tail(self) -> None:
        """After a barge-in reset, no residue of the old audio may leak out."""
        pytest.importorskip("av")
        resampler = PcmResampler(16_000, TELEPHONY_SAMPLE_RATE)
        loud = (16_000).to_bytes(2, "little", signed=True) * 1600
        silence = b"\x00\x00" * 1600

        resampler.process(loud)
        resampler.reset()
        out = _samples(resampler.process(silence))
        assert out
        assert max(abs(v) for v in out) == 0

        # Control: without a reset the FIR delay line does leak the tail —
        # if this stops failing, the reset (and its call site) is dead code.
        leaky = PcmResampler(16_000, TELEPHONY_SAMPLE_RATE)
        leaky.process(loud)
        leaked = _samples(leaky.process(silence))
        assert max(abs(v) for v in leaked) > 1_000


class TestTelephonyPlaybackTracker:
    def test_interruption_fires_clear_and_freezes_axis(self) -> None:
        now = [0.0]
        cleared: list[int] = []
        tracker = TelephonyPlaybackTracker(32_000, on_clear=lambda: cleared.append(1))
        tracker._clock = lambda: now[0]  # deterministic time

        tracker.on_audio_emitted(32_000)  # 1s of audio scheduled
        now[0] = 0.25
        tracker.on_interrupted()
        assert cleared == [1]
        frozen = tracker.played_bytes
        assert frozen == 8_000  # 250ms worth
        now[0] = 10.0
        assert tracker.played_bytes == frozen  # axis frozen after clear

    def test_mark_acks_bound_the_estimate(self) -> None:
        now = [0.0]
        tracker = TelephonyPlaybackTracker(32_000, on_clear=lambda: None)
        tracker._clock = lambda: now[0]

        tracker.on_audio_emitted(32_000)
        now[0] = 0.5
        # Provider says only 100ms actually played: ack caps the estimate.
        tracker.on_playback_ack(100.0)
        assert tracker.played_bytes == 3_200
        assert tracker.ack_received

    def test_ack_received_before_first_mark_echo(self) -> None:
        # Marks are the native clock; the first echo cannot land before the
        # carrier has played the first media frame, which is after TTS-end
        # metrics snapshot on turn 1. Report the transport as ack-capable
        # immediately so that snapshot is not a false "marks broken".
        tracker = TelephonyPlaybackTracker(32_000, on_clear=lambda: None)
        assert tracker.ack_received is True
