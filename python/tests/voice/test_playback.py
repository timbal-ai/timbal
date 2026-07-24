"""Unit tests for BufferedPlaybackTracker and heard-text mapping."""

from __future__ import annotations

from timbal.voice.playback import BufferedPlaybackTracker, map_played_bytes_to_text

BPS = 32_000  # PCM16 mono @ 16 kHz


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, secs: float) -> None:
        self.t += secs


class TestEstimateMode:
    def test_idle_tracker(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        assert tr.played_bytes == 0
        assert not tr.is_playing

    def test_position_advances_with_wall_clock(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(BPS)  # 1 second of audio
        assert tr.is_playing
        assert tr.played_bytes == 0
        clock.advance(0.5)
        assert tr.played_bytes == BPS // 2
        clock.advance(0.6)
        assert tr.played_bytes == BPS
        assert not tr.is_playing

    def test_gapless_queueing(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(BPS)
        tr.on_audio_emitted(BPS)  # queued behind the first second
        clock.advance(1.5)
        assert tr.played_bytes == int(BPS * 1.5)
        assert tr.is_playing
        clock.advance(1.0)
        assert tr.played_bytes == 2 * BPS
        assert not tr.is_playing

    def test_interrupt_discards_unplayed(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(2 * BPS)
        clock.advance(0.5)
        tr.on_interrupted()
        assert tr.played_bytes == BPS // 2
        assert not tr.is_playing
        # Position frozen after the discard.
        clock.advance(5.0)
        assert tr.played_bytes == BPS // 2

    def test_emit_after_interrupt_resumes_cleanly(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(2 * BPS)
        clock.advance(0.25)
        tr.on_interrupted()
        played_before = tr.played_bytes
        tr.on_audio_emitted(BPS)
        clock.advance(1.0)
        assert tr.played_bytes == played_before + BPS


class TestAckMode:
    def test_ack_caps_the_estimate(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(2 * BPS)
        clock.advance(1.0)
        # Estimate says 1s played; the client reports only 0.5s (slow start).
        tr.on_playback_ack(500.0)
        assert tr.played_bytes == BPS // 2

    def test_ack_extrapolates_forward(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(2 * BPS)
        tr.on_playback_ack(0.0)
        clock.advance(0.5)
        # Ack + 0.5s of real-time playback.
        assert tr.played_bytes == BPS // 2

    def test_extrapolation_capped_by_schedule(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(BPS // 2)  # only 0.5s of audio exists
        tr.on_playback_ack(0.0)
        clock.advance(2.0)
        assert tr.played_bytes == BPS // 2

    def test_backwards_ack_ignored(self) -> None:
        clock = FakeClock()
        tr = BufferedPlaybackTracker(BPS, clock=clock)
        tr.on_audio_emitted(BPS)
        clock.advance(1.0)
        tr.on_playback_ack(800.0)
        tr.on_playback_ack(300.0)  # out-of-order / stale — must not move backwards
        assert tr.played_bytes == int(0.8 * BPS)

    def test_ack_received_flips_on_first_ack(self) -> None:
        tr = BufferedPlaybackTracker(BPS, clock=FakeClock())
        assert tr.ack_received is False
        tr.on_playback_ack(0.0)
        assert tr.ack_received is True


class TestMapPlayedBytesToText:
    SEGMENTS = [("Hello there! ", 1000), ("How are you today?", 2000)]

    def test_nothing_played(self) -> None:
        assert map_played_bytes_to_text(self.SEGMENTS, 0) == ""

    def test_everything_played(self) -> None:
        assert map_played_bytes_to_text(self.SEGMENTS, 3000) == "Hello there! How are you today?"

    def test_overplayed_clamps(self) -> None:
        assert map_played_bytes_to_text(self.SEGMENTS, 10_000) == "Hello there! How are you today?"

    def test_partial_segment_snaps_to_word_boundary(self) -> None:
        # 1000 (first segment) + 1000 of 2000 → half of "How are you today?" = 9 chars
        # = "How are y" → snapped back to "How are".
        heard = map_played_bytes_to_text(self.SEGMENTS, 2000)
        assert heard == "Hello there! How are"

    def test_first_segment_partial(self) -> None:
        heard = map_played_bytes_to_text(self.SEGMENTS, 500)
        assert heard == "Hello"

    def test_zero_byte_segments_skipped(self) -> None:
        segments = [("never spoken", 0), ("spoken", 100)]
        assert map_played_bytes_to_text(segments, 100) == "spoken"
