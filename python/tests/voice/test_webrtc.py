"""WebRTC transport primitives: pacing, played-axis accounting, resampling.

The riskiest part of the WebRTC transport is the downlink pacing math —
``recv()`` timestamps drive both what the caller hears and what
interruption truncation believes was heard. These tests pin it in isolation,
before any signaling exists.
"""

from __future__ import annotations

import asyncio
import fractions
import time

import pytest

pytest.importorskip("aiortc", reason="timbal[voice] extra (aiortc) not installed")

from aiortc.mediastreams import MediaStreamError  # noqa: E402
from av import AudioFrame  # noqa: E402
from timbal.voice.webrtc import PacedPlaybackTracker, PcmQueueTrack, track_to_pcm  # noqa: E402

_SR = 16_000
_FRAME_BYTES = int(_SR * 0.02) * 2  # 20ms of PCM16 mono


class TestPcmQueueTrack:
    async def test_first_frame_is_immediate_then_paced_at_real_time(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        t0 = time.monotonic()
        for _ in range(6):
            await track.recv()
        elapsed = time.monotonic() - t0
        # 6 frames = first immediate + 5 paced gaps of 20ms.
        assert elapsed >= 0.5 * 5 * 0.02  # generous lower bound for CI jitter
        assert elapsed < 1.0

    async def test_empty_queue_yields_silence_and_no_played_bytes(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        frame = await track.recv()
        assert bytes(frame.planes[0])[: _FRAME_BYTES] == b"\x00" * _FRAME_BYTES
        assert frame.samples == int(_SR * 0.02)
        assert frame.sample_rate == _SR
        assert frame.time_base == fractions.Fraction(1, _SR)
        assert track.played_bytes == 0

    async def test_played_bytes_counts_real_pcm_only(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        track.write(b"\x01\x02" * (_FRAME_BYTES // 2))  # exactly one frame
        track.write(b"\x03\x04" * 10)  # 20 bytes: a partial second frame
        f1 = await track.recv()
        assert bytes(f1.planes[0])[:_FRAME_BYTES] == b"\x01\x02" * (_FRAME_BYTES // 2)
        assert track.played_bytes == _FRAME_BYTES
        f2 = await track.recv()
        data = bytes(f2.planes[0])[:_FRAME_BYTES]
        assert data[:20] == b"\x03\x04" * 10
        assert data[20:] == b"\x00" * (_FRAME_BYTES - 20)  # silence pad
        assert track.played_bytes == _FRAME_BYTES + 20  # pad not counted

    async def test_pts_advances_by_frame_samples(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        f1 = await track.recv()
        f2 = await track.recv()
        f3 = await track.recv()
        step = int(_SR * 0.02)
        assert (f1.pts, f2.pts, f3.pts) == (0, step, 2 * step)

    async def test_flush_drops_unsent_audio_and_freezes_played_axis(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        track.write(b"\x01\x02" * (_FRAME_BYTES // 2) * 3)
        await track.recv()
        assert track.played_bytes == _FRAME_BYTES
        dropped = track.flush()
        assert dropped == 2 * _FRAME_BYTES
        assert track.buffered_bytes == 0
        assert track.played_bytes == _FRAME_BYTES  # dropped audio was never heard
        frame = await track.recv()  # back to silence, still paced
        assert bytes(frame.planes[0])[:_FRAME_BYTES] == b"\x00" * _FRAME_BYTES

    async def test_stopped_track_raises_media_stream_error(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        track.stop()
        with pytest.raises(MediaStreamError):
            await track.recv()


class TestPacedPlaybackTracker:
    def test_position_is_native_truth(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        tracker = PacedPlaybackTracker(track)
        assert tracker.ack_received is True
        assert tracker.played_bytes == 0
        tracker.on_playback_ack(9999.0)  # acks carry no information here
        assert tracker.played_bytes == 0

    async def test_reads_through_to_the_track(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        tracker = PacedPlaybackTracker(track)
        track.write(b"\x01\x02" * _FRAME_BYTES)  # two frames
        tracker.on_audio_emitted(2 * _FRAME_BYTES)  # no-op: counting is on dequeue
        assert tracker.played_bytes == 0
        assert tracker.is_playing is True
        await track.recv()
        assert tracker.played_bytes == _FRAME_BYTES

    async def test_interruption_flushes_the_queue(self) -> None:
        track = PcmQueueTrack(sample_rate=_SR)
        tracker = PacedPlaybackTracker(track)
        track.write(b"\x01\x02" * _FRAME_BYTES * 4)
        await track.recv()
        tracker.on_interrupted()
        assert tracker.is_playing is False
        assert track.buffered_bytes == 0
        assert tracker.played_bytes == _FRAME_BYTES


class _FrameSourceTrack:
    """Duck-typed uplink track: replays canned AudioFrames, then ends."""

    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)

    async def recv(self) -> AudioFrame:
        if not self._frames:
            raise MediaStreamError
        await asyncio.sleep(0)
        return self._frames.pop(0)


def _mic_frame(pts: int, *, rate: int = 48_000, samples: int = 960, fill: bytes = b"\x11\x22") -> AudioFrame:
    frame = AudioFrame(format="s16", layout="mono", samples=samples)
    frame.planes[0].update(fill * samples)
    frame.pts = pts
    frame.sample_rate = rate
    frame.time_base = fractions.Fraction(1, rate)
    return frame


class TestTrackToPcm:
    async def test_resamples_48k_mic_to_16k_session_pcm(self) -> None:
        n_frames = 25  # 500ms of 48k audio
        frames = [_mic_frame(i * 960) for i in range(n_frames)]
        out = b"".join([chunk async for chunk in track_to_pcm(_FrameSourceTrack(frames), sample_rate=_SR)])
        expected = int(n_frames * 0.02 * _SR) * 2  # 500ms at 16k PCM16
        # The resampler's filter delay holds back a tail; allow one frame.
        assert abs(len(out) - expected) <= _FRAME_BYTES
        assert len(out) % 2 == 0

    async def test_ends_cleanly_when_the_track_ends(self) -> None:
        agen = track_to_pcm(_FrameSourceTrack([]), sample_rate=_SR)
        chunks = [c async for c in agen]
        assert chunks == []

    async def test_passthrough_at_native_rate(self) -> None:
        frames = [_mic_frame(i * 320, rate=_SR, samples=320) for i in range(10)]
        out = b"".join([chunk async for chunk in track_to_pcm(_FrameSourceTrack(frames), sample_rate=_SR)])
        assert abs(len(out) - 10 * 320 * 2) <= _FRAME_BYTES
