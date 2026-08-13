"""LiveKit paced source: played-axis accounting without the FFI.

``LkPacedSource`` accepts an injected source + frame factory so these tests
run without ``timbal[voice-livekit]``. The risk is the same as WebRTC: the
pacing math drives barge-in ``heard_text``.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from timbal.voice.livekit import LkPacedPlaybackTracker, LkPacedSource, audio_stream_to_pcm

_SR = 16_000
_FRAME_BYTES = int(_SR * 0.02) * 2  # 640
_LEAD_SECS = 0.04


class FakeSource:
    def __init__(self) -> None:
        self.captured: list[bytes] = []
        self.clears = 0
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def capture_frame(self, frame: object) -> None:
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        data = getattr(frame, "data", frame)
        self.captured.append(bytes(data) if not isinstance(data, bytes) else data)

    def clear_queue(self) -> None:
        self.clears += 1


def _source(**kwargs: object) -> tuple[LkPacedSource, FakeSource]:
    fake = FakeSource()
    src = LkPacedSource(
        _SR,
        source=fake,
        frame_factory=lambda pcm: SimpleNamespace(data=pcm),
        **kwargs,
    )
    return src, fake


class TestLkPacedSourceClock:
    def test_played_bytes_zero_before_any_real_audio(self) -> None:
        src, _ = _source()
        assert src.played_bytes == 0
        assert src.buffered_bytes == 0

    def test_clock_subtracts_lead_and_caps_at_emitted(self) -> None:
        t = [0.0]
        src, _ = _source(clock=lambda: t[0])
        src._started_at = 0.0
        src._emitted_bytes = 3 * _FRAME_BYTES
        t[0] = _LEAD_SECS
        assert src.played_bytes == 0
        t[0] = _LEAD_SECS + 0.02
        assert src.played_bytes == _FRAME_BYTES
        t[0] = _LEAD_SECS + 10.0
        assert src.played_bytes == 3 * _FRAME_BYTES  # cap at emitted

    def test_flush_freezes_played_axis_and_clears_libwebrtc(self) -> None:
        t = [0.0]
        src, fake = _source(clock=lambda: t[0])
        src.write(b"\x01\x02" * (_FRAME_BYTES // 2) * 4)
        src._started_at = 0.0
        src._emitted_bytes = _FRAME_BYTES
        t[0] = _LEAD_SECS + 0.02
        assert src.played_bytes == _FRAME_BYTES
        dropped = src.flush()
        assert dropped == 4 * _FRAME_BYTES
        assert src.buffered_bytes == 0
        assert fake.clears == 1
        assert src.played_bytes == _FRAME_BYTES
        t[0] = 100.0
        assert src.played_bytes == _FRAME_BYTES  # frozen until the next burst


class TestLkPacedSourcePusher:
    async def test_silence_underrun_is_not_played(self) -> None:
        src, fake = _source()
        await src.start()
        await asyncio.sleep(0.05)
        await src.aclose()
        assert src.played_bytes == 0
        assert len(fake.captured) >= 1
        assert fake.captured[0] == b"\x00" * _FRAME_BYTES

    async def test_real_pcm_is_captured_and_counted(self) -> None:
        src, fake = _source()
        src.write(b"\x01\x02" * (_FRAME_BYTES // 2))  # exactly one frame
        await src.start()
        deadline = time.monotonic() + 1.0
        while src.played_bytes < _FRAME_BYTES and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await src.aclose()
        assert src.played_bytes == _FRAME_BYTES
        assert fake.captured[0] == b"\x01\x02" * (_FRAME_BYTES // 2)

    async def test_flush_during_capture_does_not_count_the_frame(self) -> None:
        src, fake = _source()
        fake.gate = asyncio.Event()
        src.write(b"\x01\x02" * (_FRAME_BYTES // 2) * 2)
        await src.start()
        await asyncio.wait_for(fake.entered.wait(), timeout=1.0)
        src.flush()
        fake.gate.set()
        await asyncio.sleep(0.05)
        await src.aclose()
        assert src.played_bytes == 0
        assert fake.clears >= 1


class TestLkPacedPlaybackTracker:
    def test_position_is_native_truth(self) -> None:
        src, _ = _source()
        tracker = LkPacedPlaybackTracker(src)
        assert tracker.ack_received is True
        assert tracker.played_bytes == 0
        tracker.on_playback_ack(9999.0)
        assert tracker.played_bytes == 0

    async def test_interruption_flushes(self) -> None:
        src, fake = _source()
        tracker = src.tracker
        src.write(b"\x01\x02" * _FRAME_BYTES * 4)
        tracker.on_interrupted()
        assert tracker.is_playing is False
        assert src.buffered_bytes == 0
        assert fake.clears == 1


class TestAudioStreamToPcm:
    async def test_yields_frame_bytes_and_skips_empty(self) -> None:
        async def _stream():
            yield SimpleNamespace(frame=SimpleNamespace(data=b""))
            yield SimpleNamespace(frame=SimpleNamespace(data=b"\x01\x02"))
            yield SimpleNamespace(frame=SimpleNamespace(data=b"\x03\x04"))

        out = [c async for c in audio_stream_to_pcm(_stream())]
        assert out == [b"\x01\x02", b"\x03\x04"]
