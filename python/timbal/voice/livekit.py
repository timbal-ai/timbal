"""LiveKit transport primitives for :class:`~timbal.voice.VoiceSession`.

Requires the ``timbal[voice-livekit]`` extra. Imported only by the LiveKit
session driver and its tests, so ``timbal.voice`` stays importable without
the extra — same split as :mod:`timbal.voice.webrtc`.

LiveKit inverts the WebRTC direction: we *push* PCM into
``rtc.AudioSource.capture_frame()``, which returns when the frame is queued,
not when it is played. :class:`LkPacedSource` restores the
:class:`~timbal.voice.webrtc.PcmQueueTrack` contract — a pusher task emits
20 ms frames on a monotonic schedule at most ``_LEAD_SECS`` ahead of
realtime, and ``played_bytes`` counts only frames whose scheduled playout
instant has already passed. That number is what
:meth:`~timbal.voice.session.VoiceSession.interrupt` uses for ``heard_text``
and the recorded agent tail.

Construct :class:`LkPacedSource` (and ``rtc.Room``) inside the running event
loop, never at import or module scope. ``AudioSource.__init__`` captures
``asyncio.get_event_loop()``; a source built before a CRIU restore holds a
loop that does not survive it, and the first ``capture_frame`` hangs with no
error (measured, phase 0).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import structlog

from .playback import PlaybackTracker

logger = structlog.get_logger("timbal.voice.livekit")

_FRAME_SECS = 0.02
_LEAD_SECS = 0.04  # spike: paced mode, 40 ms lead, queue_size_ms=200
_QUEUE_SIZE_MS = 200


def _rtc():
    try:
        from livekit import rtc
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "The LiveKit voice transport requires the timbal[voice-livekit] extra: "
            "uv pip install 'timbal[voice-livekit]'"
        ) from e
    return rtc


class LkPacedSource:
    """Server-paced TTS downlink over ``rtc.AudioSource``.

    ``write()`` is faster-than-realtime; a pusher task emits 20 ms frames on
    a monotonic schedule at most ``_LEAD_SECS`` ahead of realtime. Silence
    padding on underrun does **not** advance ``played_bytes`` — same contract
    as :class:`~timbal.voice.webrtc.PcmQueueTrack`.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        source: object | None = None,
        clock: Callable[[], float] = time.monotonic,
        frame_factory: Callable[[bytes], object] | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._sr = sample_rate
        self._spf = int(sample_rate * _FRAME_SECS)
        self._frame_bytes = self._spf * 2
        self._clock = clock
        self._frame_factory = frame_factory
        self._source = (
            source
            if source is not None
            else _rtc().AudioSource(sample_rate, 1, queue_size_ms=_QUEUE_SIZE_MS)
        )
        self._buf = bytearray()
        self._emitted_bytes = 0
        self._played_frozen = 0
        self._started_at: float | None = None
        self._gen = 0
        self._closed = False
        self._task: asyncio.Task | None = None
        self.tracker = LkPacedPlaybackTracker(self)

    @property
    def source(self) -> object:
        return self._source

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def buffered_bytes(self) -> int:
        return len(self._buf)

    @property
    def played_bytes(self) -> int:
        return self._played_bytes_now()

    def write(self, pcm: bytes) -> None:
        if pcm and not self._closed:
            self._buf.extend(pcm)

    def flush(self) -> int:
        """Barge-in: drop queued PCM and the ≤200 ms already inside libwebrtc.

        Returns bytes discarded from the Python buffer. In-flight captured
        frames are dropped by ``clear_queue`` and never counted as played.
        """
        self._gen += 1
        dropped = len(self._buf)
        self._buf.clear()
        clear = getattr(self._source, "clear_queue", None)
        if clear is not None:
            clear()
        self._played_frozen = self._played_bytes_now()
        self._emitted_bytes = self._played_frozen
        self._started_at = None
        return dropped

    def _played_bytes_now(self) -> int:
        if self._started_at is None:
            return self._played_frozen
        elapsed = max(0.0, self._clock() - self._started_at - _LEAD_SECS)
        by_clock = int(round(elapsed * self._sr)) * 2
        return min(self._emitted_bytes, self._played_frozen + by_clock)

    def _freeze_if_drained(self) -> None:
        """On underrun, once the clock catches everything emitted: freeze + re-anchor.

        Without this the wall-clock term keeps running through the pause, so
        the next burst's frames count as played the instant they are queued —
        the ``_LEAD_SECS`` subtraction would only ever apply to the first
        burst, inflating barge-in ``heard_text`` and the recorded agent tail
        on later turns by audio still sitting in the lead buffer.
        """
        if self._started_at is not None and self._played_bytes_now() >= self._emitted_bytes:
            self._played_frozen = self._emitted_bytes
            self._started_at = None

    def _make_frame(self, pcm: bytes) -> object:
        if self._frame_factory is not None:
            return self._frame_factory(pcm)
        if len(pcm) < self._frame_bytes:
            pcm = pcm + b"\x00" * (self._frame_bytes - len(pcm))
        return _rtc().AudioFrame(pcm, self._sr, 1, self._spf)

    async def start(self) -> None:
        if self._task is None and not self._closed:
            self._task = asyncio.create_task(self._push_loop(), name="lk-paced-source")

    async def aclose(self) -> None:
        self._closed = True
        self._gen += 1
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _push_loop(self) -> None:
        next_at = self._clock()
        while not self._closed:
            gen = self._gen
            chunk = bytes(self._buf[: self._frame_bytes])
            del self._buf[: len(chunk)]
            real_bytes = len(chunk)
            if real_bytes < self._frame_bytes:
                chunk = chunk + b"\x00" * (self._frame_bytes - real_bytes)
            await self._source.capture_frame(self._make_frame(chunk))
            if self._gen != gen:
                # flushed during capture: clear_queue dropped this frame.
                next_at = self._clock()
                continue
            if real_bytes:
                if self._started_at is None:
                    self._started_at = self._clock()
                self._emitted_bytes += real_bytes
            else:
                self._freeze_if_drained()
            next_at += _FRAME_SECS
            sleep_for = next_at - self._clock() - _LEAD_SECS
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


class LkPacedPlaybackTracker(PlaybackTracker):
    """Playback position read from :class:`LkPacedSource`'s pacing clock.

    ``on_audio_emitted`` is a no-op: the session emits faster than real time
    and counting happens on the scheduled playout axis, inside the source.
    """

    def __init__(self, source: LkPacedSource) -> None:
        self._source = source

    def on_audio_emitted(self, num_bytes: int) -> None:
        pass

    def on_interrupted(self) -> None:
        dropped = self._source.flush()
        if dropped:
            logger.debug("livekit_playback_flushed", dropped_bytes=dropped)

    @property
    def ack_received(self) -> bool:
        return True

    @property
    def played_bytes(self) -> int:
        return self._source.played_bytes

    @property
    def is_playing(self) -> bool:
        return (
            self._source.buffered_bytes > 0
            or self._source._emitted_bytes > self._source.played_bytes
        )


async def audio_stream_to_pcm(stream: object) -> AsyncIterator[bytes]:
    """Uplink adapter: ``rtc.AudioStream`` frames → PCM16 mono for ``session.run()``.

    The SDK resamples to the requested rate; no ``av.AudioResampler`` hop.
    """
    async for ev in stream:
        data = bytes(ev.frame.data)
        if data:
            yield data
