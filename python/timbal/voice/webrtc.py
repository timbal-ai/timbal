"""WebRTC transport primitives for :class:`~timbal.voice.VoiceSession`.

Requires the ``timbal[voice]`` extra (aiortc ships with it). This module is
imported only by the RTC server route and its tests, so ``timbal.voice``
stays importable without the extra.

The session itself needs nothing from here — it already exposes the three
transport seams (audio-in iterable, events-out iterator,
:class:`~timbal.voice.playback.PlaybackTracker`). What WebRTC changes is
*who* paces the audio: the server's RTP sender pulls 20ms frames in real
time, so the played position is transport-truth instead of a client-acked
estimate, and a barge-in can drop the unsent tail server-side instead of
asking the client to clear a buffer.

Positions are exact up to the client's jitter buffer (tens of ms) — a frame
handed to the RTP sender is on the caller's speaker almost immediately.
"""

from __future__ import annotations

import asyncio
import fractions
import time
from collections.abc import AsyncIterator

import structlog

try:
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame, AudioResampler
except ImportError as e:  # pragma: no cover — exercised only without the extra
    raise ImportError(
        "The WebRTC voice transport requires the timbal[voice] extra: "
        "uv pip install 'timbal[voice]'"
    ) from e

from .playback import PlaybackTracker

logger = structlog.get_logger("timbal.voice.webrtc")

_FRAME_SECS = 0.02


class PcmQueueTrack(MediaStreamTrack):
    """Downlink audio: a paced 20ms-frame source over a PCM16 byte queue.

    The session pushes TTS PCM (faster than real time) via :meth:`write`; the
    RTP sender pulls frames via :meth:`recv`, which sleeps to hold a real-time
    cadence. When the queue is empty, silence frames keep the stream
    continuous — receivers treat a stalled track as a broken one, the same
    lesson the replay harness feeder learned about STT websockets.

    Frames are emitted at the source rate; aiortc's Opus encoder resamples to
    48kHz itself, so no resampling happens here.

    ``played_bytes`` counts only *real* PCM dequeued into sent frames —
    silence padding and audio dropped by :meth:`flush` never touch the played
    axis. That is exactly the contract interruption truncation needs.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 16_000) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._sample_rate = sample_rate
        self._samples_per_frame = int(sample_rate * _FRAME_SECS)
        self._frame_bytes = self._samples_per_frame * 2
        self._buf = bytearray()
        self._played_bytes = 0
        self._timestamp = 0
        self._started_at: float | None = None

    def write(self, pcm: bytes) -> None:
        """Queue PCM16 mono bytes for paced sending."""
        if pcm:
            self._buf.extend(pcm)

    def flush(self) -> int:
        """Barge-in: drop everything not yet handed to the RTP sender.

        Returns the number of bytes discarded. The played position is
        untouched — dropped audio was never heard.
        """
        dropped = len(self._buf)
        self._buf.clear()
        return dropped

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def buffered_bytes(self) -> int:
        """Real PCM queued but not yet sent."""
        return len(self._buf)

    @property
    def played_bytes(self) -> int:
        """Real PCM handed to the RTP sender so far (silence excluded)."""
        return self._played_bytes

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._started_at is None:
            self._started_at = time.monotonic()
        else:
            self._timestamp += self._samples_per_frame
            wait = self._started_at + (self._timestamp / self._sample_rate) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

        chunk = bytes(self._buf[: self._frame_bytes])
        del self._buf[: len(chunk)]
        self._played_bytes += len(chunk)
        if len(chunk) < self._frame_bytes:
            # Underrun (or idle): pad with silence. Not counted as played.
            chunk += b"\x00" * (self._frame_bytes - len(chunk))

        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.planes[0].update(chunk)
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        return frame


class PacedPlaybackTracker(PlaybackTracker):
    """Playback position read from the transport's pacing clock.

    No estimates, no client acks: the track knows exactly which bytes it has
    handed to the RTP sender. ``on_audio_emitted`` is a no-op because the
    session emits faster than real time — counting happens on dequeue, inside
    the track.
    """

    def __init__(self, track: PcmQueueTrack) -> None:
        self._track = track

    def on_audio_emitted(self, num_bytes: int) -> None:
        pass

    def on_interrupted(self) -> None:
        dropped = self._track.flush()
        if dropped:
            logger.debug("webrtc_playback_flushed", dropped_bytes=dropped)

    @property
    def ack_received(self) -> bool:
        return True

    @property
    def played_bytes(self) -> int:
        return self._track.played_bytes

    @property
    def is_playing(self) -> bool:
        return self._track.buffered_bytes > 0


async def track_to_pcm(track: MediaStreamTrack, sample_rate: int = 16_000) -> AsyncIterator[bytes]:
    """Uplink adapter: decoded mic frames → PCM16 mono bytes for ``session.run()``.

    Browser audio arrives Opus-decoded at 48kHz; the session's STT expects
    PCM16 mono at ``sample_rate``. Ends cleanly when the track does.
    """
    resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            return
        for out in resampler.resample(frame):
            # Plane buffers are alignment-padded; only samples*2 bytes are audio.
            data = bytes(out.planes[0])[: out.samples * 2]
            if data:
                yield data
