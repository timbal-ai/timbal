"""Client playback tracking for :class:`~timbal.voice.VoiceSession`.

TTS produces PCM faster than real time, so the client buffers audio the user
has not heard yet. The session needs the *playback* position — not the emit
position — for two things:

* barge-in gating (is the assistant still audible?), and
* interruption truncation (which part of the reply did the user actually hear,
  so transcript and memory match reality).

:class:`PlaybackTracker` is the transport-agnostic seam. The session only ever
calls this interface; how the position is known is the transport's business:

* :class:`BufferedPlaybackTracker` (default) models a gapless client-side
  playback queue with a wall-clock schedule — the same estimate the session
  used historically — and corrects it with optional client acks
  (``on_playback_ack``), the way OpenAI Realtime's ``conversation.item.truncate``
  flow relies on client-reported ``audio_end_ms``.
* A future paced transport (WebRTC, telephony) that pushes audio at real-time
  rate can implement the interface from its own pacing clock, LiveKit-style,
  without any session changes.

All positions are in **bytes on the played axis**: audio that was discarded on
interruption (emitted but never played) is not counted.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence


class PlaybackTracker(ABC):
    """How much of the emitted TTS audio the client has actually played."""

    @abstractmethod
    def on_audio_emitted(self, num_bytes: int) -> None:
        """Called by the session for every PCM chunk handed to the transport."""

    @abstractmethod
    def on_interrupted(self) -> None:
        """Client playback buffer was (or is about to be) cleared; unplayed audio is discarded."""

    def on_playback_ack(self, played_ms: float) -> None:  # noqa: B027
        """Optional client correction: cumulative milliseconds actually played this session."""

    @property
    def ack_received(self) -> bool:
        """True when the played position is client-truth, not a schedule estimate.

        Estimate-only trackers return False until the first ack arrives; paced
        transports (WebRTC) that know the position natively should return True.
        """
        return False

    @property
    @abstractmethod
    def played_bytes(self) -> int:
        """Best estimate of total bytes played so far (session lifetime, discarded audio excluded)."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """True if the client likely still has queued audio to play."""


class BufferedPlaybackTracker(PlaybackTracker):
    """Wall-clock estimate of a gapless playback queue, corrected by client acks.

    Without acks this reproduces the session's historical behavior: every
    emitted chunk is scheduled at ``max(now, queue_end)`` and assumed to play
    in real time. When the client reports ``played_ms`` acks, the position is
    the ack extrapolated forward at real-time rate, capped by the schedule
    estimate (the schedule is an upper bound — audio cannot have played faster
    than gapless real time).
    """

    def __init__(
        self,
        bytes_per_second: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be positive")
        self._bps = bytes_per_second
        self._clock = clock
        # Bytes on the played axis that will play unless interrupted.
        self._scheduled_bytes = 0
        # When the gapless queue drains (monotonic).
        self._playing_until = 0.0
        # Last client ack: (bytes, at_time). None until the first ack.
        self._last_ack: tuple[int, float] | None = None

    def on_audio_emitted(self, num_bytes: int) -> None:
        if num_bytes <= 0:
            return
        now = self._clock()
        start = max(now, self._playing_until)
        self._playing_until = start + num_bytes / self._bps
        self._scheduled_bytes += num_bytes

    def on_playback_ack(self, played_ms: float) -> None:
        ack_bytes = max(0, int(played_ms / 1000 * self._bps))
        now = self._clock()
        if self._last_ack is not None and ack_bytes < self._last_ack[0]:
            # Acks are cumulative; never move backwards.
            return
        self._last_ack = (ack_bytes, now)

    def on_interrupted(self) -> None:
        # Whatever is still queued client-side will be dropped: freeze the
        # played axis at the current position and end the schedule now.
        self._scheduled_bytes = self.played_bytes
        self._playing_until = self._clock()

    @property
    def ack_received(self) -> bool:
        return self._last_ack is not None

    @property
    def played_bytes(self) -> int:
        now = self._clock()
        remaining = max(0.0, self._playing_until - now) * self._bps
        estimate = max(0, int(self._scheduled_bytes - remaining))
        if self._last_ack is None:
            return estimate
        ack_bytes, ack_at = self._last_ack
        extrapolated = ack_bytes + max(0.0, now - ack_at) * self._bps
        return min(estimate, int(extrapolated))

    @property
    def is_playing(self) -> bool:
        return self._clock() < self._playing_until


def map_played_bytes_to_text(segments: Sequence[tuple[str, int]], played_bytes: int) -> str:
    """Reconstruct the text the user actually heard from per-segment audio sizes.

    ``segments`` are ``(spoken_text, emitted_bytes)`` records in playback order
    for a single turn. Fully played segments contribute their whole text; the
    segment the playhead landed in contributes a proportional prefix, snapped
    back to a word boundary.
    """
    parts: list[str] = []
    remaining = played_bytes
    for text, num_bytes in segments:
        if remaining <= 0:
            break
        if num_bytes <= 0:
            continue
        if remaining >= num_bytes:
            parts.append(text)
            remaining -= num_bytes
            continue
        cut = int(len(text) * (remaining / num_bytes))
        partial = text[:cut]
        if 0 < cut < len(text) and " " in partial:
            partial = partial.rsplit(" ", 1)[0]
        parts.append(partial)
        break
    return "".join(parts).strip()
