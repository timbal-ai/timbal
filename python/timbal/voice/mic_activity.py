"""Is the caller talking right now, and for how long have they talked? From the mic level alone.

Two behaviours need it and neither can borrow it from the turn detector:
``adaptive_tempo`` wants the seconds the caller spent saying an utterance (its
words over that time is how fast they talk), and ``backchannel`` wants to know
that the caller has just paused, before any transcript says so. Silero answers
both, but it runs only when local endpointing does: a session whose provider
decides the end of the turn (Deepgram Flux, the default) has none.

So this is a level meter. Each 20 ms of PCM16 is speech when it is
:attr:`MicActivity.MARGIN_DB` above the noise floor, and the floor is the
quietest 20 ms of the last three seconds: speech has a quiet instant between
words every second or two, and a hum, a street or a fan does not. A line that
gets noisier is therefore speech for three seconds at most. No model, no numpy;
a quarter of the samples are read, which is under a millisecond of work per
second of call.

It is wrong where a level meter is wrong: a television behind the caller is
speech to it. The two features that read it fail quietly then (a caller that
never seems to pause gets no listening sound; one that seems to talk slowly
gets a reply up to 10 % slower).
"""

from __future__ import annotations

import math
import sys
import time
from array import array
from collections import deque
from collections.abc import Callable


class MicActivity:
    """Speech and silence of one mic stream, from its level."""

    FRAME_SECS = 0.02
    MARGIN_DB = 12.0
    """How far above the noise floor a frame must be to count as speech."""
    MIN_SPEECH_DBFS = -50.0
    """Nothing quieter is speech, however quiet the line."""
    MAX_FLOOR_DBFS = -30.0
    """The floor is never read higher: a voice over a loud line still has to be heard."""
    WINDOW_SECS = 3.0
    """The floor is the quietest frame of this long."""
    _BUCKET_FRAMES = 25
    BRIDGE_SECS = 0.2
    """A gap this short between two stretches of speech is a consonant, not a pause."""
    STALE_SECS = 1.0
    """Without a frame for this long the meter has no opinion."""

    def __init__(self, sample_rate: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._frame_bytes = max(2, int(sample_rate * self.FRAME_SECS) * 2)
        self._clock = clock
        self._carry = b""
        # Minimum level of each finished half second of the window, and of the one under way.
        self._buckets: deque[float] = deque(
            maxlen=max(1, round(self.WINDOW_SECS / (self._BUCKET_FRAMES * self.FRAME_SECS)))
        )
        self._bucket_min: float | None = None
        self._bucket_frames = 0
        self._bridge_frames = round(self.BRIDGE_SECS / self.FRAME_SECS)
        self._loud_run = 0
        # Frames since the last one known to be speech, and frames counted as speech.
        self._quiet_frames = 0
        self._speech_frames = 0
        self._heard_speech = False
        self._last_frame_at: float | None = None

    @property
    def speech_secs(self) -> float:
        """Seconds of speech counted so far (``push(count=True)`` only), short gaps included."""
        return self._speech_frames * self.FRAME_SECS

    def push(self, chunk: bytes, *, count: bool = True) -> None:
        """Feed mic PCM16. ``count=False`` still tracks talking and pausing, without adding to :attr:`speech_secs`."""
        if not chunk:
            return
        data = self._carry + chunk
        usable = len(data) - len(data) % self._frame_bytes
        self._carry = data[usable:]
        if not usable:
            return
        samples = array("h")
        samples.frombytes(data[:usable])
        if sys.byteorder == "big":
            samples.byteswap()
        per_frame = self._frame_bytes // 2
        for start in range(0, len(samples), per_frame):
            self._frame(samples[start : start + per_frame : 4], count)
        self._last_frame_at = self._clock()

    def _frame(self, samples: array, count: bool) -> None:
        power = sum(sample * sample for sample in samples) / max(1, len(samples))
        level = 10.0 * math.log10(power / (32768.0 * 32768.0)) if power > 0 else -100.0
        self._bucket_min = level if self._bucket_min is None else min(self._bucket_min, level)
        floor = min((self._bucket_min, *self._buckets))
        self._bucket_frames += 1
        if self._bucket_frames >= self._BUCKET_FRAMES:
            self._buckets.append(self._bucket_min)
            self._bucket_min = None
            self._bucket_frames = 0
        floor = min(floor, self.MAX_FLOOR_DBFS)
        loud = level >= max(floor + self.MARGIN_DB, self.MIN_SPEECH_DBFS)
        self._loud_run = self._loud_run + 1 if loud else 0
        self._quiet_frames += 1
        # One loud frame is a click. Two in a row is a voice, and the first was part of it.
        if self._loud_run < 2:
            return
        if count:
            gap = self._quiet_frames - 2 if self._loud_run == 2 else 0
            if self._loud_run == 2:
                self._speech_frames += 2
                if self._heard_speech and gap <= self._bridge_frames:
                    self._speech_frames += gap
            else:
                self._speech_frames += 1
        self._quiet_frames = 0
        self._heard_speech = True

    @property
    def fresh(self) -> bool:
        """Mic audio arrived in the last second."""
        return self._last_frame_at is not None and self._clock() - self._last_frame_at <= self.STALE_SECS

    def quiet_secs(self) -> float | None:
        """Seconds of mic audio since the caller last made a sound. ``None`` without recent audio, or before any speech."""
        if not self.fresh or not self._heard_speech:
            return None
        return self._quiet_frames * self.FRAME_SECS

    @property
    def speaking(self) -> bool:
        """The caller is mid-word or in the short gap between two."""
        quiet = self.quiet_secs()
        return quiet is not None and quiet <= self.BRIDGE_SECS
