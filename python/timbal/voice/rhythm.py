"""Streaming pause rhythm for synthesized speech.

Neural TTS spaces many words with a 120-250 ms gap and rarely makes a pause of
0.3-0.5 s. A person does the opposite: words run together (under 90 ms apart)
and now and then there is a real pause. That gap between every word is what
reads as "word, word, word" even when pitch and rate are right.

This filter sorts each internal silence by its length and reshapes it:

* shorter than ``gap_min_secs``: left alone. It is the closure of a stop
  consonant ("p", "t", "k"), and shortening it eats the consonant.
* between ``gap_min_secs`` and ``gap_max_secs``: a gap between two words. With
  probability ``join_gaps`` it is closed to 35-85 ms. With probability
  ``promote_gaps`` it becomes a pause of 0.30-0.55 s instead (the speaker who
  stops to think mid-sentence). The rest are left alone.
* ``gap_max_secs`` or longer: a comma or sentence pause. It is kept and, with
  probability ``stretch_pauses``, lengthened by 1.4-2.3x.

Speech passes through unchanged and is never delayed. Only silence is held,
until the next speech shows how long it was. A silence longer than
``MAX_HOLD_SECS`` is released as it is, so a tool wait does not sit in the
filter. Leading silence is cut to ``LEAD_SECS``. A shortened silence loses its
middle, with a short crossfade, and a lengthened one repeats its own middle,
so the room tone of the recording is kept and no digital zeros are inserted.

``numpy`` is imported on first use. It ships with ``timbal[voice]``
(onnxruntime / transformers); :mod:`timbal.voice` stays importable without it.
"""

from __future__ import annotations

import math
import random
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

FRAME_SECS = 0.010
"""Analysis frame. Silence is judged on the RMS level of each frame."""
CROSSFADE_SECS = 0.012
"""Fade across the cut when the middle of a silence is removed."""
LEAD_SECS = 0.06
"""Silence kept before the first word. Providers open with 100-300 ms of air."""
MAX_HOLD_SECS = 0.9
"""A silence this long is released without waiting to see how it ends."""
JOINED_SECS = (0.035, 0.085)
"""What a joined word gap is shortened to."""
PROMOTED_SECS = (0.30, 0.55)
"""What a word gap promoted to a pause is lengthened to."""
STRETCH_FACTOR = (1.4, 2.3)
"""How much a lengthened pause grows."""


def _numpy():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "TTS pause rhythm requires the timbal[voice] extra (numpy): uv pip install 'timbal[voice]'"
        ) from e
    return np


def _share(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number from 0 to 1")
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return float(value)


def _span(name: str, value: Any, *, floor: float) -> tuple[float, float]:
    try:
        low, high = value
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a (low, high) pair") from e
    for bound in (low, high):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
            raise ValueError(f"{name} must be a (low, high) pair of finite numbers")
    if low < floor or high < low:
        raise ValueError(f"{name} must satisfy {floor} <= low <= high")
    return float(low), float(high)


class PauseShaper:
    """One shaper per reply (or per clip) over PCM16 little-endian mono.

    ``push`` returns what can be played now and ``finish`` the rest. ``stats``
    counts what happened to each internal silence.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        join_gaps: float = 0.0,
        stretch_pauses: float = 0.0,
        promote_gaps: float = 0.0,
        gap_min_secs: float = 0.12,
        gap_max_secs: float = 0.20,
        quiet_dbfs: float = -42.0,
        joined_secs: tuple[float, float] = JOINED_SECS,
        promoted_secs: tuple[float, float] = PROMOTED_SECS,
        stretch_factor: tuple[float, float] = STRETCH_FACTOR,
        seed: int | None = None,
    ) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        self.join_gaps = _share("join_gaps", join_gaps)
        self.stretch_pauses = _share("stretch_pauses", stretch_pauses)
        self.promote_gaps = _share("promote_gaps", promote_gaps)
        if self.join_gaps + self.promote_gaps > 1:
            raise ValueError("join_gaps and promote_gaps are shares of the same word gaps: together at most 1")
        gap_min, gap_max = _span("gap_min_secs, gap_max_secs", (gap_min_secs, gap_max_secs), floor=0.0)
        if isinstance(quiet_dbfs, bool) or not isinstance(quiet_dbfs, (int, float)) or not math.isfinite(quiet_dbfs):
            raise ValueError("quiet_dbfs must be finite")
        self._joined = _span("joined_secs", joined_secs, floor=0.0)
        self._promoted = _span("promoted_secs", promoted_secs, floor=0.0)
        self._stretch = _span("stretch_factor", stretch_factor, floor=1.0)
        self._np = _numpy()
        self._rate = sample_rate
        self._frame = max(1, int(round(sample_rate * FRAME_SECS)))
        self._fade = max(1, int(round(sample_rate * CROSSFADE_SECS)))
        self._gap_min = gap_min
        self._gap_max = gap_max
        # RMS of a full-scale int16 frame, matching 20*log10(rms/32768).
        self._quiet_rms = 32768.0 * (10.0 ** (float(quiet_dbfs) / 20.0))
        self._rng = random.Random(seed)
        self._odd = b""
        self._carry = self._np.zeros(0, dtype="<i2")
        self._held: list[Any] = []
        self._held_samples = 0
        self._started = False
        self._passing = False
        self.stats = {"gaps": 0, "joined": 0, "kept": 0, "promoted": 0, "stretched": 0}

    @property
    def enabled(self) -> bool:
        return self.join_gaps > 0 or self.stretch_pauses > 0 or self.promote_gaps > 0

    def push(self, chunk: bytes) -> bytes:
        if not self.enabled:
            return chunk
        np = self._np
        data = self._odd + chunk
        usable = len(data) - (len(data) % 2)
        self._odd = data[usable:]
        if usable <= 0:
            return b""
        samples = np.concatenate([self._carry, np.frombuffer(data[:usable], dtype="<i2")])
        frame = self._frame
        frames = len(samples) // frame
        self._carry = samples[frames * frame :]
        out: list[Any] = []
        for index in range(frames):
            block = samples[index * frame : (index + 1) * frame]
            if self._is_quiet(block):
                if self._passing:
                    out.append(block)
                    continue
                self._held.append(block)
                self._held_samples += len(block)
                if self._held_samples >= MAX_HOLD_SECS * self._rate:
                    # Too long to be a pause inside a sentence (a tool wait, a
                    # provider stall). Release it and stop holding.
                    out.append(self._release(long_silence=True))
                    self._passing = True
                continue
            if self._held:
                out.append(self._release())
            self._passing = False
            self._started = True
            out.append(block)
        if not out:
            return b""
        return np.concatenate(out).astype("<i2", copy=False).tobytes()

    def finish(self) -> bytes:
        """End of the reply: held silence and the last partial frame leave unchanged."""
        if not self.enabled:
            return b""
        np = self._np
        parts = [*self._held]
        if len(self._carry):
            parts.append(self._carry)
        self._held = []
        self._held_samples = 0
        self._carry = np.zeros(0, dtype="<i2")
        self._odd = b""
        if not parts:
            return b""
        return np.concatenate(parts).astype("<i2", copy=False).tobytes()

    def _is_quiet(self, block: Any) -> bool:
        np = self._np
        values = block.astype(np.float64)
        return float(np.sqrt(np.mean(values * values))) < self._quiet_rms

    def _release(self, *, long_silence: bool = False) -> Any:
        np = self._np
        silence = np.concatenate(self._held)
        self._held = []
        self._held_samples = 0
        if not self._started:
            # Air before the first word. Keep only its last ``LEAD_SECS``.
            keep = int(round(LEAD_SECS * self._rate))
            return silence[len(silence) - keep :] if keep < len(silence) else silence
        if long_silence:
            return silence
        return self._shape(silence)

    def _shape(self, silence: Any) -> Any:
        """The silence that replaces ``silence``, which sits between two stretches of speech."""
        np = self._np
        rng = self._rng
        size = len(silence)
        secs = size / self._rate
        self.stats["gaps"] += 1
        if secs < self._gap_min:
            return silence
        if secs < self._gap_max:
            draw = rng.random()
            if draw < self.promote_gaps:
                target = int(rng.uniform(*self._promoted) * self._rate)
                self.stats["promoted"] += 1
            elif draw < self.promote_gaps + self.join_gaps:
                target = int(rng.uniform(*self._joined) * self._rate)
                self.stats["joined"] += 1
            else:
                target = size
                self.stats["kept"] += 1
        else:
            target = size
            if self.stretch_pauses > 0 and rng.random() < self.stretch_pauses:
                target = int(size * rng.uniform(*self._stretch))
                self.stats["stretched"] += 1
        if target == size:
            return silence
        if target > size:
            # Repeat the middle of this same silence: it is the recording's own
            # room tone. Zeros would be heard as the line dropping out.
            middle = silence[size // 4 : 3 * size // 4] if size >= 8 else silence
            half = size // 2
            return np.concatenate([silence[:half], np.resize(middle, target - size), silence[half:]])
        return self._cut_middle(silence, target)

    def _cut_middle(self, silence: Any, target: int) -> Any:
        """Drop the middle and crossfade. The head is the tail of the word before
        and the end is the attack of the word after, so both are kept."""
        np = self._np
        size = len(silence)
        target = max(target, 2 * self._fade + 1)
        if target >= size:
            return silence
        head_len = target // 2
        head = silence[:head_len].astype(np.float32)
        tail = silence[size - (target - head_len) :].astype(np.float32)
        fade = min(self._fade, len(head), len(tail))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            head[-fade:] = head[-fade:] * np.cos(ramp * np.pi / 2) + tail[:fade] * np.sin(ramp * np.pi / 2)
            tail = tail[fade:]
        return np.concatenate([head, tail]).astype("<i2")


async def shape_pauses(
    pcm_chunks: AsyncIterable[bytes],
    *,
    sample_rate: int,
    join_gaps: float = 0.0,
    stretch_pauses: float = 0.0,
    promote_gaps: float = 0.0,
    gap_min_secs: float = 0.12,
    gap_max_secs: float = 0.20,
    quiet_dbfs: float = -42.0,
    seed: int | None = None,
) -> AsyncIterator[bytes]:
    """Yield ``pcm_chunks`` with word gaps joined and real pauses kept or lengthened.

    ``pcm_chunks`` are PCM16 little-endian mono at ``sample_rate``. With the
    three shares at 0 (the default) every chunk is yielded unchanged.
    ``join_gaps`` and ``promote_gaps`` are shares of the silences between
    ``gap_min_secs`` and ``gap_max_secs``; ``stretch_pauses`` is a share of the
    longer ones. ``seed`` fixes the draws. Speech is not delayed: only silence
    is held, until the speech after it arrives.
    """
    shaper = PauseShaper(
        sample_rate,
        join_gaps=join_gaps,
        stretch_pauses=stretch_pauses,
        promote_gaps=promote_gaps,
        gap_min_secs=gap_min_secs,
        gap_max_secs=gap_max_secs,
        quiet_dbfs=quiet_dbfs,
        seed=seed,
    )
    async for chunk in pcm_chunks:
        out = shaper.push(chunk)
        if out:
            yield out
    tail = shaper.finish()
    if tail:
        yield tail
