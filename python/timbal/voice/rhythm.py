"""Streaming pause rhythm for synthesized speech.

Neural TTS spaces many words with a 0.12-0.20 s gap and rarely makes a pause
of 0.3-0.5 s. A person does the opposite: words run together (under 90 ms
apart) and now and then there is a real pause. That gap between every word is
what reads as "word, word, word" even when pitch and rate are right.

This filter sorts each internal silence by its length and reshapes it:

* shorter than ``gap_min_secs``: left alone. It is the closure of a stop
  consonant ("p", "t", "k"), and shortening it eats the consonant.
* between ``gap_min_secs`` and ``gap_max_secs``: a gap between two words. With
  probability ``join_gaps`` it is closed to 23-73 ms. With probability
  ``promote_gaps`` it becomes a pause of 0.30-0.55 s instead (the speaker who
  stops to think mid-sentence). The rest are left alone.
* ``gap_max_secs`` or longer: a comma or sentence pause. It is kept and, with
  probability ``stretch_pauses``, lengthened by 1.4-2.3x, up to
  ``max_pause_secs``.

A silence is a run of 10 ms frames whose RMS level is under ``quiet_dbfs``.
That run still holds soft sound: the tail of the word before it, a final "s",
a breath. So only its true silence is ever cut or repeated: the longest
stretch of frames with no sample over ``quiet_dbfs``. A join removes the
middle of that stretch with a crossfade and always leaves 10 ms of it. A
lengthened pause gets more of that stretch's own room tone, laid down in
overlapping pieces taken at random so that it does not repeat. A silence with
no true silence to work on is left as it is.

Speech passes through unchanged and is never delayed. The silence before the
first word and after the last one pass through unchanged too. Only a silence
between two stretches of speech is held, until the speech after it shows how
long it was. It is not held past the moment the caller would hear it: when
the audio already yielded has finished playing and the provider has sent
nothing more, the held silence is released as it is, and what follows it is
not shaped (the caller has already waited). A silence of ``MAX_HOLD_SECS`` is
released as it is as well.

``numpy`` is imported on first use. It ships with ``timbal[voice]``
(onnxruntime / transformers); :mod:`timbal.voice` stays importable without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import structlog

logger = structlog.get_logger("timbal.voice.rhythm")

FRAME_SECS = 0.010
"""Analysis frame. Silence is judged on the level of each frame."""
CROSSFADE_SECS = 0.012
"""Fade across a cut, and between the pieces of room tone that lengthen a pause."""
MAX_HOLD_SECS = 0.9
"""A silence this long is released without waiting to see how it ends."""
JOINED_SECS = (0.023, 0.073)
"""What a joined word gap is shortened to."""
PROMOTED_SECS = (0.30, 0.55)
"""What a word gap promoted to a pause is lengthened to."""
STRETCH_FACTOR = (1.4, 2.3)
"""How much a lengthened pause grows."""
LENGTHENED_MAX_SECS = 0.55
"""A pause that is lengthened never ends up longer than this."""
MIN_SILENCE_SECS = 0.010
"""True silence that a joined gap always keeps."""
MIN_TONE_SECS = 0.030
"""Shortest stretch of true silence that is used to lengthen a pause."""
QUIET_DBFS_RANGE = (-90.0, -20.0)
"""Allowed ``quiet_dbfs``. Above -20 dBFS speech itself would count as silence."""
STALL_GRACE_SECS = 0.02
"""How long past the end of playback a held silence still waits for the next chunk."""


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


def _number(name: str, value: Any, *, low: float, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < low or value > high:
        raise ValueError(f"{name} must be at least {low}" + ("" if high == math.inf else f" and at most {high}"))
    return float(value)


class PauseShaper:
    """One shaper per stream of synthesized audio, over PCM16 little-endian mono.

    ``push`` returns what can be played now and ``finish`` the rest.
    ``release`` gives up the silence being held, for a caller that knows the
    listener has caught up with the audio. ``stats`` counts what happened to
    each internal silence.
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
        max_pause_secs: float = LENGTHENED_MAX_SECS,
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
        if gap_max > MAX_HOLD_SECS:
            # A silence that long is released before it can be told from a pause.
            raise ValueError(f"gap_max_secs must be at most {MAX_HOLD_SECS}")
        quiet = _number("quiet_dbfs", quiet_dbfs, low=QUIET_DBFS_RANGE[0], high=QUIET_DBFS_RANGE[1])
        max_pause = _number("max_pause_secs", max_pause_secs, low=0.0)
        self._joined = _span("joined_secs", joined_secs, floor=0.0)
        self._promoted = _span("promoted_secs", promoted_secs, floor=0.0)
        self._stretch = _span("stretch_factor", stretch_factor, floor=1.0)
        self._np = _numpy()
        self._rate = sample_rate
        self._frame = max(1, int(round(sample_rate * FRAME_SECS)))
        self._fade = max(1, int(round(sample_rate * CROSSFADE_SECS)))
        self._min_silence = max(1, int(round(sample_rate * MIN_SILENCE_SECS)))
        self._min_tone = max(1, int(round(sample_rate * MIN_TONE_SECS)))
        self._max_pause = int(max_pause * sample_rate)
        self._gap_min = gap_min
        self._gap_max = gap_max
        # A full-scale int16 is 32768, matching 20*log10(level/32768). The same
        # number is the RMS under which a frame belongs to a silence and the
        # peak under which it is true silence.
        self._quiet_level = 32768.0 * (10.0 ** (quiet / 20.0))
        # Which silences are picked must not depend on how much room tone the
        # ones before needed, so the tone has its own generator.
        self._rng = random.Random(seed)
        self._tone_rng = random.Random(None if seed is None else f"tone:{seed}")
        self._odd = b""
        self._carry = self._np.zeros(0, dtype="<i2")
        self._held: list[Any] = []
        self._held_true: list[bool] = []
        self._held_samples = 0
        # Silence passes straight through until the first speech, and again
        # after a release, until the next speech.
        self._passing = True
        self.stats = {
            "gaps": 0,
            "joined": 0,
            "kept": 0,
            "promoted": 0,
            "stretched": 0,
            "skipped": 0,
            "released": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.join_gaps > 0 or self.stretch_pauses > 0 or self.promote_gaps > 0

    @property
    def holding(self) -> bool:
        """Whether a silence is being held until the speech after it arrives."""
        return bool(self._held)

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
        if not frames:
            return b""
        blocks = samples[: frames * frame].reshape(frames, frame)
        levels = blocks.astype(np.float64)
        quiet = np.sqrt((levels * levels).mean(axis=1)) < self._quiet_level
        # abs(-32768) overflows int16. Promote before taking the magnitude.
        true_silence = np.abs(blocks.astype(np.int32)).max(axis=1) < self._quiet_level
        out: list[Any] = []
        for index in range(frames):
            block = blocks[index]
            if quiet[index]:
                if self._passing:
                    out.append(block)
                    continue
                self._held.append(block)
                self._held_true.append(bool(true_silence[index]))
                self._held_samples += frame
                if self._held_samples >= MAX_HOLD_SECS * self._rate:
                    # Too long to be a pause inside a sentence (a tool wait
                    # that the provider filled with silence). Stop holding.
                    out.append(self._take_held()[0])
                    self._passing = True
                continue
            if self._held:
                out.append(self._shape(*self._take_held()))
            self._passing = False
            out.append(block)
        if not out:
            return b""
        return np.concatenate(out).astype("<i2", copy=False).tobytes()

    def release(self) -> bytes:
        """The silence being held, unchanged. Silence then passes through until the next speech.

        For when the listener has already heard everything yielded so far:
        from then on the silence is being waited through, not measured.
        """
        if not self._held:
            return b""
        self.stats["released"] += 1
        self._passing = True
        return self._take_held()[0].astype("<i2", copy=False).tobytes()

    def finish(self) -> bytes:
        """End of the stream: held silence and the last partial frame leave unchanged."""
        if not self.enabled:
            return b""
        np = self._np
        parts = [*self._held]
        if len(self._carry):
            parts.append(self._carry)
        self._held = []
        self._held_true = []
        self._held_samples = 0
        self._carry = np.zeros(0, dtype="<i2")
        self._odd = b""
        if not parts:
            return b""
        return np.concatenate(parts).astype("<i2", copy=False).tobytes()

    def _take_held(self) -> tuple[Any, list[bool]]:
        """The held silence and, for each of its frames, whether it is true silence."""
        silence, true_frames = self._np.concatenate(self._held), self._held_true
        self._held = []
        self._held_true = []
        self._held_samples = 0
        return silence, true_frames

    def _true_silence(self, true_frames: list[bool]) -> tuple[int, int]:
        """Sample bounds, inside a held silence, of its longest run of frames with no sample over the level."""
        best = (0, 0)
        start = None
        for index, flag in enumerate([*true_frames, False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                if index - start > best[1] - best[0]:
                    best = (start, index)
                start = None
        return best[0] * self._frame, best[1] * self._frame

    def _shape(self, silence: Any, true_frames: list[bool]) -> Any:
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
                kind = "promoted"
                target = min(int(rng.uniform(*self._promoted) * self._rate), self._max_pause)
            elif draw < self.promote_gaps + self.join_gaps:
                kind = "joined"
                target = int(rng.uniform(*self._joined) * self._rate)
            else:
                self.stats["kept"] += 1
                return silence
        else:
            if not (self.stretch_pauses > 0 and rng.random() < self.stretch_pauses):
                return silence
            kind = "stretched"
            target = min(int(size * rng.uniform(*self._stretch)), self._max_pause)
        low, high = self._true_silence(true_frames)
        tone = silence[low:high]
        if kind == "joined":
            # Only true silence is removed, and some of it always stays: the
            # soft sound on either side is the end of one word and the start
            # of the next.
            remove = min(size - target, len(tone) - self._min_silence)
            if remove <= 0:
                self.stats["skipped"] += 1
                return silence
            shaped = self._cut_middle(tone, len(tone) - remove)
        else:
            extra = target - size
            if extra <= 0 or len(tone) < self._min_tone:
                # Already as long as a pause may get, or nothing but soft
                # sound in it: repeating a breath would be heard.
                self.stats["skipped"] += 1
                return silence
            shaped = self._lengthen(tone, extra)
        self.stats[kind] += 1
        return np.concatenate([silence[:low], shaped, silence[high:]])

    def _cut_middle(self, silence: Any, target: int) -> Any:
        """``silence`` cut to ``target`` samples: the middle is dropped and the two ends are crossfaded."""
        size = len(silence)
        target = max(target, 1)
        if target >= size:
            return silence
        # The crossfade plays ``fade`` samples of both ends at once, so the two
        # ends together are ``fade`` samples longer than what comes out.
        fade = min(self._fade, target, size - target)
        if fade < 2:
            fade = 0
        head_len = (target + fade) // 2
        return self._overlap_add([silence[:head_len], silence[size - (target + fade - head_len) :]], fade)

    def _lengthen(self, tone: Any, extra: int) -> Any:
        """``tone`` with ``extra`` more samples of itself in the middle.

        The added audio is pieces of ``tone`` taken at random places, in
        random lengths and directions, each crossfaded into the next. One
        slice repeated end to end is periodic, and room tone that repeats is
        heard as a flutter.
        """
        rng = self._tone_rng
        size = len(tone)
        half = size // 2
        fade = min(self._fade, half, size - half)
        if fade < 2:
            fade = 0
        # A piece gives up ``fade`` samples to the overlap with the one before
        # it, and so does the closing half of ``tone``.
        remaining = extra + fade
        most = size - fade
        least = max(fade, 1)
        pieces = [tone[:half]]
        while remaining > 0:
            if remaining <= most:
                net = remaining
            else:
                high = min(most, remaining - least)
                net = rng.randint(min(high, max(least, most // 2)), high)
            length = net + fade
            start = rng.randint(0, size - length)
            piece = tone[start : start + length]
            pieces.append(piece[::-1] if rng.random() < 0.5 else piece)
            remaining -= net
        pieces.append(tone[half:])
        return self._overlap_add(pieces, fade)

    def _overlap_add(self, pieces: list[Any], fade: int) -> Any:
        """``pieces`` in a row, each crossfaded over ``fade`` samples into the next."""
        np = self._np
        total = sum(len(piece) for piece in pieces) - fade * (len(pieces) - 1)
        out = np.zeros(total, dtype=np.float32)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32) * (np.pi / 2)
            rise, fall = np.sin(ramp), np.cos(ramp)
        position = 0
        last = len(pieces) - 1
        for index, piece in enumerate(pieces):
            piece = piece.astype(np.float32)
            if fade and index > 0:
                piece[:fade] *= rise
            if fade and index < last:
                piece[-fade:] *= fall
            out[position : position + len(piece)] += piece
            position += len(piece) - fade
        return np.clip(np.rint(out), -32768, 32767).astype("<i2")


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
    max_pause_secs: float = LENGTHENED_MAX_SECS,
    playback_rate: float = 1.0,
    seed: int | None = None,
) -> AsyncIterator[bytes]:
    """Yield ``pcm_chunks`` with word gaps joined and real pauses kept or lengthened.

    ``pcm_chunks`` are PCM16 little-endian mono at ``sample_rate``. With the
    three shares at 0 (the default) every chunk is yielded unchanged.
    ``join_gaps`` and ``promote_gaps`` are shares of the silences between
    ``gap_min_secs`` and ``gap_max_secs``; ``stretch_pauses`` is a share of the
    longer ones. No pause is lengthened past ``max_pause_secs``. ``seed``
    fixes the draws.

    Speech is yielded as it arrives. A silence is held until the speech after
    it arrives, but not past the moment the listener would hear it: once the
    audio yielded so far has finished playing, the held silence is yielded as
    it is. ``playback_rate`` is how much faster than real time the yielded
    audio is played (the tempo applied after this filter).
    """
    rate = _number("playback_rate", playback_rate, low=0.0)
    if rate <= 0:
        raise ValueError("playback_rate must be greater than 0")
    shaper = PauseShaper(
        sample_rate,
        join_gaps=join_gaps,
        stretch_pauses=stretch_pauses,
        promote_gaps=promote_gaps,
        gap_min_secs=gap_min_secs,
        gap_max_secs=gap_max_secs,
        quiet_dbfs=quiet_dbfs,
        max_pause_secs=max_pause_secs,
        seed=seed,
    )
    if not shaper.enabled:
        async for chunk in pcm_chunks:
            yield chunk
        return

    loop = asyncio.get_running_loop()
    bytes_per_sec = sample_rate * 2 * rate
    # When the listener will have heard everything yielded so far.
    played_at = 0.0
    agen = aiter(pcm_chunks)
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(agen.__anext__())
            if shaper.holding and not pending.done():
                wait = max(played_at - loop.time(), 0.0) + STALL_GRACE_SECS
                # ``asyncio.wait`` leaves ``pending`` running on a timeout and
                # raises nothing, so a timeout of the source's own is not
                # mistaken for this one.
                done, _ = await asyncio.wait({pending}, timeout=wait)
                if not done:
                    # The provider has stalled and the listener is hearing
                    # nothing. The wait is the pause now: do not add to it.
                    out = shaper.release()
                    if out:
                        played_at = max(played_at, loop.time()) + len(out) / bytes_per_sec
                        yield out
                    continue
            try:
                chunk = await pending
            except StopAsyncIteration:
                pending = None
                break
            pending = None
            out = shaper.push(chunk)
            if out:
                played_at = max(played_at, loop.time()) + len(out) / bytes_per_sec
                yield out
        tail = shaper.finish()
        if tail:
            yield tail
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        close = getattr(agen, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()
        stats = shaper.stats
        if stats["gaps"] or stats["released"]:
            # INFO on purpose: it is the only record of what was done to the
            # pauses of a reply that sounded wrong.
            logger.info("tts_rhythm_shaped", **stats)
