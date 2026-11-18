"""Streaming cap on internal TTS silence.

A sample is silence only when the last ``ANALYSIS_WINDOW_SECS`` of audio stays
under ``quiet_dbfs``. The window is long enough to cover a zero-crossing of a
voiced waveform, so the gate does not clip the body of a word the way a
per-sample threshold would. Quiet samples that still share that window with
speech are kept, and they count toward ``max_pause_secs``, so a normal cap
lands on exactly that much silence. This is the one-pass shape of Audacity's
truncate-silence: the first kept bytes are yielded before the next input
chunk is read, so the start of a reply is not held back to measure a pause
that has not arrived yet.

``numpy`` is imported on first use. It ships with ``timbal[voice]``
(onnxruntime / transformers); :mod:`timbal.voice` stays importable without it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator

# 10 ms covers a half-period down to ~50 Hz, below voiced speech, so a sine
# wave's zero-crossings stay inside a window that still contains its peak.
ANALYSIS_WINDOW_SECS = 0.010


def _numpy():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "TTS pause trimming requires the timbal[voice] extra (numpy): uv pip install 'timbal[voice]'"
        ) from e
    return np


class _Limiter:
    """Causal peak-window silence gate over PCM16 little-endian mono."""

    def __init__(self, sample_rate: int, max_pause_secs: float, quiet_dbfs: float) -> None:
        self._np = _numpy()
        self._window = max(1, int(round(sample_rate * ANALYSIS_WINDOW_SECS)))
        self._max_quiet = int(round(sample_rate * max_pause_secs))
        # dBFS relative to a full-scale int16 (32768), matching 20*log10(|s|/32768).
        self._threshold = 32768.0 * (10.0 ** (quiet_dbfs / 20.0))
        self._carry = b""
        self._i = 0
        # Decreasing (index, magnitude) deque: the front is the window peak.
        self._dq: deque[tuple[int, int]] = deque()
        self._emitted_quiet = 0

    def push(self, chunk: bytes) -> bytes:
        if not chunk and not self._carry:
            return b""
        data = self._carry + chunk
        usable = len(data) - (len(data) % 2)
        self._carry = data[usable:]
        if usable <= 0:
            return b""
        view = data[:usable]
        samples = self._np.frombuffer(view, dtype="<i2")
        # abs(-32768) overflows int16. Promote before the comparison.
        mags = self._np.abs(samples.astype(self._np.int32, copy=False))
        threshold = self._threshold
        if mags.size and int(mags.min()) >= threshold:
            self._emitted_quiet = 0
            self._note(mags)
            return view
        peak = self._dq[0][1] if self._dq else 0
        if mags.size and int(mags.max()) < threshold and peak < threshold:
            room = self._max_quiet - self._emitted_quiet
            self._note(mags)
            if room <= 0:
                return b""
            if room >= int(mags.size):
                self._emitted_quiet += int(mags.size)
                return view
            self._emitted_quiet = self._max_quiet
            return samples[:room].tobytes()
        return self._scan(samples, mags)

    def finish(self) -> bytes:
        """Drop a trailing odd byte. Nothing is held back waiting for more audio."""
        self._carry = b""
        return b""

    def _note(self, mags) -> None:
        """Advance the peak window. Only the last ``window`` magnitudes can remain."""
        window = self._window
        if len(mags) > window:
            self._i += len(mags) - window
            self._dq.clear()
            mags = mags[-window:]
        dq = self._dq
        i = self._i
        for mag in mags.tolist():
            mag = int(mag)
            while dq and dq[-1][1] <= mag:
                dq.pop()
            dq.append((i, mag))
            expire = i - window
            while dq and dq[0][0] <= expire:
                dq.popleft()
            i += 1
        self._i = i

    def _push_mag(self, mag: int) -> int:
        i = self._i
        dq = self._dq
        while dq and dq[-1][1] <= mag:
            dq.pop()
        dq.append((i, mag))
        expire = i - self._window
        while dq and dq[0][0] <= expire:
            dq.popleft()
        self._i = i + 1
        return dq[0][1]

    def _scan(self, samples, mags) -> bytes:
        kept: list[int] = []
        emitted = self._emitted_quiet
        threshold = self._threshold
        max_quiet = self._max_quiet
        for sample, mag in zip(samples.tolist(), mags.tolist(), strict=True):
            mag = int(mag)
            peak = self._push_mag(mag)
            if mag >= threshold:
                emitted = 0
                kept.append(int(sample))
                continue
            # ``peak >= threshold`` is the transient guard: this sample is quiet
            # on its own but the window still contains speech, so dropping it
            # would cut the word. It still spends the pause budget.
            if peak >= threshold or emitted < max_quiet:
                kept.append(int(sample))
                emitted += 1
        self._emitted_quiet = emitted
        if not kept:
            return b""
        return self._np.asarray(kept, dtype="<i2").tobytes()


async def limit_pauses(
    pcm_chunks: AsyncIterable[bytes],
    *,
    sample_rate: int,
    max_pause_secs: float,
    quiet_dbfs: float = -45.0,
) -> AsyncIterator[bytes]:
    """Yield ``pcm_chunks`` with internal silence shortened to ``max_pause_secs``.

    ``pcm_chunks`` are PCM16 little-endian mono at ``sample_rate``. A run of
    audio quieter than ``quiet_dbfs`` (default -45, which leaves breaths) that
    lasts longer than ``max_pause_secs`` is cut down to that length. The bytes
    that are kept are unchanged. The first chunk's kept audio is yielded
    before the following chunk is pulled.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if (
        isinstance(max_pause_secs, bool)
        or not isinstance(max_pause_secs, (int, float))
        or not math.isfinite(max_pause_secs)
        or max_pause_secs < 0
    ):
        raise ValueError("max_pause_secs must be finite and non-negative")
    if isinstance(quiet_dbfs, bool) or not isinstance(quiet_dbfs, (int, float)) or not math.isfinite(quiet_dbfs):
        raise ValueError("quiet_dbfs must be finite")
    limiter = _Limiter(sample_rate, float(max_pause_secs), float(quiet_dbfs))
    async for chunk in pcm_chunks:
        out = limiter.push(chunk)
        if out:
            yield out
    tail = limiter.finish()
    if tail:
        yield tail
