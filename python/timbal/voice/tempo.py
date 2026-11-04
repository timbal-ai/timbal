"""Pitch-preserving TTS tempo via FFmpeg's ``atempo`` filter.

The filter graph follows PyAV's ``examples/audio/atempo.py`` (BSD-licensed
PyAV). Tempos outside the range where a single ``atempo`` stage is
well-behaved (0.5–2.0) are split into chained stages, as on the FFmpeg wiki
("Changing audio speed"). Input is cut into 20 ms blocks — the same framing
LiveKit's ``AudioByteStream`` uses to keep PCM latency down — and at most
50 ms is held before a block is pushed. ``atempo`` keeps a short tail inside
the filter; a stall of ``IDLE_FLUSH_SECS`` (0.25 s) flushes it, the same idea
as Pipecat's TTS stop timeout, so a phrase spoken before a tool call finishes
instead of sitting in the filter until the next sentence.

``av`` is imported on first use. It ships with ``timbal[voice]`` (aiortc);
:mod:`timbal.voice` stays importable without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import math
from collections.abc import AsyncIterable, AsyncIterator

FRAME_MS = 20
MAX_BUFFER_MS = 50
IDLE_FLUSH_SECS = 0.25


def frame_bytes(sample_rate: int, duration_ms: int) -> int:
    """PCM16 mono byte count for ``duration_ms`` at ``sample_rate`` (at least one sample)."""
    samples = max(1, sample_rate * duration_ms // 1000)
    return samples * 2


def atempo_stages(tempo: float) -> list[float]:
    """Split ``tempo`` into ``atempo`` factors, each in ``[0.5, 2.0]``.

    The product of the stages equals ``tempo``.
    """
    if isinstance(tempo, bool) or not isinstance(tempo, (int, float)) or not math.isfinite(tempo) or tempo <= 0:
        raise ValueError(f"tempo must be finite and positive, got {tempo!r}")
    stages: list[float] = []
    remaining = float(tempo)
    while remaining > 2.0 + 1e-12:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-12:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return stages


def drain_frames(
    buf: bytearray,
    *,
    frame_bytes: int,
    max_buffer_bytes: int,
    force: bool,
) -> list[bytes]:
    """Pull whole frames out of ``buf``, leaving a short tail unless ``force``.

    Frames are ``frame_bytes`` long (20 ms at the caller's rate). A tail
    shorter than that stays buffered, and the buffer is never left at or
    above ``max_buffer_bytes`` (50 ms): the overflow is emitted as its own
    frame. ``force`` also emits an even-length tail, which is the idle and
    end-of-stream flush. A single leftover byte stays in ``buf``; it is not
    a whole sample.
    """
    if frame_bytes < 2 or frame_bytes % 2 or max_buffer_bytes < 2:
        raise ValueError("frame sizes must be positive even byte counts")
    frames: list[bytes] = []
    while len(buf) >= frame_bytes:
        frames.append(bytes(buf[:frame_bytes]))
        del buf[:frame_bytes]
    if len(buf) >= max_buffer_bytes or (force and len(buf) >= 2):
        take = len(buf) - (len(buf) % 2)
        if take:
            frames.append(bytes(buf[:take]))
            del buf[:take]
    return frames


def _av():
    try:
        import av
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "TTS tempo change requires the timbal[voice] extra (av): uv pip install 'timbal[voice]'"
        ) from e
    return av


class _Atempo:
    """One ``atempo`` graph. ``flush`` ends it; the next ``push`` opens a new one."""

    def __init__(self, sample_rate: int, tempo: float) -> None:
        self._sample_rate = sample_rate
        self._stages = atempo_stages(tempo)
        self._av = None
        self._graph = None
        self._pts = 0
        self._dirty = False

    def push(self, pcm: bytes) -> list[bytes]:
        if len(pcm) < 2:
            return []
        self._ensure()
        av = self._av
        samples = len(pcm) // 2
        frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
        frame.planes[0].update(pcm[: samples * 2])
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._pts += samples
        self._graph.push(frame)
        self._dirty = True
        return self._pull()

    def flush(self) -> list[bytes]:
        """Drain the filter tail and drop the graph. A clean graph yields nothing."""
        if self._graph is None or not self._dirty:
            self._close()
            return []
        self._graph.push(None)
        try:
            return self._pull()
        finally:
            self._close()

    def close(self) -> None:
        self._close()

    def _ensure(self) -> None:
        if self._graph is not None:
            return
        av = _av()
        self._av = av
        graph = av.filter.Graph()
        rate = self._sample_rate
        nodes = [
            graph.add_abuffer(
                format="s16",
                layout="mono",
                sample_rate=rate,
                time_base=fractions.Fraction(1, rate),
            )
        ]
        for stage in self._stages:
            nodes.append(graph.add("atempo", f"{stage:.8g}"))
        # atempo's native format is float. Pin the output back to PCM16 mono
        # at the session rate so the rest of the pipeline can count bytes.
        nodes.append(graph.add("aformat", f"sample_fmts=s16:sample_rates={rate}:channel_layouts=mono"))
        nodes.append(graph.add("abuffersink"))
        graph.link_nodes(*nodes).configure()
        self._graph = graph
        self._pts = 0
        self._dirty = False

    def _pull(self) -> list[bytes]:
        av = self._av
        graph = self._graph
        out: list[bytes] = []
        while True:
            try:
                frame = graph.pull()
            except (av.EOFError, av.BlockingIOError):
                break
            if frame is None or frame.samples <= 0:
                continue
            out.append(bytes(frame.planes[0])[: frame.samples * 2])
        return out

    def _close(self) -> None:
        self._graph = None
        self._dirty = False
        self._pts = 0


def _blocks(engine: _Atempo, buf: bytearray, frame_n: int, cap_n: int, *, force: bool) -> list[bytes]:
    out: list[bytes] = []
    for frame in drain_frames(buf, frame_bytes=frame_n, max_buffer_bytes=cap_n, force=force):
        out.extend(engine.push(frame))
    if force:
        out.extend(engine.flush())
    return [chunk for chunk in out if chunk]


async def change_tempo(
    pcm_chunks: AsyncIterable[bytes],
    *,
    sample_rate: int,
    tempo: float,
) -> AsyncIterator[bytes]:
    """Yield ``pcm_chunks`` with pitch-preserving tempo ``tempo`` (1.0 = unchanged).

    ``pcm_chunks`` are PCM16 little-endian mono at ``sample_rate``. ``tempo``
    above 1 plays faster. A tempo of 1 is a byte-for-byte pass-through (no
    filter, no extra buffering). Otherwise blocks of 20 ms are pushed into
    ``atempo``, at most 50 ms is buffered, and 0.25 s without a new chunk
    flushes the filter so the phrase ends.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(tempo, bool) or not isinstance(tempo, (int, float)) or not math.isfinite(tempo) or tempo <= 0:
        raise ValueError("tempo must be finite and positive")
    tempo = float(tempo)
    if math.isclose(tempo, 1.0, abs_tol=1e-6):
        async for chunk in pcm_chunks:
            if chunk:
                yield chunk
        return

    agen = aiter(pcm_chunks)
    engine = _Atempo(sample_rate, tempo)
    buf = bytearray()
    frame_n = frame_bytes(sample_rate, FRAME_MS)
    cap_n = frame_bytes(sample_rate, MAX_BUFFER_MS)
    pending: asyncio.Future | None = None
    seen = False
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(agen.__anext__())
            try:
                if not seen:
                    chunk = await pending
                else:
                    chunk = await asyncio.wait_for(asyncio.shield(pending), IDLE_FLUSH_SECS)
            except TimeoutError:
                for out in _blocks(engine, buf, frame_n, cap_n, force=True):
                    yield out
                continue
            except StopAsyncIteration:
                pending = None
                for out in _blocks(engine, buf, frame_n, cap_n, force=True):
                    yield out
                return
            pending = None
            seen = True
            if chunk:
                buf.extend(chunk)
            for out in _blocks(engine, buf, frame_n, cap_n, force=False):
                yield out
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        close = getattr(agen, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()
        engine.close()
