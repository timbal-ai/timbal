"""Buffered trace persistence: mid-run snapshots are stored in the background, one per run at a time.

A run saves its whole trace each time a span finishes. With ``durability="buffered"`` a save while the run
is still going is scheduled instead of awaited: the run moves on, the provider stores the run's latest
state once the flush interval since its previous save has passed, and saves requested meanwhile fold into
that one. Barriers (:func:`write_through`) store synchronously: the end of the root run (including a run
suspended for approval), a foreground task handed to the background, the end of that background
continuation, and shutdown (:func:`flush_all`). A run has one write at a time, background or barrier, so a
write that starts later stores a newer snapshot; a barrier waits for an upload in flight, never for a retry
backoff.
Background work that outlives the root run saves the same way: scheduled while it runs, and each piece's
end is a barrier.

The live trace is the buffer, so memory does not grow with pending changes; what grows is the lag, the age
of the oldest change not yet stored. A run whose uploads fail or fall behind by more than ``max_lag_s``
goes back to synchronous saves until one succeeds: the state may be the only copy, so it is retried, never
dropped. Before returning, a run with pending changes stores them at its end barrier; a crash in between
loses at most the lag.

``durability="end"`` skips mid-run saves altogether: only the barriers store. It suits runs nobody reads
while they run and that are resumed from their end, such as a subagent or a builder whose progress its caller
already streams: their traces carry the whole message history, so each mid-run save resends megabytes. A crash
loses everything since the last barrier.

Configure per provider (``Provider.configured(durability="buffered", flush_interval_s=0.5)``) or through
``TIMBAL_TRACE_DURABILITY``, ``TIMBAL_TRACE_FLUSH_INTERVAL_MS`` and ``TIMBAL_TRACE_MAX_LAG_S``.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ..context import RunContext
    from .providers.base import TracingProvider

logger = structlog.get_logger("timbal.state.tracing.uploads")

DURABILITY_ENV = "TIMBAL_TRACE_DURABILITY"
FLUSH_INTERVAL_ENV = "TIMBAL_TRACE_FLUSH_INTERVAL_MS"
MAX_LAG_ENV = "TIMBAL_TRACE_MAX_LAG_S"
METRICS_ENV = "TIMBAL_TRACE_METRICS"
DEFAULT_FLUSH_INTERVAL_S = 0.5
DEFAULT_MAX_LAG_S = 30.0
MAX_BACKOFF_S = 30.0


def _env_float(name: str, scale: float = 1.0) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw) * scale
    except ValueError:
        logger.warning("trace_upload_setting_invalid", setting=name, value=raw)
        return None


def durability(provider: type[TracingProvider]) -> str:
    """``sync`` (the default, and any value it does not know), ``buffered`` or ``end``."""
    mode = (getattr(provider, "durability", None) or os.environ.get(DURABILITY_ENV) or "sync").strip().lower()
    return mode if mode in ("buffered", "end") else "sync"


def buffered(provider: type[TracingProvider]) -> bool:
    return durability(provider) == "buffered"


def metrics_on(provider: type[TracingProvider]) -> bool:
    """Counters go into usage only when asked: they are new usage keys for every run."""
    return durability(provider) != "sync" or os.environ.get(METRICS_ENV, "").strip().lower() in ("1", "true", "yes")


def _flush_interval(provider: type[TracingProvider]) -> float:
    value = getattr(provider, "flush_interval_s", None)
    if value is None:
        value = _env_float(FLUSH_INTERVAL_ENV, 0.001)
    return max(0.0, DEFAULT_FLUSH_INTERVAL_S if value is None else value)


def _max_lag(provider: type[TracingProvider]) -> float:
    value = getattr(provider, "max_lag_s", None)
    if value is None:
        value = _env_float(MAX_LAG_ENV)
    return DEFAULT_MAX_LAG_S if value is None else value


def count(run_context: RunContext, key: str, value: int = 1) -> None:
    if value and metrics_on(run_context._tracing_provider):
        run_context.update_usage(key, int(value))


class _RunUploads:
    __slots__ = (
        "context", "provider", "task", "pending", "sending", "started_at", "unsent_since", "failures",
        "lock", "requests", "writers",
    )  # fmt: skip

    def __init__(self, context: RunContext, provider: type[TracingProvider]) -> None:
        self.context = context
        self.provider = provider
        self.task: asyncio.Task | None = None
        # A change no save has picked up yet, and when the oldest such change happened.
        self.pending = False
        self.unsent_since: float | None = None
        self.sending = False
        self.started_at = float("-inf")
        self.failures = 0
        # One write per run at a time, background upload or barrier: a write that starts later stores a
        # newer snapshot, so an older one never lands over it.
        self.lock = asyncio.Lock()
        # Saves requested so far; a write that starts after the n-th one covers it.
        self.requests = 0
        # Barrier writes holding or waiting for the lock.
        self.writers = 0


_RUNS: dict[tuple[int, str], _RunUploads] = {}


def _key(run_context: RunContext) -> tuple[int, str]:
    return id(run_context._tracing_provider), run_context.id


def schedule(run_context: RunContext) -> bool:
    """Defer this mid-run save. False: the run is too far behind, and the caller saves synchronously."""
    provider = run_context._tracing_provider
    loop = asyncio.get_running_loop()
    run = _RUNS.get(_key(run_context))
    if run is None:
        run = _RUNS[_key(run_context)] = _RunUploads(run_context, provider)
    now = loop.time()
    if run.unsent_since is not None and now - run.unsent_since > _max_lag(provider):
        count(run_context, "tracing:lag_exceeded")
        return False
    run.requests += 1
    if run.pending:
        count(run_context, "tracing:saves_coalesced")
    else:
        run.pending = True
        if run.unsent_since is None:
            run.unsent_since = now
    if run.task is None or run.task.done():
        run.task = loop.create_task(_drain(run))
    return True


def tracked(run_context: RunContext) -> bool:
    """Whether buffered saves of this run have state here (a save pending, in flight, or spacing out)."""
    return _key(run_context) in _RUNS


async def write_through(run_context: RunContext, put: Callable[[Any], Awaitable[None]]) -> None:
    """Store the run's latest state now, as its next write (a barrier).

    It waits for an upload in flight or another barrier's write (the root's end and a background task's end
    can overlap), never for a drain waiting out its interval or a retry backoff; saves requested while it
    writes are uploaded after it; what it covers is not uploaded again. Raises what ``put`` raises.
    """
    key = _key(run_context)
    run = _RUNS.get(key)
    if run is None:
        if durability(run_context._tracing_provider) == "sync":
            await put(run_context)
            return
        run = _RUNS[key] = _RunUploads(run_context, run_context._tracing_provider)
    run.writers += 1
    try:
        async with run.lock:
            covered = run.requests
            await put(run_context)
            if run.requests == covered:
                run.pending, run.unsent_since, run.failures = False, None, 0
    finally:
        run.writers -= 1
        _release(run)


def _release(run: _RunUploads) -> None:
    """Drop a run with nothing left to store, and its drain.

    Called after a barrier's write, when the drain cannot be uploading (the barrier held the lock after it):
    it is waiting out an interval or backoff, or queued for the lock, so cancelling it interrupts no write.
    """
    key = _key(run.context)
    if run.pending or run.writers or run.sending or _RUNS.get(key) is not run:
        return
    if run.task is not None and not run.task.done():
        run.task.cancel()
    del _RUNS[key]


async def flush_all(timeout: float | None = 10.0) -> None:
    """Store every buffered run's latest state now (process shutdown)."""
    runs = list(_RUNS.values())

    async def flush(run: _RunUploads) -> None:
        try:
            await write_through(run.context, run.provider.put)
        except Exception as e:
            logger.error("trace_flush_failed", run_id=run.context.id, error=str(e))

    if runs:
        try:
            await asyncio.wait_for(asyncio.gather(*map(flush, runs)), timeout)
        except TimeoutError:
            logger.error("trace_flush_timed_out", runs=[run.context.id for run in runs], timeout_s=timeout)


async def _drain(run: _RunUploads) -> None:
    loop = asyncio.get_running_loop()
    key = _key(run.context)
    interval = _flush_interval(run.provider)
    while run.pending:
        backoff = min(MAX_BACKOFF_S, interval * (2**run.failures)) if run.failures else interval
        wait = run.started_at + backoff - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        if _RUNS.get(key) is not run:
            return
        async with run.lock:
            # A barrier may have stored it meanwhile.
            if not run.pending:
                break
            # One is waiting: it stores the latest state, then this re-checks what is still pending.
            if run.writers:
                continue
            covered_since = run.unsent_since
            run.pending, run.unsent_since = False, None
            run.sending, run.started_at = True, loop.time()
            t0 = time.monotonic()
            try:
                await run.provider.put(run.context)
            except Exception as e:
                # The state may be the only copy: retry it, with backoff, until a save or a barrier stores it.
                run.failures += 1
                run.pending = True
                if covered_since is not None:
                    run.unsent_since = (
                        covered_since if run.unsent_since is None else min(run.unsent_since, covered_since)
                    )
                logger.error(
                    "Background trace save failed. Retrying...",
                    run_id=run.context.id,
                    failures=run.failures,
                    error=str(e),
                )
                count(run.context, "tracing:save_failed")
            else:
                run.failures = 0
                count(run.context, "tracing:saves")
            finally:
                run.sending = False
                count(run.context, "tracing:save_ms", round((time.monotonic() - t0) * 1000))


def pending_runs() -> list[str]:
    """Run ids with buffered state not yet stored (diagnostics and tests)."""
    return [run.context.id for run in _RUNS.values() if run.pending or run.sending]


def _reset_for_tests() -> dict[Any, Any]:
    previous = dict(_RUNS)
    _RUNS.clear()
    return previous
