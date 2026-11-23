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

``durability="sync"`` awaits every save, and holds the same order: saves of one run that overlap (the end
of the root run and the end of background work, parallel calls that end together) are written one at a
time. A save takes its snapshot when its write starts and commits some time later, so two on their way at
once could land the older one last, for good when it is the run's last save. A save that waited is not
written again when a write that began after it was asked for has landed: that write holds its state (calls
that end together while one write is in flight cost one more write, not one each). A provider that keeps
that order itself opts out (``TracingProvider.overlapping_saves``).

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
import itertools
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


def _explicit_durability(provider: type[TracingProvider]) -> str | None:
    return getattr(provider, "durability", None) or os.environ.get(DURABILITY_ENV) or None


def _mode(raw: str) -> str:
    mode = raw.strip().lower()
    return mode if mode in ("buffered", "end") else "sync"


def durability(provider: type[TracingProvider]) -> str:
    """``sync``, ``buffered`` or ``end`` (any value it does not know reads as ``sync``): the provider's
    ``durability``, else ``TIMBAL_TRACE_DURABILITY``, else the provider's ``default_durability``."""
    return _mode(_explicit_durability(provider) or getattr(provider, "default_durability", None) or "sync")


def buffered(provider: type[TracingProvider]) -> bool:
    return durability(provider) == "buffered"


def metrics_on(provider: type[TracingProvider]) -> bool:
    """Counters go into usage only when asked: they are new usage keys for every run, so a provider's
    default mode does not turn them on."""
    explicit = _explicit_durability(provider)
    if explicit is not None and _mode(explicit) != "sync":
        return True
    return os.environ.get(METRICS_ENV, "").strip().lower() in ("1", "true", "yes")


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
        _seen(context)
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
# Run contexts in the order buffered uploads first saw them (``_holds``).
_SEEN = itertools.count(1)


def _seen(run_context: RunContext) -> int:
    state = run_context.__dict__
    seen = state.get("_uploads_seen")
    if seen is None:
        seen = state["_uploads_seen"] = next(_SEEN)
    return seen


def _holds(run: _RunUploads, run_context: RunContext) -> bool:
    """Whether ``run`` stores ``run_context``. A run id used again by a fresh context (once its run finished: the
    HTTP job store allows it) takes the entry over, and what the older context left pending is no longer the id's
    to store; a save of a context older than the one held is not stored (also one that was waiting for the lock
    when the entry was taken over: ``write_through`` asks again once it holds it)."""
    if run.context is run_context:
        return True
    if _seen(run_context) < _seen(run.context):
        return False
    run.context = run_context
    run.pending, run.unsent_since, run.failures = False, None, 0
    return True


class _Turns:
    """A run's saves on a ``sync`` provider, from the moment two of them overlapped: one write at a time.

    ``started`` counts the writes begun, ``landed`` is the latest of them that succeeded. A save asked for
    when n had begun is stored once write n + 1 or a later one has landed: that write took its snapshot
    after the save was asked for.
    """

    __slots__ = ("lock", "started", "landed")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.started = 1  # the write in flight this was made for
        self.landed = 0


def _key(run_context: RunContext) -> tuple[int, str]:
    return id(run_context._tracing_provider), run_context.id


def schedule(run_context: RunContext) -> bool:
    """Defer this mid-run save. False: the run is too far behind, and the caller saves synchronously."""
    provider = run_context._tracing_provider
    loop = asyncio.get_running_loop()
    run = _RUNS.get(_key(run_context))
    if run is None:
        run = _RUNS[_key(run_context)] = _RunUploads(run_context, provider)
    elif not _holds(run, run_context):
        return True  # an older context of a run id used again: nothing of it is stored
    now = loop.time()
    # At or past the limit (inclusive: on a coarse clock, Windows before Python 3.13, a backlog started in
    # this 15.6 ms tick reads as 0 s behind, and max_lag_s=0 must still mean "always synchronous").
    if run.unsent_since is not None and now - run.unsent_since >= _max_lag(provider):
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


async def write_through(run_context: RunContext, put: Callable[[Any], Awaitable[None]]) -> bool:
    """Store the run's latest state now, as its next write (a barrier).

    It waits for an upload in flight or another barrier's write (the root's end and a background task's end
    can overlap), never for a drain waiting out its interval or a retry backoff; saves requested while it
    writes are uploaded after it; what it covers is not uploaded again. Raises what ``put`` raises.

    Every save of a ``sync`` provider comes through here, and waits the same way for one of its run that is
    being written (see the module docstring). A run whose saves never overlap takes no lock: its write in
    flight is only marked (``RunContext._save_turn``), and the save that finds the mark makes the lock the
    run's saves queue on from then on. ``put`` must not save the same run itself: that save would wait for
    the one it is part of.

    Returns False when this call wrote nothing because a write that began after it was asked for has stored
    its state (a ``sync`` save that waited): stored all the same, and not a write to count.
    """
    key = _key(run_context)
    run = _RUNS.get(key)
    if run is None:
        provider = run_context._tracing_provider
        if durability(provider) == "sync":
            if provider.overlapping_saves:
                await put(run_context)
                return True
            state = run_context.__dict__
            turn = state.get("_save_turn")
            if turn is None:
                state["_save_turn"] = True
                try:
                    await put(run_context)
                finally:
                    turn = state["_save_turn"]
                    if turn is True:
                        state["_save_turn"] = None
                    else:
                        turn.lock.release()
                return True
            if turn is True:
                # A write of this run is in flight and holds no lock: take one in its name, which it lets go
                # of when it ends. A lock nobody holds is taken without waiting, so that write cannot end first.
                turn = state["_save_turn"] = _Turns()
                await turn.lock.acquire()
            covering = turn.started + 1
            async with turn.lock:
                if turn.landed >= covering:
                    return False  # a write that began after this save was asked for has landed
                turn.started = mine = turn.started + 1
                await put(run_context)
                turn.landed = mine
            return True
        run = _RUNS[key] = _RunUploads(run_context, provider)
    elif not _holds(run, run_context):
        return False  # an older context of a run id used again: not stored, the id is the newer run's
    run.writers += 1
    try:
        async with run.lock:
            if run.context is not run_context:
                return False  # taken over meanwhile by a fresh context of the run id: this one is not stored
            covered = run.requests
            await put(run_context)
            if run.requests == covered:
                run.pending, run.unsent_since, run.failures = False, None, 0
    finally:
        run.writers -= 1
        _release(run)
    return True


def _release(run: _RunUploads, *, draining: bool = False) -> None:
    """Drop a run with nothing left to store, and its drain.

    After a barrier's write the drain cannot be uploading (the barrier held the lock after it): it is waiting
    out an interval or backoff, or queued for the lock, so cancelling it interrupts no write. ``draining``:
    the drain itself is exiting.
    """
    key = _key(run.context)
    if run.pending or run.writers or run.sending or _RUNS.get(key) is not run:
        return
    if not draining and run.task is not None and not run.task.done():
        run.task.cancel()
    del _RUNS[key]


def _finished(run_context: RunContext) -> bool:
    """The root run has ended and no background work is left on it: nothing will save it again."""
    root = run_context.root_span()
    return root is not None and root.t1 is not None and run_context._live_background == 0


async def flush_all(timeout: float | None = None) -> None:
    """Store every buffered run's latest state now, then wait for the platform saves still owed
    (process shutdown). One bound for both: ``timeout``, default ``TIMBAL_TRACE_FLUSH_TIMEOUT_SECS``
    (5 s), inside a pod's termination grace period."""
    from .providers.platform import PlatformTracingProvider, _flush_timeout

    if timeout is None:
        timeout = _flush_timeout()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
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
    await PlatformTracingProvider.flush(max(0.0, deadline - loop.time()))


async def _drain(run: _RunUploads) -> None:
    try:
        await _upload_while_pending(run)
    finally:
        # Nothing saves a finished run again: once its last change is stored (the retry of a failed end
        # barrier, or the upload a cancelled one was waiting on), let it go.
        if _finished(run.context):
            _release(run, draining=True)


async def _upload_while_pending(run: _RunUploads) -> None:
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
