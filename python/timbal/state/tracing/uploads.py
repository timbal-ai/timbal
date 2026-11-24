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
import os
import threading
import time
import weakref
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
        "lock", "requests", "writers", "loop",
    )  # fmt: skip

    def __init__(self, context: RunContext, provider: type[TracingProvider]) -> None:
        self.context = context
        self.provider = provider
        # The event loop its lock and drain belong to: a save of the run on another loop does not queue here.
        self.loop = asyncio.get_running_loop()
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


class _Writing:
    """A ``sync`` write of a run in flight on one event loop, with no queue there yet: a mark, by identity."""

    __slots__ = ()


class _Turns:
    """A run's saves on a ``sync`` provider on one event loop, from the moment two of them overlapped there: one
    write at a time.

    ``started`` counts the writes begun, ``landed`` is the latest of them that succeeded. A save asked for
    when n had begun is stored once write n + 1 or a later one has landed: that write took its snapshot
    after the save was asked for.
    """

    __slots__ = ("lock", "started", "landed")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.started = 1  # the write in flight this was made for
        self.landed = 0


class _Queues(dict):
    """A run context's ``sync`` saves, per event loop (``RunContext._save_turn``): for each loop, a ``_Writing``
    mark while one write of the run is in flight there, or the queue its saves wait in once two overlapped there
    (``_Turns``). Each loop's saves are in order among themselves. Saves of the run on two loops (a tool whose
    handler runs nested work with ``asyncio.run()`` on the same run context) are not ordered against each other
    and never wait on each other: an ``asyncio.Lock`` is one loop's. ``owner``: the run context it belongs to,
    held weakly: a copy of the context (``model_copy``) carries this object along and makes its own."""

    __slots__ = ("owner",)

    def __init__(self, owner: RunContext) -> None:
        super().__init__()
        self.owner = weakref.ref(owner)


# Every change to a run context's _Queues, from whichever thread its loop runs on. Nothing is allocated while it is
# held: an allocation can set off a collection, which can finish a save abandoned on a closed loop, whose cleanup
# changes the run's _Queues too. Re-entrant all the same, should a collection land there.
_QUEUES_LOCK = threading.RLock()


def _new_queues_lock_after_fork() -> None:
    """In a forked child: a parent thread may have held the lock at the fork, and nothing would let it go."""
    global _QUEUES_LOCK
    _QUEUES_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_queues_lock_after_fork)


def _gone(loop: asyncio.AbstractEventLoop) -> bool:
    """A loop that will not run the run's uploads further: closed, or stopped (a helper's loop between
    ``run_until_complete`` calls, kept but not running)."""
    return loop.is_closed() or not loop.is_running()


def _key(run_context: RunContext) -> tuple[int, str]:
    return id(run_context._tracing_provider), run_context.id


def schedule(run_context: RunContext) -> bool:
    """Defer this mid-run save. False: the run is too far behind, and the caller saves synchronously."""
    provider = run_context._tracing_provider
    loop = asyncio.get_running_loop()
    run = _RUNS.get(_key(run_context))
    if run is None or (run.loop is not loop and _gone(run.loop)):
        # None yet, or the loop that had them is closed or stopped, and its drain with it: this loop takes them over.
        run = _RUNS[_key(run_context)] = _RunUploads(run_context, provider)
    elif run.loop is not loop:
        return False  # the run's uploads are another live loop's: stored now, by the caller
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

    Saves are ordered per event loop: a save of the run from another loop (a tool whose handler runs nested
    work with ``asyncio.run()`` on the same run context) queues with that loop's saves, and is ordered against
    this loop's only by a provider that orders its own writes (``SqliteTracingProvider`` numbers its
    snapshots). The same for a buffered run: its uploads are the loop's that started them, and a barrier from
    another live loop is stored as it comes. A copy of the run context made while it was being written does not
    wait for the original's write.

    Returns False when this call wrote nothing because a write that began after it was asked for has stored
    its state (a ``sync`` save that waited): stored all the same, and not a write to count.
    """
    key = _key(run_context)
    run = _RUNS.get(key)
    if run is None:
        provider = run_context._tracing_provider
        if durability(provider) == "sync":
            if provider.overlapping_saves and not provider._exporters:
                # Exporters are awaited in put(): with some attached, their exports are ordered by the queue.
                await put(run_context)
                return True
            loop = asyncio.get_running_loop()
            state = run_context.__dict__
            took = False
            while True:
                # What the step below may need is made first, outside the lock (see _QUEUES_LOCK); a step that
                # finds it needs something it did not make goes round again.
                queues = state.get("_save_turn")
                fresh = _Queues(run_context) if queues is None or queues.owner() is not run_context else None
                seen = None if fresh is not None else queues.get(loop)
                mark, turns = _Writing(), (_Turns() if type(seen) is _Writing else None)
                with _QUEUES_LOCK:
                    queues = state.get("_save_turn")
                    if queues is None or queues.owner() is not run_context:
                        # None yet, or the queues of the context this one was copied from (model_copy), which
                        # this one would wait on in vain.
                        if fresh is None:
                            continue
                        queues = state["_save_turn"] = fresh
                    turn = queues.get(loop)
                    if turn is None:
                        mine = queues[loop] = mark
                    elif type(turn) is _Writing:
                        # A write of this run is in flight on this loop and holds no lock: take one in its name,
                        # which it lets go of when it ends. A lock nobody holds is taken without waiting, so that
                        # write cannot end first.
                        if turns is None:
                            continue
                        turn = queues[loop] = turns
                        took = True
                break
            if turn is None:
                try:
                    await put(run_context)
                finally:
                    with _QUEUES_LOCK:
                        turn = queues.get(loop)
                        if turn is mine:
                            del queues[loop]  # the queues object stays: it may be in use on another loop
                            turn = None
                    if turn is not None and not loop.is_closed():
                        turn.lock.release()  # the queue a save of this loop made in this write's name
                return True
            if took:
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
    elif run.loop is not asyncio.get_running_loop():
        if _gone(run.loop):
            # The loop that had the run's uploads is closed or stopped, and its drain with it: this loop takes them
            # over.
            run = _RUNS[key] = _RunUploads(run_context, run.provider)
        else:
            # A barrier on another live event loop: the run's lock and drain are that loop's. Stored as it comes.
            await put(run_context)
            return True
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
