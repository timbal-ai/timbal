"""Opt-in, process-local continuation of an agent's background completions and session events."""

import asyncio
import inspect
import math
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from ..state import get_call_id, get_run_context, set_call_id, set_parent_call_id, set_run_context
from ..state.background import forget_idle_background_store, store_for_run
from ..types.events import BaseEvent, OutputEvent
from .session_scheduler import Followup, OnBusy, SessionBusyError, SessionScheduler, take_followup

if TYPE_CHECKING:
    from ..state.context import RunContext
    from .agent import Agent

__all__ = [
    "AgentContinuation",
    "ContinuationStopReason",
    "Followup",
    "SessionBusyError",
    "continuation_for_run",
    "take_followup",
]

ContinuationStopReason = Literal[
    "idle",
    "idle_timeout",
    "continuation_limit",
    "paused",
    "run_stopped",
    "history_unavailable",
    "preempted",
    "stopped",
    "closed",
    "cancelled",
    "error",
]

_PAUSE_REASONS = frozenset({"approval_required", "input_required"})
_FOLLOWUP_DROPPED_KWARGS = frozenset({"prompt", "messages", "parent_id", "resume"})


class _StopWaiting(Exception):
    def __init__(self, reason: ContinuationStopReason) -> None:
        self.reason = reason


def continuation_for_run(run_id: str) -> "AgentContinuation | None":
    """The driver that owns ``run_id``'s session, or None.

    Found through the session's background store, so a session that has not
    spawned a background task or received an event yet has no driver to find.
    """
    store = store_for_run(run_id)
    owner = store.continuation_owner() if store is not None else None
    channel = owner.channel if isinstance(owner, SessionScheduler) else None
    return channel if isinstance(channel, AgentContinuation) else None


class AgentContinuation:
    """A stream of distinct runs, not a single collectable agent result.

    Consume on one task; close explicitly with ``contextlib.aclosing`` when
    breaking iteration. Stopping, closing, or cancelling never cancels detached
    children. Another root turn in the session preempts the driver while it
    waits between runs; while one of the driver's runs is in flight it is
    rejected (``on_busy="reject"``) or waits for that run's OUTPUT and then
    preempts it (``"enqueue"``). Finished tasks and events with
    ``announce="silent"`` start no follow-up; they reach the next turn.
    This is not a durable session worker or a voice playback scheduler.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        admit: Callable[[], Awaitable[Any] | Any] | None = None,
        max_idle_secs: float = 300.0,
        max_continuations: int = 10,
        on_busy: OnBusy = "reject",
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(max_idle_secs) or max_idle_secs <= 0:
            raise ValueError("max_idle_secs must be finite and greater than zero.")
        if isinstance(max_continuations, bool) or not isinstance(max_continuations, int) or max_continuations < 0:
            raise ValueError("max_continuations must be a non-negative integer.")
        if admit is not None and not callable(admit):
            raise TypeError("admit must be a parameterless callable.")
        self.stop_reason: ContinuationStopReason | None = None
        self.last_run_id: str | None = None
        self.last_status: str | None = None
        self.continuations = 0
        self._agent = agent
        self._admit = admit
        self._max_idle_secs = max_idle_secs
        self._max_continuations = max_continuations
        self._kwargs = kwargs
        self._scheduler = SessionScheduler(channel=self, on_busy=on_busy, on_preempt=self._preempted)
        self._context: RunContext | None = None
        self._followup: Followup | None = None
        self._stop_requested = False
        self._waiter: asyncio.Task | None = None
        self._stream = self._iterate()

    def __aiter__(self) -> "AgentContinuation":
        return self

    async def __anext__(self) -> BaseEvent:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        """Close the current run and the driver, leaving background work owned by its store."""
        await self._stream.aclose()
        if self.stop_reason is None:
            self.stop_reason = "closed"

    def stop(self) -> None:
        """End the stream at the next boundary; detached children keep running.

        Immediate while the driver waits between runs. With a run in flight,
        that run reaches its OUTPUT first and no follow-up starts after it.
        """
        if self.stop_reason is not None:
            return
        if self._scheduler.busy:
            self._stop_requested = True
        else:
            self._finish("stopped")

    def _preempted(self) -> None:
        self._finish("preempted")

    def _finish(self, reason: ContinuationStopReason) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self._scheduler.turn_finished()
        self._release()
        if self._waiter is not None:
            self._waiter.cancel()

    def _release(self) -> None:
        if self._followup is not None:
            self._followup.release()
        context = self._context
        store = context._bg_store if context is not None else None
        self._scheduler.release(store)
        if context is None:
            return
        self._scheduler.unmark(context)
        if store is not None:
            forget_idle_background_store(store)

    async def _iterate(self) -> AsyncGenerator[BaseEvent, None]:
        try:
            if get_call_id() is not None or "." in self._agent._path or self._agent.background_mode != "never":
                raise RuntimeError("Continuation streams require a foreground, top-level Agent.")
            kwargs = self._kwargs
            while self.stop_reason is None:
                self._scheduler.turn_started()
                root_call_id = None
                async with aclosing(self._agent(**kwargs)) as stream:
                    async for event in stream:
                        consumer = asyncio.current_task()
                        if consumer is not None and consumer.cancelling():
                            # The run reports its own cancellation as a cancelled OUTPUT and returns normally.
                            raise asyncio.CancelledError
                        if root_call_id is None:
                            await self._start_run(event)
                            root_call_id = event.call_id
                        if isinstance(event, OutputEvent) and event.call_id == root_call_id:
                            self._end_run(event)
                        yield event
                if self._scheduler.busy:
                    self._finish("run_stopped")
                if self.stop_reason is not None:
                    return
                followup = await self._next_followup()
                if followup is None:
                    return
                kwargs = self._followup_kwargs(followup)
        except asyncio.CancelledError:
            self._finish("cancelled")
            raise
        except GeneratorExit:
            self._finish("closed")
            raise
        except BaseException:
            self._finish("error")
            raise
        finally:
            self._release()

    async def _start_run(self, event: BaseEvent) -> None:
        context = get_run_context()
        if context is None or event.parent_call_id is not None:
            raise RuntimeError("Continuation stream did not start a top-level run.")
        if context._tracing_provider is None:
            raise ValueError("Continuation streams require a tracing provider for turn history.")
        self._scheduler.mark_own(context)
        self._context = context
        self.last_run_id = context.id
        store = context._bg_store
        if store is not None:
            other = store.continuation_owner()
            if other is not None and other is not self._scheduler:
                # A new driver's first run is a root turn like any other (and may be re-chained).
                await other.admit_foreign(context, store)
            self._scheduler.bind(store)
        self._scheduler.note_run(context.id)
        if not self.continuations and context.parent_id is not None:
            previous = await context._get_parent_trace()
            previous_root = self._agent._own_span(previous) if previous is not None else None
            if previous_root is None or previous_root.t1 is None:
                raise ValueError("Continuation streams require a completed, readable parent run.")
        if self._followup is not None:
            self._followup.started(context)
            context.root_span().metadata["trigger"]["continuation_index"] = self.continuations

    def _end_run(self, output: OutputEvent) -> None:
        self._scheduler.turn_finished()
        self.last_status = output.status.code
        if output.status.reason in _PAUSE_REASONS:
            self._finish("paused")
        elif output.status.code != "success":
            self._finish("run_stopped")
        elif self._stop_requested:
            self._finish("stopped")

    async def _next_followup(self) -> Followup | None:
        """Wait between runs: what the follow-up starts with, or None once the stream ends."""
        context = self._context
        assert context is not None
        store = context._bg_store
        if store is None:
            self._finish("idle")
            return None
        self._scheduler.bind(store)
        deadline = asyncio.get_running_loop().time() + self._max_idle_secs
        while True:
            self._waiter = asyncio.ensure_future(self._await_followup(context, deadline))
            try:
                reason = await self._waiter
            except asyncio.CancelledError:
                consumer = asyncio.current_task()
                if self.stop_reason is None or (consumer is not None and consumer.cancelling()):
                    raise
                return None
            finally:
                self._waiter = None
            if reason is not None:
                self._finish(reason)
            if self.stop_reason is not None:
                return None
            # Take and busy in one step: a root turn either preempts before it or meets a busy owner after it.
            followup = self._scheduler.take()
            if followup is not None:
                self._scheduler.turn_started()
                return followup

    async def _await_followup(self, context: "RunContext", deadline: float) -> ContinuationStopReason | None:
        """None once a follow-up is due and admitted; otherwise why the stream ends."""
        scheduler = self._scheduler
        if self.continuations >= self._max_continuations:
            return "continuation_limit" if scheduler.may_need_turn() else "idle"
        history_checked = False

        async def admit() -> None:
            nonlocal history_checked
            if not history_checked:
                root = context.root_span()
                if (
                    root is None
                    or not isinstance(root.memory, list)
                    or self._agent._find_pending_tool_uses(root.memory)
                ):
                    raise _StopWaiting("paused")
                parent = await context._get_parent_trace(context.id)
                saved = self._agent._own_span(parent) if parent is not None else None
                if saved is None or not isinstance(saved.memory, list) or saved.t1 is None:
                    raise _StopWaiting("history_unavailable")
                history_checked = True
            if self._admit is not None:
                admitted = self._admit()
                if inspect.isawaitable(admitted):
                    await admitted

        timeout = asyncio.timeout_at(deadline)
        try:
            async with timeout:
                urgency = await scheduler.wait_ready(admit)
        except _StopWaiting as stop:
            return stop.reason
        except TimeoutError:
            if timeout.expired():
                return "idle_timeout"
            raise
        return None if urgency is not None else "idle"

    def _followup_kwargs(self, followup: Followup) -> dict[str, Any]:
        context = self._context
        assert context is not None
        self.continuations += 1
        self._followup = followup
        kwargs = {key: value for key, value in self._kwargs.items() if key not in _FOLLOWUP_DROPPED_KWARGS}
        kwargs["prompt"] = followup.prompt
        kwargs["parent_id"] = context.id
        # The follow-up inherits the finished run's platform config from the ambient context.
        set_run_context(context)
        set_call_id(None)
        set_parent_call_id(None)
        return kwargs
