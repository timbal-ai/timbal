"""Opt-in, process-local continuation of an agent's background completions."""

import asyncio
import inspect
import math
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from ..state import get_call_id, get_run_context, set_call_id, set_parent_call_id, set_run_context
from ..state.background import BackgroundTaskStore, format_background_completion_notice, store_for_run
from ..types.events import BaseEvent, OutputEvent

if TYPE_CHECKING:
    from ..state.context import RunContext
    from .agent import Agent

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


def continuation_for_run(run_id: str) -> "AgentContinuation | None":
    """The driver that owns ``run_id``'s session, or None.

    Found through the session's background store, so a session that has not
    spawned a background task yet has no driver to find.
    """
    store = store_for_run(run_id)
    return store.continuation_owner() if store is not None else None


class AgentContinuation:
    """A stream of distinct runs, not a single collectable agent result.

    Consume on one task; close explicitly with ``contextlib.aclosing`` when
    breaking iteration. Stopping, closing, or cancelling never cancels detached
    children. Another root turn in the session preempts the driver while it
    waits between runs, and is rejected while one of the driver's runs is in
    flight. This is not a durable session worker or a voice playback scheduler.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        admit: Callable[[], Awaitable[Any] | Any] | None = None,
        max_idle_secs: float = 300.0,
        max_continuations: int = 10,
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
        self._context: RunContext | None = None
        self._trigger: dict[str, Any] | None = None
        # From the moment a run is committed to until its root OUTPUT.
        self._busy = False
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
        if self._busy:
            self._stop_requested = True
        else:
            self._finish("stopped")

    def _yield_turn(self) -> None:
        """Hand the session to another root turn, or refuse it while a run is in flight."""
        if self._busy:
            raise RuntimeError(
                "A continuation run is in flight in this session; wait for its OUTPUT before starting another turn."
            )
        self._finish("preempted")

    def _finish(self, reason: ContinuationStopReason) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self._busy = False
        self._release()
        if self._waiter is not None:
            self._waiter.cancel()

    def _release(self) -> None:
        context = self._context
        if context is None:
            return
        if context._bg_store is not None:
            context._bg_store.release_continuation(self)
        marker = context._continuation_owner
        if marker is not None and marker() is self:
            context._continuation_owner = None

    async def _iterate(self) -> AsyncGenerator[BaseEvent, None]:
        try:
            if get_call_id() is not None or "." in self._agent._path or self._agent.background_mode != "never":
                raise RuntimeError("Continuation streams require a foreground, top-level Agent.")
            kwargs = self._kwargs
            while self.stop_reason is None:
                self._busy = True
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
                if self._busy:
                    self._finish("run_stopped")
                if self.stop_reason is not None:
                    return
                notices = await self._next_notices()
                if notices is None:
                    return
                kwargs = self._followup_kwargs(notices)
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
        if not self.continuations and context.parent_id is not None:
            previous = await context._get_parent_trace()
            previous_root = self._agent._own_span(previous) if previous is not None else None
            if previous_root is None or previous_root.t1 is None:
                raise ValueError("Continuation streams require a completed, readable parent run.")
        context._continuation_owner = weakref.ref(self)
        self._context = context
        self.last_run_id = context.id
        store = context._bg_store
        if store is not None:
            # A new driver's first run is a root turn like any other: it preempts a waiting driver or is refused.
            store.check_continuation_turn(context)
            store.acquire_continuation(self)
        if self._trigger is not None:
            context.root_span().metadata["trigger"] = self._trigger

    def _end_run(self, output: OutputEvent) -> None:
        self._busy = False
        self.last_status = output.status.code
        if output.status.reason in _PAUSE_REASONS:
            self._finish("paused")
        elif output.status.code != "success":
            self._finish("run_stopped")
        elif self._stop_requested:
            self._finish("stopped")

    async def _next_notices(self) -> list[dict[str, Any]] | None:
        """Wait between runs: the notices a follow-up starts with, or None once the stream ends."""
        context = self._context
        assert context is not None
        store = context._bg_store
        if store is None:
            self._finish("idle")
            return None
        deadline = asyncio.get_running_loop().time() + self._max_idle_secs
        while True:
            self._waiter = asyncio.ensure_future(self._await_followup(context, store, deadline))
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
            # Drain and busy in one step: a root turn either preempts before it or is refused after it.
            notices = store.drain_completions()
            if notices:
                self._busy = True
                return notices

    async def _await_followup(
        self, context: "RunContext", store: BackgroundTaskStore, deadline: float
    ) -> ContinuationStopReason | None:
        """None once a completion is pending and the channel admits a follow-up."""
        if self.continuations >= self._max_continuations:
            return "continuation_limit" if store.has_pending_work() else "idle"
        history_checked = False
        timeout = asyncio.timeout_at(deadline)
        try:
            async with timeout:
                while True:
                    if not await store.wait_for_completion():
                        return "idle"
                    if not history_checked:
                        root = context.root_span()
                        if (
                            root is None
                            or not isinstance(root.memory, list)
                            or self._agent._find_pending_tool_uses(root.memory)
                        ):
                            return "paused"
                        parent = await context._get_parent_trace(context.id)
                        saved = self._agent._own_span(parent) if parent is not None else None
                        if saved is None or not isinstance(saved.memory, list) or saved.t1 is None:
                            return "history_unavailable"
                        history_checked = True
                    if self._admit is not None:
                        admitted = self._admit()
                        if inspect.isawaitable(admitted):
                            await admitted
                    if store.pending_completions():
                        return None
        except TimeoutError:
            if timeout.expired():
                return "idle_timeout"
            raise

    def _followup_kwargs(self, notices: list[dict[str, Any]]) -> dict[str, Any]:
        context = self._context
        assert context is not None
        self.continuations += 1
        self._trigger = {
            "kind": "background_completion",
            "continuation_index": self.continuations,
            "task_ids": [notice["task_id"] for notice in notices],
        }
        kwargs = {key: value for key, value in self._kwargs.items() if key not in _FOLLOWUP_DROPPED_KWARGS}
        kwargs["prompt"] = format_background_completion_notice(notices)
        kwargs["parent_id"] = context.id
        # The follow-up inherits the finished run's platform config from the ambient context.
        set_run_context(context)
        set_call_id(None)
        set_parent_call_id(None)
        return kwargs
