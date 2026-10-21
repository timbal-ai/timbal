"""Opt-in, process-local continuation of an agent's background completions."""

import asyncio
import math
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from ..state import get_call_id, get_run_context, set_call_id, set_parent_call_id, set_run_context
from ..state.background import BackgroundTaskStore, ensure_background_store
from ..types.events import BaseEvent, OutputEvent
from ..types.message import Message

if TYPE_CHECKING:
    from .agent import Agent

ContinuationStopReason = Literal[
    "idle",
    "idle_timeout",
    "continuation_limit",
    "paused",
    "run_stopped",
    "history_unavailable",
    "closed",
    "cancelled",
    "error",
]


class AgentContinuation:
    """A stream of distinct runs, not a single collectable agent result.

    Consume on one task. Close explicitly with ``contextlib.aclosing`` when
    breaking iteration. Cancellation/close never cancels detached children.
    Callers must stop this driver before starting another turn in its session.
    This is not a durable session worker or a voice playback scheduler.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        max_idle_secs: float,
        max_continuations: int,
        kwargs: dict[str, Any],
    ) -> None:
        if not math.isfinite(max_idle_secs) or max_idle_secs <= 0:
            raise ValueError("max_idle_secs must be finite and greater than zero.")
        if isinstance(max_continuations, bool) or not isinstance(max_continuations, int) or max_continuations < 0:
            raise ValueError("max_continuations must be a non-negative integer.")
        self.stop_reason: ContinuationStopReason | None = None
        self.last_run_id: str | None = None
        self.continuations = 0
        self._agent = agent
        self._owner = object()
        self._max_idle_secs = max_idle_secs
        self._max_continuations = max_continuations
        self._kwargs = kwargs
        self._stream = self._iterate()

    def __aiter__(self) -> "AgentContinuation":
        return self

    async def __anext__(self) -> BaseEvent:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        """Close the current run/driver, leaving background work owned by its store."""
        await self._stream.aclose()
        if self.stop_reason is None:
            self.stop_reason = "closed"

    async def _iterate(self) -> AsyncGenerator[BaseEvent, None]:
        store: BackgroundTaskStore | None = None
        acquired = False
        try:
            if get_call_id() is not None or "." in self._agent._path or self._agent.background_mode != "never":
                raise RuntimeError("Continuation streams require a foreground, top-level Agent.")
            kwargs = dict(self._kwargs)
            while True:
                context = None
                root_call_id = None
                output = None
                async with aclosing(self._agent(**kwargs)) as stream:
                    async for event in stream:
                        consumer = asyncio.current_task()
                        if consumer is not None and consumer.cancelling():
                            raise asyncio.CancelledError
                        if root_call_id is None:
                            context = get_run_context()
                            if context is None or event.parent_call_id is not None:
                                raise RuntimeError("Continuation stream did not start a top-level run.")
                            if context._tracing_provider is None:
                                raise ValueError("Continuation streams require a tracing provider for turn history.")
                            candidate = ensure_background_store(context)
                            if store is None:
                                store = candidate
                                store.acquire_continuation(self._owner)
                                acquired = True
                                if context.parent_id is not None:
                                    previous = await context._get_parent_trace()
                                    previous_root = self._agent._own_span(previous) if previous is not None else None
                                    if previous_root is None or previous_root.t1 is None:
                                        raise ValueError(
                                            "Continuation streams require a completed, readable parent run."
                                        )
                            elif candidate is not store:
                                raise RuntimeError("Continuation run lost its background-task store.")
                            store.authorize_continuation_turn(self._owner, context.id)
                            root_call_id = event.call_id
                            self.last_run_id = context.id
                            if self.continuations:
                                context._background_continuation = True
                                context.root_span().metadata["trigger"] = {
                                    "kind": "background_completion",
                                    "continuation_index": self.continuations,
                                }
                        if isinstance(event, OutputEvent) and event.call_id == root_call_id:
                            output = event
                        yield event
                if output is None or context is None or store is None:
                    self.stop_reason = "run_stopped"
                    return
                if output.status.reason in {"approval_required", "input_required"}:
                    self.stop_reason = "paused"
                    return
                if output.status.code != "success":
                    self.stop_reason = "run_stopped"
                    return
                if self.continuations >= self._max_continuations and (
                    store.pending_completions() or store.running_count()
                ):
                    self.stop_reason = "continuation_limit"
                    return
                try:
                    async with asyncio.timeout(self._max_idle_secs):
                        while True:
                            if not await store.wait_for_completion():
                                self.stop_reason = "idle"
                                return
                            if self.continuations >= self._max_continuations:
                                self.stop_reason = "continuation_limit"
                                return
                            root = context.root_span()
                            if (
                                root is None
                                or not isinstance(root.memory, list)
                                or self._agent._find_pending_tool_uses(root.memory)
                            ):
                                self.stop_reason = "paused"
                                return
                            parent = await context._get_parent_trace(context.id)
                            saved_span = self._agent._own_span(parent) if parent is not None else None
                            if saved_span is None or not isinstance(saved_span.memory, list) or saved_span.t1 is None:
                                self.stop_reason = "history_unavailable"
                                return
                            if store.pending_completions():
                                break
                except TimeoutError:
                    self.stop_reason = "idle_timeout"
                    return
                kwargs = {
                    key: value
                    for key, value in self._kwargs.items()
                    if key not in {"prompt", "messages", "parent_id", "resume"}
                }
                kwargs.update(
                    prompt=Message(
                        role="user", content=[], metadata={"source": "runtime", "kind": "background_wakeup"}
                    ),
                    messages=None,
                    parent_id=context.id,
                )
                set_run_context(context)
                set_call_id(None)
                set_parent_call_id(None)
                self.continuations += 1
        except asyncio.CancelledError:
            self.stop_reason = "cancelled"
            raise
        except GeneratorExit:
            self.stop_reason = "closed"
            raise
        except BaseException:
            self.stop_reason = "error"
            raise
        finally:
            if acquired and store is not None:
                store.release_continuation(self._owner)
