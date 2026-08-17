"""Session-scoped background-task store.

Background children (``background_mode="auto"|"always"``) used to live on the
parent ``Runnable._bg_tasks`` dict. That is process-wide for a singleton Agent:
two dock users share ids, ``get_background_task`` appears as soon as *anyone*
has a child, and a 6-char id is a cross-tenant guess.

Tasks now live on a :class:`BackgroundTaskStore` bound to the ``RunContext``
and inherited across sequential turns via ``parent_id`` (same in-process
session). Concurrent parent runs — no shared ``parent_id`` — get isolated
stores.

The event log is append-only. ``get_background_task`` peeks a summary; it does
not drain. Raw events are ``read_background_transcript(after=)`` or
:meth:`BackgroundEventLog.subscribe`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

_SUMMARY_TEXT_CHARS = 1_000
_TASK_ID_LEN = 12
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# Process-local: run_id → store. Sequential turns find the bag via parent_id.
# Concurrent siblings (no parent_id) each create a new store.
_STORES_BY_RUN_ID: dict[str, BackgroundTaskStore] = {}

_DONE = object()


def _new_task_id() -> str:
    import secrets

    return "".join(secrets.choice(_ALPHABET) for _ in range(_TASK_ID_LEN))


def _dump_event(event: Any) -> Any:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    return event


_EVENT_CLASSES: dict[str, Any] = {}


def _event_classes() -> dict[str, Any]:
    """Event classes, imported on first use.

    ``timbal.types`` imports back into ``timbal.state`` (File needs the run
    context), so importing events at module scope here deadlocks a cold
    interpreter: ``state/__init__`` → background → types → state (partial).
    """
    if not _EVENT_CLASSES:
        from ..types.events import OutputEvent
        from ..types.events.delta import DeltaEvent, Text, TextDelta, ToolUse

        _EVENT_CLASSES.update(
            delta_event=DeltaEvent,
            output_event=OutputEvent,
            text=Text,
            text_delta=TextDelta,
            tool_use=ToolUse,
        )
    return _EVENT_CLASSES


def _event_text(event: Any, classes: dict[str, Any] | None = None) -> str:
    classes = classes or _event_classes()
    if isinstance(event, classes["delta_event"]):
        item = event.item
        if isinstance(item, classes["text_delta"]):
            return item.text_delta
        if isinstance(item, classes["text"]):
            return item.text
        return ""
    if isinstance(event, classes["output_event"]) and event.output is not None:
        output = event.output
        collect = getattr(output, "collect_text", None)
        if callable(collect):
            try:
                return collect() or ""
            except Exception:
                return ""
        if isinstance(output, str):
            return output
    return ""


def _event_tool_name(event: Any, classes: dict[str, Any] | None = None) -> str | None:
    classes = classes or _event_classes()
    if isinstance(event, classes["delta_event"]) and isinstance(event.item, classes["tool_use"]):
        return event.item.name
    return None


def _lift_metadata(event: Any, metadata: dict[str, Any]) -> None:
    """Copy durable child ids off START/OUTPUT onto the task record."""
    run_id = getattr(event, "run_id", None)
    if run_id and not metadata.get("run_id"):
        metadata["run_id"] = run_id
    event_meta = getattr(event, "metadata", None)
    if not isinstance(event_meta, dict):
        return
    for key in ("cursor_agent_id", "agent_id"):
        if event_meta.get(key) and not metadata.get(key):
            metadata[key] = event_meta[key]


def _title_from_input(name: str, input: dict[str, Any]) -> str:
    for key in ("prompt", "tag", "message", "task_name", "project_name"):
        value = input.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            return text if len(text) <= 80 else text[:77] + "..."
        collect = getattr(value, "collect_text", None)
        if callable(collect):
            text = (collect() or "").strip()
            if text:
                return text if len(text) <= 80 else text[:77] + "..."
    return name


class BackgroundEventLog:
    """Append-only log. ``put_nowait`` matches :class:`asyncio.Queue` so
    ``_execute_handler`` can treat this as the event sink."""

    def __init__(self) -> None:
        self._events: list[Any] = []
        self._subscribers: list[asyncio.Queue] = []

    def put_nowait(self, event: Any) -> None:
        self._events.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)

    def peek(self, after: int = 0) -> list[Any]:
        return list(self._events[after:])

    def qsize(self) -> int:
        return len(self._events)

    def empty(self) -> bool:
        return not self._events

    async def subscribe(self, after: int = 0) -> AsyncIterator[Any]:
        """Replay from ``after``, then yield live events until :meth:`close`."""
        queue: asyncio.Queue = asyncio.Queue()
        # Register before replay so we cannot drop events that arrive mid-replay.
        self._subscribers.append(queue)
        try:
            for event in self._events[after:]:
                yield event
            while True:
                event = await queue.get()
                if event is _DONE:
                    return
                yield event
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def close(self) -> None:
        for queue in self._subscribers:
            queue.put_nowait(_DONE)


class BackgroundTask:
    """One detached child: asyncio.Task + durable event log."""

    def __init__(
        self,
        task_id: str,
        *,
        name: str,
        input: dict[str, Any],
        task: asyncio.Task,
        started_at: int,
        title: str | None = None,
        on_cancel: Callable[..., Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.input = input
        self.task = task
        self.started_at = started_at
        self.title = title or _title_from_input(name, input)
        self.log = BackgroundEventLog()
        self.metadata: dict[str, Any] = {}
        self.on_cancel = on_cancel
        self._cancel_requested = False
        self._handler_aclose: Callable[..., Any] | None = None

    async def aclose_handler(self) -> None:
        """Stop the child's handler gen, not just the wrapping asyncio.Task.

        Cancelling the Task unwinds the consumer, but an async generator
        suspended *at a yield* is left suspended — its ``finally`` never runs,
        so a handler holding a subprocess/socket would leak. We must ``aclose``
        it, and only after the Task has finished unwinding: calling ``aclose``
        while the generator is still running raises "already running".
        """
        task = self.task
        if not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                pass
        aclose = self._handler_aclose
        self._handler_aclose = None
        if aclose is None:
            return
        try:
            await aclose()
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception:
            pass

    def ingest(self, event: Any) -> None:
        _lift_metadata(event, self.metadata)
        self.log.put_nowait(event)

    def put_nowait(self, event: Any) -> None:
        """Queue-shaped sink used by ``_execute_handler``."""
        self.ingest(event)

    def status_code(self) -> str:
        task = self.task
        if self._cancel_requested or (task.done() and task.cancelled()):
            return "cancelled"
        if not task.done():
            return "running"
        if task.exception() is not None:
            return "error"
        return "completed"

    def summarize(self) -> dict[str, Any]:
        events = self.log.peek()
        classes = _event_classes()
        text_parts: list[str] = []
        tools: list[str] = []
        for event in events:
            chunk = _event_text(event, classes)
            if chunk:
                text_parts.append(chunk)
            tool_name = _event_tool_name(event, classes)
            if tool_name:
                tools.append(tool_name)
        text = "".join(text_parts)
        if len(text) > _SUMMARY_TEXT_CHARS:
            text = text[-_SUMMARY_TEXT_CHARS:]
        snapshot: dict[str, Any] = {
            "status": self.status_code(),
            "task_id": self.task_id,
            "name": self.name,
            "input": self.input,
            "title": self.title,
            "started_at": self.started_at,
            "summary": {
                "text": text,
                "tools_in_flight": tools,
                "event_count": len(events),
            },
            "transcript_cursor": len(events),
        }
        snapshot.update(self.metadata)
        if self.task.done() and not self.task.cancelled():
            exc = self.task.exception()
            if exc is not None:
                snapshot["error"] = str(exc)
            else:
                snapshot["result"] = self.task.result()
        return snapshot

    def listing(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status_code(),
            "started_at": self.started_at,
            "title": self.title,
        }


class BackgroundTaskStore:
    """Per-session bag of background children. Shared across sequential
    ``RunContext``s in the same parent_id chain; not across concurrent runs."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def add(self, record: BackgroundTask) -> None:
        self._tasks[record.task_id] = record

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list(self) -> list[dict[str, Any]]:
        return [record.listing() for record in self._tasks.values()]

    def snapshot(self, task_id: str) -> dict[str, Any]:
        record = self._tasks.get(task_id)
        if record is None:
            return {"status": "not_found", "task_id": task_id}
        return record.summarize()

    def transcript(self, task_id: str, after: int = 0) -> dict[str, Any]:
        record = self._tasks.get(task_id)
        if record is None:
            return {"status": "not_found", "task_id": task_id, "events": [], "cursor": after}
        events = record.log.peek(after)
        return {
            "status": record.status_code(),
            "task_id": task_id,
            "events": [_dump_event(event) for event in events],
            "cursor": after + len(events),
        }

    def cancel(self, task_id: str) -> dict[str, Any]:
        record = self._tasks.get(task_id)
        if record is None:
            return {"status": "not_found", "task_id": task_id}
        if not record.task.done():
            record._cancel_requested = True
            record.task.cancel()
            try:
                asyncio.get_running_loop().create_task(record.aclose_handler())
            except RuntimeError:
                pass
        if record.on_cancel is not None:
            try:
                record.on_cancel(record)
            except Exception:
                pass
        return record.summarize()


def bind_background_store(run_context: Any) -> BackgroundTaskStore | None:
    """Inherit the parent session's bag, if there is one.

    Runs on **every** ``RunContext`` construction, so it must stay cheap and
    must not allocate: the overwhelming majority of runs never spawn a
    background child. No store is created and nothing is registered in the
    process-local map until :func:`ensure_background_store` is called from the
    spawn path. Otherwise the map would grow by one entry per run forever.
    """
    store = None
    parent_id = run_context.parent_id
    if parent_id is not None:
        store = _STORES_BY_RUN_ID.get(parent_id)
        if store is not None:
            # Keep the chain alive: the next turn parented on this run finds it.
            _STORES_BY_RUN_ID[run_context.id] = store
    run_context._bg_store = store
    return store


def ensure_background_store(run_context: Any) -> BackgroundTaskStore:
    """Get (creating if needed) the session bag, and register it for chaining.

    Called from the spawn path only — this is what puts a run in the
    process-local map.
    """
    store = run_context._bg_store
    if store is None:
        store = BackgroundTaskStore()
        run_context._bg_store = store
    _STORES_BY_RUN_ID[run_context.id] = store
    return store


def store_for_run(run_id: str) -> BackgroundTaskStore | None:
    """Look up the session bag registered for a run id (tests / JobStore)."""
    return _STORES_BY_RUN_ID.get(run_id)


def current_background_store() -> BackgroundTaskStore | None:
    from . import get_run_context

    run_context = get_run_context()
    if run_context is None:
        return None
    return getattr(run_context, "_bg_store", None)


def get_background_task(task_id: str) -> dict[str, Any]:
    """Peek a summary of a background task. Does not drain the event log.

    Use this to answer questions about an in-flight or finished background
    tool. You get status, a short text summary, and any child ids (e.g.
    ``cursor_agent_id``, ``run_id``). You must have a ``task_id`` from a
    previous background-tool result or from ``list_background_tasks``.
    Raw events: ``read_background_transcript``.
    """
    store = current_background_store()
    if store is None:
        return {"status": "not_found", "task_id": task_id}
    return store.snapshot(task_id)


def list_background_tasks() -> list[dict[str, Any]]:
    """List background tools for this session (not other concurrent runs).

    Returns ``[{task_id, name, status, started_at, title}]``. Use a ``task_id``
    with ``get_background_task`` to answer questions about a child.
    """
    store = current_background_store()
    if store is None:
        return []
    return store.list()


def cancel_background_task(task_id: str) -> dict[str, Any]:
    """Cancel a running background task and stop its in-flight work.

    Cancels the asyncio.Task (the child's handler sees ``CancelledError``).
    If the child registered ``on_background_cancel``, that hook runs too
    (e.g. to stop an external harness).
    """
    store = current_background_store()
    if store is None:
        return {"status": "not_found", "task_id": task_id}
    return store.cancel(task_id)


def read_background_transcript(task_id: str, after: int = 0) -> dict[str, Any]:
    """Raw events for a background task, from cursor ``after`` (inclusive).

    Does not drain. A second read with the same ``after`` returns the same
    events. ``cursor`` is the next index to pass.
    """
    store = current_background_store()
    if store is None:
        return {"status": "not_found", "task_id": task_id, "events": [], "cursor": after}
    return store.transcript(task_id, after=after)


def register_background_task(
    *,
    name: str,
    input: dict[str, Any],
    task: asyncio.Task,
    on_cancel: Callable[..., Any] | None = None,
    started_at: int | None = None,
) -> BackgroundTask:
    """Register a running child on the current session bag."""
    from . import get_run_context

    run_context = get_run_context()
    if run_context is None:
        raise RuntimeError("Cannot register a background task without a RunContext.")
    store = ensure_background_store(run_context)
    record = BackgroundTask(
        _new_task_id(),
        name=name,
        input=input,
        task=task,
        started_at=started_at if started_at is not None else int(time.time() * 1000),
        on_cancel=on_cancel,
    )
    store.add(record)
    return record


def clear_background_stores() -> None:
    """Drop the process-local registry. Tests only."""
    _STORES_BY_RUN_ID.clear()


GET_BACKGROUND_TASK_DESCRIPTION = (
    "Get a summary of an in-flight or finished background tool so you can "
    "answer questions about it (status, last assistant text, tools it called, "
    "error). You must have a task_id from a previous background-tool result "
    "or from list_background_tasks. This does not return the full event log — "
    "use it to brief the user, not to dump a long build."
)

LIST_BACKGROUND_TASKS_DESCRIPTION = (
    "List background tools started in this session: task_id, name, status, "
    "started_at, title. Use this when the user asks what builders/tasks are "
    "running or to find a task_id for get_background_task / cancel_background_task. "
    "You only see this session — not other users."
)

CANCEL_BACKGROUND_TASK_DESCRIPTION = (
    "Cancel a running background tool by task_id (from a previous spawn result "
    "or list_background_tasks). Stops the child's in-flight work. Use when the "
    "user says to stop/cancel a builder or background task."
)
