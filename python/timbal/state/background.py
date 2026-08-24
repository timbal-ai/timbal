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

When a child reaches a terminal status, the store enqueues a one-shot
completion notice. The parent agent drains that inbox at the **start** of the
next turn and injects a synthetic user message so the LLM learns without
polling — never mid-turn while the parent is still talking.

Optional ``timeout`` (seconds) on a child cancels it as ``timed_out`` when the
deadline elapses (Task cancel + handler ``aclose`` + ``on_background_cancel``).
Optional ``stall_timeout`` cancels as ``stalled`` when no log events arrive
within the window (resets on every event).

Session caps bound fan-out: ``max_concurrent`` (in-flight) and ``max_depth``
(nesting). Defaults come from ``TIMBAL_MAX_CONCURRENT_BACKGROUND`` (20) and
``TIMBAL_MAX_BACKGROUND_DEPTH`` (unlimited); Agents can override per session.

If the parent already learned via the LLM ``get_background_task`` /
``list_background_tasks`` tools (``ack_completion=True``), the notice is
dropped so the next turn does not re-announce a completion the model handled.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

_SUMMARY_TEXT_CHARS = 1_000
_COMPLETION_SUMMARY_CHARS = 300
_RESULT_PREVIEW_CHARS = 500
_TASK_ID_LEN = 12
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled", "timed_out", "stalled"})

# Ring-buffer defaults (mirrors JobStore). ``None`` / 0 = unlimited.
DEFAULT_BG_LOG_MAX_EVENTS = 50_000
DEFAULT_BG_LOG_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_BG_TASK_RETENTION_SECS = 300.0

# Process-local: run_id → store. Sequential turns find the bag via parent_id.
# Concurrent siblings (no parent_id) each create a new store.
_STORES_BY_RUN_ID: dict[str, BackgroundTaskStore] = {}

_DONE = object()
_GAPPED = object()
_UNSET = object()

# Yielded once by :meth:`BackgroundEventLog.subscribe` when ``after`` is behind
# ``forgotten_through`` (same gap ``read``/``peek`` report via ``gapped=True``).
BACKGROUND_LOG_GAPPED = _GAPPED

# Nesting depth of the current coroutine relative to top-level session work.
# 0 = parent agent/turn; each detached child increments by 1 for its body.
_background_depth: contextvars.ContextVar[int] = contextvars.ContextVar("timbal_background_depth", default=0)


class BackgroundLimitError(RuntimeError):
    """Raised when a spawn would exceed concurrent or depth caps."""


def get_background_depth() -> int:
    return _background_depth.get()


def set_background_depth(depth: int) -> contextvars.Token[int]:
    return _background_depth.set(depth)


def reset_background_depth(token: contextvars.Token[int]) -> None:
    _background_depth.reset(token)


def _default_max_concurrent() -> int | None:
    """Default in-flight cap. ``TIMBAL_MAX_CONCURRENT_BACKGROUND`` overrides.

    Unset → 20. ``0`` / ``none`` / ``unlimited`` → no cap.
    """
    raw = os.environ.get("TIMBAL_MAX_CONCURRENT_BACKGROUND")
    if raw is None:
        return 20
    text = raw.strip().lower()
    if text in ("", "0", "none", "unlimited"):
        return None
    try:
        value = int(text)
    except ValueError:
        return 20
    return None if value <= 0 else value


def _default_max_depth() -> int | None:
    """Default spawn-depth cap. ``TIMBAL_MAX_BACKGROUND_DEPTH`` overrides.

    Unset / ``none`` / ``unlimited`` → no cap. ``1`` = only top-level may spawn.
    """
    raw = os.environ.get("TIMBAL_MAX_BACKGROUND_DEPTH")
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in ("", "none", "unlimited"):
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value < 0 else value


def _event_nbytes(event: Any) -> int:
    dump_json = getattr(event, "model_dump_json", None)
    if callable(dump_json):
        return len(dump_json())
    if isinstance(event, str):
        return len(event)
    if isinstance(event, (bytes, bytearray)):
        return len(event)
    return len(repr(event))


def _bg_log_limit(raw: str | None, default: int | None) -> int | None:
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in ("", "0", "none", "unlimited"):
        return None
    try:
        value = int(text)
    except ValueError:
        return default
    return None if value <= 0 else value


def _default_max_log_events() -> int | None:
    return _bg_log_limit(os.environ.get("TIMBAL_BG_LOG_MAX_EVENTS"), DEFAULT_BG_LOG_MAX_EVENTS)


def _default_max_log_bytes() -> int | None:
    return _bg_log_limit(os.environ.get("TIMBAL_BG_LOG_MAX_BYTES"), DEFAULT_BG_LOG_MAX_BYTES)


def _default_task_retention_secs() -> float:
    raw = os.environ.get("TIMBAL_BG_TASK_RETENTION_SECS")
    if raw is None:
        return DEFAULT_BG_TASK_RETENTION_SECS
    text = raw.strip().lower()
    if text in ("", "0", "none", "unlimited"):
        return 0.0
    try:
        return max(0.0, float(text))
    except ValueError:
        return DEFAULT_BG_TASK_RETENTION_SECS


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
        from ..types.events.delta import Custom, DeltaEvent, Text, TextDelta, ToolUse

        _EVENT_CLASSES.update(
            delta_event=DeltaEvent,
            output_event=OutputEvent,
            text=Text,
            text_delta=TextDelta,
            tool_use=ToolUse,
            custom=Custom,
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


def _coerce_pct(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = float(value)
        if 0 <= n <= 1:
            n *= 100
        if 0 <= n <= 100:
            return int(round(n))
    return None


def _progress_fields_from_mapping(data: dict[str, Any]) -> tuple[str | None, int | None]:
    phase: str | None = None
    for key in ("phase", "stage", "status", "step"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            phase = value.strip()
            break
    pct: int | None = None
    for key in ("pct", "progress", "percent", "percentage"):
        if key in data:
            pct = _coerce_pct(data[key])
            if pct is not None:
                break
    return phase, pct


def _progress_from_payload(payload: Any) -> tuple[str | None, int | None]:
    if isinstance(payload, dict):
        return _progress_fields_from_mapping(payload)
    return None, None


def _resolve_inflight_tool(
    open_tools: dict[str, str],
    *,
    tool_call_id: str | None,
    path: str | None,
) -> None:
    if tool_call_id and tool_call_id in open_tools:
        del open_tools[tool_call_id]
        return
    if not path or "." not in path:
        return
    segment = path.rsplit(".", 1)[-1]
    for call_id, name in list(open_tools.items()):
        if name == segment:
            del open_tools[call_id]
            break


def _build_summary_from_events(
    events: list[Any],
    *,
    terminal_status: str | None = None,
) -> dict[str, Any]:
    """Derive bounded briefing fields from a background child's event log."""
    classes = _event_classes()
    text_parts: list[str] = []
    open_tools: dict[str, str] = {}
    last_tool: str | None = None
    phase: str | None = None
    pct: int | None = None

    for event in events:
        chunk = _event_text(event, classes)
        if chunk:
            text_parts.append(chunk)

        if isinstance(event, classes["delta_event"]):
            item = event.item
            if isinstance(item, classes["tool_use"]):
                open_tools[item.id] = item.name
                last_tool = item.name
            elif isinstance(item, classes["custom"]):
                custom_phase, custom_pct = _progress_from_payload(item.data)
                if custom_phase is not None:
                    phase = custom_phase
                if custom_pct is not None:
                    pct = custom_pct
        elif isinstance(event, classes["output_event"]):
            meta = event.metadata if isinstance(event.metadata, dict) else {}
            custom_phase, custom_pct = _progress_from_payload(meta)
            if custom_phase is not None:
                phase = custom_phase
            if custom_pct is not None:
                pct = custom_pct
            _resolve_inflight_tool(
                open_tools,
                tool_call_id=meta.get("tool_call_id") if isinstance(meta.get("tool_call_id"), str) else None,
                path=getattr(event, "path", None),
            )

    text = "".join(text_parts)
    if len(text) > _SUMMARY_TEXT_CHARS:
        text = text[-_SUMMARY_TEXT_CHARS:]

    tools_in_flight = list(dict.fromkeys(open_tools.values()))

    if terminal_status in _TERMINAL_STATUSES:
        phase = terminal_status
        if terminal_status == "completed":
            pct = 100
    elif phase is None:
        if tools_in_flight and last_tool:
            phase = f"tool:{last_tool}"
        elif text:
            phase = "streaming"
        elif events:
            phase = "running"

    return {
        "text": text,
        "last_tool": last_tool,
        "tools_in_flight": tools_in_flight,
        "phase": phase,
        "pct": pct,
    }


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


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _preview_value(value: Any, *, limit: int = _RESULT_PREVIEW_CHARS) -> str:
    """Short, context-safe preview of a terminal result (not a full dump)."""
    if value is None:
        return ""
    collect = getattr(value, "collect_text", None)
    if callable(collect):
        try:
            text = collect() or ""
        except Exception:
            text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return _truncate(text.strip(), limit)


def format_background_completion_notice(notifications: list[dict[str, Any]]) -> Any:
    """Build the user-role message injected at the start of the next parent turn."""
    from ..types.content import TextContent
    from ..types.message import Message

    blocks: list[str] = []
    for note in notifications:
        lines = [
            f"task_id: {note['task_id']}",
            f"name: {note['name']}",
            f"status: {note['status']}",
        ]
        title = note.get("title")
        if title:
            lines.insert(2, f"title: {title}")
        if note.get("error"):
            lines.append(f"error: {note['error']}")
        elif note.get("result"):
            lines.append(f"result: {note['result']}")
        summary = note.get("summary")
        if summary:
            lines.append(f"summary: {summary}")
        blocks.append("<background_task_completed>\n" + "\n".join(lines) + "\n</background_task_completed>")

    header = (
        "Background task(s) finished since your last turn. "
        "You do not need to poll get_background_task unless you want more detail.\n\n"
    )
    return Message(role="user", content=[TextContent(text=header + "\n\n".join(blocks))])


class BackgroundEventLog:
    """Append-only ring-buffer log. ``put_nowait`` matches :class:`asyncio.Queue`.

    Cursors are **logical** indices: ``after`` is how many events the reader has
    already seen. When ``max_events`` / ``max_bytes`` is exceeded the oldest
    entries drop and :attr:`forgotten_through` advances. ``after <
    forgotten_through`` is a gap — not a silent skip to the new head.
    """

    def __init__(
        self,
        *,
        max_events: int | None = DEFAULT_BG_LOG_MAX_EVENTS,
        max_bytes: int | None = DEFAULT_BG_LOG_MAX_BYTES,
    ) -> None:
        self._max_events = max_events or 0
        self._max_bytes = max_bytes or 0
        self._events: list[Any] = []
        self._sizes: list[int] = []
        self._nbytes = 0
        self.forgotten_through = 0
        self._subscribers: list[asyncio.Queue] = []
        self._waiters: list[asyncio.Future[None]] = []
        self._closed = False

    @property
    def cursor_end(self) -> int:
        """Next logical index to pass as ``after`` (total events appended)."""
        return self.forgotten_through + len(self._events)

    def put_nowait(self, event: Any) -> None:
        if self._closed:
            return
        nbytes = _event_nbytes(event) if self._max_bytes > 0 else 0
        self._events.append(event)
        self._sizes.append(nbytes)
        self._nbytes += nbytes
        self._trim()
        for queue in self._subscribers:
            queue.put_nowait(event)
        self._wake_waiters()

    def _wake_waiters(self) -> None:
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()

    def _discard_waiter(self, waiter: asyncio.Future[None]) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            pass

    async def wait(self, after: int = 0, timeout: float | None = None) -> None:
        """Block until ``cursor_end > after``, the log :meth:`close`s, or timeout.

        Returns (does not raise) on timeout — callers re-read the snapshot.
        """
        if after < self.forgotten_through or self.cursor_end > after or self._closed:
            return
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiters.append(waiter)
        try:
            if timeout is None:
                await waiter
            elif timeout > 0:
                await asyncio.wait_for(waiter, timeout)
        except TimeoutError:
            pass
        finally:
            self._discard_waiter(waiter)

    def peek(self, after: int = 0) -> tuple[list[Any], bool]:
        """Return ``(events, gapped)`` from logical cursor ``after``."""
        if after < self.forgotten_through:
            return [], True
        offset = after - self.forgotten_through
        return list(self._events[offset:]), False

    def read(self, after: int = 0, limit: int | None = None) -> tuple[list[Any], int, bool]:
        """``(events, next_cursor, gapped)`` from logical ``after``."""
        if after < self.forgotten_through:
            return [], after, True
        events, _ = self.peek(after)
        if limit is not None and limit > 0:
            events = events[:limit]
        next_cursor = after + len(events)
        return events, next_cursor, False

    def qsize(self) -> int:
        return len(self._events)

    def empty(self) -> bool:
        return not self._events

    async def subscribe(self, after: int = 0) -> AsyncIterator[Any]:
        """Replay from logical ``after``, then yield live events until :meth:`close`.

        When ``after < forgotten_through``, yields :data:`BACKGROUND_LOG_GAPPED` once
        (matching ``gapped=True`` on :meth:`read`/``peek``), replays from the ring
        head, then continues with live events — never returns an empty iterator.
        """
        gapped = after < self.forgotten_through
        replay_from = self.forgotten_through if gapped else after
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        # Snapshot before any yield — the ring may trim while the consumer handles
        # BACKGROUND_LOG_GAPPED, which would make a post-yield offset negative.
        offset = replay_from - self.forgotten_through
        replay = list(self._events[offset:])
        try:
            if gapped:
                yield _GAPPED
            for event in replay:
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
        self._closed = True
        for queue in self._subscribers:
            queue.put_nowait(_DONE)
        self._wake_waiters()

    def _over_cap(self, n: int, nbytes: int) -> bool:
        return (self._max_events > 0 and n > self._max_events) or (
            self._max_bytes > 0 and nbytes > self._max_bytes
        )

    def _trim(self) -> None:
        """Drop oldest events until the ring fits. Never drops the tip."""
        drop = 0
        n = len(self._events)
        nbytes = self._nbytes
        while n - drop > 1 and self._over_cap(n - drop, nbytes):
            nbytes -= self._sizes[drop]
            drop += 1
        if drop:
            del self._events[:drop]
            del self._sizes[:drop]
            self._nbytes = nbytes
            self.forgotten_through += drop


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
        timeout: float | None = None,
        stall_timeout: float | None = None,
        log_max_events: int | None = DEFAULT_BG_LOG_MAX_EVENTS,
        log_max_bytes: int | None = DEFAULT_BG_LOG_MAX_BYTES,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.input = input
        self.task = task
        self.started_at = started_at
        self.title = title or _title_from_input(name, input)
        self.log = BackgroundEventLog(max_events=log_max_events, max_bytes=log_max_bytes)
        self.metadata: dict[str, Any] = {}
        self.on_cancel = on_cancel
        self.timeout = timeout
        self.stall_timeout = stall_timeout
        self.last_event_at = time.monotonic()
        self.finished_at: float | None = None
        self._cancel_requested = False
        self._timed_out = False
        self._stalled = False
        self._timeout_handle: asyncio.TimerHandle | None = None
        self._stall_handle: asyncio.TimerHandle | None = None
        self._handler_aclose: Callable[..., Any] | None = None

    def clear_watchdogs(self) -> None:
        self.clear_timeout()
        self.clear_stall()

    def schedule_timeout(self) -> None:
        """Arm a one-shot deadline; no-op when ``timeout`` is unset or non-positive."""
        if self.timeout is None or self.timeout <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._timeout_handle = loop.call_later(self.timeout, self._on_timeout)

    def clear_timeout(self) -> None:
        handle = self._timeout_handle
        self._timeout_handle = None
        if handle is not None:
            handle.cancel()

    def schedule_stall_watchdog(self) -> None:
        """Arm a one-shot idle deadline; reset on every :meth:`ingest`."""
        self.clear_stall()
        if self.stall_timeout is None or self.stall_timeout <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._stall_handle = loop.call_later(self.stall_timeout, self._on_stall)

    def clear_stall(self) -> None:
        handle = self._stall_handle
        self._stall_handle = None
        if handle is not None:
            handle.cancel()

    def _on_stall(self) -> None:
        """Cancel the child when no events arrived within ``stall_timeout``."""
        self._stall_handle = None
        if self.task.done():
            return
        self._stalled = True
        self._cancel_requested = True
        self.task.cancel()
        try:
            asyncio.get_running_loop().create_task(self.aclose_handler())
        except RuntimeError:
            pass
        if self.on_cancel is not None:
            try:
                self.on_cancel(self)
            except Exception:
                pass

    def _on_timeout(self) -> None:
        """Cancel the child and mark ``timed_out`` (distinct from user cancel)."""
        self._timeout_handle = None
        if self.task.done():
            return
        self._timed_out = True
        self._cancel_requested = True
        self.task.cancel()
        try:
            asyncio.get_running_loop().create_task(self.aclose_handler())
        except RuntimeError:
            pass
        if self.on_cancel is not None:
            try:
                self.on_cancel(self)
            except Exception:
                pass

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
        self.last_event_at = time.monotonic()
        self.log.put_nowait(event)
        self.schedule_stall_watchdog()

    def put_nowait(self, event: Any) -> None:
        """Queue-shaped sink used by ``_execute_handler``."""
        self.ingest(event)

    def status_code(self) -> str:
        task = self.task
        if self._timed_out:
            return "timed_out"
        if self._stalled:
            return "stalled"
        if self._cancel_requested or (task.done() and task.cancelled()):
            return "cancelled"
        if not task.done():
            return "running"
        if task.exception() is not None:
            return "error"
        return "completed"

    def summarize(self) -> dict[str, Any]:
        events, _ = self.log.peek(self.log.forgotten_through)
        status = self.status_code()
        briefing = _build_summary_from_events(events, terminal_status=status if status in _TERMINAL_STATUSES else None)
        snapshot: dict[str, Any] = {
            "status": status,
            "task_id": self.task_id,
            "name": self.name,
            "input": self.input,
            "title": self.title,
            "started_at": self.started_at,
            "summary": {
                **briefing,
                "event_count": self.log.cursor_end,
            },
            "transcript_cursor": self.log.cursor_end,
            "forgotten_through": self.log.forgotten_through,
        }
        if self.timeout is not None:
            snapshot["timeout"] = self.timeout
        if self.stall_timeout is not None:
            snapshot["stall_timeout"] = self.stall_timeout
        if status == "running" and self.stall_timeout is not None:
            snapshot["summary"]["seconds_since_event"] = round(time.monotonic() - self.last_event_at, 3)
        snapshot.update(self.metadata)
        if self._timed_out:
            timeout = self.timeout
            snapshot["error"] = f"Timed out after {timeout:g}s" if timeout else "Timed out"
        elif self._stalled:
            stall = self.stall_timeout
            snapshot["error"] = f"No events for {stall:g}s" if stall else "Stalled"
        elif self.task.done() and not self.task.cancelled():
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
    ``RunContext``s in the same parent_id chain; not across concurrent runs.

    Terminal children enqueue a one-shot completion notice on
    :attr:`_completion_inbox`. The parent agent drains that inbox at the
    **start** of the next turn (not mid-turn) so the LLM learns without
    polling ``get_background_task``. LLM peeks with ``ack_completion=True``
    suppress the pending notice for that child.
    """

    def __init__(
        self,
        *,
        max_concurrent: int | None | object = _UNSET,
        max_depth: int | None | object = _UNSET,
        max_log_events: int | None | object = _UNSET,
        max_log_bytes: int | None | object = _UNSET,
        task_retention_secs: float | object = _UNSET,
    ) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        # Ordered, one-shot queue of lean completion payloads for the parent LLM.
        self._completion_inbox: list[dict[str, Any]] = []
        # task_ids already enqueued, drained, or acked via peek — never re-inject.
        self._completion_notified: set[str] = set()
        self.max_concurrent = (
            _default_max_concurrent() if max_concurrent is _UNSET else max_concurrent  # type: ignore[assignment]
        )
        self.max_depth = _default_max_depth() if max_depth is _UNSET else max_depth  # type: ignore[assignment]
        self.max_log_events = (
            _default_max_log_events() if max_log_events is _UNSET else max_log_events  # type: ignore[assignment]
        )
        self.max_log_bytes = (
            _default_max_log_bytes() if max_log_bytes is _UNSET else max_log_bytes  # type: ignore[assignment]
        )
        self.task_retention_secs = (
            _default_task_retention_secs() if task_retention_secs is _UNSET else float(task_retention_secs)  # type: ignore[arg-type]
        )
        # Spawns that passed check_can_spawn but are not yet in ``_tasks``
        # (between create_task and add). Counts toward the concurrent cap.
        self._pending_spawns = 0

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def running_count(self) -> int:
        """In-flight children + spawns reserved but not yet registered."""
        live = sum(1 for record in self._tasks.values() if not record.task.done())
        return live + self._pending_spawns

    def check_can_spawn(self, depth: int | None = None) -> None:
        """Raise :class:`BackgroundLimitError` if a new spawn is not allowed."""
        if depth is None:
            depth = get_background_depth()
        if self.max_depth is not None and depth >= self.max_depth:
            raise BackgroundLimitError(
                f"Background spawn depth limit reached "
                f"(depth={depth}, max_background_depth={self.max_depth}). "
                "Nested background spawns are not allowed at this depth."
            )
        if self.max_concurrent is not None:
            running = self.running_count()
            if running >= self.max_concurrent:
                raise BackgroundLimitError(
                    f"Concurrent background limit reached "
                    f"({running}/{self.max_concurrent}). "
                    "Wait for a task to finish or cancel one before spawning another."
                )

    def begin_spawn(self, depth: int | None = None) -> None:
        """Atomically reserve a concurrent slot (check + pending++)."""
        self.check_can_spawn(depth)
        self._pending_spawns += 1

    def abort_spawn(self) -> None:
        """Release a reservation if registration fails after :meth:`begin_spawn`."""
        if self._pending_spawns > 0:
            self._pending_spawns -= 1

    def add(self, record: BackgroundTask) -> None:
        if self._pending_spawns > 0:
            self._pending_spawns -= 1
        self._tasks[record.task_id] = record
        # Enqueue when the asyncio.Task reaches a terminal state. Done callbacks
        # run on the loop thread; keep this sync and never raise.
        task = record.task
        record.schedule_timeout()
        record.schedule_stall_watchdog()
        if task.done():
            record.clear_watchdogs()
            if record.finished_at is None:
                record.finished_at = time.monotonic()
            self._enqueue_completion(record)
        else:

            def _on_done(_t: asyncio.Task) -> None:
                record.clear_watchdogs()
                if record.finished_at is None:
                    record.finished_at = time.monotonic()
                self._enqueue_completion(record)

            task.add_done_callback(_on_done)

    def reap_finished(self, now: float | None = None) -> None:
        """Drop terminal tasks whose retention window has lapsed."""
        secs = self.task_retention_secs
        if secs <= 0:
            return
        now = time.monotonic() if now is None else now
        expired = [
            task_id
            for task_id, record in self._tasks.items()
            if record.task.done()
            and record.finished_at is not None
            and now - record.finished_at >= secs
        ]
        for task_id in expired:
            del self._tasks[task_id]

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list(self, *, ack_completion: bool = False) -> list[dict[str, Any]]:
        listings = [record.listing() for record in self._tasks.values()]
        if ack_completion:
            for item in listings:
                if item.get("status") in _TERMINAL_STATUSES:
                    self.ack_completion(item["task_id"])
        return listings

    def snapshot(self, task_id: str, *, ack_completion: bool = False) -> dict[str, Any]:
        record = self._tasks.get(task_id)
        if record is None:
            return {"status": "not_found", "task_id": task_id}
        snap = record.summarize()
        if ack_completion and snap.get("status") in _TERMINAL_STATUSES:
            self.ack_completion(task_id)
        return snap

    def transcript(self, task_id: str, after: int = 0) -> dict[str, Any]:
        record = self._tasks.get(task_id)
        if record is None:
            return {
                "status": "not_found",
                "task_id": task_id,
                "events": [],
                "cursor": after,
                "gapped": False,
                "forgotten_through": 0,
            }
        events, cursor, gapped = record.log.read(after)
        return {
            "status": record.status_code(),
            "task_id": task_id,
            "events": [_dump_event(event) for event in events],
            "cursor": cursor,
            "gapped": gapped,
            "forgotten_through": record.log.forgotten_through,
        }

    def cancel(self, task_id: str) -> dict[str, Any]:
        record = self._tasks.get(task_id)
        if record is None:
            return {"status": "not_found", "task_id": task_id}
        record.clear_watchdogs()
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

    def ack_completion(self, task_id: str) -> None:
        """Mark ``task_id`` as already delivered to the parent LLM.

        Drops any pending inbox entry and blocks a later done-callback from
        re-enqueuing. Used when the model polls a terminal child mid-turn.
        """
        self._completion_notified.add(task_id)
        if self._completion_inbox:
            self._completion_inbox = [n for n in self._completion_inbox if n["task_id"] != task_id]

    def _enqueue_completion(self, record: BackgroundTask) -> None:
        """Push a lean notice if this child just became terminal (once)."""
        if record.task_id in self._completion_notified:
            return
        # status_code() can report cancelled as soon as cancel is requested,
        # before the asyncio.Task has actually finished — wait for done.
        if not record.task.done():
            return
        status = record.status_code()
        if status not in _TERMINAL_STATUSES:
            return
        self._completion_notified.add(record.task_id)
        snap = record.summarize()
        summary_text = ""
        summary = snap.get("summary")
        if isinstance(summary, dict):
            summary_text = _truncate(str(summary.get("text") or ""), _COMPLETION_SUMMARY_CHARS)
        note: dict[str, Any] = {
            "task_id": snap["task_id"],
            "name": snap["name"],
            "title": snap.get("title"),
            "status": status,
            "summary": summary_text,
        }
        if status in ("error", "timed_out", "stalled"):
            note["error"] = str(snap.get("error") or ("Timed out" if status == "timed_out" else "Stalled"))
        elif status == "completed" and "result" in snap:
            note["result"] = _preview_value(snap["result"])
        self._completion_inbox.append(note)

    def pending_completions(self) -> list[dict[str, Any]]:
        """Peek the completion inbox without draining (tests / UIs)."""
        return list(self._completion_inbox)

    def drain_completions(self) -> list[dict[str, Any]]:
        """Take all pending completion notices. Empty after a successful drain."""
        notices = self._completion_inbox
        self._completion_inbox = []
        return notices


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


def apply_background_limits(
    run_context: Any,
    *,
    max_concurrent: int | None | object = _UNSET,
    max_depth: int | None | object = _UNSET,
    max_log_events: int | None | object = _UNSET,
    max_log_bytes: int | None | object = _UNSET,
    task_retention_secs: float | object = _UNSET,
) -> None:
    """Stash session caps on the RunContext and update an existing store."""
    if max_concurrent is not _UNSET:
        run_context._bg_max_concurrent = max_concurrent
    if max_depth is not _UNSET:
        run_context._bg_max_depth = max_depth
    if max_log_events is not _UNSET:
        run_context._bg_max_log_events = max_log_events
    if max_log_bytes is not _UNSET:
        run_context._bg_max_log_bytes = max_log_bytes
    if task_retention_secs is not _UNSET:
        run_context._bg_task_retention_secs = task_retention_secs
    store = getattr(run_context, "_bg_store", None)
    if store is None:
        return
    if max_concurrent is not _UNSET:
        store.max_concurrent = max_concurrent  # type: ignore[assignment]
    if max_depth is not _UNSET:
        store.max_depth = max_depth  # type: ignore[assignment]
    if max_log_events is not _UNSET:
        store.max_log_events = max_log_events  # type: ignore[assignment]
    if max_log_bytes is not _UNSET:
        store.max_log_bytes = max_log_bytes  # type: ignore[assignment]
    if task_retention_secs is not _UNSET:
        store.task_retention_secs = float(task_retention_secs)  # type: ignore[arg-type]
    store.reap_finished()


def ensure_background_store(run_context: Any) -> BackgroundTaskStore:
    """Get (creating if needed) the session bag, and register it for chaining.

    Called from the spawn path only — this is what puts a run in the
    process-local map.
    """
    store = run_context._bg_store
    if store is None:
        max_concurrent = (
            run_context._bg_max_concurrent
            if hasattr(run_context, "_bg_max_concurrent")
            else _UNSET
        )
        max_depth = run_context._bg_max_depth if hasattr(run_context, "_bg_max_depth") else _UNSET
        max_log_events = (
            run_context._bg_max_log_events if hasattr(run_context, "_bg_max_log_events") else _UNSET
        )
        max_log_bytes = (
            run_context._bg_max_log_bytes if hasattr(run_context, "_bg_max_log_bytes") else _UNSET
        )
        task_retention_secs = (
            run_context._bg_task_retention_secs if hasattr(run_context, "_bg_task_retention_secs") else _UNSET
        )
        store = BackgroundTaskStore(
            max_concurrent=max_concurrent,
            max_depth=max_depth,
            max_log_events=max_log_events,
            max_log_bytes=max_log_bytes,
            task_retention_secs=task_retention_secs,
        )
        run_context._bg_store = store
    store.reap_finished()
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


def get_background_task(task_id: str, *, ack_completion: bool = False) -> dict[str, Any]:
    """Peek a summary of a background task. Does not drain the event log.

    Use this to answer questions about an in-flight or finished background
    tool. You get status, structured progress (phase, pct, last_tool,
    tools_in_flight), a short text summary, and any child ids (e.g.
    ``cursor_agent_id``, ``run_id``). You must have a ``task_id`` from a
    previous background-tool result or from ``list_background_tasks``.
    Raw events: ``read_background_transcript``.

    When ``ack_completion`` is true and the child is terminal, any pending
    completion notice for this task is dropped (LLM tool path).
    """
    store = current_background_store()
    if store is None:
        return {"status": "not_found", "task_id": task_id}
    return store.snapshot(task_id, ack_completion=ack_completion)


def list_background_tasks(*, ack_completion: bool = False) -> list[dict[str, Any]]:
    """List background tools for this session (not other concurrent runs).

    Returns ``[{task_id, name, status, started_at, title}]``. Use a ``task_id``
    with ``get_background_task`` to answer questions about a child.

    When ``ack_completion`` is true, terminal children are treated as already
    delivered (LLM tool path) so the next turn does not re-inject notices.
    """
    store = current_background_store()
    if store is None:
        return []
    return store.list(ack_completion=ack_completion)


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
    """Raw events for a background task, from logical cursor ``after``.

    Does not drain. Returns ``gapped=True`` when ``after`` is behind
    ``forgotten_through`` (events were dropped from the ring). ``cursor`` is
    the next logical index to pass.
    """
    store = current_background_store()
    if store is None:
        return {
            "status": "not_found",
            "task_id": task_id,
            "events": [],
            "cursor": after,
            "gapped": False,
            "forgotten_through": 0,
        }
    return store.transcript(task_id, after=after)


async def wait_for_background(
    task_id: str,
    *,
    timeout: float | None = None,
    after: int | None = None,
) -> dict[str, Any]:
    """Block until a background child is ready to report, then return its snapshot.

    Without ``after``, waits for a **terminal** status (``completed``,
    ``error``, ``cancelled``, ``timed_out``). With ``after``, long-polls until
    the event log advances past that logical cursor, the child finishes, or
    ``timeout`` elapses — whichever comes first.

    Returns the same dict shape as :func:`get_background_task` (does not ack
    completion notices). On timeout while still running, returns the current
    snapshot so callers can poll again.
    """
    store = current_background_store()
    if store is None:
        return {"status": "not_found", "task_id": task_id}

    record = store.get(task_id)
    if record is None:
        return {"status": "not_found", "task_id": task_id}

    def _is_ready(rec: BackgroundTask) -> bool:
        if after is not None and rec.log.cursor_end > after:
            return True
        return rec.task.done() and rec.status_code() in _TERMINAL_STATUSES

    async def _yield_for_callbacks(rec: BackgroundTask) -> None:
        if rec.task.done():
            await asyncio.sleep(0)

    if _is_ready(record):
        await _yield_for_callbacks(record)
        fresh = store.get(task_id)
        return fresh.summarize() if fresh is not None else {"status": "not_found", "task_id": task_id}

    if after is None:
        wait_set = {record.task}
        if timeout is None:
            await asyncio.wait(wait_set)
        else:
            await asyncio.wait(wait_set, timeout=timeout)
        await _yield_for_callbacks(record)
        fresh = store.get(task_id)
        return fresh.summarize() if fresh is not None else {"status": "not_found", "task_id": task_id}

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        rec = store.get(task_id)
        if rec is None:
            return {"status": "not_found", "task_id": task_id}
        if _is_ready(rec):
            await _yield_for_callbacks(rec)
            return rec.summarize()

        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return rec.summarize()

        wait_timeout = remaining if remaining is not None else None
        await rec.log.wait(after, timeout=wait_timeout)


def register_background_task(
    *,
    name: str,
    input: dict[str, Any],
    task: asyncio.Task,
    on_cancel: Callable[..., Any] | None = None,
    started_at: int | None = None,
    timeout: float | None = None,
    stall_timeout: float | None = None,
) -> BackgroundTask:
    """Register a running child on the current session bag.

    ``timeout`` is seconds until the child is cancelled as ``timed_out``.
    ``stall_timeout`` is seconds without log events until ``stalled``.
    Raises :class:`BackgroundLimitError` if concurrent/depth caps would be exceeded.
    """
    from . import get_run_context

    run_context = get_run_context()
    if run_context is None:
        raise RuntimeError("Cannot register a background task without a RunContext.")
    store = ensure_background_store(run_context)
    # Slot already reserved by Runnable._spawn_background_task via begin_spawn;
    # direct callers must begin_spawn (or accept check_can_spawn here without reserve).
    if store._pending_spawns == 0:
        store.check_can_spawn()
    record = BackgroundTask(
        _new_task_id(),
        name=name,
        input=input,
        task=task,
        started_at=started_at if started_at is not None else int(time.time() * 1000),
        on_cancel=on_cancel,
        timeout=timeout,
        stall_timeout=stall_timeout,
        log_max_events=store.max_log_events,
        log_max_bytes=store.max_log_bytes,
    )
    store.add(record)
    return record


def clear_background_stores() -> None:
    """Drop the process-local registry. Tests only."""
    _STORES_BY_RUN_ID.clear()


GET_BACKGROUND_TASK_DESCRIPTION = (
    "Get a summary of an in-flight or finished background tool so you can "
    "answer questions about it (status, phase, pct, last_tool, tools_in_flight, "
    "last assistant text, error). You must have a task_id from a previous "
    "background-tool result or from list_background_tasks. This does not return "
    "the full event log — use it to brief the user, not to dump a long build."
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

READ_BACKGROUND_TRANSCRIPT_DESCRIPTION = (
    "Read raw events from a background task's append-only log. Use when "
    "get_background_task's summary is not enough — e.g. to inspect streaming "
    "output or tool calls mid-build. Pass task_id and optional after (logical "
    "cursor from a prior read or from get_background_task's transcript_cursor). "
    "Does not drain — safe to re-read. Page with after on long tasks; when "
    "gapped is true, events before forgotten_through were dropped from the ring."
)
