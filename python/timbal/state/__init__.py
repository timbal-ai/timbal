"""Manages the run context for Timbal using contextvars.

This module provides a concurrency-safe way to access the current run's context.

Public API:
    - RunContext: The data model for the run context.
    - get_run_context(): Retrieves the current run context.
    - get_or_create_run_context(): Retrieves the current run context, creating a new one if necessary.
    - emit(): Fire-and-forget custom DELTA event on the current call's stream.
    - mark_salient(): mark a moment worth remembering (span ``metadata.memory.salient``).
    - get_background_task / list_background_tasks / cancel_background_task /
      read_background_transcript / wait_for_background: session-scoped background children.
    - message_background_task: a message a running background agent reads after its current step.
    - post_session_event / SessionEvent: events posted to a session from outside its runs.
    - report_progress: progress from inside a background task, as an event keyed on the task.

The context is typically managed automatically by the framework. Advanced users
may need to set the context manually when creating custom execution flows.
See the function docstrings for usage details and warnings.
"""

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from .background import (
    BACKGROUND_LOG_GAPPED as BACKGROUND_LOG_GAPPED,
)
from .background import (
    SessionEvent as SessionEvent,
)
from .background import (
    cancel_background_task as cancel_background_task,
)
from .background import (
    get_background_task as get_background_task,
)
from .background import (
    list_background_tasks as list_background_tasks,
)
from .background import (
    message_background_task as message_background_task,
)
from .background import (
    post_session_event as post_session_event,
)
from .background import (
    read_background_transcript as read_background_transcript,
)
from .background import (
    report_progress as report_progress,
)
from .background import (
    wait_for_background as wait_for_background,
)
from .context import RunContext

# INTERNAL: This variable holds the context. Do not access directly.
_run_context_var: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def get_run_context() -> RunContext | None:
    """Retrieves the current run context."""
    return _run_context_var.get()


def get_or_create_run_context() -> RunContext:
    """Retrieves the current run context, creating a new one if necessary."""
    run_context = get_run_context()
    if not run_context:
        run_context = RunContext()
        set_run_context(run_context)
    return run_context


def set_run_context(context: RunContext | None) -> None:
    """Sets the run context for the current async task or thread.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    context can lead to unexpected behavior if not handled correctly.
    """
    _run_context_var.set(context)


def emit(data: Any) -> None:
    """Fire-and-forget: broadcast a custom DELTA event on the current call's stream.

    Convenience wrapper over :meth:`RunContext.emit` that swallows the call
    when no run context is active, so handlers (and helpers deep inside them)
    stay unit-testable without a context harness. See :meth:`RunContext.emit`
    for delivery and ordering semantics.
    """
    run_context = get_run_context()
    if run_context is None:
        return
    run_context.emit(data)


MAX_SALIENT_CHARS = 500
# How many times a mark copies a span's memory that the copy itself keeps changing (see mark_salient).
_SALIENT_COPIES = 8
# Re-entrant: the copy runs code of the caller's (a mapping subclass), which may mark again on the same thread.
_SALIENT_LOCK = threading.RLock()


def _new_salient_lock() -> None:
    global _SALIENT_LOCK
    _SALIENT_LOCK = threading.RLock()  # a forked child starts with none held, whatever the parent was doing


if hasattr(os, "register_at_fork"):  # (not on Windows)
    os.register_at_fork(after_in_child=_new_salient_lock)


def mark_salient(why: str, *, span: Any = None) -> bool:
    """Mark a moment as worth remembering: appends ``why`` to ``span.metadata["memory"]["salient"]``.

    For memory consolidation (monolith#565's dreams read marked spans first); it writes nothing to a
    memory store and costs no model call. ``span`` defaults to the current span. Returns False (and
    marks nothing) outside a run, for an empty reason, or when the span's ``memory`` is not a mapping (the
    runnable's own metadata keeps something else there) or one that copying it replaces every time (code
    of a mapping subclass that writes the span's ``memory``; tried 8 times); reasons are capped at
    :data:`MAX_SALIENT_CHARS` characters. Counted in usage as ``memory:salient``.
    """
    run_context = get_run_context()
    why = why.strip() if isinstance(why, str) else ""
    if run_context is None or not why:
        return False
    if span is None:
        try:
            span = run_context.current_span()
        except RuntimeError:
            return False
    # Copied, never written in place: a span's metadata starts as a shallow copy of its runnable's own, so a
    # ``memory`` dict there may be the runnable's configuration, which every later run of it starts from. Under
    # a lock: a tool run on a thread may mark the same span (read, copy and assign would lose a mark), and the
    # count is a read, an increment and a write too.
    with _SALIENT_LOCK:
        for _ in range(_SALIENT_COPIES):
            memory = span.metadata.get("memory")
            if memory is not None and not isinstance(memory, Mapping):
                return False  # the runnable keeps something else under that key: nowhere to put the mark
            copy = dict(memory) if memory is not None else {}
            salient = copy.get("salient")
            marks = [*(salient if isinstance(salient, list | tuple) else ()), why[:MAX_SALIENT_CHARS]]
            if span.metadata.get("memory") is memory:
                break
            # The copy ran code of the caller's (a mapping or list subclass) that marked this span: what it
            # stored is the latest, so copy that. (It is a plain dict and list now, and runs nothing.)
        else:
            return False  # the copy changes the span's memory every time: nothing settles to add the mark to
        copy["salient"] = marks
        span.metadata["memory"] = copy
        run_context.update_usage("memory:salient", 1)
    return True


# INTERNAL: This variables hold the call ids. Do not access directly.
_call_id: ContextVar[str | None] = ContextVar("call_id", default=None)
_parent_call_id: ContextVar[str | None] = ContextVar("parent_call_id", default=None)


def get_call_id() -> str | None:
    """Retrieves the current call ID."""
    return _call_id.get()


def get_parent_call_id() -> str | None:
    """Retrieves the current parent call ID."""
    return _parent_call_id.get()


def set_call_id(call_id: str | None) -> None:
    """Sets the current call ID.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    call ID can lead to unexpected behavior if not handled correctly.
    """
    _call_id.set(call_id)


def set_parent_call_id(parent_call_id: str | None) -> None:
    """Sets the current parent call ID.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    parent call ID can lead to unexpected behavior if not handled correctly.
    """
    _parent_call_id.set(parent_call_id)


def _suspension_id_for(path: str, payload: Any, tool_call_id: str | None = None) -> str:
    """Compute a stable suspension_id for a ``suspend()`` call.

    Hashes ``(path, payload[, tool_call_id])`` so the same suspension resumes
    deterministically across the handler re-execution.

    When the suspend happens inside an agent tool, ``tool_call_id`` (the LLM's
    ``tool_use`` block id) is folded in. This disambiguates two parallel calls to
    the *same* tool in a single turn even when their payloads are identical — each
    tool_use block gets its own id and can be resumed with its own answer. The id
    is stable across re-execution because the assistant message (and thus the
    tool_use ids) is replayed verbatim from the parent trace on resume.

    Outside an agent (direct ``Tool`` calls, workflow steps) ``tool_call_id`` is
    ``None`` and the id reduces to ``hash(path, payload)``. There, two identical
    payloads on the same path still share one id (mirrors approval's
    ``(path, input)`` semantics); add a discriminator field to the payload if you
    need distinct ids.
    """
    keyed: dict[str, Any] = {"path": path, "payload": payload}
    if tool_call_id is not None:
        keyed["tool_call_id"] = tool_call_id
    return hashlib.sha256(json.dumps(keyed, sort_keys=True, default=str).encode()).hexdigest()[:32]


def suspend(
    payload: dict[str, Any],
    *,
    kind: str = "suspend",
    response_schema: dict[str, Any] | None = None,
) -> Any:
    """Pause the current run and wait for an externally-supplied resume value.

    Call this from inside any handler (e.g. an ``ask_user`` tool). On the first
    pass it raises :class:`~timbal.errors.Suspend`, ending the run with status
    ``cancelled`` / reason ``input_required`` and emitting an ``InteractionEvent``
    carrying ``payload`` for the frontend to render.

    Resume by calling the runnable again with the original run as ``parent_id``
    and ``resume={suspension_id: value}``. The handler re-executes from the top
    and this call returns ``value``.

    To cancel instead of answering, resume with ``Cancel(reason=...)``: the run
    terminates (status ``cancelled`` / reason ``cancelled``) and this call never
    returns.

    IMPORTANT: because the handler re-runs on resume, ``suspend()`` must come
    before any non-idempotent side-effect in the handler (same caveat as
    LangGraph's ``interrupt()``).

    Args:
        payload: JSON-serializable data describing what the caller must supply
            (e.g. ``{"question": "...", "options": [...]}``).
        kind: Discriminator the frontend uses to pick a renderer
            (e.g. ``"ask_user"``, ``"confirm"``).
        response_schema: Optional JSON Schema describing the shape the resume
            value must match. Surfaced on the ``InteractionEvent`` so the
            frontend can validate the user's input before resuming.

    Returns:
        The value supplied on resume.

    Raises:
        Suspend: on the first (un-resumed) pass.
        RunCancelled: if resumed with a ``Cancel`` value.
        RuntimeError: if called outside a run context.
    """
    from ..errors import RunCancelled, Suspend
    from ..types.approval import Cancel

    run_context = get_run_context()
    if run_context is None:
        raise RuntimeError("suspend() called outside of a run context.")
    span = run_context.current_span()
    tool_call_id = (span.metadata or {}).get("tool_call_id")
    suspension_id = _suspension_id_for(span.path, payload, tool_call_id)
    if suspension_id in run_context._resume_values:
        run_context._used_resume_ids.add(suspension_id)
        value = run_context._resume_values[suspension_id]
        if isinstance(value, Cancel):
            raise RunCancelled(value.reason or "Run cancelled by user.")
        return value
    raise Suspend(suspension_id, payload, kind, response_schema)


def _normalize_tool_request_kind(kind: str) -> str:
    """Lowercase snake_case label safe for usage keys (``[a-z0-9_]+``)."""
    t = kind.strip().lower().replace("-", "_")
    t = re.sub(r"[^a-z0-9_]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "standard"


_billing_id: ContextVar[str | None] = ContextVar("billing_id", default=None)


def get_billing_id() -> str | None:
    """Retrieves the canonical billing model id (e.g. ``openai/gpt-4o``) for the current LLM call."""
    return _billing_id.get()


def set_billing_id(billing_id: str | None) -> None:
    """Sets the billing model id for the current LLM call.

    Called by the LLM router before yielding chunks so that collectors
    can use the stable timbal model id instead of whatever the API echoes.
    """
    _billing_id.set(billing_id)


def _record_tool_requests(tool_name: str, count: int = 1, *, kind: str | None = None) -> None:
    """Framework-only: increment ``{tool_name}:requests`` (or kind-suffixed) on the current span.

    Used by :class:`~timbal.core.tool.Tool` completion for default billing keys; not a supported
    extension point—prefer letting the framework record usage or using :meth:`RunContext.update_usage`
    directly for custom metrics.

    No-op when there is no run context or ``count`` <= 0.
    """
    if count <= 0:
        return
    run_context = get_run_context()
    if not run_context:
        return
    if kind:
        suffix = f"{_normalize_tool_request_kind(kind)}_requests"
    else:
        suffix = "requests"
    run_context.update_usage(f"{tool_name}:{suffix}", count)
