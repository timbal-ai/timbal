"""Manages the run context for Timbal using contextvars.

This module provides a concurrency-safe way to access the current run's context.

Public API:
    - RunContext: The data model for the run context.
    - get_run_context(): Retrieves the current run context.
    - get_or_create_run_context(): Retrieves the current run context, creating a new one if necessary.

The context is typically managed automatically by the framework. Advanced users
may need to set the context manually when creating custom execution flows.
See the function docstrings for usage details and warnings.
"""

import hashlib
import json
import re
from contextvars import ContextVar
from typing import Any

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


def set_run_context(context: RunContext) -> None:
    """Sets the run context for the current async task or thread.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    context can lead to unexpected behavior if not handled correctly.
    """
    _run_context_var.set(context)


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


def _suspension_id_for(path: str, payload: Any) -> str:
    """Compute a stable suspension_id for a ``suspend()`` call.

    Hashes ``(path, payload)`` so the same payload on the same runnable resumes
    deterministically across the handler re-execution. Two identical payloads on
    the same path share one id (mirrors approval's ``(path, input)`` semantics);
    add a discriminator field to the payload if you need distinct ids.
    """
    keyed = {"path": path, "payload": payload}
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
    suspension_id = _suspension_id_for(span.path, payload)
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
