"""Manages the run context for Timbal using contextvars.

This module provides a concurrency-safe way to access the current run's context.

Public API:
    - RunContext: The data model for the run context.
    - get_run_context(): Retrieves the current run context.
    - get_or_create_run_context(): Retrieves the current run context, creating a new one if necessary.
    - ref(): Creates runtime-resolved references to shared data.

The context is typically managed automatically by the framework. Advanced users
may need to set the context manually when creating custom execution flows.
See the function docstrings for usage details and warnings.
"""
# ruff: noqa: F401
from contextvars import ContextVar

from .context import RunContext
from .ref import ref

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


def set_call_id(call_id: str) -> None:
    """Sets the current call ID.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    call ID can lead to unexpected behavior if not handled correctly.
    """
    _call_id.set(call_id)


def set_parent_call_id(parent_call_id: str) -> None:
    """Sets the current parent call ID.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    parent call ID can lead to unexpected behavior if not handled correctly.
    """
    _parent_call_id.set(parent_call_id)
