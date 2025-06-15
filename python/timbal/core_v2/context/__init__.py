"""Manages the run context for Timbal using contextvars.

This module provides a concurrency-safe way to access the current run's context.

Public API:
    - RunContext: The data model for the run context.
    - get_run_context(): Retrieves the current run context.
    - set_run_context(): Sets the current run context.

The context is typically managed automatically by the framework. Advanced users
may need to set the context manually when creating custom execution flows.
See the function docstrings for usage details and warnings.
"""
from contextvars import ContextVar

from .run_context import RunContext


# INTERNAL: This variable holds the context. Do not access directly.
_run_context_var: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def get_run_context() -> RunContext | None:
    """Retrieves the current run context.

    This is the safe, recommended way to access run-specific information.

    Returns:
        The current RunContext instance, or None if no context is set.
    """
    return _run_context_var.get()


def set_run_context(context: RunContext) -> None:
    """Sets the run context for the current async task or thread.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    context can lead to unexpected behavior if not handled correctly.
    """
    _run_context_var.set(context)


__all__ = [
    "RunContext",
    "get_run_context",
    "set_run_context",
]
