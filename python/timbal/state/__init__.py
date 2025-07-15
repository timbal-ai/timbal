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
# ruff: noqa: F401
from . import savers
from .context import RunContext
from .snapshot import Snapshot
import os
from contextvars import ContextVar

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

def resolve_platform_auth(context: RunContext | None = None) -> tuple[str, dict[str, str]]:
    """Resolves platform authentication configuration."""
    if context and context.timbal_platform_config:
        host = context.timbal_platform_config.host
        auth = context.timbal_platform_config.auth
        headers = {auth.header_key: auth.header_value}
    else:
        host = os.getenv("TIMBAL_API_HOST", "api.timbal.ai")
        token = os.getenv("TIMBAL_API_KEY") or os.getenv("TIMBAL_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
    return host, headers


__all__ = [
    "RunContext",
    "Snapshot",
    "get_run_context",
    "resolve_platform_auth",
    "savers",
    "set_run_context",
]
