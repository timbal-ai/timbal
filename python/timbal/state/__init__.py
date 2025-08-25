"""
Manages the run context for Timbal using contextvars.

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
import os
from contextvars import ContextVar

from .context import PlatformConfig, RunContext

# INTERNAL: This variable holds the context. Do not access directly.
_run_context_var: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def get_run_context() -> RunContext | None:
    """
    Retrieves the current run context.

    This is the safe, recommended way to access run-specific information.

    Returns:
        The current RunContext instance, or None if no context is set.
    """
    return _run_context_var.get()


def set_run_context(context: RunContext) -> None:
    """
    Sets the run context for the current async task or thread.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    context can lead to unexpected behavior if not handled correctly.
    """
    _run_context_var.set(context)


# INTERNAL: This variables hold the call ids. Do not access directly.
_call_id: ContextVar[str | None] = ContextVar("call_id", default=None)
_parent_call_id: ContextVar[str | None] = ContextVar("parent_call_id", default=None)


def get_call_id() -> str | None:
    """
    Retrieves the current call ID.

    Returns:
        The current call ID, or None if no call ID is set.
    """
    return _call_id.get()


def get_parent_call_id() -> str | None:
    """
    Retrieves the current parent call ID.

    Returns:
        The current parent call ID, or None if no parent call ID is set.
    """
    return _parent_call_id.get()


def set_call_id(call_id: str) -> None:
    """
    Sets the current call ID.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    call ID can lead to unexpected behavior if not handled correctly.
    """
    _call_id.set(call_id)


def set_parent_call_id(parent_call_id: str) -> None:
    """
    Sets the current parent call ID.

    WARNING: This function is for advanced use cases, such as creating
    custom `Runnable` components or execution flows. Manually setting the
    parent call ID can lead to unexpected behavior if not handled correctly.
    """
    _parent_call_id.set(parent_call_id)


# TODO Remove this method in favor of custom init in the RunContext
def resolve_platform_config() -> PlatformConfig:
    """
    Resolves the active Timbal platform configuration for the current execution context.

    This function provides a unified way to obtain the PlatformConfig used for authentication
    and resource identification in Timbal's agentic and workflow runs. It first attempts to retrieve
    the platform configuration from the current RunContext (if one is set via contextvars). If no
    RunContext is active, it falls back to constructing a configuration from environment variables:
    - TIMBAL_API_HOST (required)
    - TIMBAL_API_KEY or TIMBAL_API_TOKEN (required)

    Raises:
        ValueError: If neither a RunContext nor the required environment variables are available.

    Returns:
        PlatformConfig: The resolved platform configuration, ready for use in API calls,
        resource resolution, and authentication.

    Usage:
        Prefer this function over direct environment variable access or manual context inspection
        to ensure consistent, concurrency-safe configuration resolution throughout the Timbal framework.
    """
    current_context = get_run_context()

    if current_context and current_context.platform_config:
        return current_context.platform_config
    else:
        host = os.getenv("TIMBAL_API_HOST")
        if not host:
            raise ValueError("Missing TIMBAL_API_HOST environment variable.")
        token = os.getenv("TIMBAL_API_KEY") or os.getenv("TIMBAL_API_TOKEN")
        if not token:
            raise ValueError("Missing TIMBAL_API_KEY environment variable.")
        return PlatformConfig.model_validate({
            "host": host,
            "auth": {
                "type": "bearer",
                "token": token
            }
        })
