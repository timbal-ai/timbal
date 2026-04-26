import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from uuid_extensions import uuid7

from ..errors import SpanNotFound
from .config import PlatformConfig
from .tracing.providers import (
    TRACING_UNSET,
    InMemoryTracingProvider,
    PlatformTracingProvider,
    TracingProvider,
    _TracingProviderUnset,
)
from .tracing.span import Span
from .tracing.trace import Trace


def _get_logger():
    import structlog

    return structlog.get_logger("timbal.state.context")


class _NoDefault:
    """INTERNAL: Sentinel to distinguish 'no default provided' from 'default is None'."""

    def __repr__(self) -> str:
        return "<no default>"


_NO_DEFAULT = _NoDefault()


class RunContext(BaseModel):
    """Runtime execution context shared across all components in a run.

    The RunContext provides a centralized location for:
    - Execution tracing and monitoring
    - Data sharing between steps and components
    - Usage tracking and statistics
    - Parent-child run relationships

    This context is automatically created and managed by the framework and is accessible
    through get_run_context() in runtime callables like default param callables and hooks.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    id: str = Field(
        default_factory=lambda: uuid7(as_type="str").replace("-", ""),  # type: ignore
        description="Unique identifier for the run.",
    )
    parent_id: str | None = Field(
        default=None,
        description="Whether this run is a direct child of another run.",
    )

    platform_config: PlatformConfig | None = Field(
        default=None,
        description="Platform configuration for the run.",
    )

    tracing_provider: Any = Field(
        default=TRACING_UNSET,
        description=(
            "Tracing provider to use for this run. "
            "TRACING_UNSET (default) → auto-detect from env/config. "
            "None → disable tracing entirely. "
            "A TracingProvider subclass → use that provider."
        ),
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_platform_config(cls, data: Any) -> Any:
        """Normalize platform_config from legacy 'timbal_platform_config' key."""
        if isinstance(data, dict):
            # Only use legacy key if platform_config is not present
            if "timbal_platform_config" in data and "platform_config" not in data:
                data["platform_config"] = data.pop("timbal_platform_config")
            elif "timbal_platform_config" in data:
                del data["timbal_platform_config"]
        return data

    _base_path: Path | None = PrivateAttr(default=None)
    _trace: Trace = PrivateAttr()
    _tracing_provider: type[TracingProvider] = PrivateAttr()
    _session_data: dict[str, Any] | None = PrivateAttr(default=None)
    _approval_decisions: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Active approval resolutions keyed by approval_id (values: ApprovalResolution).
    Typed as ``Any`` to avoid an import cycle with ``..types.approval`` — the
    invariant is enforced at write time by ``_normalize_approval_decisions``."""
    _used_approval_ids: set[str] = PrivateAttr(default_factory=set)
    """Approval IDs that matched a gate during this run. Used to warn about
    unrecognized decisions (typos, stale IDs) at run completion."""

    def pending_approvals(self) -> list[dict[str, Any]]:
        """Return metadata for every span currently waiting on approval.

        Useful when an OutputEvent's status is cancelled/approval_required and
        the caller wants to enumerate every approval that needs a decision
        before retrying the run with ``approval_decisions={...}``. Includes
        ``expired``/``expired_at`` when the previous decision was rejected for
        TTL reasons so a UI can flag stale approvals to the operator.

        Tolerates both ``RunStatus`` instances and dicts, since traces loaded
        from JSONL/SQLite providers carry status as a dict.
        """
        pending: list[dict[str, Any]] = []
        for span in self._trace.values():
            status = span.status
            code = status.get("code") if isinstance(status, dict) else getattr(status, "code", None)
            reason = status.get("reason") if isinstance(status, dict) else getattr(status, "reason", None)
            if code == "cancelled" and reason == "approval_required":
                approval = (span.metadata or {}).get("approval")
                if approval and approval.get("id"):
                    entry = {
                        "approval_id": approval["id"],
                        "path": span.path,
                        "call_id": span.call_id,
                        "prompt": approval.get("prompt"),
                        "description": approval.get("description"),
                        "metadata": approval.get("metadata", {}),
                        "input": approval.get("input"),
                    }
                    if approval.get("expired"):
                        entry["expired"] = True
                        entry["expired_at"] = approval.get("expired_at")
                    pending.append(entry)
        return pending

    def model_post_init(self, __context: Any) -> None:
        """Initialize the RunContext after Pydantic model creation.

        Sets up the tracing provider based on available configuration.
        Defaults to in-memory tracing if no custom provider is configured.

        If no platform_config is provided, attempts to resolve it from
        environment variables and ~/.timbal/ config files.
        """
        from .config_loader import resolve_platform_config

        self._trace = Trace()

        # Explicit provider set on the runnable — skip auto-detection entirely.
        # None means tracing is disabled; a class means use that provider.
        if not isinstance(self.tracing_provider, _TracingProviderUnset):
            if self.tracing_provider is not None and (
                not isinstance(self.tracing_provider, type) or not issubclass(self.tracing_provider, TracingProvider)
            ):
                raise TypeError(
                    f"tracing_provider must be a TracingProvider subclass, None, or TRACING_UNSET — "
                    f"got {self.tracing_provider!r}. "
                    f"Pass the class itself (e.g. MyProvider), not an instance. "
                    f"Use MyProvider.configured(...) to set provider-specific options."
                )
            self._tracing_provider = self.tracing_provider
            return

        self.platform_config = resolve_platform_config(self.platform_config)

        if self.platform_config:
            use_platform_traces = self.platform_config.sync_traces_enabled is not False
            if use_platform_traces and self.platform_config.subject and self.platform_config.subject.app_id:
                _get_logger().info(
                    f"Platform configuration found (subject: {self.platform_config.subject}). "
                    "Using platform tracing provider.",
                    event_name="tracing_setup",
                    run_id=self.id,
                )
                self._tracing_provider = PlatformTracingProvider
                return
            if self.platform_config.sync_traces_enabled is False:
                _get_logger().info(
                    "Sync traces disabled (sync_traces_enabled=False). Using in-memory tracing provider.",
                    event_name="tracing_setup",
                    run_id=self.id,
                )
            else:
                _get_logger().warning(
                    "Platform configuration found but no valid subject. "
                    "Please set TIMBAL_ORG_ID and TIMBAL_APP_ID environment variables to enable platform tracing.",
                    event_name="tracing_setup",
                    run_id=self.id,
                )
        _get_logger().info(
            "Using in-memory tracing provider.",
            event_name="tracing_setup",
            run_id=self.id,
        )
        self._tracing_provider = InMemoryTracingProvider

    async def _get_parent_trace(self) -> Trace | None:
        """Load the trace data for the parent run.

        INTERNAL METHOD: This method is intended for internal framework use.
        Use with caution as it involves async I/O operations and direct
        interaction with the tracing provider.

        Returns:
            The parent run's tracing data, or None if this is a root run.
        """
        if self.parent_id and self._tracing_provider is not None:
            return await self._tracing_provider.get(self)
        return None

    async def _save_trace(self) -> None:
        """Save the trace data for the run.

        INTERNAL METHOD: This method is intended for internal framework use.
        It persists the current run's trace data using the configured
        tracing provider. Manual calls to this method may interfere with
        the framework's automatic tracing lifecycle.
        """
        # Sync session data to root span before saving
        root = self.root_span()
        if root is not None and self._session_data is not None:
            from ..utils import dump

            root.session = self._session_data
            root._session_dump = await dump(self._session_data)
        if self._tracing_provider is not None:
            await self._tracing_provider.put(self)

    async def get_session(self) -> dict[str, Any]:
        """Get session data that persists across runs."""
        if self._session_data is None:
            self._session_data = {}
            if self.parent_id and self._tracing_provider is not None:
                trace = await self._tracing_provider.get(self)
                if trace is None or trace._root_call_id is None:
                    _get_logger().error(
                        "Parent trace not found. Continuing without session data...",
                        parent_id=self.parent_id,
                        run_id=self.id,
                    )
                    return self._session_data
                root_span = trace.get(trace._root_call_id)
                assert root_span is not None, "Root span not found"
                if root_span.session is not None:
                    self._session_data.update(root_span.session)
        return self._session_data

    def root_span(self) -> Span | None:
        """Get the root span of the trace (the first span with no parent)."""
        if self._trace._root_call_id is None:
            return None
        return self._trace.get(self._trace._root_call_id)

    def parent_of(self, span: Span) -> Span | None:
        """Get the parent of a span. If the span has no parent, it's the root."""
        if span.parent_call_id is None:
            return None
        return self._trace.get(span.parent_call_id)

    def current_span(self) -> Span:
        """Get the span for the current call."""
        from . import get_call_id

        call_id = get_call_id()
        span = self._trace.get(call_id)
        if not span:
            raise RuntimeError(f"Could not resolve current span for call ID {call_id}")
        return span

    def parent_span(self) -> Span:
        """Get the span for the parent call."""
        from . import get_parent_call_id

        parent_call_id = get_parent_call_id()
        parent_span = self._trace.get(parent_call_id)
        if not parent_span:
            raise RuntimeError(f"Could not resolve parent span for call ID {parent_call_id}")
        return parent_span

    def step_span(self, name: str, default: Any = _NO_DEFAULT) -> Span | Any:
        """Get the span for a neighbor step by name.

        Uses get_parent_call_id() to find sibling spans. This allows workflows to
        temporarily set parent_call_id before evaluating lambdas, enabling step_span
        to find the correct siblings without requiring a span for the current call.

        Args:
            name: The name of the step to find.
            default: Value to return if span not found. If not provided, raises SpanNotFound.

        Returns:
            The span for the requested step, or the default value if provided and not found.

        Raises:
            SpanNotFound: If the step's span doesn't exist and no default is provided.
        """
        from . import get_parent_call_id

        parent_call_id = get_parent_call_id()
        for span in self._trace.values():
            if span.parent_call_id == parent_call_id and span.path.endswith("." + name):
                return span

        if isinstance(default, _NoDefault):
            raise SpanNotFound(name)
        return default

    def update_usage(self, key: str, value: int) -> None:
        """Update usage statistics for the current call and all parent calls.

        This method propagates usage statistics up the call stack, ensuring
        that parent components can track cumulative usage from their children.
        Commonly used for tracking token usage, API calls, or other metrics.

        Note: This method is safe under asyncio concurrency because it contains
        no await points — the entire read-modify-write is atomic with respect
        to the event loop. Do not add await points inside this method.

        Args:
            key: The usage metric key (e.g., 'tokens', 'api_calls')
            value: The value to add to the current usage for this key
        """
        from . import get_call_id

        call_id = get_call_id() or self._trace._root_call_id
        # Update usage for all parents in the call stack
        while call_id:
            assert call_id in self._trace, f"RunContext.update_usage: Call ID {call_id} not found in trace."
            span = self._trace[call_id]
            span.usage[key] = span.usage.get(key, 0) + value
            call_id = span.parent_call_id

    def resolve_cwd(self, path: str | None = None) -> Path:
        """Get the current working directory or resolve a path relative to it.

        This method handles:
        - Returning the base_path (CWD) when no path is provided
        - Environment variable expansion (e.g., $HOME)
        - User home directory expansion (e.g., ~)
        - Relative path resolution (relative to base_path/CWD if set)
        - Security validation (ensures path is within base_path/CWD if set)

        Args:
            path: Optional file path to resolve (can be relative or absolute).
                  If None, returns the current working directory (base_path).

        Returns:
            Resolved absolute Path object

        Raises:
            ValueError: If base_path is set and the resolved path is outside it
        """
        # If no path provided, return the CWD (base_path or current directory)
        if path is None:
            if self._base_path is not None:
                return self._base_path.resolve()
            else:
                return Path.cwd()

        # Expand environment variables and user home directory
        expanded_path = Path(os.path.expandvars(os.path.expanduser(path)))

        # Only enforce base_path restrictions if explicitly set
        if self._base_path is not None:
            base_path = self._base_path.resolve()

            # If path is relative, resolve it relative to base_path
            if not expanded_path.is_absolute():
                resolved_path = (base_path / expanded_path).resolve()
            else:
                resolved_path = expanded_path.resolve()

            # Security check: ensure resolved path is within base_path
            try:
                resolved_path.relative_to(base_path)
            except ValueError:
                raise ValueError(
                    f"Access denied: path '{path}' resolves to '{resolved_path}' "
                    f"which is outside the allowed base path '{base_path}'"
                ) from None

            return resolved_path
        else:
            # No base_path restriction - just resolve the path normally
            return expanded_path.resolve()
