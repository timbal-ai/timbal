import os
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from uuid_extensions import uuid7

from .config import PlatformConfig, PlatformSubject
from .tracing.providers import InMemoryTracingProvider, PlatformTracingProvider, TracingProvider
from .tracing.span import Span
from .tracing.trace import Trace

logger = structlog.get_logger("timbal.state.context")


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
        default_factory=lambda: uuid7(as_type="str").replace("-", ""),
        description="Unique identifier for the run.",
    )
    parent_id: str | None = Field(
        None,
        description="Whether this run is a direct child of another run.",
    )

    # TODO We should also be able to deserialize from "platform_config"
    platform_config: PlatformConfig | None = Field(
        None,
        description="Platform configuration for the run.",
        validation_alias="timbal_platform_config",  # Legacy
    )

    _base_path: Path | None = PrivateAttr(default=None)
    _trace: Trace = PrivateAttr()
    _tracing_provider: type[TracingProvider] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the RunContext after Pydantic model creation.

        Sets up the tracing provider based on available configuration.
        Defaults to in-memory tracing if no custom provider is configured.

        If no platform_config is provided, attempts to resolve it from the environment.
        """
        # Resolve platform config from environment if not provided at initialization
        if not self.platform_config:
            host = os.getenv("TIMBAL_API_HOST")
            if host:
                token = os.getenv("TIMBAL_API_KEY") or os.getenv("TIMBAL_API_TOKEN")
                if token:
                    auth = {"type": "bearer", "token": token}
                    self.platform_config = PlatformConfig(
                        host=host,
                        auth=auth,
                    )

        # We validate this afterwards. We might want to send a platform config from the platform, and rely on the subject to be set locally with the env
        if self.platform_config and not self.platform_config.subject:
            org_id = os.getenv("TIMBAL_ORG_ID")
            app_id = os.getenv("TIMBAL_APP_ID")
            project_id = os.getenv("TIMBAL_PROJECT_ID")
            if org_id:
                if app_id:
                    version_id = os.getenv("TIMBAL_VERSION_ID")
                    self.platform_config.subject = PlatformSubject(
                        org_id=org_id,
                        app_id=app_id,
                        version_id=version_id,
                    )
                elif project_id:
                    self.platform_config.subject = PlatformSubject(
                        org_id=org_id,
                        project_id=project_id,
                    )
                else:
                    self.platform_config.subject = PlatformSubject(
                        org_id=org_id,
                    )

        self._trace = Trace()
        # TODO Enable custom tracing providers
        if self.platform_config:
            if self.platform_config.subject:
                if self.platform_config.subject.app_id or self.platform_config.subject.project_id:
                    logger.info(
                        f"Platform configuration found (subject: {self.platform_config.subject}). "
                        "Using platform tracing provider.",
                        event_name="tracing_setup",
                        run_id=self.id,
                    )
                    self._tracing_provider = PlatformTracingProvider
                    return
            logger.warning(
                "Platform configuration found but no valid subject. "
                "Please set TIMBAL_ORG_ID and TIMBAL_APP_ID or TIMBAL_PROJECT_ID environment variables to enable platform tracing.",
                event_name="tracing_setup",
                run_id=self.id,
            )
        logger.info(
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
        if self.parent_id:
            return await self._tracing_provider.get(self)
        return None

    async def _save_trace(self) -> None:
        """Save the trace data for the run.

        INTERNAL METHOD: This method is intended for internal framework use.
        It persists the current run's trace data using the configured
        tracing provider. Manual calls to this method may interfere with
        the framework's automatic tracing lifecycle.
        """
        await self._tracing_provider.put(self)

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
        span = self.current_span()
        parent_call_id = span.parent_call_id
        parent_span = self._trace.get(parent_call_id)
        if not parent_span:
            raise RuntimeError(f"Could not resolve parent span for call ID {parent_call_id}")
        return parent_span

    def step_span(self, name: str) -> Span:
        """Get the span for a neighbor step by name."""
        span = self.current_span()
        parent_call_id = span.parent_call_id
        for span in self._trace.values():
            if (
                span.parent_call_id == parent_call_id
                and
                # span.call_id != span.call_id and  # Don't match self
                span.path.endswith("." + name)
            ):
                return span
        raise RuntimeError(f"Could not resolve step span for call ID {parent_call_id} and step name {name}")

    def update_usage(self, key: str, value: int) -> None:
        """Update usage statistics for the current call and all parent calls.

        This method propagates usage statistics up the call stack, ensuring
        that parent components can track cumulative usage from their children.
        Commonly used for tracking token usage, API calls, or other metrics.

        Args:
            key: The usage metric key (e.g., 'tokens', 'api_calls')
            value: The value to add to the current usage for this key
        """
        from . import get_call_id

        call_id = get_call_id()
        # Update usage for all parents in the call stack
        while call_id:
            assert call_id in self._trace, f"RunContext.update_usage: Call ID {call_id} not found in trace."
            span = self._trace[call_id]
            current_value = span.usage.get(key, 0)
            span.usage[key] = current_value + value
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
                )

            return resolved_path
        else:
            # No base_path restriction - just resolve the path normally
            return expanded_path.resolve()
