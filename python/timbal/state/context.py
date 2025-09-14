import os
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from uuid_extensions import uuid7

from .config import PlatformConfig
from .tracing import Tracing
from .tracing.providers import InMemoryTracingProvider, PlatformTracingProvider, TracingProvider
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
        extra="ignore",
    )

    id: str = Field(
        default_factory=lambda: uuid7(as_type="str").replace("-", ""),
        description="Unique identifier for the run.",
    )
    parent_id: str | None = Field(
        None,
        description="Whether this run is a direct child of another run.",
    )

    platform_config: PlatformConfig | None = Field(
        None,
        description="Platform configuration for the run.",
        validation_alias="timbal_platform_config", # Legacy
    )

    _tracing: Tracing = PrivateAttr()
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
                    auth = {
                        "type": "bearer",
                        "token": token
                    }
                    subject = None
                    org_id = os.getenv("TIMBAL_ORG_ID")
                    app_id = os.getenv("TIMBAL_APP_ID")
                    if org_id and app_id:
                        version_id = os.getenv("TIMBAL_VERSION_ID")
                        subject = {
                            "org_id": org_id,
                            "app_id": app_id,
                            "version_id": version_id
                        }
                    self.platform_config = PlatformConfig.model_validate({
                        "host": host,
                        "auth": auth,
                        "subject": subject
                    })
        # ? Crazy idea. Merge RunContext and Tracing classes
        self._tracing = Tracing()
        # TODO Enable custom tracingproviders
        if self.platform_config:
            if not self.platform_config.subject:
                logger.warning(
                    "Platform configuration found but no subject. "
                    "Please set TIMBAL_ORG_ID and TIMBAL_APP_ID environment variables to enable platform tracing.", 
                    run_id=self.id,
                )
            else:
                logger.info(
                    f"Platform configuration found (subject: {self.platform_config.subject}). "
                    "Using platform tracing provider.", 
                    run_id=self.id
                )
                self._tracing_provider = PlatformTracingProvider
                return
        logger.info(
            "Using in-memory tracing provider.", 
            run_id=self.id
        )
        self._tracing_provider = InMemoryTracingProvider


    async def _get_parent_tracing(self) -> Tracing | None:
        """Load the tracing data for the parent run.
        
        INTERNAL METHOD: This method is intended for internal framework use.
        Use with caution as it involves async I/O operations and direct
        interaction with the tracing provider.
        
        Returns:
            The parent run's tracing data, or None if this is a root run.
        """
        if self.parent_id:
            return await self._tracing_provider.get(self)
        return None


    async def _save_tracing(self) -> None:
        """Save the tracing data for the run.
        
        INTERNAL METHOD: This method is intended for internal framework use.
        It persists the current run's tracing data using the configured
        tracing provider. Manual calls to this method may interfere with
        the framework's automatic tracing lifecycle.
        """
        await self._tracing_provider.put(self)


    def current_trace(self) -> Trace:
        """Get the trace for the current call."""
        from . import get_call_id
        call_id = get_call_id()
        trace = self._tracing.get(call_id)
        if not trace:
            raise RuntimeError(f"Could not resolve current trace for call ID {call_id}")
        return trace

    
    def parent_trace(self) -> Trace:
        """Get the trace for the parent call."""
        trace = self.current_trace()
        parent_call_id = trace.parent_call_id
        parent_trace = self._tracing.get(parent_call_id)
        if not parent_trace:
            raise RuntimeError(f"Could not resolve parent trace for call ID {parent_call_id}")
        return parent_trace

    
    def step_trace(self, name: str) -> Trace:
        """Get the trace for a neighbor step by name."""
        trace = self.current_trace()
        parent_call_id = trace.parent_call_id
        for trace in self._tracing.values():
            if (trace.parent_call_id == parent_call_id and 
                # trace.call_id != current_trace.call_id and  # Don't match self
                trace.path.endswith("." + name)):
                return trace
        raise RuntimeError(f"Could not resolve step trace for call ID {parent_call_id} and step name {name}")
    

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
            assert call_id in self._tracing, f"RunContext.update_usage: Call ID {call_id} not found in tracing."
            trace = self._tracing[call_id]
            current_value = trace.usage.get(key, 0)
            trace.usage[key] = current_value + value
            call_id = trace.parent_call_id
