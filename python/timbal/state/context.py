import os
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from uuid_extensions import uuid7

from .config import PlatformConfig
from .tracing import Tracing
from .tracing.providers import InMemoryTracingProvider, PlatformTracingProvider, TracingProvider

logger = structlog.get_logger("timbal.state.context")


class RunContext(BaseModel):
    """Runtime execution context shared across all components in a run.
    
    The RunContext provides a centralized location for:
    - Execution tracing and monitoring
    - Data sharing between steps and components
    - Usage tracking and statistics
    - Parent-child run relationships
    
    This context is automatically created and managed by the framework and is
    accessible through get_run_context() in runtime callables like system prompt
    functions and hooks.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    id: str = Field(
        default_factory=lambda: uuid7(as_type="str"),
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
    tracing: Tracing = Field(
        default_factory=Tracing,
        description=(
            "Execution traces for the run."
            "Stores detailed execution information including input, output, error, timing, and usage."
            "Usage data is stored within each trace entry under the 'usage' key."
        ),
    )
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
        # TODO Enable custom providers
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
            assert call_id in self.tracing, f"RunContext.update_usage: Call ID {call_id} not found in tracing."
            trace = self.tracing[call_id]
            current_value = trace.usage.get(key, 0)
            trace.usage[key] = current_value + value
            call_id = trace.parent_call_id


    def get_data(self, key: str) -> Any:
        """Get data using ref key format: .input.field, ..output, step_name.shared.key"""
        from . import get_call_id
        current_call_id = get_call_id()
        if not current_call_id:
            raise RuntimeError("get_data() can only be called within a Runnable execution context")
        current_trace = self.tracing.get(current_call_id)
        if not current_trace:
            raise RuntimeError(f"Call ID {current_call_id} not found in tracing.")
        target_trace, data_key = self._resolve_target_trace(key, current_trace)
        return self._extract_data_from_trace(target_trace, data_key)


    def _resolve_target_trace(self, key: str, current_trace) -> tuple[Any, str]:
        """Resolve which trace to get data from and what key to use."""
        if key.startswith(".."):
            if not current_trace.parent_call_id:
                raise ValueError(f"No parent call found for key: {key}")
            parent_trace = self.tracing.get(current_trace.parent_call_id)
            if not parent_trace:
                raise RuntimeError(f"Parent call ID {current_trace.parent_call_id} not found in tracing.")
            return parent_trace, key[2:]  # Remove ".."
        elif key.startswith("."):
            return current_trace, key[1:]  # Remove "."
        else:
            key_parts = key.split(".", 1)
            if len(key_parts) < 2:
                raise ValueError(f"Invalid key format for sibling reference: {key}")
            sibling_name = key_parts[0]
            remaining_key = key_parts[1]
            sibling_trace = self._find_sibling_trace(current_trace, sibling_name)
            if not sibling_trace:
                raise ValueError(f"Sibling '{sibling_name}' not found")
            return sibling_trace, remaining_key


    def _find_sibling_trace(self, current_trace, sibling_name: str):
        """Find a sibling trace by name (same parent, path ends with name)."""
        parent_call_id = current_trace.parent_call_id
        if not parent_call_id:
            return None
        for trace in self.tracing.values():
            if (trace.parent_call_id == parent_call_id and 
                trace.call_id != current_trace.call_id and  # Don't match self
                trace.path.endswith("." + sibling_name)):
                return trace
        return None


    def _extract_data_from_trace(self, trace, data_key: str) -> Any:
        """Extract data from trace using the data key (input.field, output, shared.key)."""
        key_parts = data_key.split(".")
        if not key_parts:
            raise ValueError("Empty data key")
        data_source = key_parts[0]
        field_path = key_parts[1:] if len(key_parts) > 1 else []
        # TODO Convert to an assertion
        if data_source == "input":
            current_data = trace.input
        elif data_source == "output":
            current_data = trace.output
        elif data_source == "shared":
            current_data = trace.shared
        else:
            raise ValueError(f"Invalid data source: {data_source}. Must be input, output, or shared.")
        for field in field_path:
            if isinstance(current_data, dict) and field in current_data:
                current_data = current_data[field]
            elif hasattr(current_data, field):
                current_data = getattr(current_data, field)
            else:
                raise ValueError(f"Field '{field}' not found in {data_source}")
        return current_data


    def set_data(self, key: str, value: Any) -> None:
        """"""
        from . import get_call_id
        current_call_id = get_call_id()
        if not current_call_id:
            raise RuntimeError("set_data() can only be called within a Runnable execution context")
        trace = self.tracing[current_call_id]
        trace.shared[key] = value
