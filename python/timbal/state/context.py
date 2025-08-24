from collections import UserDict
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from uuid_extensions import uuid7

from .config import TimbalPlatformConfig
from .data import BaseData, DataValue
from .tracing import Tracing
from .tracing.providers import InMemoryTracingProvider, TracingProvider

logger = structlog.get_logger("timbal.state.context")


class RunContextData(UserDict):

    # TODO We should call get_data_key internally for this.
    def __getitem__(self, key: str):
        return super().__getitem__(key).resolve()

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, BaseData):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, DataValue(value=value))

    def as_dict(self) -> dict[str, BaseData]:
        return dict(self.data)


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
    idempotency_key: str | None = Field(
        None,
        description="Idempotency key for the run."
    )
    data: RunContextData = Field(
        default_factory=RunContextData,
        description="Data to be shared between steps in an agent or workflow."
    )
    timbal_platform_config: TimbalPlatformConfig | None = Field(
        None,
        description="Platform configuration for the run."
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
        """
        # TODO Enable custom providers
        # TODO Enable platform provider
        logger.warning(
            "Neither custom tracing provider nor platform config found. "
            "Using in-memory tracing provider.",
            run_id=self.id,
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
            return await self._tracing_provider.get(self.parent_id)
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


    def get_current_input(self) -> dict[str, Any] | None:
        """Get the input parameters for the currently executing component.
        
        This method retrieves the input parameters that were passed to the
        component currently being executed. Useful in system prompt functions
        and hooks to access the original input data.
        
        Returns:
            Dictionary containing the input parameters, or None if no current call
        """
        from . import get_call_id
        call_id = get_call_id()
        if not call_id: 
            return None
        assert call_id in self.tracing, f"RunContext.get_current_input: Call ID {call_id} not found in tracing."
        trace = self.tracing[call_id]
        return trace.input
