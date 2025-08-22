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
    """Context for a run.
    This is shared between all steps in an agent/workflow (including nested agents/workflows).
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
        """"""
        # TODO Enable custom providers
        # TODO Enable platform provider
        logger.warning(
            "Neither custom tracing provider nor platform config found. "
            "Using in-memory tracing provider.",
            run_id=self.id,
        )
        self._tracing_provider = InMemoryTracingProvider

    def update_usage(self, key: str, value: int) -> None:
        """Update usage statistics within traces with the runnable path from call stack inspection."""
        from . import get_call_id
        call_id = get_call_id()
        # Update usage for all parents in the call stack
        while call_id:
            assert call_id in self.tracing, f"RunContext.update_usage: Call ID {call_id} not found in tracing."
            trace = self.tracing[call_id]
            current_value = trace.usage.get(key, 0)
            trace.usage[key] = current_value + value
            call_id = trace.parent_call_id

    async def get_parent_tracing(self) -> Tracing | None:
        """Load the tracing data for the parent run."""
        if self.parent_id:
            return await self._tracing_provider.get(self.parent_id)
        return None

    async def save_tracing(self) -> None:
        """Save the tracing data for the run."""
        await self._tracing_provider.put(self)
