import inspect
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
        # Find the first Runnable instance in the call stack
        call_id = self._get_call_id_from_stack()
        
        # Update usage for all parents in the call stack
        processed_root = False
        while not processed_root:
            assert call_id in self.tracing, f"RunContext.update_usage: Call ID {call_id} not found in tracing."
            if call_id is None:
                processed_root = True
            trace = self.tracing[call_id]
            current_value = trace["usage"].get(key, 0)
            trace["usage"][key] = current_value + value
            call_id = trace.get("parent_call_id", None)

    def _get_call_id_from_stack(self) -> str:
        """Inspect the call stack to find the first Runnable instance and return its path and call_id."""
        # Import Runnable here to avoid circular imports
        from ..core_v2.runnable import Runnable
        
        try:
            stack_frames = inspect.stack()
        except Exception as e:
            logger.error("inspect_stack_error", error=e)
            return None
            
        runnable_path = None
        for frame_info in stack_frames:
            try:
                frame = frame_info.frame
                frame_locals = frame.f_locals
                    
                if "self" in frame_locals and not runnable_path:
                    obj = frame_locals["self"]
                    if isinstance(obj, Runnable):
                        runnable_path = obj._path

                if not runnable_path:
                    if frame_info.function == "_next":
                        runnable_path = frame_locals.get("_path", None)
                        _call_id = frame_locals.get("_call_id", None)
                        return _call_id
                    continue
                    
                _call_id = frame_locals.get("_call_id", None)
                if _call_id:
                    return _call_id

                tool_call = frame_locals.get("tool_call", None)
                if tool_call and hasattr(tool_call, 'id'):
                    return tool_call.id
                        
            except Exception as e:
                # Skip problematic frames
                logger.error("inspect_stack_error", error=e)
                continue
            
        return None
