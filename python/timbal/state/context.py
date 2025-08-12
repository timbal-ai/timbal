import inspect
from collections import UserDict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7

from .config import TimbalPlatformConfig
from .data import BaseData, DataValue


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
    # TODO We could use a custom model for this
    tracing: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "Execution traces for the run."
            "Stores detailed execution information including input, output, error, timing, and usage."
            "Usage data is stored within each trace entry under the 'usage' key."
            "Structure: {runnable_path: {call_id: {usage: {...}, ...}}}"
        ),
    )

    def update_usage(self, key: str, value: int) -> None:
        """Update usage statistics within traces with the runnable path from call stack inspection."""
        
        # Find the first Runnable instance in the call stack
        runnable_path, call_id = self._get_runnable_path_from_stack()
        
        # Initialize runnable_path entry if it doesn't exist
        if runnable_path not in self.tracing:
            self.tracing[runnable_path] = {}
        
        # Initialize call_id entry if it doesn't exist
        if call_id not in self.tracing[runnable_path]:
            self.tracing[runnable_path][call_id] = {"usage": {}}
        
        # Initialize usage within the trace if it doesn't exist
        if "usage" not in self.tracing[runnable_path][call_id]:
            self.tracing[runnable_path][call_id]["usage"] = {}
        
        # Update the usage value
        current_value = self.tracing[runnable_path][call_id]["usage"].get(key, 0)
        self.tracing[runnable_path][call_id]["usage"][key] = current_value + value

    def _get_runnable_path_from_stack(self) -> tuple[str, str]:
        """Inspect the call stack to find the first Runnable instance and return its path and call_id."""
        try:
            # Import Runnable here to avoid circular imports
            from ..core_v2.runnable import Runnable
        except ImportError:
            return ("unknown", "unknown")
        
        try:
            stack_frames = inspect.stack()
        except Exception:
            return ("unknown", "unknown")
            
        runnable_path = "unknown"
        for frame_info in stack_frames:
            try:
                frame = frame_info.frame
                frame_locals = frame.f_locals
                    
                if "self" in frame_locals and runnable_path == "unknown":
                    obj = frame_locals["self"]
                    if isinstance(obj, Runnable):
                        runnable_path = obj._path

                if runnable_path == "unknown":
                    if frame_info.function == "_next":
                        runnable_path = frame_locals.get("_path", "unknown")
                        _call_id = frame_locals.get("_call_id", "unknown")
                        return (runnable_path, _call_id)
                    continue
                    
                _call_id = frame_locals.get("_call_id", None)
                if _call_id:
                    return (runnable_path, _call_id)

                tool_call = frame_locals.get("tool_call", None)
                if tool_call and hasattr(tool_call, 'id'):
                    return (runnable_path, tool_call.id)
                        
            except Exception:
                # Skip problematic frames
                continue
            
        return (runnable_path, "unknown")
