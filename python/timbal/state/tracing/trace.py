from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# class TraceValue:
#     """Container for trace data that supports both raw and serialized access."""
    
#     def __init__(self, raw_value: Any):
#         self.raw = raw_value
#         self._serialized = None
#         self._serialized_computed = False
    
#     async def serialized(self) -> Any:
#         """Get the serialized version of the raw value."""
#         if not self._serialized_computed:
#             self._serialized = await dump(self.raw)
#             self._serialized_computed = True
#         return self._serialized
    
#     def invalidate_serialized(self) -> None:
#         """Mark serialized cache as invalid when raw value changes."""
#         self._serialized = None
#         self._serialized_computed = False
    
#     def __repr__(self):
#         return f"TraceValue(raw={self.raw})"


class Trace(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    path: str = Field(
        ...,
        description="The path of the runnable.",
    )
    call_id: str = Field(
        ...,
        description="The call id of the runnable.",
    )
    parent_call_id: str | None = Field(
        None,
        description="The parent call id of the runnable.",
    )
    t0: int = Field(
        ...,
        description="The start time of the runnable.",
    )
    input: Any = Field(
        None,
        description="The input of the runnable. Will be None if the runnable has not yet started or if there was an error gathering the input.",
    )
    t1: int | None = Field(
        None, 
        description="The end time of the runnable. Will be None if the runnable has not yet completed.",
    )
    output: Any = Field(
        None,
        description="The output of the runnable. Will be None if the runnable has not yet completed or if there was an error.",
    )
    error: Any = Field(
        None,
        description="The error of the runnable. Will be None if the runnable has not yet completed or if there was no error.",
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="The usage of the runnable.",
    )
