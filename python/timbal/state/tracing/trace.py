from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Trace(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
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
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata storage for run-specific metrics and data.",
    )

    _input_dump: Any = PrivateAttr()
    """The dumped/serialized version of input for internal use."""
    _output_dump: Any = PrivateAttr()
    """The dumped/serialized version of output for internal use."""

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to use dumped versions of input and output during serialization."""
        data = super().model_dump(**kwargs) # Pydantic ignores private attributes by default
        # Use dumped versions if available, otherwise fall back to originals
        if hasattr(self, "_input_dump"):
            data["input"] = self._input_dump
        if hasattr(self, "_output_dump"):
            data["output"] = self._output_dump
        return data
