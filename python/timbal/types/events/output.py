from typing import Any, Literal

from pydantic import PrivateAttr

from .base import BaseEvent


class OutputEvent(BaseEvent):
    """Event emitted when a step completes with its full output."""
    type: Literal["OUTPUT"] = "OUTPUT"

    input: Any
    """The input arguments passed to the step."""
    output: Any
    """The result of the step."""
    error: Any
    """The error that occurred during the step."""
    t0: int 
    """The start time of the step in milliseconds."""
    t1: int 
    """The end time of the step in milliseconds."""
    usage: dict[str, int]
    """The usage of the step."""
    metadata: dict[str, Any]
    """Additional metadata about the step."""

    _input_dump: Any = PrivateAttr()
    """The dumped/serialized version of input for internal use."""
    _output_dump: Any = PrivateAttr()
    """The dumped/serialized version of output for internal use."""

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to use dumped versions of input and output during serialization."""
        data = super().model_dump(**kwargs)
        # Use dumped versions if available, otherwise fall back to originals
        if hasattr(self, "_input_dump"):
            data["input"] = self._input_dump
        if hasattr(self, "_output_dump"):
            data["output"] = self._output_dump
        return data
