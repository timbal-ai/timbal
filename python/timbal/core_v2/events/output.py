from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import Event


class OutputEventData(BaseModel):
    """"""
    t0: int = Field(
        ...,
        description="The start time of the step in milliseconds.",
    )
    t1: int = Field(
        ...,
        description="The end time of the step in milliseconds.",
    )
    input: Any = Field(
        ...,
        description="The input to the step.",
    )
    output: Any | None = Field(
        None,
        description="The output of the step.",
    )
    error: Any | None = Field(
        None,
        description="The error that occurred during the step (if any).",
    )
    usage: dict[str, Any] | None = Field(
        None,
        description="The usage of the step.",
    )


class OutputEvent(Event[OutputEventData]):
    """"""
    type: Literal["output"] = "output"
