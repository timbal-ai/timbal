from typing import Any

from .base import BaseEvent


class OutputEvent(BaseEvent):
    """Event emitted when a step completes with its full output."""
    type: str = "OUTPUT"

    input: Any
    """The input arguments passed to the step."""
    output: Any | None = None
    """The result of the step (if any)."""
    error: Any | None = None
    """The error that occurred during the step (if any)."""
    t0: int 
    """The start time of the step in milliseconds."""
    t1: int 
    """The end time of the step in milliseconds."""
    usage: dict[str, int] = {}
    """The usage of the step."""
