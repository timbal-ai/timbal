from typing import Any

from pydantic import BaseModel

from ....types.message import Message


class LLMResult(BaseModel):
    """Helper class to wrap LLM results."""

    input: dict[str, Any]
    """The input kwargs to the LLM router."""
    output: Message | None = None
    """The output message of the LLM. Will be None if the LLM returned an error."""
    error: dict[str, Any] | None = None
    """Store if any error occurred while running the LLM."""
    t0: int 
    """The start time of the LLM in milliseconds."""
    t1: int
    """The end time of the LLM in milliseconds."""
    usage: dict[str, int] = {}
    """The usage of the LLM."""
