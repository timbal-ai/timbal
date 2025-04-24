from typing import Any

from pydantic import BaseModel


class LLMChunk(BaseModel):
    """Helper class to wrap LLM output chunks."""

    output: str
    """The output chunk of the LLM."""
