
from typing import Any

from pydantic import BaseModel


class Tool(BaseModel):
    """
    A class representing a tool or function that can be used by AI models.
    """
    name: str 
    """The name of the tool."""
    description: str 
    """A description of the tool."""
    input_schema: dict[str, Any]
    """The input schema for the tool."""

    def to_openai(self) -> dict[str, Any]:
        """Convert the tool to OpenAI's expected format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }

    def to_anthropic(self) -> dict[str, Any]:
        """Convert the tool to Anthropic's expected format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
