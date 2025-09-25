import json
from ast import literal_eval
from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import field_validator

from .base import BaseContent

logger = structlog.get_logger("timbal.types.content.tool_use")


class ToolUseContent(BaseContent):
    """Tool use content type for chat messages."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]

    @field_validator("input", mode="before")
    def validate_input(cls, v: Any):
        """Aux function to parse tool use input (output from LLMs) into python objects."""
        if not isinstance(v, dict):
            try:
                v = json.loads(v)
            except Exception:
                try:
                    v = literal_eval(v)
                except Exception:
                    logger.error(
                        "Both json.loads and literal_eval failed when parsing tool_use input", 
                        input=v,
                        exc_info=True
                    )
        return v

    @override
    def to_openai_responses_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "call_id": self.id,
            "type": "function_call",
            "name": self.name,
            "arguments": json.dumps(self.input),
        }

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "arguments": json.dumps(self.input),
                "name": self.name
            }
        }

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }
