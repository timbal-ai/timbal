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

from ...utils import coerce_to_dict
from .base import BaseContent

logger = structlog.get_logger("timbal.types.content.tool_use")


class ToolUseContent(BaseContent):
    """Tool use content type for chat messages."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]
    thought_signature: str | None = None
    is_server_tool_use: bool = False

    @field_validator("input", mode="before")
    def validate_input(cls, v: Any):
        return coerce_to_dict(v)

    @override
    def to_openai_responses_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        # TODO Review is_server_tool_use
        if self.is_server_tool_use:
            raise NotImplementedError("is_server_tool_use is not supported for OpenAI responses yet")
        return {
            "call_id": self.id,
            "type": "function_call",
            "name": self.name,
            "arguments": json.dumps(self.input),
        }

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        if self.is_server_tool_use:
            raise ValueError("is_server_tool_use is not supported for OpenAI chat completions")

        data = {"id": self.id, "type": "function", "function": {"arguments": json.dumps(self.input), "name": self.name}}

        if self.thought_signature:
            data["extra_content"] = {"google": {"thought_signature": self.thought_signature}}

        return data

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "tool_use" if not self.is_server_tool_use else "server_tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }
