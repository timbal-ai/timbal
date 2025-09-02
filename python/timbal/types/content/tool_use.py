import json
from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent


class ToolUseContent(BaseContent):
    """Tool use content type for chat messages."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]

    @override
    def to_openai_input(self) -> dict[str, Any]:
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
    def to_anthropic_input(self) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }
