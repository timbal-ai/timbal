from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent
from .file import FileContent
from .text import TextContent


class ToolResultContent(BaseContent):
    """Tool result content type for chat messages."""
    type: Literal["tool_result"] = "tool_result"
    id: str
    content: list[TextContent | FileContent]

    @override
    def to_openai_input(self) -> dict[str, Any]:
        """See base class."""
        return {
            "role": "tool",
            "tool_call_id": self.id,
            "content": [item.to_openai_input() for item in self.content],
        }

    @override
    def to_anthropic_input(self) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": [item.to_anthropic_input() for item in self.content],
        }
