from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent


class TextContent(BaseContent):
    """Text content type for chat messages."""
    type: Literal["text"] = "text"
    text: str 

    @override
    def to_openai_input(self) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "text", 
            "text": self.text
        }

    @override
    def to_anthropic_input(self) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "text", 
            "text": self.text
        }
