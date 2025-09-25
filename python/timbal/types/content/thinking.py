from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent


class ThinkingContent(BaseContent):
    """Thinking content type for chat messages."""
    type: Literal["thinking"] = "thinking"
    thinking: str 

    @override
    def to_openai_responses_input(self, role: str, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        type = "output_text" if role == "assistant" else "input_text"
        return {
            "type": type,
            "text": self.thinking
        }

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "text", 
            "text": self.thinking
        }

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "text", 
            "text": self.thinking
        }
