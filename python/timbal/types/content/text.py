from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent


class TextContent(BaseContent):
    """Text content type for chat messages.

    ``phase`` is the OpenAI Responses assistant-message phase (``"commentary"`` for a
    preamble before tool calls, ``"final_answer"`` for the completed answer). GPT-5.4+
    emit it on every ``message`` output item and want it back verbatim when the
    history is replayed manually — a dropped phase makes a preamble read as a final
    answer and the model stops early or garbles the next tool call. ``None`` for text
    from any other source; it never reaches other providers.
    """
    type: Literal["text"] = "text"
    text: str
    phase: str | None = None

    @override
    def to_openai_responses_input(self, role: str, **kwargs: Any) -> dict[str, Any]:
        """See base class. The phase belongs on the enclosing ``message`` item, which
        ``Message.to_openai_responses_input`` builds — see there."""
        type = "output_text" if role == "assistant" else "input_text"
        return {
            "type": type,
            "text": self.text
        }

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "text", 
            "text": self.text
        }

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "text", 
            "text": self.text
        }
