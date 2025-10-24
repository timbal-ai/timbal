from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent


class CustomContent(BaseContent):
    """Custom content type for chat messages."""
    type: Literal["custom"] = "custom"
    value: dict[str, Any] 

    @override
    def to_openai_responses_input(self, role: str, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return self.value

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return self.value

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return self.value
