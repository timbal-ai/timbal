from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseContent(ABC, BaseModel):
    """Abstract base class for all message content types."""
    type: str

    @abstractmethod
    def to_openai_responses_input(self, **kwargs: Any) -> dict[str, Any]:
        """Convert the content to the input format required by OpenAI's responses api."""
        pass
    
    @abstractmethod
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any] | None:
        """Convert the content to the input format required by OpenAI's chat completions api.

        Return ``None`` when the content is not represented as a content block
        (e.g. thinking is carried as top-level ``reasoning_content`` on the message).
        """
        pass
    
    @abstractmethod
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """Convert the content to the input format required by Anthropic's api."""
        pass
