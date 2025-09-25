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
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any]:
        """Convert the content to the input format required by OpenAI's chat completions api."""
        pass
    
    @abstractmethod
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """Convert the content to the input format required by Anthropic's api."""
        pass
