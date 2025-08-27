from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseContent(ABC, BaseModel):
    """Abstract base class for all message content types."""
    type: str

    @abstractmethod
    def to_openai_input(self) -> dict[str, Any]:
        """Convert the content to the input format required by OpenAI."""
        pass
    
    @abstractmethod
    def to_anthropic_input(self) -> dict[str, Any]:
        """Convert the content to the input format required by Anthropic."""
        pass
