"""
WebSearch is a specification-only tool.

Specification-only tools are tools that exist purely to define parameter
schemas and return types for LLM interaction, without containing any
executable logic. They serve as "interface contracts" that tell the LLM what
parameters are expected and what the tool conceptually returns, but the
actual execution happens elsewhere (or not at all).
"""

from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import BaseModel, Field, computed_field

from ..core.tool import Tool

logger = structlog.get_logger("timbal.tools.web_search")


class WebSearchConfig(BaseModel):
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    # User location example: {
    #     "type": "approximate",
    #     "country": "GB",
    #     "city": "London",
    #     "region": "London",
    #     "timezone": "Europe/London"
    # }
    user_location: dict[str, Any] | None = None


class WebSearch(Tool):
    config: WebSearchConfig = Field(default_factory=WebSearchConfig)

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
        name: str = "web_search",
        description: str | None = None,
        # ? anthropic's max_uses
        **kwargs: Any,
    ) -> None:
        # Legacy param support: fold direct params into config when config
        # isn't explicitly provided, preserving backward compatibility.
        if "config" not in kwargs:
            kwargs["config"] = WebSearchConfig(
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                user_location=user_location,
            )

        def _unreachable_handler():
            raise NotImplementedError("This is a specification-only tool")

        super().__init__(name=name, description=description, handler=_unreachable_handler, **kwargs)

    @override
    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """See base class."""
        schema = {
            "type": "web_search",
        }

        if self.config.allowed_domains:
            schema["filters"] = {"allowed_domains": self.config.allowed_domains}
        if self.config.blocked_domains:
            logger.warning("Blocked domains are not supported by OpenAI.")
        if self.config.user_location:
            schema["user_location"] = self.config.user_location

        return schema

    @override
    @computed_field(repr=False)
    @cached_property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """See base class."""
        raise ValueError("WebSearch is not compatible with OpenAI's chat completions API.")

    @override
    @computed_field
    @cached_property
    def anthropic_schema(self) -> dict[str, Any]:
        """See base class."""
        anthropic_schema = {
            "type": "web_search_20250305",
            "name": "web_search",
        }

        if self.config.allowed_domains:
            anthropic_schema["allowed_domains"] = self.config.allowed_domains
        if self.config.blocked_domains:
            anthropic_schema["blocked_domains"] = self.config.blocked_domains
        if self.config.user_location:
            anthropic_schema["user_location"] = self.config.user_location

        return anthropic_schema
