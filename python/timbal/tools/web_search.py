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

from pydantic import computed_field

from ..core.tool import Tool


def _get_logger():
    import structlog
    return structlog.get_logger("timbal.tools.web_search")


class WebSearch(Tool):
    name: str = "web_search"
    # Anthropic and OpenAI recognize this tool natively by type, so description is not required.
    description: str | None = None
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

    @override
    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "allowed_domains": self.allowed_domains,
                    "blocked_domains": self.blocked_domains,
                    "user_location": self.user_location,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        if "handler" not in kwargs:

            def _unreachable():
                raise NotImplementedError("This is a specification-only tool")

            kwargs["handler"] = _unreachable
        super().__init__(**kwargs)

    @override
    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """See base class."""
        schema: dict[str, Any] = {
            "type": "web_search",
        }

        if self.allowed_domains:
            schema["filters"] = {"allowed_domains": self.allowed_domains}
        if self.blocked_domains:
            _get_logger().warning("Blocked domains are not supported by OpenAI.")
        if self.user_location:
            schema["user_location"] = self.user_location

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
        anthropic_schema: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
        }

        if self.allowed_domains:
            anthropic_schema["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            anthropic_schema["blocked_domains"] = self.blocked_domains
        if self.user_location:
            anthropic_schema["user_location"] = self.user_location

        return anthropic_schema
