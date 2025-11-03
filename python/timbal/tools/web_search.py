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
from pydantic import computed_field

from ..core.tool import Tool

logger = structlog.get_logger("timbal.tools.web_search")


class WebSearch(Tool):

    def __init__(
        self, 
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict[str, Any] | None = None,
        # ? anthropic's max_uses
        **kwargs: Any,
    ) -> None:

        def _unreachable_handler():
            raise NotImplementedError("This is a specification-only tool")

        super().__init__(
            name="web_search",
            handler=_unreachable_handler,
            **kwargs
        )

        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        # User location example: {
        #     "type": "approximate",
        #     "country": "GB",
        #     "city": "London",
        #     "region": "London",
        #     "timezone": "Europe/London"
        # }
        self.user_location = user_location

    
    @override
    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """See base class."""
        schema = {"type": "web_search",}

        if self.allowed_domains:
            schema["filters"] = {"allowed_domains": self.allowed_domains}
        if self.blocked_domains:
            logger.warning("Blocked domains are not supported by OpenAI.")
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
        anthropic_schema = {
            "type": "web_search_20250305", # TODO Review
            "name": "web_search",
        }

        if self.allowed_domains:
            anthropic_schema["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            anthropic_schema["blocked_domains"] = self.blocked_domains
        if self.user_location:
            anthropic_schema["user_location"] = self.user_location

        return anthropic_schema
    