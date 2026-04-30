"""
XSearch is a specification-only tool for searching X (Twitter) posts.

This tool is only supported by xAI models via the Responses API.
"""

from functools import cached_property
from typing import Any

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from pydantic import computed_field, model_serializer

from ..core.tool import Tool


class XSearch(Tool):
    name: str = "x_search"
    description: str | None = None

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
        return {"type": "x_search"}

    @override
    @computed_field(repr=False)
    @cached_property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """See base class."""
        raise ValueError("XSearch is only supported by xAI models via the Responses API.")

    @override
    @computed_field
    @cached_property
    def anthropic_schema(self) -> dict[str, Any]:
        """See base class."""
        raise ValueError("XSearch is only supported by xAI models via the Responses API.")

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        """Use the responses schema for serialization since this tool is xAI-only."""
        return self.openai_responses_schema
