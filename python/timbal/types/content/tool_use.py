import json
from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from pydantic import field_validator

from ...utils import coerce_to_dict
from .base import BaseContent


def canonical_input(value: Any) -> Any:
    """``value`` with every object's keys sorted, recursively.

    A replayed tool input must render the same on every request: storage (e.g. Postgres
    ``jsonb``) does not keep the model's key order, and an input whose keys come back reordered
    breaks the provider's prompt cache from that block on (and, on Claude 5.5 models, the
    prefix binding of every later thinking block).
    """
    if isinstance(value, dict):
        return {key: canonical_input(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_input(item) for item in value]
    return value


class ToolUseContent(BaseContent):
    """Tool use content type for chat messages."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]
    thought_signature: str | None = None
    is_server_tool_use: bool = False

    @field_validator("input", mode="before")
    def validate_input(cls, v: Any):
        return coerce_to_dict(v)

    @override
    def to_openai_responses_input(self, **kwargs: Any) -> dict[str, Any] | None:
        """See base class."""
        # Server-side tool calls are provider-internal execution records (e.g.
        # Anthropic web search replayed after a fallback switch to OpenAI).
        # They can't be represented in another provider's API; skip them — the
        # assistant text already carries the results.
        if self.is_server_tool_use:
            return None
        return {
            "call_id": self.id,
            "type": "function_call",
            "name": self.name,
            "arguments": json.dumps(self.input, sort_keys=True),
        }

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any] | None:
        """See base class."""
        if self.is_server_tool_use:
            return None  # see to_openai_responses_input

        data = {
            "id": self.id,
            "type": "function",
            "function": {"arguments": json.dumps(self.input, sort_keys=True), "name": self.name},
        }

        if self.thought_signature:
            data["extra_content"] = {"google": {"thought_signature": self.thought_signature}}

        return data

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any]:
        """See base class."""
        return {
            "type": "tool_use" if not self.is_server_tool_use else "server_tool_use",
            "id": self.id,
            "name": self.name,
            "input": canonical_input(self.input),
        }
