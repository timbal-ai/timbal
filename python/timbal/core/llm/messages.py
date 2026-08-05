"""Anthropic Messages API request adapter (anthropic)."""

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from ...types.message import Message
from ...utils import transform_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runnable import Runnable


def prepare_messages_request(
    *,
    client: Any,
    model_name: str,
    request_headers: dict[str, str],
    system_prompt: str | None,
    messages: list[Message],
    tools: "list[Runnable] | None",
    max_tokens: int | None,
    temperature: float | None,
    output_model: "type[BaseModel] | None",
    provider_params: dict[str, Any],
) -> tuple[Callable[[], AsyncIterator[Any]], str]:
    """Build the Messages API kwargs and return (create_stream, context_label)."""
    anthropic_kwargs = {
        "model": model_name,
        "messages": [message.to_anthropic_input() for message in messages],
        "max_tokens": max_tokens,
        "stream": True,
    }

    if system_prompt:
        anthropic_kwargs["system"] = system_prompt

    if tools:
        anthropic_tools = [tool.anthropic_schema for tool in tools]
        if anthropic_tools:
            anthropic_kwargs["tools"] = anthropic_tools

    if temperature is not None:
        anthropic_kwargs["temperature"] = temperature

    anthropic_kwargs.update(provider_params)

    async def _create_stream():
        if output_model is not None:
            anthropic_kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": transform_schema(output_model),
                }
            }
        res = await client.messages.create(extra_headers=request_headers, **anthropic_kwargs)
        async for chunk in res:
            yield chunk

    return _create_stream, "Anthropic"
