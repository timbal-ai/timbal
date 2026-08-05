"""OpenAI Responses API request adapter (openai, xai)."""

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from ...types.message import Message
from ...utils import transform_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runnable import Runnable


def prepare_responses_request(
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
    """Build the Responses API kwargs and return (create_stream, context_label)."""
    responses_kwargs = {
        "model": model_name,
        "stream": True,
        "store": False,
        "include": ["web_search_call.action.sources"],
    }

    if system_prompt:
        responses_kwargs["instructions"] = system_prompt

    responses_kwargs["input"] = sum([message.to_openai_responses_input() for message in messages], [])

    if tools:
        responses_tools = [tool.openai_responses_schema for tool in tools]
        if responses_tools:
            responses_kwargs["tools"] = responses_tools
            responses_kwargs["parallel_tool_calls"] = True

    if max_tokens:
        responses_kwargs["max_output_tokens"] = max_tokens

    if temperature is not None:
        responses_kwargs["temperature"] = temperature

    if output_model is not None:
        responses_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": output_model.__name__,
                "schema": transform_schema(output_model),
                "strict": True,
            }
        }

    responses_kwargs.update(provider_params)

    async def _create_stream():
        res = await client.responses.create(extra_headers=request_headers, **responses_kwargs)
        async for chunk in res:
            yield chunk

    return _create_stream, "OpenAI Responses"
