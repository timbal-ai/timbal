"""OpenAI Chat Completions request adapter.

Serves every OpenAI-compatible provider (google, groq, togetherai, cerebras,
fireworks, moonshot, sambanova, xiaomi, byteplus, ...) plus openai/xai when
``TIMBAL_OPENAI_API`` is set to the legacy Chat Completions mode. Provider
quirks (reasoning content, stream options, text flattening) come from the
router's ``_ProviderConfig``.
"""

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from ...types.message import Message
from ...utils import transform_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runnable import Runnable
    from .registry import _ProviderConfig


def prepare_chat_completions_request(
    *,
    provider: str,
    config: "_ProviderConfig",
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
    """Build the Chat Completions kwargs and return (create_stream, context_label)."""
    chat_completions_messages = []
    if system_prompt:
        chat_completions_messages.append({"role": "system", "content": system_prompt})
    reasoning_as = "reasoning_content" if config.supports_chat_reasoning_content else "omit"
    for message in messages:
        chat_completions_message = message.to_openai_chat_completions_input(reasoning_as=reasoning_as)
        chat_completions_messages.append(chat_completions_message)

    # Some providers have incomplete OpenAI chat completions support.
    # Flatten text-only content arrays to plain strings for compatibility.
    if config.flatten_text_content:
        for msg in chat_completions_messages:
            content = msg.get("content")
            if isinstance(content, list) and all(
                isinstance(item, dict) and item.get("type") == "text" for item in content
            ):
                msg["content"] = "\n".join(item["text"] for item in content)

    chat_completions_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": chat_completions_messages,
        "stream": True,
    }

    if config.supports_stream_options:
        chat_completions_kwargs["stream_options"] = {"include_usage": True}

    if tools:
        chat_completions_tools = [tool.openai_chat_completions_schema for tool in tools]
        if chat_completions_tools:
            chat_completions_kwargs["tools"] = chat_completions_tools

    if max_tokens:
        chat_completions_kwargs["max_completion_tokens"] = max_tokens

    if temperature is not None:
        chat_completions_kwargs["temperature"] = temperature

    if output_model is not None:
        chat_completions_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "schema": transform_schema(output_model),
                "strict": True,
            },
        }

    chat_completions_kwargs.update(provider_params)

    async def _create_stream():
        res = await client.chat.completions.create(extra_headers=request_headers, **chat_completions_kwargs)
        async for chunk in res:
            yield chunk

    return _create_stream, f"{provider} Chat Completions"
