"""OpenAI Responses API request adapter (openai, xai)."""

import re
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from ...types.message import Message
from ...utils import transform_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runnable import Runnable


# OpenAI reasoning models: the `o` series, GPT-5.x, GPT-6.x and the codex line. With
# `store: false` these only keep their chain of thought across a tool loop if every
# request asks for `reasoning.encrypted_content` and replays the items it gets back.
# Non-reasoning models (gpt-4o, gpt-4.1) and xAI's Responses-compatible endpoint
# are left alone so an unsupported `include` value can never 400 a request.
_ENCRYPTED_REASONING_MODEL_RE = re.compile(r"^(?:o\d|gpt-5|gpt-6|codex)", re.IGNORECASE)

REASONING_ENCRYPTED_CONTENT_INCLUDE = "reasoning.encrypted_content"


def supports_encrypted_reasoning(model_name: str) -> bool:
    """Whether `include: ["reasoning.encrypted_content"]` should be requested for this model."""
    return bool(_ENCRYPTED_REASONING_MODEL_RE.match((model_name or "").strip()))


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
    include = ["web_search_call.action.sources"]
    if supports_encrypted_reasoning(model_name):
        include.append(REASONING_ENCRYPTED_CONTENT_INCLUDE)
    responses_kwargs = {
        "model": model_name,
        "stream": True,
        "store": False,
        "include": include,
    }

    if system_prompt:
        responses_kwargs["instructions"] = system_prompt

    input_items = sum([message.to_openai_responses_input() for message in messages], [])
    if not supports_encrypted_reasoning(model_name):
        # History written by an OpenAI reasoning model, replayed to something else
        # (FallbackModel to xAI, a gpt-4.1 follow-up): `reasoning` items with another
        # model's encrypted payload and the assistant `phase` field are not accepted
        # there. The function_call each reasoning item preceded stays.
        input_items = [
            {k: v for k, v in item.items() if k != "phase"}
            for item in input_items
            if not (item.get("type") == "reasoning" and item.get("encrypted_content"))
        ]
    responses_kwargs["input"] = input_items

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

    # provider_params may carry provider-native (server-side) tool defs, e.g.
    # {"type": "web_search"}. Merge them with the client tools instead of
    # letting dict.update clobber the generated list.
    provider_params = dict(provider_params)
    extra_tools = provider_params.pop("tools", None)
    # A caller-supplied `include` extends ours rather than replacing it: dropping
    # `reasoning.encrypted_content` silently would reintroduce the lost-chain-of-thought bug.
    extra_include = provider_params.pop("include", None)
    if extra_include:
        responses_kwargs["include"] = list(dict.fromkeys([*responses_kwargs["include"], *extra_include]))
    responses_kwargs.update(provider_params)
    if extra_tools:
        responses_kwargs["tools"] = [*responses_kwargs.get("tools", []), *extra_tools]

    async def _create_stream():
        res = await client.responses.create(extra_headers=request_headers, **responses_kwargs)
        async for chunk in res:
            yield chunk

    return _create_stream, "OpenAI Responses"
