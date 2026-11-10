"""Anthropic Messages API request adapter (anthropic)."""

from collections.abc import AsyncIterator, Callable, Collection
from typing import TYPE_CHECKING, Any

from ...types.message import Message
from ...utils import transform_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runnable import Runnable


ANTHROPIC_TOOL_SEARCH_TYPE = "tool_search_tool_bm25_20251119"
ANTHROPIC_TOOL_SEARCH_NAME = "tool_search_tool_bm25"


def _drop_stale_tool_references(messages: list[dict[str, Any]], offered: set[str]) -> list[dict[str, Any]]:
    """Drop past tool-search references to tools this request doesn't offer.

    The API expands every ``tool_reference`` it is sent and rejects one with no definition in
    ``tools`` (HTTP 400). A tool found earlier can be gone now: a narrower ``tool_policy``, a
    forced tool-less final call, the denial breaker. Replaces only the blocks it changes (the
    content lists are fresh per request), so memory keeps the verbatim result, and a request
    whose tools still cover every reference is sent byte-identical.
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_search_tool_result":
                continue
            result = block.get("content")
            references = result.get("tool_references") if isinstance(result, dict) else None
            if not references:
                continue
            kept = [ref for ref in references if ref.get("tool_name") in offered]
            if len(kept) != len(references):
                content[index] = {**block, "content": {**result, "tool_references": kept}}
    return messages


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
    deferred_tools: "Collection[str] | None" = None,
) -> tuple[Callable[[], AsyncIterator[Any]], str]:
    """Build the Messages API kwargs and return (create_stream, context_label).

    ``deferred_tools``: names sent with ``defer_loading`` behind Anthropic's own
    tool search (``Agent.tool_search``), which is added to the tools when any are.
    """
    anthropic_kwargs = {
        "model": model_name,
        "messages": [message.to_anthropic_input() for message in messages],
        "max_tokens": max_tokens,
        "stream": True,
    }

    if system_prompt:
        anthropic_kwargs["system"] = system_prompt

    if tools:
        if not deferred_tools:
            anthropic_tools = [tool.anthropic_schema for tool in tools]
        else:
            deferred = set(deferred_tools)
            anthropic_tools = [
                {**tool.anthropic_schema, "defer_loading": True} if tool.name in deferred else tool.anthropic_schema
                for tool in tools
            ]
            if deferred.intersection(tool.name for tool in tools):
                anthropic_tools.append({"type": ANTHROPIC_TOOL_SEARCH_TYPE, "name": ANTHROPIC_TOOL_SEARCH_NAME})
        if anthropic_tools:
            anthropic_kwargs["tools"] = anthropic_tools

    if temperature is not None:
        anthropic_kwargs["temperature"] = temperature

    # Server-side automatic prompt caching (anthropic SDK >= 0.83): the API
    # places a breakpoint on the last cacheable block and advances it as the
    # conversation grows. Cache-read rates and minimum cacheable prefix lengths
    # vary by model; short prefixes are simply not cached, so this is
    # safe to default on. Opt out with model_params={"cache_control": None}.
    anthropic_kwargs["cache_control"] = {"type": "ephemeral"}

    # provider_params may carry provider-native (server-side) tool defs, e.g.
    # {"type": "web_search_20250305", ...}. Merge them with the client tools
    # instead of letting dict.update clobber the generated list.
    provider_params = dict(provider_params)
    extra_tools = provider_params.pop("tools", None)
    # Caller headers (e.g. `anthropic-beta: fast-mode-2026-02-01` for `speed: "fast"`)
    # are merged under Timbal's own tracing headers instead of colliding with them.
    extra_headers = {**(provider_params.pop("extra_headers", None) or {}), **request_headers}
    anthropic_kwargs.update(provider_params)
    if extra_tools:
        anthropic_kwargs["tools"] = [*anthropic_kwargs.get("tools", []), *extra_tools]
    if anthropic_kwargs["cache_control"] is None:
        del anthropic_kwargs["cache_control"]
    _drop_stale_tool_references(
        anthropic_kwargs["messages"], {tool.get("name") for tool in anthropic_kwargs.get("tools") or ()}
    )

    async def _create_stream():
        if output_model is not None:
            anthropic_kwargs["output_config"] = {
                **anthropic_kwargs.get("output_config", {}),
                "format": {
                    "type": "json_schema",
                    "schema": transform_schema(output_model),
                }
            }
        res = await client.messages.create(extra_headers=extra_headers, **anthropic_kwargs)
        async for chunk in res:
            yield chunk

    return _create_stream, "Anthropic"
