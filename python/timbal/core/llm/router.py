"""Model-string dispatch: fallback/TestModel short-circuits, client resolution,
file loading, and per-API request dispatch through the shared retry loop."""

import asyncio
from typing import Any

from pydantic import BaseModel, SecretStr

from ...state import get_call_id, get_or_create_run_context, set_billing_id
from ...types.message import Message
from ..runnable import Runnable
from .chat_completions import prepare_chat_completions_request
from .clients import _get_file_client, _resolve_client
from .messages import prepare_messages_request
from .registry import _PROVIDERS, TIMBAL_OPENAI_API
from .responses import prepare_responses_request
from .retry import _retry_on_error


async def _llm_router(
    model: Any,  # Model | str | TestModel — typed as Any so Pydantic doesn't reject TestModel instances
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Runnable] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    output_model: type[BaseModel] | None = None,
    base_url: str | SecretStr | None = None,
    api_key: str | SecretStr | None = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    fail_fast_rate_limit: bool = False,
    provider_params: dict[str, Any] | None = None,
) -> Message:  # type: ignore[misc]  # Declared as Message for framework schema generation; runtime is an async generator of provider-specific chunks.
    """
    Internal LLM router function.

    WARNING: This function is for internal use only and may change frequently
    as LLM providers update their APIs. Use the stable Agent/Workflow APIs instead.
    """
    messages = messages or []
    provider_params = provider_params or {}

    if getattr(model, "__timbal_fallback_model__", False):
        async for chunk in model.route(
            _llm_router,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            output_model=output_model,
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider_params=provider_params,
        ):
            yield chunk  # type: ignore[return-type]
        return

    # Convert SecretStr to str if needed
    if isinstance(base_url, SecretStr):
        base_url = base_url.get_secret_value()
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()

    # TestModel short-circuit — delegates to model.stream() with no network call.
    if hasattr(model, "stream"):
        async for chunk in model.stream(messages=messages):
            yield chunk  # type: ignore[return-type]
        return

    if "/" not in model:
        raise ValueError("Model must be in format 'provider/model_name'")

    set_billing_id(model)
    provider, model_name = model.split("/", 1)

    config = _PROVIDERS.get(provider)
    if config is None:
        raise ValueError(f"Unsupported provider: {provider}")

    # Anthropic requires max_tokens
    if provider == "anthropic" and not max_tokens:
        raise ValueError("'max_tokens' is required for claude models.")

    run_context = get_or_create_run_context()
    call_id = get_call_id()
    # Per-request headers: change every call, so passed via extra_headers on each .create().
    request_headers: dict[str, str] = {
        "x-timbal-run-id": run_context.id,
    }
    if call_id:
        request_headers["x-timbal-call-id"] = call_id
    if run_context.platform_config and run_context.platform_config.subject:
        if run_context.platform_config.subject.app_id:
            request_headers["x-timbal-app-id"] = run_context.platform_config.subject.app_id

    client, base_url = _resolve_client(provider, config, api_key, base_url, run_context)

    # Eagerly load all unloaded file content (async, concurrent) before
    # serialization.  This scan lives here — not inside to_*_input() — because:
    #   1. gather() needs the full list upfront for concurrent downloads.
    #   2. to_*_input() stays sync and pure (format conversion, no I/O).
    # The double iteration (scan here + serialize in to_*_input) is intentional:
    # content arrays are small (1-5 items) and the cost is negligible vs the
    # network calls that follow.
    from ...types.content import FileContent

    _unloaded_files = [
        c.file
        for m in messages
        for c in m.content
        if isinstance(c, FileContent) and object.__getattribute__(c.file, "__fileobj__") is None
    ]
    if _unloaded_files:
        await asyncio.gather(*(f.load(client=_get_file_client()) for f in _unloaded_files))

    # Per-API request builders return the stream factory; the single retry
    # loop below is shared, so the per-chunk generator nesting is identical
    # to an inline implementation.
    request_kwargs: dict[str, Any] = {
        "client": client,
        "model_name": model_name,
        "request_headers": request_headers,
        "system_prompt": system_prompt,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "output_model": output_model,
        "provider_params": provider_params,
    }
    if provider == "anthropic":
        create_stream, context = prepare_messages_request(**request_kwargs)
    elif provider in ("openai", "xai") and TIMBAL_OPENAI_API == "responses":
        create_stream, context = prepare_responses_request(**request_kwargs)
    else:
        # OpenAI Chat Completions compatible providers
        create_stream, context = prepare_chat_completions_request(
            provider=provider, config=config, **request_kwargs,
        )

    async for res_chunk in _retry_on_error(
        create_stream, max_retries, retry_delay, context, fail_fast_rate_limit=fail_fast_rate_limit,
    ):
        yield res_chunk  # type: ignore[return-type]
