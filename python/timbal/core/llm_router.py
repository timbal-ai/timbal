"""
INTERNAL USE ONLY

This module is intended for internal use and will be subject to frequent changes
as LLM providers constantly update their APIs and add new features. The external
APIs (Runnables, Agents, Workflows) will remain stable, but this module will
evolve to match provider changes.

Do not rely on this module's interface in external code.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal

import structlog
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError  # APIError as AnthropicAPIError,
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError  # APIError as OpenAIAPIError,
from pydantic import BaseModel, SecretStr

from ..errors import APIKeyNotFoundError
from ..state import get_call_id, get_or_create_run_context
from ..types.message import Message
from ..utils import transform_schema
from .runnable import Runnable

logger = structlog.get_logger("timbal.core.llm_router")

# Module-level client cache keyed by (client_class, api_key, base_url, provider).
# Reusing clients preserves the underlying httpx connection pool, avoiding a
# fresh TCP+TLS handshake on every LLM call (~200-300ms saved per request).
# Per-request tracing headers (run_id, call_id) are passed via extra_headers
# on each individual .create() call instead.
_CLIENT_CACHE: dict[tuple, AsyncOpenAI | AsyncAnthropic] = {}


def _get_client(cls: type, api_key: str, base_url: str | None, provider: str) -> AsyncOpenAI | AsyncAnthropic:
    cache_key = (cls, api_key, base_url, provider)
    if cache_key not in _CLIENT_CACHE:
        kwargs: dict[str, Any] = {"api_key": api_key, "default_headers": {"x-provider": provider}}
        if base_url:
            kwargs["base_url"] = base_url
        _CLIENT_CACHE[cache_key] = cls(**kwargs)
    return _CLIENT_CACHE[cache_key]


TIMBAL_OPENAI_API = os.getenv("TIMBAL_OPENAI_API", "responses")
if TIMBAL_OPENAI_API != "responses":
    logger.warning(
        "Using legacy Chat Completions API. OpenAI is transitioning to the new Responses API, "
        "which should be preferred for all new development. Set TIMBAL_OPENAI_API=responses to switch."
    )


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _ProviderConfig:
    """Static configuration for a single LLM provider."""

    env_key: str
    """Environment variable name for the API key (e.g. ``OPENAI_API_KEY``)."""

    default_base_url: str | None = None
    """Default API base URL.  ``None`` uses the SDK default."""

    proxy_name: str = "openai-completions"
    """Platform proxy path segment (e.g. ``openai-responses``, ``anthropic``)."""

    proxy_suffix: str = "/v1"
    """Appended to the proxy URL.  Anthropic uses ``""``."""

    client_type: Literal["openai", "anthropic"] = "openai"
    """Which SDK client to create."""

    flatten_text_content: bool = False
    """Flatten text-only content arrays to plain strings for providers with incomplete Chat Completions support."""

    supports_stream_options: bool = True
    """Whether the provider supports ``stream_options`` in Chat Completions."""


_PROVIDERS: dict[str, _ProviderConfig] = {
    "openai": _ProviderConfig(
        env_key="OPENAI_API_KEY",
        proxy_name="openai-responses" if TIMBAL_OPENAI_API == "responses" else "openai-completions",
    ),
    "anthropic": _ProviderConfig(
        env_key="ANTHROPIC_API_KEY",
        proxy_name="anthropic",
        proxy_suffix="",
        client_type="anthropic",
    ),
    "google": _ProviderConfig(
        env_key="GEMINI_API_KEY",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "togetherai": _ProviderConfig(
        env_key="TOGETHER_API_KEY",
        default_base_url="https://api.together.xyz/v1/",
    ),
    "xai": _ProviderConfig(
        env_key="XAI_API_KEY",
        default_base_url="https://api.x.ai/v1",
        proxy_name="openai-responses",
    ),
    "groq": _ProviderConfig(
        env_key="GROQ_API_KEY",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    "fireworks": _ProviderConfig(
        env_key="FIREWORKS_API_KEY",
        default_base_url="https://api.fireworks.ai/inference/v1",
    ),
    "byteplus": _ProviderConfig(
        env_key="BYTEPLUS_API_KEY",
        default_base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    ),
    "xiaomi": _ProviderConfig(
        env_key="XIAOMI_API_KEY",
        default_base_url="https://api.xiaomimimo.com/v1",
        flatten_text_content=True,
        supports_stream_options=False,
    ),
    "cerebras": _ProviderConfig(
        env_key="CEREBRAS_API_KEY",
        default_base_url="https://api.cerebras.ai/v1",
    ),
    "sambanova": _ProviderConfig(
        env_key="SAMBANOVA_API_KEY",
        default_base_url="https://api.sambanova.ai/v1",
        flatten_text_content=True,
    ),
}


def _resolve_client(
    provider: str,
    config: _ProviderConfig,
    api_key: str | None,
    base_url: str | None,
    run_context: Any,
) -> tuple[AsyncOpenAI | AsyncAnthropic, str | None]:
    """Resolve API key, base URL, and return the appropriate SDK client.

    Returns:
        (client, resolved_base_url) — base_url may have been updated for platform proxies.
    """
    if not api_key:
        api_key = os.getenv(config.env_key)
    if not api_key:
        if run_context.platform_config is not None and run_context.platform_config.subject is not None:
            api_key = run_context.platform_config.auth.header_value
            base_url = (
                f"https://{run_context.platform_config.host}"
                f"/orgs/{run_context.platform_config.subject.org_id}"
                f"/proxies/{config.proxy_name}{config.proxy_suffix}"
            )
    if not api_key:
        raise APIKeyNotFoundError(f"{config.env_key} not found.")

    if config.client_type == "anthropic":
        return _get_client(AsyncAnthropic, api_key, base_url, "anthropic"), base_url
    return _get_client(AsyncOpenAI, api_key, base_url or config.default_base_url, provider), base_url


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _retry_on_error(async_gen_func, max_retries: int, retry_delay: float, context: str):
    """Helper to retry an async generator function on transient failures.

    Retryable errors (using SDK exception types):
    - Empty streams (StopAsyncIteration)
    - Rate limiting (RateLimitError from OpenAI/Anthropic SDKs)
    - Timeouts (APITimeoutError from OpenAI/Anthropic SDKs)
    - Connection errors (APIConnectionError from OpenAI/Anthropic SDKs)
    - Server errors (APIStatusError with 500, 502, 503, 504 status codes)
    - Overloaded/capacity errors (APIError with "overload" or "capacity" in message)

    Non-retryable errors (fail immediately):
    - Authentication errors (401, 403)
    - Invalid requests (400, 404)
    - Other 4xx client errors

    Args:
        async_gen_func: Async callable that returns an async generator
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay for exponential backoff
        context: Description for logging (e.g., "Anthropic API")

    Yields:
        Items from the async generator

    Raises:
        Exception: Original exception if not retryable or max retries exceeded
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            async_gen = async_gen_func()
            # Try to get the first item to detect empty streams
            first_item = await async_gen.__anext__()
            # Success - yield first item and then all remaining items
            yield first_item
            async for item in async_gen:
                yield item
            return  # Successfully completed

        except StopAsyncIteration as e:
            # Empty stream detected
            last_error = e
            error_type = "empty_stream"
            error_msg = "Empty stream"

        except Exception as e:
            # Check if it's a retryable error
            last_error = e
            error_type = type(e).__name__
            error_msg = str(e)

            # Determine if error is retryable based on SDK exception types
            is_retryable = False

            # OpenAI SDK exceptions
            if isinstance(e, (OpenAIRateLimitError, AnthropicRateLimitError)):
                is_retryable = True
                error_type = "rate_limit"

            elif isinstance(e, (OpenAIAPITimeoutError, AnthropicAPITimeoutError)):
                is_retryable = True
                error_type = "timeout"

            elif isinstance(e, (OpenAIAPIConnectionError, AnthropicAPIConnectionError)):
                is_retryable = True
                error_type = "connection_error"

            elif isinstance(e, (OpenAIAPIStatusError, AnthropicAPIStatusError)):
                # Check status code for retryable HTTP errors
                status_code = getattr(e, "status_code", None)
                if status_code in [500, 502, 503, 504]:
                    is_retryable = True
                    if status_code == 503:
                        error_type = "service_unavailable"
                    else:
                        error_type = f"server_error_{status_code}"
                # Don't retry on 4xx errors (client errors like 400, 401, 403, 404)

            # If not retryable, re-raise immediately
            if not is_retryable:
                logger.error(
                    "Non-retryable error from LLM provider", context=context, error_type=error_type, error=error_msg
                )
                raise

        # Retry logic for retryable errors
        if attempt < max_retries:
            delay = retry_delay * (2**attempt)
            logger.warning(
                "Retryable error from LLM provider, retrying...",
                context=context,
                error_type=error_type,
                error=error_msg,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
            )
            await asyncio.sleep(delay)
        else:
            # Max retries exceeded
            logger.error(
                "Max retries exceeded for LLM provider", context=context, error_type=error_type, max_retries=max_retries
            )
            # Re-raise the last error
            raise last_error


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

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
    provider_params: dict[str, Any] | None = None,
) -> Message:  # type: ignore[misc]  # Declared as Message for framework schema generation; runtime is an async generator of provider-specific chunks.
    """
    Internal LLM router function.

    WARNING: This function is for internal use only and may change frequently
    as LLM providers update their APIs. Use the stable Agent/Workflow APIs instead.
    """
    messages = messages or []
    provider_params = provider_params or {}

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
        "x-timbal-call-id": call_id,
    }
    if run_context.platform_config and run_context.platform_config.subject:
        if run_context.platform_config.subject.app_id:
            request_headers["x-timbal-app-id"] = run_context.platform_config.subject.app_id
        if run_context.platform_config.subject.version_id:
            request_headers["x-timbal-version-id"] = run_context.platform_config.subject.version_id

    client, base_url = _resolve_client(provider, config, api_key, base_url, run_context)

    if provider == "anthropic":
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
                # TODO: Review when Anthropic promotes structured outputs to stable API.
                # Currently using beta endpoint because structured outputs (output_format with json_schema)
                # is only available via the beta API with the "structured-outputs-2025-11-13" feature flag.
                # See: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
                anthropic_kwargs["output_format"] = {
                    "type": "json_schema",
                    "schema": transform_schema(output_model),
                }
                res = await client.beta.messages.create(betas=["structured-outputs-2025-11-13"], extra_headers=request_headers, **anthropic_kwargs)  # type: ignore[attr-defined]
            else:
                res = await client.messages.create(extra_headers=request_headers, **anthropic_kwargs)  # type: ignore[attr-defined]
            async for chunk in res:
                yield chunk

        async for res_chunk in _retry_on_error(_create_stream, max_retries, retry_delay, "Anthropic"):
            yield res_chunk  # type: ignore[return-type]

    elif provider in ("openai", "xai") and TIMBAL_OPENAI_API == "responses":
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
            res = await client.responses.create(extra_headers=request_headers, **responses_kwargs)  # type: ignore[attr-defined]
            async for chunk in res:
                yield chunk

        async for res_chunk in _retry_on_error(_create_stream, max_retries, retry_delay, "OpenAI Responses"):
            yield res_chunk  # type: ignore[return-type]

    # OpenAI Chat Completions compatible providers
    else:
        chat_completions_messages = []
        if system_prompt:
            chat_completions_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            chat_completions_message = message.to_openai_chat_completions_input()
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
            res = await client.chat.completions.create(extra_headers=request_headers, **chat_completions_kwargs)  # type: ignore[attr-defined]
            async for chunk in res:
                yield chunk

        async for res_chunk in _retry_on_error(
            _create_stream, max_retries, retry_delay, f"{provider} Chat Completions"
        ):
            yield res_chunk  # type: ignore[return-type]
