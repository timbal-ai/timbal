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

TIMBAL_OPENAI_API = os.getenv("TIMBAL_OPENAI_API", "responses")
if TIMBAL_OPENAI_API != "responses":
    logger.warning(
        "Using legacy Chat Completions API. OpenAI is transitioning to the new Responses API, "
        "which should be preferred for all new development. Set TIMBAL_OPENAI_API=responses to switch."
    )

# Model type with provider prefixes
Model = Literal[
    # OpenAI models
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o3",
    "openai/o3-mini",
    "openai/o3-pro",
    "openai/o3-deep-research",
    "openai/o4-mini",
    "openai/o4-mini-deep-research",
    "openai/gpt-4o-mini-audio-preview",
    # Anthropic models
    "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-opus-4-1",
    "anthropic/claude-opus-4-0",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-sonnet-4-0",
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-3-7-sonnet-latest",
    "anthropic/claude-3-5-haiku-latest",
    "anthropic/claude-3-opus-latest",
    "anthropic/claude-3-haiku-20240307",
    # TogetherAI models
    "togetherai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "togetherai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "togetherai/meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "togetherai/Qwen/Qwen3.5-397B-A17B",
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507",
    "togetherai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "togetherai/Qwen/Qwen3-Coder-Next-FP8",
    "togetherai/Qwen/Qwen3-Next-80B-A3B-Instruct",
    "togetherai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "togetherai/deepseek-ai/DeepSeek-V3.1",
    "togetherai/deepseek-ai/DeepSeek-R1",
    "togetherai/moonshotai/Kimi-K2.5",
    "togetherai/moonshotai/Kimi-K2-Instruct",
    "togetherai/MiniMaxAI/MiniMax-M2.5",
    "togetherai/zai-org/GLM-5",
    "togetherai/zai-org/GLM-4.7",
    "togetherai/openai/gpt-oss-120b",
    "togetherai/google/gemma-3n-E4B-it",
    "togetherai/google/gemma-3-27b-it",
    "togetherai/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "togetherai/deepcogito/cogito-v2-1-671b",
    "togetherai/mistralai/Mistral-Small-24B-Instruct-2501",
    "togetherai/mistralai/Ministral-3-14B-Instruct-2512",
    # Gemini models
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-native-audio-preview-12-2025",
    "google/gemini-2.5-flash-image",
    "google/gemini-2.5-flash-preview-tts",
    "google/gemini-2.5-pro-preview-tts",
    # x.ai/Grok models
    "xai/grok-4",
    "xai/grok-4-fast-reasoning",
    "xai/grok-4-fast-non-reasoning",
    "xai/grok-4-1-fast-reasoning",
    "xai/grok-4-1-fast-non-reasoning",
]


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


async def _llm_router(
    model: Model | str,
    system_prompt: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Runnable] | None = None,
    max_tokens: int | None = None,
    thinking: Any = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    cache_control: Any = None,
    output_model: type[BaseModel] | None = None,
    base_url: str | SecretStr | None = None,
    api_key: str | SecretStr | None = None,
) -> Message:  # type: ignore[misc]  # Declared as Message for framework schema generation; runtime is an async generator of provider-specific chunks.
    """
    Internal LLM router function.

    WARNING: This function is for internal use only and may change frequently
    as LLM providers update their APIs. Use the stable Agent/Workflow APIs instead.
    """
    messages = messages or []

    # Convert SecretStr to str if needed
    if isinstance(base_url, SecretStr):
        base_url = base_url.get_secret_value()
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()

    if "/" not in model:
        raise ValueError("Model must be in format 'provider/model_name'")

    provider, model_name = model.split("/", 1)

    run_context = get_or_create_run_context()
    call_id = get_call_id()
    default_headers = {
        "x-timbal-run-id": run_context.id,
        "x-timbal-call-id": call_id,
    }
    if run_context.platform_config and run_context.platform_config.subject:
        if run_context.platform_config.subject.app_id:
            default_headers["x-timbal-app-id"] = run_context.platform_config.subject.app_id
        if run_context.platform_config.subject.version_id:
            default_headers["x-timbal-version-id"] = run_context.platform_config.subject.version_id

    if provider == "openai":
        default_headers["x-provider"] = "openai"
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if run_context.platform_config is not None and run_context.platform_config.subject is not None:
                api_key = run_context.platform_config.auth.header_value
                proxy_api = "openai-responses"
                if TIMBAL_OPENAI_API != "responses":
                    proxy_api = "openai-completions"
                base_url = f"https://{run_context.platform_config.host}/orgs/{run_context.platform_config.subject.org_id}/proxies/{proxy_api}/v1"
        if not api_key:
            raise APIKeyNotFoundError("OPENAI_API_KEY not found.")
        if base_url is not None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        else:
            client = AsyncOpenAI(api_key=api_key, default_headers=default_headers)

    elif provider == "anthropic":
        default_headers["x-provider"] = "anthropic"
        if not max_tokens:
            raise ValueError("'max_tokens' is required for claude models.")
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            if run_context.platform_config is not None and run_context.platform_config.subject is not None:
                api_key = run_context.platform_config.auth.header_value
                base_url = f"https://{run_context.platform_config.host}/orgs/{run_context.platform_config.subject.org_id}/proxies/anthropic"
        if not api_key:
            raise APIKeyNotFoundError("ANTHROPIC_API_KEY not found.")
        if base_url is not None:
            client = AsyncAnthropic(api_key=api_key, base_url=base_url, default_headers=default_headers)
        else:
            client = AsyncAnthropic(api_key=api_key, default_headers=default_headers)

    elif provider == "google":
        default_headers["x-provider"] = "google"
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            if run_context.platform_config is not None and run_context.platform_config.subject is not None:
                api_key = run_context.platform_config.auth.header_value
                base_url = f"https://{run_context.platform_config.host}/orgs/{run_context.platform_config.subject.org_id}/proxies/openai-completions/v1"
        if not api_key:
            raise APIKeyNotFoundError("GEMINI_API_KEY not found.")
        if base_url is not None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        else:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                default_headers=default_headers,
            )

    elif provider == "togetherai":
        default_headers["x-provider"] = "togetherai"
        if not api_key:
            api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            if run_context.platform_config is not None and run_context.platform_config.subject is not None:
                api_key = run_context.platform_config.auth.header_value
                base_url = f"https://{run_context.platform_config.host}/orgs/{run_context.platform_config.subject.org_id}/proxies/openai-completions/v1"
        if not api_key:
            raise APIKeyNotFoundError("TOGETHER_API_KEY not found.")
        if base_url is not None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        else:
            client = AsyncOpenAI(
                api_key=api_key, base_url="https://api.together.xyz/v1/", default_headers=default_headers
            )

    elif provider == "xai":
        default_headers["x-provider"] = "xai"
        if not api_key:
            api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            if run_context.platform_config is not None and run_context.platform_config.subject is not None:
                api_key = run_context.platform_config.auth.header_value
                base_url = f"https://{run_context.platform_config.host}/orgs/{run_context.platform_config.subject.org_id}/proxies/openai-completions/v1"
        if not api_key:
            raise APIKeyNotFoundError("XAI_API_KEY not found.")
        if base_url is not None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        else:
            client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1", default_headers=default_headers)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "anthropic":
        anthropic_messages = []
        for message in messages:
            anthropic_message = message.to_anthropic_input()
            anthropic_messages.append(anthropic_message)

        anthropic_kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if system_prompt:
            if cache_control:
                anthropic_kwargs["system"] = [{"type": "text", "text": system_prompt, "cache_control": cache_control}]
            else:
                anthropic_kwargs["system"] = system_prompt

        if tools:
            anthropic_tools = []
            for tool in tools:
                anthropic_tools.append(tool.anthropic_schema)
            if anthropic_tools:
                anthropic_kwargs["tools"] = anthropic_tools

        if thinking:
            # {"type": "enabled", "budget_tokens": int}
            # budget_tokens must be >= 1024
            anthropic_kwargs["thinking"] = thinking

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
                res = await client.beta.messages.create(betas=["structured-outputs-2025-11-13"], **anthropic_kwargs)  # type: ignore[attr-defined]
            else:
                res = await client.messages.create(**anthropic_kwargs)  # type: ignore[attr-defined]
            async for chunk in res:
                yield chunk

        async for res_chunk in _retry_on_error(_create_stream, max_retries, retry_delay, "Anthropic"):
            yield res_chunk  # type: ignore[return-type]

    elif provider == "openai" and TIMBAL_OPENAI_API == "responses":
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

        if thinking:
            # {"effort": enum["minimal", "low", "medium", "high"], "summary": enum["auto", "concise", "detailed"]}
            responses_kwargs["reasoning"] = thinking

        if output_model is not None:
            responses_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_model.__name__,
                    "schema": transform_schema(output_model),
                    "strict": True,
                }
            }

        async def _create_stream():
            res = await client.responses.create(**responses_kwargs)  # type: ignore[attr-defined]
            async for chunk in res:
                yield chunk

        async for res_chunk in _retry_on_error(_create_stream, max_retries, retry_delay, "OpenAI Responses"):
            yield res_chunk  # type: ignore[return-type]

    # Try with OpenAI Chat Completions compatible providers
    else:
        chat_completions_messages = []
        if system_prompt:
            chat_completions_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            chat_completions_message = message.to_openai_chat_completions_input()
            chat_completions_messages.append(chat_completions_message)

        chat_completions_kwargs = {
            "model": model_name,
            "messages": chat_completions_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            chat_completions_tools = []
            for tool in tools:
                chat_completions_tools.append(tool.openai_chat_completions_schema)
            if chat_completions_tools:
                chat_completions_kwargs["tools"] = chat_completions_tools

        if max_tokens:
            chat_completions_kwargs["max_completion_tokens"] = max_tokens

        if output_model is not None:
            chat_completions_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "schema": transform_schema(output_model),
                    "strict": True,
                },
            }

        async def _create_stream():
            res = await client.chat.completions.create(**chat_completions_kwargs)  # type: ignore[attr-defined]
            async for chunk in res:
                yield chunk

        async for res_chunk in _retry_on_error(
            _create_stream, max_retries, retry_delay, f"{provider} Chat Completions"
        ):
            yield res_chunk  # type: ignore[return-type]
