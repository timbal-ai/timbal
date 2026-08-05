"""Retry loop for transient provider failures (rate limits, timeouts, 5xx)."""

import asyncio
import random

import structlog

from ..provider_errors import provider_error_classes

logger = structlog.get_logger("timbal.core.llm")

MAX_RETRY_DELAY = 30.0


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
            # (classes resolved lazily — we're on an LLM error path, the SDK
            # that raised is already imported).
            is_retryable = False
            err_cls = provider_error_classes()

            if isinstance(e, err_cls["rate_limit"]):
                is_retryable = True
                error_type = "rate_limit"

            elif isinstance(e, err_cls["timeout"]):
                is_retryable = True
                error_type = "timeout"

            elif isinstance(e, err_cls["connection"]):
                is_retryable = True
                error_type = "connection_error"

            elif isinstance(e, err_cls["status"]):
                # Check status code for retryable HTTP errors
                status_code = getattr(e, "status_code", None)
                if status_code in [429, 500, 502, 503, 504]:
                    is_retryable = True
                    if status_code == 429:
                        error_type = "rate_limit"
                    elif status_code == 503:
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
            cap = min(retry_delay * (2**attempt), MAX_RETRY_DELAY)
            delay = random.uniform(0, cap)
            retry_after = _retry_after_seconds(last_error)
            if retry_after is not None:
                delay = max(delay, retry_after)
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


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None
