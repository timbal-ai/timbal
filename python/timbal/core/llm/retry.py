"""Retry loop for transient provider failures (rate limits, timeouts, 5xx)."""

import asyncio
import contextlib
import random

import structlog

from ...errors import ModelStreamStalled
from ..provider_errors import provider_error_classes

logger = structlog.get_logger("timbal.core.llm")

MAX_RETRY_DELAY = 30.0


async def _next_chunk(stream, idle_timeout: float | None, *, after_output: bool):
    """The stream's next chunk, or ``ModelStreamStalled`` after ``idle_timeout`` seconds of silence."""
    if idle_timeout is None:
        return await stream.__anext__()
    try:
        async with asyncio.timeout(idle_timeout):
            return await stream.__anext__()
    except TimeoutError as e:
        raise ModelStreamStalled(idle_timeout, after_output=after_output) from e


async def _retry_on_error(
    async_gen_func,
    max_retries: int,
    retry_delay: float,
    context: str,
    fail_fast_rate_limit: bool = False,
    idle_timeout: float | None = None,
):
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

    Retry delays honor a ``Retry-After`` response header as a floor, but are
    always capped at ``MAX_RETRY_DELAY`` — a provider-side cooldown of minutes
    must never turn into an in-place sleep of minutes.

    Args:
        async_gen_func: Async callable that returns an async generator
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay for exponential backoff
        context: Description for logging (e.g., "Anthropic API")
        fail_fast_rate_limit: If True, rate-limit errors (RateLimitError / 429)
            are raised immediately instead of retried in place. Set by
            FallbackModel for every entry that still has a fallback behind it:
            a rate-limited provider is unavailable for a while by definition,
            so the chain should move on rather than sleep.
        idle_timeout: Seconds the stream may stay silent between chunks
            (``None``: forever). A stall before the first chunk is retried
            like a timeout; one after it raises ``ModelStreamStalled``.

    Yields:
        Items from the async generator

    Raises:
        Exception: Original exception if not retryable or max retries exceeded
    """
    last_error = None

    for attempt in range(max_retries + 1):
        async_gen = None
        try:
            async_gen = async_gen_func()
            # Try to get the first item to detect empty streams
            first_item = await _next_chunk(async_gen, idle_timeout, after_output=False)
            # Success - yield first item and then all remaining items
            yield first_item
            while True:
                try:
                    item = await _next_chunk(async_gen, idle_timeout, after_output=True)
                except StopAsyncIteration:
                    break
                yield item
            return  # Successfully completed

        except StopAsyncIteration as e:
            # Empty stream detected
            last_error = e
            error_type = "empty_stream"
            error_msg = "Empty stream"

        except ModelStreamStalled as e:
            # The abandoned stream still holds its connection.
            with contextlib.suppress(Exception):
                await async_gen.aclose()
            if e.after_output:
                logger.error("Model stream stalled after output", context=context, idle_timeout=idle_timeout)
                raise
            last_error = e
            error_type = "stream_stalled"
            error_msg = str(e)

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

            if fail_fast_rate_limit and error_type == "rate_limit":
                logger.warning(
                    "Rate limited and a fallback model is configured, failing over without retrying",
                    context=context,
                    error=error_msg,
                )
                raise

        # Retry logic for retryable errors
        if attempt < max_retries:
            cap = min(retry_delay * (2**attempt), MAX_RETRY_DELAY)
            delay = random.uniform(0, cap)
            retry_after = _retry_after_seconds(last_error)
            if retry_after is not None:
                delay = min(max(delay, retry_after), MAX_RETRY_DELAY)
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
