from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import RateLimitError as OpenAIRateLimitError

from ..errors import FallbackExhausted

logger = structlog.get_logger("timbal.core.fallback_model")

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model in a fallback chain."""

    model: str
    max_retries: int = 2
    retry_delay: float = 1.0
    api_key: str | None = None
    base_url: str | None = None


class FallbackModel:
    """Ordered fallback chain for LLM providers.

    The first model is tried first. If it fails with a retryable provider error
    after its per-model retries are exhausted, the next entry is attempted.
    """

    __timbal_fallback_model__ = True
    provider = "fallback"

    def __init__(
        self,
        *models: str | ModelEntry,
        fallback_on: type[BaseException]
        | tuple[type[BaseException], ...]
        | list[type[BaseException]]
        | Callable[[BaseException], bool]
        | None = None,
    ) -> None:
        if not models:
            raise ValueError("FallbackModel requires at least one model.")

        self.entries = tuple(entry if isinstance(entry, ModelEntry) else ModelEntry(entry) for entry in models)
        self.fallback_on = fallback_on
        self.model_name = " -> ".join(entry.model for entry in self.entries)

    def __str__(self) -> str:
        return self.entries[0].model

    async def route(
        self,
        router: Callable[..., AsyncGenerator[Any, None]],
        **llm_router_kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        errors: list[tuple[str, BaseException]] = []

        for index, entry in enumerate(self.entries):
            started = False
            kwargs = {
                **llm_router_kwargs,
                "model": entry.model,
                "max_retries": entry.max_retries,
                "retry_delay": entry.retry_delay,
            }
            if entry.api_key is not None:
                kwargs["api_key"] = entry.api_key
            if entry.base_url is not None:
                kwargs["base_url"] = entry.base_url

            try:
                async for chunk in router(**kwargs):
                    started = True
                    yield chunk
                return
            except Exception as exc:
                if started:
                    raise
                if not self._should_fallback(exc):
                    raise

                errors.append((entry.model, exc))
                next_model = self.entries[index + 1].model if index + 1 < len(self.entries) else None
                logger.warning(
                    "Falling back to next LLM model",
                    failed_model=entry.model,
                    next_model=next_model,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        raise FallbackExhausted(errors)

    def _should_fallback(self, exc: BaseException) -> bool:
        if self.fallback_on is None:
            return is_retryable_provider_error(exc)
        if isinstance(self.fallback_on, type) and issubclass(self.fallback_on, BaseException):
            return isinstance(exc, self.fallback_on)
        if isinstance(self.fallback_on, (tuple, list)):
            return isinstance(exc, self.fallback_on)
        return bool(self.fallback_on(exc))


def is_retryable_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, (OpenAIRateLimitError, AnthropicRateLimitError)):
        return True
    if isinstance(exc, (OpenAIAPITimeoutError, AnthropicAPITimeoutError)):
        return True
    if isinstance(exc, (OpenAIAPIConnectionError, AnthropicAPIConnectionError)):
        return True
    if isinstance(exc, (OpenAIAPIStatusError, AnthropicAPIStatusError)):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, StopAsyncIteration):
        return True

    message = str(exc).lower()
    return "overload" in message or "capacity" in message
