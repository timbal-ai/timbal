from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from ..errors import FallbackExhausted
from .provider_errors import provider_error_classes

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

    The first model is tried first. If it fails *before any chunk has been
    yielded*, the next entry is attempted. By default any ``Exception`` triggers
    fallback — the rationale is that if you bothered to configure a fallback,
    you almost certainly want it to actually take over (auth failures, bad
    requests, missing keys, transient network errors, etc.). ``BaseException``
    subclasses (``KeyboardInterrupt``, ``SystemExit``, ``asyncio.CancelledError``)
    are *not* caught and propagate normally.

    Once any chunk has streamed from the current model, errors propagate
    immediately — switching mid-stream would corrupt the response.

    Pass ``fallback_on=`` (an exception class, tuple/list of classes, or a
    predicate) to narrow the default. Use :func:`is_retryable_provider_error`
    for the conservative "transient provider errors only" behavior.
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
            has_fallback = index + 1 < len(self.entries)
            kwargs = {
                **llm_router_kwargs,
                "model": entry.model,
                "max_retries": entry.max_retries,
                "retry_delay": entry.retry_delay,
                # A rate limit means the provider is unavailable for a while by
                # definition — while another model is still available, fail over
                # immediately instead of sleeping through Retry-After in place.
                # The last entry has nowhere to go, so it retries normally.
                "fail_fast_rate_limit": has_fallback,
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
                next_model = self.entries[index + 1].model if has_fallback else None
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
            # Default: fallback on anything that isn't a BaseException-only
            # subclass (KeyboardInterrupt, SystemExit, CancelledError). The
            # outer `except Exception` in route() already filters those out,
            # so reaching here means the caller wanted recovery.
            return True
        if isinstance(self.fallback_on, type) and issubclass(self.fallback_on, BaseException):
            return isinstance(exc, self.fallback_on)
        if isinstance(self.fallback_on, (tuple, list)):
            return isinstance(exc, self.fallback_on)
        return bool(self.fallback_on(exc))


def is_retryable_provider_error(exc: BaseException) -> bool:
    # Classes resolved lazily — this runs on an LLM error path, where the SDK
    # that raised is already imported (see provider_errors).
    err_cls = provider_error_classes()
    if isinstance(exc, err_cls["rate_limit"]):
        return True
    if isinstance(exc, err_cls["timeout"]):
        return True
    if isinstance(exc, err_cls["connection"]):
        return True
    if isinstance(exc, err_cls["status"]):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, StopAsyncIteration):
        return True

    message = str(exc).lower()
    return "overload" in message or "capacity" in message
