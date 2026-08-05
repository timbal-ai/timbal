"""SDK client cache and resolution (API keys, platform proxy fallback).

Provider SDKs (openai, anthropic) are imported lazily — at client resolution,
never at module import. They account for ~460ms (a third) of
``from timbal import Agent`` otherwise.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import structlog

from ...errors import APIKeyNotFoundError
from .registry import _PROVIDERS, _ProviderConfig

logger = structlog.get_logger("timbal.core.llm")

# Module-level client cache keyed by (client_class, api_key, base_url, provider).
# Reusing clients preserves the underlying httpx connection pool, avoiding a
# fresh TCP+TLS handshake on every LLM call (~200-300ms saved per request).
# Per-request tracing headers (run_id, call_id) are passed via extra_headers
# on each individual .create() call instead.
# Concurrency-safe: all coroutines run on one thread in asyncio; the GIL
# ensures check-and-assign is atomic (no await between the if and the write).
# Values are AsyncOpenAI | AsyncAnthropic instances (SDKs imported lazily).
_CLIENT_CACHE: dict[tuple, Any] = {}

# Shared httpx client for async file loading. Reused across LLM calls to
# preserve connection pools to file origins (CDNs, S3, etc.). Lazy-initialized
# on first use — never created if the conversation has no files.
# Loop-aware: httpx.AsyncClient is bound to the event loop at creation time.
# When the loop changes (e.g. between pytest-asyncio tests with per-function
# loop scope), the old client becomes unusable ("Event loop is closed").
# We detect this and transparently recreate the client.
_FILE_CLIENT: Any = None
_FILE_CLIENT_LOOP: Any = None


def _get_file_client() -> Any:
    global _FILE_CLIENT, _FILE_CLIENT_LOOP
    loop = asyncio.get_running_loop()
    if _FILE_CLIENT is not None and (_FILE_CLIENT.is_closed or _FILE_CLIENT_LOOP is not loop):
        _FILE_CLIENT = None
    if _FILE_CLIENT is None:
        import httpx

        _FILE_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        _FILE_CLIENT_LOOP = loop
    return _FILE_CLIENT


def _get_client(cls: type, api_key: str, base_url: str | None, provider: str) -> Any:
    cache_key = (cls, api_key, base_url, provider)
    if cache_key not in _CLIENT_CACHE:
        kwargs: dict[str, Any] = {"api_key": api_key, "default_headers": {"x-provider": provider}}
        if base_url:
            kwargs["base_url"] = base_url
        _CLIENT_CACHE[cache_key] = cls(**kwargs)
    return _CLIENT_CACHE[cache_key]


def _resolve_client(
    provider: str,
    config: _ProviderConfig,
    api_key: str | None,
    base_url: str | None,
    run_context: Any,
) -> tuple[Any, str | None]:
    """Resolve API key, base URL, and return the appropriate SDK client.

    Returns:
        (client, resolved_base_url) — base_url may have been updated for platform proxies.
    """
    if not api_key:
        api_key = os.getenv(config.env_key)
    if not api_key:
        if (
            config.supports_platform_proxy
            and run_context.platform_config is not None
            and run_context.platform_config.subject is not None
        ):
            api_key = run_context.platform_config.auth.header_value
            base_url = (
                f"https://{run_context.platform_config.host}"
                f"/orgs/{run_context.platform_config.subject.org_id}"
                f"/proxies/{config.proxy_name}{config.proxy_suffix}"
            )
    if not api_key:
        raise APIKeyNotFoundError(f"{config.env_key} not found.")

    # Lazy SDK import: after the first call this is a sys.modules lookup (~1µs).
    if config.client_type == "anthropic":
        from anthropic import AsyncAnthropic

        return _get_client(AsyncAnthropic, api_key, base_url, "anthropic"), base_url
    from openai import AsyncOpenAI

    return _get_client(AsyncOpenAI, api_key, base_url or config.default_base_url, provider), base_url


async def warmup_llm_connection(model: str) -> None:
    """Pre-establish the provider's HTTPS connection pool for ``model``.

    The first LLM call of a process pays TCP+TLS(+HTTP/2) setup before any
    token arrives — measured ~1.3s extra TTFT cold vs warm against
    api.openai.com. This issues one lightweight authenticated GET
    (``/models``) through the same cached SDK client the real calls will use,
    so the pool is hot by the time the first request fires.

    Best-effort and side-effect free: any failure (no API key, unsupported
    endpoint on OpenAI-compatible providers, timeout) is logged at DEBUG and
    ignored. Callers fire-and-forget (e.g. voice sessions at startup).
    """
    try:
        provider, _ = model.split("/", 1)
    except ValueError:
        return
    config = _PROVIDERS.get(provider)
    if config is None:
        return
    try:
        # _resolve_client only reads run_context.platform_config; don't create
        # a real RunContext here (warmup must not touch tracing state).
        ctx = SimpleNamespace(platform_config=None)
        client, _ = _resolve_client(provider, config, None, None, ctx)
        await asyncio.wait_for(client.models.list(), timeout=5.0)
        logger.debug("llm_connection_warmed", model=model)
    except Exception as e:
        logger.debug("llm_warmup_skipped", model=model, error=str(e))
