import asyncio
import json as json_lib
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal

import httpx
import structlog

from ..errors import PlatformError
from ..state import get_or_create_run_context

logger = structlog.get_logger("timbal.platform.utils")

# Connect/pool default. Write/read default to unbounded so large uploads (KB
# ingest, recording multipart, …) don't die with an opaque httpx.WriteTimeout
# after 10s of body send. Override via TIMBAL_HTTP_*_TIMEOUT env vars.
_DEFAULT_CONNECT_TIMEOUT = 10.0


def _env_timeout_seconds(name: str, default: float | None) -> float | None:
    """Parse a timeout env var: float seconds, or none/null/inf/unlimited → None."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("", "none", "null", "inf", "unlimited"):
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {name}={raw!r}: expected a number of seconds, or 'none'."
        ) from exc


def _default_timeout() -> httpx.Timeout:
    """Resolve the default httpx timeout for platform HTTP calls.

    Env knobs (seconds; ``none`` = unbounded):
      - ``TIMBAL_HTTP_TIMEOUT`` — connect + pool (default 10)
      - ``TIMBAL_HTTP_WRITE_TIMEOUT`` — request body send (default unbounded)
      - ``TIMBAL_HTTP_READ_TIMEOUT`` — response body read (default unbounded)
    """
    connect = _env_timeout_seconds("TIMBAL_HTTP_TIMEOUT", _DEFAULT_CONNECT_TIMEOUT)
    if connect is None:
        # Connect must stay finite — otherwise a dead host hangs forever.
        connect = _DEFAULT_CONNECT_TIMEOUT
    write = _env_timeout_seconds("TIMBAL_HTTP_WRITE_TIMEOUT", None)
    read = _env_timeout_seconds("TIMBAL_HTTP_READ_TIMEOUT", None)
    return httpx.Timeout(connect, read=read, write=write)


def _timeout_error_message(exc: httpx.TimeoutException, url: str, timeout: httpx.Timeout) -> str:
    """Human-readable PlatformError body for httpx timeout failures."""
    if isinstance(exc, httpx.WriteTimeout):
        phase = "write"
        detail = "sending the request body"
        knob = "TIMBAL_HTTP_WRITE_TIMEOUT"
        configured = timeout.write
    elif isinstance(exc, httpx.ReadTimeout):
        phase = "read"
        detail = "waiting for / reading the response"
        knob = "TIMBAL_HTTP_READ_TIMEOUT"
        configured = timeout.read
    elif isinstance(exc, httpx.ConnectTimeout):
        phase = "connect"
        detail = "establishing the connection"
        knob = "TIMBAL_HTTP_TIMEOUT"
        configured = timeout.connect
    else:
        phase = "timeout"
        detail = "completing the request"
        knob = "TIMBAL_HTTP_TIMEOUT"
        configured = timeout.connect

    limit = "unbounded" if configured is None else f"{configured:g}s"
    return (
        f"\n"
        f"  URL: {url}\n"
        f"  Error: {phase} timeout while {detail} (limit={limit})\n"
        f"  Hint: raise the limit via {knob}=<seconds> (or '{knob}=none' for "
        f"unbounded), or pass timeout= to the platform request helper."
    )


def _resolve_url_and_headers(
    service: str | None,
    path: str,
    headers: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Build the final URL and auth headers for a request.

    When service is None, uses platform_config (existing behavior).
    When service is set, resolves the target URL and auth headers based on
    the environment: TIMBAL_PROJECT_ENV_ID distinguishes remote from local.
    """
    run_context = get_or_create_run_context()

    # If service is None, we want to perform a platform API call.
    if service is None:
        if not run_context.platform_config:
            raise ValueError("No platform config available for platform API calls.")
        platform_config = run_context.platform_config
        url = f"https://{platform_config.host}/{path}"
        headers = {
            **headers,
            platform_config.auth.header_key: platform_config.auth.header_value,
        }
        return url, headers

    # We want to resolve a call to a service in the project environment.
    env_id = os.environ.get("TIMBAL_PROJECT_ENV_ID")

    # If TIMBAL_PROJECT_ENV_ID is set, all services live behind a shared gateway.
    if env_id:
        if not run_context.platform_config:
            raise ValueError("No platform config available for platform API calls.")
        platform_config = run_context.platform_config
        # e{env_id} is the canonical public host for a project env. TIMBAL_PROJECT_ENV_ORIGIN
        # (full origin) or TIMBAL_DEPLOYMENTS_DOMAIN let the platform override this at deploy
        # time so a future domain-scheme change doesn't require an SDK release.
        base = os.environ.get("TIMBAL_PROJECT_ENV_ORIGIN")
        if not base:
            domain = os.environ.get("TIMBAL_DEPLOYMENTS_DOMAIN", "deployments.timbal.ai")
            base = f"https://e{env_id}.{domain}"
        if service == "api":
            url = f"{base}/api/{path}"
        elif service == "ui":
            url = f"{base}/{path}"
        else:
            url = f"{base}/api/workforce/{service}/{path}"
        headers = {
            **headers,
            platform_config.auth.header_key: platform_config.auth.header_value,
        }
        return url, headers
    else:
        # Local dev — each service runs on its own port, so there's no shared gateway that needs path-based routing.
        if service == "api":
            port = os.environ.get("TIMBAL_START_API_PORT")
            if not port:
                raise ValueError("Cannot resolve service 'api': TIMBAL_START_API_PORT is not set.")
            url = f"http://localhost:{port}/{path}"
        elif service == "ui":
            port = os.environ.get("TIMBAL_START_UI_PORT")
            if not port:
                raise ValueError("Cannot resolve service 'ui': TIMBAL_START_UI_PORT is not set.")
            url = f"http://localhost:{port}/{path}"
        else:
            workforce = os.environ.get("TIMBAL_START_WORKFORCE")
            if not workforce:
                # TODO: Remove once the API blueprint reads TIMBAL_START_WORKFORCE.
                workforce = os.environ.get("TIMBAL_WORKFORCE")
            if not workforce:
                raise ValueError(f"Cannot resolve service '{service}': TIMBAL_START_WORKFORCE is not set.")
            members = dict(entry.split(":") for entry in workforce.split(","))
            if service not in members:
                raise ValueError(f"Cannot resolve service '{service}': not found in TIMBAL_START_WORKFORCE.")
            url = f"http://localhost:{members[service]}/{path}"
        # User might have api key based auth configured in the env
        if run_context.platform_config is not None:
            headers = {
                **headers,
                run_context.platform_config.auth.header_key: run_context.platform_config.auth.header_value,
            }
        return url, headers


async def _request(
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    max_retries: int = 3,
    backoff: Callable[[int], float] | None = None,
    timeout: httpx.Timeout | float | None = None,
    service: str | None = None,
) -> Any:
    """Utility function for making HTTP requests.

    ``backoff`` overrides the retry delay: called with the 0-based attempt
    index, returns seconds to wait (default: 0.1s doubling). A ``Retry-After``
    on 429 still wins when longer. ``timeout`` overrides
    :func:`_default_timeout` (connect 10s, read/write unbounded; see
    ``TIMBAL_HTTP_*_TIMEOUT``).
    """
    url, headers = _resolve_url_and_headers(service, path, headers)
    if timeout is None:
        timeout = _default_timeout()
    elif isinstance(timeout, (int, float)):
        # Match httpx: a bare number is an all-phases timeout. Keep read
        # unbounded so long-running responses still work.
        timeout = httpx.Timeout(float(timeout), read=None)
    payload_kwargs = {}
    # `is not None` so an empty dict still sends a JSON body + Content-Type
    # (parameterless POSTs would otherwise 415 Unsupported Media Type).
    if json is not None:
        payload_kwargs["json"] = json
    elif content:
        payload_kwargs["content"] = content
    elif files:
        payload_kwargs["files"] = files

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    **payload_kwargs,
                )
                res.raise_for_status()
                return res
        except httpx.HTTPStatusError as exc:
            try:
                error_body = exc.response.json()
            except Exception:
                error_body = exc.response.text
            # Don't retry on client errors (4xx except 429)
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                raise PlatformError(
                    f"\n"
                    f"  URL: {exc.request.url}\n"
                    f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                    f"  Response body: {error_body or None}",
                    status_code=exc.response.status_code,
                ) from exc
            # Retry on 429, 5xx, or other errors
            if attempt == max_retries:
                raise PlatformError(
                    f"\n"
                    f"  URL: {exc.request.url}\n"
                    f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                    f"  Response body: {error_body or None}",
                    status_code=exc.response.status_code,
                ) from exc
            # Default exponential backoff: 100ms, 200ms, 400ms
            wait_time = backoff(attempt) if backoff is not None else 0.1 * (2**attempt)
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_time = max(wait_time, float(retry_after))
                    except ValueError:
                        pass
            logger.warning(
                f"Request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                status_code=exc.response.status_code,
            )
            await asyncio.sleep(wait_time)
        except httpx.TimeoutException as exc:
            if attempt == max_retries:
                raise PlatformError(_timeout_error_message(exc, url, timeout)) from exc
            wait_time = backoff(attempt) if backoff is not None else 0.1 * (2**attempt)
            logger.warning(
                f"Request timed out, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=type(exc).__name__,
            )
            await asyncio.sleep(wait_time)
        except Exception as exc:
            # Retry on any other error (network, etc.)
            if attempt == max_retries:
                raise
            wait_time = backoff(attempt) if backoff is not None else 0.1 * (2**attempt)
            logger.warning(
                f"Request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(exc),
            )
            await asyncio.sleep(wait_time)


async def _stream(
    method: Literal["GET", "POST"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    max_retries: int = 3,
    service: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Utility function for making streaming HTTP requests with SSE."""
    url, headers = _resolve_url_and_headers(service, path, headers)
    headers = {
        **headers,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    payload_kwargs = {}
    # `is not None` so an empty dict still sends a JSON body + Content-Type
    # (parameterless POSTs would otherwise 415 Unsupported Media Type).
    if json is not None:
        payload_kwargs["json"] = json
    elif content:
        payload_kwargs["content"] = content
    elif files:
        payload_kwargs["files"] = files

    timeout = _default_timeout()
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(method, url, headers=headers, params=params, **payload_kwargs) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        data_str = line[len("data:") :].strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            yield json_lib.loads(data_str)
                        except json_lib.JSONDecodeError:
                            logger.warning(f"Received non-JSON SSE data: {data_str}")
                            continue
                    return  # Successful completion

        except httpx.HTTPStatusError as exc:
            try:
                # Read the raw bytes first
                content = await exc.response.aread()
                try:
                    error_body = exc.response.json()
                except Exception:
                    error_body = content.decode(errors="replace")
            except Exception:
                error_body = None
            # Don't retry on client errors (4xx except 429)
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                raise PlatformError(
                    f"\n"
                    f"  URL: {exc.request.url}\n"
                    f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                    f"  Response body: {error_body or None}",
                    status_code=exc.response.status_code,
                ) from exc
            # Retry on 429, 5xx, or other errors
            if attempt == max_retries:
                raise PlatformError(
                    f"\n"
                    f"  URL: {exc.request.url}\n"
                    f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                    f"  Response body: {error_body or None}",
                    status_code=exc.response.status_code,
                ) from exc
            wait_time = 0.1 * (2**attempt)
            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_time = max(wait_time, float(retry_after))
                    except ValueError:
                        pass
            logger.warning(
                f"Stream request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                status_code=exc.response.status_code,
            )
            await asyncio.sleep(wait_time)
        except httpx.TimeoutException as exc:
            if attempt == max_retries:
                raise PlatformError(_timeout_error_message(exc, url, timeout)) from exc
            wait_time = 0.1 * (2**attempt)
            logger.warning(
                f"Stream request timed out, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=type(exc).__name__,
            )
            await asyncio.sleep(wait_time)
        except Exception as exc:
            # Retry on any other error (network, etc.)
            if attempt == max_retries:
                raise
            wait_time = 0.1 * (2**attempt)
            logger.warning(
                f"Stream request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(exc),
            )
            await asyncio.sleep(wait_time)
