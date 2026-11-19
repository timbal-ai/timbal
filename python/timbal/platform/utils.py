import asyncio
import concurrent.futures
import contextvars
import json as json_lib
import os
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal

import httpx
import structlog

from ..errors import PlatformError
from ..state import get_or_create_run_context
from ._http_session import request_client

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


class _TotalTimeoutExceeded(Exception):
    """A request was still in flight at its ``total_timeout`` deadline."""


class _RetryBudget:
    """Deadline for one call and all its retries (``total_timeout``).

    An attempt is cancelled at the deadline (:meth:`run`), whatever the phase it
    is in: httpx timeouts are per phase and per read, so a response dripping in
    under the read timeout would otherwise run on. No retry starts when its
    wait would end past the deadline (:meth:`exhausted_by`).
    """

    __slots__ = ("deadline",)

    def __init__(self, total: float) -> None:
        self.deadline = time.monotonic() + total

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def exhausted_by(self, wait: float) -> bool:
        return self.left() <= wait

    async def run(self, attempt: Callable[[], Awaitable[httpx.Response]]) -> httpx.Response:
        timeout = asyncio.timeout(self.left())
        try:
            async with timeout:
                return await attempt()
        except TimeoutError as exc:
            if not timeout.expired():
                raise  # the attempt's own TimeoutError, not this deadline
            raise _TotalTimeoutExceeded from exc


def _total_timeout_error(url: str, total_timeout: float) -> PlatformError:
    return PlatformError(
        f"\n"
        f"  URL: {url}\n"
        f"  Error: no response within the call's total timeout (total_timeout={total_timeout:g}s, retries included)"
    )


def _status_error(exc: httpx.HTTPStatusError) -> PlatformError:
    """PlatformError for a failed platform response."""
    try:
        error_body = exc.response.json()
    except Exception:
        error_body = exc.response.text
    return PlatformError(
        f"\n"
        f"  URL: {exc.request.url}\n"
        f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
        f"  Response body: {error_body or None}",
        status_code=exc.response.status_code,
    )


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
    total_timeout: float | None = None,
) -> Any:
    """Utility function for making HTTP requests.

    ``backoff`` overrides the retry delay: called with the 0-based attempt
    index, returns seconds to wait (default: 0.1s doubling). A ``Retry-After``
    on 429 still wins when longer. ``timeout`` overrides
    :func:`_default_timeout` (connect 10s, read/write unbounded; see
    ``TIMBAL_HTTP_*_TIMEOUT``). ``total_timeout`` is a deadline, in seconds,
    for the whole call, retries and their waits included. An attempt still in
    flight at the deadline is cancelled and the call raises
    :class:`PlatformError`. A retry whose wait would end past the deadline is
    not attempted: the call ends as it does after its last retry, raising that
    failure. ``timeout`` alone cannot give this bound: it applies per phase
    and per read, and again on every retry.
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
    budget = _RetryBudget(total_timeout) if total_timeout is not None else None

    async def send() -> httpx.Response:
        async with request_client(timeout) as client:
            res = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                **payload_kwargs,
            )
            res.raise_for_status()
            return res

    for attempt in range(max_retries + 1):
        try:
            if budget is None:
                return await send()
            return await budget.run(send)
        except _TotalTimeoutExceeded as exc:
            raise _total_timeout_error(url, total_timeout) from exc
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
            if budget is not None and budget.exhausted_by(wait_time):
                raise _status_error(exc) from exc
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
            if budget is not None and budget.exhausted_by(wait_time):
                raise PlatformError(_timeout_error_message(exc, url, timeout)) from exc
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
            if budget is not None and budget.exhausted_by(wait_time):
                raise
            logger.warning(
                f"Request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(exc),
            )
            await asyncio.sleep(wait_time)


def _request_sync(
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    max_retries: int = 3,
    timeout: httpx.Timeout | float | None = None,
    service: str | None = None,
    total_timeout: float | None = None,
) -> httpx.Response:
    """Blocking :func:`_request`, for code that cannot await (a lazy ``File`` read).

    Runs :func:`_request` on a private event loop in a helper thread and waits
    for it, so it is safe on a thread whose own loop is running (it never
    touches that loop) and auth, retries and errors are :func:`_request`'s. The
    helper thread runs in a copy of the caller's context (run context included).

    The calling thread is blocked until the call ends, and never longer than
    ``total_timeout``: the request is cancelled at that deadline, and the wait
    itself stops there too should the helper thread fail to finish in time.
    """
    done: concurrent.futures.Future[httpx.Response] = concurrent.futures.Future()

    async def call() -> None:
        try:
            res = await _request(
                method,
                path,
                headers=headers,
                params=params,
                json=json,
                max_retries=max_retries,
                timeout=timeout,
                service=service,
                total_timeout=total_timeout,
            )
        except BaseException as exc:
            done.set_exception(exc)
        else:
            done.set_result(res)

    def run() -> None:
        try:
            asyncio.run(call())
        except BaseException as exc:
            if not done.done():  # the loop itself failed, before or after the call
                done.set_exception(exc)

    # The result is published from inside the loop: closing the loop afterwards
    # can outlast the deadline (it joins asyncio's executor, e.g. a DNS lookup
    # still in flight), and the caller must not wait for that.
    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(run,), name="timbal-request-sync", daemon=True).start()
    finished, _ = concurrent.futures.wait([done], timeout=total_timeout)
    if not finished:
        url, _ = _resolve_url_and_headers(service, path, headers)
        raise _total_timeout_error(url, total_timeout)
    return done.result()


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
