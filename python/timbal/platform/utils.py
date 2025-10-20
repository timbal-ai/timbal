import asyncio
import json as json_lib
from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx
import structlog

from ..errors import PlatformError
from ..state import get_or_create_run_context

logger = structlog.get_logger("timbal.platform.utils")


async def _request(
    method: Literal["GET", "POST", "PATCH", "DELETE"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    max_retries: int = 3,
) -> Any:
    """Utility function for making platform API calls."""
    run_context = get_or_create_run_context()
    if not run_context.platform_config:
        raise ValueError("No platform config available for platform API calls.")
    platform_config = run_context.platform_config

    url = f"https://{platform_config.host}/{path}"
    headers = {
        **headers, 
        platform_config.auth.header_key: platform_config.auth.header_value,
    }
    payload_kwargs = {}
    if json:
        payload_kwargs["json"] = json
    elif content: 
        payload_kwargs["content"] = content
    elif files:
        payload_kwargs["files"] = files
   
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
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
                    f"  Response body: {error_body or None}"
                ) from exc
            # Retry on 429, 5xx, or other errors
            if attempt == max_retries:
                raise PlatformError(
                    f"\n"
                    f"  URL: {exc.request.url}\n"
                    f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                    f"  Response body: {error_body or None}"
                ) from exc
            wait_time = 0.1 * (2 ** attempt)  # Exponential backoff: 100ms, 200ms, 400ms
            logger.warning(
                f"Request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                status_code=exc.response.status_code,
            )
            await asyncio.sleep(wait_time)
        except Exception as exc:
            # Retry on any other error (network, timeout, etc.)
            if attempt == max_retries:
                raise
            wait_time = 0.1 * (2 ** attempt)
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
) -> AsyncGenerator[dict, None]:
    """Utility function for making streaming platform API calls and handling Server-Sent Events (SSE)."""
    run_context = get_or_create_run_context()
    if not run_context.platform_config:
        raise ValueError("No platform config available for platform API calls.")
    platform_config = run_context.platform_config

    url = f"https://{platform_config.host}/{path}"
    headers = {
        **headers,
        platform_config.auth.header_key: platform_config.auth.header_value,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    payload_kwargs = {}
    if json:
        payload_kwargs["json"] = json
    elif content:
        payload_kwargs["content"] = content
    elif files:
        payload_kwargs["files"] = files
    
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
                async with client.stream(method, url, headers=headers, params=params, **payload_kwargs) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        data_str = line[len("data:"):].strip()
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
                    f"  Response body: {error_body or None}"
                ) from exc
            # Retry on 429, 5xx, or other errors
            if attempt == max_retries:
                raise PlatformError(
                    f"\n"
                    f"  URL: {exc.request.url}\n"
                    f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                    f"  Response body: {error_body or None}"
                ) from exc
            wait_time = 0.1 * (2 ** attempt)
            logger.warning(
                f"Stream request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                status_code=exc.response.status_code,
            )
            await asyncio.sleep(wait_time)
        except Exception as exc:
            # Retry on any other error (network, timeout, etc.)
            if attempt == max_retries:
                raise
            wait_time = 0.1 * (2 ** attempt)
            logger.warning(
                f"Stream request failed, retrying in {wait_time:.1f}s",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(exc),
            )
            await asyncio.sleep(wait_time)
        