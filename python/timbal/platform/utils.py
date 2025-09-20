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
   
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
        try:
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
            raise PlatformError(
                f"\n"
                f"  URL: {exc.request.url}\n"
                f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                f"  Response body: {error_body or None}"
            ) from exc


async def _stream(
    method: Literal["GET", "POST"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
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
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
        try:
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
            raise PlatformError(
                f"\n"
                f"  URL: {exc.request.url}\n"
                f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                f"  Response body: {error_body or None}"
            ) from exc
        