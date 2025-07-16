import json as json_lib
from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx
import structlog

from .errors import PlatformError
from .state import resolve_platform_config

logger = structlog.get_logger("timbal.utils")


async def _platform_api_call(
    method: Literal["GET", "POST", "PATCH", "DELETE"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
) -> Any:
    """Utility function for making platform API calls."""
    platform_config = resolve_platform_config()

    url = f"https://{platform_config.host}/{path}"
    headers = {
        **headers, 
        platform_config.auth.header_key: platform_config.auth.header_value,
    }
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
        try:
            res = await client.request(
                method, 
                url, 
                headers=headers, 
                params=params, 
                json=json,
                content=content,
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


async def _platform_api_stream_call(
    method: Literal["GET", "POST"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> AsyncGenerator[dict, None]:
    """Utility function for making streaming platform API calls and handling Server-Sent Events (SSE)."""
    platform_config = resolve_platform_config()

    url = f"https://{platform_config.host}/{path}"
    headers = {
        **headers,
        platform_config.auth.header_key: platform_config.auth.header_value,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
        try:
            async with client.stream(method, url, headers=headers, params=params, json=json) as response:
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
                error_body = exc.response.json()
            except Exception:
                error_body = exc.response.text
            raise PlatformError(
                f"\n"
                f"  URL: {exc.request.url}\n"
                f"  Status: {exc.response.status_code} {exc.response.reason_phrase}\n"
                f"  Response body: {error_body or None}"
            ) from exc
        