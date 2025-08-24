import asyncio
import contextvars
import json as json_lib
from collections.abc import AsyncGenerator, Generator
from typing import Any, Literal

import httpx
import structlog
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .errors import PlatformError
from .state import resolve_platform_config

logger = structlog.get_logger("timbal.utils")


# TODO We might implement a decorator to wrap all handlers to handle this automatically
def resolve_default(key: str, value: Any) -> Any:
    """Resolve the default value of a field.
    Use this function to resolve default kwargs when calling a function that uses Field defaults.
    """
    if isinstance(value, FieldInfo):
        if value.default == PydanticUndefined:
            raise ValueError(f"{key} is required")
        return value.default
    return value


async def sync_to_async_gen(
    gen: Generator[Any, None, None], 
    loop: asyncio.AbstractEventLoop,
    ctx: contextvars.Context,
) -> AsyncGenerator[Any, None]:
    """Auxiliary function to convert a sync generator to an async generator.
    This function also shares the context of the caller to the executor.
    """
    while True:
        # StopIteration is special in Python. It's used to implement generator protocol and can't
        # be pickled/transferred across threads properly. By catching it explicitly in the executor 
        # function and converting it to a sentinel value, we avoid problematic exception propagation.
        def _next():
            try:
                return next(gen)
            except StopIteration: 
                return None
        value = await loop.run_in_executor(None, lambda: ctx.run(_next))
        if value is None:
            break
        yield value


async def _platform_api_call(
    method: Literal["GET", "POST", "PATCH", "DELETE"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
) -> Any:
    """Utility function for making platform API calls."""
    platform_config = resolve_platform_config()

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


async def _platform_api_stream_call(
    method: Literal["GET", "POST"],
    path: str,
    headers: dict[str, str] = {},
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    content: bytes | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
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
        