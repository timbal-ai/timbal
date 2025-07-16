from typing import Any, Literal

import httpx

from .errors import PlatformError
from .state import resolve_platform_config


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
    
    async with httpx.AsyncClient() as client:
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
        