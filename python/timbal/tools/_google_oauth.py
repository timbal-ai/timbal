"""Shared Google OAuth credential resolution for marketing/product tools.

Resolution order:
1. ``Integration.resolve()`` → ``token`` or ``access_token``
2. Explicit ``token`` argument (SecretStr)
3. Product-specific access token env var (short-lived, dev/testing)
4. Refresh token exchange via ``GOOGLE_CLIENT_ID`` + ``GOOGLE_CLIENT_SECRET`` +
   product-specific refresh env var (or global ``GOOGLE_REFRESH_TOKEN``)
"""

import os
import time
from typing import Any

from pydantic import SecretStr

from ..platform.integrations import Integration

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_TOKEN_CACHE_TTL_SECONDS = 3300  # refresh before typical 3600s expiry


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def config_fields(tool: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "integration": tool.integration,
        "token": tool.token,
    }
    if extra:
        fields.update(extra)
    return tool._annotate_config(fields)


async def _refresh_access_token(*, client_id: str, client_secret: str, refresh_token: str) -> str:
    cache_key = (client_id, refresh_token)
    cached = _TOKEN_CACHE.get(cache_key)
    now = time.time()
    if cached and cached[1] > now:
        return cached[0]

    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            },
        )
        if not response.is_success:
            detail = response.text
            try:
                body = response.json()
                detail = body.get("error_description") or body.get("error") or detail
            except Exception:
                pass
            hint = ""
            if "unauthorized_client" in detail or "invalid_grant" in detail:
                hint = (
                    " If you used OAuth Playground, open the gear icon and enable "
                    "'Use your own OAuth credentials' with the same GOOGLE_CLIENT_ID/SECRET as .env."
                )
            raise ValueError(f"Google OAuth refresh failed ({response.status_code}): {detail}.{hint}") from None
        access_token = response.json()["access_token"]

    _TOKEN_CACHE[cache_key] = (access_token, now + _TOKEN_CACHE_TTL_SECONDS)
    return access_token


def _credential_from_integration(credentials: dict[str, Any]) -> str | None:
    for key in ("token", "access_token"):
        value = credentials.get(key)
        if value:
            return value
    return None


async def resolve_google_token(
    *,
    provider_name: str,
    integration: Any = None,
    token: SecretStr | None = None,
    refresh_token: SecretStr | None = None,
    refresh_token_env: str,
    access_token_env: str | None = None,
) -> str:
    """Resolve a Google OAuth access token for API calls."""
    if isinstance(integration, Integration):
        credentials = await integration.resolve()
        resolved = _credential_from_integration(credentials)
        if resolved:
            return resolved
        integration_refresh = credentials.get("refresh_token")
        if integration_refresh:
            client_id = credentials.get("client_id") or os.getenv("GOOGLE_CLIENT_ID")
            client_secret = credentials.get("client_secret") or os.getenv("GOOGLE_CLIENT_SECRET")
            if client_id and client_secret:
                return await _refresh_access_token(
                    client_id=client_id,
                    client_secret=client_secret,
                    refresh_token=integration_refresh,
                )

    if token is not None:
        return token.get_secret_value()

    if access_token_env:
        env_access = os.getenv(access_token_env)
        if env_access:
            return env_access

    if refresh_token is not None:
        env_refresh = refresh_token.get_secret_value()
    else:
        env_refresh = os.getenv(refresh_token_env) or os.getenv("GOOGLE_REFRESH_TOKEN")

    if env_refresh:
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(f"{provider_name} OAuth refresh requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
        return await _refresh_access_token(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=env_refresh,
        )

    raise ValueError(
        f"{provider_name} credentials not found. Configure an integration, pass token, "
        f"or set GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET with {refresh_token_env} "
        f"(or GOOGLE_REFRESH_TOKEN)."
    )
