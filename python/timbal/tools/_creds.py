"""Shared credential resolution helper for HTTP-based tool providers.

Resolution order:
1. Explicit ``api_key`` argument (SecretStr).
2. ``Integration`` object — calls ``.resolve()`` and reads ``credentials["api_key"]``.
3. Environment variable named by ``env_var``.

Raises ``ValueError`` if none of the above yields a key. The ``provider_name``
argument is included in the error message for user-facing clarity.
"""

import os
from typing import Any

from pydantic import SecretStr

from ..platform.integrations import Integration


async def resolve_api_key(
    *,
    env_var: str,
    provider_name: str,
    integration: Any = None,
    api_key: SecretStr | None = None,
) -> str:
    """Resolve a provider API key from explicit arg, Integration, or env var (in that order)."""
    if isinstance(integration, Integration):
        credentials = await integration.resolve()
        return credentials["api_key"]
    if api_key is not None:
        return api_key.get_secret_value()
    env_key = os.getenv(env_var)
    if env_key:
        return env_key
    raise ValueError(
        f"{provider_name} API key not found. Pass `api_key` on the tool, "
        f"configure an Integration, or set the {env_var} environment variable."
    )
