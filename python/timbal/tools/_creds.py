"""Credential resolution for tools.

Per credential, first match wins: explicit value on the tool → Integration
credentials → env var. Raises CredentialNotAvailable when nothing resolves.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import SecretStr

from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration

__all__ = [
    "CredentialNotAvailable",
    "resolve_api_key",
]


def _secret_value(value: Any) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return str(value)


def _first_present(credentials: Any, keys: Sequence[str]) -> Any:
    """First truthy value for `keys` in a credentials dict or object."""
    if isinstance(credentials, Mapping):
        for key in keys:
            value = credentials.get(key)
            if value:
                return value
        return None
    for key in keys:
        value = getattr(credentials, key, None)
        if value:
            return value
    return None


async def resolve_api_key(
    *,
    provider_name: str,
    env_var: str | None = None,
    integration: Any = None,
    api_key: SecretStr | str | None = None,
    tool: Any = None,
    explicit_attr: str = "api_key",
    integration_keys: Sequence[str] = ("api_key",),
) -> str:
    """Resolve a single provider secret.

    Pass `tool=self` to pull `integration` / `explicit_attr` off it, or pass
    `integration` / `api_key` explicitly. `integration_keys` are the keys tried
    in the resolved integration credentials.
    """
    if tool is not None:
        if integration is None:
            integration = getattr(tool, "integration", None)
        if api_key is None:
            api_key = getattr(tool, explicit_attr, None)

    if api_key is not None:
        return _secret_value(api_key)

    if isinstance(integration, Integration):
        credentials = await integration.resolve()
        value = _first_present(credentials, integration_keys)
        if value is not None:
            return str(value)

    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

    raise CredentialNotAvailable(
        provider_name,
        missing=[explicit_attr],
        env_vars=[env_var] if env_var else [],
    )
