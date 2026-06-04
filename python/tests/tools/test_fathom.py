from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from timbal.platform.integrations import Integration
from timbal.tools.fathom import _fathom_headers, _resolve_auth


def test_fathom_headers_api_key():
    assert _fathom_headers(kind="api_key", credential="key-123") == {"X-Api-Key": "key-123"}


def test_fathom_headers_bearer():
    assert _fathom_headers(kind="bearer", credential="oauth-tok") == {
        "Authorization": "Bearer oauth-tok",
    }


@pytest.mark.asyncio
async def test_resolve_auth_uses_bearer_for_integration_token():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"token": "platform-oauth-token"})
    tool = SimpleNamespace(integration=integration, api_key=None)

    assert await _resolve_auth(tool) == ("bearer", "platform-oauth-token")


@pytest.mark.asyncio
async def test_resolve_auth_uses_api_key_header_for_integration_api_key():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "fathom-key"})
    tool = SimpleNamespace(integration=integration, api_key=None)

    assert await _resolve_auth(tool) == ("api_key", "fathom-key")


@pytest.mark.asyncio
async def test_resolve_auth_prefers_api_key_when_both_present():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(
        return_value={"api_key": "fathom-key", "token": "oauth-tok"},
    )
    tool = SimpleNamespace(integration=integration, api_key=None)

    assert await _resolve_auth(tool) == ("api_key", "fathom-key")


@pytest.mark.asyncio
async def test_resolve_auth_uses_access_token_from_integration():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"access_token": "oauth-access"})
    tool = SimpleNamespace(integration=integration, api_key=None)

    assert await _resolve_auth(tool) == ("bearer", "oauth-access")


@pytest.mark.asyncio
async def test_resolve_auth_tool_access_token():
    tool = SimpleNamespace(integration=None, api_key=None, access_token=SecretStr("local-bearer"))

    assert await _resolve_auth(tool) == ("bearer", "local-bearer")


@pytest.mark.asyncio
async def test_resolve_auth_prefers_tool_api_key_over_access_token():
    tool = SimpleNamespace(
        integration=None,
        api_key=SecretStr("local-key"),
        access_token=SecretStr("local-bearer"),
    )

    assert await _resolve_auth(tool) == ("api_key", "local-key")


@pytest.mark.asyncio
async def test_resolve_auth_fathom_api_key_auth_env(monkeypatch):
    monkeypatch.delenv("FATHOM_API_KEY", raising=False)
    monkeypatch.setenv("FATHOM_API_KEY_AUTH", "env-key-auth")
    tool = SimpleNamespace(integration=None, api_key=None)

    assert await _resolve_auth(tool) == ("api_key", "env-key-auth")


@pytest.mark.asyncio
async def test_resolve_auth_fathom_bearer_auth_env(monkeypatch):
    monkeypatch.delenv("FATHOM_API_KEY", raising=False)
    monkeypatch.delenv("FATHOM_API_KEY_AUTH", raising=False)
    monkeypatch.setenv("FATHOM_BEARER_AUTH", "env-bearer")
    tool = SimpleNamespace(integration=None, api_key=None)

    assert await _resolve_auth(tool) == ("bearer", "env-bearer")


@pytest.mark.asyncio
async def test_resolve_auth_prefers_api_key_env_over_bearer_env(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "env-key")
    monkeypatch.setenv("FATHOM_BEARER_AUTH", "env-bearer")
    tool = SimpleNamespace(integration=None, api_key=None)

    assert await _resolve_auth(tool) == ("api_key", "env-key")
