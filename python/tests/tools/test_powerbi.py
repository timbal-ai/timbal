"""Tests for Power BI auth (user-delegated OAuth + app-delegated service principal)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.tools.powerbi import (
    PowerBIListWorkspaces,
    _raise_for_execute_queries,
    _resolve_token,
    _service_principal_parts,
)


def _tool(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "integration": None,
        "api_key": None,
        "token": None,
        "tenant_id": None,
        "client_id": None,
        "client_secret": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _clear_powerbi_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "POWERBI_API_KEY",
        "POWERBI_ACCESS_TOKEN",
        "POWERBI_TENANT_ID",
        "POWERBI_CLIENT_ID",
        "POWERBI_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.asyncio
async def test_explicit_api_key_still_works() -> None:
    tool = _tool(api_key=SecretStr("app-bearer"))
    assert await _resolve_token(tool) == "app-bearer"


@pytest.mark.asyncio
async def test_explicit_oauth_token() -> None:
    tool = _tool(token=SecretStr("user-oauth"))
    assert await _resolve_token(tool) == "user-oauth"


@pytest.mark.asyncio
async def test_explicit_api_key_beats_explicit_token() -> None:
    tool = _tool(api_key=SecretStr("app-bearer"), token=SecretStr("user-oauth"))
    assert await _resolve_token(tool) == "app-bearer"


@pytest.mark.asyncio
async def test_oauth_token_from_integration() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"token": "platform-user-token"})
    assert await _resolve_token(_tool(integration=integration)) == "platform-user-token"


@pytest.mark.asyncio
async def test_oauth_access_token_alias_from_integration() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"access_token": "platform-user-access"})
    assert await _resolve_token(_tool(integration=integration)) == "platform-user-access"


@pytest.mark.asyncio
async def test_api_key_from_integration_still_works() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "platform-app-key"})
    assert await _resolve_token(_tool(integration=integration)) == "platform-app-key"


@pytest.mark.asyncio
async def test_integration_oauth_beats_api_key_when_both_present() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"token": "user-oauth", "api_key": "app-key"})
    assert await _resolve_token(_tool(integration=integration)) == "user-oauth"


@pytest.mark.asyncio
async def test_service_principal_from_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(
        return_value={"tenant_id": "tid", "client_id": "cid", "client_secret": "csecret"},
    )
    monkeypatch.setattr(
        "timbal.tools.powerbi._get_token_from_client_credentials",
        AsyncMock(return_value="sp-access-token"),
    )
    assert await _resolve_token(_tool(integration=integration)) == "sp-access-token"


@pytest.mark.asyncio
async def test_service_principal_from_tool_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "timbal.tools.powerbi._get_token_from_client_credentials",
        AsyncMock(return_value="sp-access-token"),
    )
    tool = _tool(tenant_id="tid", client_id="cid", client_secret=SecretStr("csecret"))
    assert await _resolve_token(tool) == "sp-access-token"


@pytest.mark.asyncio
async def test_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POWERBI_API_KEY", "env-app-key")
    assert await _resolve_token(_tool()) == "env-app-key"


@pytest.mark.asyncio
async def test_env_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POWERBI_ACCESS_TOKEN", "env-user-token")
    assert await _resolve_token(_tool()) == "env-user-token"


@pytest.mark.asyncio
async def test_env_api_key_beats_env_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POWERBI_API_KEY", "env-app-key")
    monkeypatch.setenv("POWERBI_ACCESS_TOKEN", "env-user-token")
    assert await _resolve_token(_tool()) == "env-app-key"


def test_service_principal_parts_incomplete() -> None:
    assert _service_principal_parts(_tool(tenant_id="tid", client_id="cid"), {}) is None


@pytest.mark.asyncio
async def test_missing_credentials_raises() -> None:
    with pytest.raises(CredentialNotAvailable) as exc_info:
        await _resolve_token(_tool())
    assert exc_info.value.provider_name == "PowerBI"


def test_list_workspaces_config_exposes_both_auth_fields() -> None:
    config = PowerBIListWorkspaces().get_config()
    assert "api_key" in config
    assert "token" in config
    assert "tenant_id" in config
    assert "client_id" in config
    assert "client_secret" in config
    assert "integration" in config


def test_execute_queries_401_names_rls_limitation() -> None:
    response = SimpleNamespace(
        status_code=401,
        text='{"error":{"code":"PowerBINotAuthorizedException"}}',
        raise_for_status=lambda: None,
    )
    with pytest.raises(ValueError, match="user-delegated OAuth") as exc_info:
        _raise_for_execute_queries(response)
    assert "PowerBINotAuthorizedException" in str(exc_info.value)
    assert "impersonatedUserName" in str(exc_info.value)
