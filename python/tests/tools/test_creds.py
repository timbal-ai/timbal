"""Tests for timbal.tools._creds.resolve_api_key."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.state.config_loader import FileConfig
from timbal.tools._creds import resolve_api_key


def _empty_file_config(*_args, **_kwargs) -> FileConfig:
    return FileConfig(None, None, None, None)


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch):
    monkeypatch.setattr(
        "timbal.state.config_loader.load_file_config",
        _empty_file_config,
    )
    monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
    monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
    for var in ("TAVILY_API_KEY", "SLACK_API_TOKEN", "TEST_API_KEY", "FIRECRAWL_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.asyncio
async def test_resolve_explicit_api_key_param():
    key = await resolve_api_key(
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
        api_key=SecretStr("explicit"),
    )
    assert key == "explicit"


@pytest.mark.asyncio
async def test_resolve_explicit_via_tool():
    tool = SimpleNamespace(
        integration=None,
        api_key=SecretStr("from-tool"),
    )
    key = await resolve_api_key(
        tool=tool,
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
    )
    assert key == "from-tool"


@pytest.mark.asyncio
async def test_resolve_explicit_attr_override():
    tool = SimpleNamespace(
        integration=None,
        api_token=SecretStr("slack-token"),
        api_key=None,
    )
    key = await resolve_api_key(
        tool=tool,
        provider_name="Slack",
        env_var="SLACK_API_TOKEN",
        explicit_attr="api_token",
        integration_keys=("api_token",),
    )
    assert key == "slack-token"


@pytest.mark.asyncio
async def test_resolve_from_integration():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "platform-key"})
    key = await resolve_api_key(
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
        integration=integration,
    )
    assert key == "platform-key"
    integration.resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_integration_keys_fallback():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"token": "mongo-token"})
    key = await resolve_api_key(
        provider_name="MongoDB",
        env_var="MONGODB_API_KEY",
        integration=integration,
        integration_keys=("api_key", "token"),
    )
    assert key == "mongo-token"


@pytest.mark.asyncio
async def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    key = await resolve_api_key(
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
    )
    assert key == "env-key"


@pytest.mark.asyncio
async def test_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    key = await resolve_api_key(
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
        api_key="explicit-wins",
    )
    assert key == "explicit-wins"


@pytest.mark.asyncio
async def test_integration_beats_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "integration-wins"})
    key = await resolve_api_key(
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
        integration=integration,
    )
    assert key == "integration-wins"


@pytest.mark.asyncio
async def test_credential_not_available_fields_and_message():
    with pytest.raises(CredentialNotAvailable) as exc_info:
        await resolve_api_key(
            provider_name="Firecrawl",
            env_var="FIRECRAWL_API_KEY",
            explicit_attr="api_key",
        )

    err = exc_info.value
    assert err.provider_name == "Firecrawl"
    assert err.missing == ["api_key"]
    assert err.env_vars == ["FIRECRAWL_API_KEY"]
    assert "Firecrawl credentials not found" in str(err)
    assert "FIRECRAWL_API_KEY" in str(err)


@pytest.mark.asyncio
async def test_credential_not_available_is_value_error():
    with pytest.raises(ValueError):
        await resolve_api_key(provider_name="Test", env_var="TEST_API_KEY")


@pytest.mark.asyncio
async def test_non_integration_object_skips_resolve(monkeypatch):
    """String integration ids are not resolved until pydantic validates to Integration."""
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    key = await resolve_api_key(
        provider_name="Tavily",
        env_var="TAVILY_API_KEY",
        integration="org-integration-id-string",
    )
    assert key == "env-key"
