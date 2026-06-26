from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.platform.integrations import Integration
from timbal.tools._creds import resolve_api_key
from timbal.tools.happy_scribe import (
    HappyScribeListTranscriptions,
    _headers,
    _resolve_organization_id,
)


def test_headers():
    assert _headers("hs-key") == {
        "Authorization": "Bearer hs-key",
        "Content-Type": "application/json",
        "User-Agent": "timbal-happy-scribe-tools/1.0",
    }


@pytest.mark.asyncio
async def test_resolve_api_key_from_tool():
    key = await resolve_api_key(
        env_var="HAPPYSCRIBE_API_KEY",
        provider_name="Happy Scribe",
        integration=None,
        api_key=SecretStr("local-key"),
    )
    assert key == "local-key"


@pytest.mark.asyncio
async def test_resolve_api_key_from_integration():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "platform-key"})
    key = await resolve_api_key(
        env_var="HAPPYSCRIBE_API_KEY",
        provider_name="Happy Scribe",
        integration=integration,
        api_key=None,
    )
    assert key == "platform-key"


@pytest.mark.asyncio
async def test_resolve_organization_id_from_call():
    tool = SimpleNamespace(default_organization_id="99", integration=None)
    assert await _resolve_organization_id(tool, "123") == "123"


@pytest.mark.asyncio
async def test_resolve_organization_id_from_default():
    tool = SimpleNamespace(default_organization_id="456", integration=None)
    assert await _resolve_organization_id(tool, None) == "456"


@pytest.mark.asyncio
async def test_resolve_organization_id_from_integration():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"organization_id": 789})
    tool = SimpleNamespace(default_organization_id=None, integration=integration)
    assert await _resolve_organization_id(tool, None) == "789"


@pytest.mark.asyncio
async def test_resolve_organization_id_from_env(monkeypatch):
    monkeypatch.setenv("HAPPYSCRIBE_ORGANIZATION_ID", "env-org")
    tool = SimpleNamespace(default_organization_id=None, integration=None)
    assert await _resolve_organization_id(tool, None) == "env-org"


@pytest.mark.asyncio
async def test_resolve_organization_id_required_raises():
    tool = SimpleNamespace(default_organization_id=None, integration=None)
    with pytest.raises(ValueError, match="organization_id"):
        await _resolve_organization_id(tool, None)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_list_transcriptions_builds_query_params():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b'{"results": []}'
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = HappyScribeListTranscriptions(
            api_key=SecretStr("hs-key"),
            default_organization_id="42",
        )
        out = await tool.handler(
            organization_id=None,
            folder_id=7,
            page=1,
            per_page=10,
            tags=["todo"],
        )
        assert out == {"results": []}

    mock_client.request.assert_awaited_once()
    call_kwargs = mock_client.request.await_args.kwargs
    assert call_kwargs["params"] == [
        ("organization_id", "42"),
        ("folder_id", "7"),
        ("page", "1"),
        ("per_page", "10"),
        ("tags[]", "todo"),
    ]
