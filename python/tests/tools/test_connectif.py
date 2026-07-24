"""Tests for Connectif integration tools (mocked; no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from timbal.platform.integrations import Integration
from timbal.tools.connectif import (
    ConnectifGetStore,
    ConnectifRequest,
    ConnectifUpsertContactByEmail,
    _auth_headers,
    _connectif_request,
    _join_url,
    _resolve_api_key,
)


class TestConnectifHelpers:
    def test_join_url(self) -> None:
        assert _join_url("/store/") == "https://api.connectif.cloud/store/"
        assert _join_url("contacts/email/a@b.com") == "https://api.connectif.cloud/contacts/email/a@b.com"

    def test_auth_headers(self) -> None:
        headers = _auth_headers("key-id:secret")
        assert headers["Authorization"] == "apiKey key-id:secret"
        assert headers["Accept"] == "application/json"


class TestConnectifPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import ConnectifGetStore as exported

        assert exported is ConnectifGetStore


class TestConnectifCredentials:
    @pytest.mark.asyncio
    async def test_resolve_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONNECTIF_API_KEY", "kid:sec")
        tool = ConnectifGetStore()
        assert await _resolve_api_key(tool) == "kid:sec"

    @pytest.mark.asyncio
    async def test_resolve_from_split_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONNECTIF_API_KEY_ID", "kid")
        monkeypatch.setenv("CONNECTIF_API_KEY_SECRET", "sec")
        tool = ConnectifGetStore()
        assert await _resolve_api_key(tool) == "kid:sec"

    @pytest.mark.asyncio
    async def test_resolve_from_explicit_field(self) -> None:
        tool = ConnectifGetStore(api_key=SecretStr("explicit:token"))
        assert await _resolve_api_key(tool) == "explicit:token"

    @pytest.mark.asyncio
    async def test_resolve_from_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool = ConnectifGetStore(integration=Integration("connectif", "int-1"))

        async def fake_resolve() -> dict[str, str]:
            return {"api_key": "platform:secret"}

        monkeypatch.setattr(tool.integration, "resolve", fake_resolve)
        assert await _resolve_api_key(tool) == "platform:secret"

    @pytest.mark.asyncio
    async def test_resolve_missing_raises(self) -> None:
        tool = ConnectifGetStore()
        with pytest.raises(ValueError, match="Connectif API key not found"):
            await _resolve_api_key(tool)


@pytest.mark.asyncio
async def test_connectif_request_builds_url_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTIF_API_KEY", "kid:sec")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"store":{}}'
    mock_response.json.return_value = {"store": {}}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    tool = ConnectifGetStore()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _connectif_request(tool, method="GET", path="/store/")

    assert result == {"store": {}}
    call_args = mock_client.request.call_args
    assert call_args.args[0] == "GET"
    assert call_args.args[1] == "https://api.connectif.cloud/store/"
    assert call_args.kwargs["headers"]["Authorization"] == "apiKey kid:sec"


@pytest.mark.asyncio
async def test_connectif_get_store_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTIF_API_KEY", "kid:sec")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert path == "/store/"
        return {"store": {"id": "1"}}

    monkeypatch.setattr("timbal.tools.connectif._connectif_request", fake_request)
    out = await ConnectifGetStore()().collect()
    assert out.output == {"store": {"id": "1"}}


@pytest.mark.asyncio
async def test_connectif_upsert_contact_encodes_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTIF_API_KEY", "kid:sec")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "PATCH"
        assert path == "/contacts/email/user%40example.com"
        assert json_body == {"_name": "Ada"}
        return {"contact": {"email": "user@example.com"}}

    monkeypatch.setattr("timbal.tools.connectif._connectif_request", fake_request)
    out = await ConnectifUpsertContactByEmail()(
        email="user@example.com",
        updates={"_name": "Ada"},
    ).collect()
    assert out.output["contact"]["email"] == "user@example.com"


@pytest.mark.asyncio
async def test_connectif_request_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTIF_API_KEY", "kid:sec")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "POST"
        assert path == "purchases/"
        assert params == {"triggerPurchaseEvent": False}
        assert json_body == {"purchaseId": "ORD-1"}
        return {"purchase": {"id": "p1"}}

    monkeypatch.setattr("timbal.tools.connectif._connectif_request", fake_request)
    out = await ConnectifRequest()(
        method="POST",
        path="purchases/",
        query_params={"triggerPurchaseEvent": False},
        body={"purchaseId": "ORD-1"},
    ).collect()
    assert out.output == {"purchase": {"id": "p1"}}
