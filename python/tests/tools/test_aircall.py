"""Tests for Aircall integration tools (mocked; no network)."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from timbal.platform.integrations import Integration
from timbal.tools.aircall import (
    AircallListCalls,
    AircallRequest,
    _aircall_request,
    _auth_headers,
    _basic_auth_headers,
    _join_path,
    _resolve_basic_credentials,
    _resolve_bearer_token,
)


class TestAircallHelpers:
    def test_join_path(self) -> None:
        assert _join_path("calls") == "https://api.aircall.io/v1/calls"
        assert _join_path("/calls/1") == "https://api.aircall.io/v1/calls/1"
        assert _join_path("/v1/ping") == "https://api.aircall.io/v1/ping"

    def test_basic_auth_headers(self) -> None:
        headers = _basic_auth_headers("api-id", "api-token")
        expected = base64.b64encode(b"api-id:api-token").decode("ascii")
        assert headers["Authorization"] == f"Basic {expected}"
        assert headers["Accept"] == "application/json"


class TestAircallPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import AircallListCalls as exported

        assert exported is AircallListCalls


class TestAircallAuth:
    @pytest.mark.asyncio
    async def test_resolve_basic_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIRCALL_API_ID", "id-123")
        monkeypatch.setenv("AIRCALL_API_TOKEN", "tok-456")
        tool = AircallListCalls()
        creds = await _resolve_basic_credentials(tool)
        assert creds == ("id-123", "tok-456")

    @pytest.mark.asyncio
    async def test_resolve_basic_from_explicit_fields(self) -> None:
        tool = AircallListCalls(api_id=SecretStr("explicit-id"), api_token=SecretStr("explicit-token"))
        creds = await _resolve_basic_credentials(tool)
        assert creds == ("explicit-id", "explicit-token")

    @pytest.mark.asyncio
    async def test_resolve_bearer_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIRCALL_ACCESS_TOKEN", "oauth-token")
        tool = AircallListCalls()
        assert await _resolve_bearer_token(tool) == "oauth-token"

    @pytest.mark.asyncio
    async def test_resolve_bearer_from_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool = AircallListCalls(integration=Integration("aircall", "int-1"))

        async def fake_resolve() -> dict[str, str]:
            return {"access_token": "platform-oauth"}

        monkeypatch.setattr(tool.integration, "resolve", fake_resolve)
        assert await _resolve_bearer_token(tool) == "platform-oauth"

    @pytest.mark.asyncio
    async def test_auth_headers_prefers_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIRCALL_API_ID", "id")
        monkeypatch.setenv("AIRCALL_API_TOKEN", "tok")
        monkeypatch.setenv("AIRCALL_ACCESS_TOKEN", "bearer-token")
        tool = AircallListCalls()
        headers = await _auth_headers(tool)
        assert headers["Authorization"] == "Bearer bearer-token"

    @pytest.mark.asyncio
    async def test_auth_headers_basic_when_no_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIRCALL_API_ID", "my-id")
        monkeypatch.setenv("AIRCALL_API_TOKEN", "my-tok")
        monkeypatch.delenv("AIRCALL_ACCESS_TOKEN", raising=False)
        tool = AircallListCalls()
        headers = await _auth_headers(tool)
        encoded = base64.b64encode(b"my-id:my-tok").decode("ascii")
        assert headers["Authorization"] == f"Basic {encoded}"

    @pytest.mark.asyncio
    async def test_auth_headers_missing_raises(self) -> None:
        tool = AircallListCalls()
        with pytest.raises(ValueError, match="Aircall credentials not found"):
            await _auth_headers(tool)


@pytest.mark.asyncio
async def test_aircall_request_builds_url_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRCALL_API_ID", "kid")
    monkeypatch.setenv("AIRCALL_API_TOKEN", "sec")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"calls":[]}'
    mock_response.json.return_value = {"calls": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    tool = AircallListCalls()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _aircall_request(tool, method="GET", path="/calls", params={"page": 2, "per_page": 10})

    assert result == {"calls": []}
    call_args = mock_client.request.call_args
    assert call_args.args[0] == "GET"
    assert call_args.args[1] == "https://api.aircall.io/v1/calls"
    assert call_args.kwargs["params"] == {"page": 2, "per_page": 10}
    encoded = base64.b64encode(b"kid:sec").decode("ascii")
    assert call_args.kwargs["headers"]["Authorization"] == f"Basic {encoded}"


@pytest.mark.asyncio
async def test_aircall_list_calls_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRCALL_API_ID", "id")
    monkeypatch.setenv("AIRCALL_API_TOKEN", "tok")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert path == "/calls"
        assert params == {"page": 1, "per_page": 25, "order": "desc"}
        assert json_body is None
        return {"calls": [{"id": 812}]}

    monkeypatch.setattr("timbal.tools.aircall._aircall_request", fake_request)
    tool = AircallListCalls()
    out = await tool(page=1, per_page=25, order="desc").collect()
    assert out.output == {"calls": [{"id": 812}]}


@pytest.mark.asyncio
async def test_aircall_request_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRCALL_API_ID", "id")
    monkeypatch.setenv("AIRCALL_API_TOKEN", "tok")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "POST"
        assert path == "contacts"
        assert params == {"page": 1}
        assert json_body == {"first_name": "Ada"}
        return {"contact": {"id": 1}}

    monkeypatch.setattr("timbal.tools.aircall._aircall_request", fake_request)
    tool = AircallRequest()
    out = await tool(
        method="POST",
        path="contacts",
        query_params={"page": 1},
        body={"first_name": "Ada"},
    ).collect()
    assert out.output == {"contact": {"id": 1}}
