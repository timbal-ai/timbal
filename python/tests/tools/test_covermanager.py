"""Tests for CoverManager integration tools (mocked; no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from timbal.platform.integrations import Integration
from timbal.tools.covermanager import (
    CoverManagerGetReservs,
    CoverManagerRequest,
    CoverManagerReserv,
    _covermanager_request,
    _format_path,
    _resolve_api_key,
)


class TestCoverManagerHelpers:
    def test_format_path_injects_api_key_and_segments(self) -> None:
        path = _format_path(
            "/api/restaurant/get_reservs/{api_key}/{restaurant}/{date_start}/{date_end}/{page}/{table}",
            "my-key",
            {
                "restaurant": "casa-carlos",
                "date_start": "2026-03-01",
                "date_end": "2026-03-02",
                "page": "0",
                "table": "0",
            },
        )
        assert path.startswith("/api/restaurant/get_reservs/my-key/casa-carlos/")


class TestCoverManagerPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import CoverManagerRequest as exported

        assert exported is CoverManagerRequest


class TestCoverManagerCredentials:
    @pytest.mark.asyncio
    async def test_resolve_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COVERMANAGER_API_KEY", "api-key-123")
        tool = CoverManagerRequest()
        assert await _resolve_api_key(tool) == "api-key-123"

    @pytest.mark.asyncio
    async def test_resolve_from_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool = CoverManagerRequest(integration=Integration("covermanager", "int-1"))

        async def fake_resolve() -> dict[str, str]:
            return {"api_key": "platform-key"}

        monkeypatch.setattr(tool.integration, "resolve", fake_resolve)
        assert await _resolve_api_key(tool) == "platform-key"

    @pytest.mark.asyncio
    async def test_resolve_missing_raises(self) -> None:
        tool = CoverManagerRequest()
        with pytest.raises(ValueError, match="CoverManager API key not found"):
            await _resolve_api_key(tool)


@pytest.mark.asyncio
async def test_covermanager_request_post_sends_apikey_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERMANAGER_API_KEY", "kid")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"resp":1}'
    mock_response.json.return_value = {"resp": 1}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    tool = CoverManagerReserv()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _covermanager_request(
            tool,
            method="POST",
            path="/api/reserv/reserv",
            json_body={"restaurant": "casa-carlos"},
        )

    assert result == {"resp": 1}
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["headers"]["apikey"] == "kid"
    assert call_kwargs["json"] == {"restaurant": "casa-carlos"}


@pytest.mark.asyncio
async def test_covermanager_get_reservs_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERMANAGER_API_KEY", "kid")
    monkeypatch.setenv("COVERMANAGER_RESTAURANT_SLUG", "casa-carlos")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert "/api/restaurant/get_reservs/kid/casa-carlos/" in path
        return {"reservs": []}

    monkeypatch.setattr("timbal.tools.covermanager._covermanager_request", fake_request)
    out = await CoverManagerGetReservs()(
        date_start="2026-03-01",
        date_end="2026-03-01",
        page="0",
        table="0",
    ).collect()
    assert out.output == {"reservs": []}


@pytest.mark.asyncio
async def test_covermanager_request_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERMANAGER_API_KEY", "kid")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "POST"
        assert path == "/api/reserv/availability"
        assert json_body == {"restaurant": "casa-carlos", "date": "2026-03-01"}
        return {"available": True}

    monkeypatch.setattr("timbal.tools.covermanager._covermanager_request", fake_request)
    out = await CoverManagerRequest()(
        method="POST",
        path="/api/reserv/availability",
        body={"restaurant": "casa-carlos", "date": "2026-03-01"},
    ).collect()
    assert out.output == {"available": True}
