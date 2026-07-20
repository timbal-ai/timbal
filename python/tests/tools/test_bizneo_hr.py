"""Tests for Bizneo HR integration tools (mocked; no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from timbal.platform.integrations import Integration
from timbal.tools.bizneo_hr import (
    BizneoHRListJobs,
    BizneoHRRequest,
    _auth_headers,
    _bizneo_request,
    _join_path,
    _normalize_base_url,
    _resolve_credentials,
)


class TestBizneoHRHelpers:
    def test_normalize_base_url_strips_trailing_slash(self) -> None:
        assert _normalize_base_url("https://acme.bizneohr.com/") == "https://acme.bizneohr.com"

    def test_join_path_adds_api_prefix(self) -> None:
        base = "https://acme.bizneohr.com"
        assert _join_path(base, "jobs") == "https://acme.bizneohr.com/api/v1/jobs"
        assert _join_path(base, "/jobs/1") == "https://acme.bizneohr.com/api/v1/jobs/1"
        assert _join_path(base, "/api/v1/jobs") == "https://acme.bizneohr.com/api/v1/jobs"

    def test_auth_headers_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BIZNEO_API_KEY_HEADER", raising=False)
        monkeypatch.delenv("BIZNEO_API_SECRET_HEADER", raising=False)
        headers = _auth_headers("key-id", "secret")
        assert headers["X-API-KEY"] == "key-id"
        assert headers["X-API-SECRET"] == "secret"
        assert headers["Accept"] == "application/json"

    def test_auth_headers_custom_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIZNEO_API_KEY_HEADER", "Api-Key")
        monkeypatch.setenv("BIZNEO_API_SECRET_HEADER", "Api-Secret")
        headers = _auth_headers("k", "s")
        assert headers["Api-Key"] == "k"
        assert headers["Api-Secret"] == "s"


class TestBizneoHRPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import BizneoHRListJobs as exported

        assert exported is BizneoHRListJobs


class TestBizneoHRCredentials:
    @pytest.mark.asyncio
    async def test_resolve_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIZNEO_BASE_URL", "https://tenant.bizneohr.com/")
        monkeypatch.setenv("BIZNEO_API_KEY", "key-id")
        monkeypatch.setenv("BIZNEO_API_SECRET", "secret")
        tool = BizneoHRListJobs()
        base, key, secret = await _resolve_credentials(tool)
        assert base == "https://tenant.bizneohr.com"
        assert key == "key-id"
        assert secret == "secret"

    @pytest.mark.asyncio
    async def test_resolve_from_explicit_fields(self) -> None:
        tool = BizneoHRListJobs(
            base_url="https://co.bizneohr.com",
            api_key=SecretStr("explicit-key"),
            api_secret=SecretStr("explicit-secret"),
        )
        base, key, secret = await _resolve_credentials(tool)
        assert base == "https://co.bizneohr.com"
        assert key == "explicit-key"
        assert secret == "explicit-secret"

    @pytest.mark.asyncio
    async def test_resolve_from_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool = BizneoHRListJobs(integration=Integration("bizneo_hr", "int-1"))

        async def fake_resolve() -> dict[str, str]:
            return {
                "base_url": "https://platform.bizneohr.com",
                "api_key": "platform-key",
                "api_secret": "platform-secret",
            }

        monkeypatch.setattr(tool.integration, "resolve", fake_resolve)
        base, key, secret = await _resolve_credentials(tool)
        assert base == "https://platform.bizneohr.com"
        assert key == "platform-key"
        assert secret == "platform-secret"

    @pytest.mark.asyncio
    async def test_resolve_missing_raises(self) -> None:
        tool = BizneoHRListJobs()
        with pytest.raises(ValueError, match="Bizneo HR credentials not found"):
            await _resolve_credentials(tool)


@pytest.mark.asyncio
async def test_bizneo_request_builds_url_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZNEO_BASE_URL", "https://tenant.bizneohr.com")
    monkeypatch.setenv("BIZNEO_API_KEY", "kid")
    monkeypatch.setenv("BIZNEO_API_SECRET", "sec")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"items":[]}'
    mock_response.json.return_value = {"items": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    tool = BizneoHRListJobs()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _bizneo_request(
            tool,
            method="GET",
            path="/jobs",
            params={"page": 2, "per_page": 10},
        )

    assert result == {"items": []}
    call_args = mock_client.request.call_args
    assert call_args.args[0] == "GET"
    assert call_args.args[1] == "https://tenant.bizneohr.com/api/v1/jobs"
    assert call_args.kwargs["params"] == {"page": 2, "per_page": 10}
    assert call_args.kwargs["headers"]["X-API-KEY"] == "kid"
    assert call_args.kwargs["headers"]["X-API-SECRET"] == "sec"


@pytest.mark.asyncio
async def test_bizneo_hr_list_jobs_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZNEO_BASE_URL", "https://tenant.bizneohr.com")
    monkeypatch.setenv("BIZNEO_API_KEY", "k")
    monkeypatch.setenv("BIZNEO_API_SECRET", "s")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert path == "/jobs"
        assert params == {"page": 1, "per_page": 25}
        assert json_body is None
        return {"jobs": [{"id": "1"}]}

    monkeypatch.setattr("timbal.tools.bizneo_hr._bizneo_request", fake_request)
    tool = BizneoHRListJobs()
    out = await tool(page=1, per_page=25).collect()
    assert out.output == {"jobs": [{"id": "1"}]}


@pytest.mark.asyncio
async def test_bizneo_hr_request_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZNEO_BASE_URL", "https://tenant.bizneohr.com")
    monkeypatch.setenv("BIZNEO_API_KEY", "k")
    monkeypatch.setenv("BIZNEO_API_SECRET", "s")

    async def fake_request(
        _tool: Any,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "POST"
        assert path == "custom"
        assert params == {"q": "x"}
        assert json_body == {"name": "test"}
        return {"ok": True}

    monkeypatch.setattr("timbal.tools.bizneo_hr._bizneo_request", fake_request)
    tool = BizneoHRRequest()
    out = await tool(method="POST", path="custom", query_params={"q": "x"}, body={"name": "test"}).collect()
    assert out.output == {"ok": True}
