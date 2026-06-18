"""Tests for Google Analytics tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools._google_oauth import resolve_google_token
from timbal.tools.google_analytics import (
    GoogleAnalyticsListAccountSummaries,
    GoogleAnalyticsRunReport,
    _normalize_property_id,
)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def test_normalize_property_id():
    assert _normalize_property_id("123456789") == "properties/123456789"
    assert _normalize_property_id("properties/123456789") == "properties/123456789"


@pytest.mark.asyncio
async def test_resolve_google_token_uses_explicit_token():
    token = await resolve_google_token(
        provider_name="Google Analytics",
        integration=None,
        token=SecretStr("access-token"),
        refresh_token_env="GOOGLE_ANALYTICS_REFRESH_TOKEN",
    )
    assert token == "access-token"


@pytest.mark.asyncio
async def test_google_analytics_list_account_summaries():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"accountSummaries": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleAnalyticsListAccountSummaries(token=SecretStr("ga-token"))
        out = await tool.handler(page_size=200, page_token=None)

    assert out == {"accountSummaries": []}
    mock_client.get.assert_awaited_once()
    assert mock_client.get.await_args.kwargs["headers"]["Authorization"] == "Bearer ga-token"


@pytest.mark.asyncio
async def test_google_analytics_run_report(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_ANALYTICS_PROPERTY_ID", "999")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"rows": []}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleAnalyticsRunReport(token=SecretStr("ga-token"))
        out = await tool.handler(
            property_id=None,
            start_date="7daysAgo",
            end_date="today",
            dimensions=["country"],
            metrics=["activeUsers"],
            limit=100,
            offset=0,
            dimension_filter=None,
            metric_filter=None,
            order_bys=None,
        )

    assert out == {"rows": []}
    url = mock_client.post.await_args[0][0]
    assert url.endswith("/properties/999:runReport")
    body = mock_client.post.await_args.kwargs["json"]
    assert body["metrics"] == [{"name": "activeUsers"}]


@pytest.mark.asyncio
async def test_resolve_google_token_refreshes_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GOOGLE_ANALYTICS_REFRESH_TOKEN", "refresh-token")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"access_token": "fresh-token"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        token = await resolve_google_token(
            provider_name="Google Analytics",
            integration=None,
            token=None,
            refresh_token_env="GOOGLE_ANALYTICS_REFRESH_TOKEN",
        )

    assert token == "fresh-token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_analytics_integration_list_accounts():
    if not (
        os.getenv("GOOGLE_CLIENT_ID")
        and os.getenv("GOOGLE_CLIENT_SECRET")
        and (os.getenv("GOOGLE_ANALYTICS_REFRESH_TOKEN") or os.getenv("GOOGLE_REFRESH_TOKEN"))
    ):
        pytest.skip("GA4 integration: set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and refresh token env vars.")

    tool = GoogleAnalyticsListAccountSummaries()
    result = await tool(page_size=10).collect()
    assert result.error is None
    assert "accountSummaries" in result.output
