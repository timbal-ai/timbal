"""Tests for Google Analytics tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.google_analytics import (
    GoogleAnalyticsListAccountSummaries,
    GoogleAnalyticsRunReport,
    _normalize_property_id,
    _resolve_token,
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
async def test_resolve_token_uses_explicit_token():
    tool = GoogleAnalyticsListAccountSummaries(token=SecretStr("access-token"))
    assert await _resolve_token(tool) == "access-token"


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
    assert mock_client.post.await_args[0][0].endswith("/properties/999:runReport")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_analytics_integration_list_accounts():
    token = os.getenv("GOOGLE_ANALYTICS_ACCESS_TOKEN")
    if not token:
        pytest.skip("GA4 integration: set GOOGLE_ANALYTICS_ACCESS_TOKEN or configure a platform integration.")

    tool = GoogleAnalyticsListAccountSummaries(token=SecretStr(token))
    result = await tool(page_size=10).collect()
    assert result.error is None
    assert "accountSummaries" in result.output
