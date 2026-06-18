"""Tests for Google Search Console tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.google_search_console import (
    GoogleSearchConsoleListSites,
    GoogleSearchConsoleSearchAnalytics,
    _encoded_site_url,
)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def test_encoded_site_url():
    assert _encoded_site_url("https://example.com/") == "https%3A%2F%2Fexample.com%2F"


@pytest.mark.asyncio
async def test_google_search_console_list_sites():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"siteEntry": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleSearchConsoleListSites(token=SecretStr("gsc-token"))
        out = await tool.handler()

    assert out == {"siteEntry": []}
    assert mock_client.get.await_args[0][0].endswith("/sites")


@pytest.mark.asyncio
async def test_google_search_console_search_analytics(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_SEARCH_CONSOLE_SITE_URL", "https://example.com/")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"rows": []}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleSearchConsoleSearchAnalytics(token=SecretStr("gsc-token"))
        out = await tool.handler(
            site_url=None,
            start_date="2026-01-01",
            end_date="2026-01-31",
            dimensions=["query"],
            search_type="web",
            row_limit=100,
            start_row=0,
            dimension_filter_groups=None,
            aggregation_type=None,
            data_state=None,
        )

    assert out == {"rows": []}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_search_console_integration_list_sites():
    token = os.getenv("GOOGLE_SEARCH_CONSOLE_ACCESS_TOKEN")
    if not token:
        pytest.skip("GSC integration: set GOOGLE_SEARCH_CONSOLE_ACCESS_TOKEN or configure a platform integration.")

    tool = GoogleSearchConsoleListSites(token=SecretStr(token))
    result = await tool().collect()
    assert result.error is None
