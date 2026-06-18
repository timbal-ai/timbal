"""Tests for Google Business Profile tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.google_business import (
    GoogleBusinessListAccounts,
    GoogleBusinessListLocations,
    _normalize_account_id,
)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def test_normalize_account_id():
    assert _normalize_account_id("123") == "accounts/123"


@pytest.mark.asyncio
async def test_google_business_list_accounts():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"accounts": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleBusinessListAccounts(token=SecretStr("business-token"))
        out = await tool.handler(page_size=20, page_token=None)

    assert out == {"accounts": []}


@pytest.mark.asyncio
async def test_google_business_list_locations(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_BUSINESS_ACCOUNT_ID", "accounts/111")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"locations": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleBusinessListLocations(token=SecretStr("business-token"))
        out = await tool.handler(
            account_id=None,
            page_size=100,
            page_token=None,
            read_mask="name,title",
        )

    assert out == {"locations": []}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_business_integration_list_accounts():
    token = os.getenv("GOOGLE_BUSINESS_ACCESS_TOKEN")
    if not token:
        pytest.skip("Business integration: set GOOGLE_BUSINESS_ACCESS_TOKEN or configure a platform integration.")

    tool = GoogleBusinessListAccounts(token=SecretStr(token))
    result = await tool(page_size=10).collect()
    assert result.error is None
