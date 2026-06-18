"""Tests for Google Merchant Center tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.google_merchant_center import (
    GoogleMerchantListAccounts,
    GoogleMerchantListProducts,
)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_google_merchant_list_accounts():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"resources": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleMerchantListAccounts(token=SecretStr("merchant-token"))
        out = await tool.handler(max_results=250, page_token=None)

    assert out == {"resources": []}
    assert mock_client.get.await_args[0][0].endswith("/accounts")


@pytest.mark.asyncio
async def test_google_merchant_list_products(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_MERCHANT_ID", "123456789")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"resources": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleMerchantListProducts(token=SecretStr("merchant-token"))
        out = await tool.handler(merchant_id=None, max_results=250, page_token=None)

    assert out == {"resources": []}
    assert "/123456789/products" in mock_client.get.await_args[0][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_merchant_integration_list_accounts():
    token = os.getenv("GOOGLE_MERCHANT_CENTER_ACCESS_TOKEN")
    if not token:
        pytest.skip("Merchant integration: set GOOGLE_MERCHANT_CENTER_ACCESS_TOKEN or configure a platform integration.")

    tool = GoogleMerchantListAccounts(token=SecretStr(token))
    result = await tool(max_results=10).collect()
    assert result.error is None
