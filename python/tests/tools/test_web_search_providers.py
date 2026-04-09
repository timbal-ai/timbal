"""Smoke tests for provider-backed WebSearch and Firecrawl search payloads (httpx mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.firecrawl import FirecrawlSearch
from timbal.tools.web_search import WebSearch


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_web_search_tavily_posts_search_payload():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"results": [{"title": "x"}]}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(provider="tavily", api_key=SecretStr("test-key"))
        out = await tool.handler(
            query="hello world",
            max_results=7,
            search_depth="basic",
            topic="general",
            time_range=None,
        )

    assert out == {"results": [{"title": "x"}]}
    mock_client.post.assert_awaited_once()
    url = mock_client.post.await_args[0][0]
    assert url.endswith("/search")
    body = mock_client.post.await_args.kwargs["json"]
    assert body["query"] == "hello world"
    assert body["max_results"] == 7


@pytest.mark.asyncio
async def test_web_search_firecrawl_posts_location_and_country():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": []}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(provider="firecrawl", api_key=SecretStr("fc-key"))
        await tool.handler(query="q", limit=3, location="Berlin", country="DE")

    body = mock_client.post.await_args.kwargs["json"]
    assert body["query"] == "q"
    assert body["limit"] == 3
    assert body["location"] == "Berlin"
    assert body["country"] == "DE"


@pytest.mark.asyncio
async def test_firecrawl_search_tool_sends_v2_location_fields():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = FirecrawlSearch(api_key=SecretStr("k"))
        await tool.handler(query="x", location="Germany", country="DE")

    body = mock_client.post.await_args.kwargs["json"]
    assert body["location"] == "Germany"
    assert body["country"] == "DE"
