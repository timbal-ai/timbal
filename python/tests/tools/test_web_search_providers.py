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


@pytest.mark.asyncio
async def test_web_search_google_get_passes_key_cx_q(monkeypatch):
    monkeypatch.delenv("GOOGLE_CUSTOM_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"items": [{"title": "x"}]}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(provider="google", api_key=SecretStr("k"), cx="cx-123")
        out = await tool.handler(query="hello", num=5, start=11, gl="us")

    assert out["items"][0]["title"] == "x"
    mock_client.get.assert_awaited_once()
    url = mock_client.get.await_args[0][0]
    assert url == "https://www.googleapis.com/customsearch/v1"
    params = mock_client.get.await_args.kwargs["params"]
    assert params["key"] == "k"
    assert params["cx"] == "cx-123"
    assert params["q"] == "hello"
    assert params["num"] == 5
    assert params["start"] == 11
    assert params["gl"] == "us"


@pytest.mark.asyncio
async def test_web_search_google_resolves_credentials_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "env-key")
    monkeypatch.setenv("GOOGLE_CSE_CX", "env-cx")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(provider="google")
        await tool.handler(query="q", num=10)

    params = mock_client.get.await_args.kwargs["params"]
    assert params["key"] == "env-key"
    assert params["cx"] == "env-cx"
    assert params["q"] == "q"
    assert params["num"] == 10


@pytest.mark.asyncio
async def test_web_search_scraperapi_get_passes_query_and_country():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"organic_results": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(provider="scraperapi", api_key=SecretStr("sa-key"))
        await tool.handler(query="best espresso", country_code="es", hl="es", start=10)

    mock_client.get.assert_awaited_once()
    url = mock_client.get.await_args[0][0]
    assert "/google/search" in url
    params = mock_client.get.await_args.kwargs["params"]
    assert params["api_key"] == "sa-key"
    assert params["query"] == "best espresso"
    assert params["country_code"] == "es"
    assert params["hl"] == "es"
    assert params["start"] == 10


@pytest.mark.asyncio
async def test_web_search_scraperapi_uses_fixed_user_location_country():
    """`user_location={'country': 'GB'}` at construction overrides per-call country_code default."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(
            provider="scraperapi",
            api_key=SecretStr("k"),
            user_location={"country": "GB"},
        )
        await tool.handler(query="q")

    params = mock_client.get.await_args.kwargs["params"]
    assert params["country_code"] == "GB"


@pytest.mark.asyncio
async def test_web_search_cala_posts_natural_language_query():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"answer": "..."}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = WebSearch(provider="cala", api_key=SecretStr("cala-key"))
        out = await tool.handler(query="what is the capital of Mali")

    assert out == {"answer": "..."}
    mock_client.post.assert_awaited_once()
    url = mock_client.post.await_args[0][0]
    assert url.endswith("/knowledge/search")
    headers = mock_client.post.await_args.kwargs["headers"]
    assert headers["x-api-key"] == "cala-key"
    body = mock_client.post.await_args.kwargs["json"]
    assert body == {"input": "what is the capital of Mali"}


# ---------------------------------------------------------------------------
# Schema-mode branching: provider-less vs provider-set
# ---------------------------------------------------------------------------


def test_websearch_native_mode_anthropic_schema_has_no_function_shape():
    tool = WebSearch()
    schema = tool.anthropic_schema
    assert schema["type"] == "web_search_20250305"
    assert "input_schema" not in schema


def test_websearch_native_mode_openai_chat_completions_schema_raises():
    tool = WebSearch()
    with pytest.raises(ValueError, match="not compatible"):
        _ = tool.openai_chat_completions_schema


def test_websearch_provider_mode_anthropic_schema_is_function_shape():
    tool = WebSearch(provider="tavily", api_key=SecretStr("k"))
    schema = tool.anthropic_schema
    assert schema["name"] == "web_search"
    assert "input_schema" in schema
    assert "type" not in schema or schema.get("type") != "web_search_20250305"


def test_websearch_provider_mode_openai_chat_completions_schema_is_function_shape():
    tool = WebSearch(provider="tavily", api_key=SecretStr("k"))
    schema = tool.openai_chat_completions_schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "parameters" in schema["function"]


def test_websearch_provider_mode_openai_responses_schema_is_function_shape():
    tool = WebSearch(provider="firecrawl", api_key=SecretStr("k"))
    schema = tool.openai_responses_schema
    assert schema["type"] == "function"
    assert schema["name"] == "web_search"
    assert "parameters" in schema
