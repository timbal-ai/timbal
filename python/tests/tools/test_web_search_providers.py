"""Smoke tests for provider-backed WebSearch and Firecrawl search payloads (httpx mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.tools.firecrawl import FirecrawlSearch
from timbal.tools.web_search import WebSearch
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


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
        await tool(query="x", location="Germany", country="DE").collect()

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
    assert body == {
        "input": "what is the capital of Mali",
        "explainability": True,
        "return_entities": True,
    }


# ---------------------------------------------------------------------------
# Schema-mode branching: provider-less vs provider-set
# ---------------------------------------------------------------------------


def test_websearch_native_mode_anthropic_schema_has_no_function_shape():
    tool = WebSearch()
    schema = tool.anthropic_schema
    assert schema["type"] == "web_search_20250305"
    assert "input_schema" not in schema


def test_websearch_native_mode_openai_chat_completions_schema_is_function_shape():
    schema = WebSearch().openai_chat_completions_schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert schema["function"]["description"]
    assert schema["function"]["parameters"]["required"] == ["query"]


def test_websearch_catalog_description_is_the_class_default():
    """Tool catalogs read the class field default, not an instance."""
    assert WebSearch.model_fields["description"].default
    assert WebSearch(provider="tavily", api_key=SecretStr("k")).description == WebSearch().description


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["timbal", "google", "groq"])
async def test_native_websearch_on_chat_completions_sends_a_function_tool(provider):
    """`timbal/auto` (and every Chat Completions provider) used to fail the whole turn here."""
    from timbal.core.llm.chat_completions import prepare_chat_completions_request
    from timbal.core.llm.registry import _PROVIDERS

    async def _no_chunks():
        return
        yield

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_no_chunks())
    create_stream, _ = prepare_chat_completions_request(
        provider=provider,
        config=_PROVIDERS[provider],
        client=client,
        model_name="auto",
        request_headers={},
        system_prompt=None,
        messages=[],
        tools=[WebSearch()],
        max_tokens=None,
        temperature=None,
        output_model=None,
        provider_params={},
    )
    async for _ in create_stream():
        pass

    (tool,) = client.chat.completions.create.await_args.kwargs["tools"]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"


def test_websearch_provider_mode_anthropic_schema_is_function_shape():
    tool = WebSearch(provider="tavily", api_key=SecretStr("k"))
    schema = tool.anthropic_schema
    assert schema["name"] == "tavily_web_search"
    assert "input_schema" in schema
    assert "type" not in schema or schema.get("type") != "web_search_20250305"


def test_websearch_provider_mode_openai_chat_completions_schema_is_function_shape():
    tool = WebSearch(provider="tavily", api_key=SecretStr("k"))
    schema = tool.openai_chat_completions_schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "tavily_web_search"
    assert "parameters" in schema["function"]


def test_websearch_provider_mode_openai_responses_schema_is_function_shape():
    tool = WebSearch(provider="firecrawl", api_key=SecretStr("k"))
    schema = tool.openai_responses_schema
    assert schema["type"] == "function"
    assert schema["name"] == "firecrawl_web_search"
    assert "parameters" in schema


def test_websearch_provider_name_is_prefixed_by_default():
    """Each provider gets a unique default name so multiple instances can coexist."""
    assert WebSearch(provider="tavily", api_key=SecretStr("k")).name == "tavily_web_search"
    assert WebSearch(provider="scraperapi", api_key=SecretStr("k")).name == "scraperapi_web_search"
    assert WebSearch(provider="cala", api_key=SecretStr("k")).name == "cala_web_search"
    assert WebSearch(provider="firecrawl", api_key=SecretStr("k")).name == "firecrawl_web_search"
    assert WebSearch(provider="google", api_key=SecretStr("k"), cx="cx").name == "google_web_search"
    # Native mode (Anthropic server-tool) must keep `web_search`.
    assert WebSearch().name == "web_search"
    # Explicit override still wins.
    assert WebSearch(provider="tavily", api_key=SecretStr("k"), name="custom").name == "custom"


# ---------------------------------------------------------------------------
# End-to-end Agent loop: TestModel emits a tool_use, the provider-set
# WebSearch handler executes against mocked httpx, agent loop completes.
# These guard the Tool wrapping / param validation / dispatch path that the
# isolated handler tests above don't exercise.
# ---------------------------------------------------------------------------


def _tool_use_response(name: str, input_: dict) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id="c1", name=name, input=input_)],
        stop_reason="tool_use",
    )


def _mock_post_client(json_payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = json_payload if json_payload is not None else {}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


def _mock_get_client(json_payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = json_payload if json_payload is not None else {}
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_agent_invokes_tavily_web_search_via_tool_use():
    mock_client = _mock_post_client({"results": [{"title": "x"}]})
    model = TestModel(responses=[
        _tool_use_response("tavily_web_search", {"query": "hi"}),
        "done.",
    ])
    agent = Agent(
        name="t",
        model=model,
        tools=[WebSearch(provider="tavily", api_key=SecretStr("k"))],
    )
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        result = await agent(prompt="search please").collect()

    assert result.status.code == "success"
    mock_client.post.assert_awaited_once()
    assert mock_client.post.await_args[0][0].endswith("/search")
    body = mock_client.post.await_args.kwargs["json"]
    assert body["query"] == "hi"


@pytest.mark.asyncio
async def test_agent_invokes_scraperapi_web_search_via_tool_use():
    mock_client = _mock_get_client({"organic_results": []})
    model = TestModel(responses=[
        _tool_use_response("scraperapi_web_search", {"query": "espresso"}),
        "done.",
    ])
    agent = Agent(
        name="t",
        model=model,
        tools=[WebSearch(provider="scraperapi", api_key=SecretStr("k"))],
    )
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        result = await agent(prompt="search please").collect()

    assert result.status.code == "success"
    mock_client.get.assert_awaited_once()
    assert "/google/search" in mock_client.get.await_args[0][0]
    assert mock_client.get.await_args.kwargs["params"]["query"] == "espresso"


@pytest.mark.asyncio
async def test_agent_invokes_cala_web_search_via_tool_use():
    mock_client = _mock_post_client({"answer": "Bamako"})
    model = TestModel(responses=[
        _tool_use_response("cala_web_search", {"query": "capital of Mali"}),
        "Bamako.",
    ])
    agent = Agent(
        name="t",
        model=model,
        tools=[WebSearch(provider="cala", api_key=SecretStr("k"))],
    )
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        result = await agent(prompt="ask cala").collect()

    assert result.status.code == "success"
    mock_client.post.assert_awaited_once()
    assert mock_client.post.await_args[0][0].endswith("/knowledge/search")
    assert mock_client.post.await_args.kwargs["json"] == {
        "input": "capital of Mali",
        "explainability": True,
        "return_entities": True,
    }


@pytest.mark.asyncio
async def test_agent_invokes_firecrawl_web_search_via_tool_use():
    mock_client = _mock_post_client({"data": []})
    model = TestModel(responses=[
        _tool_use_response("firecrawl_web_search", {"query": "berlin", "country": "DE"}),
        "done.",
    ])
    agent = Agent(
        name="t",
        model=model,
        tools=[WebSearch(provider="firecrawl", api_key=SecretStr("k"))],
    )
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        result = await agent(prompt="search please").collect()

    assert result.status.code == "success"
    mock_client.post.assert_awaited_once()
    body = mock_client.post.await_args.kwargs["json"]
    assert body["query"] == "berlin"
    assert body["country"] == "DE"


@pytest.mark.asyncio
async def test_agent_invokes_google_web_search_via_tool_use(monkeypatch):
    monkeypatch.delenv("GOOGLE_CUSTOM_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    mock_client = _mock_get_client({"items": [{"title": "x"}]})
    model = TestModel(responses=[
        _tool_use_response("google_web_search", {"query": "timbal"}),
        "done.",
    ])
    agent = Agent(
        name="t",
        model=model,
        tools=[WebSearch(provider="google", api_key=SecretStr("k"), cx="cx-1")],
    )
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        result = await agent(prompt="search please").collect()

    assert result.status.code == "success"
    mock_client.get.assert_awaited_once()
    params = mock_client.get.await_args.kwargs["params"]
    assert params["q"] == "timbal"
    assert params["key"] == "k"
    assert params["cx"] == "cx-1"


@pytest.mark.asyncio
async def test_two_provider_web_searches_coexist_in_one_agent(monkeypatch):
    """Two `WebSearch(provider=...)` instances must register and dispatch
    independently. Proves the provider-prefixed default name avoids
    collision when several WebSearch variants live in the same Agent."""
    monkeypatch.delenv("GOOGLE_CUSTOM_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)  # tavily uses post
    client.get = AsyncMock(return_value=response)  # google uses get

    model = TestModel(responses=[
        _tool_use_response("tavily_web_search", {"query": "from tavily"}),
        _tool_use_response("google_web_search", {"query": "from google"}),
        "done.",
    ])
    agent = Agent(
        name="t",
        model=model,
        tools=[
            WebSearch(provider="tavily", api_key=SecretStr("k")),
            WebSearch(provider="google", api_key=SecretStr("k"), cx="cx-1"),
        ],
    )

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(client)):
        result = await agent(prompt="run both").collect()

    assert result.status.code == "success"
    # Both tools were dispatched by name with no collision/hallucination error.
    client.post.assert_awaited_once()
    client.get.assert_awaited_once()
    assert client.post.await_args.kwargs["json"]["query"] == "from tavily"
    assert client.get.await_args.kwargs["params"]["q"] == "from google"


# ---------------------------------------------------------------------------
# Native mode on Chat Completions: the platform's funded search runs the call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_native_web_search_runs_through_platform_proxy():
    proxy = AsyncMock(return_value={"data": {"web": []}})
    model = TestModel(responses=[
        _tool_use_response("web_search", {"query": "timbal"}),
        "done.",
    ])
    agent = Agent(name="t", model=model, tools=[WebSearch()])
    with patch("timbal.tools.web_search.execute_tool_proxy", proxy):
        result = await agent(prompt="search please").collect()

    assert result.status.code == "success"
    proxy.assert_awaited_once_with("firecrawl_search", {"query": "timbal", "limit": 5})


@pytest.mark.asyncio
async def test_native_web_search_maps_native_options_onto_platform_search():
    proxy = AsyncMock(return_value={})
    tool = WebSearch(
        allowed_domains=["a.com", "b.com"],
        blocked_domains=["c.com"],
        user_location={"type": "approximate", "country": "DE"},
        max_results=3,
    )
    with patch("timbal.tools.web_search.execute_tool_proxy", proxy):
        await tool.handler(query="espresso", limit=10, tbs="qdr:w")

    proxy.assert_awaited_once_with(
        "firecrawl_search",
        {"query": "espresso", "limit": 3, "tbs": "qdr:w", "country": "DE", "include_domains": ["a.com", "b.com"]},
    )


@pytest.mark.asyncio
async def test_native_web_search_enforces_domains_on_results():
    """Firecrawl takes one list per request and older platforms drop both: results are filtered here too."""
    urls = [
        "https://a.com/1",
        "https://news.b.com/2",
        "https://x.a.com/3",
        "https://nota.com/4",
        "https://c.com/5",
    ]
    proxy = AsyncMock(return_value={"success": True, "data": {"web": [{"url": u} for u in urls], "news": []}})
    tool = WebSearch(allowed_domains=["a.com", "B.com"], blocked_domains=["x.a.com", "c.com"])
    with patch("timbal.tools.web_search.execute_tool_proxy", proxy):
        out = await tool.handler(query="q", limit=5, tbs=None)

    # One list per request: the allowlist, with the carve-out inside it excluded upstream.
    sent = proxy.await_args[0][1]
    assert sent["query"] == "q -site:x.a.com"
    assert sent["include_domains"] == ["a.com", "b.com"]
    assert "exclude_domains" not in sent
    assert [r["url"] for r in out["data"]["web"]] == ["https://a.com/1", "https://news.b.com/2"]
    assert out["data"]["news"] == []
    assert out["success"] is True


@pytest.mark.asyncio
async def test_native_web_search_blocklist_only_sends_exclude_domains():
    proxy = AsyncMock(return_value={"data": [{"url": "https://spam.com/x"}, {"url": "https://ok.com/y"}]})
    tool = WebSearch(blocked_domains=["spam.com"])
    with patch("timbal.tools.web_search.execute_tool_proxy", proxy):
        out = await tool.handler(query="q", limit=5, tbs=None)

    assert proxy.await_args[0][1]["exclude_domains"] == ["spam.com"]
    assert "include_domains" not in proxy.await_args[0][1]
    assert out["data"] == [{"url": "https://ok.com/y"}]


@pytest.mark.asyncio
async def test_firecrawl_search_sends_one_domain_list():
    web = [{"url": "https://a.com/1"}, {"url": "https://x.a.com/2"}, {"url": "https://c.com/3"}]
    mock_client = _mock_post_client({"data": {"web": web}})
    tool = FirecrawlSearch(api_key=SecretStr("k"))
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        out = await tool(query="q", include_domains=["a.com"], exclude_domains=["x.a.com", "c.com"]).collect()

    body = mock_client.post.await_args.kwargs["json"]
    assert body["includeDomains"] == ["a.com"]
    assert "excludeDomains" not in body
    assert body["query"] == "q -site:x.a.com"
    assert out.output["data"]["web"] == [{"url": "https://a.com/1"}]

    mock_client = _mock_post_client({"data": {"web": web}})
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        out = await tool(query="q", exclude_domains=["c.com"]).collect()

    body = mock_client.post.await_args.kwargs["json"]
    assert body["excludeDomains"] == ["c.com"]
    assert "includeDomains" not in body
    assert body["query"] == "q"
    assert [r["url"] for r in out.output["data"]["web"]] == ["https://a.com/1", "https://x.a.com/2"]


@pytest.mark.asyncio
async def test_native_web_search_without_platform_says_what_to_do():
    from timbal.errors import ToolProxyUnavailable

    proxy = AsyncMock(side_effect=ToolProxyUnavailable("no subject"))
    with patch("timbal.tools.web_search.execute_tool_proxy", proxy):
        with pytest.raises(ToolProxyUnavailable, match=r"WebSearch\(provider=\.\.\.\)"):
            await WebSearch().handler(query="x", limit=5, tbs=None)
