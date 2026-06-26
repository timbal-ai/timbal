"""Tests for platform tool proxy client and Tool credential fallback."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr
from timbal.core.tool import Tool
from timbal.errors import PlatformError
from timbal.platform.tool_proxy import build_tool_proxy_headers, execute_tool_proxy
from timbal.state import RunContext, get_run_context, set_call_id, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.config_loader import FileConfig


def _platform_config() -> PlatformConfig:
    return PlatformConfig(
        host="api.example.com",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token=SecretStr("platform-key")),
        subject=PlatformSubject(org_id="org-1", app_id="app-9", project_id="proj-7", rev="main"),
    )


def _empty_file_config(*_args, **_kwargs) -> FileConfig:
    return FileConfig(None, None, None, None)


@pytest.fixture(autouse=True)
def _isolated_context(monkeypatch):
    monkeypatch.setattr(
        "timbal.state.config_loader.load_file_config",
        _empty_file_config,
    )
    monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
    monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
    for var in (
        "TIMBAL_APP_ID",
        "TIMBAL_PROJECT_ID",
        "TIMBAL_REV",
        "TIMBAL_API_KEY",
        "TIMBAL_API_HOST",
        "TIMBAL_ORG_ID",
        "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    set_run_context(None)  # type: ignore[arg-type]
    set_call_id(None)


def test_build_tool_proxy_headers_from_subject():
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)
    set_call_id("call-abc")

    headers = build_tool_proxy_headers()

    assert headers["x-timbal-run-id"] == run_context.id
    assert headers["x-timbal-call-id"] == "call-abc"
    assert headers["x-timbal-app-id"] == "app-9"
    assert headers["x-timbal-project-id"] == "proj-7"
    assert headers["x-timbal-rev"] == "main"
    assert "x-timbal-version" in headers


def test_build_tool_proxy_headers_env_fallback(monkeypatch):
    monkeypatch.setenv("TIMBAL_APP_ID", "env-app")
    monkeypatch.setenv("TIMBAL_PROJECT_ID", "env-project")

    run_context = RunContext(
        platform_config=PlatformConfig(
            host="api.example.com",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token=SecretStr("k")),
            subject=PlatformSubject(org_id="org-1"),
        ),
        tracing_provider=None,
    )
    set_run_context(run_context)

    headers = build_tool_proxy_headers()
    assert headers["x-timbal-app-id"] == "env-app"
    assert headers["x-timbal-project-id"] == "env-project"
    assert "x-timbal-call-id" not in headers


@pytest.mark.asyncio
async def test_execute_tool_proxy_posts_to_tools_path():
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)
    set_call_id("call-1")

    mock_response = MagicMock()
    mock_response.json.return_value = {"markdown": "hello"}

    with patch("timbal.platform.tool_proxy._request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        result = await execute_tool_proxy(
            "firecrawl_scrape",
            {"url": "https://example.com", "formats": ["markdown"]},
        )

    assert result == {"markdown": "hello"}
    mock_request.assert_awaited_once()
    args, kwargs = mock_request.await_args
    assert args[0] == "POST"
    assert args[1] == "orgs/org-1/proxies/v1/tools/firecrawl_scrape"
    assert kwargs["json"] == {"url": "https://example.com", "formats": ["markdown"]}
    assert kwargs["headers"]["x-timbal-app-id"] == "app-9"
    assert kwargs["headers"]["x-timbal-project-id"] == "proj-7"
    assert kwargs["headers"]["x-timbal-run-id"] == run_context.id
    assert kwargs["headers"]["x-timbal-call-id"] == "call-1"


@pytest.mark.asyncio
async def test_execute_tool_proxy_resolves_platform_config_from_env(monkeypatch):
    monkeypatch.setenv("TIMBAL_API_KEY", "sk-platform")
    monkeypatch.setenv("TIMBAL_ORG_ID", "org-from-env")
    monkeypatch.setenv("TIMBAL_API_HOST", "api.env.test")

    run_context = RunContext(tracing_provider=None)
    set_run_context(run_context)
    assert run_context.platform_config is None

    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}

    with patch("timbal.platform.tool_proxy._request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        result = await execute_tool_proxy("tavily_search", {"query": "hello"})

    assert result == {"ok": True}
    ctx = get_run_context()
    assert ctx is not None
    assert ctx.platform_config is not None
    assert ctx.platform_config.host == "api.env.test"
    assert ctx.platform_config.subject is not None
    assert ctx.platform_config.subject.org_id == "org-from-env"

    args, _ = mock_request.await_args
    assert args[1] == "orgs/org-from-env/proxies/v1/tools/tavily_search"


@pytest.mark.asyncio
async def test_execute_tool_proxy_requires_platform_auth(monkeypatch):
    monkeypatch.delenv("TIMBAL_API_KEY", raising=False)
    monkeypatch.delenv("TIMBAL_ORG_ID", raising=False)

    run_context = RunContext(tracing_provider=None)
    set_run_context(run_context)

    with pytest.raises(ValueError, match="Tool proxy requires platform_config"):
        await execute_tool_proxy("firecrawl_scrape", {"url": "https://example.com"})


@pytest.mark.asyncio
async def test_execute_tool_proxy_raises_platform_error_on_http_failure():
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)

    request = httpx.Request("POST", "https://api.example.com/orgs/org-1/proxies/v1/tools/t")
    httpx.Response(404, request=request, text="not found")

    with patch(
        "timbal.platform.tool_proxy._request",
        new_callable=AsyncMock,
        side_effect=PlatformError("proxy failed"),
    ):
        with pytest.raises(PlatformError, match="proxy failed"):
            await execute_tool_proxy("firecrawl_scrape", {"url": "https://example.com"})


@pytest.mark.asyncio
async def test_tool_falls_back_to_proxy_on_credential_not_available():
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)

    async def _handler(url: str) -> dict:
        from timbal.tools._creds import resolve_api_key

        await resolve_api_key(
            provider_name="Firecrawl",
            env_var="FIRECRAWL_API_KEY",
            integration=None,
            api_key=None,
        )
        return {"url": url}

    tool = Tool(name="firecrawl_scrape", handler=_handler, tracing_provider=None)

    with patch(
        "timbal.core.tool.execute_tool_proxy",
        new_callable=AsyncMock,
        return_value={"proxied": True},
    ) as mock_proxy:
        result = await tool(url="https://example.com").collect()

    mock_proxy.assert_awaited_once_with("firecrawl_scrape", {"url": "https://example.com"})
    assert result.status.code == "success"
    assert result.output == {"proxied": True}


@pytest.mark.asyncio
async def test_tool_proxy_failure_surfaces_as_tool_error():
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)

    async def _handler(url: str) -> dict:
        from timbal.tools._creds import resolve_api_key

        await resolve_api_key(
            provider_name="Firecrawl",
            env_var="FIRECRAWL_API_KEY",
            integration=None,
            api_key=None,
        )
        return {"url": url}

    tool = Tool(name="firecrawl_scrape", handler=_handler, tracing_provider=None)

    with patch(
        "timbal.core.tool.execute_tool_proxy",
        new_callable=AsyncMock,
        side_effect=PlatformError("upstream proxy error"),
    ):
        result = await tool(url="https://example.com").collect()

    assert result.status.code == "error"
    assert result.error is not None
    assert result.error["type"] == "PlatformError"
    assert "upstream proxy error" in result.error["message"]


@pytest.mark.parametrize("status_code", [403, 404, 501])
@pytest.mark.asyncio
async def test_tool_proxy_not_implemented_reraises_credential_error(status_code):
    """When the platform has no proxy for this tool (403, or 404/501 fallbacks),
    surface the actionable CredentialNotAvailable (set local creds) instead of
    the opaque platform error."""
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)

    async def _handler(url: str) -> dict:
        from timbal.tools._creds import resolve_api_key

        await resolve_api_key(
            provider_name="Firecrawl",
            env_var="FIRECRAWL_API_KEY",
            integration=None,
            api_key=None,
        )
        return {"url": url}

    tool = Tool(name="firecrawl_scrape", handler=_handler, tracing_provider=None)

    with patch(
        "timbal.core.tool.execute_tool_proxy",
        new_callable=AsyncMock,
        side_effect=PlatformError("not found", status_code=status_code),
    ):
        result = await tool(url="https://example.com").collect()

    assert result.status.code == "error"
    assert result.error is not None
    assert result.error["type"] == "CredentialNotAvailable"
    assert "Firecrawl credentials not found" in result.error["message"]


@pytest.mark.asyncio
async def test_tool_proxy_other_4xx_surfaces_platform_error():
    """A real proxy failure (e.g. 400 bad request) must surface as PlatformError,
    not be masked by the credential error."""
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)

    async def _handler(url: str) -> dict:
        from timbal.tools._creds import resolve_api_key

        await resolve_api_key(
            provider_name="Firecrawl",
            env_var="FIRECRAWL_API_KEY",
            integration=None,
            api_key=None,
        )
        return {"url": url}

    tool = Tool(name="firecrawl_scrape", handler=_handler, tracing_provider=None)

    with patch(
        "timbal.core.tool.execute_tool_proxy",
        new_callable=AsyncMock,
        side_effect=PlatformError("bad request", status_code=400),
    ):
        result = await tool(url="https://example.com").collect()

    assert result.status.code == "error"
    assert result.error is not None
    assert result.error["type"] == "PlatformError"
    assert "bad request" in result.error["message"]


@pytest.mark.asyncio
async def test_tool_collect_passes_run_call_id_in_proxy_headers():
    run_context = RunContext(platform_config=_platform_config(), tracing_provider=None)
    set_run_context(run_context)

    async def _handler(url: str) -> dict:
        from timbal.tools._creds import resolve_api_key

        await resolve_api_key(
            provider_name="Firecrawl",
            env_var="FIRECRAWL_API_KEY",
            integration=None,
            api_key=None,
        )
        return {"url": url}

    tool = Tool(name="firecrawl_scrape", handler=_handler, tracing_provider=None)

    mock_response = MagicMock()
    mock_response.json.return_value = {"proxied": True}

    with patch("timbal.platform.tool_proxy._request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        result = await tool(url="https://example.com").collect()

    assert result.status.code == "success"
    _, kwargs = mock_request.await_args
    headers = kwargs["headers"]
    assert headers["x-timbal-run-id"] == run_context.id
    assert headers["x-timbal-call-id"]  # set by Tool.__call__ before handler runs


@pytest.mark.asyncio
async def test_firecrawl_scrape_uses_proxy_without_local_credentials(monkeypatch):
    monkeypatch.setenv("TIMBAL_API_KEY", "sk-platform")
    monkeypatch.setenv("TIMBAL_ORG_ID", "org-fc")
    monkeypatch.setenv("TIMBAL_API_HOST", "api.example.com")

    from timbal.tools.firecrawl import FirecrawlScrape

    run_context = RunContext(tracing_provider=None)
    set_run_context(run_context)

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": {"markdown": "# Example"}}

    with patch("timbal.platform.tool_proxy._request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        tool = FirecrawlScrape(tracing_provider=None)
        result = await tool(url="https://example.com", formats=["markdown"]).collect()

    assert result.status.code == "success"
    assert result.output == {"data": {"markdown": "# Example"}}
    args, kwargs = mock_request.await_args
    assert args[1] == "orgs/org-fc/proxies/v1/tools/firecrawl_scrape"
    assert kwargs["json"]["url"] == "https://example.com"
    assert kwargs["json"]["formats"] == ["markdown"]
