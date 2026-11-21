"""Agicap and Payhawk through their official hosted MCP servers.

Unit tests cover credential resolution and the approval default; the end-to-end test runs a real
streamable-http MCP server that rejects requests without the bearer token, so the whole path
(``OfficialMCPServer`` -> ``MCPServer`` -> ``IntegrationBearerAuth`` -> wire) is exercised offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr
from timbal.core import MCPServer
from timbal.platform.integrations import Integration
from timbal.tools import AgicapAuth, AgicapMCP, PayhawkAuth, PayhawkMCP
from timbal.tools._mcp_oauth import IntegrationBearerAuth, gate_unless_read_only

pytest.importorskip("mcp")

VENDORS = [
    pytest.param(PayhawkMCP, PayhawkAuth, "payhawk", "https://mcp.payhawk.com/mcp", "PAYHAWK_MCP_TOKEN", id="payhawk"),
    pytest.param(AgicapMCP, AgicapAuth, "agicap", "https://mcp.agicap.com/mcp", "AGICAP_MCP_TOKEN", id="agicap"),
]


@pytest.fixture(autouse=True)
def _no_env_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAYHAWK_MCP_TOKEN", raising=False)
    monkeypatch.delenv("AGICAP_MCP_TOKEN", raising=False)


async def _headers(auth: httpx.Auth) -> httpx.Headers:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=auth) as client:
        await client.get("https://h/mcp")
    return seen[0].headers


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("server_cls", "_auth", "provider", "url", "_env"), VENDORS)
def test_preset_points_at_the_official_server(server_cls, _auth, provider, url, _env) -> None:
    server = server_cls(token="t")
    assert isinstance(server, MCPServer)
    assert (server.transport, server.url, server.name) == ("http", url, provider)
    assert isinstance(server.auth, IntegrationBearerAuth)
    assert server.approval is gate_unless_read_only


@pytest.mark.parametrize(("server_cls", "_auth", "_provider", "_url", "_env"), VENDORS)
def test_overrides_and_stdio_rejection(server_cls, _auth, _provider, _url, _env) -> None:
    server = server_cls(token="t", url="https://eu.example/mcp", name="pay", approval=None, defer_loading=True)
    assert (server.url, server.name, server.approval, server.defer_loading) == (
        "https://eu.example/mcp",
        "pay",
        None,
        True,
    )
    with pytest.raises(ValueError, match="only the http transport"):
        server_cls(token="t", transport="stdio", command="x")


@pytest.mark.parametrize(("server_cls", "_auth", "_provider", "_url", "_env"), VENDORS)
def test_integration_id_is_serialised_not_the_token(server_cls, _auth, _provider, _url, _env) -> None:
    server = server_cls(integration="org-int-1", token="secret-token")
    dumped = server.model_dump()
    assert dumped["integration"] == "org-int-1"
    assert "secret-token" not in repr(dumped)


# ---------------------------------------------------------------------------
# Credentials: explicit token -> integration -> env
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("_cls", "auth_cls", "_provider", "_url", "env"), VENDORS)
async def test_token_precedence(_cls, auth_cls, _provider, _url, env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env, "from-env")
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"access_token": "from-integration"})

    assert (await _headers(auth_cls(token="explicit")))["Authorization"] == "Bearer explicit"
    assert (await _headers(auth_cls(token=SecretStr("secret"))))["Authorization"] == "Bearer secret"

    auth = auth_cls(integration)
    await auth.prepare()
    assert (await _headers(auth))["Authorization"] == "Bearer from-integration"
    assert (await _headers(auth_cls()))["Authorization"] == "Bearer from-env"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["access_token", "token", "bearer_auth"])
async def test_integration_token_keys(key: str) -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={key: "tok"})
    auth = PayhawkAuth(integration)
    await auth.prepare()
    assert (await _headers(auth))["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_integration_without_a_token_is_an_actionable_error() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "not-oauth"})
    with pytest.raises(ValueError, match="Reconnect the integration"):
        await AgicapAuth(integration).prepare()


@pytest.mark.asyncio
@pytest.mark.parametrize(("_cls", "auth_cls", "_provider", "_url", "env"), VENDORS)
async def test_missing_credentials_name_the_env_var(_cls, auth_cls, _provider, _url, env) -> None:
    with pytest.raises(ValueError, match=env):
        await auth_cls().prepare()


_CALLER = contextvars.ContextVar("caller", default=None)


@pytest.mark.asyncio
async def test_integration_resolves_in_caller_context_for_context_free_requests() -> None:
    """MCPServer sends from an owner task with an empty context; the integration must resolve in the caller's."""

    async def resolve() -> dict[str, Any]:
        assert _CALLER.get() == "run-1", "Integration.resolve ran outside the caller's context"
        return {"access_token": "tok"}

    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(side_effect=resolve)
    auth = PayhawkAuth(integration)

    async def send() -> httpx.Headers:
        return await _headers(auth)

    _CALLER.set("run-1")
    await auth.prepare()
    headers = await asyncio.get_running_loop().create_task(send(), context=contextvars.Context())
    assert headers["Authorization"] == "Bearer tok"
    assert integration.resolve.await_count == 1


@pytest.mark.asyncio
async def test_unprepared_integration_refuses_without_run_context() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"access_token": "tok"})
    with pytest.raises(RuntimeError, match="caller's run context"):
        await _headers(AgicapAuth(integration))
    integration.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_request_prepares_auth_in_caller_task() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"access_token": "tok"})
    server = PayhawkMCP(integration=integration)
    session = SimpleNamespace(list_tools=AsyncMock(return_value="tools"))
    server._connect = AsyncMock(return_value=session)  # type: ignore[method-assign]
    assert await server._request(lambda s: s.list_tools(), what="tools/list") == "tools"
    integration.resolve.assert_awaited_once()


# ---------------------------------------------------------------------------
# Approval default
# ---------------------------------------------------------------------------


def _tool(read_only: bool | None, destructive: bool | None = None) -> Any:
    annotations = None if read_only is None else SimpleNamespace(readOnlyHint=read_only, destructiveHint=destructive)
    return SimpleNamespace(name="t", annotations=annotations)


def test_gate_unless_read_only_is_fail_closed() -> None:
    assert gate_unless_read_only(_tool(True)) is False
    assert gate_unless_read_only(_tool(False)) is True
    assert gate_unless_read_only(_tool(None)) is True  # unannotated: treated as a write
    assert gate_unless_read_only(SimpleNamespace(name="t", annotations=SimpleNamespace(readOnlyHint=None))) is True


# ---------------------------------------------------------------------------
# End to end against a real bearer-protected MCP server
# ---------------------------------------------------------------------------

SERVER_SCRIPT = '''
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("vendor", host="127.0.0.1", port=int(sys.argv[1]))
READ = ToolAnnotations(readOnlyHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False)


@mcp.tool(annotations=READ)
def list_expenses() -> str:
    """List expenses."""
    return "2 expenses"


@mcp.tool(annotations=WRITE)
def create_expense(amount: float) -> str:
    """Create an expense."""
    return f"created {amount}"


@mcp.tool()
def mystery() -> str:
    """No annotations."""
    return "?"


app = mcp.streamable_http_app()


class Bearer:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != b"Bearer good-token":
                await send({"type": "http.response.start", "status": 401, "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b'Bearer resource_metadata="http://x/.well-known"')]})
                await send({"type": "http.response.body", "body": b'{"error":"invalid_token"}'})
                return
        await self.inner(scope, receive, send)


uvicorn.run(Bearer(app), host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def vendor_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER_SCRIPT)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    url = f"http://127.0.0.1:{port}/mcp"
    deadline = time.time() + 20
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            httpx.get(url, timeout=0.2)
            break
        time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError("vendor MCP test server did not start")
    yield url
    proc.kill()
    proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(("server_cls", "_auth", "provider", "_url", "_env"), VENDORS)
async def test_end_to_end_tools_and_approval(vendor_server, server_cls, _auth, provider, _url, _env) -> None:
    server = server_cls(url=vendor_server, token="good-token")
    try:
        tools = {t.name: t for t in await server.resolve()}
        assert set(tools) == {f"{provider}__list_expenses", f"{provider}__create_expense", f"{provider}__mystery"}
        assert tools[f"{provider}__list_expenses"].requires_approval is False
        assert tools[f"{provider}__create_expense"].requires_approval is True
        assert tools[f"{provider}__mystery"].requires_approval is True

        result = await tools[f"{provider}__list_expenses"]().collect()
        assert result.status.code == "success", result.error
        assert "2 expenses" in str(result.output)

        gated = await tools[f"{provider}__create_expense"](amount=12.5).collect()
        assert gated.status.reason == "approval_required"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_end_to_end_wrong_token_is_rejected(vendor_server) -> None:
    server = PayhawkMCP(url=vendor_server, token="bad-token")
    try:
        with pytest.raises(Exception, match=r"401|Unauthorized|invalid_token|connect"):
            await server.resolve()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_end_to_end_approval_can_be_relaxed(vendor_server) -> None:
    server = AgicapMCP(url=vendor_server, token="good-token", approval=None)
    try:
        tools = {t.name: t for t in await server.resolve()}
        assert not any(t.requires_approval for t in tools.values())
        out = await tools["agicap__create_expense"](amount=3).collect()
        assert out.status.code == "success", out.error
        assert "created 3" in str(out.output)
    finally:
        await server.close()
