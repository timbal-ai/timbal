"""Streamable HTTP failure modes the SDK leaves to the client: dead transport mid-call,
stale session after a server restart, unreachable server, request timeout.

Runs a real FastMCP streamable-http server in a subprocess so the transport code path
(POST/SSE, Mcp-Session-Id, 404 -> "Session terminated") is the SDK's, not a mock.
"""

import contextlib
import socket
import subprocess
import sys
import time

import httpx
import pytest
import structlog
from timbal.core.mcp import MCPServer
from timbal.core.test_model import TestModel

try:
    from mcp import types as mcp_types
except ImportError:  # pragma: no cover
    mcp_types = None

pytestmark = pytest.mark.skipif(mcp_types is None, reason="mcp package not installed")


SERVER_SCRIPT = '''
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent
from pydantic import BaseModel

stateless = len(sys.argv) > 2 and sys.argv[2] == "stateless"
mcp = FastMCP("http-server", host="127.0.0.1", port=int(sys.argv[1]), stateless_http=stateless)


@mcp.tool()
def ping() -> str:
    """Ping."""
    return "pong"


class Confirm(BaseModel):
    confirm: bool


@mcp.tool()
async def swap_kb(kb_id: int, ctx: Context) -> str:
    """Swap the KB (elicits)."""
    result = await ctx.elicit(message=f"Swap KB to {kb_id}?", schema=Confirm)
    return "swapped" if result.action == "accept" else "not swapped"


@mcp.tool()
async def ask_llm(question: str, ctx: Context) -> str:
    """Sample from the client's LLM."""
    # related_request_id routes the request onto this tools/call's own SSE stream, which is
    # the only stream a stateless server has. Without it the server SDK tries the standalone
    # GET stream, which does not exist in stateless mode, and drops the request before it
    # ever leaves the server (ClosedResourceError) -- nothing a client can detect.
    result = await ctx.session.create_message(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=question))],
        max_tokens=50,
        related_request_id=ctx.request_id,
    )
    return result.content.text


mcp.run(transport="streamable-http")
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HttpServerProcess:
    def __init__(self, script: str, port: int, mode: str = "") -> None:
        self.script = script
        self.port = port
        self.mode = mode
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, self.script, str(self.port), self.mode],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 15
        while time.time() < deadline:
            with contextlib.suppress(Exception):
                httpx.get(self.url, timeout=0.2)  # any HTTP answer means uvicorn is up
                return
            time.sleep(0.05)
        raise RuntimeError("streamable-http test server did not start")

    def kill(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def restart(self) -> None:
        self.kill()
        self.start()


@pytest.fixture
def http_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER_SCRIPT)
    server = HttpServerProcess(str(script), _free_port())
    server.start()
    yield server
    server.kill()


@pytest.fixture
def stateless_http_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER_SCRIPT)
    server = HttpServerProcess(str(script), _free_port(), mode="stateless")
    server.start()
    yield server
    server.kill()


async def _ping(client: MCPServer):
    tools = {t.name: t for t in await client.resolve()}
    return await tools["ping"]().collect()


class TestHttpResilience:
    async def test_dead_server_mid_call_fails_fast_then_reconnects(self, http_server):
        client = MCPServer(transport="http", url=http_server.url)
        try:
            assert (await _ping(client)).output == "pong"

            http_server.kill()
            t0 = time.monotonic()
            result = await _ping(client)
            elapsed = time.monotonic() - t0
            # Previously this hung forever: the owner task died but the pending
            # call_tool future was never failed.
            assert result.status.code == "error", result
            assert elapsed < 5, elapsed
            assert "TaskGroup" not in result.error["message"]

            http_server.restart()
            assert (await _ping(client)).output == "pong"
        finally:
            await client.close()

    async def test_stale_session_after_idle_restart_is_transparent(self, http_server):
        """Server restarts while we are idle: the SDK swallows the GET-stream error, keeps the
        old Mcp-Session-Id, and every request 404s ("Session terminated") forever. We must
        reset and retry once."""
        client = MCPServer(transport="http", url=http_server.url)
        try:
            assert (await _ping(client)).output == "pong"
            first_owner = client._session_task

            http_server.restart()
            assert client._session_task is first_owner  # nothing noticed yet

            result = await _ping(client)
            assert result.status.code == "success", result.error
            assert result.output == "pong"
            assert client._session_task is not first_owner
            assert first_owner.done()

            # And it stays healthy.
            assert (await _ping(client)).output == "pong"
        finally:
            await client.close()

    async def test_sessionful_server_elicitation_round_trips(self, http_server):
        """Control for the stateless tests: over a sessionful HTTP connection the server's
        elicit() reaches us and our answer reaches it."""
        client = MCPServer(transport="http", url=http_server.url)
        try:
            swap = {t.name: t for t in await client.resolve()}["swap_kb"]
            assert client._get_session_id() is not None
            first = await swap(kb_id=5).collect()
            assert first.status.reason == "input_required"
            sid = first.output["suspension_id"]
            resumed = await swap(kb_id=5, parent_id=first.run_id, resume={sid: {"confirm": True}}).collect()
            assert resumed.output == "swapped"
        finally:
            await client.close()

    async def test_unreachable_server_gives_readable_error(self):
        client = MCPServer(name="ghost", transport="http", url=f"http://127.0.0.1:{_free_port()}/mcp")
        with pytest.raises(ConnectionError) as info:
            await client.resolve()
        message = str(info.value)
        assert "ghost" in message
        assert "failed to connect" in message
        assert "TaskGroup" not in message
        assert client._session is None and client._session_task is None


class TestStatelessHttp:
    """A 2025-era stateless server (no Mcp-Session-Id) treats every POST as independent, so
    our reply to a server-initiated request lands on a transport that has no idea who asked
    and the server's elicit()/create_message() waits forever. Plain tool calls are fine."""

    async def test_plain_calls_work_and_survive_idle_restart(self, stateless_http_server):
        client = MCPServer(transport="http", url=stateless_http_server.url)
        try:
            assert (await _ping(client)).output == "pong"
            assert client._get_session_id() is None
            stateless_http_server.restart()
            assert (await _ping(client)).output == "pong"  # nothing to go stale
        finally:
            await client.close()

    async def test_elicitation_fails_fast_instead_of_hanging(self, stateless_http_server):
        client = MCPServer(name="sl", transport="http", url=stateless_http_server.url)
        try:
            with structlog.testing.capture_logs() as logs:
                tools = {t.name: t for t in await client.resolve()}
            assert any("stateless" in entry.get("event", "") for entry in logs), logs

            t0 = time.monotonic()
            result = await tools["sl__swap_kb"](kb_id=5).collect()
            assert time.monotonic() - t0 < 5
            assert result.status.code == "error"
            assert "stateless" in result.error["message"]
            assert "swap_kb" in result.error["message"]
            assert result.status.reason != "input_required"

            # The connection is still fine for tools that don't call back.
            assert (await tools["sl__ping"]().collect()).output == "pong"
        finally:
            await client.close()

    async def test_sampling_fails_fast_instead_of_hanging(self, stateless_http_server):
        client = MCPServer(transport="http", url=stateless_http_server.url, sampling_model=TestModel(responses=["x"]))
        try:
            ask = {t.name: t for t in await client.resolve()}["ask_llm"]
            t0 = time.monotonic()
            result = await ask(question="?").collect()
            assert time.monotonic() - t0 < 5
            assert result.status.code == "error"
            assert "sampling" in result.error["message"] and "stateless" in result.error["message"]
        finally:
            await client.close()
