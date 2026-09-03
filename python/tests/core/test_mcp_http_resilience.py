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
from timbal.core.mcp import MCPServer

try:
    from mcp import types as mcp_types
except ImportError:  # pragma: no cover
    mcp_types = None

pytestmark = pytest.mark.skipif(mcp_types is None, reason="mcp package not installed")


SERVER_SCRIPT = '''
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("http-server", host="127.0.0.1", port=int(sys.argv[1]))


@mcp.tool()
def ping() -> str:
    """Ping."""
    return "pong"


mcp.run(transport="streamable-http")
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HttpServerProcess:
    def __init__(self, script: str, port: int) -> None:
        self.script = script
        self.port = port
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, self.script, str(self.port)],
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

    async def test_unreachable_server_gives_readable_error(self):
        client = MCPServer(name="ghost", transport="http", url=f"http://127.0.0.1:{_free_port()}/mcp")
        with pytest.raises(ConnectionError) as info:
            await client.resolve()
        message = str(info.value)
        assert "ghost" in message
        assert "failed to connect" in message
        assert "TaskGroup" not in message
        assert client._session is None and client._session_task is None
