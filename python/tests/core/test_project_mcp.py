"""Real Streamable HTTP checks for project routing and per-call identities."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from timbal.core import MCPServer
from timbal.state import set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig
from timbal.state.context import RunContext


def caller(token):
    set_run_context(
        RunContext(
            platform_config=PlatformConfig(
                host="unused.test",
                auth=PlatformAuth(type=PlatformAuthType.BEARER, token=token),
            ),
            tracing_provider=None,
        )
    )


@pytest.fixture
def endpoint(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload=None):
            data = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(405)

        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers["content-length"])))
            token = self.headers.get("Authorization")
            requests.append((self.path, token, msg["method"]))
            if self.path not in ("/mcp", "/api/mcp"):
                return self.reply(404)
            if token not in ("Bearer alice", "Bearer bob"):
                return self.reply(401)
            if "id" not in msg:
                return self.reply(202)
            if msg["method"] == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "project", "version": "1"},
                }
            elif msg["method"] == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "who",
                            "description": "Caller identity",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            else:
                result = {"content": [{"type": "text", "text": token.removeprefix("Bearer ")}]}
            self.reply(200, {"jsonrpc": "2.0", "id": msg["id"], "result": result})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("TIMBAL_PROJECT_ENV_ID", raising=False)
    monkeypatch.setenv("TIMBAL_START_API_PORT", str(server.server_port))
    yield server.server_port, requests
    server.shutdown()
    server.server_close()
    thread.join()
    set_run_context(None)


@pytest.mark.asyncio
async def test_local_discovery_and_invocation(endpoint):
    _, requests = endpoint
    caller("alice")
    server = MCPServer(name="crm", transport="http", service="api", connect_timeout=2)
    tools = await server.resolve()
    assert [t.name for t in tools] == ["crm__who"]
    assert await tools[0].handler() == "alice"
    assert {path for path, _, _ in requests} == {"/mcp"}
    assert server.url is None and not server.headers and server._session_task is None


@pytest.mark.asyncio
async def test_gateway_route_and_concurrent_identity_isolation(endpoint, monkeypatch):
    port, requests = endpoint
    monkeypatch.setenv("TIMBAL_PROJECT_ENV_ID", "123")
    monkeypatch.setenv("TIMBAL_PROJECT_ENV_ORIGIN", f"http://127.0.0.1:{port}")
    caller("alice")
    server = MCPServer(transport="http", service="api", connect_timeout=2)
    tool = (await server.resolve())[0]

    async def invoke(token):
        caller(token)
        return await tool.handler()

    assert await asyncio.gather(invoke("alice"), invoke("bob")) == ["alice", "bob"]
    assert {path for path, _, _ in requests} == {"/api/mcp"}
    assert server._session_task is None and server._tools_cache is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("endpoint")
async def test_failed_discovery_does_not_poison_next_caller():
    caller("rejected")
    server = MCPServer(transport="http", service="api", connect_timeout=2)
    with pytest.raises(ConnectionError):
        await server.resolve()
    caller("bob")
    assert await (await server.resolve())[0].handler() == "bob"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "https://outside.test/mcp"},
        {"headers": {"Authorization": "Bearer fake"}},
        {"transport": "stdio", "command": "unused"},
    ],
)
def test_service_mode_rejects_manual_transport_identity(kwargs):
    with pytest.raises(ValueError, match="service='api'"):
        MCPServer(**({"transport": "http", "service": "api"} | kwargs))
