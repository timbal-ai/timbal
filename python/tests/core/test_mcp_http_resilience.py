"""Streamable HTTP failure modes the SDK leaves to the client: dead transport mid-call,
stale session after a server restart, unreachable server, request timeout.

Runs a real FastMCP streamable-http server in a subprocess so the transport code path
(POST/SSE, Mcp-Session-Id, 404 -> "Session terminated") is the SDK's, not a mock.
"""

import asyncio
import contextlib
import gzip
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
import asyncio
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


@mcp.tool()
async def slow(seconds: float) -> str:
    """Sleep, then answer."""
    await asyncio.sleep(seconds)
    return f"slept {seconds}"


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

    async def test_tool_slower_than_httpx_default_read_timeout_completes(self, http_server):
        """A ``tools/call`` answer is the first bytes of the POST's SSE stream. With httpx's
        5s default read timeout the SDK's SSE reader hit ``ReadTimeout``, logged it at debug
        and returned without failing the request: transport up, owner task alive, call hung
        forever (Composer bootstrap on ``codegen test``, 2026-09-07). A parallel fast call on
        the same session, as the agent multiplexer issues them, must not change that."""
        client = MCPServer(transport="http", url=http_server.url)
        try:
            tools = {t.name: t for t in await client.resolve()}
            t0 = time.monotonic()
            ping, slow = await asyncio.wait_for(
                asyncio.gather(tools["ping"]().collect(), tools["slow"](seconds=6).collect()),
                timeout=30,
            )
            elapsed = time.monotonic() - t0
            assert ping.output == "pong"
            assert slow.status.code == "success", slow.error
            assert slow.output == "slept 6.0"
            # Windows time.monotonic() has ~15.6ms tick granularity (GetTickCount64),
            # so a 6.0s sleep can measure ~5.984s.
            assert 5.8 <= elapsed < 15, elapsed
            # The session is still the same one and still serves.
            assert (await _ping(client)).output == "pong"
        finally:
            await client.close()

    async def test_timeout_bounds_a_slow_http_tool(self, http_server):
        client = MCPServer(transport="http", url=http_server.url, timeout=1)
        try:
            slow = {t.name: t for t in await client.resolve()}["slow"]
            t0 = time.monotonic()
            result = await slow(seconds=30).collect()
            assert time.monotonic() - t0 < 5
            assert result.status.code == "error"
            assert result.error["type"] == "TimeoutError"
            assert "tools/call slow" in result.error["message"]
            # The session survives the timeout; the next call works.
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


class _CutStreamHandler(BaseHTTPRequestHandler):
    """An MCP server whose answer to a request is an event stream (#292). ``server.mode`` decides what the stream holds:

    - ``eof`` / ``reset``: a comment, then the body ends cleanly / is cut short, with no answer (a gateway cutting the
      stream of a server that died);
    - ``event id``: an event that carries an id (and ``retry: 10``), then the end: the SDK resumes from it, and this
      server refuses the resumption (405);
    - ``resumed``: the same, and the resumption (GET with ``Last-Event-ID``) sends the answer;
    - ``cut in an event``: the stream ends inside an event, after its ``id`` line: no event was dispatched;
    - ``answer``, ``gzip answer``, ``bare cr answer``: the answer, as the last bytes of the stream (gzip-encoded, or
      with bare CR line endings, which the SDK's parser dispatches only at the end of the stream).

    ``server.cut`` names the methods answered that way (others get a JSON answer); ``server.calls`` counts them."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, payload):
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _chunk(self, data: bytes):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _stream(self, gzip_body=False):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if gzip_body:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    @staticmethod
    def _answer(request_id, eol=b"\n"):
        result = {"content": [{"type": "text", "text": "answered"}], "isError": False}
        data = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode()
        return b"event: message" + eol + b"data: " + data + eol + eol

    def do_GET(self):
        if self.server.mode == "resumed" and self.headers.get("Last-Event-ID") == "1":
            self.server.calls.append("resume")
            self._stream()
            self._chunk(self._answer(self.server.request_id))
            self._chunk(b"")
            self.close_connection = True
            return
        self._empty(405)  # no standalone stream, and no resumption

    def do_DELETE(self):
        self._empty(204)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        method, server = body.get("method"), self.server
        if method == "initialize":
            return self._json(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": body["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "cut", "version": "0"},
                    },
                }
            )
        if method not in ("tools/call", "tools/list"):  # notifications, our answers to the server
            return self._empty(202)
        server.calls.append(method)
        if method not in server.cut or (method == "tools/list" and server.calls.count("tools/list") > 1):
            return self._json({"jsonrpc": "2.0", "id": body["id"], "result": {"tools": []}})
        server.request_id, mode = body["id"], server.mode
        if mode in ("answer", "gzip answer", "bare cr answer"):
            payload = self._answer(body["id"], b"\r" if mode == "bare cr answer" else b"\n")
            self._stream(gzip_body=mode == "gzip answer")
            self._chunk(gzip.compress(payload) if mode == "gzip answer" else payload)
            self._chunk(b"")
            self.close_connection = True
            return
        self._stream()
        if mode in ("event id", "resumed"):
            note = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "x"}}
            self._chunk(b"id: 1\nretry: 10\ndata: %s\n\n" % json.dumps(note).encode())
        elif mode == "cut in an event":  # the body ends right after: the event never completes
            self._chunk(b'id: 7\ndata: {"jsonrpc": "2.0", ')
            self._chunk(b"")
            self.close_connection = True
            return
        self._chunk(b": working\n\n")
        time.sleep(0.2)
        if mode != "reset":
            self._chunk(b"")  # the chunked body's end: a clean end of the stream
        self.close_connection = True


@contextlib.contextmanager
def _cut_stream_server(mode, cut=("tools/call",)):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CutStreamHandler)
    server.mode, server.cut, server.calls, server.request_id = mode, cut, [], None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}/mcp"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


class _ReadsBodies(httpx.Auth):
    """An auth whose flow reads every response body before the client sees it (httpx's ``requires_response_body``,
    as a refresh-token flow does)."""

    requires_response_body = True

    def auth_flow(self, request):
        yield request


async def until(condition, timeout):
    """Wait for ``condition()``, failing after ``timeout`` seconds."""

    async def poll():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def _call(client):
    return await asyncio.wait_for(client._request(lambda s: s.call_tool("anything", {}), what="tools/call"), 10)


class TestResponseStreamEnds:
    """A request whose response stream ends before its answer arrives fails at once as a lost connection, instead
    of waiting its whole ``timeout`` (600 s by default) for an answer nothing will send (#292). The SDK's own client
    waits too: it reads the stream, finds no answer, and returns without failing the request. Decided once the SDK
    is done with the stream (and with resuming it), so a request whose answer did come is never failed."""

    @pytest.mark.parametrize("mode", ["eof", "reset", "cut in an event", "event id"])
    async def test_a_tool_call_whose_stream_ends_unanswered_fails_at_once(self, mode):
        """Also when the stream is cut inside an event after its ``id`` (no event was dispatched, so the SDK has
        nothing to resume from), and when the SDK resumes and the resumption fails (it gives up after two tries)."""
        with _cut_stream_server(mode) as (server, url):
            client = MCPServer(name="cut", transport="http", url=url, timeout=60)
            try:
                t0 = time.monotonic()
                with pytest.raises(ConnectionError, match="connection lost during tools/call"):
                    await _call(client)
                assert time.monotonic() - t0 < 5
                assert server.calls == ["tools/call"]  # not sent again: the tool may have run
                with pytest.raises(ConnectionError):  # the session is still up: the next call is its own
                    await _call(client)
                assert server.calls == ["tools/call", "tools/call"]
            finally:
                await client.close()

    @pytest.mark.parametrize("mode", ["answer", "gzip answer", "bare cr answer", "resumed"])
    @pytest.mark.parametrize("auth", [None, "reads bodies"])
    async def test_a_tool_call_whose_answer_comes_gets_it(self, mode, auth):
        """An answer that comes is never taken for a missing one: as the stream's last bytes, compressed, with bare
        CR line endings (dispatched only at the stream's end), read first by an auth flow, or on a resumed stream
        (review of the fix for #292: deciding from the stream's bytes failed each of these)."""
        with _cut_stream_server(mode) as (server, url):
            client = MCPServer(name="cut", transport="http", url=url, timeout=60, auth=_ReadsBodies() if auth else None)
            try:
                result = await _call(client)
                assert result.content[0].text == "answered" and not result.isError
                assert server.calls.count("tools/call") == 1
                assert ("resume" in server.calls) == (mode == "resumed")
            finally:
                await client.close()

    async def test_a_tools_list_whose_stream_ends_is_asked_again(self):
        """``tools/list`` changes nothing, so a lost answer is asked for again, as after a lost connection."""
        with _cut_stream_server("eof", cut=("tools/list",)) as (server, url):
            client = MCPServer(name="cut", transport="http", url=url, timeout=60)
            try:
                assert await asyncio.wait_for(client.resolve(), 10) == []
                assert server.calls == ["tools/list", "tools/list"]
            finally:
                await client.close()

    async def test_a_call_that_stopped_waiting_leaves_no_warning(self, monkeypatch):
        """A request that timed out before its stream ended gets the error all the same, from the SDK's task: the
        session finds no request waiting for it and passes it on as an error, which is not worth a warning."""
        seen, on_message = [], MCPServer._on_message

        async def recording(self, message):
            seen.append(message)
            return await on_message(self, message)

        monkeypatch.setattr(MCPServer, "_on_message", recording)
        with _cut_stream_server("eof") as (_server, url), structlog.testing.capture_logs() as logs:
            client = MCPServer(name="cut", transport="http", url=url, timeout=0.05)
            try:
                with pytest.raises(TimeoutError):
                    await _call(client)
                await until(lambda: any("timbal: the response stream ended" in str(m) for m in seen), 10)
                assert not [entry for entry in logs if entry.get("event") == "MCP session error"], logs
            finally:
                await client.close()
