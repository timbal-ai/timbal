import asyncio
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from timbal import Agent
from timbal.core.mcp import (
    PROVENANCE_PARENT_KEY,
    PROVENANCE_RUN_KEY,
    MCPServer,
    MCPTool,
    _convert_call_tool_result,
    _pseudonym,
)
from timbal.types.content import FileContent, TextContent, ToolUseContent
from timbal.types.message import Message

try:
    from mcp import types as mcp_types
except ImportError:  # pragma: no cover
    mcp_types = None

pytestmark = pytest.mark.skipif(mcp_types is None, reason="mcp package not installed")


SERVER_SCRIPT = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@mcp.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


mcp.run()
'''


@pytest.fixture
def stdio_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER_SCRIPT)
    return MCPServer(transport="stdio", command=sys.executable, args=[str(script)])


class TestMCPServerValidation:
    def test_stdio_requires_command(self):
        with pytest.raises(ValidationError, match="'command' is required"):
            MCPServer(transport="stdio")

    def test_http_requires_url(self):
        with pytest.raises(ValidationError, match="'url' is required"):
            MCPServer(transport="http")


class TestMCPServerResolve:
    @pytest.mark.asyncio
    async def test_resolve_returns_tools_with_server_schema(self, stdio_server):
        try:
            tools = await stdio_server.resolve()
            by_name = {t.name: t for t in tools}
            assert set(by_name) == {"add", "greet", "boom"}
            assert all(isinstance(t, MCPTool) for t in tools)

            add = by_name["add"]
            assert add.description == "Add two integers."
            schema = add.anthropic_schema["input_schema"]
            assert schema["properties"]["a"]["type"] == "integer"
            assert schema["properties"]["b"]["type"] == "integer"
            assert set(schema["required"]) == {"a", "b"}
        finally:
            await stdio_server.close()

    @pytest.mark.asyncio
    async def test_resolve_caches_tools(self, stdio_server):
        try:
            tools1 = await stdio_server.resolve()
            tools2 = await stdio_server.resolve()
            assert tools1 is tools2
        finally:
            await stdio_server.close()

    @pytest.mark.asyncio
    async def test_close_clears_cache(self, stdio_server):
        tools = await stdio_server.resolve()
        assert tools
        await stdio_server.close()
        assert stdio_server._tools_cache is None
        assert stdio_server._session is None


META_SERVER_SCRIPT = """
import json
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("meta-server")


@mcp.tool()
def echo_meta(ctx: Context) -> str:
    \"\"\"Return the request's _meta.\"\"\"
    meta = ctx.request_context.meta
    return json.dumps(meta.model_dump(exclude_none=True) if meta is not None else {})


mcp.run()
"""


class TestProvenance:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provenance", [True, None])
    async def test_run_and_call_ids_ride_in_meta_when_enabled(self, tmp_path, provenance):
        import json

        from timbal.core.mcp import PROVENANCE_CALL_KEY, PROVENANCE_RUN_KEY
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        script = tmp_path / "meta_server.py"
        script.write_text(META_SERVER_SCRIPT)
        server = MCPServer(transport="stdio", command=sys.executable, args=[str(script)], provenance=provenance)
        call = Message(
            role="assistant", content=[ToolUseContent(id="m1", name="echo_meta", input={})], stop_reason="tool_use"
        )
        agent = Agent(name="agent", model=TestModel(responses=[call, "done"]), tools=[server])
        try:
            events = [event async for event in agent(prompt="go")]
        finally:
            await server.close()
        tool_out = next(e for e in events if getattr(e, "type", "") == "OUTPUT" and e.path == "agent.echo_meta")
        meta = json.loads(tool_out.output)
        if provenance:
            # A third party that opted in: pseudonyms of the run and of the tool's own span.
            assert meta[PROVENANCE_RUN_KEY] == _pseudonym(server._origin(), tool_out.run_id)
            assert meta[PROVENANCE_CALL_KEY] == _pseudonym(server._origin(), tool_out.call_id)
            assert tool_out.run_id not in json.dumps(meta) and "traceparent" not in meta
        else:
            assert PROVENANCE_RUN_KEY not in meta  # third-party servers never see run ids by default

    @pytest.mark.asyncio
    async def test_a_chat_session_run_sends_its_session_id_but_no_credential_to_a_third_party(self, tmp_path):
        import json

        from timbal.core.chat_session import ChatSession
        from timbal.core.mcp import PROVENANCE_SESSION_KEY
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        script = tmp_path / "meta_server.py"
        script.write_text(META_SERVER_SCRIPT)
        server = MCPServer(transport="stdio", command=sys.executable, args=[str(script)], provenance=True)
        seen = []

        def handler(messages):
            results = [m for m in messages if m.role == "tool"]
            if results:
                seen.append(results[-1].content[0].content[0].text)
                return "done"
            return Message(
                role="assistant", content=[ToolUseContent(id="m1", name="echo_meta", input={})], stop_reason="tool_use"
            )

        agent = Agent(name="agent", model=TestModel(handler=handler), tools=[server])
        try:
            async with ChatSession(agent, session_id="chat-123") as session:
                session.send("go")
                async for item in session.events():
                    if isinstance(item, dict) and item.get("kind") == "turn_finished":
                        break
        finally:
            await server.close()
        meta = json.loads(seen[0])
        # A caller's session id is its key, never a run id: the first turn runs under one of its own, and the
        # hint is the caller's id (the platform's chat reserves it as the thread's root run). This server is a
        # third party that opted in, so it sees both as its own pseudonyms.
        origin = server._origin()
        assert session.root_run_id == "chat-123"
        assert meta[PROVENANCE_SESSION_KEY] == _pseudonym(origin, "chat-123")
        assert meta[PROVENANCE_RUN_KEY] not in (meta[PROVENANCE_SESSION_KEY], "chat-123", None)
        assert PROVENANCE_PARENT_KEY not in meta  # the root has no parent
        assert not any("token" in key for key in meta)  # identity is the request's credential, never _meta

    def test_a_chained_run_names_its_parent_and_no_credential(self):
        from timbal.state import set_run_context
        from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
        from timbal.state.context import RunContext

        ctx = RunContext(parent_id="run-parent")
        ctx.platform_config = PlatformConfig(
            host="api.example.test",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="secret-credential"),
            subject=PlatformSubject(org_id="1", app_id="2"),
        )
        set_run_context(ctx)
        try:
            meta = platform_server()._provenance_meta()
        finally:
            set_run_context(None)
        assert meta[PROVENANCE_PARENT_KEY] == "run-parent" and meta[PROVENANCE_RUN_KEY] == ctx.id
        assert "secret-credential" not in str(meta)

    def test_on_by_default_only_for_the_platform_server(self):
        assert MCPServer(transport="http", service="api")._sends_provenance()
        assert not MCPServer(transport="http", url="https://example.com/mcp")._sends_provenance()

    def test_the_platform_gets_the_tools_span_as_traceparent(self):
        from timbal.state import set_call_id, set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.exporters.otel import OTelExporter

        ctx = RunContext()
        set_run_context(ctx)
        set_call_id("call-1")
        try:
            meta = platform_server()._provenance_meta()
        finally:
            set_run_context(None)
            set_call_id(None)
        # The same ids OTelExporter exports the run under: the server's spans join that trace.
        assert meta["traceparent"] == f"00-{OTelExporter._trace_id(ctx.id)}-{OTelExporter._span_id('call-1')}-01"

    def test_an_active_otel_span_wins_and_reaches_third_parties_too(self, monkeypatch):
        import types

        from timbal.state import set_call_id, set_run_context
        from timbal.state.context import RunContext

        valid = types.SimpleNamespace(is_valid=True)
        trace_mod = types.ModuleType("opentelemetry.trace")
        trace_mod.get_current_span = lambda: types.SimpleNamespace(get_span_context=lambda: valid)
        propagate_mod = types.ModuleType("opentelemetry.propagate")
        propagate_mod.inject = lambda carrier: carrier.update(
            traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01", tracestate="k=v"
        )
        root = types.ModuleType("opentelemetry")
        root.trace, root.propagate = trace_mod, propagate_mod
        for name, module in (
            ("opentelemetry", root),
            ("opentelemetry.trace", trace_mod),
            ("opentelemetry.propagate", propagate_mod),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        set_run_context(RunContext())
        set_call_id("call-1")
        try:
            first = platform_server()._provenance_meta()
            third = MCPServer(transport="http", url="https://tools.example.com/mcp", provenance=True)._provenance_meta()
        finally:
            set_run_context(None)
            set_call_id(None)
        for meta in (first, third):
            assert meta["traceparent"] == "00-" + "a" * 32 + "-" + "b" * 16 + "-01" and meta["tracestate"] == "k=v"

    def test_third_parties_get_pseudonyms_stable_per_origin_and_unlinkable_across(self, monkeypatch):
        from timbal.state import set_call_id, set_run_context
        from timbal.state.context import RunContext

        ctx = RunContext()
        set_run_context(ctx)
        set_call_id("call-1")
        try:
            one = MCPServer(transport="http", url="https://a.example.com/mcp", provenance=True)
            same_origin = MCPServer(transport="http", url="https://A.example.com/other", provenance=True)
            other = MCPServer(transport="http", url="https://b.example.com/mcp", provenance=True)
            a, a2, b = one._provenance_meta(), same_origin._provenance_meta(), other._provenance_meta()
            monkeypatch.setenv("TIMBAL_PROVENANCE_SECRET", "shared")
            keyed = one._provenance_meta()
        finally:
            set_run_context(None)
            set_call_id(None)
        assert a == a2 and a[PROVENANCE_RUN_KEY] != b[PROVENANCE_RUN_KEY]
        assert keyed[PROVENANCE_RUN_KEY] == _pseudonym(one._origin(), ctx.id) != a[PROVENANCE_RUN_KEY]
        assert ctx.id not in str((a, b, keyed)) and "traceparent" not in a


def platform_server():
    server = MCPServer(transport="http", url="https://api.example.test/mcp", provenance=True)
    server._first_party = True
    return server


async def until(condition, timeout=30.0):
    """Wait for ``condition()``. The budget is for a slow machine: one where it never holds fails here."""

    async def poll():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


class TestPlatformConnectionPerRun:
    @pytest.fixture
    def platform(self, tmp_path, monkeypatch):
        """``service="api"`` connections, served by the stdio meta server; records each one opened."""
        script = tmp_path / "meta_server.py"
        script.write_text(META_SERVER_SCRIPT)
        opened = []
        credential = {"value": "Bearer run-a"}
        monkeypatch.setattr(
            "timbal.platform.utils._resolve_url_and_headers",
            lambda *_: ("https://api.example.test/mcp", {"Authorization": credential["value"]}),
        )

        def new(_self, _url, headers):
            server = MCPServer(transport="stdio", command=sys.executable, args=[str(script)])
            server._first_party = True
            opened.append((server, headers["Authorization"]))
            return server

        monkeypatch.setattr(MCPServer, "_new_project_server", new)
        return opened, credential

    @staticmethod
    def calls(n, parallel=False):
        def handler(messages):
            done = sum(1 for m in messages if m.role == "tool")
            if done >= n:
                return "done"
            ids = [f"m{done}-{i}" for i in range(n if parallel else 1)]
            return Message(
                role="assistant",
                content=[ToolUseContent(id=i, name="echo_meta", input={}) for i in ids],
                stop_reason="tool_use",
            )

        return handler

    @pytest.mark.asyncio
    async def test_one_connection_per_run_reused_and_closed_at_its_end(self, platform):
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        for parallel in (False, True):
            set_run_context(None)
            set_call_id(None)
            set_parent_call_id(None)
            agent = Agent(name="agent", model=TestModel(handler=self.calls(3, parallel)), tools=[service])
            out = await agent(prompt="go").collect()
            assert out.status.code == "success", out.error
        # tools/list on every step and five tool calls across two runs: one connection each, closed in the
        # background once its run ended.
        assert len(opened) == 2
        await until(lambda: all(server._session is None for server, _ in opened))
        assert not service._run_connections
        await service.close()

    @pytest.mark.asyncio
    async def test_a_changed_credential_gets_a_new_connection(self, platform):
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, credential = platform
        service = MCPServer(transport="http", service="api")
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        def handler(messages):
            if sum(1 for m in messages if m.role == "tool") == 1:
                credential["value"] = "Bearer run-a-refreshed"
            return self.calls(2)(messages)

        out = await Agent(name="agent", model=TestModel(handler=handler), tools=[service])(prompt="go").collect()
        assert out.status.code == "success", out.error
        assert [auth for _, auth in opened] == ["Bearer run-a", "Bearer run-a-refreshed"]
        await until(lambda: all(server._session is None for server, _ in opened))
        await service.close()

    @pytest.mark.asyncio
    async def test_a_cancelled_run_still_closes_its_connection(self, platform):
        """The run-end hooks run in ``_stream``'s ``finally``: a run cut while it ends does not leak its connection."""
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        task = None

        def handler(messages):
            if any(m.role == "tool" for m in messages):
                task.cancel()  # lands at the run's next await, after the connection was used
                return "done"
            return self.calls(1)(messages)

        agent = Agent(name="agent", model=TestModel(handler=handler), tools=[service])
        task = asyncio.ensure_future(agent(prompt="go").collect())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
        assert len(opened) == 1
        await until(lambda: opened[0][0]._session is None and not service._run_connections)
        await service.close()

    @pytest.mark.asyncio
    async def test_a_run_cut_while_its_end_is_saved_still_closes_its_connection(self, platform):
        """A cut that lands in the run's last save goes through ``_stream``'s ``except``: the hooks still run."""
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context
        from timbal.state.tracing.providers import InMemoryTracingProvider

        opened, _ = platform
        holding, release = asyncio.Event(), asyncio.Event()

        class HoldsTheEnd(InMemoryTracingProvider):
            @classmethod
            async def _store(cls, run_context):
                root = next(s for s in run_context._trace.values() if s.parent_call_id is None)
                if root.t1 is not None and not holding.is_set():
                    holding.set()
                    await release.wait()  # cut here
                await super()._store(run_context)

        service = MCPServer(transport="http", service="api")
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        agent = Agent(
            name="agent", model=TestModel(handler=self.calls(1)), tools=[service], tracing_provider=HoldsTheEnd
        )
        task = asyncio.ensure_future(agent(prompt="go").collect())
        await asyncio.wait_for(holding.wait(), 30)
        assert len(opened) == 1 and service._run_connections  # used, and not idle-closed yet
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
        assert not service._run_connections  # retired by the end hook, not left to the 60 s idle close
        await until(lambda: opened[0][0]._session is None)
        await service.close()

    @pytest.mark.asyncio
    async def test_the_connection_is_closed_before_the_run_returns(self, platform):
        """The end hook waits for the close: a loop that ends with the run (``asyncio.run``) cancels nothing."""
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        out = await Agent(name="agent", model=TestModel(handler=self.calls(2)), tools=[service])(prompt="go").collect()
        assert out.status.code == "success", out.error
        assert len(opened) == 1 and opened[0][0]._session is None and not service._run_connections
        assert not service._close_tasks
        await service.close()

    @pytest.mark.asyncio
    async def test_close_leaves_a_connection_in_use_to_its_call(self, platform, monkeypatch):
        """``close()`` does not cut a call in flight (a per-call connection was the call's own); its end closes it."""
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        opened, _ = platform
        monkeypatch.setattr("timbal.core.mcp._run_connection_key", lambda _context: "run", raising=False)  # one goes on
        set_run_context(RunContext())
        service = MCPServer(transport="http", service="api")
        try:
            async with service._project_connection() as server:
                await server.resolve()
                assert server._session is not None
                await service.close()
                assert server._session is not None  # still the call's
                assert [t.name for t in await server.resolve()] == ["echo_meta"]
            assert server._session is None and not service._run_connections and len(opened) == 1
        finally:
            set_run_context(None)
            await service.close()

    @pytest.mark.asyncio
    async def test_close_waits_for_the_closes_already_started(self):
        """A connection let go of earlier (its run ended, its credential changed) is closed when ``close()`` returns."""
        from timbal.core.mcp import _RunConnection

        release, closed = asyncio.Event(), []

        class SlowClose:
            async def close(self):
                await release.wait()
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(SlowClose(), "fingerprint")
        service._retire_run_connection("run")  # started, not waited for
        asyncio.get_running_loop().call_soon(release.set)
        await asyncio.wait_for(service.close(), 30)
        assert closed == [True]
        assert not service._close_tasks

    @pytest.mark.asyncio
    async def test_a_cancelled_close_lets_the_closes_finish(self):
        from timbal.core.mcp import _RunConnection

        release, closed = asyncio.Event(), []

        class SlowClose:
            async def close(self):
                await release.wait()
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(SlowClose(), "fingerprint")
        closing = asyncio.ensure_future(service.close())
        await until(lambda: service._close_tasks)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 30)
        release.set()
        await until(lambda: closed == [True])

    @pytest.mark.asyncio
    async def test_a_call_on_another_loop_opens_its_own(self, platform):
        """An offloaded handler running ``asyncio.run`` carries the run's context; the run's session belongs to
        the run's loop, so that call gets a connection of its own, closed with its loop's work."""
        from timbal import Tool
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        listed = []

        def offloaded() -> str:
            async def list_there():
                tools = await asyncio.wait_for(service.resolve(), 20)
                listed.append([t.name for t in tools])

            asyncio.run(list_there())
            return "listed"

        responses = iter(
            [
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="a", name="echo_meta", input={})],
                    stop_reason="tool_use",
                ),
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="b", name="offloaded", input={})],
                    stop_reason="tool_use",
                ),
            ]
        )
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        tools = [service, Tool(name="offloaded", handler=offloaded, offload_blocking=True)]
        model = TestModel(handler=lambda _messages: next(responses, "done"))
        out = await Agent(name="agent", model=model, tools=tools)(prompt="go").collect()
        assert out.status.code == "success", out.error
        assert listed == [["echo_meta"]]
        assert len(opened) == 2 and all(server._session is None for server, _ in opened)
        assert not service._run_connections
        await service.close()

    @pytest.mark.asyncio
    async def test_close_from_another_loop_closes_each_session_on_its_own(self, platform):
        """``close()`` through ``asyncio.run`` in a worker thread: the run's session is closed on the run's loop."""
        from timbal import Tool
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        outcome = []

        def offloaded() -> str:
            try:
                asyncio.run(asyncio.wait_for(service.close(), 20))
                outcome.append("closed" if opened[0][0]._session is None else "still open")
            except TimeoutError:
                outcome.append("timed out")
            return "done"

        responses = iter(
            [
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="a", name="echo_meta", input={})],
                    stop_reason="tool_use",
                ),
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="b", name="offloaded", input={})],
                    stop_reason="tool_use",
                ),
            ]
        )
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        tools = [service, Tool(name="offloaded", handler=offloaded, offload_blocking=True)]
        model = TestModel(handler=lambda _messages: next(responses, "done"))
        out = await Agent(name="agent", model=model, tools=tools)(prompt="go").collect()
        assert out.status.code == "success", out.error
        assert outcome == ["closed"]  # closed by the time close() returned in the worker
        # The run's next step listed tools again on a new connection, closed at the run's end.
        assert all(server._session is None for server, _ in opened) and not service._run_connections
        await service.close()

    @pytest.mark.asyncio
    async def test_two_runs_with_one_context_id_do_not_share_a_connection(self, platform):
        """``context.id`` is the caller's: two concurrent runs may carry the same one."""
        from timbal import Tool
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context
        from timbal.state.context import RunContext

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        first_waiting, go_on = asyncio.Event(), asyncio.Event()

        async def gate() -> str:
            first_waiting.set()
            await asyncio.wait_for(go_on.wait(), 30)
            return "on"

        async def release() -> str:
            go_on.set()
            return "released"

        def script(*names):
            steps = iter(
                Message(role="assistant", content=[ToolUseContent(id=n, name=n, input={})], stop_reason="tool_use")
                for n in names
            )
            return TestModel(handler=lambda _messages: next(steps, "done"))

        async def run(model):
            set_run_context(RunContext(id="the-same-id"))
            set_call_id(None)
            set_parent_call_id(None)
            tools = [service, Tool(name="gate", handler=gate), Tool(name="release", handler=release)]
            return await Agent(name="agent", model=model, tools=tools)(prompt="go").collect()

        first = asyncio.ensure_future(run(script("echo_meta", "gate")))
        await asyncio.wait_for(first_waiting.wait(), 30)  # the first run holds its connection
        second = await asyncio.wait_for(run(script("echo_meta", "release")), 30)
        outs = [await asyncio.wait_for(first, 30), second]
        assert all(out.status.code == "success" for out in outs), [out.error for out in outs]
        assert len(opened) == 2  # one each
        assert all(server._session is None for server, _ in opened) and not service._run_connections
        await service.close()

    def test_a_stream_closed_from_another_loop_leaves_its_session_to_a_close_on_its_own(self, platform):
        """A streaming tool opened the run's platform session on an ``asyncio.Runner``'s loop; another loop closes
        the stream while that one is paused. The end does not try to close the session from there (it hung): the
        connection is kept, said in a warning, and a ``close()`` on its own loop closes it."""
        from timbal import Tool
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")

        async def streamer():
            await service.resolve()
            yield "first"
            await asyncio.Event().wait()

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        stream = Tool(name="streamer", handler=streamer)._stream()

        async def read_first():
            async for event in stream:
                if event.type == "DELTA":
                    return

        with asyncio.Runner() as runner:
            runner.run(asyncio.wait_for(read_first(), 30))
            assert len(opened) == 1 and opened[0][0]._session is not None
            asyncio.run(asyncio.wait_for(stream.aclose(), 10))  # from another loop, the owner paused
            assert opened[0][0]._session is not None and service._run_connections
            runner.run(asyncio.wait_for(service.close(), 30))
            assert opened[0][0]._session is None and not service._run_connections

    @pytest.mark.asyncio
    async def test_outside_a_run_each_call_closes_its_own(self, platform):
        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        from timbal.state import set_run_context

        set_run_context(None)
        tools = await service.resolve()
        assert [t.name for t in tools] == ["echo_meta"]
        assert len(opened) == 1 and opened[0][0]._session is None and not service._run_connections


class TestPlatformConnectionLifetime:
    def test_close_says_so_when_a_connection_belongs_to_a_paused_loop(self):
        """An ``asyncio.Runner`` between two ``run()`` calls owns the session: ``close()`` from another loop cannot
        close it there and says so instead of waiting; ``close()`` on that loop closes it."""
        from timbal.core.mcp import _RunConnection

        closed = []

        class Fake:
            async def close(self):
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        with asyncio.Runner() as runner:
            service._run_connections["run"] = _RunConnection(Fake(), "fingerprint", runner.get_loop())
            with pytest.raises(RuntimeError, match="not running"):
                asyncio.run(asyncio.wait_for(service.close(), 10))
            assert closed == [] and "run" in service._run_connections
            runner.run(service.close())
        assert closed == [True] and not service._run_connections

    def test_close_says_so_when_a_close_already_started_waits_on_a_paused_loop(self):
        """Let go of before close() (an idle close, a credential change): its close was started on the owner loop,
        which paused before it finished. close() from another loop cannot finish it, and says so."""
        from timbal.core.mcp import _RunConnection

        closed = []

        class SlowClose:
            def __init__(self):
                self.release = asyncio.Event()

            async def close(self):
                await self.release.wait()
                closed.append(True)

        fake = SlowClose()
        service = MCPServer(transport="http", service="api")
        with asyncio.Runner() as runner:

            async def let_go():
                service._run_connections["run"] = _RunConnection(fake, "fingerprint", asyncio.get_running_loop())
                service._retire_run_connection("run")  # started; the runner pauses before it finishes

            runner.run(let_go())
            assert not service._run_connections and service._close_tasks
            with pytest.raises(RuntimeError, match="open or closing"):
                asyncio.run(asyncio.wait_for(service.close(), 10))

            async def finish():
                fake.release.set()
                await service.close()

            runner.run(finish())
        assert closed == [True] and not service._close_tasks

    @staticmethod
    def closing_loop():
        """A loop that ran when it was checked and had closed by the time work was handed to it."""

        class ClosingLoop:
            def __init__(self):
                self.loop = asyncio.new_event_loop()
                self.loop.close()
                self.handed = False

            def is_closed(self):
                return self.handed

            def is_running(self):
                return not self.handed

            def call_soon_threadsafe(self, *args, **kwargs):
                self.handed = True
                return self.loop.call_soon_threadsafe(*args, **kwargs)  # raises: closed

        return ClosingLoop()

    @pytest.mark.asyncio
    async def test_an_owner_that_closes_before_the_handoff_takes_its_sessions_with_it(self):
        """close() and the run's end hook drop a connection whose loop closed between their check and the handoff,
        as when it was closed at the check, instead of failing or leaving it cached."""
        from timbal.core.mcp import _RunConnection

        class Session:
            async def close(self):
                raise AssertionError("closed on the wrong loop")

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(Session(), "fingerprint", self.closing_loop())
        await asyncio.wait_for(service.close(), 10)
        assert not service._run_connections

        owner = self.closing_loop()
        service._run_connections["ended"] = _RunConnection(Session(), "fingerprint", owner)
        service._run_hooked["ended"] = owner
        await asyncio.wait_for(service._run_ended("ended"), 10)
        assert not service._run_connections and "ended" not in service._run_closes

    def test_close_says_so_when_the_owner_stops_after_the_handoff(self):
        """The owner ran when close() handed it the work, then stopped before the session was closed: close() says so
        instead of waiting for it to run again, and the close ends once it does."""
        import threading

        from timbal.core.mcp import _RunConnection

        owner = asyncio.new_event_loop()
        started, finished, closed = threading.Event(), threading.Event(), []

        class SlowClose:
            async def close(self):
                started.set()
                await release.wait()
                closed.append(True)
                finished.set()

        async def make_event():
            return asyncio.Event()

        thread = threading.Thread(target=owner.run_forever)
        thread.start()
        release = asyncio.run_coroutine_threadsafe(make_event(), owner).result(10)
        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(SlowClose(), "fingerprint", owner)

        async def main():
            closing = asyncio.ensure_future(service.close())
            assert await asyncio.to_thread(started.wait, 10)  # the owner is closing it
            owner.call_soon_threadsafe(owner.stop)
            await asyncio.to_thread(thread.join, 10)
            with pytest.raises(RuntimeError, match="not running"):
                await asyncio.wait_for(closing, 10)

        try:
            asyncio.run(main())
            assert closed == []
            thread = threading.Thread(target=owner.run_forever)
            thread.start()
            owner.call_soon_threadsafe(release.set)
            assert finished.wait(10) and closed == [True]  # finished once its loop ran again
        finally:
            owner.call_soon_threadsafe(owner.stop)
            thread.join(10)
            owner.close()

    @pytest.mark.asyncio
    async def test_connections_let_go_of_are_not_kept_alive_by_the_run(self, monkeypatch):
        """One end hook per run, which finds the connection at the run's end: the hooks keep no server (with its
        credential headers) that a credential change or close() already let go of."""
        import gc
        import weakref

        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        refs, credential = [], {"n": 0}

        class Fake:
            async def close(self):
                pass

        def new(_self, _url, _headers):
            server = Fake()
            refs.append(weakref.ref(server))
            return server

        monkeypatch.setattr(
            "timbal.platform.utils._resolve_url_and_headers",
            lambda *_: ("https://api.example.test/mcp", {"Authorization": f"Bearer {credential['n']}"}),
        )
        monkeypatch.setattr(MCPServer, "_new_project_server", new)
        monkeypatch.setattr("timbal.core.mcp._run_connection_key", lambda _context: "run")
        context = RunContext()
        set_run_context(context)
        service = MCPServer(transport="http", service="api")
        try:
            for n in range(5):
                credential["n"] = n  # a refreshed credential: a new connection each time
                async with service._project_connection():
                    pass
            await service.close()
            gc.collect()
            assert len(refs) == 5 and [ref() for ref in refs] == [None] * 5
            assert len(context._end_hooks) == 1
        finally:
            set_run_context(None)


class TestPlatformConnectionOutsideARun:
    """With the real resolver, which creates and sets a run context when there is none."""

    @pytest.fixture
    def platform(self, tmp_path, monkeypatch):
        script = tmp_path / "meta_server.py"
        script.write_text(META_SERVER_SCRIPT)
        for name in [name for name in os.environ if name.startswith("TIMBAL_")]:
            monkeypatch.delenv(name)
        monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.timbal credentials
        monkeypatch.setenv("TIMBAL_START_API_PORT", "1")
        opened = []

        def new(_self, url, _headers):
            assert url == "http://localhost:1/mcp"  # the real resolver ran
            server = MCPServer(transport="stdio", command=sys.executable, args=[str(script)])
            server._first_party = True
            opened.append(server)
            return server

        monkeypatch.setattr(MCPServer, "_new_project_server", new)
        return opened

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context", ["none", "set by hand"])
    async def test_each_call_opens_and_closes_its_own(self, platform, context):
        from timbal.state import get_run_context, set_call_id, set_run_context
        from timbal.state.context import RunContext

        set_run_context(None if context == "none" else RunContext())
        set_call_id(None)
        service = MCPServer(transport="http", service="api")
        try:
            tools = await service.resolve()
            assert get_run_context() is not None  # the resolver set one: not a run all the same
            assert isinstance(await tools[0].handler(), str)  # the call went through
            assert len(platform) == 2 and all(server._session is None for server in platform)
            assert not service._run_connections
            assert not get_run_context()._end_hooks  # nothing left waiting for an end that never comes
        finally:
            set_run_context(None)
            await service.close()


class TestMCPToolNamespacing:
    def test_unnamed_server_keeps_bare_tool_names(self):
        server = MCPServer(transport="http", url="https://example.com/mcp")
        tool = server._make_tool(
            mcp_types.Tool(
                name="greet",
                description="Greet someone.",
                inputSchema={"type": "object", "properties": {}},
            )
        )
        assert tool.name == "greet"
        assert tool.description == "Greet someone."

    def test_named_server_prefixes_tool_names(self):
        server = MCPServer(name="alpha", transport="http", url="https://example.com/mcp")
        tool = server._make_tool(
            mcp_types.Tool(
                name="greet",
                description="Greet someone.",
                inputSchema={"type": "object", "properties": {}},
            )
        )
        assert tool.name == "alpha__greet"
        assert tool.description == "[alpha] Greet someone."

    @pytest.mark.asyncio
    async def test_two_named_servers_with_same_tool_both_register(self):
        """Two MCPServers exposing the same bare tool name must both be usable."""
        from timbal.core.test_model import TestModel
        from timbal.types.events import OutputEvent

        calls: list[tuple[str, str]] = []

        def _fake_server(name: str) -> MCPServer:
            server = MCPServer(name=name, transport="http", url=f"https://{name}.example/mcp")

            async def resolve() -> list:
                tool = server._make_tool(
                    mcp_types.Tool(
                        name="greet",
                        description=f"Greet via {name}.",
                        inputSchema={
                            "type": "object",
                            "properties": {"who": {"type": "string"}},
                            "required": ["who"],
                        },
                    )
                )

                # Replace the real MCP handler with one that records which server ran.
                async def _handler(**kwargs):
                    calls.append((name, kwargs["who"]))
                    return f"{name}:{kwargs['who']}"

                tool.handler = _handler
                return [tool]

            server.resolve = resolve  # type: ignore[method-assign]
            return server

        alpha = _fake_server("alpha")
        beta = _fake_server("beta")

        tool_calls = Message(
            role="assistant",
            content=[
                ToolUseContent(id="c1", name="alpha__greet", input={"who": "A"}),
                ToolUseContent(id="c2", name="beta__greet", input={"who": "B"}),
            ],
            stop_reason="tool_use",
        )
        agent = Agent(
            name="multi_mcp",
            model=TestModel(responses=[tool_calls, "done"]),
            tools=[alpha, beta],
            max_iter=3,
        )
        events = [event async for event in agent(prompt="greet both")]
        final = events[-1]
        assert isinstance(final, OutputEvent)
        assert final.status.code == "success"

        paths = {e.path for e in events if isinstance(e, OutputEvent)}
        assert "multi_mcp.alpha__greet" in paths
        assert "multi_mcp.beta__greet" in paths
        assert sorted(calls) == [("alpha", "A"), ("beta", "B")]

    @pytest.mark.asyncio
    async def test_qualified_name_still_calls_bare_mcp_tool(self, stdio_server):
        """Wire call must use the bare MCP name even when the agent sees a prefix."""
        named = MCPServer(
            name="demo",
            transport="stdio",
            command=sys.executable,
            args=stdio_server.args,
        )
        try:
            tools = {t.name: t for t in await named.resolve()}
            assert "demo__greet" in tools
            assert "greet" not in tools
            result = await tools["demo__greet"](name="Zed").collect()
            assert result.status.code == "success"
            assert result.output == "Hello, Zed!"
        finally:
            await named.close()


class TestMCPConnectConcurrency:
    @pytest.mark.asyncio
    async def test_parallel_connect_opens_session_once(self):
        """Two concurrent _connect() calls must not each open a transport."""
        server = MCPServer(transport="http", url="https://example.com/mcp")
        opens = 0
        gate = asyncio.Event()

        @asynccontextmanager
        async def slow_connect():
            nonlocal opens
            opens += 1
            await gate.wait()
            yield AsyncMock(name="session")

        server._connect_http = slow_connect  # type: ignore[method-assign]

        t1 = asyncio.create_task(server._connect())
        t2 = asyncio.create_task(server._connect())
        # Let both tasks reach the lock / check before releasing the open.
        await asyncio.sleep(0.05)
        gate.set()
        s1, s2 = await asyncio.gather(t1, t2)

        assert opens == 1
        assert s1 is s2
        assert server._session is s1
        await server.close()

    @pytest.mark.asyncio
    async def test_parallel_resolve_lists_tools_once(self):
        server = MCPServer(transport="http", url="https://example.com/mcp")
        opens = 0
        list_calls = 0

        class FakeSession:
            async def list_tools(self):
                nonlocal list_calls
                list_calls += 1
                await asyncio.sleep(0.05)
                return mcp_types.ListToolsResult(
                    tools=[
                        mcp_types.Tool(
                            name="ping",
                            description="ping",
                            inputSchema={"type": "object", "properties": {}},
                        )
                    ]
                )

        @asynccontextmanager
        async def instant_connect():
            nonlocal opens
            opens += 1
            yield FakeSession()

        server._connect_http = instant_connect  # type: ignore[method-assign]

        tools1, tools2 = await asyncio.gather(server.resolve(), server.resolve())
        assert opens == 1
        assert list_calls == 1
        assert tools1 is tools2
        await server.close()


class TestMCPToolExecution:
    @pytest.mark.asyncio
    async def test_call_text_tool(self, stdio_server):
        try:
            tools = {t.name: t for t in await stdio_server.resolve()}
            result = await tools["greet"](name="Alice").collect()
            assert result.status.code == "success"
            assert result.output == "Hello, Alice!"
        finally:
            await stdio_server.close()

    @pytest.mark.asyncio
    async def test_call_structured_tool(self, stdio_server):
        try:
            tools = {t.name: t for t in await stdio_server.resolve()}
            result = await tools["add"](a=2, b=3).collect()
            assert result.status.code == "success"
            # The text block carries the canonical serialized result.
            assert result.output == "5"
        finally:
            await stdio_server.close()

    @pytest.mark.asyncio
    async def test_call_error_tool(self, stdio_server):
        try:
            tools = {t.name: t for t in await stdio_server.resolve()}
            result = await tools["boom"]().collect()
            assert result.status.code == "error"
            assert "kaboom" in result.error["message"]
        finally:
            await stdio_server.close()


class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_agent_calls_mcp_tool(self, stdio_server):
        from timbal.core.test_model import TestModel

        tool_call = Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name="greet", input={"name": "Bob"})],
            stop_reason="tool_use",
        )
        agent = Agent(
            name="mcp_agent",
            model=TestModel(responses=[tool_call, "done"]),
            tools=[stdio_server],
            max_iter=3,
        )
        try:
            from timbal.types.events import OutputEvent

            events = [event async for event in agent(prompt="Greet Bob")]
            final = events[-1]
            assert isinstance(final, OutputEvent)
            assert final.status.code == "success"
            assert final.output.collect_text() == "done"

            tool_events = [e for e in events if isinstance(e, OutputEvent) and e.path == "mcp_agent.greet"]
            assert len(tool_events) == 1
            assert tool_events[0].output == "Hello, Bob!"
        finally:
            await stdio_server.close()


@pytest.mark.integration
class TestTimbalPlatformMCP:
    """Live integration tests against the real Timbal platform MCP server.

    Run explicitly (requires TIMBAL_API_KEY in the environment or .env):

        uv run pytest python/tests/core/test_mcp.py -m integration -v
    """

    @pytest.fixture
    def platform_server(self):
        api_key = os.getenv("TIMBAL_API_KEY")
        if not api_key:
            pytest.skip("Timbal MCP integration: set TIMBAL_API_KEY in the environment or .env")
        url = os.getenv("TIMBAL_MCP_URL", "https://api.timbal.ai/mcp")
        return MCPServer(
            transport="http",
            url=url,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @pytest.mark.asyncio
    async def test_resolve_platform_tools(self, platform_server):
        try:
            tools = await platform_server.resolve()
            by_name = {t.name: t for t in tools}
            assert "whoami" in by_name
            assert "search_tools" in by_name
            # Server-declared schemas must survive the MCPTool wrapping.
            search_schema = by_name["search_tools"].anthropic_schema["input_schema"]
            assert search_schema["properties"]
        finally:
            await platform_server.close()

    @pytest.mark.asyncio
    async def test_call_whoami(self, platform_server):
        try:
            tools = {t.name: t for t in await platform_server.resolve()}
            result = await tools["whoami"]().collect()
            assert result.status.code == "success"
            assert result.output
        finally:
            await platform_server.close()

    @pytest.mark.asyncio
    async def test_agent_with_platform_mcp(self, platform_server):
        """Full agent loop against the live server, no LLM key needed (TestModel)."""
        from timbal.core.test_model import TestModel
        from timbal.types.events import OutputEvent

        tool_call = Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name="whoami", input={})],
            stop_reason="tool_use",
        )
        agent = Agent(
            name="timbal_mcp_agent",
            model=TestModel(responses=[tool_call, "done"]),
            tools=[platform_server],
            max_iter=3,
        )
        try:
            events = [event async for event in agent(prompt="Who am I?")]
            final = events[-1]
            assert isinstance(final, OutputEvent)
            assert final.status.code == "success"
            assert final.output.collect_text() == "done"

            tool_events = [e for e in events if isinstance(e, OutputEvent) and e.path == "timbal_mcp_agent.whoami"]
            assert len(tool_events) == 1
            assert tool_events[0].status.code == "success"
            assert tool_events[0].output
        finally:
            await platform_server.close()


class TestConvertCallToolResult:
    def test_single_text(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="hi")],
        )
        assert _convert_call_tool_result("t", result) == "hi"

    def test_multiple_texts(self):
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="a"),
                mcp_types.TextContent(type="text", text="b"),
            ],
        )
        assert _convert_call_tool_result("t", result) == ["a", "b"]

    def test_empty_content(self):
        result = mcp_types.CallToolResult(content=[])
        assert _convert_call_tool_result("t", result) is None

    def test_text_wins_over_structured_content(self):
        # Text blocks are the spec's canonical serialization of structuredContent.
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text='{"x": 1}')],
            structuredContent={"x": 1},
        )
        assert _convert_call_tool_result("t", result) == '{"x": 1}'

    def test_structured_content_without_text(self):
        result = mcp_types.CallToolResult(content=[], structuredContent={"x": 1})
        assert _convert_call_tool_result("t", result) == {"x": 1}

    def test_error_raises(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="bad input")],
            isError=True,
        )
        with pytest.raises(RuntimeError, match="bad input"):
            _convert_call_tool_result("t", result)

    def test_error_without_text_raises_generic(self):
        result = mcp_types.CallToolResult(content=[], isError=True)
        with pytest.raises(RuntimeError, match="MCP tool 'mytool' call failed"):
            _convert_call_tool_result("mytool", result)

    def test_image_content_becomes_message_with_file(self):
        # 1x1 transparent PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="here is your image"),
                mcp_types.ImageContent(type="image", data=png_b64, mimeType="image/png"),
            ],
        )
        converted = _convert_call_tool_result("t", result)
        assert isinstance(converted, Message)
        assert any(isinstance(c, TextContent) and c.text == "here is your image" for c in converted.content)
        assert any(isinstance(c, FileContent) for c in converted.content)
