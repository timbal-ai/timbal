import asyncio
import contextlib
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


class TestMCPServerCopies:
    """An unopened server is configuration: it copies and pickles, as it did before connections were per run (no lock
    nor loop in its state until it connects)."""

    @pytest.mark.parametrize(
        "config",
        [{"transport": "stdio", "command": "npx", "args": ["server"]}, {"transport": "http", "service": "api"}],
    )
    def test_an_unopened_server_copies_and_pickles(self, config):
        import copy
        import pickle

        from pydantic import BaseModel

        server = MCPServer(**config)
        for copied in (copy.deepcopy(server), server.model_copy(deep=True), pickle.loads(pickle.dumps(server))):
            assert copied.model_dump() == server.model_dump() and copied is not server

        class Holder(BaseModel):
            server: MCPServer = MCPServer(**config)  # a field default: copied for each instance

        assert Holder().server.model_dump() == server.model_dump()
        agent = Agent(name="agent", model="openai/gpt-4o-mini", tools=[server])
        assert agent.model_copy(deep=True).name == "agent"


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
        """In the W3C format whatever the global propagator is: one configured for B3 writes no traceparent."""
        import types

        from timbal.state import set_call_id, set_run_context
        from timbal.state.context import RunContext

        valid = types.SimpleNamespace(is_valid=True)
        trace_mod = types.ModuleType("opentelemetry.trace")
        trace_mod.get_current_span = lambda: types.SimpleNamespace(get_span_context=lambda: valid)
        propagate_mod = types.ModuleType("opentelemetry.propagate")  # the global propagator: B3 here
        propagate_mod.inject = lambda carrier: carrier.update(b3="a" * 32 + "-" + "b" * 16 + "-1")
        tracecontext_mod = types.ModuleType("opentelemetry.trace.propagation.tracecontext")

        class TraceContextTextMapPropagator:
            def inject(self, carrier):
                carrier.update(traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01", tracestate="k=v")

        tracecontext_mod.TraceContextTextMapPropagator = TraceContextTextMapPropagator
        propagation_mod = types.ModuleType("opentelemetry.trace.propagation")
        propagation_mod.tracecontext = tracecontext_mod
        trace_mod.propagation = propagation_mod
        root = types.ModuleType("opentelemetry")
        root.trace, root.propagate = trace_mod, propagate_mod
        for name, module in (
            ("opentelemetry", root),
            ("opentelemetry.trace", trace_mod),
            ("opentelemetry.propagate", propagate_mod),
            ("opentelemetry.trace.propagation", propagation_mod),
            ("opentelemetry.trace.propagation.tracecontext", tracecontext_mod),
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
    async def test_close_from_another_loop_leaves_the_runs_connection_to_the_run(self, platform):
        """``close()`` through ``asyncio.run`` in a worker thread, during a run: the run's connection belongs to the
        run, on the run's loop. It is not closed from there, nor a reason to fail (``main``'s close() never touched a
        call's connection, from any loop): the run's next step reuses it, and the run's end closes it."""
        from timbal import Tool
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        outcome = []

        def offloaded() -> str:
            asyncio.run(asyncio.wait_for(service.close(), 20))
            outcome.append("closed" if opened[0][0]._session is None else "still open")
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
        assert outcome == ["still open"]  # left to the run by the close() in the worker
        assert len(opened) == 1  # the run's next step listed tools on the same connection
        assert opened[0][0]._session is None and not service._run_connections  # closed at the run's end
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

    def test_a_stream_closed_from_another_loop_leaves_its_session_to_a_close_on_its_own(self, platform, monkeypatch):
        """A streaming tool opened the run's platform session on an ``asyncio.Runner``'s loop; another loop closes
        the stream while that one is paused. The run's end there closes nothing (the session is the Runner loop's):
        its hook raises, which the run logs; the connection stays, and a ``close()`` on its own loop closes it."""
        from timbal import Tool
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        ended, raised = MCPServer._run_ended, []

        def recording(self, key, owner):
            try:
                return ended(self, key, owner)
            except RuntimeError as error:
                raised.append(str(error))
                raise

        monkeypatch.setattr(MCPServer, "_run_ended", recording)

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
            assert len(raised) == 1 and "close() on the loop that owns the connection" in raised[0]
            assert opened[0][0]._session is not None and service._run_connections
            runner.run(asyncio.wait_for(service.close(), 30))
            assert opened[0][0]._session is None and not service._run_connections

    def test_a_stream_closed_from_another_loop_after_close_ends_quietly(self, platform):
        """As above, but ``close()`` on the Runner's loop closed the run's connection first: the run's end on the other
        loop has nothing left to close, so its hook returns instead of raising a failure the run logs
        (``run_end_hook_failed``, saying the connection still had to close; review on #334)."""
        import structlog
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
            runner.run(asyncio.wait_for(service.close(), 30))
            assert opened[0][0]._session is None and not service._run_connections and not service._unclosed
            with structlog.testing.capture_logs() as logs:
                asyncio.run(asyncio.wait_for(stream.aclose(), 10))  # from another loop, the owner paused
            assert not [entry for entry in logs if entry.get("event") == "run_end_hook_failed"], logs

    @pytest.mark.asyncio
    async def test_outside_a_run_each_call_closes_its_own(self, platform):
        opened, _ = platform
        service = MCPServer(transport="http", service="api")
        from timbal.state import set_run_context

        set_run_context(None)
        tools = await service.resolve()
        assert [t.name for t in tools] == ["echo_meta"]
        assert len(opened) == 1 and opened[0][0]._session is None and not service._run_connections

    @pytest.mark.asyncio
    async def test_a_close_that_never_ends_holds_the_run_for_a_bounded_time(self, platform, monkeypatch):
        """The platform never answers the session's teardown: the run still returns, ``_RUN_END_CLOSE_S`` after its
        end (a call's own close ran inside the run's ``timeout`` before connections were per run), and the close goes
        on behind it."""
        from timbal.core import mcp as mcp_module
        from timbal.core.test_model import TestModel
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        opened, _ = platform
        release = asyncio.Event()
        real_close = MCPServer.close

        async def close(server):
            if server._first_party:  # the run's connection: its teardown hangs until the test lets it go
                await release.wait()
            await real_close(server)

        monkeypatch.setattr(MCPServer, "close", close)
        monkeypatch.setattr(mcp_module, "_RUN_END_CLOSE_S", 0.2)
        service = MCPServer(transport="http", service="api")
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        agent = Agent(name="agent", model=TestModel(handler=self.calls(1)), tools=[service])
        try:
            out = await asyncio.wait_for(agent(prompt="go").collect(), 5)
            assert out.status.code == "success", out.error
            assert len(opened) == 1 and opened[0][0]._session is not None  # still closing, behind the run
        finally:
            release.set()
        await until(lambda: opened[0][0]._session is None)
        await asyncio.wait_for(service.close(), 10)

    @pytest.mark.asyncio
    async def test_a_runnable_out_of_time_does_not_wait_for_its_close(self, monkeypatch):
        """A root Tool with a 200 ms timeout used the platform connection, then ran past its budget: its timeout result
        comes back at once, the close going on behind it. A call's own close ran inside the budget before connections
        were per run; a fresh 10 s wait at the run's end after the budget had run out broke that."""
        from timbal.core.tool import Tool
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        closing, release = asyncio.Event(), asyncio.Event()

        class SlowTeardown:
            async def close(self):
                closing.set()
                await release.wait()

        monkeypatch.setattr(
            "timbal.platform.utils._resolve_url_and_headers",
            lambda *_: ("https://api.example.test/mcp", {"Authorization": "Bearer run-a"}),
        )
        monkeypatch.setattr(MCPServer, "_new_project_server", lambda _self, _url, _headers: SlowTeardown())
        service = MCPServer(transport="http", service="api")

        async def handler():
            async with service._project_connection():
                pass  # the run's connection: opened, and kept for the run's next call
            await asyncio.Event().wait()  # then the tool runs past its timeout

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        try:
            out = await asyncio.wait_for(Tool(name="slow", handler=handler, timeout=0.2)().collect(), 5)
            assert out.status.code == "timeout"
            await until(closing.is_set)
            assert service._close_tasks  # still closing, behind the result
        finally:
            release.set()
        await until(lambda: not service._close_tasks)
        await asyncio.wait_for(service.close(), 10)


class TestPlatformConnectionLifetime:
    @pytest.mark.asyncio
    async def test_a_cut_close_does_not_cut_the_session_close(self):
        """close() is cancelled after it let go of the connection and started its close: the close goes on, and a
        later close() waits for it instead of returning with the session still open."""
        from timbal.core.mcp import _RunConnection

        release, closed = asyncio.Event(), []

        class SlowClose:
            async def close(self):
                await release.wait()
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(SlowClose(), "fingerprint", asyncio.get_running_loop())
        closing = asyncio.ensure_future(service.close())
        await until(lambda: service._close_tasks)  # let go of, its close started
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert service._close_tasks and not closed  # still closing
        later = asyncio.ensure_future(service.close())
        done, _ = await asyncio.wait({later}, timeout=0.3)
        assert not done  # waiting for that close
        release.set()
        await asyncio.wait_for(later, 10)
        assert closed == [True]

    @pytest.mark.asyncio
    async def test_a_session_close_that_is_cut_is_kept_for_the_next_close(self):
        """The close task itself cancelled (a loop cancelling its tasks) before the session closed: the connection is
        kept, ended and out of the pool, and the next close() closes it."""
        from timbal.core.mcp import _RunConnection

        started, closed = asyncio.Event(), []

        class CutOnce:
            calls = 0

            async def close(self):
                CutOnce.calls += 1
                if CutOnce.calls == 1:
                    started.set()
                    await asyncio.Event().wait()  # cut here
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(CutOnce(), "fingerprint", asyncio.get_running_loop())
        task = service._retire_run_connection("run")
        await asyncio.wait_for(started.wait(), 10)
        task.cancel()
        await asyncio.wait({task})
        await until(lambda: service._unclosed)  # kept, out of the pool
        assert next(iter(service._unclosed.values()))[1].ended and not service._run_connections
        await asyncio.wait_for(service.close(), 10)
        assert closed == [True] and not service._run_connections and not service._unclosed

    @staticmethod
    async def cut(service, key, entry):
        """``entry``'s close cut before it closed: kept for a later close (``_close_done`` of a cancelled close)."""
        task = asyncio.ensure_future(asyncio.sleep(30))
        task.cancel()
        await asyncio.wait({task})
        service._close_done(key, entry, task)

    @pytest.mark.asyncio
    async def test_a_closed_run_loop_drops_the_runs_cut_connections(self):
        """The run's loop closed (and took its sessions): its cut connections go too, credentials included."""
        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                pass

        service = MCPServer(transport="http", service="api")
        gone = asyncio.new_event_loop()
        gone.close()
        await self.cut(service, "run", _RunConnection(Server(), "fingerprint", gone))
        service._run_hooked["run"] = gone
        assert service._run_ended("run", gone) is None
        assert not service._unclosed and "run" not in service._run_hooked

    @pytest.mark.asyncio
    async def test_a_session_close_cut_while_close_waits_is_reported(self):
        """The session's own close task cancelled while close() waits for it: the session is still open (kept for
        the next close()), and close() says so instead of returning as if it were closed."""
        from timbal.core.mcp import _RunConnection

        started = asyncio.Event()

        class Hangs:
            async def close(self):
                started.set()
                await asyncio.Event().wait()

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(Hangs(), "fingerprint", asyncio.get_running_loop())
        closing = asyncio.ensure_future(service.close())
        await asyncio.wait_for(started.wait(), 10)
        next(iter(service._close_tasks)).cancel()
        with pytest.raises(RuntimeError, match="still open"):
            await asyncio.wait_for(closing, 10)
        assert service._unclosed  # owed, for the next close()

    @pytest.mark.asyncio
    async def test_two_close_calls_both_wait_for_the_retry_one_of_them_took(self):
        """After a cut, two close() at once: the second sees the retry the first took and waits for it."""
        from timbal.core.mcp import _RunConnection

        release, closed = asyncio.Event(), []

        class Slow:
            async def close(self):
                await release.wait()
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        await self.cut(service, "run", _RunConnection(Slow(), "fingerprint", asyncio.get_running_loop()))
        first = asyncio.ensure_future(service.close())
        second = asyncio.ensure_future(service.close())
        done, _ = await asyncio.wait({second}, timeout=0.3)
        assert not done  # waiting for the first's retry
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 10)
        assert closed == [True]

    @pytest.mark.asyncio
    async def test_a_later_call_of_the_run_does_not_reuse_a_cut_connection(self, monkeypatch):
        """Its server may still have a session closing: the run's next call gets a connection of its own, and the
        run's end (or close()) closes the cut one again."""
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        servers, started = [], asyncio.Event()

        class Server:
            def __init__(self):
                self.closes = 0

            async def close(self):
                self.closes += 1
                if self is servers[0] and self.closes == 1:
                    started.set()
                    await asyncio.Event().wait()  # cut here

        def new(_self, _url, _headers):
            servers.append(Server())
            return servers[-1]

        monkeypatch.setattr("timbal.platform.utils._resolve_url_and_headers", lambda *_: ("https://x.test/mcp", {}))
        monkeypatch.setattr(MCPServer, "_new_project_server", new)
        monkeypatch.setattr("timbal.core.mcp._run_connection_key", lambda _context: "run")
        context = RunContext()
        set_run_context(context)
        service = MCPServer(transport="http", service="api")
        try:
            async with service._project_connection() as first:
                pass
            task = service._retire_run_connection("run")
            await asyncio.wait_for(started.wait(), 10)
            task.cancel()
            await asyncio.wait({task})
            async with service._project_connection() as second:
                assert second is not first
            await asyncio.wait_for(service.close(), 10)
            assert first.closes == 2 and second.closes == 1
        finally:
            set_run_context(None)

    def test_closes_of_a_closed_loop_leave_both_collections(self):
        """A close that finished on a loop that stopped before its done callback ran: dropped from the per-run sets
        too, so the run keeps neither the task nor its loop."""
        gone = asyncio.new_event_loop()

        async def nothing():
            pass

        task = gone.create_task(nothing())
        gone.run_until_complete(task)
        gone.close()
        service = MCPServer(transport="http", service="api")
        service._close_tasks.add(task)
        service._run_closes["run"] = {task}
        asyncio.run(asyncio.wait_for(service.close(), 10))
        assert not service._close_tasks and not service._run_closes

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

    @pytest.mark.asyncio
    async def test_close_holds_what_it_took_so_a_new_connection_is_never_counted_as_one_of_them(self):
        """close() let go of a connection, which closed, and still waits (here for the server's own lock): it holds
        that connection, so a new one a run opens meanwhile cannot take its id (a freed one's id is the next one's
        as a rule) and be counted as a connection close() set out to close and left open."""
        import gc
        import weakref

        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                pass

        class Connection(_RunConnection):
            __slots__ = ("__weakref__",)  # for the test's weak reference only

        service = MCPServer(transport="http", service="api")
        loop = asyncio.get_running_loop()
        service._run_connections["run"] = Connection(Server(), "fingerprint", loop)
        taken = weakref.ref(service._run_connections["run"])
        ident = id(taken())
        holding, free = asyncio.Event(), asyncio.Event()

        async def hold():
            async with service._get_lock():
                holding.set()
                await free.wait()

        holder = asyncio.ensure_future(hold())
        await asyncio.wait_for(holding.wait(), 10)
        closing = asyncio.ensure_future(service.close())  # its own session's close waits for the lock
        try:
            await until(lambda: "run" not in service._run_connections and not service._close_tasks)  # closed
            gc.collect()
            assert taken() is not None  # held by close() until it returns
            others = []  # a run's first call meanwhile: at the taken one's address, if that one were freed
            for _ in range(100):
                others.append(Connection(Server(), "fingerprint", loop))
                if id(others[-1]) == ident:
                    break
            service._run_connections["other"] = others[-1]
        finally:
            free.set()
        await asyncio.wait_for(closing, 10)  # nothing it set out to close is left open
        await asyncio.wait_for(holder, 10)

    @pytest.mark.parametrize("reader", ["_wait_loop_closes", "_forget_close"])
    @pytest.mark.asyncio
    async def test_closes_are_read_from_a_copy_while_another_loop_changes_them(self, reader):
        """Another loop's thread starts or drops its own closes (its runs') while close() goes over them: each reader
        goes over a copy. Going over the set or dict itself raises "changed size during iteration". The change is
        made from inside the read here (a close's ``get_loop()``, a hash), where another thread's would land."""
        service = MCPServer(transport="http", service="api")
        elsewhere = asyncio.new_event_loop()
        elsewhere.close()

        class Close:
            """A close of another loop, which starts one more while it is read."""

            def get_loop(self):
                service._close_tasks.add(Close())
                return elsewhere

        class Forgotten:
            """A close being forgotten: while it is looked up, another run starts closing."""

            def __hash__(self):
                service._run_closes.setdefault(f"run-{len(service._run_closes)}", set())
                return 0

        if reader == "_forget_close":
            service._run_closes.update({"a": {object()}, "b": {object()}})
            service._forget_close(Forgotten())
            return
        service._close_tasks.update({Close(), Close()})
        await service._wait_loop_closes()  # none of them is this loop's: nothing to wait for

    @pytest.mark.asyncio
    async def test_a_close_that_starts_after_the_settle_is_not_reported(self, monkeypatch):
        """A call's own close (its connection was in use, left to it) starting after close() waited for the closes
        going: still closing as designed, not reported as cancelled."""
        from timbal.core.mcp import _RunConnection

        release = asyncio.Event()

        class Slow:
            async def close(self):
                await release.wait()

        settle = MCPServer._settle_closes

        async def settle_then_a_call_closes_its_own(self, loop):
            await settle(self, loop)
            self._retire_run_connection("late", _RunConnection(Slow(), "fingerprint", loop))

        monkeypatch.setattr(MCPServer, "_settle_closes", settle_then_a_call_closes_its_own)
        service = MCPServer(transport="http", service="api")
        try:
            await asyncio.wait_for(service.close(), 10)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*service._close_tasks), 10)

    @pytest.mark.asyncio
    async def test_a_session_close_that_fails_is_kept_for_the_next_close(self):
        """The connection's own close() raised (what it could not close is still open), not cancelled: the connection
        is kept for the next close(), and this one says it is still open."""
        from timbal.core.mcp import _RunConnection

        calls = []

        class FailsOnce:
            async def close(self):
                calls.append(True)
                if len(calls) == 1:
                    raise RuntimeError("its session is still shutting down")

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(FailsOnce(), "fingerprint", asyncio.get_running_loop())
        with pytest.raises(RuntimeError, match="still open"):
            await asyncio.wait_for(service.close(), 10)
        assert service._unclosed
        await asyncio.wait_for(service.close(), 10)  # retried
        assert len(calls) == 2 and not service._unclosed


class TestStandaloneServerClose:
    @pytest.mark.asyncio
    async def test_a_retried_close_waits_for_the_owner_a_cut_close_signalled(self):
        """close() forgets the session, signals its owner task, and is cut waiting for it: the owner is still
        shutting down, and the next close() waits for it."""
        server = MCPServer(transport="stdio", command="unused")
        closed, release = asyncio.Event(), asyncio.Event()

        async def owner():
            await closed.wait()
            await release.wait()  # a transport that takes its time to shut down

        owner_task = asyncio.ensure_future(owner())
        server._session_task, server._session_closed = owner_task, closed
        server._owner_task = server._session_task  # as _open_session leaves them
        first = asyncio.ensure_future(server.close())
        await asyncio.wait_for(closed.wait(), 10)  # signalled
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        retry = asyncio.ensure_future(server.close())
        done, _ = await asyncio.wait({retry}, timeout=0.3)
        assert not done  # waiting for that owner
        release.set()
        await asyncio.wait_for(retry, 10)
        assert owner_task.done()

    @pytest.mark.asyncio
    async def test_the_owner_is_recorded_before_the_session_is_forgotten(self, monkeypatch):
        """A close() on another thread must see the session or its owner, never neither: the owner is in
        _owners_closing by the time the refs go."""
        server = MCPServer(transport="stdio", command="unused")
        closed = asyncio.Event()

        async def owner():
            await closed.wait()

        task = asyncio.ensure_future(owner())
        server._session_task, server._session_closed = task, closed
        server._owner_task = server._session_task  # as _open_session leaves them
        forget, seen = MCPServer._forget_session, []

        def watched(self, *args):
            if self._session_task is not None:
                seen.append(self._session_task in self._owners_closing)
            return forget(self, *args)

        monkeypatch.setattr(MCPServer, "_forget_session", watched)
        await asyncio.wait_for(server.close(), 10)
        assert seen == [True]

    @pytest.mark.asyncio
    async def test_one_owner_reference_from_the_handshake_to_the_sessions_end(self, monkeypatch):
        """The owner task is the server's owner reference as soon as it exists, stays it when the session is
        published, and goes only once close() has recorded it as shutting down."""
        gates = {"release": asyncio.Event()}

        async def run_session(_self, ready, closed):
            await gates["release"].wait()
            ready.set_result(object())
            await closed.wait()

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        connecting = asyncio.ensure_future(server._connect())
        await until(lambda: server._owner_task is not None)
        owner = server._owner_task
        gates["release"].set()
        await asyncio.wait_for(connecting, 10)
        assert server._owner_task is owner is server._session_task  # the same reference once published
        await asyncio.wait_for(server.close(), 10)
        assert server._owner_task is None and owner.done()

    @pytest.mark.asyncio
    async def test_a_connect_cut_twice_during_its_handshake_leaves_no_opening_owner(self, monkeypatch):
        """Cancelled in its handshake, then again while it waits for its owner to shut down: the opening owner is
        still cleared, so no later close() finds a session that is gone, nor a reconnect a stale owner."""
        closing, release = asyncio.Event(), asyncio.Event()

        async def run_session(_self, _ready, closed):  # a handshake that never ends, and a slow teardown
            await closed.wait()
            closing.set()
            await release.wait()

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        connecting = asyncio.ensure_future(server._connect())
        await until(lambda: server._owner_task is not None)
        owner = server._owner_task
        connecting.cancel()
        await asyncio.wait_for(closing.wait(), 10)  # waiting for its owner's teardown
        connecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connecting
        try:
            assert server._owner_task is None
            assert owner in server._owners_closing  # still shutting down: a later close() waits for it
        finally:
            release.set()
            await asyncio.wait_for(asyncio.wait({owner}), 10)

    @pytest.mark.asyncio
    async def test_a_reconnect_after_close_let_go_of_the_session_is_not_its_failure(self, monkeypatch):
        """A call reconnects once close() has closed the old session: the new handshake is not close()'s to report,
        and close() succeeds."""
        gates = {}

        async def run_session(_self, ready, closed):
            if "release" in gates:  # the reconnect: its handshake waits
                await gates["release"].wait()
            ready.set_result(object())
            await closed.wait()

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        await asyncio.wait_for(server._connect(), 10)  # the session close() is for
        wait_owners, reconnect = MCPServer._wait_owners_closing, {}

        async def then_a_call_reconnects(self, loop):
            await wait_owners(self, loop)
            gates["release"] = asyncio.Event()
            reconnect["task"] = asyncio.ensure_future(self._connect())
            await until(lambda: self._owner_task is not None)  # its handshake under way as close() counts

        monkeypatch.setattr(MCPServer, "_wait_owners_closing", then_a_call_reconnects)
        await asyncio.wait_for(server.close(), 10)
        gates["release"].set()
        await asyncio.wait_for(reconnect["task"], 10)
        monkeypatch.setattr(MCPServer, "_wait_owners_closing", wait_owners)
        await asyncio.wait_for(server.close(), 10)


class TestCloseOnAnotherLoop:
    """``close()`` and a run's end act only on what their own event loop owns. A session of another loop that has
    not closed is refused (``RuntimeError``) before anything is touched; another loop's run connections are left to
    their run; what a closed loop held is forgotten."""

    OWNS = r"close\(\) on the loop that owns the connection"

    @staticmethod
    def owner_thread():
        import threading

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()

        async def cancel_the_rest():  # every task still pending there, cancelled and awaited on its loop
            others = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in others:
                task.cancel()
            await asyncio.gather(*others, return_exceptions=True)

        def stop():
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(cancel_the_rest(), loop).result(10)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

        return loop, stop

    @staticmethod
    def state(server):
        """What a close() changes: the session's references and owners, the caches, the run connections."""
        return (
            server._session,
            server._session_task,
            server._owner_task,
            server._session_closed,
            set(server._owners_closing),
            server._tools_cache,
            server._tools_cache_run_id,
            server._tools_stale,
            dict(server._inflight),
            dict(server._run_connections),
            dict(server._unclosed),
            set(server._close_tasks),
            {key: set(closes) for key, closes in server._run_closes.items()},
            dict(server._run_hooked),
        )

    @staticmethod
    def call_on(loop):
        """The record of a tools/call in flight on ``loop``, as ``_begin_call`` leaves it (its abort future is made
        there)."""
        from types import SimpleNamespace

        return SimpleNamespace(abort=loop.create_future())

    @staticmethod
    def open_session(monkeypatch):
        """A transport that connects at once and stays up until closed."""

        async def run_session(_self, ready, closed):
            ready.set_result(object())
            await closed.wait()

        monkeypatch.setattr(MCPServer, "_run_session", run_session)

    def test_a_session_of_another_running_loop_is_refused_before_anything_is_touched(self, monkeypatch):
        """A standalone server's session is open on a loop running in another thread: close() from this loop raises
        and changes nothing (the owner task, the locks and the transport are that loop's; ``main``'s close() failed
        there too). Its lock is not taken. close() on that loop closes it."""
        self.open_session(monkeypatch)
        owner_loop, stop = self.owner_thread()
        server = MCPServer(transport="stdio", command="unused")
        try:
            asyncio.run_coroutine_threadsafe(server._connect(), owner_loop).result(10)
            server._tools_cache, server._tools_cache_run_id = ["the list"], "run"
            server._inflight[1] = self.call_on(owner_loop)
            before = self.state(server)
            with pytest.raises(RuntimeError, match=self.OWNS) as refused:
                asyncio.run(asyncio.wait_for(server.close(), 10))
            assert self.state(server) == before
            assert not server._get_lock().locked() and not server._get_tools_lock().locked()
            # It says how: on a Runner, or handed to a loop running in another thread (blocking, or awaited).
            assert "runner.run(server.close())" in str(refused.value)
            assert "asyncio.run_coroutine_threadsafe(server.close(), loop).result()" in str(refused.value)
            assert "await asyncio.wrap_future(" in str(refused.value)

            async def as_it_says():  # the awaited form, from async code on another loop
                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(server.close(), owner_loop))

            asyncio.run(asyncio.wait_for(as_it_says(), 10))
            assert server._session is None and server._owner_task is None and server._tools_cache is None
        finally:
            stop()

    def test_a_handshake_under_way_on_another_loop_is_refused(self, monkeypatch):
        """The first session is still in its handshake on its loop (its owner task recorded, nothing published): close()
        from another loop is refused, and the handshake goes on undisturbed; close() on its loop closes it."""
        import threading

        owner_loop, stop = self.owner_thread()
        shaking, gates = threading.Event(), {}

        async def run_session(_self, ready, closed):  # the transport: its handshake waits for the test
            gates["release"] = asyncio.Event()
            shaking.set()
            await gates["release"].wait()
            ready.set_result(object())
            await closed.wait()

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        try:
            connecting = asyncio.run_coroutine_threadsafe(server._connect(), owner_loop)
            assert shaking.wait(10)
            before = self.state(server)
            with pytest.raises(RuntimeError, match=self.OWNS):
                asyncio.run(asyncio.wait_for(server.close(), 10))
            assert self.state(server) == before and server._owner_task is not None
            owner_loop.call_soon_threadsafe(gates["release"].set)
            connecting.result(10)  # published, as if close() had not been called
            asyncio.run_coroutine_threadsafe(server.close(), owner_loop).result(10)
            assert server._session is None and server._owner_task is None
        finally:
            stop()

    def test_a_session_of_a_paused_loop_is_refused_and_closed_there(self, monkeypatch):
        """An ``asyncio.Runner`` between two ``run()`` calls owns the session: close() from another loop is refused
        instead of waiting for that loop, and close() on it closes the session."""
        self.open_session(monkeypatch)
        server = MCPServer(transport="stdio", command="unused")
        with asyncio.Runner() as runner:
            runner.run(asyncio.wait_for(server._connect(), 10))
            before = self.state(server)
            with pytest.raises(RuntimeError, match=self.OWNS):
                asyncio.run(asyncio.wait_for(server.close(), 10))
            assert self.state(server) == before
            runner.run(asyncio.wait_for(server.close(), 10))
        assert server._session is None and server._owner_task is None

    def test_an_owner_a_cut_close_signalled_on_another_loop_is_refused(self):
        """close() on the session's loop signalled the owner and was cut waiting for it (the session's refs are gone,
        its owner still shutting down): a retry from another loop is refused, changing nothing; a retry on that loop
        waits for the owner."""
        owner_loop, stop = self.owner_thread()
        server = MCPServer(transport="stdio", command="unused")
        try:

            async def install():
                closed, release = asyncio.Event(), asyncio.Event()

                async def owner():
                    await closed.wait()
                    await release.wait()  # a transport that takes its time to shut down

                task = asyncio.ensure_future(owner())
                server._session_task, server._session_closed = task, closed
                server._owner_task = server._session_task  # as _open_session leaves them
                return task, closed, release

            owner, closed, release = asyncio.run_coroutine_threadsafe(install(), owner_loop).result(10)
            first = asyncio.run_coroutine_threadsafe(server.close(), owner_loop)
            asyncio.run(until(closed.is_set))  # the first close() has signalled the owner
            first.cancel()
            assert server._owner_task is None and owner in server._owners_closing
            before = self.state(server)
            with pytest.raises(RuntimeError, match=self.OWNS):
                asyncio.run(asyncio.wait_for(server.close(), 10))
            assert self.state(server) == before
            retry = asyncio.run_coroutine_threadsafe(server.close(), owner_loop)
            owner_loop.call_soon_threadsafe(release.set)
            retry.result(10)
            assert owner.done()  # waited for, there
        finally:
            stop()

    @staticmethod
    def ended_owner(monkeypatch, until_ended, error=None):
        """A session on a loop of its own thread whose transport ends (the server process exits) once ``end()`` is
        called, ``until_ended`` running in the owner task as it ends (it pauses the loop, or queues a callback that
        does not yield, ahead of the task's done callbacks). Returns (server, loop, thread, end)."""
        import threading

        owner_loop, gates = asyncio.new_event_loop(), {}

        async def run_session(_self, ready, _closed):
            gates["drop"] = asyncio.Event()
            ready.set_result(object())
            await gates["drop"].wait()  # the transport, until the server process exits
            until_ended(asyncio.get_running_loop())
            if error is not None:
                raise error

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        thread = threading.Thread(target=owner_loop.run_forever)
        thread.start()
        asyncio.run_coroutine_threadsafe(server._connect(), owner_loop).result(10)
        return server, owner_loop, thread, lambda: owner_loop.call_soon_threadsafe(gates["drop"].set)

    @staticmethod
    def stop_ended(owner_loop, thread):
        """Let the owner's loop run what it has pending (its task's done callbacks), then stop and close it."""
        import threading

        if not owner_loop.is_closed():
            if not thread.is_alive():
                thread = threading.Thread(target=owner_loop.run_forever)
                thread.start()
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            thread.join(10)
            owner_loop.close()

    @pytest.mark.parametrize("owner", ["paused", "busy"])
    def test_a_session_whose_owner_has_ended_is_left_to_its_loop(self, monkeypatch, owner):
        """The transport's owner task ended on its loop (the server process exited), and that loop paused, or is busy
        in a callback, before the task's done callback forgot the session. Nothing of it is live: close() from
        another loop does not refuse (as on 0e7071e4), and writes nothing of it, that loop being free to reconnect
        meanwhile; the done callback forgets it once that loop runs it (review on #334)."""
        import threading

        hold, holding = threading.Event(), threading.Event()

        def block():  # a callback that does not yield
            holding.set()
            hold.wait(30)

        def until_ended(loop):
            if owner == "paused":
                loop.stop()  # stops before the task's done callbacks run
            else:
                loop.call_soon(block)  # queued ahead of the task's done callbacks

        server, owner_loop, thread, end = self.ended_owner(monkeypatch, until_ended)
        try:
            server._tools_cache, server._inflight[1] = ["the list"], self.call_on(owner_loop)
            end()
            if owner == "paused":
                thread.join(10)
                assert not thread.is_alive() and not owner_loop.is_running()
            else:
                assert holding.wait(10)
            assert server._owner_task.done() and server._session is not None  # its done callback has not run
            before = self.state(server)
            asyncio.run(asyncio.wait_for(server.close(), 10))
            assert self.state(server) == before
        finally:
            hold.set()
            self.stop_ended(owner_loop, thread)
        assert server._session is None and server._session_task is None and server._owner_task is None

    @pytest.mark.parametrize("then", ["its loop runs again", "its loop closed"])
    def test_an_ended_owners_error_is_read_once_whoever_forgets_it(self, monkeypatch, then):
        """The transport died with an error and its loop paused before the owner task's done callback ran. close()
        from another loop leaves it to that loop, whose callback reads and logs the error when it runs; if that loop
        closes instead (its callbacks dropped), close() forgets the session and reads and logs the error itself, as a
        close on its loop does. Either way once, and asyncio reports nothing as never retrieved (Bugbot on #334)."""
        import gc

        from timbal.core import mcp as mcp_module

        logged, unretrieved = [], []

        class Recorder:
            def __getattr__(self, level):
                return lambda _event, **fields: logged.append(fields.get("error"))

        monkeypatch.setattr(mcp_module, "logger", Recorder())
        error = ConnectionError("the server process exited")
        server, owner_loop, thread, end = self.ended_owner(monkeypatch, lambda loop: loop.stop(), error)
        owner_loop.set_exception_handler(lambda _loop, context: unretrieved.append(context["message"]))
        try:
            end()
            thread.join(10)
            assert server._owner_task.done() and server._session is not None  # its done callback has not run
            if then == "its loop closed":
                owner_loop.close()  # its pending callbacks dropped: they never run
            asyncio.run(asyncio.wait_for(server.close(), 10))
        finally:
            self.stop_ended(owner_loop, thread)
        assert server._session is None
        assert logged.count("ConnectionError: the server process exited") == 1
        del error
        gc.collect()
        assert not [message for message in unretrieved if "never retrieved" in message]

    def test_an_ended_owners_replacement_and_its_calls_are_left_alone(self):
        """close() on another loop finds an owner task that has ended, on a loop that still runs; that loop publishes a
        new session, with a call in flight on it, just as close() decides what to do with the ended one. close() writes
        nothing of that loop's: the new session's references and the call's record stay (cleared, the new session was
        left open with nothing referring to it, and the call lost its elicitation prompt; review on #334)."""
        other = asyncio.new_event_loop()  # the owner's loop: alive, not this one
        ended, new, closed = other.create_future(), other.create_future(), asyncio.Event()
        ended.set_result(None)
        fresh, call = object(), object()
        inside, published = [], []

        class Raced(MCPServer):
            async def _close_own_session(self, loop):
                inside.append(True)
                try:
                    return await super()._close_own_session(loop)
                finally:
                    inside.clear()

            def __getattr__(self, name):
                value = super().__getattr__(name)
                if inside and name == "_owner_task" and not published:
                    published.append(True)  # it read the ended owner; the owner's loop now opens a new session
                    self.__pydantic_private__.update(
                        _owner_task=new, _session_task=new, _session=fresh, _session_closed=closed
                    )
                    self._inflight[7] = call
                return value

        server = Raced(transport="stdio", command="unused")
        server.__pydantic_private__.update(_owner_task=ended, _session_task=ended, _session=object())
        try:
            asyncio.run(asyncio.wait_for(server.close(), 10))
            assert published
            assert server._session is fresh and server._session_task is new and server._owner_task is new
            assert server._session_closed is closed and server._inflight == {7: call}
        finally:
            other.close()

    @pytest.mark.asyncio
    async def test_the_refusal_reads_the_owner_before_the_owners_shutting_down(self):
        """A close on the session's loop moves its owner to ``_owners_closing`` and then lets go of the reference.
        close() on another thread reads the reference first, so it finds the owner in one of the two, and refuses,
        even when that move happens between its two reads (reading ``_owners_closing`` first, it found neither and
        went on to take the session's locks on its own loop)."""
        paused = asyncio.new_event_loop()  # the owner's loop, not running
        owner = paused.create_future()
        closer, armed = asyncio.get_running_loop(), [True]

        class Raced(MCPServer):
            def __getattr__(self, name):
                value = super().__getattr__(name)
                if armed and name == "_owners_closing":
                    armed.clear()
                    before = set(value)  # what this read sees; then, on the owner's loop, the move:
                    value.add(owner)
                    self._forget_session()
                    return before
                return value

        server = Raced(transport="stdio", command="unused")
        server.__pydantic_private__.update(_session_task=owner, _owner_task=owner)
        try:
            with pytest.raises(RuntimeError, match=self.OWNS):
                server._refuse_other_loops(closer)
            assert not armed
        finally:
            paused.close()

    def test_a_close_that_looks_up_the_owner_as_the_handshake_publishes_still_finds_it(self, monkeypatch):
        """close() on another thread looks up the session's owner just as the handshake publishes the session, and the
        owner's loop pauses right after: the lookup still finds the owner, so close() refuses (the session is that
        loop's). With one field for the handshake and another once published, it read the first before the publish
        and the second after the publish had cleared it, and went on as if nothing were open."""
        import threading

        owner_loop = asyncio.new_event_loop()
        first = threading.Thread(target=owner_loop.run_forever)
        shaking, gates, connecting = threading.Event(), {}, {}

        async def run_session(_self, ready, closed):  # the transport: its handshake waits for the test
            gates["release"] = asyncio.Event()
            shaking.set()
            await gates["release"].wait()
            ready.set_result(object())
            await closed.wait()

        closer, armed = threading.current_thread(), []

        class Raced(MCPServer):
            def __getattr__(self, name):
                value = super().__getattr__(name)
                owner_fields = ("_owner_task", "_opening_task", "_session_task")
                if armed and name in owner_fields and threading.current_thread() is closer:
                    armed.clear()  # close()'s first look at the owner; before its next one:
                    owner_loop.call_soon_threadsafe(gates["release"].set)
                    connecting["future"].result(10)  # the handshake published the session
                    owner_loop.call_soon_threadsafe(owner_loop.stop)
                    first.join(10)  # and its loop paused
                return value

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = Raced(transport="stdio", command="unused")
        first.start()
        running = first
        try:
            connecting["future"] = asyncio.run_coroutine_threadsafe(server._connect(), owner_loop)
            assert shaking.wait(10)
            armed.append(True)
            with pytest.raises(RuntimeError, match=self.OWNS):
                asyncio.run(asyncio.wait_for(server.close(), 10))
            assert not armed and not owner_loop.is_running()
        finally:
            armed.clear()
            first.join(10)
            running = first if first.is_alive() else threading.Thread(target=owner_loop.run_forever)
            if running is not first:
                running.start()

            async def settle():  # on its own loop: the close happens there, then whatever is left is cancelled
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(server.close(), 10)
                others = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
                for task in others:
                    task.cancel()
                await asyncio.gather(*others, return_exceptions=True)

            asyncio.run_coroutine_threadsafe(settle(), owner_loop).result(20)
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            running.join(10)
            owner_loop.close()
        assert server._session is None

    def test_close_from_another_loop_clears_the_tool_cache_after_the_transport_dropped(self, monkeypatch):
        """The session's transport dropped (its owner ended, the session forgotten) on a loop now paused: close() from
        another loop has nothing to close there, and clears the tool cache where it runs, as close() always did."""

        async def run_session(_self, ready, _closed):
            ready.set_result(object())  # and the transport drops at once

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        with asyncio.Runner() as runner:

            async def connect_and_drop():
                await server._connect()
                await until(lambda: server._session is None)

            runner.run(asyncio.wait_for(connect_and_drop(), 10))
            server._tools_cache = ["the old list"]
            asyncio.run(asyncio.wait_for(server.close(), 10))
            assert server._tools_cache is None

    @pytest.mark.parametrize("how", ["published", "in its handshake"])
    @pytest.mark.asyncio
    async def test_a_session_whose_loop_closed_still_clears_the_caches(self, how):
        """The session's loop closed with it, nothing having cancelled its tasks: close() forgets the session and
        clears the tool list and the records of the calls in flight that died with it, as every close() does (a call
        left there would make every later elicitation look ambiguous, answered ``cancel``). A call that another loop,
        still running, has begun (it is about to reconnect) keeps its record: cleared, an elicitation during it would
        find no call to ask for, and the tool would complete without the question (Bugbot on #334)."""
        loop, other = asyncio.new_event_loop(), asyncio.new_event_loop()
        dead, theirs = self.call_on(loop), self.call_on(other)
        loop.close()

        class Owner:
            """The session's owner task, on that loop."""

            def get_loop(self):
                return loop

            def done(self):
                return False

        server = MCPServer(transport="stdio", command="unused")
        server._owner_task = Owner()
        if how == "published":
            server._session_task = server._owner_task  # as _open_session leaves them
        server._tools_cache, server._tools_cache_run_id, server._tools_stale = [object()], "run", True
        server._inflight[1], server._inflight[2] = dead, theirs
        try:
            await asyncio.wait_for(server.close(), 10)
            assert server._session_task is None and server._owner_task is None
            assert server._tools_cache is None and server._tools_cache_run_id is None and not server._tools_stale
            assert server._inflight == {2: theirs}
        finally:
            other.close()

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Known gap (CLAUDE.md, MCPServer 'One loop only'): forgetting a closed loop's session reference is a "
        "check and then a write, and a handshake another thread starts between the two loses its owner reference. "
        "Closing it takes a lock shared by every publish of a session; the window is a few bytecodes.",
    )
    @pytest.mark.asyncio
    async def test_a_handshake_another_thread_starts_as_close_forgets_a_closed_loops_owner_keeps_its_owner(self):
        """The session's loop closed in its handshake. close() checks that the owner reference is still that one and
        then lets go of it; a call on a loop running in another thread starts a handshake right after the check (its
        owner reference recorded). That reference should stay: without it, close() on another loop no longer finds
        the live handshake to refuse."""
        loop, other = asyncio.new_event_loop(), asyncio.new_event_loop()
        new = other.create_future()  # the other loop's handshake
        loop.close()
        reads = []

        class Owner:
            def get_loop(self):
                return loop

            def done(self):
                return False

        class Raced(MCPServer):
            async def _close_own_session(self, loop):
                reads.append(0)
                try:
                    return await super()._close_own_session(loop)
                finally:
                    reads.clear()

            def __getattr__(self, name):
                value = super().__getattr__(name)
                if reads and name == "_owner_task":
                    reads.append(name)
                    if len(reads) == 3:  # the check, after close() read the owner once: a handshake starts now
                        self.__pydantic_private__["_owner_task"] = new
                return value

        server = Raced(transport="stdio", command="unused")
        server.__pydantic_private__["_owner_task"] = Owner()
        try:
            await asyncio.wait_for(server.close(), 10)
            assert server._owner_task is new
        finally:
            other.close()

    @pytest.mark.parametrize("how", ["nothing open", "its session open here"])
    @pytest.mark.asyncio
    async def test_close_drops_only_the_records_of_calls_whose_loop_closed(self, monkeypatch, how):
        """close() on the loop that owns the session, or with nothing open, drops the records of the calls that died
        with a loop that has closed. A call on a loop that has not closed, this one or another thread's, keeps its
        record, which it removes as it ends (cleared, as before, an elicitation during it would find no call to ask
        for)."""
        self.open_session(monkeypatch)
        server = MCPServer(transport="stdio", command="unused")
        if how == "its session open here":
            await server._connect()
        gone, other = asyncio.new_event_loop(), asyncio.new_event_loop()
        mine, dead, theirs = self.call_on(asyncio.get_running_loop()), self.call_on(gone), self.call_on(other)
        gone.close()
        server._inflight.update({1: mine, 2: dead, 3: theirs})
        try:
            await asyncio.wait_for(server.close(), 10)
            assert server._session is None and server._owner_task is None and server._tools_cache is None
            assert server._inflight == {1: mine, 3: theirs}
        finally:
            other.close()

    def test_what_a_closed_loop_held_is_forgotten(self):
        """A run's loop closed without running its cleanup, taking its sessions with it: close() from another loop
        forgets its pooled and cut connections (they hold their credential headers) and its pending closes, closes
        nothing of them, and succeeds."""
        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                raise AssertionError("closed on the wrong loop")

        gone = asyncio.new_event_loop()
        pending = gone.create_future()  # a close that never ran its callback
        gone.close()
        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(Server(), "fingerprint", gone)
        cut = _RunConnection(Server(), "fingerprint", gone)
        service._unclosed[id(cut)] = ("run", cut)
        service._close_tasks.add(pending)
        service._run_closes["run"] = {pending}
        asyncio.run(asyncio.wait_for(service.close(), 10))
        assert not service._run_connections and not service._unclosed
        assert not service._close_tasks and not service._run_closes

    @pytest.mark.parametrize("owner", ["running", "paused"])
    def test_a_run_connection_of_another_loop_is_left_to_its_run(self, owner):
        """``service="api"``: a run's connections belong to the run, on its loop (running in another thread, or an
        ``asyncio.Runner`` between two ``run()`` calls). close() from another loop neither closes them nor raises
        (``main``'s close() never touched a call's connection, from any loop), changes nothing of them, and does not
        wait for a close already going there: they are left to their run's end or idle close, or a close() there."""
        import threading

        from timbal.core.mcp import _RunConnection

        if owner == "running":
            owner_loop, stop = self.owner_thread()

            def there(coroutine):
                return asyncio.run_coroutine_threadsafe(coroutine, owner_loop).result(10)

        else:
            runner = asyncio.Runner()
            owner_loop, stop = runner.get_loop(), runner.close

            def there(coroutine):
                return runner.run(asyncio.wait_for(coroutine, 10))

        closed, release = [], threading.Event()

        class Server:
            async def close(self):
                closed.append(True)

        class SlowClose:
            async def close(self):
                await asyncio.to_thread(release.wait, 30)
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        try:
            pooled, cut = _RunConnection(Server(), "fingerprint", owner_loop), _RunConnection(Server(), "f", owner_loop)
            service._run_connections["run"] = pooled
            service._unclosed[id(cut)] = ("run", cut)

            async def let_go():  # a credential change there: its close is going when close() comes
                service._run_connections["changed"] = _RunConnection(SlowClose(), "fingerprint", owner_loop)
                service._retire_run_connection("changed")

            there(let_go())
            before = self.state(service)
            asyncio.run(asyncio.wait_for(service.close(), 10))  # does not wait for the slow close
            assert self.state(service) == before and closed == [] and not pooled.ended
            release.set()
            there(service.close())
            assert closed == [True, True, True] and not service._run_connections and not service._unclosed
        finally:
            release.set()
            stop()

    def test_a_runs_end_on_another_loop_is_refused_and_leaves_the_connection_to_its_loop(self):
        """A run's end hook runs on another loop than the run's (its stream closed from there): it closes nothing and
        raises (the run logs it, ``run_end_hook_failed``), leaving the connection as it was but for its own hook
        record; close() on the run's loop closes it."""
        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()
        closed = []

        class Server:
            async def close(self):
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        try:
            entry = _RunConnection(Server(), "fingerprint", owner_loop)
            service._run_connections["run"] = entry
            service._run_hooked["run"] = owner_loop

            async def end_the_run():
                return service._run_ended("run", owner_loop)

            with pytest.raises(RuntimeError, match=self.OWNS) as refused:
                asyncio.run(asyncio.wait_for(end_the_run(), 10))
            assert "closes on its own loop once idle (60 s after its last call or a close that failed)" in str(
                refused.value
            )
            assert service._run_connections == {"run": entry} and not entry.ended and entry.close_task is None
            assert not service._close_tasks and closed == []
            assert "run" not in service._run_hooked  # the hook ran, once
            asyncio.run_coroutine_threadsafe(service.close(), owner_loop).result(10)
            assert closed == [True] and not service._run_connections
        finally:
            stop()

    @pytest.mark.parametrize(
        "left",
        [
            "nothing",
            "a cut connection",
            "a close still going",
            "a close that failed, not yet recorded",
            "a close that was cut, not yet recorded",
        ],
    )
    def test_a_runs_end_on_another_loop_refuses_only_while_the_run_has_something_to_close(self, left):
        """The run's end on another loop than the run's: with nothing of the run's left (a close() on its loop closed it
        all), it returns, so the run logs no failure (review on #334); with a connection whose close was cut, or a
        close still going there, it raises as when the connection is pooled. So it does for a close that ended cut
        or failed while its loop paused before ``_close_done`` recorded it as cut: the session is still open (it
        returned quietly; review on #334)."""
        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()
        service = MCPServer(transport="http", service="api")
        try:
            if left == "a cut connection":
                cut = _RunConnection(object(), "fingerprint", owner_loop)
                service._unclosed[id(cut)] = ("run", cut)
            elif left == "a close still going":
                service._run_closes["run"] = {owner_loop.create_future()}
            elif left.startswith("a close that"):
                ended = owner_loop.create_future()
                if "failed" in left:
                    owner_loop.call_soon_threadsafe(ended.set_exception, RuntimeError("the server's close() raised"))
                else:
                    owner_loop.call_soon_threadsafe(ended.cancel)
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0), owner_loop).result(10)
                assert ended.done()
                service._run_closes["run"] = {ended}  # its done callback, _close_done, has not run
            elif left == "nothing":
                done = owner_loop.create_future()
                owner_loop.call_soon_threadsafe(done.set_result, None)
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0), owner_loop).result(10)
                service._run_closes["run"] = {done}  # one that finished, its record not dropped yet
            service._run_hooked["run"] = owner_loop

            async def end_the_run():
                return service._run_ended("run", owner_loop)

            if left == "nothing":
                assert asyncio.run(asyncio.wait_for(end_the_run(), 10)) is None
            else:
                with pytest.raises(RuntimeError, match=self.OWNS):
                    asyncio.run(asyncio.wait_for(end_the_run(), 10))
            assert "run" not in service._run_hooked
        finally:
            stop()

    @pytest.mark.parametrize("then", ["its idle retry", "close()"])
    @pytest.mark.asyncio
    async def test_a_connection_being_closed_is_seen_by_the_runs_end_at_every_step(self, then):
        """A run's connection on its way out, on its loop: retired from the pool, its close failing (it is kept as cut),
        then closed again by its idle retry or by close(). A run's end on another thread may read at any step: at
        each one (every write to the pool, the cut list and the close records) the connection is still found, until
        a close has succeeded. Each was taken out of one place before its next was recorded, so a read in between
        returned quietly with the session open (review on #334)."""
        from timbal.core.mcp import _RunConnection

        service = MCPServer(transport="http", service="api")
        closes, closed, seen = [], [], []

        def look(step):
            seen.append((step, bool(closed) or service._run_holds("run")))

        class Watched(dict):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                look(f"{self.name}[{key!r}] =")

            def pop(self, key, *default):
                value = super().pop(key, *default)
                look(f"{self.name}.pop")
                return value

        class WatchedSet(set):
            def add(self, item):
                super().add(item)
                look("close recorded")

            def discard(self, item):
                super().discard(item)
                look("close discarded")

        class Closes(Watched):
            def setdefault(self, key, _default=None):  # each run's close records watched too
                return super().setdefault(key, WatchedSet())

        class Server:
            async def close(self):
                closes.append(True)
                if len(closes) == 1:
                    raise RuntimeError("the server's close() raised")
                closed.append(True)  # what follows may say the run holds nothing: it closed

        private = service.__pydantic_private__
        private["_run_connections"], private["_unclosed"], private["_run_closes"] = (
            Watched("pool"),
            Watched("cut"),
            Closes("closes"),
        )
        entry = _RunConnection(Server(), "fingerprint", asyncio.get_running_loop())
        service._run_connections["run"] = entry
        first = service._retire_run_connection("run", entry)
        await asyncio.wait({first})
        await asyncio.sleep(0)  # its done callbacks
        assert service._unclosed  # kept as cut
        if then == "its idle retry":
            entry.cancel_idle()
            service._retry_unclosed("run", entry)
            second = entry.close_task
            await asyncio.wait({second})
            await asyncio.sleep(0)
        else:
            await asyncio.wait_for(service.close(), 10)
        assert closes == [True, True] and closed == [True]
        assert not service._run_connections and not service._unclosed and not service._run_closes
        assert [step for step, holds in seen if not holds] == [], seen

    def test_a_run_ended_elsewhere_as_its_first_call_registers_the_hook_is_refused(self, monkeypatch):
        """Another loop closes the run's stream right after the run's first call registered the end hook, before that
        call went on. The hook knows the run's loop from its registration: it is refused there, the connection is
        closed on its own loop once idle, never on the loop that ended the run, and no record of the hook is left
        behind. With the loop looked up after the registration, the end found none, took the run's connection on the
        wrong loop, and its teardown failed there (review on #334)."""
        import threading

        from timbal.core import mcp as mcp_module
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        closed_on = []

        class Server:
            async def close(self):
                closed_on.append(asyncio.get_running_loop())

        monkeypatch.setattr("timbal.platform.utils._resolve_url_and_headers", lambda *_: ("https://x.test/mcp", {}))
        monkeypatch.setattr(MCPServer, "_new_project_server", lambda _self, _url, _headers: Server())
        monkeypatch.setattr(mcp_module, "_run_connection_key", lambda _context: "run")
        monkeypatch.setattr(mcp_module, "_RUN_CONNECTION_IDLE_S", 0.2)
        on_end, ended = RunContext.on_end, []

        def end_elsewhere_once_accepted(self, hook):
            accepted = on_end(self, hook)
            if accepted and not ended:  # another loop closes the run's stream, now
                ended.append(True)
                thread = threading.Thread(target=lambda: asyncio.run(self._run_end_hooks()))
                thread.start()
                thread.join(10)
            return accepted

        monkeypatch.setattr(RunContext, "on_end", end_elsewhere_once_accepted)
        service = MCPServer(transport="http", service="api")

        async def first_call():
            set_run_context(RunContext())
            try:
                async with service._project_connection():
                    pass
                await until(lambda: closed_on, 10)  # closed once idle
                return asyncio.get_running_loop()
            finally:
                set_run_context(None)

        run_loop = asyncio.run(first_call())
        assert ended and closed_on == [run_loop]
        assert not service._run_hooked and not service._run_connections

    def test_a_runs_cut_connection_is_closed_on_its_loop_when_its_end_is_refused(self, monkeypatch):
        """A run's connection whose close failed is kept, out of the pool, for another try. The run's end on another
        loop is refused and does nothing, so that try is the connection's own: on its loop, once idle, as a pooled
        connection's idle close. Before, nothing but a close() on that loop closed it (Bugbot on #334)."""
        from timbal.core import mcp as mcp_module
        from timbal.core.mcp import _RunConnection

        monkeypatch.setattr(mcp_module, "_RUN_CONNECTION_IDLE_S", 0.2)
        owner_loop, stop = self.owner_thread()
        closes = []

        class FailsOnce:
            async def close(self):
                closes.append(True)
                if len(closes) == 1:
                    raise RuntimeError("its session is still shutting down")

        service = MCPServer(transport="http", service="api")
        try:

            async def fail_its_close():  # a credential change there, whose close failed
                service._run_connections["run"] = _RunConnection(FailsOnce(), "fingerprint", owner_loop)
                await asyncio.wait({service._retire_run_connection("run")})
                await until(lambda: service._unclosed)  # kept, out of the pool

            asyncio.run_coroutine_threadsafe(fail_its_close(), owner_loop).result(10)
            service._run_hooked["run"] = owner_loop

            async def end_the_run():
                return service._run_ended("run", owner_loop)

            with pytest.raises(RuntimeError, match=self.OWNS):
                asyncio.run(asyncio.wait_for(end_the_run(), 10))
            # No close() anywhere: closed on its own loop, once idle.
            asyncio.run(until(lambda: len(closes) == 2 and not service._unclosed and not service._close_tasks, 10))
        finally:
            stop()

    @staticmethod
    def race_at(service, attribute, mapping, other, other_first=False):
        """Install ``mapping`` as ``service``'s ``attribute`` and run ``other`` on a thread of its own at this thread's
        first look into it (after the look, or before it with ``other_first``), going on once ``other`` has finished:
        where another thread's change would land. Returns a list that holds that thread once it started."""
        import threading

        armed, started = [True], []

        def race():
            if armed and threading.current_thread() is threading.main_thread():
                armed.clear()  # this thread is looking; the other one goes now
                started.append(threading.Thread(target=other))
                started[0].start()
                started[0].join(10)

        class Watched(dict):
            def get(self, key, default=None):
                if other_first:
                    race()
                value = super().get(key, default)
                race()
                return value

            def __getitem__(self, key):
                if other_first:
                    race()
                value = super().__getitem__(key)
                race()
                return value

        service.__pydantic_private__[attribute] = Watched(mapping)
        return started

    def test_two_closes_on_two_threads_take_a_closed_loops_connection_once(self):
        """close() on two threads both find a connection whose loop has closed: one takes it out of the pool and the
        other finds it gone, instead of passing the check too and failing on the delete (``KeyError``, which ended
        its cleanup)."""
        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                pass

        gone = asyncio.new_event_loop()
        gone.close()
        service = MCPServer(transport="http", service="api")
        results = []

        def close_here():
            try:
                asyncio.run(asyncio.wait_for(service.close(), 10))
                results.append(None)
            except BaseException as error:
                results.append(error)

        pool = {"run": _RunConnection(Server(), "fingerprint", gone)}
        other = self.race_at(service, "_run_connections", pool, close_here)
        close_here()
        other[0].join(10)
        assert results == [None, None] and not service._run_connections

    def test_two_closes_on_two_threads_forget_a_closed_loops_close_record_once(self):
        """A close finished on a loop that then closed before its callback ran: its record is left, with no pooled
        connection. close() on two threads both took that close to forget it, and both found its record; whichever
        looks it up second finds it gone instead of failing on the lookup (``KeyError``, which ended its cleanup).
        The other close() is run from that point: its ``_forget_close``, at this one's lookup."""
        gone = asyncio.new_event_loop()
        record = gone.create_future()  # the close, its loop closed under it
        gone.close()
        service = MCPServer(transport="http", service="api")
        service._close_tasks.add(record)
        errors = []

        def the_other_close():
            try:
                service._forget_close(record)
            except BaseException as error:
                errors.append(error)

        other = self.race_at(service, "_run_closes", {"run": {record}}, the_other_close, other_first=True)
        asyncio.run(asyncio.wait_for(service.close(), 10))
        other[0].join(10)
        assert not errors and not service._run_closes and not service._close_tasks

    def test_a_run_dropped_while_close_takes_its_connection_is_taken_once(self):
        """The run's loop closed: its end hook drops the run (``_drop_run``) on one thread while close() on another
        takes the run's connection out of the pool: whichever comes second finds it gone instead of failing on the
        delete (``KeyError``, which ended close()'s cleanup)."""
        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                pass

        gone = asyncio.new_event_loop()
        gone.close()
        service = MCPServer(transport="http", service="api")
        errors = []

        def drop():
            try:
                service._drop_run("run")
            except BaseException as error:
                errors.append(error)

        pool = {"run": _RunConnection(Server(), "fingerprint", gone)}
        other = self.race_at(service, "_run_connections", pool, drop)
        asyncio.run(asyncio.wait_for(service.close(), 10))
        other[0].join(10)
        assert not errors and not service._run_connections


class TestInflightCallsPerSession:
    """The records of the ``tools/call``s in flight (``_inflight``), which the elicitation and sampling callbacks read
    to find the call that asked. A record lives as long as its call (a ``close()`` keeps it); a server request on a
    session counts only the calls whose request is outstanding on that session (``_InflightCall.session``)."""

    QUESTION = {"message": "Which one?", "requestedSchema": {"type": "object", "properties": {}}}

    @staticmethod
    def on(session):
        """The request context the SDK hands a callback: the session the server's request arrived on."""
        from types import SimpleNamespace

        return SimpleNamespace(session=session)

    @staticmethod
    def sent_on(call, session):
        """The call's request sent on ``session``, as ``_request`` notes it."""
        call.session = session
        return call

    @staticmethod
    def asking_session(server, on):
        """A session whose ``tools/call`` of ``ask`` makes the server ask the question first (it answers with what the
        client said); any other tool just answers."""

        class Session:
            async def call_tool(self, name, **_request):
                if name != "ask":
                    return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="done")])
                params = mcp_types.ElicitRequestFormParams(**TestInflightCallsPerSession.QUESTION)
                answer = await on(self)(server._on_elicitation, params)
                return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=answer.action)])

        return Session

    @staticmethod
    def assert_asked(result):
        """The question reached the call: its run stopped for the answer, instead of the tool completing on a
        ``cancel`` it never showed."""
        assert result.status.reason == "input_required", (result.status, result.output)
        assert result.output["kind"] == "mcp_elicitation" and result.output["payload"]["message"] == "Which one?"

    @pytest.mark.asyncio
    async def test_an_elicitation_counts_only_the_calls_with_a_request_on_its_session(self):
        """A call still waiting to send (in ``auth.prepare()``, or to connect) and one sent on another session are not
        counted: the question goes to the one call whose request is on its session (outside a run it is declined),
        not answered ``cancel`` for three calls in flight. Two calls sent on that session are ambiguous, as before."""
        server = MCPServer(transport="stdio", command="unused")
        session, other = object(), object()
        asking = self.sent_on(server._begin_call("ask"), session)
        waiting = server._begin_call("waiting")
        elsewhere = self.sent_on(server._begin_call("elsewhere"), other)
        result = await server._on_elicitation(self.on(session), mcp_types.ElicitRequestFormParams(**self.QUESTION))
        assert result.action == "decline"
        assert asking.ambiguous_with == waiting.ambiguous_with == elsewhere.ambiguous_with == 0
        also = self.sent_on(server._begin_call("also"), session)
        result = await server._on_elicitation(self.on(session), mcp_types.ElicitRequestFormParams(**self.QUESTION))
        assert result.action == "cancel" and asking.ambiguous_with == also.ambiguous_with == 2
        assert waiting.ambiguous_with == elsewhere.ambiguous_with == 0

    @pytest.mark.asyncio
    async def test_a_sampling_request_counts_only_the_calls_with_a_request_on_its_session(self, monkeypatch):
        """With another call still waiting to send, sampling runs in the context of the one call whose request is on
        its session (its span under the tool), not outside the run."""
        import contextvars

        from timbal.core.test_model import TestModel

        whose = contextvars.ContextVar("whose", default="nobody's")

        async def sample(_self, _params):
            return whose.get()

        monkeypatch.setattr(MCPServer, "_sample", sample)
        server = MCPServer(transport="stdio", command="unused", sampling_model=TestModel(responses=["unused"]))
        session = object()
        server._begin_call("waiting")
        token = whose.set("the call's")
        self.sent_on(server._begin_call("ask"), session)  # its context is taken here
        whose.reset(token)
        params = mcp_types.CreateMessageRequestParams(messages=[], maxTokens=16)
        result = await server._on_sampling(self.on(session), params)
        assert result.content.text == "the call's"

    @pytest.mark.asyncio
    async def test_a_call_waiting_in_prepare_across_close_leaves_another_calls_question_to_it(self, monkeypatch):
        """One loop: a call waits in ``auth.prepare()`` while close() runs, and outlives it (its record kept); another
        call then opens a new session, and the server asks it a question. The question reaches that call, the first
        one having sent nothing (counted, the two were ambiguous and the question answered ``cancel``; review on
        #334). The first call then completes on the new session."""
        import httpx

        held, release = asyncio.Event(), asyncio.Event()

        class Auth(httpx.Auth):
            async def prepare(self):
                if not held.is_set():
                    held.set()
                    await release.wait()  # the first call only: a token fetch that takes a while

        def on(session):
            return lambda callback, params: callback(self.on(session), params)

        async def run_session(_self, ready, closed):
            ready.set_result(session_class())
            await closed.wait()

        server = MCPServer(name="srv", transport="http", url="http://unused.test/mcp", auth=Auth())
        session_class = self.asking_session(server, on)
        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        await server._connect()
        other = server._make_tool(mcp_types.Tool(name="other", inputSchema={"type": "object"}))
        ask = server._make_tool(mcp_types.Tool(name="ask", inputSchema={"type": "object"}))
        try:
            first = asyncio.create_task(other().collect())
            await asyncio.wait_for(held.wait(), 10)
            await asyncio.wait_for(server.close(), 10)
            assert len(server._inflight) == 1  # the first call's record, kept
            self.assert_asked(await asyncio.wait_for(ask().collect(), 10))
            release.set()
            assert (await asyncio.wait_for(first, 10)).output == "done"
        finally:
            release.set()
            await server.close()

    def test_a_call_on_another_loop_using_this_session_is_asked(self, monkeypatch):
        """The tools were resolved, and the session opened, on a loop running in another thread; the tool is called on
        this loop and its request goes out on that session, whose callbacks run on that loop. The question reaches
        the call (it was answered ``cancel`` when only the calls of the callback's loop counted; review on #334)."""
        import threading

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()

        def on(session):  # the SDK runs the callback on the session's receive loop: that thread's
            return lambda callback, params: asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(callback(self.on(session), params), loop)
            )

        async def run_session(_self, ready, closed):
            ready.set_result(session_class())
            await closed.wait()

        server = MCPServer(name="srv", transport="stdio", command="unused")
        session_class = self.asking_session(server, on)
        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        try:
            asyncio.run_coroutine_threadsafe(server._connect(), loop).result(10)
            ask = server._make_tool(mcp_types.Tool(name="ask", inputSchema={"type": "object"}))
            self.assert_asked(asyncio.run(asyncio.wait_for(ask().collect(), 10)))
        finally:
            asyncio.run_coroutine_threadsafe(server.close(), loop).result(10)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

    def test_a_stateless_servers_question_fails_a_call_on_another_loop_on_that_loop(self):
        """A stateless HTTP server (no ``Mcp-Session-Id``) asks a question during a call made on another loop than the
        session's: the client cannot deliver an answer, so it fails the call fast through its ``abort`` future. That
        future is the call's loop's and is not thread-safe: it is completed there (``call_soon_threadsafe``), not from
        the session's loop, where asyncio's debug check refused it and a loop not in debug could miss the wake-up and
        leave the call waiting for its timeout (Bugbot on #334)."""
        import threading

        loop = asyncio.new_event_loop()
        loop.set_debug(True)  # asyncio then checks which thread schedules on it
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        server = MCPServer(transport="http", url="http://unused.test/mcp")
        server._get_session_id = lambda: None  # connected, and the server gave no session id
        session = object()

        async def begin():  # the call sends on the session and waits, as _request races its abort future
            call = self.sent_on(server._begin_call("ask"), session)
            waiting = asyncio.ensure_future(asyncio.wait({call.abort}))
            await asyncio.sleep(0)
            return call, waiting

        async def outcome(waiting):
            await asyncio.wait_for(waiting, 10)
            return waiting

        try:
            call, waiting = asyncio.run_coroutine_threadsafe(begin(), loop).result(10)
            params = mcp_types.ElicitRequestFormParams(**self.QUESTION)
            answer = asyncio.run(asyncio.wait_for(server._on_elicitation(self.on(session), params), 10))
            assert answer.code == mcp_types.INVALID_REQUEST
            asyncio.run_coroutine_threadsafe(outcome(waiting), loop).result(10)
            assert "stateless" in str(call.abort.exception())
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

    def test_a_call_another_thread_began_is_asked_after_close_from_another_loop(self):
        """A call begun on a loop running in another thread, not connected yet: close() from this loop has nothing open
        to refuse, and keeps that call's record. When the call has sent its request and the server's question
        arrives on that session, it reaches the call (its payload recorded for the suspend), instead of finding no
        call and being answered ``cancel`` (Bugbot on #334)."""
        import threading
        from types import SimpleNamespace

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        server = MCPServer(transport="stdio", command="unused")
        session = object()

        async def begin():
            call = server._begin_call("ask")
            call.run_context = SimpleNamespace(_resume_values={}, _used_resume_ids=set())
            call.path = "agent.ask"
            return call

        async def ask():
            return await server._on_elicitation(self.on(session), mcp_types.ElicitRequestFormParams(**self.QUESTION))

        try:
            call = asyncio.run_coroutine_threadsafe(begin(), loop).result(10)
            asyncio.run(asyncio.wait_for(server.close(), 10))
            assert list(server._inflight.values()) == [call]
            self.sent_on(call, session)
            asyncio.run_coroutine_threadsafe(ask(), loop).result(10)
            assert call.pending_payload is not None and call.pending_payload["message"] == "Which one?"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

    @pytest.mark.asyncio
    async def test_a_call_waiting_to_connect_while_close_shuts_down_is_still_asked(self, monkeypatch):
        """One loop: close() is shutting the session down (it holds the session's lock) when a tool call begins and
        waits for that lock; it then reconnects, and the server asks a question during it. The question reaches the
        call, instead of being answered ``cancel`` for no call in flight while the tool returns as if nothing was
        asked: close() dropped the record of a call that went on (review on #334)."""
        shutting_down, finish, sessions = asyncio.Event(), asyncio.Event(), []

        def on(session):
            return lambda callback, params: callback(self.on(session), params)

        async def run_session(_self, ready, closed):
            sessions.append(True)
            ready.set_result(session_class())
            await closed.wait()
            if len(sessions) == 1:
                shutting_down.set()
                await finish.wait()  # a slow teardown, the lock held meanwhile

        server = MCPServer(name="srv", transport="stdio", command="unused")
        session_class = self.asking_session(server, on)
        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        await server._connect()
        tool = server._make_tool(mcp_types.Tool(name="ask", inputSchema={"type": "object"}))
        try:
            closing = asyncio.create_task(server.close())
            await asyncio.wait_for(shutting_down.wait(), 10)
            calling = asyncio.create_task(tool().collect())
            await until(lambda: len(server._inflight) == 1)  # begun, waiting for the lock
            finish.set()
            await asyncio.wait_for(closing, 10)
            result = await asyncio.wait_for(calling, 10)
            assert len(sessions) == 2  # it reconnected
            self.assert_asked(result)
        finally:
            finish.set()
            await server.close()

    @pytest.mark.asyncio
    async def test_calls_begun_at_once_get_keys_of_their_own(self):
        """Another thread begins a call just as this one takes its key: each gets its own, wherever in ``_begin_call``
        the other call starts (an instance counter's ``+= 1``, read and then written, handed both the same key, so one
        record replaced the other and the first call's end removed the second's). Tried at every attribute of the
        server that ``_begin_call`` reads."""
        state = {"at": 0, "reads": -1, "nested": []}  # reads < 0: not counting

        class Raced(MCPServer):
            def __getattr__(self, name):
                value = super().__getattr__(name)
                if state["reads"] >= 0 and name.startswith("_"):
                    state["reads"] += 1
                    if state["reads"] == state["at"]:  # another thread begins a call here
                        state["reads"] = -1
                        state["nested"].append(self._begin_call("theirs"))
                return value

        tried = 0
        while True:
            server = Raced(transport="stdio", command="unused")
            state.update(at=tried + 1, reads=0, nested=[])  # count only what _begin_call reads
            mine = server._begin_call("mine")
            state["reads"] = -1
            if not state["nested"]:
                break  # _begin_call reads fewer attributes than that: every point was tried
            theirs = state["nested"][0]
            assert mine.key != theirs.key, f"at read {state['at']}"
            assert server._inflight == {mine.key: mine, theirs.key: theirs}
            server._end_call(mine)
            assert server._inflight == {theirs.key: theirs}
            tried += 1
        assert tried >= 1


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
            assert not get_run_context().__dict__.get("_end_hooks")  # nothing left waiting for an end that never comes
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
        assert tool.interruption_mode == "allow"

    def test_server_interruption_mode_is_copied_onto_tools(self):
        server = MCPServer(
            name="alpha",
            transport="http",
            url="https://example.com/mcp",
            interruption_mode="disable_during_tool",
        )
        tool = server._make_tool(
            mcp_types.Tool(
                name="greet",
                description="Greet someone.",
                inputSchema={"type": "object", "properties": {}},
            )
        )
        assert tool.interruption_mode == "disable_during_tool"

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
