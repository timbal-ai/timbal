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
    async def test_a_loop_that_closes_during_the_handoff_takes_its_cut_connections_with_it(self):
        """A cut connection of another loop is left in _unclosed for that loop; the loop closes as close() hands it
        over: the row goes too, credential headers included, instead of staying until a later close()."""
        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                raise AssertionError("closed on the wrong loop")

        service = MCPServer(transport="http", service="api")
        entry = _RunConnection(Server(), "fingerprint", self.closing_loop())
        service._unclosed[id(entry)] = ("run", entry)
        await asyncio.wait_for(service.close(), 10)
        assert not service._unclosed

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

    def test_a_handoff_the_owner_cancels_is_reported_and_the_connection_kept(self):
        """The owner cancelled the handed work before it started (a loop cancelling its tasks as it shuts down): the
        future is done, but nothing was closed. close() says the connection is still open; the end hook keeps it."""
        import threading

        from timbal.core.mcp import _RunConnection

        owner = asyncio.new_event_loop()

        def cancelling(loop, coro, **kwargs):
            task = asyncio.Task(coro, loop=loop, **kwargs)
            task.cancel()
            return task

        owner.set_task_factory(cancelling)
        thread = threading.Thread(target=owner.run_forever)
        thread.start()

        class Session:
            async def close(self):
                raise AssertionError("never started")

        service = MCPServer(transport="http", service="api")
        try:
            service._run_connections["run"] = _RunConnection(Session(), "fingerprint", owner)
            with pytest.raises(RuntimeError, match="still open"):
                asyncio.run(asyncio.wait_for(service.close(), 10))
            assert "run" in service._run_connections  # the cleanup is still owed
            service._run_hooked["run"] = owner

            async def end_the_run():  # the run's end hook, on another loop than the run's
                waiting = service._run_ended("run")
                if waiting is not None:
                    await waiting

            asyncio.run(asyncio.wait_for(end_the_run(), 10))
            assert service._run_connections["run"].ended  # kept, for a close() later
        finally:
            owner.call_soon_threadsafe(owner.stop)
            thread.join(10)
            owner.close()

    @pytest.mark.asyncio
    async def test_a_cut_teardown_does_not_cut_the_session_close(self):
        """The owner's cleanup is cancelled after it let go of the connection and started its close: the close goes
        on, and a later close() waits for it instead of returning with the session still open."""
        from timbal.core.mcp import _RunConnection

        release, closed = asyncio.Event(), []

        class SlowClose:
            async def close(self):
                await release.wait()
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        entry = _RunConnection(SlowClose(), "fingerprint", asyncio.get_running_loop())
        service._run_connections["run"] = entry
        teardown = asyncio.ensure_future(service._close_run_connections([("run", entry)]))
        await until(lambda: service._close_tasks)  # let go of, its close started
        teardown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await teardown
        assert service._close_tasks and not closed  # still closing
        release.set()
        await asyncio.wait_for(service.close(), 10)
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
    async def test_the_run_end_and_a_close_elsewhere_never_both_take_a_cut_close(self, monkeypatch):
        """close() on another loop takes the cut connections while the run's end is going through them: the end
        skips the ones taken, instead of failing on them, and takes none twice."""
        from timbal.core.mcp import _RunConnection

        class Server:
            async def close(self):
                pass

        service = MCPServer(transport="http", service="api")
        loop = asyncio.get_running_loop()
        for _ in range(2):
            await self.cut(service, "run", _RunConnection(Server(), "fingerprint", loop))
        retire, taken = MCPServer._retire_run_connection, []

        def retire_while_close_takes_the_rest(self, key, entry=None):
            if not taken:
                self._unclosed.clear()  # what close() on the other loop does meanwhile
            taken.append(entry)
            return retire(self, key, entry)

        monkeypatch.setattr(MCPServer, "_retire_run_connection", retire_while_close_takes_the_rest)
        waiting = service._end_run_here("run")
        if waiting is not None:
            await asyncio.wait_for(waiting, 10)
        assert len(taken) == 1

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
        assert service._run_ended("run") is None
        assert not service._unclosed
        owner = self.closing_loop()  # closes during the handoff instead
        await self.cut(service, "other", _RunConnection(Server(), "fingerprint", owner))
        service._run_hooked["other"] = owner
        await asyncio.wait_for(service._run_ended("other"), 10)
        assert not service._unclosed

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

    @pytest.mark.parametrize("reader", ["_wait_loop_closes", "_finish_handoff", "_forget_close"])
    @pytest.mark.asyncio
    async def test_closes_are_read_from_a_copy_while_another_loop_changes_them(self, reader):
        """Another loop's thread starts or drops closes while close() goes over them: each reader goes over a copy.
        Going over the set or dict itself raises "changed size during iteration". The change is made from inside
        the read here (a close's ``get_loop()``, a hash), where another thread's would land."""
        from timbal.core.mcp import _Left

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
        if reader == "_wait_loop_closes":
            assert await service._wait_loop_closes() == _Left()
        else:  # that loop closed: its closes are forgotten, which goes over them
            assert await service._finish_handoff(None, elsewhere, []) == _Left()


class TestStandaloneServerClose:
    def test_a_retry_from_another_loop_waits_for_the_owner_a_cut_close_signalled(self):
        """A stdio server's first close() ran on its owner's loop, signalled the owner and was cut waiting for it; a
        retry from another, running loop waits for that owner on its own loop instead of returning at once."""
        import threading

        owner_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=owner_loop.run_forever)
        thread.start()
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

            async def retry():
                await until(closed.is_set)  # the first close() has signalled the owner
                first.cancel()
                attempt = asyncio.ensure_future(server.close())
                done, _ = await asyncio.wait({attempt}, timeout=0.3)
                assert not done  # waiting for that owner, on its loop
                owner_loop.call_soon_threadsafe(release.set)
                await asyncio.wait_for(attempt, 10)
                assert owner.done()

            asyncio.run(retry())
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            thread.join(10)
            owner_loop.close()

    @pytest.mark.parametrize("how", ["closed", "closes during the handoff"])
    @pytest.mark.asyncio
    async def test_a_session_whose_loop_closed_still_clears_the_caches(self, how):
        """The session's loop closed with it, nothing having cancelled its tasks: close() forgets the session and
        clears the tool list and the calls in flight, as every close() does. A call left there would make every
        later elicitation look ambiguous (answered ``cancel``)."""
        loop = TestPlatformConnectionLifetime.closing_loop() if how != "closed" else asyncio.new_event_loop()
        if how == "closed":
            loop.close()

        class Owner:
            """The session's owner task, on that loop."""

            def get_loop(self):
                return loop

            def done(self):
                return False

        server = MCPServer(transport="stdio", command="unused")
        server._session_task = Owner()
        server._owner_task = server._session_task  # as _open_session leaves them
        server._tools_cache, server._tools_cache_run_id, server._tools_stale = [object()], "run", True
        server._inflight[1] = object()  # a call that died with its loop
        await asyncio.wait_for(server.close(), 10)
        assert server._session_task is None
        assert server._tools_cache is None and server._tools_cache_run_id is None and not server._tools_stale
        assert not server._inflight


class TestCrossLoopClose:
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
                loop.set_task_factory(None)  # a test's own factory (one that cancels tasks) must not cancel this
                asyncio.run_coroutine_threadsafe(cancel_the_rest(), loop).result(10)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

        return loop, stop

    def test_a_retry_from_another_loop_closes_the_session_on_its_loop_behind_its_locks(self):
        """The session's loop holds the server's lock (work in flight there) when a close() from another loop comes:
        the session's close runs on its own loop, waiting there for the lock, instead of failing on a lock bound
        to that loop."""
        owner_loop, stop = self.owner_thread()
        server = MCPServer(transport="stdio", command="unused")
        try:

            async def install():
                closed, free, held = asyncio.Event(), asyncio.Event(), asyncio.Event()

                async def owner():
                    await closed.wait()

                async def hold():
                    async with server._get_lock():
                        held.set()
                        await free.wait()

                task = asyncio.ensure_future(owner())
                server._session_task, server._session_closed = task, closed
                server._owner_task = server._session_task  # as _open_session leaves them
                asyncio.ensure_future(hold())
                await held.wait()
                waiter = asyncio.ensure_future(server._get_lock().acquire())  # contended there: bound to that loop
                await asyncio.sleep(0)
                waiter.cancel()
                return task, free

            owner, free = asyncio.run_coroutine_threadsafe(install(), owner_loop).result(10)

            async def retry():
                attempt = asyncio.ensure_future(server.close())
                done, _ = await asyncio.wait({attempt}, timeout=0.3)
                assert not done  # on the session's loop, waiting for its lock
                owner_loop.call_soon_threadsafe(free.set)
                await asyncio.wait_for(attempt, 10)
                assert owner.done()

            asyncio.run(retry())
        finally:
            stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ended", ["with a cut", "failed"])
    async def test_how_anothers_handoff_ended_is_this_close_too(self, ended):
        """All a close() finds is another close()'s handoff, still going: it waits for it, and when that one leaves
        a session open (its close cut there, or the handoff failed) it says so, as that other close() does."""
        import concurrent.futures

        from timbal.core.mcp import _Left, _RunConnection

        class Server:
            async def close(self):
                pass

        loop = asyncio.get_running_loop()
        handed = concurrent.futures.Future()
        service = MCPServer(transport="http", service="api")
        service._handoffs[id(handed)] = (handed, loop)
        entry = _RunConnection(Server(), "fingerprint", loop)

        def finish():  # what the other close()'s handoff leaves: the connection kept for a later close()
            service._unclosed[id(entry)] = ("run", entry)
            if ended == "with a cut":
                handed.set_result(_Left(cut=1))
            else:
                handed.set_exception(RuntimeError("the owner loop cancelled it"))

        loop.call_later(0.05, finish)
        with pytest.raises(RuntimeError, match="1 platform connection"):
            await asyncio.wait_for(service.close(), 10)

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

    def test_a_retried_connection_whose_handoff_fails_is_counted_once(self):
        """A cut connection retried by close(), its handoff cancelled by its loop: one connection still open."""
        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()

        def cancelling(loop, coro, **kwargs):
            task = asyncio.Task(coro, loop=loop, **kwargs)
            task.cancel()
            return task

        class Server:
            async def close(self):
                pass

        service = MCPServer(transport="http", service="api")
        try:
            owner_loop.set_task_factory(cancelling)
            entry = _RunConnection(Server(), "fingerprint", owner_loop)
            service._unclosed[id(entry)] = ("run", entry)
            with pytest.raises(RuntimeError, match=r"MCP server 'http': 1 platform connection\(s\) are still open"):
                asyncio.run(asyncio.wait_for(service.close(), 10))
        finally:
            stop()

    def test_a_signalled_owner_on_a_paused_loop_is_counted_once(self):
        """A standalone server whose earlier close() forgot the session and was cut: its owner, on a paused loop, is
        one session still shutting down, whichever step finds it."""
        server = MCPServer(transport="stdio", command="unused")
        with asyncio.Runner() as runner:

            async def start():
                return asyncio.ensure_future(asyncio.Event().wait())

            owner = runner.run(start())
            server._owners_closing.add(owner)  # its refs already forgotten
            with pytest.raises(RuntimeError) as failure:
                asyncio.run(asyncio.wait_for(server.close(), 10))
            assert "1 session(s) of this server are still shutting down" in str(failure.value)

    def test_a_cancelled_handoff_does_not_end_close_before_the_closes_it_started(self):
        """The owner cancelled close()'s handoff after it had let go of the connection and started its close: that
        close goes on there, and close() waits for it instead of returning while the session is open."""
        import threading

        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()
        started, closed = threading.Event(), []

        async def make_event():
            return asyncio.Event()

        release = asyncio.run_coroutine_threadsafe(make_event(), owner_loop).result(10)

        class Slow:
            async def close(self):
                started.set()
                await release.wait()
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        try:
            service._run_connections["run"] = _RunConnection(Slow(), "fingerprint", owner_loop)

            async def main():
                closing = asyncio.ensure_future(service.close())
                assert await asyncio.to_thread(started.wait, 10)  # let go of there, its close going
                handed, _ = next(iter(service._handoffs.values()))
                handed.cancel()  # the owner cancels the handed work
                done, _ = await asyncio.wait({closing}, timeout=0.3)
                assert not done and not closed  # waiting for the close it started
                owner_loop.call_soon_threadsafe(release.set)
                await asyncio.wait_for(closing, 10)
                assert closed == [True]

            asyncio.run(main())
        finally:
            stop()

    def test_a_close_from_another_loop_during_the_first_handshake_closes_the_session_there(self, monkeypatch):
        """The first session is still in its handshake on its loop (no owner task recorded yet, its lock held) when a
        close() comes from another loop: the close runs on the session's loop, after the handshake, and closes it."""
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

            async def close_elsewhere():
                attempt = asyncio.ensure_future(server.close())
                done, _ = await asyncio.wait({attempt}, timeout=0.3)
                assert not done  # on the session's loop, behind its handshake
                owner_loop.call_soon_threadsafe(gates["release"].set)
                await asyncio.wait_for(attempt, 10)

            asyncio.run(close_elsewhere())
            connecting.result(10)
            assert server._session is None and server._session_task is None  # closed, not left open
        finally:
            stop()

    def test_a_cut_wait_for_a_signalled_owner_is_reported(self):
        """The wait for an owner an earlier close() signalled is cancelled on the owner's loop while the owner still
        runs: close() says that session is still shutting down instead of returning."""
        owner_loop, stop = self.owner_thread()

        def cancel_the_waits(loop, coro, **kwargs):
            task = asyncio.Task(coro, loop=loop, **kwargs)
            if getattr(coro, "__qualname__", "").endswith("_wait_for_task"):
                task.cancel()
            return task

        async def start():
            return asyncio.ensure_future(asyncio.Event().wait())

        server = MCPServer(transport="stdio", command="unused")
        owner = asyncio.run_coroutine_threadsafe(start(), owner_loop).result(10)
        try:
            owner_loop.set_task_factory(cancel_the_waits)
            server._owners_closing.add(owner)  # signalled by an earlier close(), its refs forgotten
            with pytest.raises(RuntimeError, match="still open or shutting down"):
                asyncio.run(asyncio.wait_for(server.close(), 10))
        finally:
            stop()  # cancels the owner and awaits it there before stopping its loop

    def test_the_runs_end_finds_a_cut_connection_a_close_elsewhere_is_about_to_retry(self):
        """close() on another loop is about to retry a cut connection on its loop when the run ends there first: the
        run's end still finds it and waits for its close, instead of returning with the transport open."""
        import threading

        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()
        closed = []

        class Server:
            async def close(self):
                closed.append(True)

        service = MCPServer(transport="http", service="api")
        entry = _RunConnection(Server(), "fingerprint", owner_loop)
        service._unclosed[id(entry)] = ("run", entry)
        service._run_hooked["run"] = owner_loop

        async def end_the_run():
            waiting = service._run_ended("run")
            if waiting is None:
                return False  # nothing to wait for
            await waiting
            return bool(closed)

        try:
            gate = threading.Event()
            owner_loop.call_soon_threadsafe(gate.wait, 10)  # its loop busy: what comes next waits in line
            ended = asyncio.run_coroutine_threadsafe(end_the_run(), owner_loop)  # the run's end, first in line

            async def close_elsewhere():
                closing = asyncio.ensure_future(service.close())
                await until(lambda: service._handoffs)  # its retry handed over, behind the run's end
                gate.set()
                await asyncio.wait_for(closing, 10)

            asyncio.run(close_elsewhere())
            assert ended.result(10) is True and closed == [True]
        finally:
            stop()

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

    def test_a_handshake_paused_on_its_loop_is_reported(self, monkeypatch):
        """A first handshake under way on an asyncio.Runner paused between two run() calls: close() from another
        loop cannot close that session there, and says so; a close() on that loop closes it."""
        gates = {}

        async def run_session(_self, ready, closed):  # the transport: its handshake waits for the test
            gates["shaking"].set()
            await gates["release"].wait()
            ready.set_result(object())
            await closed.wait()

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        server = MCPServer(transport="stdio", command="unused")
        with asyncio.Runner() as runner:

            async def start():
                gates["shaking"], gates["release"] = asyncio.Event(), asyncio.Event()
                connecting = asyncio.ensure_future(server._connect())
                await gates["shaking"].wait()
                return connecting

            connecting = runner.run(start())  # the runner pauses with the handshake under way
            with pytest.raises(RuntimeError, match="not running"):
                asyncio.run(asyncio.wait_for(server.close(), 10))

            async def finish():
                gates["release"].set()
                await connecting
                await server.close()

            runner.run(asyncio.wait_for(finish(), 10))
        assert server._session is None and server._owner_task is None

    def test_a_handshake_that_publishes_while_close_runs_and_then_pauses_is_reported(self, monkeypatch):
        """close() begins during a first handshake on another loop; the handshake publishes its session and that loop
        pauses before close()'s handed close runs: still the server's session, it is reported, not missed."""
        import threading

        owner_loop = asyncio.new_event_loop()
        shaking, gates = threading.Event(), {}

        async def run_session(_self, ready, closed):  # the transport: its handshake waits for the test
            gates["release"] = asyncio.Event()
            shaking.set()
            await gates["release"].wait()
            ready.set_result(object())
            await closed.wait()

        open_session = MCPServer._open_session

        async def open_then_pause(self):
            session = await open_session(self)
            asyncio.get_running_loop().stop()  # its loop pauses right after the publish
            return session

        monkeypatch.setattr(MCPServer, "_run_session", run_session)
        monkeypatch.setattr(MCPServer, "_open_session", open_then_pause)
        server = MCPServer(transport="stdio", command="unused")
        first = threading.Thread(target=owner_loop.run_forever)
        first.start()
        try:
            asyncio.run_coroutine_threadsafe(server._connect(), owner_loop)
            assert shaking.wait(10)

            async def close_elsewhere():
                closing = asyncio.ensure_future(server.close())
                await until(lambda: server._lock is not None and server._lock._waiters)  # its close, behind the lock
                owner_loop.call_soon_threadsafe(gates["release"].set)
                with pytest.raises(RuntimeError, match="not running"):
                    await asyncio.wait_for(closing, 10)

            asyncio.run(close_elsewhere())
        finally:
            first.join(10)  # it stopped after the publish (or the test failed before: then it still runs)
            running = first if first.is_alive() else threading.Thread(target=owner_loop.run_forever)
            if running is not first:
                running.start()  # the loop runs again: the close it was handed happens there

            async def settle():  # on its own loop: the close finished, then whatever is left cancelled and awaited
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

    def test_a_close_that_looks_up_the_owner_as_the_handshake_publishes_still_finds_it(self, monkeypatch):
        """close() on another thread looks up the session's owner just as the handshake publishes the session, and the
        owner's loop pauses right after: the lookup still finds the owner, so close() says the session is left on a
        loop that is not running. With one field for the handshake and another once published, it read the first
        before the publish and the second after the publish had cleared it, and returned as if nothing were open."""
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
            with pytest.raises(RuntimeError, match="not running"):
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

    @pytest.mark.asyncio
    async def test_what_close_owes_reads_the_owner_before_the_owners_shutting_down(self):
        """A close on the session's loop moves its owner to ``_owners_closing`` and then lets go of the reference.
        close() on another thread reads the reference first, so it finds the owner in one of the two even when that
        move happens between its two reads (reading ``_owners_closing`` first, it found neither)."""
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
            left = server._owed(closer, {}, owner, set())
            assert left.owners == 1 and not armed
        finally:
            paused.close()

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
        and close() succeeds, as it did before close() worked across loops."""
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

    def test_a_close_pending_when_close_began_whose_waits_are_cut_is_reported(self):
        """A teardown already going on another running loop when close() began, every wait close() sends for it
        cancelled there: still going after them, it is owed."""
        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()

        async def make_event():
            return asyncio.Event()

        release = asyncio.run_coroutine_threadsafe(make_event(), owner_loop).result(10)

        class Slow:
            async def close(self):
                await release.wait()

        service = MCPServer(transport="http", service="api")
        service._run_connections["run"] = _RunConnection(Slow(), "fingerprint", owner_loop)

        async def let_go():
            service._retire_run_connection("run")  # its teardown going there before close() begins

        def cancel_the_waits(loop, coro, **kwargs):
            task = asyncio.Task(coro, loop=loop, **kwargs)
            if getattr(coro, "__qualname__", "").endswith(("_wait_loop_closes", "_close_run_connections")):
                task.cancel()
            return task

        try:
            asyncio.run_coroutine_threadsafe(let_go(), owner_loop).result(10)
            owner_loop.set_task_factory(cancel_the_waits)
            with pytest.raises(RuntimeError, match="still closing"):
                asyncio.run(asyncio.wait_for(service.close(), 10))
        finally:
            owner_loop.call_soon_threadsafe(release.set)
            stop()

    def test_a_close_waiting_on_anothers_handoff_reports_what_it_left(self):
        """Two close() at once on one loop, the run connection on another: the second waits for the first's
        handoff, and when a session close was cut there it says the session is still open, as the first does."""
        import threading

        from timbal.core.mcp import _RunConnection

        owner_loop, stop = self.owner_thread()
        started = threading.Event()

        class Hangs:
            async def close(self):
                started.set()
                await asyncio.Event().wait()

        service = MCPServer(transport="http", service="api")
        try:
            service._run_connections["run"] = _RunConnection(Hangs(), "fingerprint", owner_loop)

            async def both():
                first = asyncio.ensure_future(service.close())
                second = asyncio.ensure_future(service.close())
                assert await asyncio.to_thread(started.wait, 10)

                def cut():
                    for task in list(service._close_tasks):
                        if task.get_loop() is owner_loop:
                            task.cancel()

                owner_loop.call_soon_threadsafe(cut)
                outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 10)
                assert all(isinstance(o, RuntimeError) and "still open" in str(o) for o in outcomes), outcomes

            asyncio.run(both())
        finally:
            stop()

    def test_a_connection_closed_before_a_handoff_failed_is_not_kept_as_open(self):
        """The owner cancelled the work handed to it after a connection had closed there: only the connections not
        closed are kept for a later close(). Kept, a closed one was counted as still open by every close() from
        another loop until its run ended."""
        import concurrent.futures

        from timbal.core.mcp import _HandoffFailed, _RunConnection

        class Server:
            async def close(self):
                pass

        owner_loop, stop = self.owner_thread()
        service = MCPServer(transport="http", service="api")
        try:
            entry = _RunConnection(Server(), "fingerprint", owner_loop)

            async def close_there():
                entry.close_task = asyncio.ensure_future(entry.server.close())
                await entry.close_task

            asyncio.run_coroutine_threadsafe(close_there(), owner_loop).result(10)
            handed = concurrent.futures.Future()
            handed.cancel()  # the work handed there, cancelled once that close was done

            async def main():
                with pytest.raises(_HandoffFailed):
                    await asyncio.wait_for(service._finish_handoff(handed, owner_loop, [("run", entry)]), 10)
                await asyncio.wait_for(service.close(), 10)  # from another loop: nothing is still open

            asyncio.run(main())
            assert not service._unclosed
        finally:
            stop()

    def test_a_busy_owner_loop_does_not_hold_the_run_end_past_its_bound(self, monkeypatch):
        """The run's stream is closed from another loop while the run's loop is busy in a callback that does not yield:
        the end hook waits for the cleanup it hands over at most ``_RUN_END_CLOSE_S``, counted on the calling loop
        (the bound kept on the owner's loop has not started). The cleanup stays queued and runs once that loop is
        free."""
        import threading

        from timbal.core import mcp as mcp_module
        from timbal.core.mcp import _RunConnection

        closed = []

        class Server:
            async def close(self):
                closed.append(True)

        monkeypatch.setattr(mcp_module, "_RUN_END_CLOSE_S", 0.2)
        owner_loop, stop = self.owner_thread()
        service = MCPServer(transport="http", service="api")
        busy, free = threading.Event(), threading.Event()
        try:
            service._run_connections["run"] = _RunConnection(Server(), "fingerprint", owner_loop)
            service._run_hooked["run"] = owner_loop

            def hold():  # a callback that does not yield
                busy.set()
                free.wait(30)

            owner_loop.call_soon_threadsafe(hold)
            assert busy.wait(10)

            async def end_the_run():  # the run's end hook, on the loop that closed its stream
                waiting = service._run_ended("run")
                if waiting is not None:
                    await waiting

            asyncio.run(asyncio.wait_for(end_the_run(), 5))  # back within its bound
            assert closed == [] and service._run_connections["run"].ended  # queued there, not done yet
            free.set()
            asyncio.run(until(lambda: closed == [True] and not service._run_connections))  # done there once free
        finally:
            free.set()
            stop()

    @staticmethod
    def race_at(monkeypatch, service, attribute, mapping, other, other_first=False):
        """Install ``mapping`` as ``service``'s ``attribute`` and run ``other`` on a thread of its own at this thread's
        first look into it (after the look, or before it with ``other_first``), going on once ``other`` has finished
        or waits on the run-state lock this thread holds: where another thread's change would land. Returns a list
        that holds that thread once it started."""
        import threading

        from timbal.core import mcp as mcp_module

        class ObservedLock:
            """The run-state lock, saying when a second taker waits on it."""

            def __init__(self):
                self.lock, self.waited = threading.Lock(), threading.Event()

            def __enter__(self):
                if not self.lock.acquire(blocking=False):
                    self.waited.set()
                    self.lock.acquire()
                return self

            def __exit__(self, *_):
                self.lock.release()

        lock, armed, started = ObservedLock(), [True], []

        def race():
            if armed and threading.current_thread() is threading.main_thread():
                armed.clear()  # this thread is looking; the other one goes now
                started.append(threading.Thread(target=other))
                started[0].start()
                for _ in range(1000):  # until it is done, or waits on the lock this thread holds
                    if not started[0].is_alive() or lock.waited.wait(0.01):
                        break

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
        monkeypatch.setattr(mcp_module, "_RUN_STATE_LOCK", lock, raising=False)
        return started

    def test_two_closes_on_two_threads_take_a_closed_loops_connection_once(self, monkeypatch):
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
        other = self.race_at(monkeypatch, service, "_run_connections", pool, close_here)
        close_here()
        other[0].join(10)
        assert results == [None, None] and not service._run_connections

    def test_two_closes_on_two_threads_forget_a_closed_loops_close_record_once(self, monkeypatch):
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

        other = self.race_at(monkeypatch, service, "_run_closes", {"run": {record}}, the_other_close, other_first=True)
        asyncio.run(asyncio.wait_for(service.close(), 10))
        other[0].join(10)
        assert not errors and not service._run_closes and not service._close_tasks

    def test_the_run_budget_left_is_read_on_the_clock_that_set_it(self, monkeypatch):
        """A root Tool's stream is closed from another loop, whose clock has another origin (120 s behind), once its
        budget has run out: the run returns at once, the close of its platform connection going on behind it.
        Read on the closing loop's clock, the budget looked 120 s long and the run waited for the close."""
        import threading

        from timbal import Tool
        from timbal.state import set_call_id, set_parent_call_id, set_run_context

        closing, closed = threading.Event(), threading.Event()

        class SlowTeardown:
            async def close(self):
                closing.set()
                await asyncio.to_thread(release.wait, 30)
                closed.set()

        release = threading.Event()
        monkeypatch.setattr(
            "timbal.platform.utils._resolve_url_and_headers",
            lambda *_: ("https://api.example.test/mcp", {"Authorization": "Bearer run-a"}),
        )
        monkeypatch.setattr(MCPServer, "_new_project_server", lambda _self, _url, _headers: SlowTeardown())
        service = MCPServer(transport="http", service="api")

        async def streamer():
            async with service._project_connection():
                pass  # the run's connection, kept for its next call
            yield "first"
            await asyncio.Event().wait()

        class Behind(asyncio.SelectorEventLoop):
            def time(self):
                return super().time() - 120

        owner_loop, stop = self.owner_thread()
        try:

            async def read_first():
                set_run_context(None)
                set_call_id(None)
                set_parent_call_id(None)
                stream = Tool(name="streamer", handler=streamer, timeout=0.05)._stream()
                async for event in stream:
                    if event.type == "DELTA":
                        return stream, asyncio.get_running_loop().time()

            stream, read_at = asyncio.run_coroutine_threadsafe(read_first(), owner_loop).result(30)
            while owner_loop.time() < read_at + 0.1:  # the budget, set before that, has run out on its own clock
                threading.Event().wait(0.01)
            with asyncio.Runner(loop_factory=Behind) as runner:
                runner.run(asyncio.wait_for(stream.aclose(), 5))  # back at once, no budget left to wait in
            assert closing.wait(10) and not closed.is_set()  # the close goes on, behind the run
        finally:
            release.set()
            assert closed.wait(10)
            stop()

    def test_a_run_dropped_while_close_takes_its_connection_is_taken_once(self, monkeypatch):
        """The run's loop closed: its end hook drops the run (``_drop_run``) on one thread while close() on another
        takes the run's connection out of the pool. Both take the pool lock, so whichever comes second finds it gone
        instead of failing on the delete (``KeyError``, which ended close()'s cleanup)."""
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
        other = self.race_at(monkeypatch, service, "_run_connections", pool, drop)
        asyncio.run(asyncio.wait_for(service.close(), 10))
        other[0].join(10)
        assert not errors and not service._run_connections


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
