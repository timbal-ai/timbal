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
            assert meta[PROVENANCE_RUN_KEY] == tool_out.run_id
            assert meta[PROVENANCE_CALL_KEY] == tool_out.call_id  # the tool's own span
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
        assert meta[PROVENANCE_SESSION_KEY] == "chat-123"
        assert meta[PROVENANCE_RUN_KEY] == "chat-123"  # a new thread's first turn runs under the session id
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
            meta = MCPServer(transport="http", url="https://api.example.test/mcp", provenance=True)._provenance_meta()
        finally:
            set_run_context(None)
        assert meta[PROVENANCE_PARENT_KEY] == "run-parent" and meta[PROVENANCE_RUN_KEY] == ctx.id
        assert "secret-credential" not in str(meta)

    def test_on_by_default_only_for_the_platform_server(self):
        assert MCPServer(transport="http", service="api")._sends_provenance()
        assert not MCPServer(transport="http", url="https://example.com/mcp")._sends_provenance()


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
