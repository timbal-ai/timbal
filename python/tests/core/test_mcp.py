import os
import sys

import pytest
from pydantic import ValidationError
from timbal import Agent
from timbal.core.mcp import MCPServer, MCPTool, _convert_call_tool_result
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

            tool_events = [
                e for e in events if isinstance(e, OutputEvent) and e.path == "timbal_mcp_agent.whoami"
            ]
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
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
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
