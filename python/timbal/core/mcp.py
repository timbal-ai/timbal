import asyncio
from contextlib import asynccontextmanager
from functools import cached_property
from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import httpx
import structlog
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import Field, PrivateAttr, computed_field, model_validator

from ..types.content import FileContent, TextContent
from ..types.file import File
from ..types.message import Message
from .runnable import Runnable
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.mcp")


def _convert_call_tool_result(tool_name: str, result: mcp_types.CallToolResult) -> Any:
    """Convert an MCP CallToolResult into a value Timbal can feed back to the LLM.

    Text-only results collapse to a plain string, structured results return the
    structured payload, and binary content (images, audio, blob resources) is
    wrapped in Timbal Files inside a Message so the agent forwards them as
    FileContent tool results.
    """
    texts: list[str] = []
    files: list[File] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
        elif isinstance(block, mcp_types.ImageContent | mcp_types.AudioContent):
            files.append(File(f"data:{block.mimeType};base64,{block.data}"))
        elif isinstance(block, mcp_types.EmbeddedResource):
            resource = block.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                texts.append(resource.text)
            else:
                mime_type = resource.mimeType or "application/octet-stream"
                files.append(File(f"data:{mime_type};base64,{resource.blob}"))
        elif isinstance(block, mcp_types.ResourceLink):
            texts.append(str(block.uri))

    if result.isError:
        raise RuntimeError("\n".join(texts) or f"MCP tool '{tool_name}' call failed.")

    if files:
        content: list[Any] = [TextContent(text=text) for text in texts]
        content.extend(FileContent(file=file) for file in files)
        return Message(role="assistant", content=content)
    # Per the MCP spec, text blocks carry the canonical serialized result even when
    # structuredContent is present (e.g. FastMCP wraps returns as {"result": ...}),
    # so prefer text and only fall back to the structured payload.
    if len(texts) == 1:
        return texts[0]
    if texts:
        return texts
    return result.structuredContent


class MCPTool(Tool):
    """A Tool whose parameter schema comes from an MCP server instead of handler introspection.

    The handler is a ``**kwargs`` passthrough that forwards the arguments to the
    MCP session, so ``params_model`` accepts anything; the schema the LLM sees is
    the server-declared JSON schema.
    """

    input_schema: dict[str, Any] = Field(default_factory=dict)
    """The JSON schema declared by the MCP server for this tool's arguments."""

    @override
    @computed_field
    @cached_property
    def params_model_schema(self) -> dict[str, Any]:
        """See base class."""
        schema = dict(self.input_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema


class MCPServer(ToolSet):
    """A single MCP server connection that resolves its tools at runtime.

    Each MCPServer instance represents one MCP server. Configure the
    transport type and its required parameters:

        MCPServer(transport="stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
        MCPServer(transport="http", url="https://api.timbal.ai/mcp")
    """

    name: str | None = None
    """Optional identifier for this server.

    When set, each resolved tool is exposed to the agent as
    ``{name}__{tool}`` so two servers that declare the same bare tool name
    don't collide in the agent's flat registry. The bare name is still used
    for ``session.call_tool``. Also used by codegen (``remove-tool --name``).
    """

    transport: Literal["stdio", "http"]

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    _session: Any | None = PrivateAttr(default=None)
    _tools_cache: list[Runnable] | None = PrivateAttr(default=None)
    _context: Any | None = PrivateAttr(default=None)
    _lock: asyncio.Lock | None = PrivateAttr(default=None)

    def _get_lock(self) -> asyncio.Lock:
        # Lazily create so model construction doesn't require a running loop.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServer":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("'command' is required for stdio transport")
        elif self.transport == "http":
            if not self.url:
                raise ValueError("'url' is required for http transport")
        return self

    @asynccontextmanager
    async def _connect_stdio(self):
        assert self.command is not None
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env if self.env else None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Connected to MCP server via stdio", command=self.command)
                yield session

    @asynccontextmanager
    async def _connect_http(self):
        assert self.url is not None
        async with httpx.AsyncClient(headers=self.headers if self.headers else None) as http_client:
            async with streamable_http_client(self.url, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("Connected to MCP server via http", url=self.url)
                    yield session

    async def _open_session(self) -> ClientSession:
        """Open the transport. Caller must hold ``_get_lock()`` and check ``_session`` first."""
        if self.transport == "stdio":
            ctx = self._connect_stdio()
        else:
            ctx = self._connect_http()

        session = await ctx.__aenter__()
        self._session = session
        self._context = ctx
        return session

    async def _connect(self) -> ClientSession:
        """Establish connection and store session for reuse.

        Synchronized so parallel tool calls (agent multiplexing) don't each
        open a duplicate stdio subprocess / HTTP session and orphan the first.
        """
        if self._session is not None:
            return self._session

        async with self._get_lock():
            if self._session is not None:
                return self._session
            return await self._open_session()

    def _qualified_tool_name(self, tool_name: str) -> str:
        """Name exposed to the agent/LLM for an MCP tool.

        Prefix with ``{server}__`` when this server has a ``name``, so multiple
        MCPServer instances can coexist without their tools clobbering each
        other in the agent's flat registry. Without a server name the bare
        MCP tool name is kept (fine for a single unnamed server).
        """
        if self.name:
            return f"{self.name}__{tool_name}"
        return tool_name

    def _make_tool(self, mcp_tool: mcp_types.Tool) -> MCPTool:
        # Bare name for the wire call; qualified name for the agent registry.
        bare_name = mcp_tool.name
        exposed_name = self._qualified_tool_name(bare_name)
        description = mcp_tool.description or ""
        if self.name and description:
            description = f"[{self.name}] {description}"
        elif self.name:
            description = f"[{self.name}] {bare_name}"

        async def _handler(**kwargs: Any) -> Any:
            session = await self._connect()
            result = await session.call_tool(bare_name, arguments=kwargs)
            return _convert_call_tool_result(bare_name, result)

        return MCPTool(
            name=exposed_name,
            description=description,
            handler=_handler,
            input_schema=mcp_tool.inputSchema or {},
        )

    async def resolve(self) -> list[Runnable]:
        """See base class. Lists the server's tools once and caches them until close()."""
        if self._tools_cache is not None:
            return self._tools_cache

        async with self._get_lock():
            if self._tools_cache is not None:
                return self._tools_cache

            if self._session is None:
                await self._open_session()
            assert self._session is not None

            result = await self._session.list_tools()
            tools: list[Runnable] = [self._make_tool(mcp_tool) for mcp_tool in result.tools]
            logger.info("Resolved MCP tools", tools=[t.name for t in tools])
            self._tools_cache = tools
            return tools

    async def close(self) -> None:
        """Close the MCP server connection."""
        async with self._get_lock():
            if self._context is not None:
                try:
                    await self._context.__aexit__(None, None, None)
                except Exception as e:
                    logger.error("Error closing MCP connection", error=str(e))
                self._session = None
                self._context = None
                self._tools_cache = None

    def __del__(self):
        session = getattr(self, "_session", None)
        if session:
            logger.warning("MCPServer deleted without calling close()")
