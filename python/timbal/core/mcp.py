from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import Field, PrivateAttr, model_validator

from .runnable import Runnable
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.mcp")


class MCPServer(ToolSet):
    """A single MCP server connection that resolves its tools at runtime.

    Each MCPServer instance represents one MCP server. Configure the
    transport type and its required parameters:

        MCPServer(transport="stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
        MCPServer(transport="http", url="https://api.timbal.ai/mcp")
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

    async def _connect(self) -> ClientSession:
        """Establish connection and store session for reuse."""
        if self._session is not None:
            return self._session

        if self.transport == "stdio":
            ctx = self._connect_stdio()
        else:
            ctx = self._connect_http()

        session = await ctx.__aenter__()
        self._session = session
        self._context = ctx
        return session

    async def resolve(self) -> list[Runnable]:
        session = await self._connect()
        result = await session.list_tools()
        print(result)
        # TODO
        return []

    async def close(self) -> None:
        """Close the MCP server connection."""
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
