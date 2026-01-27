import json
from contextlib import asynccontextmanager
from functools import cached_property
from pathlib import Path
from typing import Any

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model, model_serializer

from .runnable import Runnable
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.mcp")


def _json_schema_to_python_type(schema: dict[str, Any]) -> type:
    """Convert JSON Schema type to Python type."""
    # Handle anyOf/oneOf (union types)
    if "anyOf" in schema or "oneOf" in schema:
        union_schemas = schema.get("anyOf") or schema.get("oneOf", [])
        # For union types, use Any to accept all variants
        # Pydantic will coerce as needed
        return Any
    
    schema_type = schema.get("type", "string")
    
    # Handle array of types (e.g., ["string", "null"])
    if isinstance(schema_type, list):
        # Use Any for complex type arrays
        return Any
    
    if schema_type == "array":
        items = schema.get("items", {})
        return list[_json_schema_to_python_type(items)]
    if schema_type == "object":
        return dict[str, Any]
    
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
    }
    return type_map.get(schema_type, Any)


def _validate_server_config(name: str, config: dict[str, Any]) -> None:
    """Validate server configuration has required fields for its transport type."""
    transport = config.get("transport")
    if not transport:
        raise ValueError(f"Server '{name}': 'transport' is required")
    
    if transport == "stdio":
        if not config.get("command"):
            raise ValueError(f"Server '{name}': 'command' is required for stdio transport")
    elif transport in ("http", "sse"):
        if not config.get("url"):
            raise ValueError(f"Server '{name}': 'url' is required for {transport} transport")
    else:
        raise ValueError(f"Server '{name}': unsupported transport '{transport}'")


def _load_mcp_json(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load and parse mcp.json file, returning normalized servers dict."""
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both formats:
    # {"mcpServers": {...}} - Cursor-style
    # {"server1": {...}} - Direct format
    if "mcpServers" in data:
        servers = data["mcpServers"]
    else:
        # Check if it looks like direct server configs
        servers = {k: v for k, v in data.items() if isinstance(v, dict) and "transport" in v}
        if not servers:
            servers = data  # Let validation catch issues
    
    return servers


class MCPTool(Tool):
    """
    A Tool that wraps an MCP server tool.
    
    Unlike regular Tools that derive their schema from the handler's function
    signature, MCPTool uses the MCP inputSchema directly. This preserves
    complex nested structures for LLM tool calling.
    """

    mcp_input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="The original MCP inputSchema for this tool",
    )
    
    _params_model_cache: type[BaseModel] | None = PrivateAttr(default=None)

    @property
    def params_model(self) -> type[BaseModel]:
        """Create params_model from MCP schema instead of handler introspection.
        
        This is critical: the base Tool class creates params_model from the handler
        signature. Since MCP handlers use **kwargs, that model would be empty and
        all parameters would be stripped during validation (extra="ignore").
        
        We override to create a model from the MCP inputSchema with extra="allow"
        so all parameters are passed through to the handler.
        """
        if self._params_model_cache is None:
            schema = self.mcp_input_schema
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            
            fields = {}
            for field_name, field_schema in properties.items():
                field_type = _json_schema_to_python_type(field_schema)
                description = field_schema.get("description", "")
                
                if field_name in required:
                    if description:
                        fields[field_name] = (field_type, Field(..., description=description))
                    else:
                        fields[field_name] = (field_type, ...)
                else:
                    default = field_schema.get("default")
                    if description:
                        fields[field_name] = (field_type | None, Field(default, description=description))
                    else:
                        fields[field_name] = (field_type | None, default)
            
            model_name = self.name.title().replace("_", "").replace("-", "") + "Params"
            # Use extra="allow" to pass through any additional parameters
            self._params_model_cache = create_model(
                model_name, 
                __config__=ConfigDict(extra="allow"),
                **fields
            )
        
        return self._params_model_cache

    def format_params_model_schema(self) -> dict[str, Any]:
        """Return the original MCP schema directly.
        
        This overrides the base implementation to use the MCP inputSchema,
        preserving complex nested structures that LLMs need for correct
        parameter formatting.
        """
        schema = dict(self.mcp_input_schema)
        
        # Apply schema filtering if configured
        if self.schema_params_mode == "required":
            selected = set(schema.get("required", []))
        else:
            selected = set(schema.get("properties", {}).keys())
        
        if self.schema_include_params:
            selected.update(self.schema_include_params)
        if self.schema_exclude_params:
            selected.difference_update(self.schema_exclude_params)
        
        # Filter properties
        properties = {k: v for k, v in schema.get("properties", {}).items() if k in selected}
        
        # Handle background mode
        if self.background_mode != "never":
            properties["run_in_background"] = {
                "type": "boolean",
                "default": self.background_mode == "always",
                "description": "Run in the background",
            }
        
        return {**schema, "properties": properties}

    @property
    def anthropic_schema(self) -> dict[str, Any]:
        """Generate Anthropic-compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description or "",
            "input_schema": self.format_params_model_schema(),
        }

    @property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """Generate OpenAI chat completions tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.format_params_model_schema(),
            },
        }

    @property
    def openai_responses_schema(self) -> dict[str, Any]:
        """Generate OpenAI responses API tool schema."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description or "",
            "parameters": self.format_params_model_schema(),
        }

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        """Serialize using anthropic schema."""
        return self.anthropic_schema


class MCPToolSet(ToolSet):
    """ToolSet that dynamically loads tools from MCP servers.
    
    Supports loading configuration from:
    - mcp.json file (Cursor-compatible format)
    - Direct server configuration dict
    - Auto-discovery from project directory
    """

    config_path: str | Path | None = None
    """Path to mcp.json configuration file."""

    servers: dict[str, dict[str, Any]] | None = None
    """Direct server configuration dictionary."""

    _sessions: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Active MCP sessions by server name."""

    _tools_cache: list[Runnable] | None = PrivateAttr(default=None)
    """Cached tools from MCP servers."""

    _contexts: list[Any] = PrivateAttr(default_factory=list)
    """Active context managers for cleanup."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        servers: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        if config_path is None and servers is None:
            raise ValueError("Either 'config_path' or 'servers' must be provided")
        super().__init__(config_path=config_path, servers=servers, **kwargs)

    @classmethod
    def from_project(cls, start_path: str | Path | None = None) -> "MCPToolSet":
        """Create MCPToolSet by auto-discovering mcp.json in project.
        
        Searches for mcp.json starting from the given path and walking up
        the directory tree until found.
        """
        if start_path is None:
            current = Path.cwd()
        else:
            current = Path(start_path).expanduser().resolve()
            if current.is_file():
                current = current.parent
        
        while True:
            config_path = current / "mcp.json"
            if config_path.exists():
                logger.info("Discovered MCP configuration", path=str(config_path))
                return cls(config_path=config_path)
            
            parent = current.parent
            if parent == current:
                break
            current = parent
        
        raise FileNotFoundError(f"Could not find mcp.json in {start_path or Path.cwd()} or parent directories")

    def _get_servers(self) -> dict[str, dict[str, Any]]:
        """Load and validate server configurations."""
        if self.config_path:
            path = Path(self.config_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"MCP config file not found: {path}")
            servers = _load_mcp_json(path)
        elif self.servers:
            servers = self.servers
        else:
            raise ValueError("No configuration provided")
        
        # Validate all server configs
        for name, config in servers.items():
            _validate_server_config(name, config)
        
        return servers

    @asynccontextmanager
    async def _connect_stdio(self, name: str, config: dict[str, Any]):
        """Connect to an MCP server via stdio transport."""
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env", {}),
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Connected to MCP server via stdio", server=name)
                yield session

    @asynccontextmanager
    async def _connect_http(self, name: str, config: dict[str, Any]):
        """Connect to an MCP server via HTTP transport."""
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise ImportError("HTTP transport requires: pip install mcp[http]") from exc
        
        async with streamable_http_client(config["url"]) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Connected to MCP server via http", server=name, url=config["url"])
                yield session

    @asynccontextmanager
    async def _connect_sse(self, name: str, config: dict[str, Any]):
        """Connect to an MCP server via SSE transport."""
        try:
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise ImportError("SSE transport requires: pip install mcp[sse]") from exc
        
        headers = config.get("headers", {})
        async with sse_client(config["url"], headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Connected to MCP server via sse", server=name, url=config["url"])
                yield session

    async def _connect_server(self, name: str, config: dict[str, Any]) -> ClientSession:
        """Connect to an MCP server and store the session.
        
        Note: This method enters async context managers that must be cleaned up
        by calling close(). The contexts are stored in self._contexts.
        """
        transport = config["transport"]
        
        if transport == "stdio":
            ctx = self._connect_stdio(name, config)
        elif transport == "http":
            ctx = self._connect_http(name, config)
        elif transport == "sse":
            ctx = self._connect_sse(name, config)
        else:
            raise ValueError(f"Unsupported transport: {transport}")
        
        # Enter the async context manager and store for later cleanup
        # We need to use the async context manager protocol manually here
        # because we want to keep the session alive across multiple calls
        session = await ctx.__aenter__()  # type: ignore[union-attr]
        self._sessions[name] = session
        self._contexts.append((name, ctx))
        return session

    def _create_tool(self, name: str, mcp_tool: Any, session: ClientSession) -> MCPTool:
        """Create an MCPTool from an MCP tool definition."""
        tool_name = mcp_tool.name
        
        async def handler(**kwargs: Any) -> Any:
            """Call the MCP tool."""
            try:
                result = await session.call_tool(tool_name, kwargs)
                
                # Extract content from MCP response
                if hasattr(result, "content") and result.content:
                    items = result.content
                    if len(items) == 1 and hasattr(items[0], "text"):
                        return items[0].text
                    return [getattr(item, "text", str(item)) for item in items]
                return str(result)
            except Exception as e:
                error_msg = f"Error calling MCP tool '{tool_name}': {e}"
                logger.error(error_msg, tool=tool_name, server=name)
                return error_msg
        
        handler.__name__ = tool_name
        
        return MCPTool(
            name=tool_name,
            description=mcp_tool.description or f"MCP tool: {tool_name}",
            handler=handler,
            mcp_input_schema=mcp_tool.inputSchema,
            metadata={"type": "MCP_Tool", "mcp_server": name},
        )

    async def resolve(self) -> list[Runnable]:
        """Resolve and return tools from all configured MCP servers."""
        if self._tools_cache is not None:
            return self._tools_cache
        
        try:
            servers = self._get_servers()
            if not servers:
                logger.warning("No MCP servers configured")
                self._tools_cache = []
                return []
            
            all_tools = []
            
            for name, config in servers.items():
                try:
                    session = await self._connect_server(name, config)
                    result = await session.list_tools()
                    
                    for mcp_tool in result.tools:
                        tool = self._create_tool(name, mcp_tool, session)
                        all_tools.append(tool)
                    
                    logger.info("Loaded tools from MCP server", server=name, count=len(result.tools))
                    
                except Exception as e:
                    logger.warning("Failed to connect to MCP server", server=name, error=str(e))
            
            logger.info("Resolved MCP tools", total=len(all_tools), servers=len(servers))
            self._tools_cache = all_tools
            return all_tools
            
        except Exception as e:
            logger.error("Failed to resolve MCP tools", error=str(e))
            self._tools_cache = []
            return []

    async def close(self) -> None:
        """Close all MCP server connections."""
        for name, ctx in self._contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.error("Error closing MCP connection", server=name, error=str(e))
        
        self._sessions.clear()
        self._contexts.clear()
        self._tools_cache = None
        logger.info("Closed MCP connections")

    def __del__(self):
        """Warn if connections weren't properly closed."""
        # Use getattr with default to avoid AttributeError during garbage collection
        # when __pydantic_private__ may not be initialized
        sessions = getattr(self, "_sessions", None)
        if sessions:
            logger.warning("MCPToolSet deleted without calling close()")
