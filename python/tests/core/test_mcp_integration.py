import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from timbal import Agent, MCPToolSet
from timbal.core.mcp import MCPTool, _validate_server_config


class TestConfigValidation:
    """Test MCP configuration validation."""

    def test_stdio_config_valid(self):
        """Test valid stdio configuration passes validation."""
        _validate_server_config("test", {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
        })

    def test_stdio_config_missing_command(self):
        """Test stdio configuration without command raises error."""
        with pytest.raises(ValueError, match="'command' is required"):
            _validate_server_config("test", {"transport": "stdio"})

    def test_http_config_valid(self):
        """Test valid http configuration passes validation."""
        _validate_server_config("test", {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        })

    def test_http_config_missing_url(self):
        """Test http configuration without url raises error."""
        with pytest.raises(ValueError, match="'url' is required"):
            _validate_server_config("test", {"transport": "http"})

    def test_sse_config_valid(self):
        """Test valid sse configuration passes validation."""
        _validate_server_config("test", {
            "transport": "sse",
            "url": "http://localhost:8000/sse",
        })

    def test_sse_config_missing_url(self):
        """Test sse configuration without url raises error."""
        with pytest.raises(ValueError, match="'url' is required"):
            _validate_server_config("test", {"transport": "sse"})

    def test_missing_transport(self):
        """Test configuration without transport raises error."""
        with pytest.raises(ValueError, match="'transport' is required"):
            _validate_server_config("test", {"command": "python"})

    def test_unsupported_transport(self):
        """Test unsupported transport raises error."""
        with pytest.raises(ValueError, match="unsupported transport"):
            _validate_server_config("test", {"transport": "unknown"})


class TestMCPTool:
    """Test MCPTool class."""

    def test_mcp_tool_schema(self):
        """Test MCPTool uses original MCP schema."""
        async def handler(**kwargs):
            return "result"
        
        handler.__name__ = "test_tool"
        
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            handler=handler,
            mcp_input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        )
        
        schema = tool.format_params_model_schema()
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_mcp_tool_params_model(self):
        """Test MCPTool creates params_model from MCP schema, not handler."""
        async def handler(**kwargs):
            return kwargs  # Return kwargs to verify they're passed through
        
        handler.__name__ = "test_tool"
        
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            handler=handler,
            mcp_input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL"},
                },
                "required": ["url"],
            },
        )
        
        # Verify params_model has the correct fields
        assert "url" in tool.params_model.model_fields
        
        # Verify validation works and doesn't strip parameters
        validated = tool.params_model.model_validate({"url": "https://example.com"})
        assert validated.url == "https://example.com"

    async def test_mcp_tool_handler_receives_params(self):
        """Test that MCP tool handler receives all parameters."""
        received_kwargs = {}
        
        async def handler(**kwargs):
            received_kwargs.update(kwargs)
            return "success"
        
        handler.__name__ = "test_tool"
        
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            handler=handler,
            mcp_input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["url"],
            },
        )
        
        # Call the tool with parameters
        async for event in tool(url="https://example.com", timeout=30):
            pass
        
        # Verify handler received all parameters
        assert received_kwargs.get("url") == "https://example.com"
        assert received_kwargs.get("timeout") == 30

    async def test_mcp_tool_anyof_union_types(self):
        """Test that MCP tool handles anyOf union types (e.g., int | str)."""
        received_kwargs = {}
        
        async def handler(**kwargs):
            received_kwargs.update(kwargs)
            return "success"
        
        handler.__name__ = "send_message"
        
        # This schema matches Telegram MCP's send_message tool
        tool = MCPTool(
            name="send_message",
            description="Send a message",
            handler=handler,
            mcp_input_schema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "title": "Chat Id"
                    },
                    "message": {"type": "string", "title": "Message"},
                },
                "required": ["chat_id", "message"],
            },
        )
        
        # Test with integer chat_id (should not raise ValidationError)
        received_kwargs.clear()
        async for event in tool(chat_id=6038558870, message="Hello"):
            pass
        assert received_kwargs.get("chat_id") == 6038558870
        assert received_kwargs.get("message") == "Hello"
        
        # Test with string chat_id
        received_kwargs.clear()
        async for event in tool(chat_id="@username", message="Hello"):
            pass
        assert received_kwargs.get("chat_id") == "@username"

    def test_mcp_tool_anthropic_schema(self):
        """Test MCPTool generates correct Anthropic schema."""
        async def handler(**kwargs):
            return "result"
        
        handler.__name__ = "test_tool"
        
        tool = MCPTool(
            name="test_tool",
            description="Test description",
            handler=handler,
            mcp_input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        
        schema = tool.anthropic_schema
        assert schema["name"] == "test_tool"
        assert schema["description"] == "Test description"
        assert "input_schema" in schema

    def test_mcp_tool_openai_schema(self):
        """Test MCPTool generates correct OpenAI schema."""
        async def handler(**kwargs):
            return "result"
        
        handler.__name__ = "test_tool"
        
        tool = MCPTool(
            name="test_tool",
            description="Test description",
            handler=handler,
            mcp_input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        
        schema = tool.openai_responses_schema
        assert schema["type"] == "function"
        assert schema["name"] == "test_tool"
        assert "parameters" in schema


class TestMCPToolSet:
    """Test MCPToolSet public API."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test mcp.json file."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["test_server.py"],
                }
            }
        }
        config_file.write_text(json.dumps(config_data))
        return config_file

    @pytest.fixture
    def direct_config_file(self, tmp_path):
        """Create a test mcp.json file with direct format."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "test": {
                "transport": "stdio",
                "command": "python",
                "args": ["test_server.py"],
            }
        }
        config_file.write_text(json.dumps(config_data))
        return config_file

    def test_initialization_with_path(self, config_file):
        """Test MCPToolSet initialization with config path."""
        toolset = MCPToolSet(config_path=config_file)
        assert toolset.config_path == config_file

    def test_initialization_with_servers(self):
        """Test MCPToolSet initialization with server dict."""
        servers = {
            "test": {
                "transport": "stdio",
                "command": "python",
                "args": ["test.py"],
            }
        }
        toolset = MCPToolSet(servers=servers)
        assert toolset.servers == servers

    def test_initialization_no_config(self):
        """Test MCPToolSet initialization without config raises error."""
        with pytest.raises(ValueError, match="Either 'config_path' or 'servers'"):
            MCPToolSet()

    def test_from_project(self, config_file):
        """Test creating MCPToolSet from project discovery."""
        toolset = MCPToolSet.from_project(config_file.parent)
        assert toolset.config_path == config_file

    def test_from_project_nested(self, tmp_path):
        """Test discovery works from nested directories."""
        # Create nested directory structure
        project_dir = tmp_path / "project"
        sub_dir = project_dir / "src" / "app"
        sub_dir.mkdir(parents=True)

        # Create mcp.json in project root
        config_file = project_dir / "mcp.json"
        config_file.write_text('{"mcpServers": {"test": {"transport": "stdio", "command": "python"}}}')

        # Discover from subdirectory
        toolset = MCPToolSet.from_project(sub_dir)
        assert toolset.config_path == config_file

    def test_from_project_not_found(self, tmp_path):
        """Test discovery failure when mcp.json doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Could not find mcp.json"):
            MCPToolSet.from_project(tmp_path)

    def test_cursor_format_config(self, config_file):
        """Test loading Cursor-style mcp.json format."""
        toolset = MCPToolSet(config_path=config_file)
        servers = toolset._get_servers()
        assert "test" in servers
        assert servers["test"]["transport"] == "stdio"

    def test_direct_format_config(self, direct_config_file):
        """Test loading direct server dict format."""
        toolset = MCPToolSet(config_path=direct_config_file)
        servers = toolset._get_servers()
        assert "test" in servers
        assert servers["test"]["transport"] == "stdio"

    def test_invalid_server_config(self, tmp_path):
        """Test invalid server configuration raises error."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {"test": {"transport": "stdio"}}}')
        
        toolset = MCPToolSet(config_path=config_file)
        with pytest.raises(ValueError, match="'command' is required"):
            toolset._get_servers()

    @patch("timbal.core.mcp.stdio_client")
    async def test_resolve_stdio(self, mock_stdio_client, config_file):
        """Test resolving tools from stdio MCP server."""
        # Mock the MCP session
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        
        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        
        # Mock the context managers
        mock_read = MagicMock()
        mock_write = MagicMock()
        
        mock_stdio_ctx = AsyncMock()
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio_ctx.__aexit__ = AsyncMock()
        mock_stdio_client.return_value = mock_stdio_ctx
        
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock()
        
        with patch("timbal.core.mcp.ClientSession", return_value=mock_session_ctx):
            toolset = MCPToolSet(config_path=config_file)
            tools = await toolset.resolve()
            
            assert len(tools) == 1
            assert tools[0].name == "test_tool"
            assert isinstance(tools[0], MCPTool)

    async def test_resolve_empty_servers(self):
        """Test resolving with no servers configured returns empty list."""
        toolset = MCPToolSet(servers={})
        # Manually set empty servers (bypassing validation)
        toolset.servers = {}
        tools = await toolset.resolve()
        assert tools == []

    async def test_tools_cached(self, config_file):
        """Test that resolved tools are cached."""
        with patch("timbal.core.mcp.stdio_client") as mock_client:
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
            
            mock_read = MagicMock()
            mock_write = MagicMock()
            
            mock_stdio_ctx = AsyncMock()
            mock_stdio_ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
            mock_stdio_ctx.__aexit__ = AsyncMock()
            mock_client.return_value = mock_stdio_ctx
            
            mock_session_ctx = AsyncMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock()
            
            with patch("timbal.core.mcp.ClientSession", return_value=mock_session_ctx):
                toolset = MCPToolSet(config_path=config_file)
                
                # First call
                tools1 = await toolset.resolve()
                # Second call should use cache
                tools2 = await toolset.resolve()
                
                assert tools1 is tools2
                # stdio_client should only be called once
                assert mock_client.call_count == 1


class TestMCPIntegrationWithAgent:
    """Test MCP integration with Timbal Agent."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test mcp.json file."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["test_server.py"],
                }
            }
        }
        config_file.write_text(json.dumps(config_data))
        return config_file

    async def test_agent_with_mcp_toolset(self, config_file):
        """Test creating an agent with MCPToolSet."""
        mcp_tools = MCPToolSet(config_path=config_file)

        agent = Agent(
            name="test_agent",
            model="openai/gpt-4.1-nano",
            tools=[mcp_tools],
        )

        assert agent.name == "test_agent"
        assert any(isinstance(tool, MCPToolSet) for tool in agent.tools)

    async def test_agent_with_inline_servers(self):
        """Test creating an agent with inline MCP server config."""
        mcp_tools = MCPToolSet(servers={
            "test": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
            }
        })

        agent = Agent(
            name="test_agent",
            model="openai/gpt-4.1-nano",
            tools=[mcp_tools],
        )

        assert agent.name == "test_agent"
        assert any(isinstance(tool, MCPToolSet) for tool in agent.tools)
