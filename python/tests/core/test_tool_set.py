import asyncio
import time
from typing import Any

import pytest
from pydantic import ValidationError
from timbal import Agent, Tool
from timbal.core.tool_set import ToolSet
from timbal.types.message import Message

from .conftest import (
    assert_has_output_event,
    skip_if_agent_error,
)

# ==============================================================================
# Test ToolSet Implementations
# ==============================================================================

class StaticToolSet(ToolSet):
    """Simple ToolSet that returns a static list of tools."""
    
    async def resolve(self) -> list[Tool]:
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        def add(a: int, b: int) -> int:
            return a + b
        
        return [
            Tool(handler=greet),
            Tool(handler=add),
        ]


class ConditionalToolSet(ToolSet):
    """ToolSet that returns different tools based on a condition."""
    
    role: str
    
    async def resolve(self) -> list[Tool]:
        def view_profile() -> str:
            return "Viewing profile..."
        
        def delete_user(user_id: int) -> str:
            return f"Deleted user {user_id}"
        
        def modify_permissions(user_id: int, permission: str) -> str:
            return f"Modified permissions for user {user_id}: {permission}"
        
        if self.role == "admin":
            return [
                Tool(handler=view_profile),
                Tool(handler=delete_user),
                Tool(handler=modify_permissions),
            ]
        else:
            return [
                Tool(handler=view_profile),
            ]


class LazyToolSet(ToolSet):
    """ToolSet that lazily initializes resources."""
    
    connection_string: str
    _initialized: bool = False
    _call_count: int = 0
    
    async def resolve(self) -> list[Tool]:
        # Simulate lazy initialization
        if not self._initialized:
            await asyncio.sleep(0.01)  # Simulate connection setup
            self._initialized = True
        
        self._call_count += 1
        
        def query(sql: str) -> str:
            return f"Query result for: {sql}"
        
        def get_stats() -> dict[str, Any]:
            return {"connection": self.connection_string, "calls": self._call_count}
        
        return [
            Tool(handler=query),
            Tool(handler=get_stats),
        ]


class EmptyToolSet(ToolSet):
    """ToolSet that returns no tools."""
    
    async def resolve(self) -> list[Tool]:
        return []


class AsyncInitToolSet(ToolSet):
    """ToolSet that performs async operations during resolution."""
    
    delay: float = 0.01
    
    async def resolve(self) -> list[Tool]:
        # Simulate async API call or database query
        await asyncio.sleep(self.delay)
        
        def process(data: str) -> str:
            return f"Processed: {data}"
        
        return [Tool(handler=process)]


class CounterToolSet(ToolSet):
    """ToolSet that tracks how many times resolve() is called."""
    
    _resolve_count: int = 0
    
    async def resolve(self) -> list[Tool]:
        self._resolve_count += 1
        
        def get_count() -> int:
            return self._resolve_count
        
        return [Tool(handler=get_count)]


# ==============================================================================
# Test ToolSet Creation and Validation
# ==============================================================================

class TestToolSetCreation:
    """Test ToolSet instantiation and validation."""
    
    def test_cannot_instantiate_abstract_toolset(self):
        """Test that ToolSet cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ToolSet()
    
    def test_static_toolset_creation(self):
        """Test creating a simple static ToolSet."""
        toolset = StaticToolSet()
        assert isinstance(toolset, ToolSet)
    
    def test_conditional_toolset_with_params(self):
        """Test creating a ToolSet with Pydantic parameters."""
        admin_toolset = ConditionalToolSet(role="admin")
        assert admin_toolset.role == "admin"
        
        user_toolset = ConditionalToolSet(role="user")
        assert user_toolset.role == "user"
    
    def test_lazy_toolset_with_validation(self):
        """Test that Pydantic validation works for ToolSet parameters."""
        toolset = LazyToolSet(connection_string="postgresql://localhost/db")
        assert toolset.connection_string == "postgresql://localhost/db"
        
        # Test that required fields are validated
        with pytest.raises(ValidationError):
            LazyToolSet()  # Missing connection_string
    
    def test_empty_toolset_creation(self):
        """Test creating a ToolSet that returns no tools."""
        toolset = EmptyToolSet()
        assert isinstance(toolset, ToolSet)


# ==============================================================================
# Test ToolSet Resolution
# ==============================================================================

class TestToolSetResolution:
    """Test ToolSet resolve() method behavior."""
    
    @pytest.mark.asyncio
    async def test_static_resolution(self):
        """Test resolving a static list of tools."""
        toolset = StaticToolSet()
        tools = await toolset.resolve()
        
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert all(isinstance(tool, Tool) for tool in tools)
        assert tools[0].name == "greet"
        assert tools[1].name == "add"
    
    @pytest.mark.asyncio
    async def test_conditional_resolution_admin(self):
        """Test conditional resolution for admin role."""
        toolset = ConditionalToolSet(role="admin")
        tools = await toolset.resolve()
        
        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "view_profile" in tool_names
        assert "delete_user" in tool_names
        assert "modify_permissions" in tool_names
    
    @pytest.mark.asyncio
    async def test_conditional_resolution_user(self):
        """Test conditional resolution for regular user role."""
        toolset = ConditionalToolSet(role="user")
        tools = await toolset.resolve()
        
        assert len(tools) == 1
        assert tools[0].name == "view_profile"
    
    @pytest.mark.asyncio
    async def test_lazy_initialization(self):
        """Test that lazy initialization happens on first resolve."""
        toolset = LazyToolSet(connection_string="test://db")
        
        assert not toolset._initialized
        assert toolset._call_count == 0
        
        # First resolve should initialize
        tools = await toolset.resolve()
        assert toolset._initialized
        assert toolset._call_count == 1
        assert len(tools) == 2
        
        # Second resolve should reuse initialized state
        tools = await toolset.resolve()
        assert toolset._call_count == 2
    
    @pytest.mark.asyncio
    async def test_empty_resolution(self):
        """Test that ToolSet can return empty list."""
        toolset = EmptyToolSet()
        tools = await toolset.resolve()
        
        assert isinstance(tools, list)
        assert len(tools) == 0
    
    @pytest.mark.asyncio
    async def test_async_resolution(self):
        """Test that resolve() supports async operations."""
        toolset = AsyncInitToolSet(delay=0.02)
        
        start_time = time.time()
        tools = await toolset.resolve()
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.02  # Should have waited for async operation
        assert len(tools) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_resolutions(self):
        """Test that resolve() can be called multiple times."""
        toolset = CounterToolSet()
        
        # First resolution
        tools1 = await toolset.resolve()
        assert toolset._resolve_count == 1
        
        # Second resolution
        tools2 = await toolset.resolve()
        assert toolset._resolve_count == 2
        
        # Third resolution
        tools3 = await toolset.resolve()
        assert toolset._resolve_count == 3


# ==============================================================================
# Test ToolSet Integration with Agent
# ==============================================================================

class TestToolSetAgentIntegration:
    """Test ToolSet integration with Agent."""
    
    @pytest.mark.asyncio
    async def test_agent_with_static_toolset(self):
        """Test agent using a static ToolSet."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            tools=[StaticToolSet()],
            max_iter=2,
        )
        
        prompt = Message.validate({"role": "user", "content": "Say hello to Alice"})
        result = agent(prompt=prompt)
        output = await result.collect()
        
        skip_if_agent_error(output, "agent_with_static_toolset")
        assert_has_output_event(output)
    
    @pytest.mark.asyncio
    async def test_agent_with_conditional_toolset(self):
        """Test agent with conditional ToolSet based on role."""
        admin_agent = Agent(
            name="admin_agent",
            model="openai/gpt-4o-mini",
            tools=[ConditionalToolSet(role="admin")],
            max_iter=2,
        )
        
        prompt = Message.validate({"role": "user", "content": "Delete user 123"})
        result = admin_agent(prompt=prompt)
        output = await result.collect()
        
        skip_if_agent_error(output, "agent_with_conditional_toolset_admin")
        assert_has_output_event(output)
    
    @pytest.mark.asyncio
    async def test_agent_with_mixed_tools_and_toolsets(self):
        """Test agent with both regular tools and ToolSets."""
        def standalone_tool(x: str) -> str:
            return f"Standalone: {x}"
        
        agent = Agent(
            name="mixed_agent",
            model="openai/gpt-4o-mini",
            tools=[
                standalone_tool,
                StaticToolSet(),
            ],
            max_iter=2,
        )
        
        prompt = Message.validate({"role": "user", "content": "Greet Bob"})
        result = agent(prompt=prompt)
        output = await result.collect()
        
        skip_if_agent_error(output, "agent_with_mixed_tools")
        assert_has_output_event(output)
    
    @pytest.mark.asyncio
    async def test_agent_with_empty_toolset(self):
        """Test agent with ToolSet that returns no tools."""
        agent = Agent(
            name="empty_toolset_agent",
            model="openai/gpt-4o-mini",
            tools=[EmptyToolSet()],
            max_iter=1,
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello"})
        result = agent(prompt=prompt)
        output = await result.collect()
        
        # Should work fine, just no tools available
        assert_has_output_event(output)
    
    @pytest.mark.asyncio
    async def test_agent_resolves_toolset_per_iteration(self):
        """Test that agent resolves ToolSet on each iteration."""
        counter_toolset = CounterToolSet()
        
        agent = Agent(
            name="counter_agent",
            model="openai/gpt-4o-mini",
            tools=[counter_toolset],
            max_iter=3,
        )
        
        prompt = Message.validate({"role": "user", "content": "What's the count?"})
        result = agent(prompt=prompt)
        output = await result.collect()
        
        skip_if_agent_error(output, "agent_resolves_toolset_per_iteration")
        # The toolset should have been resolved at least once
        assert counter_toolset._resolve_count >= 1
    
    @pytest.mark.asyncio
    async def test_agent_with_lazy_toolset(self):
        """Test agent with lazy-initialized ToolSet."""
        lazy_toolset = LazyToolSet(connection_string="test://lazy")
        
        agent = Agent(
            name="lazy_agent",
            model="openai/gpt-4o-mini",
            tools=[lazy_toolset],
            max_iter=2,
        )
        
        assert not lazy_toolset._initialized
        
        prompt = Message.validate({"role": "user", "content": "Query the database"})
        result = agent(prompt=prompt)
        output = await result.collect()
        
        skip_if_agent_error(output, "agent_with_lazy_toolset")
        # Lazy initialization should have happened
        assert lazy_toolset._initialized


# ==============================================================================
# Test ToolSet Edge Cases
# ==============================================================================

class TestToolSetEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_toolset_returning_duplicate_tool_names(self):
        """Test ToolSet that returns tools with duplicate names."""
        class DuplicateToolSet(ToolSet):
            async def resolve(self) -> list[Tool]:
                def tool_a() -> str:
                    return "A"
                
                def tool_b() -> str:
                    return "B"
                
                return [
                    Tool(name="same_name", handler=tool_a),
                    Tool(name="same_name", handler=tool_b),
                ]
        
        toolset = DuplicateToolSet()
        tools = await toolset.resolve()
        
        # Should return the tools (agent will handle duplicates)
        assert len(tools) == 2
        assert tools[0].name == tools[1].name
    
    @pytest.mark.asyncio
    async def test_toolset_with_pydantic_private_attrs(self):
        """Test that ToolSet can use Pydantic private attributes."""
        from pydantic import PrivateAttr
        
        class PrivateAttrToolSet(ToolSet):
            _cache: dict[str, Any] = PrivateAttr(default_factory=dict)
            
            async def resolve(self) -> list[Tool]:
                if "tools" not in self._cache:
                    def cached_tool() -> str:
                        return "cached"
                    self._cache["tools"] = [Tool(handler=cached_tool)]
                
                return self._cache["tools"]
        
        toolset = PrivateAttrToolSet()
        tools1 = await toolset.resolve()
        tools2 = await toolset.resolve()
        
        # Should return the same cached tools
        assert tools1 is tools2
    
    @pytest.mark.asyncio
    async def test_toolset_with_complex_pydantic_model(self):
        """Test ToolSet with complex Pydantic configuration."""
        from pydantic import Field, field_validator
        
        class ComplexToolSet(ToolSet):
            max_tools: int = Field(default=5, ge=1, le=10)
            prefix: str = Field(default="tool")
            
            @field_validator("prefix")
            @classmethod
            def validate_prefix(cls, v: str) -> str:
                if not v.isalnum():
                    raise ValueError("Prefix must be alphanumeric")
                return v
            
            async def resolve(self) -> list[Tool]:
                tools = []
                for i in range(self.max_tools):
                    def handler(x: str, idx=i) -> str:
                        return f"{self.prefix}_{idx}: {x}"
                    tools.append(Tool(name=f"{self.prefix}_{i}", handler=handler))
                return tools
        
        toolset = ComplexToolSet(max_tools=3, prefix="custom")
        tools = await toolset.resolve()
        
        assert len(tools) == 3
        assert all(tool.name.startswith("custom_") for tool in tools)
        
        # Test validation
        with pytest.raises(ValidationError):
            ComplexToolSet(max_tools=20)  # Exceeds max
        
        with pytest.raises(ValidationError):
            ComplexToolSet(prefix="invalid-prefix")  # Non-alphanumeric
