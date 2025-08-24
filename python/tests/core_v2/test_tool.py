import asyncio
import time
from functools import partial
from typing import Any

import pytest
from pydantic import BaseModel
from timbal.core_v2.handlers import llm_router
from timbal.core_v2.tool import Tool
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from .conftest import (
    ERROR_SCENARIOS,
    TOOL_HANDLERS,
    Timer,
    assert_has_output_event,
    assert_no_errors,
    async_gen_handler,
    async_handler,
    error_handler,
    sync_gen_handler,
    sync_handler,
)


class TestToolCreation:
    """Test Tool instantiation and validation."""
    
    def test_missing_handler(self):
        """Test that Tool requires a handler."""
        with pytest.raises(ValueError, match="You must provide a handler"):
            Tool()
    
    def test_invalid_handler(self):
        """Test that non-callable objects are rejected."""
        handler = object()
        with pytest.raises(ValueError, match="Handler must be a function"):
            Tool(handler=handler)
    
    def test_lambda_without_name(self):
        """Test that lambdas require explicit names."""
        with pytest.raises(ValueError, match="A name must be specified when using a lambda"):
            Tool(handler=lambda x: x)
    
    def test_lambda_with_name(self):
        """Test that named lambdas work correctly."""
        tool = Tool(name="identity", handler=lambda x: x)
        assert tool.name == "identity"
        assert tool.params_model_schema["title"] == "IdentityParams"
    
    def test_function_without_name(self):
        """Test auto-naming from function name."""
        def identity(x: Any) -> Any:
            return x
        
        tool = Tool(handler=identity)
        assert tool.name == "identity"
        assert tool.params_model_schema["title"] == "IdentityParams"
    
    def test_function_with_explicit_name(self):
        """Test explicit naming overrides function name."""
        def identity(x: Any) -> Any:
            return x
        
        tool = Tool(name="my_tool", handler=identity)
        assert tool.name == "my_tool"
        assert tool.params_model_schema["title"] == "MyToolParams"
    
    def test_partial_function(self):
        """Test that partial functions require explicit names."""
        def add(a: int, b: int) -> int:
            return a + b
        
        partial_add = partial(add, b=10)
        
        with pytest.raises(ValueError, match="please provide a 'name' explicitly"):
            Tool(handler=partial_add)
        
        # Should work with explicit name
        tool = Tool(name="add_ten", handler=partial_add)
        assert tool.name == "add_ten"
    
    def test_tool_introspection(self):
        """Test that tools correctly introspect handler characteristics."""
        # Sync function
        sync_tool = Tool(handler=sync_handler)
        assert not sync_tool._is_coroutine
        assert not sync_tool._is_gen
        assert not sync_tool._is_async_gen
        assert not sync_tool._is_orchestrator
        
        # Async function
        async_tool = Tool(handler=async_handler)
        assert async_tool._is_coroutine
        assert not async_tool._is_gen
        assert not async_tool._is_async_gen
        
        # Sync generator
        sync_gen_tool = Tool(handler=sync_gen_handler)
        assert not sync_gen_tool._is_coroutine
        assert sync_gen_tool._is_gen
        assert not sync_gen_tool._is_async_gen
        
        # Async generator
        async_gen_tool = Tool(handler=async_gen_handler)
        assert not async_gen_tool._is_coroutine
        assert not async_gen_tool._is_gen
        assert async_gen_tool._is_async_gen


class TestToolExecution:
    """Test Tool execution patterns."""
    
    @pytest.mark.parametrize("handler_type,handler", TOOL_HANDLERS)
    @pytest.mark.asyncio
    async def test_handler_execution(self, handler_type, handler):
        """Test that different handler types execute correctly."""
        if handler_type == "sync_gen":
            tool = Tool(handler=handler)
            result = tool(count=3)
        else:
            tool = Tool(handler=handler)
            if handler_type in ["sync", "async"]:
                result = tool(x="test")
            else:  # async_gen
                result = tool(count=3)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
    
    @pytest.mark.asyncio
    async def test_sync_execution(self):
        """Test synchronous handler execution."""
        tool = Tool(handler=sync_handler)
        result = tool(x="hello")
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.output == "sync:hello"
    
    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test asynchronous handler execution."""
        tool = Tool(handler=async_handler)
        result = tool(x="hello")
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.output == "async:hello"
    
    @pytest.mark.asyncio
    async def test_sync_generator_execution(self):
        """Test synchronous generator handler execution."""
        tool = Tool(handler=sync_gen_handler)
        result = tool(count=3)
        
        output = await result.collect()
        
        # Should have a final output event with collected results
        assert isinstance(output, OutputEvent)
        assert output.output is not None
    
    @pytest.mark.asyncio
    async def test_async_generator_execution(self):
        """Test asynchronous generator handler execution."""
        tool = Tool(handler=async_gen_handler)
        result = tool(count=3)
        
        output = await result.collect()
        
        # Should have a final output event with collected results
        assert isinstance(output, OutputEvent)
        assert output.output is not None
    
    @pytest.mark.asyncio
    async def test_default_params(self):
        """Test that default_params are merged correctly."""
        tool = Tool(
            handler=sync_handler,
            default_params={"x": "fixed_value"}
        )
        
        # Should work without providing x
        result = tool()
        output = await result.collect()
        assert output.output == "sync:fixed_value"
        
        # Runtime params should override fixed params
        result = tool(x="runtime_value")
        output = await result.collect()
        assert output.output == "sync:runtime_value"
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test that invalid parameters are rejected."""
        def typed_handler(x: int, y: str = "default") -> str:
            return f"{x}:{y}"
        
        tool = Tool(handler=typed_handler)
        
        # Valid parameters should work
        result = tool(x=42, y="test")
        output = await result.collect()
        assert output.output == "42:test"
        
        # Invalid type should cause validation error (captured in OutputEvent)
        result = tool(x="not_an_int")
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.error is not None  # Should have validation error


class TestToolSchemas:
    """Test Tool schema generation."""
    
    def test_params_model_generation(self):
        """Test that parameter models are generated correctly."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        tool = Tool(handler=handler)
        params_model = tool.params_model
        
        # Should be a Pydantic model
        assert issubclass(params_model, BaseModel)
        
        # Check schema properties
        schema = tool.params_model_schema
        assert "properties" in schema
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert "c" in schema["properties"]
        
        # Check required fields
        assert "required" in schema
        assert "a" in schema["required"]
        assert "b" not in schema["required"]  # Has default
        assert "c" not in schema["required"]  # Has default
    
    def test_return_model_extraction(self):
        """Test that return type annotations are extracted."""
        def int_handler() -> int:
            return 42
        
        def str_handler() -> str:
            return "hello"
        
        int_tool = Tool(handler=int_handler)
        str_tool = Tool(handler=str_handler)
        
        assert int_tool.return_model == int
        assert str_tool.return_model == str
    
    def test_openai_schema_format(self):
        """Test OpenAI schema generation."""
        def handler(query: str, limit: int = 10) -> list:
            return []
        
        tool = Tool(
            name="search",
            description="Search for items",
            handler=handler
        )
        
        schema = tool.openai_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for items"
        assert "parameters" in schema["function"]
        assert "query" in schema["function"]["parameters"]["properties"]
    
    def test_anthropic_schema_format(self):
        """Test Anthropic schema generation."""
        def handler(query: str, limit: int = 10) -> list:
            return []
        
        tool = Tool(
            name="search",
            description="Search for items",
            handler=handler
        )
        
        schema = tool.anthropic_schema
        assert schema["name"] == "search"
        assert schema["description"] == "Search for items"
        assert "input_schema" in schema
        assert "query" in schema["input_schema"]["properties"]


class TestToolNesting:
    """Test Tool nesting functionality."""
    
    def test_initial_path(self):
        """Test that tools start with their name as path."""
        tool = Tool(name="test_tool", handler=sync_handler)
        assert tool._path == "test_tool"
    
    def test_nesting(self):
        """Test that nesting updates paths correctly."""
        tool = Tool(name="child", handler=sync_handler)
        tool.nest("parent")
        assert tool._path == "parent.child"
        
        # Test nested nesting
        tool.nest("grandparent.parent")
        assert tool._path == "grandparent.parent.child"
    
    @pytest.mark.asyncio
    async def test_path_in_events(self):
        """Test that events contain the correct path."""
        tool = Tool(name="test_tool", handler=sync_handler)
        tool.nest("parent")
        
        result = tool(x="test")
        output = await result.collect()
        
        # Output event should have the nested path
        assert output.path == "parent.test_tool"


class TestErrorHandling:
    """Test Tool error handling."""
    
    @pytest.mark.parametrize("error_type,handler", ERROR_SCENARIOS)
    @pytest.mark.asyncio
    async def test_error_capture(self, error_type, handler):
        """Test that handler errors are captured properly."""
        tool = Tool(handler=handler)
        result = tool(message="test error")
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.error is not None
        assert "test error" in output.error['message']
        assert output.error['type'] in ['ValueError']
    
    @pytest.mark.asyncio
    async def test_error_does_not_break_execution(self):
        """Test that errors don't prevent event generation."""
        tool = Tool(handler=error_handler)
        result = tool(message="test error")
        
        output = await result.collect()
        
        # Should still have output event with error
        assert isinstance(output, OutputEvent)
        assert output.error is not None


class TestPerformance:
    """Test Tool performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_sync_tools(self):
        """Test that sync tools can run concurrently."""
        def slow_handler(x: str) -> str:
            time.sleep(0.1)  # 100ms delay
            return f"slow:{x}"
        
        tool = Tool(handler=slow_handler)
        
        with Timer() as timer:
            results = await asyncio.gather(
                tool(x="1").collect(),
                tool(x="2").collect(),
                tool(x="3").collect()
            )
        
        # Should complete concurrently, not sequentially
        assert timer.elapsed < 0.2, f"Tools did not run concurrently: {timer.elapsed}s"
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_collect_performance(self):
        """Test collect() method performance."""
        tool = Tool(handler=sync_handler)
        result = tool(x="test")
        
        # First collect
        with Timer() as timer1:
            output1 = await result.collect()
        
        # Second collect (cached)
        with Timer() as timer2:
            output2 = await result.collect()
        
        # Second should be much faster
        assert timer2.elapsed < timer1.elapsed / 2
        assert output1 == output2


class TestLLMIntegration:
    """Test LLM integration through llm_router."""
    
    @pytest.mark.asyncio
    async def test_llm_router_tool(self):
        """Test that llm_router works as a tool."""
        tool = Tool(handler=llm_router)
        
        # Create a simple message using Message.validate()
        message = Message.validate({
            "role": "user", 
            "content": "Hello, how are you?"
        })
        
        result = tool(model="openai/gpt-4o-mini", messages=[message])
        output = await result.collect()
        
        assert isinstance(output, OutputEvent)
        assert output.error is None
        
        # Output should be a Message
        assert isinstance(output.output, Message)
        assert output.output.role == "assistant"
        assert len(output.output.content) > 0
