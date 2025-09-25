import asyncio
import time
from functools import partial
from typing import Any

import pytest
from pydantic import BaseModel
from timbal import Tool
from timbal.core.llm_router import _llm_router
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
    
    def test_runnable_handler_rejected(self):
        """Test that Runnable instances cannot be used as handlers."""
        # Create a simple tool to use as a Runnable
        def simple_func(x: str) -> str:
            return f"simple:{x}"
        
        runnable_tool = Tool(handler=simple_func)
        
        # Try to create a tool with a Runnable as handler
        with pytest.raises(ValueError, match="Handler cannot be a Runnable instance"):
            Tool(handler=runnable_tool)
        
        # Error message should suggest using Agent or Workflow
        with pytest.raises(ValueError, match="use an Agent or Workflow instead"):
            Tool(handler=runnable_tool)
    
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
    
    def test_openai_chat_completions_schema_format(self):
        """Test OpenAI schema generation."""
        def handler(query: str, limit: int = 10) -> list:
            return []
        
        tool = Tool(
            name="search",
            description="Search for items",
            handler=handler
        )
        
        schema = tool.openai_chat_completions_schema
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
        tool = Tool(handler=_llm_router)
        
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


class TestToolSchemaConfiguration:
    """Test Tool schema configuration options (params_mode, include/exclude params)."""
    
    def test_schema_params_mode_all(self):
        """Test schema_params_mode='all' includes all parameters."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        tool = Tool(handler=handler, schema_params_mode="all")
        schema = tool.format_params_model_schema()
        
        # Should include all parameters
        assert "a" in schema["properties"]
        assert "b" in schema["properties"] 
        assert "c" in schema["properties"]
        assert len(schema["properties"]) == 3
    
    def test_schema_params_mode_required(self):
        """Test schema_params_mode='required' includes only required parameters."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        tool = Tool(handler=handler, schema_params_mode="required")
        schema = tool.format_params_model_schema()
        
        # Should only include required parameters
        assert "a" in schema["properties"]
        assert "b" not in schema["properties"]  # Has default
        assert "c" not in schema["properties"]  # Has default
        assert len(schema["properties"]) == 1
    
    def test_schema_include_params(self):
        """Test schema_include_params adds specific parameters."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        tool = Tool(
            handler=handler, 
            schema_params_mode="required",
            schema_include_params=["b", "c"]
        )
        schema = tool.format_params_model_schema()
        
        # Should include required param 'a' plus explicitly included 'b' and 'c'
        assert "a" in schema["properties"]  # Required
        assert "b" in schema["properties"]  # Explicitly included
        assert "c" in schema["properties"]  # Explicitly included
        assert len(schema["properties"]) == 3
    
    def test_schema_exclude_params(self):
        """Test schema_exclude_params removes specific parameters."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="all",
            schema_exclude_params=["c"]
        )
        schema = tool.format_params_model_schema()
        
        # Should include all params except excluded ones
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert "c" not in schema["properties"]  # Explicitly excluded
        assert len(schema["properties"]) == 2
    
    def test_schema_include_exclude_combination(self):
        """Test combining include and exclude parameters."""
        def handler(a: int, b: str = "default", c: float = 1.0, d: bool = True) -> str:
            return f"{a}-{b}-{c}-{d}"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="required",  # Only 'a' by default
            schema_include_params=["b", "c", "d"],  # Add these
            schema_exclude_params=["d"]  # But remove this one
        )
        schema = tool.format_params_model_schema()
        
        # Should have 'a' (required) + 'b', 'c' (included) - 'd' (excluded)
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert "c" in schema["properties"]
        assert "d" not in schema["properties"]  # Excluded overrides included
        assert len(schema["properties"]) == 3
    
    def test_openai_chat_completions_schema_with_configuration(self):
        """Test OpenAI schema generation with parameter configuration."""
        def handler(required_param: str, optional_param: int = 10) -> str:
            return f"{required_param}:{optional_param}"
        
        tool = Tool(
            name="configured_tool",
            description="A configured tool",
            handler=handler,
            schema_params_mode="required"
        )
        
        schema = tool.openai_chat_completions_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "configured_tool"
        assert schema["function"]["description"] == "A configured tool"
        
        # Should only have required parameter
        properties = schema["function"]["parameters"]["properties"]
        assert "required_param" in properties
        assert "optional_param" not in properties
        assert len(properties) == 1
    
    def test_anthropic_schema_with_configuration(self):
        """Test Anthropic schema generation with parameter configuration."""
        def handler(required_param: str, optional_param: int = 10) -> str:
            return f"{required_param}:{optional_param}"
        
        tool = Tool(
            name="configured_tool",
            description="A configured tool",
            handler=handler,
            schema_params_mode="required"
        )
        
        schema = tool.anthropic_schema
        assert schema["name"] == "configured_tool"
        assert schema["description"] == "A configured tool"
        
        # Should only have required parameter
        properties = schema["input_schema"]["properties"]
        assert "required_param" in properties
        assert "optional_param" not in properties
        assert len(properties) == 1


class TestToolConfigurationEdgeCases:
    """Test edge cases in tool configuration."""
    
    def test_empty_include_params_list(self):
        """Test that empty include_params list works correctly."""
        def handler(a: int, b: str = "default") -> str:
            return f"{a}:{b}"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="required",
            schema_include_params=[]  # Empty list
        )
        schema = tool.format_params_model_schema()
        
        # Should only have required parameters (empty include list adds nothing)
        assert "a" in schema["properties"]
        assert "b" not in schema["properties"]
        assert len(schema["properties"]) == 1
    
    def test_empty_exclude_params_list(self):
        """Test that empty exclude_params list works correctly."""
        def handler(a: int, b: str = "default") -> str:
            return f"{a}:{b}"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="all",
            schema_exclude_params=[]  # Empty list
        )
        schema = tool.format_params_model_schema()
        
        # Should have all parameters (empty exclude list removes nothing)
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert len(schema["properties"]) == 2
    
    def test_nonexistent_include_params(self):
        """Test including parameters that don't exist in the handler."""
        def handler(a: int) -> str:
            return str(a)
        
        tool = Tool(
            handler=handler,
            schema_include_params=["nonexistent_param"]
        )
        schema = tool.format_params_model_schema()
        
        # Should only include parameters that actually exist
        assert "a" in schema["properties"]
        assert "nonexistent_param" not in schema["properties"]
        assert len(schema["properties"]) == 1
    
    def test_nonexistent_exclude_params(self):
        """Test excluding parameters that don't exist in the handler."""
        def handler(a: int, b: str = "default") -> str:
            return f"{a}:{b}"
        
        tool = Tool(
            handler=handler,
            schema_exclude_params=["nonexistent_param"]
        )
        schema = tool.format_params_model_schema()
        
        # Should include all existing parameters (excluding non-existent has no effect)
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert len(schema["properties"]) == 2
    
    def test_tool_with_no_parameters(self):
        """Test tool configuration with handler that has no parameters."""
        def handler() -> str:
            return "no params"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="all",
            schema_include_params=["nonexistent"],
            schema_exclude_params=["also_nonexistent"]
        )
        schema = tool.format_params_model_schema()
        
        # Should have no parameters regardless of configuration
        assert len(schema["properties"]) == 0
    
    def test_complex_parameter_types(self):
        """Test schema configuration with complex parameter types."""
        def handler(
            simple_int: int,
            required_str: str,
            optional_list: list[str] = None,
            optional_dict: dict[str, Any] = None
        ) -> str:
            return "complex"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="required",
            schema_include_params=["optional_list"]
        )
        schema = tool.format_params_model_schema()
        
        # Should include required params + explicitly included optional params
        assert "simple_int" in schema["properties"]  # Required
        assert "required_str" in schema["properties"]  # Required
        assert "optional_list" in schema["properties"]  # Explicitly included
        assert "optional_dict" not in schema["properties"]  # Optional, not included
        
        # Verify the types are preserved correctly
        assert schema["properties"]["simple_int"]["type"] == "integer"
        assert schema["properties"]["required_str"]["type"] == "string"
        assert schema["properties"]["optional_list"]["type"] == "array"


class TestToolConfigurationWithDictCreation:
    """Test tool configuration when creating tools from dictionaries."""
    
    def test_dict_tool_creation_with_schema_config(self):
        """Test creating tools from dictionary configs with schema parameters."""
        def search_handler(query: str, limit: int = 10, include_meta: bool = False) -> list:
            return []
        
        # Test creating tool with dict configuration
        tool_config = {
            "name": "search",
            "description": "Search for items",
            "handler": search_handler,
            "schema_params_mode": "required",
            "schema_include_params": ["limit"]
        }
        
        tool = Tool(**tool_config)
        
        assert tool.name == "search"
        assert tool.description == "Search for items"
        assert tool.schema_params_mode == "required"
        assert tool.schema_include_params == ["limit"]
        
        # Test schema generation
        schema = tool.format_params_model_schema()
        assert "query" in schema["properties"]  # Required
        assert "limit" in schema["properties"]  # Explicitly included
        assert "include_meta" not in schema["properties"]  # Optional, not included
    
    def test_invalid_schema_params_mode(self):
        """Test that invalid schema_params_mode values are rejected."""
        def handler(x: int) -> int:
            return x
        
        with pytest.raises(ValueError):
            Tool(handler=handler, schema_params_mode="invalid_mode")
    
    @pytest.mark.asyncio
    async def test_configured_tool_execution(self):
        """Test that schema configuration doesn't affect tool execution."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        tool = Tool(
            handler=handler,
            schema_params_mode="required",  # Schema only shows 'a'
            schema_exclude_params=["b"]     # Schema excludes 'b'
        )
        
        # Despite schema configuration, tool should execute with all parameters
        result = tool(a=1, b="custom", c=2.5)
        output = await result.collect()
        
        assert_has_output_event(output)
        assert_no_errors(output)
        assert output.output == "1-custom-2.5"
        
        # Test with only required parameter (others should use defaults)
        result2 = tool(a=42)
        output2 = await result2.collect()
        
        assert_has_output_event(output2)
        assert_no_errors(output2)
        assert output2.output == "42-default-1.0"
