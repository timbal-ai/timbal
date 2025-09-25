import asyncio

import pytest
from timbal import Tool
from timbal.collectors.base import BaseCollector
from timbal.types.events import OutputEvent

from .conftest import (
    Timer,
)


class TestRunnableBase:
    """Test base Runnable functionality through Tool implementation."""
    
    def test_schema_generation(self, simple_tool):
        """Test that schemas are generated correctly."""
        # Should have both OpenAI and Anthropic schemas
        assert hasattr(simple_tool, 'openai_chat_completions_schema')
        assert hasattr(simple_tool, 'anthropic_schema')
        
        openai_chat_completions_schema = simple_tool.openai_chat_completions_schema
        assert openai_chat_completions_schema['type'] == 'function'
        assert openai_chat_completions_schema['function']['name'] == 'simple'
        assert 'parameters' in openai_chat_completions_schema['function']
        
        anthropic_schema = simple_tool.anthropic_schema
        assert anthropic_schema['name'] == 'simple'
        assert 'input_schema' in anthropic_schema
    
    def test_params_filtering(self):
        """Test parameter filtering functionality."""
        def handler(a: int, b: str = "default", c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"
        
        # Test with different schema_params_mode settings
        tool_all = Tool(handler=handler, schema_params_mode="all")
        tool_required = Tool(handler=handler, schema_params_mode="required")
        
        all_props = tool_all.format_params_model_schema()['properties']
        required_props = tool_required.format_params_model_schema()['properties']
        
        # All mode should include all parameters
        assert 'a' in all_props
        assert 'b' in all_props
        assert 'c' in all_props
        
        # Required mode should only include required parameters
        assert 'a' in required_props
        assert len(required_props) == 1  # Only 'a' is required
    
    def test_include_exclude_params(self):
        """Test schema_include_params and schema_exclude_params functionality."""
        def handler(a: int, b: str, c: float) -> str:
            return f"{a}-{b}-{c}"
        
        # Test schema_include_params
        tool_include = Tool(
            handler=handler, 
            schema_params_mode="required",  # Would normally include only required
            schema_include_params=["b", "c"]  # But we explicitly include b and c
        )
        
        include_props = tool_include.format_params_model_schema()['properties']
        assert 'a' in include_props  # Required param
        assert 'b' in include_props  # Explicitly included
        assert 'c' in include_props  # Explicitly included
        
        # Test schema_exclude_params
        tool_exclude = Tool(
            handler=handler,
            schema_params_mode="all",
            schema_exclude_params=["c"]
        )
        
        exclude_props = tool_exclude.format_params_model_schema()['properties']
        assert 'a' in exclude_props
        assert 'b' in exclude_props
        assert 'c' not in exclude_props  # Explicitly excluded
    
    def test_default_params(self):
        """Test default_params functionality."""
        def handler(a: str, b: str) -> str:
            return f"{a}:{b}"
        
        tool = Tool(
            handler=handler,
            default_params={"b": "fixed_value"}
        )
        
        # Should be able to call with only 'a' parameter
        result = tool(a="test")
        assert isinstance(result, BaseCollector)
    
    @pytest.mark.asyncio
    async def test_nested_paths(self):
        """Test that nested paths work correctly."""
        def handler(x: str) -> str:
            return x
        
        tool = Tool(name="child", handler=handler)
        original_path = tool._path
        
        # Nest under a parent
        tool.nest("parent")
        nested_path = tool._path
        
        assert original_path == "child"
        assert nested_path == "parent.child"
        
        # Test that the runnable has the correct nested path
        result = await tool(x="test").collect()
        assert result.path == "parent.child"
    
    @pytest.mark.asyncio
    async def test_serialization(self, simple_tool):
        """Test that tools can be serialized."""
        serialized = simple_tool.serialize()
        
        # Should contain the anthropic schema format
        assert 'name' in serialized
        assert 'input_schema' in serialized
        assert serialized['name'] == 'simple'


class TestErrorHandling:
    """Test error handling in Runnable execution."""
    
    @pytest.mark.asyncio
    async def test_handler_exception(self, error_tool):
        """Test that handler exceptions are captured in OutputEvent."""
        result = error_tool(message="test error")
        output = await result.collect()
        
        # Should have output event with error
        assert isinstance(output, OutputEvent)
        assert output.error is not None
        assert "Test error: test error" in output.error['message']
        assert output.error['type'] == 'ValueError'
    
    @pytest.mark.asyncio
    async def test_collect_with_error(self, error_tool):
        """Test that collect() works even when handler raises exception."""
        result = error_tool(message="test error")
        
        # collect() should return the OutputEvent with error
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.error is not None


class TestPerformance:
    """Test performance characteristics of Runnable execution."""
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that multiple runnables can execute concurrently."""
        def slow_handler(x: str) -> str:
            import time
            time.sleep(0.1)  # 100ms delay
            return f"slow:{x}"
        
        tool = Tool(handler=slow_handler)
        
        with Timer() as timer:
            # Run 3 tools concurrently
            results = await asyncio.gather(
                tool(x="1").collect(),
                tool(x="2").collect(),
                tool(x="3").collect()
            )
        
        # Should complete in roughly 100ms, not 300ms
        assert timer.elapsed < 0.2, f"Concurrent execution took too long: {timer.elapsed}s"
        
        # All results should be present
        assert len(results) == 3
        assert all(isinstance(r, OutputEvent) for r in results)
    
    @pytest.mark.asyncio
    async def test_event_caching_efficiency(self, simple_tool):
        """Test that event caching doesn't impact performance significantly."""
        result = simple_tool(x="test")
        
        # First collect
        with Timer() as timer1:
            output1 = await result.collect()
        
        # Second collect (should be cached)
        with Timer() as timer2:
            output2 = await result.collect()
        
        # Second collect should be much faster (cached)
        assert timer2.elapsed < timer1.elapsed / 2
        assert output1 == output2