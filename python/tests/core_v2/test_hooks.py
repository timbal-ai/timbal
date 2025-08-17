from typing import Any

import pytest
from pydantic import ValidationError
from timbal.core_v2.tool import Tool
from timbal.state import get_run_context
from timbal.types.events import OutputEvent

from .conftest import (
    assert_no_errors,
)


class TestHooks:
    """Test the pre_hook and post_hook functionality in Runnable components."""

    @pytest.mark.asyncio
    async def test_pre_hook_input_modification(self):
        """Test that pre_hook can modify input parameters before validation."""
        modified_values = []
        
        async def modify_input(input_dict: dict[str, Any]) -> None:
            """Pre-hook that modifies input parameters."""
            # Record original value
            modified_values.append(f"original:{input_dict['x']}")
            # Modify the input
            input_dict['x'] = f"modified:{input_dict['x']}"
            # Add new parameter
            input_dict['new_param'] = "added_by_hook"
        
        def handler(x: str, new_param: str = None) -> str:
            """Handler that uses both original and new parameters."""
            return f"result:{x}|{new_param}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            pre_hook=modify_input
        )
        
        # Execute tool
        result = await tool(x="original_value").collect()
        
        # Verify the hook was called and modified input
        assert len(modified_values) == 1
        assert modified_values[0] == "original:original_value"
        
        # Verify the handler received modified input
        assert isinstance(result, OutputEvent)
        assert_no_errors(result)
        assert result.output == "result:modified:original_value|added_by_hook"

    @pytest.mark.asyncio
    async def test_pre_hook_parameter_removal(self):
        """Test that pre_hook can remove parameters from input."""
        async def remove_param(input_dict: dict[str, Any]) -> None:
            """Pre-hook that removes a parameter."""
            if 'unwanted_param' in input_dict:
                del input_dict['unwanted_param']
        
        def handler(x: str) -> str:
            """Handler that only accepts x parameter."""
            return f"clean:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            pre_hook=remove_param
        )
        
        # Execute tool with unwanted parameter
        result = await tool(x="test", unwanted_param="should_be_removed").collect()
        
        # Verify execution succeeded (parameter was removed)
        assert isinstance(result, OutputEvent)
        assert_no_errors(result)
        assert result.output == "clean:test"

    @pytest.mark.asyncio
    async def test_post_hook_output_inspection(self):
        """Test that post_hook can inspect and record output."""
        captured_outputs = []
        
        async def capture_output(output: Any) -> None:
            """Post-hook that captures output for inspection."""
            captured_outputs.append(f"captured:{output}")
        
        def handler(x: str) -> str:
            """Simple handler."""
            return f"processed:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            post_hook=capture_output
        )
        
        # Execute tool
        result = await tool(x="test").collect()
        
        # Verify the hook captured the output
        assert len(captured_outputs) == 1
        assert captured_outputs[0] == "captured:processed:test"
        
        # Verify original output is unchanged
        assert isinstance(result, OutputEvent)
        assert_no_errors(result)
        assert result.output == "processed:test"

    @pytest.mark.asyncio
    async def test_post_hook_mutable_output_modification(self):
        """Test that post_hook can modify mutable outputs in-place."""
        async def modify_output(output: Any) -> None:
            """Post-hook that modifies mutable output."""
            if isinstance(output, dict):
                # Record how many keys were there before modification
                original_count = len(output)
                output['modified_by_hook'] = True
                output['original_count'] = original_count
        
        def handler(items: list[str]) -> dict[str, Any]:
            """Handler that returns a mutable dict."""
            return {
                'items': items,
                'count': len(items),
                'processed': True
            }
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            post_hook=modify_output
        )
        
        # Execute tool
        result = await tool(items=["a", "b", "c"]).collect()
        
        # Verify the hook modified the output
        assert isinstance(result, OutputEvent)
        assert_no_errors(result)
        output = result.output
        assert output['modified_by_hook'] is True
        assert output['original_count'] == 3  # 3 original keys before modification
        assert output['items'] == ["a", "b", "c"]  # Original data preserved
        assert len(output) == 5  # 3 original + 2 added by hook

    @pytest.mark.asyncio
    async def test_context_access_in_hooks(self):
        """Test that hooks can access run context via get_run_context()."""
        context_data = []
        
        async def pre_hook_with_context(input_dict: dict[str, Any]) -> None:
            """Pre-hook that uses run context."""
            run_context = get_run_context()
            # Store data in context for later use
            run_context.data['pre_hook_executed'] = True
            run_context.data['input_keys'] = list(input_dict.keys())
            context_data.append("pre_hook_executed")
        
        async def post_hook_with_context(output: Any) -> None:
            """Post-hook that reads from run context."""
            run_context = get_run_context()
            # Read data stored by pre_hook
            if run_context.data.get('pre_hook_executed'):
                context_data.append("post_hook_found_pre_data")
            # Add more data
            run_context.data['output_type'] = type(output).__name__
        
        def handler(x: str) -> str:
            """Simple handler."""
            return f"handled:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            pre_hook=pre_hook_with_context,
            post_hook=post_hook_with_context
        )
        
        # Execute tool
        result = await tool(x="test").collect()
        
        # Verify both hooks executed and shared data via context
        assert len(context_data) == 2
        assert "pre_hook_executed" in context_data
        assert "post_hook_found_pre_data" in context_data
        
        # Verify context data was set
        run_context = get_run_context()
        assert run_context.data['pre_hook_executed'] is True
        assert run_context.data['input_keys'] == ['x']
        assert run_context.data['output_type'] == 'str'

    @pytest.mark.asyncio
    async def test_both_hooks_together(self):
        """Test that pre_hook and post_hook work together in sequence."""
        execution_order = []
        
        async def pre_hook(input_dict: dict[str, Any]) -> None:
            """Pre-hook that modifies input and records execution."""
            execution_order.append("pre_hook")
            input_dict['x'] = f"pre:{input_dict['x']}"
        
        async def post_hook(output: Any) -> None:
            """Post-hook that records execution."""
            execution_order.append("post_hook")
        
        def handler(x: str) -> str:
            """Handler that records execution."""
            execution_order.append("handler")
            return f"handled:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            pre_hook=pre_hook,
            post_hook=post_hook
        )
        
        # Execute tool
        result = await tool(x="test").collect()
        
        # Verify execution order
        assert execution_order == ["pre_hook", "handler", "post_hook"]
        
        # Verify the pre_hook modification was applied
        assert isinstance(result, OutputEvent)
        assert_no_errors(result)
        assert result.output == "handled:pre:test"

    def test_hook_validation_non_callable(self):
        """Test that non-callable hooks are rejected during validation."""
        with pytest.raises(ValidationError) as exc_info:
            Tool(
                name="test_tool",
                handler=lambda x: x,
                pre_hook="not_callable"  # Should fail validation
            )
        
        # Pydantic provides its own callable validation message
        assert "Input should be callable" in str(exc_info.value)

    def test_hook_validation_sync_function(self):
        """Test that sync functions are rejected as hooks."""
        def sync_hook(input_dict: dict[str, Any]) -> None:
            pass
        
        with pytest.raises(ValidationError) as exc_info:
            Tool(
                name="test_tool",
                handler=lambda x: x,
                pre_hook=sync_hook  # Should fail validation
            )
        
        assert "Hook must be an async function" in str(exc_info.value)

    def test_hook_validation_generator_function(self):
        """Test that generator functions are rejected as hooks."""
        def sync_gen_hook(input_dict: dict[str, Any]):
            yield "not_allowed"
        
        with pytest.raises(ValidationError) as exc_info:
            Tool(
                name="test_tool",
                handler=lambda x: x,
                pre_hook=sync_gen_hook  # Should fail validation
            )
        
        # The generator check happens first now
        assert "Hook must not be a generator or async generator" in str(exc_info.value)

    def test_hook_validation_async_generator_function(self):
        """Test that async generator functions are rejected as hooks."""
        async def async_gen_hook(input_dict: dict[str, Any]):
            yield "not_allowed"
        
        with pytest.raises(ValidationError) as exc_info:
            Tool(
                name="test_tool",
                handler=lambda x: x,
                pre_hook=async_gen_hook  # Should fail validation
            )
        
        assert "Hook must not be a generator or async generator" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """Test that errors in hooks are properly handled."""
        async def failing_pre_hook(input_dict: dict[str, Any]) -> None:
            """Pre-hook that raises an error."""
            raise ValueError("Pre-hook error")
        
        def handler(x: str) -> str:
            """Handler that should not be called due to pre-hook error."""
            return f"handled:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            pre_hook=failing_pre_hook
        )
        
        # Execute tool
        result = await tool(x="test").collect()
        
        # Verify the error was captured
        assert isinstance(result, OutputEvent)
        assert result.error is not None
        assert "Pre-hook error" in result.error['message']
        assert result.output is None

    @pytest.mark.asyncio
    async def test_post_hook_error_handling(self):
        """Test that errors in post_hook don't affect the output."""
        async def failing_post_hook(output: Any) -> None:
            """Post-hook that raises an error."""
            raise ValueError("Post-hook error")
        
        def handler(x: str) -> str:
            """Handler that completes successfully."""
            return f"handled:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            post_hook=failing_post_hook
        )
        
        # Execute tool
        result = await tool(x="test").collect()
        
        # Verify the post-hook error was captured but output is preserved
        assert isinstance(result, OutputEvent)
        assert result.error is not None
        assert "Post-hook error" in result.error['message']
        # The output should still be present since handler succeeded
        # (This behavior depends on implementation - adjust as needed)

    @pytest.mark.asyncio
    async def test_hooks_with_none_values(self):
        """Test that hooks work correctly with None values."""
        async def handle_none_pre_hook(input_dict: dict[str, Any]) -> None:
            """Pre-hook that handles None values."""
            if input_dict.get('x') is None:
                input_dict['x'] = "default_value"
        
        async def handle_none_post_hook(output: Any) -> None:
            """Post-hook that inspects None output."""
            run_context = get_run_context()
            run_context.data['output_was_none'] = output is None
        
        def handler(x: str) -> str:
            """Handler that might return None based on input."""
            if x == "return_none":
                return None
            return f"handled:{x}"
        
        tool = Tool(
            name="test_tool",
            handler=handler,
            pre_hook=handle_none_pre_hook,
            post_hook=handle_none_post_hook
        )
        
        # Test with None input
        result1 = await tool(x=None).collect()
        assert isinstance(result1, OutputEvent)
        assert_no_errors(result1)
        assert result1.output == "handled:default_value"
        
        # Test with input that causes None output
        result2 = await tool(x="return_none").collect()
        assert isinstance(result2, OutputEvent)
        assert_no_errors(result2)
        assert result2.output is None
        
        # Verify context data
        run_context = get_run_context()
        assert run_context.data['output_was_none'] is True