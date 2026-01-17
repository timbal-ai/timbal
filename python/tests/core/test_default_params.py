import asyncio
import time
from datetime import datetime

import pytest
from timbal import Agent, Tool
from timbal.types.message import Message

from .conftest import assert_has_output_event, assert_no_errors, skip_if_agent_error

# ==============================================================================
# Test Utility Functions for Default Params
# ==============================================================================


def get_current_time() -> str:
    """Get current time as string."""
    return datetime.now().strftime("%H:%M:%S")


def get_current_date() -> str:
    """Get current date as string."""
    return datetime.now().strftime("%Y-%m-%d")


def get_static_value() -> str:
    """Get a static test value."""
    return "test_value"


def get_counter() -> int:
    """Get an incrementing counter value."""
    if not hasattr(get_counter, "count"):
        get_counter.count = 0
    get_counter.count += 1
    return get_counter.count


async def async_get_timestamp() -> str:
    """Async function to get timestamp."""
    await asyncio.sleep(0.01)
    return str(int(time.time()))


async def async_get_random() -> int:
    """Async function to get a pseudo-random number."""
    await asyncio.sleep(0.01)
    return int(time.time() % 1000)


def get_context_info(**kwargs) -> str:
    """Function with optional parameters (should be rejected)."""
    return f"Context: {kwargs.get('type', 'default')}"


def get_required_param(value: str) -> str:
    """Function with required parameter (should be rejected)."""
    return f"Value: {value}"


# ==============================================================================
# Test Classes
# ==============================================================================


class TestDefaultParamsBasic:
    """Test basic default params functionality."""

    def test_tool_without_default_params(self):
        """Test tool creation without default params."""

        def simple_handler(x: str) -> str:
            return f"result:{x}"

        tool = Tool(name="simple_tool", handler=simple_handler)
        assert tool.default_params == {}
        assert len(tool._default_fixed_params) == 0
        assert len(tool._default_runtime_params) == 0

    def test_tool_with_static_default_params(self):
        """Test tool with only static default params."""

        def handler(message: str, prefix: str = "default") -> str:
            return f"{prefix}:{message}"

        tool = Tool(name="static_tool", handler=handler, default_params={"prefix": "static_value"})

        assert tool.default_params == {"prefix": "static_value"}
        assert tool._default_fixed_params == {"prefix": "static_value"}
        assert len(tool._default_runtime_params) == 0

    def test_tool_with_callable_default_params(self):
        """Test tool with only callable default params."""

        def handler(message: str, time_str: str) -> str:
            return f"{message} at {time_str}"

        tool = Tool(name="callable_tool", handler=handler, default_params={"time_str": get_current_time})

        assert len(tool._default_fixed_params) == 0
        assert len(tool._default_runtime_params) == 1
        assert "time_str" in tool._default_runtime_params
        assert tool._default_runtime_params["time_str"]["callable"] is get_current_time
        assert isinstance(tool._default_runtime_params["time_str"]["is_coroutine"], bool)

    def test_tool_with_mixed_default_params(self):
        """Test tool with both static and callable default params."""

        def handler(message: str, time_str: str, static_val: str) -> str:
            return f"{message} at {time_str} ({static_val})"

        tool = Tool(
            name="mixed_tool",
            handler=handler,
            default_params={
                "time_str": get_current_time,  # Callable
                "static_val": "fixed_value",  # Static
            },
        )

        assert tool._default_fixed_params == {"static_val": "fixed_value"}
        assert len(tool._default_runtime_params) == 1
        assert "time_str" in tool._default_runtime_params


class TestDefaultParamsValidation:
    """Test validation of callable default params."""

    def test_valid_parameterless_function(self):
        """Test that parameterless functions are accepted."""

        def handler(x: str, time_str: str) -> str:
            return f"{x}:{time_str}"

        tool = Tool(name="valid_tool", handler=handler, default_params={"time_str": get_current_time})

        callable_info = tool._default_runtime_params["time_str"]
        assert callable_info["callable"] is get_current_time
        assert callable_info["is_coroutine"] is False

    def test_valid_async_parameterless_function(self):
        """Test that async parameterless functions are accepted."""

        def handler(x: str, timestamp: str) -> str:
            return f"{x}:{timestamp}"

        tool = Tool(name="async_tool", handler=handler, default_params={"timestamp": async_get_timestamp})

        callable_info = tool._default_runtime_params["timestamp"]
        assert callable_info["callable"] is async_get_timestamp
        assert callable_info["is_coroutine"] is True

    def test_function_with_optional_params(self):
        """Test that functions with optional parameters are accepted."""

        def handler(x: str, context: str) -> str:
            return f"{x}:{context}"

        # This should work since get_context_info has no required params
        tool = Tool(name="optional_tool", handler=handler, default_params={"context": get_context_info})

        assert "context" in tool._default_runtime_params

    def test_invalid_function_with_required_params(self):
        """Test that functions with required parameters are rejected."""

        def handler(x: str, value: str) -> str:
            return f"{x}:{value}"

        with pytest.raises(ValueError, match="Callable must not have any required parameters"):
            Tool(name="invalid_tool", handler=handler, default_params={"value": get_required_param})

    def test_invalid_non_callable(self):
        """Test that non-callable values in default_params are treated as static."""

        def handler(x: str, value: str) -> str:
            return f"{x}:{value}"

        tool = Tool(name="non_callable_tool", handler=handler, default_params={"value": "not_a_function"})

        # Should be treated as static
        assert tool._default_fixed_params == {"value": "not_a_function"}
        assert len(tool._default_runtime_params) == 0


class TestDefaultParamsExecution:
    """Test execution of tools with default params."""

    @pytest.mark.asyncio
    async def test_static_default_params_execution(self):
        """Test execution with static default params."""

        def handler(message: str, prefix: str) -> str:
            return f"{prefix}:{message}"

        tool = Tool(name="static_exec_tool", handler=handler, default_params={"prefix": "DEFAULT"})

        result = tool(message="test")
        output = await result.collect()

        assert_has_output_event(output)
        assert_no_errors(output)
        assert output.output == "DEFAULT:test"

    @pytest.mark.asyncio
    async def test_callable_default_params_execution(self):
        """Test execution with callable default params."""

        def handler(message: str, counter: int) -> str:
            return f"{message}:{counter}"

        tool = Tool(name="callable_exec_tool", handler=handler, default_params={"counter": get_counter})

        # Execute multiple times to verify callable is executed each time
        result1 = tool(message="first")
        output1 = await result1.collect()

        result2 = tool(message="second")
        output2 = await result2.collect()

        assert_has_output_event(output1)
        assert_has_output_event(output2)
        assert_no_errors(output1)
        assert_no_errors(output2)

        # Counter should increment each time
        assert "first:1" in str(output1.output)
        assert "second:2" in str(output2.output)

    @pytest.mark.asyncio
    async def test_async_callable_default_params_execution(self):
        """Test execution with async callable default params."""

        def handler(message: str, timestamp: str) -> str:
            return f"{message}@{timestamp}"

        tool = Tool(name="async_exec_tool", handler=handler, default_params={"timestamp": async_get_timestamp})

        result = tool(message="test")
        output = await result.collect()

        assert_has_output_event(output)
        assert_no_errors(output)
        assert "test@" in str(output.output)
        # Should contain a timestamp (numeric string)
        timestamp_part = str(output.output).split("@")[1]
        assert timestamp_part.isdigit()

    @pytest.mark.asyncio
    async def test_mixed_default_params_execution(self):
        """Test execution with both static and callable default params."""

        def handler(message: str, time_str: str, prefix: str) -> str:
            return f"{prefix}:{message}@{time_str}"

        tool = Tool(
            name="mixed_exec_tool",
            handler=handler,
            default_params={
                "time_str": get_current_time,  # Callable
                "prefix": "LOG",  # Static
            },
        )

        result = tool(message="event")
        output = await result.collect()

        assert_has_output_event(output)
        assert_no_errors(output)
        result_str = str(output.output)
        assert "LOG:event@" in result_str
        # Should contain time format
        assert ":" in result_str.split("@")[1]  # Time contains colons

    @pytest.mark.asyncio
    async def test_override_default_params(self):
        """Test that runtime kwargs override default params."""

        def handler(message: str, prefix: str) -> str:
            return f"{prefix}:{message}"

        tool = Tool(name="override_tool", handler=handler, default_params={"prefix": "DEFAULT"})

        # Should use default
        result1 = tool(message="test1")
        output1 = await result1.collect()

        # Should override default
        result2 = tool(message="test2", prefix="OVERRIDE")
        output2 = await result2.collect()

        assert_has_output_event(output1)
        assert_has_output_event(output2)
        assert_no_errors(output1)
        assert_no_errors(output2)

        assert output1.output == "DEFAULT:test1"
        assert output2.output == "OVERRIDE:test2"


class TestDefaultParamsPerformance:
    """Test performance characteristics of default params resolution."""

    @pytest.mark.asyncio
    async def test_resolution_performance(self):
        """Test that default params resolution is reasonably fast."""

        def handler(message: str, time1: str, time2: str, static: str) -> str:
            return f"{message}:{time1}:{time2}:{static}"

        tool = Tool(
            name="perf_tool",
            handler=handler,
            default_params={"time1": get_current_time, "time2": get_current_date, "static": "fixed"},
        )

        start_time = time.time()
        resolved_params = await tool._resolve_input_params()
        end_time = time.time()

        # Should resolve quickly (under 1 second for simple functions)
        assert end_time - start_time < 1.0
        assert "time1" in resolved_params
        assert "time2" in resolved_params
        assert resolved_params["static"] == "fixed"

    @pytest.mark.asyncio
    async def test_concurrent_resolution(self):
        """Test that multiple tools can resolve default params concurrently."""

        def handler(message: str, time_str: str) -> str:
            return f"{message}:{time_str}"

        tools = [
            Tool(name=f"concurrent_tool_{i}", handler=handler, default_params={"time_str": get_current_time})
            for i in range(3)
        ]

        start_time = time.time()

        # Resolve all params concurrently
        tasks = [tool._resolve_input_params() for tool in tools]
        results = await asyncio.gather(*tasks)

        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 2.0
        assert len(results) == 3
        assert all("time_str" in result for result in results)

    @pytest.mark.asyncio
    async def test_parallel_callable_execution(self):
        """Test that multiple callable params are executed in parallel."""

        def slow_func1() -> str:
            time.sleep(0.1)  # Simulate work
            return "func1"

        def slow_func2() -> str:
            time.sleep(0.1)  # Simulate work
            return "func2"

        def handler(message: str, val1: str, val2: str) -> str:
            return f"{message}:{val1}:{val2}"

        tool = Tool(name="parallel_tool", handler=handler, default_params={"val1": slow_func1, "val2": slow_func2})

        start_time = time.time()
        resolved_params = await tool._resolve_input_params()
        end_time = time.time()

        # Should be faster than sequential execution (less than 0.15s vs 0.2s)
        assert end_time - start_time < 0.15
        assert resolved_params["val1"] == "func1"
        assert resolved_params["val2"] == "func2"


class TestDefaultParamsEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_default_params(self):
        """Test tools with empty default_params."""

        def handler(x: str) -> str:
            return f"result:{x}"

        tool = Tool(name="empty_tool", handler=handler, default_params={})

        resolved_params = await tool._resolve_input_params()
        assert resolved_params == {}

    @pytest.mark.asyncio
    async def test_callable_returning_none(self):
        """Test handling of callables that return None."""

        def return_none() -> None:
            return None

        def handler(message: str, value: str) -> str:
            return f"{message}:{value}"

        tool = Tool(name="none_tool", handler=handler, default_params={"value": return_none})

        resolved_params = await tool._resolve_input_params()
        assert resolved_params["value"] is None

    @pytest.mark.asyncio
    async def test_mixed_none_and_valid_results(self):
        """Test mix of None-returning and valid callables."""

        def return_none() -> None:
            return None

        def return_value() -> str:
            return "valid"

        def handler(message: str, none_val: str, valid_val: str) -> str:
            return f"{message}:{none_val}:{valid_val}"

        tool = Tool(
            name="mixed_none_tool", handler=handler, default_params={"none_val": return_none, "valid_val": return_value}
        )

        resolved_params = await tool._resolve_input_params()
        assert resolved_params["none_val"] is None
        assert resolved_params["valid_val"] == "valid"

    @pytest.mark.asyncio
    async def test_callable_exception_handling(self):
        """Test handling of exceptions in callable default params."""

        def failing_callable() -> str:
            raise ValueError("Test error")

        def handler(message: str, value: str) -> str:
            return f"{message}:{value}"

        tool = Tool(name="failing_tool", handler=handler, default_params={"value": failing_callable})

        # The exception should propagate when resolving default params
        with pytest.raises(ValueError, match="Test error"):
            await tool._resolve_input_params()


class TestDefaultParamsWithAgents:
    """Test default params functionality with Agent classes."""

    @pytest.mark.asyncio
    async def test_agent_with_static_default_params(self):
        """Test agent with static default params."""
        agent = Agent(name="static_agent", model="openai/gpt-4o-mini", default_params={"context": "test_context"})

        assert agent._default_fixed_params == {"context": "test_context"}
        assert len(agent._default_runtime_params) == 0

        # Test execution
        prompt = Message.validate({"role": "user", "content": "Hello"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "agent_with_static_default_params")

    @pytest.mark.asyncio
    async def test_agent_with_callable_default_params(self):
        """Test agent with callable default params."""
        agent = Agent(
            name="callable_agent",
            model="openai/gpt-4o-mini",
            default_params={"current_time": get_current_time, "session_id": lambda: "session_123"},
        )

        assert len(agent._default_fixed_params) == 0
        assert len(agent._default_runtime_params) == 2
        assert "current_time" in agent._default_runtime_params
        assert "session_id" in agent._default_runtime_params

        # Test resolution
        resolved_params = await agent._resolve_input_params()
        assert "current_time" in resolved_params
        assert "session_id" in resolved_params
        assert resolved_params["session_id"] == "session_123"
        # Time should be in HH:MM:SS format
        assert ":" in resolved_params["current_time"]

    @pytest.mark.asyncio
    async def test_agent_with_mixed_default_params(self):
        """Test agent with both static and callable default params."""
        agent = Agent(
            name="mixed_agent",
            model="openai/gpt-4o-mini",
            default_params={
                "static_context": "production",  # Static
                "timestamp": get_current_time,  # Callable
                "counter": get_counter,  # Callable with state
            },
        )

        assert agent._default_fixed_params == {"static_context": "production"}
        assert len(agent._default_runtime_params) == 2

        # Test multiple resolutions to verify callables are executed each time
        resolved1 = await agent._resolve_input_params()
        resolved2 = await agent._resolve_input_params()

        # Static value should be same
        assert resolved1["static_context"] == resolved2["static_context"] == "production"
        # Counter should increment
        assert resolved2["counter"] > resolved1["counter"]

    @pytest.mark.asyncio
    async def test_agent_default_params_with_system_prompt_templating(self):
        """Test that default params work alongside system prompt templating."""
        agent = Agent(
            name="template_agent",
            model="openai/gpt-4o-mini",
            system_prompt="Current directory: {os::getcwd}. Process ID: {os::getpid}",
            default_params={
                "context": lambda: "AI_ASSISTANT_SESSION",
                "user_id": "user_123",  # Static
            },
        )

        # Should have both system prompt callables and default param callables
        assert len(agent._system_prompt_templates) == 2  # os::getcwd and os::getpid
        assert len(agent._default_runtime_params) == 1  # context callable
        assert agent._default_fixed_params == {"user_id": "user_123"}

        # Test that both are resolved independently
        resolved_system = await agent._resolve_system_prompt()
        resolved_defaults = await agent._resolve_input_params()

        # System prompt should have template functions resolved
        assert "{os::getcwd}" not in resolved_system
        assert "{os::getpid}" not in resolved_system
        assert "Current directory:" in resolved_system
        assert "Process ID:" in resolved_system

        # Default params should be resolved
        assert resolved_defaults["context"] == "AI_ASSISTANT_SESSION"
        assert resolved_defaults["user_id"] == "user_123"


class TestDefaultParamsWithNestedTools:
    """Test default params functionality with tools nested inside agents."""

    @pytest.mark.asyncio
    async def test_nested_tools_with_default_params(self):
        """Test that tools with default params work when nested in agents."""

        # Create tools with default params
        def time_logger(message: str, timestamp: str, prefix: str) -> str:
            return f"[{prefix}] {timestamp}: {message}"

        def counter_tool(action: str, current_count: int) -> str:
            if action == "increment":
                return f"Count incremented to {current_count + 1}"
            return f"Current count: {current_count}"

        time_tool = Tool(
            name="time_logger",
            handler=time_logger,
            default_params={
                "timestamp": get_current_time,  # Callable
                "prefix": "LOG",  # Static
            },
        )

        counter_tool_instance = Tool(
            name="counter",
            handler=counter_tool,
            default_params={
                "current_count": get_counter  # Callable with state
            },
        )

        # Create agent with these tools
        agent = Agent(name="nested_tools_agent", model="openai/gpt-4o-mini", tools=[time_tool, counter_tool_instance])

        # Verify tools maintain their default params after nesting
        assert len(time_tool._default_runtime_params) == 1
        assert time_tool._default_fixed_params == {"prefix": "LOG"}
        assert len(counter_tool_instance._default_runtime_params) == 1

        # Test tool execution within agent context
        prompt = Message.validate(
            {"role": "user", "content": "Please log the message 'System started' and then increment the counter"}
        )
        result = agent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "nested_tools_with_default_params")

    @pytest.mark.asyncio
    async def test_agent_and_nested_tools_both_have_default_params(self):
        """Test scenario where both agent and its tools have default params."""

        def user_action(action: str, user_context: str, session_id: str) -> str:
            return f"User {user_context} performed {action} in session {session_id}"

        # Tool with default params
        action_tool = Tool(
            name="user_action",
            handler=user_action,
            default_params={
                "user_context": lambda: "anonymous_user",  # Callable
                "action": "unknown",  # Static (will be overridden by user)
            },
        )

        # Agent with default params
        agent = Agent(
            name="user_agent",
            model="openai/gpt-4o-mini",
            tools=[action_tool],
            default_params={
                "session_id": lambda: f"session_{int(time.time())}",  # Callable
                "environment": "production",  # Static
            },
        )

        # Test that both agent and tool default params are properly initialized
        assert len(agent._default_runtime_params) == 1
        assert agent._default_fixed_params == {"environment": "production"}
        assert len(action_tool._default_runtime_params) == 1
        assert action_tool._default_fixed_params == {"action": "unknown"}

        # Test resolution works for both
        agent_resolved = await agent._resolve_input_params()
        tool_resolved = await action_tool._resolve_input_params()

        assert "session_" in agent_resolved["session_id"]
        assert agent_resolved["environment"] == "production"
        assert tool_resolved["user_context"] == "anonymous_user"
        assert tool_resolved["action"] == "unknown"

    @pytest.mark.asyncio
    async def test_nested_agent_default_params(self):
        """Test default params with nested agents (subagents)."""

        # Create a subagent with default params
        def simple_task(task: str, priority: str) -> str:
            return f"Task '{task}' with priority {priority} completed"

        subagent = Agent(
            name="task_subagent",
            model="openai/gpt-4o-mini",
            tools=[simple_task],
            default_params={
                "priority": "medium"  # Static default
            },
        )

        # Create parent agent with the subagent as a tool
        parent_agent = Agent(
            name="parent_agent",
            model="openai/gpt-4o-mini",
            tools=[subagent],
            default_params={
                "context": lambda: "parent_context",  # Callable
                "session": "main_session",  # Static
            },
        )

        # Verify both agents have their default params
        assert parent_agent._default_fixed_params == {"session": "main_session"}
        assert len(parent_agent._default_runtime_params) == 1
        assert subagent._default_fixed_params == {"priority": "medium"}
        assert len(subagent._default_runtime_params) == 0

        # Test execution (may be complex due to nested nature)
        prompt = Message.validate({"role": "user", "content": "Please complete a simple task using the subagent"})
        result = parent_agent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "nested_agent_default_params")


class TestDefaultParamsIntegration:
    """Test integration scenarios with default params across the framework."""

    @pytest.mark.asyncio
    async def test_default_params_with_concurrent_tool_execution(self):
        """Test default params work correctly during concurrent tool execution."""

        def slow_task(task_name: str, delay: float, context: str) -> str:
            time.sleep(delay)
            return f"Task {task_name} completed in context {context}"

        # Create multiple tools with callable default params
        tools = []
        for i in range(3):
            tool = Tool(
                name=f"slow_task_{i}",
                handler=slow_task,
                default_params={
                    "context": lambda i=i: f"context_{i}_{int(time.time())}",
                    "delay": 0.01,  # Static short delay for testing
                },
            )
            tools.append(tool)

        agent = Agent(name="concurrent_agent", model="openai/gpt-4o-mini", tools=tools)

        # Test that each tool gets its own resolved default params
        resolved_params = []
        for tool in tools:
            params = await tool._resolve_input_params()
            resolved_params.append(params)

        # Each tool should have unique context values
        contexts = [params["context"] for params in resolved_params]
        assert len(set(contexts)) == 3  # All should be unique

        # All should have the same static delay
        delays = [params["delay"] for params in resolved_params]
        assert all(delay == 0.01 for delay in delays)

    @pytest.mark.asyncio
    async def test_default_params_error_handling_in_agent_context(self):
        """Test error handling when default param callables fail in agent context."""

        def failing_default() -> str:
            raise RuntimeError("Default param failed")

        def simple_handler(message: str, context: str) -> str:
            return f"{message} in {context}"

        tool_with_failing_default = Tool(
            name="failing_tool", handler=simple_handler, default_params={"context": failing_default}
        )

        agent = Agent(name="error_agent", model="openai/gpt-4o-mini", tools=[tool_with_failing_default])

        # Test that the tool's default param resolution fails as expected
        with pytest.raises(RuntimeError, match="Default param failed"):
            await tool_with_failing_default._resolve_input_params()

        # This confirms the error handling behavior without needing LLM calls

    @pytest.mark.asyncio
    async def test_default_params_with_streaming_responses(self):
        """Test that default params work correctly with streaming tool responses."""

        def streaming_task(message: str, chunk_size: int, prefix: str) -> str:
            # Simulate a task that could stream but returns final result
            chunks = [message[i : i + chunk_size] for i in range(0, len(message), chunk_size)]
            return f"{prefix}: " + " | ".join(chunks)

        streaming_tool = Tool(
            name="streaming_task",
            handler=streaming_task,
            default_params={
                "chunk_size": 5,  # Static
                "prefix": lambda: f"STREAM_{int(time.time())}",  # Callable
            },
        )

        # Test the tool directly without involving LLM calls
        resolved_params = await streaming_tool._resolve_input_params()
        assert resolved_params["chunk_size"] == 5
        assert "STREAM_" in resolved_params["prefix"]

        # Test tool execution with default params
        result = streaming_tool(message="Hello World This Is A Test")
        output = await result.collect()

        assert_has_output_event(output)
        assert_no_errors(output)
        assert "STREAM_" in str(output.output)
        assert "Hello |" in str(output.output)  # First chunk should be "Hello"
        assert "|" in str(output.output)  # Should have chunk separators
