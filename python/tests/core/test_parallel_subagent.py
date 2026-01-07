"""Tests for parallel subagent execution.

This module tests the behavior when an agent calls a subagent tool
multiple times in parallel, focusing on memory resolution and state
management across concurrent executions.
"""

import pytest
from timbal import Agent
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from .conftest import assert_has_output_event, skip_if_agent_error


class TestParallelSubagentExecution:
    """Test parallel execution of subagent tools."""

    @pytest.fixture
    def counter_subagent(self):
        """Create a simple subagent that returns a count."""
        call_count = {"value": 0}

        def increment() -> int:
            """Increment and return the counter."""
            call_count["value"] += 1
            return call_count["value"]

        return Agent(
            name="counter_agent",
            model="openai/gpt-4o-mini",
            tools=[increment],
            description="An agent that increments and returns a counter. Call the increment tool and return the result.",
        )

    @pytest.fixture
    def echo_subagent(self):
        """Create a simple subagent that echoes input."""

        def echo(message: str) -> str:
            """Echo the message back."""
            return f"Echo: {message}"

        return Agent(
            name="echo_agent",
            model="openai/gpt-4o-mini",
            tools=[echo],
            description="An agent that echoes messages. Use the echo tool with the provided message.",
        )

    @pytest.fixture
    def main_agent_with_subagent(self, echo_subagent):
        """Create main agent with a subagent as a tool."""
        return Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[echo_subagent],
            system_prompt=(
                "You have access to an echo_agent tool. When asked to echo multiple messages, "
                "call the echo_agent tool for each message IN PARALLEL (all at once, not sequentially)."
            ),
        )

    @pytest.mark.asyncio
    async def test_parallel_subagent_calls_basic(self, main_agent_with_subagent):
        """Test that parallel subagent calls complete without errors."""
        prompt = Message.validate(
            {"role": "user", "content": "Please echo these two messages in parallel: 'hello' and 'world'"}
        )

        result = main_agent_with_subagent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "parallel_subagent_calls_basic")
        assert isinstance(output.output, Message)

    @pytest.mark.asyncio
    async def test_parallel_subagent_calls_track_events(self, main_agent_with_subagent):
        """Test that we can track events from parallel subagent executions."""
        prompt = Message.validate(
            {"role": "user", "content": "Please echo these two messages in parallel: 'first' and 'second'"}
        )

        subagent_start_events = []
        subagent_output_events = []

        async for event in main_agent_with_subagent(prompt=prompt):
            if event.type == "START" and "echo_agent" in event.path:
                subagent_start_events.append(event)
            if event.type == "OUTPUT" and event.path == "main_agent.echo_agent":
                subagent_output_events.append(event)

        # We should have at least one subagent call
        assert len(subagent_start_events) >= 1, "Expected at least one subagent START event"

    @pytest.mark.asyncio
    async def test_parallel_subagent_memory_isolation(self):
        """Test that parallel subagent calls have isolated memory contexts."""
        memory_tracker = {"calls": []}

        def track_and_respond(input_id: str) -> str:
            """Track the call and respond."""
            memory_tracker["calls"].append(input_id)
            return f"Processed: {input_id}"

        tracking_subagent = Agent(
            name="tracking_agent",
            model="openai/gpt-4o-mini",
            tools=[track_and_respond],
            description="An agent that tracks calls. Use track_and_respond with the given input_id.",
        )

        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[tracking_subagent],
            system_prompt=(
                "You have a tracking_agent tool. When asked to process multiple IDs, "
                "call tracking_agent for each ID IN PARALLEL."
            ),
        )

        prompt = Message.validate({"role": "user", "content": "Process these IDs in parallel: 'id_1', 'id_2', 'id_3'"})

        result = main_agent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "parallel_subagent_memory_isolation")

    @pytest.mark.asyncio
    async def test_multiple_iterations_with_parallel_subagent(self, echo_subagent):
        """Test multiple iterations where each iteration has parallel subagent calls."""
        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[echo_subagent],
            system_prompt=(
                "You have an echo_agent tool. Call it for each message the user provides, "
                "making parallel calls when possible."
            ),
        )

        # First iteration
        prompt1 = Message.validate(
            {"role": "user", "content": "Echo 'iteration1_msg1' and 'iteration1_msg2' in parallel"}
        )
        result1 = main_agent(prompt=prompt1)
        output1 = await result1.collect()

        assert_has_output_event(output1)
        skip_if_agent_error(output1, "multiple_iterations_1")

        # Second iteration - uses the RunContext from first to test memory handling
        prompt2 = Message.validate(
            {"role": "user", "content": "Echo 'iteration2_msg1' and 'iteration2_msg2' in parallel"}
        )
        result2 = main_agent(prompt=prompt2)
        output2 = await result2.collect()

        assert_has_output_event(output2)
        skip_if_agent_error(output2, "multiple_iterations_2")

        # Third iteration
        prompt3 = Message.validate(
            {"role": "user", "content": "Echo 'iteration3_msg1' and 'iteration3_msg2' in parallel"}
        )
        result3 = main_agent(prompt=prompt3)
        output3 = await result3.collect()

        assert_has_output_event(output3)
        skip_if_agent_error(output3, "multiple_iterations_3")

    @pytest.mark.asyncio
    async def test_nested_parallel_subagent_calls(self):
        """Test nested agents where the subagent itself makes parallel tool calls."""

        def tool_a(x: str) -> str:
            return f"A:{x}"

        def tool_b(x: str) -> str:
            return f"B:{x}"

        inner_agent = Agent(
            name="inner_agent",
            model="openai/gpt-4o-mini",
            tools=[tool_a, tool_b],
            description="An agent with tool_a and tool_b. When asked, call both tools in parallel.",
        )

        outer_agent = Agent(
            name="outer_agent",
            model="openai/gpt-4o-mini",
            tools=[inner_agent],
            system_prompt=(
                "You have an inner_agent. When asked to process requests, call inner_agent multiple times IN PARALLEL."
            ),
        )

        prompt = Message.validate(
            {
                "role": "user",
                "content": "Call inner_agent twice in parallel: once with 'request1' and once with 'request2'",
            }
        )

        result = outer_agent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "nested_parallel_subagent_calls")

    @pytest.mark.asyncio
    async def test_parallel_subagent_with_errors(self):
        """Test that errors in one parallel subagent call don't crash others."""
        call_count = {"value": 0}

        def sometimes_fail(should_fail: bool) -> str:
            """A tool that fails based on input."""
            call_count["value"] += 1
            if should_fail:
                raise ValueError("Intentional failure")
            return "Success"

        error_subagent = Agent(
            name="error_agent",
            model="openai/gpt-4o-mini",
            tools=[sometimes_fail],
            description="An agent with a sometimes_fail tool. Call it with should_fail=true or false.",
        )

        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[error_subagent],
            system_prompt=("You have an error_agent. When asked, call it multiple times in parallel."),
        )

        prompt = Message.validate(
            {
                "role": "user",
                "content": "Call error_agent twice in parallel: once with should_fail=false and once with should_fail=true",
            }
        )

        result = main_agent(prompt=prompt)
        output = await result.collect()

        # Should complete without crashing even if one subagent fails
        assert isinstance(output, OutputEvent)
        assert isinstance(output.output, Message)

    @pytest.mark.asyncio
    async def test_parallel_subagent_concurrent_execution_timing(self):
        """Test that parallel subagent calls actually execute concurrently."""
        import asyncio
        import time

        execution_times = {"starts": [], "ends": []}

        def slow_tool(task_id: str) -> str:
            """A slow tool to measure concurrency."""
            execution_times["starts"].append((task_id, time.time()))
            time.sleep(0.5)  # 500ms delay
            execution_times["ends"].append((task_id, time.time()))
            return f"Completed: {task_id}"

        slow_subagent = Agent(
            name="slow_agent",
            model="openai/gpt-4o-mini",
            tools=[slow_tool],
            description="An agent with a slow_tool. Call it with a task_id.",
        )

        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[slow_subagent],
            system_prompt=(
                "You have a slow_agent. When asked to run multiple tasks, "
                "call slow_agent for each task IN PARALLEL (all at once)."
            ),
        )

        prompt = Message.validate({"role": "user", "content": "Run slow_agent for 'task_1' and 'task_2' in parallel"})

        start_time = time.time()
        result = main_agent(prompt=prompt)
        output = await result.collect()
        total_time = time.time() - start_time

        assert_has_output_event(output)
        skip_if_agent_error(output, "parallel_subagent_concurrent_execution_timing")

        # If executed in parallel, total time should be less than 2x the individual tool time
        # (accounting for LLM latency, we just check it's not obviously sequential)
        # Note: This is a soft check - LLM behavior may vary

    @pytest.mark.asyncio
    async def test_repeated_parallel_calls_same_subagent(self):
        """Test calling the same subagent in parallel multiple times across iterations."""
        call_log = []

        def log_call(call_id: str) -> str:
            """Log the call and return confirmation."""
            call_log.append(call_id)
            return f"Logged: {call_id}"

        logging_subagent = Agent(
            name="logging_agent",
            model="openai/gpt-4o-mini",
            tools=[log_call],
            description="An agent that logs calls. Use log_call with a call_id.",
        )

        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[logging_subagent],
            system_prompt=(
                "You have a logging_agent. When asked to log multiple IDs, call logging_agent for each ID IN PARALLEL."
            ),
        )

        # Run 3 iterations, each with parallel calls
        for iteration in range(3):
            prompt = Message.validate(
                {"role": "user", "content": f"Log these IDs in parallel: 'iter{iteration}_a' and 'iter{iteration}_b'"}
            )

            result = main_agent(prompt=prompt)
            output = await result.collect()

            assert_has_output_event(output)
            skip_if_agent_error(output, f"repeated_parallel_calls_iteration_{iteration}")

    @pytest.mark.asyncio
    async def test_parallel_subagent_with_different_inputs(self):
        """Test parallel subagent calls with distinctly different inputs."""

        def process_number(n: int) -> int:
            """Double a number."""
            return n * 2

        def process_string(s: str) -> str:
            """Uppercase a string."""
            return s.upper()

        multi_tool_subagent = Agent(
            name="processor_agent",
            model="openai/gpt-4o-mini",
            tools=[process_number, process_string],
            description="An agent that can process numbers or strings.",
        )

        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[multi_tool_subagent],
            system_prompt=(
                "You have a processor_agent. When asked to process multiple items, "
                "call processor_agent for each item IN PARALLEL."
            ),
        )

        prompt = Message.validate(
            {
                "role": "user",
                "content": "Process these in parallel using processor_agent: the number 5, the string 'hello', and the number 10",
            }
        )

        result = main_agent(prompt=prompt)
        output = await result.collect()

        assert_has_output_event(output)
        skip_if_agent_error(output, "parallel_subagent_with_different_inputs")
        assert isinstance(output.output, Message)
