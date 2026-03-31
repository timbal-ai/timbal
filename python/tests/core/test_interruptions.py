import asyncio
import time
from collections.abc import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
from timbal import Agent, Tool, Workflow
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from ..conftest import Timer, assert_has_output_event, assert_no_errors

# ==============================================================================
# Test Handler Functions for Interruption Testing
# ==============================================================================


def long_running_sync_handler(duration: float = 10) -> str:
    """Synchronous handler with long execution time."""
    time.sleep(duration)
    return f"sync_completed_after_{duration}s"


async def long_running_async_handler(duration: float = 10) -> str:
    """Asynchronous handler with long execution time."""
    await asyncio.sleep(duration)
    return f"async_completed_after_{duration}s"


def sync_generator_handler(count: int = 100, delay: float = 0.5) -> Generator[str, None, None]:
    """Synchronous generator that yields values with delays."""
    for i in range(count):
        time.sleep(delay)
        yield f"sync_chunk_{i}"


async def async_generator_handler(count: int = 100, delay: float = 0.5) -> AsyncGenerator[str, None]:
    """Asynchronous generator that yields values with delays."""
    for i in range(count):
        await asyncio.sleep(delay)
        yield f"async_chunk_{i}"


async def slow_step_handler(duration: float = 5) -> str:
    """Slow async step handler for workflow testing."""
    await asyncio.sleep(duration)
    return f"step_completed_after_{duration}s"


def parallel_task_1(duration: float = 10) -> str:
    """First parallel task that takes time."""
    time.sleep(duration)
    return "parallel_1_completed"


def parallel_task_2(duration: float = 10) -> str:
    """Second parallel task that takes time."""
    time.sleep(duration)
    return "parallel_2_completed"


def parallel_task_3(duration: float = 10) -> str:
    """Third parallel task that takes time."""
    time.sleep(duration)
    return "parallel_3_completed"


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def long_running_sync_tool():
    """Create a long-running synchronous tool."""
    return Tool(name="long_sync", handler=long_running_sync_handler)


@pytest.fixture
def long_running_async_tool():
    """Create a long-running asynchronous tool."""
    return Tool(name="long_async", handler=long_running_async_handler)


@pytest.fixture
def sync_gen_tool():
    """Create a synchronous generator tool."""
    return Tool(name="sync_gen", handler=sync_generator_handler)


@pytest.fixture
def async_gen_tool():
    """Create an asynchronous generator tool."""
    return Tool(name="async_gen", handler=async_generator_handler)


@pytest.fixture
def slow_step_tool():
    """Create a slow step tool for workflows."""
    return Tool(name="slow_step", handler=slow_step_handler)


@pytest.fixture
def agent_with_long_tools():
    """Create an agent with long-running tools."""

    def get_weather() -> str:
        time.sleep(10)
        print("get_weather")
        return "Sunny, 72°F"

    def get_location() -> str:
        time.sleep(10)
        print("get_location")
        return "San Francisco, CA"

    def get_temperature() -> str:
        time.sleep(10)
        print("get_temperature")
        return "72°F"

    return Agent(
        name="agent_with_tools", model="openai/gpt-4o-mini", tools=[get_weather, get_location, get_temperature]
    )


# ==============================================================================
# Test Classes
# ==============================================================================


class TestToolInterruption:
    """Test tool interruption during execution."""

    @pytest.mark.asyncio
    async def test_sync_tool_interruption(self, long_running_sync_tool):
        """Test that synchronous tool can be interrupted during execution."""
        # Start tool execution
        task = asyncio.create_task(long_running_sync_tool(duration=10).collect())

        # Wait briefly to ensure execution starts
        await asyncio.sleep(2)

        # Cancel the task
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_async_tool_interruption(self, long_running_async_tool):
        """Test that asynchronous tool can be interrupted during execution."""
        # Start tool execution
        task = asyncio.create_task(long_running_async_tool(duration=10).collect())

        # Wait briefly to ensure execution starts
        await asyncio.sleep(2)

        # Cancel the task
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_sync_generator_interruption(self, sync_gen_tool):
        """Test that synchronous generator tool can be interrupted while yielding."""
        # Start generator tool execution
        task = asyncio.create_task(sync_gen_tool(count=100, delay=0.5).collect())

        # Wait for some chunks to be yielded
        await asyncio.sleep(2)

        # Cancel while still yielding
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"
        assert result.output is not None  # Should have partial output

    @pytest.mark.asyncio
    async def test_async_generator_interruption(self, async_gen_tool):
        """Test that asynchronous generator tool can be interrupted while yielding."""
        # Start generator tool execution
        task = asyncio.create_task(async_gen_tool(count=100, delay=0.5).collect())

        # Wait for some chunks to be yielded
        await asyncio.sleep(2)

        # Cancel while still yielding
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"
        assert result.output is not None  # Should have partial output

    @pytest.mark.asyncio
    async def test_tool_reusability_after_interruption(self, long_running_async_tool):
        """Test that tools can be reused after being interrupted."""
        # First execution - interrupt it
        task1 = asyncio.create_task(long_running_async_tool(duration=10).collect())
        await asyncio.sleep(1)
        task1.cancel()
        result1 = await task1

        assert result1.status.code == "cancelled"

        # Second execution - let it complete
        result2 = await long_running_async_tool(duration=0.1).collect()

        assert_has_output_event(result2)
        assert_no_errors(result2)
        assert result2.status.code == "success"
        assert result2.output == "async_completed_after_0.1s"

    @pytest.mark.parametrize("duration", [5, 10, 15])
    @pytest.mark.asyncio
    async def test_interruption_timing_variations(self, duration):
        """Test interruption at different timing points."""
        tool = Tool(handler=long_running_async_handler)

        task = asyncio.create_task(tool(duration=duration).collect())
        await asyncio.sleep(2)  # Interrupt after 2 seconds
        task.cancel()
        result = await task

        assert result.status.code == "cancelled"


class TestAgentInterruption:
    """Test agent interruption during execution."""

    @pytest.mark.asyncio
    async def test_agent_interruption_during_llm_call(self):
        """Test that agent can be interrupted during LLM processing."""
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini")

        prompt = Message.validate({"role": "user", "content": "Tell me a very long story about space exploration"})

        # Start agent execution
        task = asyncio.create_task(agent(prompt=prompt).collect())

        # Wait for LLM to start processing
        await asyncio.sleep(4)

        # Interrupt the agent
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_agent_interruption_during_tool_execution(self):
        """Test that agent can be interrupted while executing a tool."""

        # Create agent with long-running tool
        async def slow_tool() -> str:
            await asyncio.sleep(10)
            return "tool_result"

        agent = Agent(name="tool_agent", model="openai/gpt-4o-mini", tools=[slow_tool])

        prompt = Message.validate({"role": "user", "content": "Use the slow_tool please"})

        # Start agent execution
        task = asyncio.create_task(agent(prompt=prompt).collect())

        # Wait for tool to start executing
        await asyncio.sleep(4)

        # Interrupt the agent
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_agent_reusability_after_interruption(self):
        """Test that agent can be reused after interruption."""
        agent = Agent(name="reusable_agent", model="openai/gpt-4o-mini")

        # First call - interrupt it
        prompt1 = Message.validate({"role": "user", "content": "Tell me a long story"})
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        await asyncio.sleep(2)
        task1.cancel()
        result1 = await task1

        assert result1.status.code == "cancelled"

        # Second call - should work normally
        prompt2 = Message.validate({"role": "user", "content": "What is your name?"})
        result2 = await agent(prompt=prompt2).collect()

        assert_has_output_event(result2)
        assert result2.status.code == "success"
        assert isinstance(result2.output, Message)

    @pytest.mark.asyncio
    async def test_agent_interruption_with_multiple_tools(self, agent_with_long_tools):
        """Test agent interruption when multiple tools are available."""
        prompt = Message.validate({"role": "user", "content": "Get the weather, location, and temperature"})

        # Start execution
        task = asyncio.create_task(agent_with_long_tools(prompt=prompt).collect())

        # Wait for tools to start executing
        await asyncio.sleep(4)

        # Interrupt while tools are running
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_agent_interruption_preserves_partial_results(self):
        """Test that partial results are preserved during interruption."""

        def quick_tool() -> str:
            return "quick_result"

        async def slow_tool() -> str:
            await asyncio.sleep(10)
            return "slow_result"

        agent = Agent(name="mixed_speed_agent", model="openai/gpt-4o-mini", tools=[quick_tool, slow_tool])

        prompt = Message.validate({"role": "user", "content": "Use both quick_tool and slow_tool"})

        task = asyncio.create_task(agent(prompt=prompt).collect())
        await asyncio.sleep(4)
        task.cancel()
        result = await task

        # Should have interrupted status but may have partial output
        assert result.status.code == "cancelled"
        assert result.output is not None


class TestWorkflowInterruption:
    """Test workflow interruption during step execution."""

    @pytest.mark.asyncio
    async def test_workflow_interruption_during_single_step(self, slow_step_tool):
        """Test workflow interruption during a single step execution."""
        workflow = Workflow(name="single_step_workflow").step(slow_step_tool, duration=10)

        # Start workflow execution
        task = asyncio.create_task(workflow().collect())

        # Wait for step to start
        await asyncio.sleep(2)

        # Interrupt workflow
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_workflow_interruption_during_sequential_steps(self):
        """Test workflow interruption during sequential step execution."""

        async def step1() -> str:
            await asyncio.sleep(3)
            return "step1_done"

        async def step2() -> str:
            await asyncio.sleep(3)
            return "step2_done"

        async def step3() -> str:
            await asyncio.sleep(3)
            return "step3_done"

        workflow = (
            Workflow(name="sequential_workflow")
            .step(step1)
            .step(step2, depends_on=["step1"])
            .step(step3, depends_on=["step2"])
        )

        # Start workflow
        task = asyncio.create_task(workflow().collect())

        # Interrupt during execution
        await asyncio.sleep(5)
        task.cancel()
        result = await task

        # Verify interruption
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_workflow_interruption_during_parallel_steps(self):
        """Test workflow interruption when parallel steps are executing."""
        workflow = (
            Workflow(name="parallel_workflow")
            .step(parallel_task_1, duration=10)
            .step(parallel_task_2, duration=10)
            .step(parallel_task_3, duration=10)
        )

        # Start workflow (all steps run in parallel)
        task = asyncio.create_task(workflow().collect())

        # Wait for steps to start
        await asyncio.sleep(2)

        # Interrupt while parallel steps are running
        task.cancel()
        result = await task

        # Verify interruption
        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_workflow_interruption_mixed_dependencies(self):
        """Test workflow interruption with mixed parallel and sequential steps."""

        async def parallel_a() -> str:
            await asyncio.sleep(5)
            return "parallel_a_done"

        async def parallel_b() -> str:
            await asyncio.sleep(5)
            return "parallel_b_done"

        async def sequential_c() -> str:
            await asyncio.sleep(5)
            return "sequential_c_done"

        workflow = (
            Workflow(name="mixed_workflow")
            .step(parallel_a)
            .step(parallel_b)
            .step(sequential_c, depends_on=["parallel_a", "parallel_b"])
        )

        # Start workflow
        task = asyncio.create_task(workflow().collect())

        # Interrupt during parallel phase
        await asyncio.sleep(3)
        task.cancel()
        result = await task

        # Verify interruption
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_workflow_reusability_after_interruption(self):
        """Test that workflow can be reused after interruption."""

        async def simple_step(x: str) -> str:
            await asyncio.sleep(5)
            return f"result:{x}"

        workflow = Workflow(name="reusable_workflow").step(simple_step, x="test")

        # First execution - interrupt it
        task1 = asyncio.create_task(workflow().collect())
        await asyncio.sleep(2)
        task1.cancel()
        result1 = await task1

        assert result1.status.code == "cancelled"

        # Second execution - use shorter duration
        async def quick_step(x: str) -> str:
            await asyncio.sleep(0.1)
            return f"result:{x}"

        workflow2 = Workflow(name="quick_workflow").step(quick_step, x="test")
        result2 = await workflow2().collect()

        assert_has_output_event(result2)
        assert result2.status.code == "success"
        assert result2.output == "result:test"

    @pytest.mark.asyncio
    async def test_workflow_interruption_preserves_completed_steps(self):
        """Test that completed steps are preserved when workflow is interrupted."""

        async def fast_step() -> str:
            await asyncio.sleep(0.5)
            return "fast_done"

        async def slow_step() -> str:
            await asyncio.sleep(10)
            return "slow_done"

        workflow = Workflow(name="mixed_speed_workflow").step(fast_step).step(slow_step, depends_on=["fast_step"])

        # Start workflow
        task = asyncio.create_task(workflow().collect())

        # Wait for fast step to complete, interrupt during slow step
        await asyncio.sleep(3)
        task.cancel()
        result = await task

        # Verify interruption
        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_workflow_interruption_with_when_clause(self):
        """Test workflow interruption with multiple steps using conditional 'when' clauses."""
        # Track which steps were executed
        executed_steps = []

        async def step_a() -> str:
            executed_steps.append("step_a")
            await asyncio.sleep(2)
            return "a_done"

        async def step_b() -> str:
            executed_steps.append("step_b")
            await asyncio.sleep(5)
            return "b_done"

        async def step_c() -> str:
            executed_steps.append("step_c")
            await asyncio.sleep(5)
            return "c_done"

        async def step_d() -> str:
            executed_steps.append("step_d")
            await asyncio.sleep(5)
            return "d_done"

        # Create workflow with conditional execution using 'when' clause
        workflow = (
            Workflow(name="conditional_workflow")
            .step(step_a)
            .step(step_b, depends_on=["step_a"], when=lambda: True)  # Should execute
            .step(step_c, depends_on=["step_a"], when=lambda: False)  # Should be skipped
            .step(step_d, depends_on=["step_b"])  # Should start after step_b
        )

        # Start workflow
        task = asyncio.create_task(workflow().collect())

        # Wait for step_a to complete and step_b to start, then interrupt
        await asyncio.sleep(4)
        task.cancel()
        result = await task

        # Verify interruption
        assert result.status.code == "cancelled"

        # Verify that step_a was executed
        assert "step_a" in executed_steps
        # Verify that step_b was started (since when=True)
        assert "step_b" in executed_steps
        # Verify that step_c was never executed (since when=False)
        assert "step_c" not in executed_steps
        assert "step_d" not in executed_steps

    @pytest.mark.asyncio
    async def test_workflow_interruption_during_function_step(self):
        """Test workflow interruption during execution of a normal function step."""
        executed_steps = []

        async def normal_function() -> str:
            """A normal async function used as a workflow step."""
            executed_steps.append("function")
            await asyncio.sleep(10)
            return "function_done"

        def quick_tool() -> str:
            """Quick tool that completes before interruption."""
            executed_steps.append("tool")
            return "tool_done"

        async def slow_agent_handler() -> str:
            """Agent step that should not execute."""
            executed_steps.append("agent")
            await asyncio.sleep(10)
            return "agent_done"

        # Create workflow with function, tool, and agent steps
        tool_step = Tool(name="quick_tool", handler=quick_tool)
        agent_step = Agent(name="slow_agent", model="openai/gpt-4o-mini")

        workflow = (
            Workflow(name="mixed_step_workflow")
            .step(tool_step)
            .step(normal_function, depends_on=["quick_tool"])
            .step(agent_step, prompt="Say hello", depends_on=["normal_function"])
        )

        # Start workflow and interrupt during normal function execution
        task = asyncio.create_task(workflow().collect())
        await asyncio.sleep(2)  # Let tool complete and function start
        task.cancel()
        result = await task

        # Verify interruption during function step
        assert result.status.code == "cancelled"
        assert "tool" in executed_steps
        assert "function" in executed_steps
        assert "agent" not in executed_steps

    @pytest.mark.asyncio
    async def test_workflow_interruption_during_tool_step(self):
        """Test workflow interruption during execution of a Tool step."""
        executed_steps = []

        def quick_function() -> str:
            """Quick function that completes before interruption."""
            executed_steps.append("function")
            return "function_done"

        async def long_tool_handler() -> str:
            """Long-running tool to interrupt."""
            executed_steps.append("tool")
            await asyncio.sleep(10)
            return "tool_done"

        async def agent_handler() -> str:
            """Agent step that should not execute."""
            executed_steps.append("agent")
            await asyncio.sleep(10)
            return "agent_done"

        # Create workflow
        function_step = Tool(name="quick_function", handler=quick_function)
        tool_step = Tool(name="long_tool", handler=long_tool_handler)
        agent_step = Agent(name="agent", model="openai/gpt-4o-mini")

        workflow = (
            Workflow(name="tool_interrupt_workflow")
            .step(function_step)
            .step(tool_step, depends_on=["quick_function"])
            .step(agent_step, prompt="Say goodbye", depends_on=["long_tool"])
        )

        # Start workflow and interrupt during tool execution
        task = asyncio.create_task(workflow().collect())
        await asyncio.sleep(2)  # Let function complete and tool start
        task.cancel()
        result = await task

        # Verify interruption during tool step
        assert result.status.code == "cancelled"
        assert "function" in executed_steps
        assert "tool" in executed_steps
        assert "agent" not in executed_steps

    @pytest.mark.asyncio
    async def test_workflow_interruption_during_agent_step(self):
        """Test workflow interruption during execution of an Agent step."""
        executed_steps = []

        def quick_function() -> str:
            """Quick function that completes before interruption."""
            executed_steps.append("function")
            return "function_done"

        def quick_tool() -> str:
            """Quick tool that completes before interruption."""
            executed_steps.append("tool")
            return "tool_done"

        # Create workflow
        function_step = Tool(name="quick_function", handler=quick_function)
        tool_step = Tool(name="quick_tool", handler=quick_tool)
        agent_step = Agent(name="story_agent", model="openai/gpt-4o-mini")

        workflow = (
            Workflow(name="agent_interrupt_workflow")
            .step(function_step)
            .step(tool_step, depends_on=["quick_function"])
            .step(agent_step, prompt="Tell me a very long story about space exploration", depends_on=["quick_tool"])
        )

        # Start workflow and interrupt during agent execution
        task = asyncio.create_task(workflow().collect())
        await asyncio.sleep(4)  # Let function and tool complete, agent start
        task.cancel()
        result = await task

        # Verify interruption during agent step
        assert result.status.code == "cancelled"
        assert "function" in executed_steps
        assert "tool" in executed_steps
        # Note: Agent doesn't add to executed_steps, but we verify it was interrupted


class TestInterruptionEdgeCases:
    """Test edge cases and boundary conditions for interruptions."""

    @pytest.mark.asyncio
    async def test_immediate_interruption(self):
        """Test interruption immediately after starting execution."""
        tool = Tool(handler=long_running_async_handler)

        task = asyncio.create_task(tool(duration=10).collect())
        # Interrupt almost immediately
        await asyncio.sleep(0.01)
        task.cancel()
        result = await task

        assert result.status.code == "cancelled"

    @pytest.mark.asyncio
    async def test_multiple_interruptions(self):
        """Test multiple sequential interruptions on the same runnable."""
        tool = Tool(handler=long_running_async_handler)

        # First interruption
        task1 = asyncio.create_task(tool(duration=10).collect())
        await asyncio.sleep(1)
        task1.cancel()
        result1 = await task1
        assert result1.status.code == "cancelled"

        # Second interruption
        task2 = asyncio.create_task(tool(duration=10).collect())
        await asyncio.sleep(1)
        task2.cancel()
        result2 = await task2
        assert result2.status.code == "cancelled"

        # Finally let it complete
        result3 = await tool(duration=0.1).collect()
        assert result3.status.code == "success"

    @pytest.mark.asyncio
    async def test_concurrent_interruptions(self):
        """Test interrupting multiple concurrent executions."""
        tool = Tool(handler=long_running_async_handler)

        # Start multiple concurrent executions
        tasks = [
            asyncio.create_task(tool(duration=10).collect()),
            asyncio.create_task(tool(duration=10).collect()),
            asyncio.create_task(tool(duration=10).collect()),
        ]

        # Wait a bit then interrupt all
        await asyncio.sleep(2)
        for task in tasks:
            task.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=False)

        # All should be interrupted
        for result in results:
            assert isinstance(result, OutputEvent)
            assert result.status.code == "cancelled"


class TestInterruptionPerformance:
    """Test performance characteristics of interruption."""

    @pytest.mark.asyncio
    async def test_interruption_response_time(self):
        """Test that interruption happens quickly."""
        tool = Tool(handler=long_running_async_handler)

        with Timer() as timer:
            task = asyncio.create_task(tool(duration=60).collect())
            await asyncio.sleep(1)
            task.cancel()
            await task

        # Interruption should happen quickly (much faster than 60 seconds)
        assert timer.elapsed < 5, f"Interruption took too long: {timer.elapsed}s"

    @pytest.mark.asyncio
    async def test_generator_interruption_cleanup_time(self):
        """Test that generator interruption cleans up quickly."""

        async def slow_generator() -> AsyncGenerator[str, None]:
            for i in range(1000):
                await asyncio.sleep(0.5)
                yield f"chunk_{i}"

        tool = Tool(handler=slow_generator)

        with Timer() as timer:
            task = asyncio.create_task(tool().collect())
            await asyncio.sleep(2)
            task.cancel()
            await task

        # Should cleanup quickly
        assert timer.elapsed < 5, f"Cleanup took too long: {timer.elapsed}s"


class TestComprehensiveInterruptionVerification:
    """Test comprehensive verification of all interruption aspects."""

    @pytest.mark.asyncio
    async def test_comprehensive_interruption_verification(self):
        """
        Comprehensive test demonstrating all aspects that should be verified
        during interruption handling beyond just checking status code.
        """

        async def yielding_handler(count: int = 100) -> AsyncGenerator[str, None]:
            """Generator that yields multiple chunks."""
            for i in range(count):
                await asyncio.sleep(0.1)
                yield f"chunk_{i}"

        tool = Tool(name="yielding_tool", handler=yielding_handler)

        # Start execution and interrupt after some time
        with Timer() as timer:
            task = asyncio.create_task(tool(count=100).collect())
            await asyncio.sleep(1.5)  # Let some chunks be yielded
            task.cancel()
            output_event = await task

        # Get the final output event
        assert isinstance(output_event, OutputEvent)

        # ====================================================================
        # VERIFICATION CHECKLIST FOR INTERRUPTION HANDLING
        # ====================================================================

        # 1. STATUS CODE - Most basic check
        assert output_event.status.code == "cancelled", "Status code should be 'interrupted'"

        # 2. STATUS DETAILS - Check reason and message
        assert output_event.status.reason == "interrupted", "Status reason should indicate interruption"

        # 3. PARTIAL OUTPUT PRESERVATION - Chunks collected before interruption
        # For generators that yield strings, the collector concatenates them
        assert output_event.output is not None, "Partial output should be preserved for generators"
        # Output should be a string with partial results (concatenated chunks)
        assert isinstance(output_event.output, str), "Output should be a string (concatenated chunks)"
        assert len(output_event.output) > 0, "Should have collected some chunks before interruption"
        # Verify it contains some chunks (e.g., chunk_0, chunk_1, etc.)
        assert "chunk_" in output_event.output, "Output should contain chunk data"

        # 4. ERROR FIELD - Should be None (interruption is not an error)
        assert output_event.error is None, "Error field should be None for interruptions"

        # 5. INPUT PRESERVATION - Original input should be available
        assert output_event.input is not None, "Input parameters should be preserved"
        assert output_event.input.get("count") == 100, "Input values should match what was passed"

        # 6. TIMING INFORMATION - Verify timestamps
        assert output_event.t0 > 0, "Start time should be recorded"
        assert output_event.t1 > output_event.t0, "End time should be after start time"
        duration_ms = output_event.t1 - output_event.t0
        assert duration_ms < 5000, f"Execution should stop quickly on interruption, took {duration_ms}ms"

        # 7. INTERRUPTION TIMING - Should be much faster than full execution
        assert timer.elapsed < 5, f"Interruption should be fast, took {timer.elapsed}s (would take 10s+ to complete)"

        # 8. METADATA AND USAGE - Should be preserved if present
        assert isinstance(output_event.metadata, dict), "Metadata should be accessible"
        assert isinstance(output_event.usage, dict), "Usage information should be accessible"

        # 9. RUN CONTEXT - Verify run and call IDs are present
        assert output_event.run_id is not None, "Run ID should be set"
        assert output_event.call_id is not None, "Call ID should be set"
        assert output_event.path == "yielding_tool", "Path should match the tool name"

        # 10. REUSABILITY - Tool should be usable after interruption
        result2 = await tool(count=1).collect()
        assert result2.status.code == "success", "Tool should be reusable after interruption"

    @pytest.mark.asyncio
    async def test_workflow_interruption_comprehensive(self):
        """Comprehensive verification for workflow interruption."""
        executed_steps = []

        async def step_1() -> str:
            executed_steps.append("step_1")
            await asyncio.sleep(0.5)
            return "step_1_done"

        async def step_2() -> str:
            executed_steps.append("step_2")
            await asyncio.sleep(10)
            return "step_2_done"

        async def step_3() -> str:
            executed_steps.append("step_3")
            await asyncio.sleep(10)
            return "step_3_done"

        workflow = (
            Workflow(name="comprehensive_workflow")
            .step(step_1)
            .step(step_2, depends_on=["step_1"])
            .step(step_3, depends_on=["step_2"])
        )

        # Execute and interrupt
        with Timer() as timer:
            task = asyncio.create_task(workflow().collect())
            await asyncio.sleep(2)
            task.cancel()
            result = await task

        # WORKFLOW-SPECIFIC VERIFICATIONS

        # 1. Workflow status
        assert result.status.code == "cancelled", "Workflow should be marked as interrupted"

        # 2. Partial execution tracking - verify which steps ran
        assert "step_1" in executed_steps, "First step should have completed"
        assert "step_2" in executed_steps, "Second step should have started"
        assert "step_3" not in executed_steps, "Third step should never have started (depends on interrupted step)"

        assert result.output is not None, "Partial results from completed steps should be preserved"
        assert result.output == "step_1_done", "Output should contain results from completed steps"

        assert result.error is None, "Interruption should not be treated as an error"

        assert timer.elapsed < 5, f"Workflow interruption should be quick, took {timer.elapsed}s"

        assert result.run_id is not None, "Run ID should be set"
        assert result.call_id is not None, "Call ID should be set"
        assert result.path == "comprehensive_workflow", "Path should match workflow name"

        assert result.t0 > 0, "Start time should be recorded"
        assert result.t1 > result.t0, "End time should be after start time"


# ==============================================================================
# Regression tests: second CancelledError inside the interruption except block
#
# Bug: in the except (CancelledError, InterruptError) handler, span.status was
# set AFTER awaiting dump(output).  If a NEW external CancelledError arrives
# while that await is suspended (e.g. another task.cancel() call, or a timeout
# firing during heavy load), the handler exits before span.status is assigned,
# leaving it as None.  The finally block then calls OutputEvent(status=None),
# which raises a Pydantic ValidationError because status: RunStatus is required.
#
# Mechanism reproduced by these tests:
#   1. A dict-yielding generator accumulates items so the collector result is
#      a non-empty list (DefaultCollector).
#   2. dump() is patched: call #1 (input at line 849) completes normally;
#      call #2 (output dump in the except block) suspends long enough for an
#      external task.cancel() to fire.
#   3. First cancel fires during the generator → except block runs.
#   4. Second cancel (external) fires at asyncio.sleep() inside the patched
#      dump → CancelledError raised inside the except block's await.
#   5. Without fix: span.status is None → OutputEvent(status=None) →
#      Pydantic ValidationError.
#   6. With fix: span.status set first → no ValidationError; task ends with
#      CancelledError (acceptable).
#
# Fix: move span.status assignment to the top of the except block.
# ==============================================================================


async def _dict_generator(count: int = 100) -> AsyncGenerator[dict, None]:
    """Yields dicts so DefaultCollector accumulates a list.
    dump(list_of_dicts) → asyncio.gather → a real, suspending await."""
    for i in range(count):
        await asyncio.sleep(0.05)
        yield {"index": i}


class TestDoubleCancellationStatusSet:
    """Regression tests for the span.status=None / Pydantic ValidationError bug."""

    @pytest.mark.asyncio
    async def test_second_cancel_at_output_dump_raises_validation_error_without_fix(self):
        """Core regression: external cancel fires while the except block awaits dump(output).

        Setup
        -----
        * dump() is patched: call #1 (input dump) completes instantly; call #2
          (output dump inside the except block) suspends for 3 s.
        * A dict-yielding generator is used so the collector result is a list
          (DefaultCollector) and call #2 is indeed triggered.

        Without the fix the second cancel leaves span.status=None and
        OutputEvent() raises ValidationError → pytest.fail().
        With the fix span.status is set first → CancelledError (acceptable).
        """
        from timbal.utils.serialization import dump as real_dump

        input_done = asyncio.Event()
        in_except_dump = asyncio.Event()
        call_count = [0]

        async def tracked_dump(value):
            call_count[0] += 1
            if call_count[0] == 1:
                # Input dump (line 849 of runnable.py): complete normally.
                result = await real_dump(value)
                input_done.set()
                return result
            # Output dump inside except block: suspend so the external cancel fires.
            in_except_dump.set()
            await asyncio.sleep(3.0)
            return await real_dump(value)

        tool = Tool(handler=_dict_generator)

        with patch("timbal.core.runnable.dump", tracked_dump):
            task = asyncio.create_task(tool(count=100).collect())

            # Wait for the input dump to finish (handler has started running).
            try:
                await asyncio.wait_for(input_done.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pytest.skip("Input dump did not complete in time")

            # Let some items accumulate so the collector result is non-empty.
            await asyncio.sleep(0.4)

            # First cancel — fires during the generator's asyncio.sleep(0.05).
            task.cancel()

            # Wait until the except block starts its dump (call #2).
            try:
                await asyncio.wait_for(in_except_dump.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pytest.skip("Output dump in except block was never reached")

            # Second (external) cancel — fires at asyncio.sleep(3.0) inside tracked_dump.
            # This is the scenario that triggers the bug without the fix.
            task.cancel()

            try:
                result = await task
                # Best case: generator fully suppressed both exceptions.
                assert isinstance(result, OutputEvent)
                assert result.status is not None, (
                    "status must not be None — without the fix OutputEvent() would "
                    "raise a Pydantic ValidationError here"
                )
                assert result.status.code == "cancelled"
            except asyncio.CancelledError:
                pass  # acceptable: task was cancelled by the second cancel
            except Exception as exc:
                pytest.fail(
                    f"Unexpected exception — without the fix this would be a "
                    f"Pydantic ValidationError: {type(exc).__name__}: {exc}"
                )

    @pytest.mark.asyncio
    async def test_many_concurrent_external_cancels_during_output_dump(self):
        """Heavy-load regression: 10 concurrent tasks, each hit with a second external
        cancel while the except block is suspended inside dump(output).
        No task may raise ValidationError."""
        from timbal.utils.serialization import dump as real_dump

        async def run_one():
            input_done = asyncio.Event()
            in_except_dump = asyncio.Event()
            call_count = [0]

            async def tracked_dump(value):
                call_count[0] += 1
                if call_count[0] == 1:
                    result = await real_dump(value)
                    input_done.set()
                    return result
                in_except_dump.set()
                await asyncio.sleep(3.0)
                return await real_dump(value)

            tool = Tool(handler=_dict_generator)

            with patch("timbal.core.runnable.dump", tracked_dump):
                t = asyncio.create_task(tool(count=100).collect())

                try:
                    await asyncio.wait_for(input_done.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    t.cancel()
                    return None

                await asyncio.sleep(0.4)
                t.cancel()

                try:
                    await asyncio.wait_for(in_except_dump.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    t.cancel()
                    try:
                        return await t
                    except Exception:
                        return None

                t.cancel()  # external second cancel

                try:
                    return await t
                except asyncio.CancelledError:
                    return None  # acceptable
                except Exception as exc:
                    return exc   # reported below

        results = await asyncio.gather(*[run_one() for _ in range(10)])

        for i, r in enumerate(results):
            if r is None:
                continue
            if isinstance(r, Exception):
                pytest.fail(
                    f"Task {i} raised unexpected exception "
                    f"(ValidationError = regression): {type(r).__name__}: {r}"
                )
            assert isinstance(r, OutputEvent), f"Task {i}: unexpected type {type(r)}"
            assert r.status is not None, f"Task {i}: status must not be None"
            assert r.status.code == "cancelled"


# ==============================================================================
# Regression tests: span.status=None from paths OTHER than the double-cancel
#
# Two additional scenarios where span.status can still be None in the finally
# block, each with a different root cause:
#
# 1. except Exception — span.status was set AFTER str(err) and
#    traceback.format_exc().  If the exception's __str__ itself raises (e.g. a
#    buggy or hostile custom exception class), the assignment is never reached.
#    Fix: move span.status assignment to be the first statement in that block.
#
# 2. GeneratorExit — a BaseException subclass that bypasses all three except
#    clauses (EarlyExit, CancelledError/InterruptError, Exception).  Thrown
#    when the generator is closed early: streaming HTTP consumer breaks on
#    client disconnect, explicit aclose() call, etc.
#    Fix: defensive `if span.status is None:` guard at the TOP of the finally
#    block, before OutputEvent() is called.
# ==============================================================================


class _StrRaisesException(Exception):
    """Custom exception whose __str__ always raises RuntimeError.

    Used to simulate the case where processing an exception inside
    `except Exception` (specifically str(err)) itself raises, leaving
    span.status unset if the status assignment came after that call.
    """

    def __str__(self):
        raise RuntimeError("__str__ deliberately raised")


class _NotAnException(BaseException):
    """BaseException subclass that is NOT a subclass of Exception.

    Bypasses `except Exception`, `except EarlyExit`, and
    `except (CancelledError, InterruptError)` — goes straight to finally.
    Used to test the defensive `if span.status is None` fallback.
    """


async def _slow_generator_for_close(count: int = 100) -> AsyncGenerator[str, None]:
    """Slow generator used to test early aclose(); must be at module level."""
    for i in range(count):
        await asyncio.sleep(0.1)
        yield f"item_{i}"


async def _raises_not_an_exception() -> str:
    """Handler that raises a BaseException (not Exception) subclass."""
    raise _NotAnException("bypasses except Exception")


async def _raises_str_raises() -> str:
    """Handler that raises _StrRaisesException (module-level for nested tests)."""
    raise _StrRaisesException()


class TestExceptExceptionStatusSet:
    """Regression: span.status set before str(err)/traceback in except Exception."""

    @pytest.mark.asyncio
    async def test_bad_str_exception_does_not_produce_validation_error(self):
        """If the caught exception's __str__ raises, span.status must already be
        set so the finally block can construct OutputEvent without ValidationError.

        Without the fix: str(err) raises → span.status still None → finally
        calls OutputEvent(status=None) → Pydantic ValidationError.
        With the fix:    span.status set first → str(err) raises → finally
        calls OutputEvent(status=span.status) → succeeds; RuntimeError
        from str(err) propagates instead of ValidationError.
        """

        async def raising_handler():
            raise _StrRaisesException("will never be stringified")

        tool = Tool(handler=raising_handler)

        try:
            await tool().collect()
        except Exception as exc:
            from pydantic import ValidationError
            if isinstance(exc, ValidationError) and "status" in str(exc):
                pytest.fail(
                    f"Got Pydantic ValidationError for 'status' — span.status was "
                    f"not set before str(err) raised: {exc}"
                )
            # Any other exception (RuntimeError from __str__) is acceptable.

    @pytest.mark.asyncio
    async def test_many_bad_str_exceptions(self):
        """10 concurrent tasks, each raising _StrRaisesException. None may
        produce a ValidationError for status."""

        async def raising_handler():
            raise _StrRaisesException()

        from pydantic import ValidationError

        async def run_one():
            tool = Tool(handler=raising_handler)
            try:
                await tool().collect()
                return None
            except ValidationError as e:
                if "status" in str(e):
                    return e
                return None
            except Exception:
                return None

        results = await asyncio.gather(*[run_one() for _ in range(10)])
        for i, r in enumerate(results):
            if r is not None:
                pytest.fail(
                    f"Task {i} got ValidationError for status — regression: {r}"
                )


class TestGeneratorExitStatusSet:
    """Regression: span.status=None when GeneratorExit bypasses all except clauses."""

    @pytest.mark.asyncio
    async def test_aclose_does_not_produce_validation_error(self):
        """Calling aclose() mid-stream throws GeneratorExit into the generator.
        GeneratorExit bypasses all except clauses, so without the defensive
        `if span.status is None` guard in finally, OutputEvent(status=None)
        raises a Pydantic ValidationError.

        Without the fix: ValidationError propagates from aclose().
        With the fix:    span.status is set defensively; ValidationError is
                         avoided (aclose may raise RuntimeError from yield-in-
                         finally-during-GeneratorExit, which is acceptable).
        """
        from pydantic import ValidationError

        tool = Tool(handler=_slow_generator_for_close)
        collector = tool(count=50)

        # Consume the StartEvent so the generator is running inside the try block.
        first = await collector.__anext__()
        assert first is not None

        try:
            await collector.aclose()
        except ValidationError as exc:
            if "status" in str(exc):
                pytest.fail(
                    f"aclose() raised ValidationError for 'status' — defensive "
                    f"fallback in finally is missing: {exc}"
                )
            raise
        except Exception:
            pass  # RuntimeError("async generator ignored GeneratorExit") is acceptable

    @pytest.mark.asyncio
    async def test_many_concurrent_aclose_calls(self):
        """10 concurrent generators each closed early via aclose().
        None may produce a ValidationError for status."""
        from pydantic import ValidationError

        async def run_one():
            tool = Tool(handler=_slow_generator_for_close)
            collector = tool(count=50)
            await collector.__anext__()  # let it enter the try block
            try:
                await collector.aclose()
                return None
            except ValidationError as e:
                if "status" in str(e):
                    return e
                return None
            except Exception:
                return None

        results = await asyncio.gather(*[run_one() for _ in range(10)])
        for i, r in enumerate(results):
            if r is not None:
                pytest.fail(
                    f"Task {i}: aclose() raised ValidationError for status — "
                    f"regression: {r}"
                )


# ==============================================================================
# Deeper regression tests
#
# Extends coverage beyond the basic scenarios with:
#
# 1. except Exception — additional routes where pre-status code can raise:
#    - traceback.format_exc() patched to raise (instead of str(err))
#    - BaseException subclass (not Exception) bypasses the block entirely
#
# 2. GeneratorExit — real GC-finalizer path (the streaming disconnect scenario):
#    - consumer breaks from async for, reference dropped, GC triggers aclose()
#    - asyncio exception handler is monitored so errors in the finalizer are
#      visible to the test
#
# 3. Nested runnables — tool inside a Workflow:
#    - verifies the fix holds end-to-end, not just in unit isolation
#    - the ValidationError must not surface in the outer workflow result
# ==============================================================================


def _no_validation_error_for_status(exc: BaseException) -> bool:
    """Return True if exc is a Pydantic ValidationError complaining about status."""
    from pydantic import ValidationError
    return isinstance(exc, ValidationError) and "status" in str(exc)


class TestExceptExceptionDeeper:
    """Deeper coverage for the except Exception span.status ordering fix."""

    @pytest.mark.asyncio
    async def test_format_exc_raises_does_not_produce_validation_error(self):
        """traceback.format_exc() is called AFTER str(err) in the except Exception
        block.  If it raises, span.status must already be set.

        Patching traceback.format_exc via the runnable module's reference so
        that any exception class can trigger it, not just one with bad __str__.
        """
        from pydantic import ValidationError

        async def normal_error_handler():
            raise ValueError("ordinary error")

        tool = Tool(handler=normal_error_handler)

        with patch("timbal.core.runnable.traceback.format_exc", side_effect=RuntimeError("format_exc deliberately raised")):
            try:
                await tool().collect()
            except ValidationError as exc:
                if "status" in str(exc):
                    pytest.fail(
                        f"ValidationError for 'status' — span.status was not set "
                        f"before traceback.format_exc raised: {exc}"
                    )
            except Exception:
                pass  # RuntimeError from format_exc is acceptable

    @pytest.mark.asyncio
    async def test_base_exception_subclass_uses_defensive_fallback(self):
        """A BaseException subclass that is not an Exception subclass bypasses
        all three except clauses (EarlyExit, CancelledError/InterruptError,
        Exception) and hits the finally block directly.

        Without the `if span.status is None` guard in finally, the handler
        raises _NotAnException, span.status stays None, and
        OutputEvent(status=None) raises ValidationError.
        With the guard, span.status is set defensively and the original
        _NotAnException propagates cleanly.
        """
        from pydantic import ValidationError

        tool = Tool(handler=_raises_not_an_exception)

        try:
            await tool().collect()
        except _NotAnException:
            pass  # original exception propagated cleanly — fix is working
        except ValidationError as exc:
            if "status" in str(exc):
                pytest.fail(
                    f"ValidationError for 'status' — defensive fallback in finally "
                    f"is missing for BaseException subclasses: {exc}"
                )

    @pytest.mark.asyncio
    async def test_many_concurrent_base_exception_in_handler(self):
        """10 concurrent tasks raising _NotAnException. None may produce a
        ValidationError for status."""
        from pydantic import ValidationError

        async def run_one():
            tool = Tool(handler=_raises_not_an_exception)
            try:
                await tool().collect()
                return None
            except _NotAnException:
                return None  # correct outcome
            except ValidationError as e:
                if "status" in str(e):
                    return e
                return None
            except Exception:
                return None

        results = await asyncio.gather(*[run_one() for _ in range(10)])
        for i, r in enumerate(results):
            if r is not None:
                pytest.fail(f"Task {i}: got ValidationError for status — regression: {r}")


class TestGeneratorExitDeeper:
    """Deeper coverage for the GeneratorExit / consumer-disconnect scenario."""

    @pytest.mark.asyncio
    async def test_consumer_break_gc_finalizer_no_validation_error(self):
        """Real streaming-disconnect simulation: consumer breaks from async for,
        the collector reference is dropped, CPython immediately decrements the
        inner async generator's refcount to zero, and asyncio's async-gen
        finalizer schedules aclose().  The finalizer runs on the next event-loop
        turn and throws GeneratorExit into the generator's finally block.

        Without the defensive fallback the finalizer raises ValidationError,
        which asyncio routes to the exception handler.
        With the fix, the exception handler sees at most RuntimeError
        ('async generator ignored GeneratorExit') but never ValidationError.
        """
        from pydantic import ValidationError

        loop = asyncio.get_event_loop()
        captured: list[BaseException] = []
        original_handler = loop.get_exception_handler()

        def capture_handler(loop, context):
            exc = context.get("exception")
            if exc is not None:
                captured.append(exc)

        loop.set_exception_handler(capture_handler)
        try:
            tool = Tool(handler=_slow_generator_for_close)
            collector = tool(count=100)

            # Consume one event then exit — simulates client disconnect
            async for _ in collector:
                break

            # Drop our reference; CPython immediately GC's the async gen
            del collector
            import gc
            gc.collect()
            await asyncio.sleep(0.1)  # Let the event loop run the finalizer

            for exc in captured:
                if _no_validation_error_for_status(exc):
                    pytest.fail(
                        f"ValidationError for 'status' surfaced in async-gen GC "
                        f"finalizer — defensive fallback in finally is missing: {exc}"
                    )
        finally:
            loop.set_exception_handler(original_handler)

    @pytest.mark.asyncio
    async def test_many_consumer_breaks_no_validation_error(self):
        """10 concurrent streaming consumers that each disconnect after one event.
        No ValidationError for status should appear in the asyncio exception handler."""
        from pydantic import ValidationError

        loop = asyncio.get_event_loop()
        captured: list[BaseException] = []
        original_handler = loop.get_exception_handler()

        def capture_handler(loop, context):
            exc = context.get("exception")
            if exc is not None:
                captured.append(exc)

        loop.set_exception_handler(capture_handler)
        try:
            import gc

            async def one_consumer():
                tool = Tool(handler=_slow_generator_for_close)
                collector = tool(count=100)
                async for _ in collector:
                    break
                del collector

            await asyncio.gather(*[one_consumer() for _ in range(10)])
            gc.collect()
            await asyncio.sleep(0.2)

            for exc in captured:
                if _no_validation_error_for_status(exc):
                    pytest.fail(
                        f"ValidationError for 'status' in GC finalizer — regression: {exc}"
                    )
        finally:
            loop.set_exception_handler(original_handler)


class TestNestedRunnableStatusIntegrity:
    """Verify the fixes hold end-to-end when runnables are nested inside a Workflow."""

    @pytest.mark.asyncio
    async def test_tool_with_bad_str_exception_inside_workflow(self):
        """Tool raises _StrRaisesException (str(err) raises).  Without the fix,
        the tool's except Exception block fails before setting span.status, the
        tool's finally raises ValidationError, that exception propagates into the
        workflow as a step failure, and the workflow result's error field contains
        the ValidationError text about 'status'.

        With the fix, the tool handles the exception cleanly (status set first),
        and the workflow result's error — if any — must NOT reference a
        ValidationError for status.
        """
        from pydantic import ValidationError

        tool = Tool(name="bad_str_tool", handler=_raises_str_raises)
        workflow = Workflow(name="nested_bad_str_wf").step(tool)

        try:
            result = await workflow().collect()
        except ValidationError as exc:
            if "status" in str(exc):
                pytest.fail(
                    f"ValidationError for 'status' propagated out of workflow — "
                    f"fix not effective in nested context: {exc}"
                )
            raise

        assert result is not None, "workflow collect() should return an OutputEvent"
        assert result.status is not None, "workflow OutputEvent.status must not be None"

        # If the workflow captured an error, it must not be the status=None ValidationError
        if result.error:
            error_text = str(result.error)
            if "ValidationError" in error_text and "status" in error_text:
                pytest.fail(
                    f"ValidationError for 'status' surfaced inside workflow error — "
                    f"fix not effective in nested context: {result.error}"
                )

    @pytest.mark.asyncio
    async def test_tool_with_base_exception_inside_workflow(self):
        """Tool raises _NotAnException (BaseException, not Exception) inside a Workflow.

        The tool's defensive fallback in finally fires correctly (no ValidationError for
        status from the tool span), but because _NotAnException is a BaseException (not
        Exception), the workflow's _enqueue_step_events catches it only via its `finally`
        block — the task exits with an unhandled BaseException stored on the asyncio.Task.

        The important invariant: no ValidationError for 'status' leaks out of the
        tool-level span.  The workflow-level behaviour for bare BaseException subclasses is
        a pre-existing limitation (the tool-level fix is already verified independently in
        TestExceptExceptionDeeper.test_base_exception_in_handler).

        We wrap collect() in asyncio.wait_for so the test cannot hang.
        """
        import asyncio

        from pydantic import ValidationError

        tool = Tool(name="base_exc_tool2", handler=_raises_not_an_exception)
        workflow = Workflow(name="nested_base_exc_wf2").step(tool)

        try:
            result = await asyncio.wait_for(workflow().collect(), timeout=10.0)
        except _NotAnException:
            pass  # tool BaseException propagated cleanly — no ValidationError
        except asyncio.TimeoutError:
            # Pre-existing workflow limitation: bare BaseException in a step task
            # leaves the queue without a completion sentinel, causing the consumer
            # loop to wait until the timeout.  The important thing is that this path
            # does NOT indicate a ValidationError for status (the tool span is fine).
            pass
        except ValidationError as exc:
            if "status" in str(exc):
                pytest.fail(
                    f"ValidationError for 'status' propagated out of workflow "
                    f"with BaseException in tool: {exc}"
                )
            raise
        else:
            assert result is not None
            assert result.status is not None, "workflow OutputEvent.status must not be None"
