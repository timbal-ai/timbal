import asyncio
import time
from collections.abc import AsyncGenerator, Generator

import pytest
from timbal import Agent, Tool, Workflow
from timbal.types.events import ChunkEvent, OutputEvent, StartEvent
from timbal.errors import InterruptError
from timbal.types.message import Message

from .conftest import (
    assert_has_output_event,
    assert_no_errors,
    Timer,
)

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
        name="agent_with_tools",
        model="openai/gpt-4o-mini",
        tools=[get_weather, get_location, get_temperature]
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
        agent = Agent(
            name="simple_agent",
            model="openai/gpt-4o-mini"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Tell me a very long story about space exploration"
        })
        
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
        
        agent = Agent(
            name="tool_agent",
            model="openai/gpt-4o-mini",
            tools=[slow_tool]
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Use the slow_tool please"
        })
        
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
        agent = Agent(
            name="reusable_agent",
            model="openai/gpt-4o-mini"
        )
        
        # First call - interrupt it
        prompt1 = Message.validate({
            "role": "user",
            "content": "Tell me a long story"
        })
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        await asyncio.sleep(2)
        task1.cancel()
        result1 = await task1
        
        assert result1.status.code == "cancelled"
        
        # Second call - should work normally
        prompt2 = Message.validate({
            "role": "user",
            "content": "What is your name?"
        })
        result2 = await agent(prompt=prompt2).collect()
        
        assert_has_output_event(result2)
        assert result2.status.code == "success"
        assert isinstance(result2.output, Message)
    
    @pytest.mark.asyncio
    async def test_agent_interruption_with_multiple_tools(self, agent_with_long_tools):
        """Test agent interruption when multiple tools are available."""
        prompt = Message.validate({
            "role": "user",
            "content": "Get the weather, location, and temperature"
        })
        
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
        
        agent = Agent(
            name="mixed_speed_agent",
            model="openai/gpt-4o-mini",
            tools=[quick_tool, slow_tool]
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Use both quick_tool and slow_tool"
        })
        
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
        
        workflow = (Workflow(name="sequential_workflow")
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
        workflow = (Workflow(name="parallel_workflow")
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
        
        workflow = (Workflow(name="mixed_workflow")
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
        
        workflow = (Workflow(name="mixed_speed_workflow")
            .step(fast_step)
            .step(slow_step, depends_on=["fast_step"])
        )
        
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
        workflow = (Workflow(name="conditional_workflow")
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
        
        workflow = (Workflow(name="mixed_step_workflow")
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
        
        workflow = (Workflow(name="tool_interrupt_workflow")
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
        
        workflow = (Workflow(name="agent_interrupt_workflow")
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
        assert output_event.status.code == "cancelled", \
            "Status code should be 'interrupted'"
        
        # 2. STATUS DETAILS - Check reason and message
        assert output_event.status.reason == "interrupted", \
            "Status reason should indicate interruption"
        
        # 3. PARTIAL OUTPUT PRESERVATION - Chunks collected before interruption
        # For generators that yield strings, the collector concatenates them
        assert output_event.output is not None, \
            "Partial output should be preserved for generators"
        # Output should be a string with partial results (concatenated chunks)
        assert isinstance(output_event.output, str), \
            "Output should be a string (concatenated chunks)"
        assert len(output_event.output) > 0, \
            "Should have collected some chunks before interruption"
        # Verify it contains some chunks (e.g., chunk_0, chunk_1, etc.)
        assert "chunk_" in output_event.output, \
            "Output should contain chunk data"
        
        # 4. ERROR FIELD - Should be None (interruption is not an error)
        assert output_event.error is None, \
            "Error field should be None for interruptions"
        
        # 5. INPUT PRESERVATION - Original input should be available
        assert output_event.input is not None, \
            "Input parameters should be preserved"
        assert output_event.input.get("count") == 100, \
            "Input values should match what was passed"
        
        # 6. TIMING INFORMATION - Verify timestamps
        assert output_event.t0 > 0, \
            "Start time should be recorded"
        assert output_event.t1 > output_event.t0, \
            "End time should be after start time"
        duration_ms = output_event.t1 - output_event.t0
        assert duration_ms < 5000, \
            f"Execution should stop quickly on interruption, took {duration_ms}ms"
        
        # 7. INTERRUPTION TIMING - Should be much faster than full execution
        assert timer.elapsed < 5, \
            f"Interruption should be fast, took {timer.elapsed}s (would take 10s+ to complete)"
        
        # 8. METADATA AND USAGE - Should be preserved if present
        assert isinstance(output_event.metadata, dict), \
            "Metadata should be accessible"
        assert isinstance(output_event.usage, dict), \
            "Usage information should be accessible"
        
        # 9. RUN CONTEXT - Verify run and call IDs are present
        assert output_event.run_id is not None, \
            "Run ID should be set"
        assert output_event.call_id is not None, \
            "Call ID should be set"
        assert output_event.path == "yielding_tool", \
            "Path should match the tool name"
        
        # 10. REUSABILITY - Tool should be usable after interruption
        result2 = await tool(count=1).collect()
        assert result2.status.code == "success", \
            "Tool should be reusable after interruption"
    
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
        
        workflow = (Workflow(name="comprehensive_workflow")
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
        assert result.status.code == "cancelled", \
            "Workflow should be marked as interrupted"
        
        # 2. Partial execution tracking - verify which steps ran
        assert "step_1" in executed_steps, \
            "First step should have completed"
        assert "step_2" in executed_steps, \
            "Second step should have started"
        assert "step_3" not in executed_steps, \
            "Third step should never have started (depends on interrupted step)"
        
        # 3. Partial results preservation
        assert result.output is not None, \
            "Partial results from completed steps should be preserved"
        # The output should be an OutputEvent from the completed step
        assert hasattr(result.output, 'output'), \
            "Output should contain results from completed steps"
        
        # 4. No error despite interruption
        assert result.error is None, \
            "Interruption should not be treated as an error"
        
        # 5. Timing - should complete quickly on interruption
        assert timer.elapsed < 5, \
            f"Workflow interruption should be quick, took {timer.elapsed}s"
        
        # 6. Run identifiers should be set
        assert result.run_id is not None, \
            "Run ID should be set"
        assert result.call_id is not None, \
            "Call ID should be set"
        assert result.path == "comprehensive_workflow", \
            "Path should match workflow name"
        
        # 7. Timestamps should be recorded
        assert result.t0 > 0, \
            "Start time should be recorded"
        assert result.t1 > result.t0, \
            "End time should be after start time"
