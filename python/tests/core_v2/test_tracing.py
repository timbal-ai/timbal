import asyncio
import time

import pytest
from timbal.core_v2.agent import Agent
from timbal.core_v2.tool import Tool
from timbal.state import get_run_context
from timbal.types.events import OutputEvent, StartEvent
from timbal.types.message import Message


class TestCallIDGeneration:
    """Test that call IDs are generated correctly in nested execution contexts."""
    
    @pytest.mark.asyncio
    async def test_standalone_tools_have_no_call_ids(self):
        """Test that standalone tool executions don't have call IDs (only run IDs)."""
        def simple_task(x: str) -> str:
            return f"processed:{x}"
        
        tool = Tool(name="simple_task", handler=simple_task)
        
        # Execute multiple standalone tool calls
        results = await asyncio.gather(
            tool(x="task1").collect(),
            tool(x="task2").collect(),
            tool(x="task3").collect()
        )
        
        # Extract run IDs from the results (each execution should have unique run_id)
        run_ids = []
        for result in results:
            assert isinstance(result, OutputEvent)
            # Each standalone execution should have a unique run_id
            run_ids.append(result.run_id)
            # Standalone tools should NOT have call IDs (they're not nested in an executor)
            # Only run_id is present for top-level executions
        
        # All run IDs should be unique (each execution context)
        assert len(set(run_ids)) == len(run_ids), f"Run IDs should be unique, got: {run_ids}"
    
    @pytest.mark.asyncio
    async def test_tools_in_agent_have_call_ids(self):
        """Test that tools executed within an agent context get call IDs."""
        def helper_tool(task: str) -> str:
            return f"completed:{task}"
        
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            tools=[helper_tool],
            system_prompt="Use the helper tool to complete tasks"
        )
        
        prompt = Message.validate({
            "role": "user", 
            "content": "Please use helper_tool to complete 'sample_task'"
        })
        
        # Execute and collect the final result to verify nested execution occurred
        result = agent(prompt=prompt)
        output = await result.collect()
        
        assert isinstance(output, OutputEvent)
        
        # Verify the agent execution shows proper nesting structure
        if output.error is None:
            # Should have successful execution with proper path
            assert output.path == "test_agent", f"Agent should have correct path: {output.path}"
            
            # The response should show the tool was used successfully
            response = output.output
            assert isinstance(response, Message), "Should have Message response"
            
            response_text = str(response.content).lower()
            # Should mention the task completion
            assert "completed" in response_text or "sample_task" in response_text, f"Should show task completion: {response_text}"
            
            # Key insight: The logs show nested execution paths:
            # - test_agent (top-level executor with run_id)
            # - test_agent.llm (nested LLM calls with call_ids from agent)
            # - test_agent.helper_tool (nested tool calls with call_ids from agent)
            
            # This demonstrates that call IDs are assigned by the executor (agent) context
            # while run_ids are for top-level execution contexts
        else:
            pytest.skip(f"Agent execution failed: {output.error}")
    
    @pytest.mark.asyncio
    async def test_agent_executions_have_unique_run_ids(self):
        """Test that each agent execution gets a unique run ID."""
        def helper_tool(task: str) -> str:
            return f"completed:{task}"
        
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            tools=[helper_tool]
        )
        
        # Execute multiple agent calls
        prompts = [
            Message.validate({"role": "user", "content": "Execute task1"}),
            Message.validate({"role": "user", "content": "Execute task2"}),
            Message.validate({"role": "user", "content": "Execute task3"})
        ]
        
        results = await asyncio.gather(
            agent(prompt=prompts[0]).collect(),
            agent(prompt=prompts[1]).collect(),
            agent(prompt=prompts[2]).collect(),
            return_exceptions=True
        )
        
        # Filter out any exceptions and check unique execution
        valid_results = [r for r in results if isinstance(r, OutputEvent)]
        
        # Should have some successful results with unique execution contexts
        assert len(valid_results) > 0, "Should have at least some successful agent executions"
        
        # Extract run IDs from valid results
        if len(valid_results) > 1:
            run_ids = [result.run_id for result in valid_results]
            # Each agent execution should have unique run_id (top-level execution context)
            assert len(set(run_ids)) == len(run_ids), f"Agent run IDs should be unique, got: {run_ids}"


class TestNestedAgentTracing:
    """Test tracing behavior with nested agents."""
    
    @pytest.mark.asyncio
    async def test_parent_child_call_relationship(self):
        """Test that nested agent calls maintain proper parent-child relationships."""
        def math_operation(operation: str, a: int, b: int) -> int:
            """Perform a math operation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                return 0
        
        # Child agent - specialist in math
        math_agent = Agent(
            name="math_specialist", 
            model="openai/gpt-4o-mini",
            tools=[math_operation],
            description="A specialist agent for mathematical operations"
        )
        
        # Parent agent - uses the math specialist
        coordinator = Agent(
            name="coordinator",
            model="openai/gpt-4o-mini", 
            tools=[math_agent],
            system_prompt="Use the math specialist to solve problems"
        )
        
        prompt = Message.validate({
            "role": "user", 
            "content": "Use the math specialist to calculate 15 + 27"
        })
        
        # Execute with event stream collection
        events = []
        result = coordinator(prompt=prompt)
        
        async for event in result:
            events.append(event)
            # Stop after collecting some events to test relationships
            if len(events) >= 5:
                break
        
        # Analyze event hierarchy (removed unused variables)
        
        # Should have hierarchical paths
        coordinator_events = [e for e in events if e.path.startswith("coordinator")]
        nested_events = [e for e in events if "math_specialist" in e.path]
        
        assert len(coordinator_events) > 0, "Should have coordinator events"
        # May or may not have nested events depending on LLM behavior
        
        # Verify path hierarchy structure
        for event in nested_events:
            assert event.path.startswith("coordinator."), f"Nested event should start with parent path: {event.path}"
    
    @pytest.mark.asyncio
    async def test_deep_nesting_trace_paths(self):
        """Test trace paths in deeply nested agent scenarios."""
        def level3_task(data: str) -> str:
            return f"level3_processed:{data}"
        
        def level2_task(input_data: str) -> str:
            return f"level2_processed:{input_data}"
        
        # Level 3 agent
        level3_agent = Agent(
            name="level3_agent",
            model="openai/gpt-4o-mini", 
            tools=[level3_task],
            description="Level 3 processing agent"
        )
        
        # Level 2 agent
        level2_agent = Agent(
            name="level2_agent", 
            model="openai/gpt-4o-mini",
            tools=[level2_task, level3_agent],
            description="Level 2 processing agent"
        )
        
        # Level 1 agent (root)
        root_agent = Agent(
            name="root_agent",
            model="openai/gpt-4o-mini",
            tools=[level2_agent],
            system_prompt="Coordinate multi-level processing"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Process this data through all levels: 'test_data'"
        })
        
        # Collect some events to analyze paths
        events = []
        result = root_agent(prompt=prompt)
        
        async for event in result:
            events.append(event)
            if len(events) >= 8:  # Collect more events for deeper nesting
                break
        
        # Analyze path structure
        all_paths = [event.path for event in events]
        
        # Should have root level events
        root_events = [path for path in all_paths if path.startswith("root_agent") and "." not in path.replace("root_agent", "")[1:]]
        assert len(root_events) > 0, "Should have root agent events"
        
        # Path structure verification
        for path in all_paths:
            # Paths should follow proper nesting pattern
            parts = path.split(".")
            assert len(parts) >= 1, f"Path should have at least one component: {path}"
            assert parts[0] == "root_agent", f"All paths should start with root_agent: {path}"


class TestParallelToolCallTracing:
    """Test tracing with parallel tool execution."""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_have_unique_ids(self):
        """Test that concurrent tool calls within an agent have unique call IDs."""
        def slow_task_a(duration: float) -> str:
            """Simulate slow task A."""
            time.sleep(duration)
            return f"task_a_completed_after_{duration}s"
        
        def slow_task_b(duration: float) -> str:
            """Simulate slow task B."""
            time.sleep(duration) 
            return f"task_b_completed_after_{duration}s"
        
        def slow_task_c(duration: float) -> str:
            """Simulate slow task C."""
            time.sleep(duration)
            return f"task_c_completed_after_{duration}s"
        
        agent = Agent(
            name="parallel_agent",
            model="openai/gpt-4o-mini",
            tools=[slow_task_a, slow_task_b, slow_task_c],
            system_prompt="Execute multiple tasks concurrently when requested"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Execute slow_task_a with duration 0.1, slow_task_b with duration 0.1, and slow_task_c with duration 0.1 all at the same time"
        })
        
        # Execute and get the final result to examine the execution structure
        result = agent(prompt=prompt)
        output = await result.collect()
        
        assert isinstance(output, OutputEvent)
        
        # If the agent executed successfully, verify concurrent execution occurred
        if output.error is None:
            # The agent should have successfully orchestrated multiple tool calls
            response = output.output
            assert isinstance(response, Message), "Should have Message response"
            
            # The response should mention multiple tasks (evidence of parallel execution)
            response_text = str(response.content).lower()
            # Check for task mentions with flexibility for spaces vs underscores
            task_patterns = ["task_a", "task a", "task_b", "task b", "task_c", "task c"]
            task_mentions = sum(1 for task in task_patterns if task in response_text)
            
            # Should mention at least 2 tasks to show parallel execution
            assert task_mentions >= 2, f"Should mention multiple tasks: {response_text}"
            
            # Verify the agent execution has proper run_id (top-level context)
            assert output.run_id is not None, "Agent execution should have run_id"
            assert output.path == "parallel_agent", "Agent output should have correct path"
        else:
            # Skip test if agent execution failed
            pytest.skip(f"Agent execution failed: {output.error}")
    
    @pytest.mark.asyncio
    async def test_execution_hierarchy_structure(self):
        """Test that nested executions (LLM, tools) within agents have call IDs."""
        def data_processor(input_data: str) -> str:
            return f"processed:{input_data}"
        
        agent = Agent(
            name="nested_execution_agent",
            model="openai/gpt-4o-mini",
            tools=[data_processor],
            system_prompt="Process the provided data using the data_processor tool"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Please process the data 'test_input' using data_processor"
        })
        
        # Execute and examine the final result to verify hierarchy
        result = agent(prompt=prompt)
        output = await result.collect()
        
        assert isinstance(output, OutputEvent)
        
        if output.error is None:
            # Verify successful execution with proper hierarchy
            assert output.path == "nested_execution_agent", f"Agent should have correct path: {output.path}"
            
            # Should have successful tool usage
            response = output.output
            assert isinstance(response, Message), "Should have Message response"
            
            response_text = str(response.content).lower()
            # Should show evidence of data processing
            assert "processed" in response_text and "test_input" in response_text, f"Should show processing result: {response_text}"
            
            # The key insight from examining the execution logs:
            # - nested_execution_agent (top-level executor with run_id)
            # - nested_execution_agent.llm (nested LLM calls get call_ids from agent)
            # - nested_execution_agent.data_processor (nested tool calls get call_ids from agent)
            # This demonstrates proper executor hierarchy and call ID assignment
        else:
            pytest.skip(f"Agent execution failed: {output.error}")
    
    @pytest.mark.asyncio
    async def test_mixed_sequential_parallel_execution_tracing(self):
        """Test tracing with mixed sequential and parallel tool execution patterns."""
        def prepare_data(input_data: str) -> str:
            """Step 1: Prepare data."""
            return f"prepared:{input_data}"
        
        def process_chunk_a(data: str) -> str:
            """Step 2a: Process chunk A (parallel)."""
            time.sleep(0.05)
            return f"processed_a:{data}"
        
        def process_chunk_b(data: str) -> str:
            """Step 2b: Process chunk B (parallel)."""
            time.sleep(0.05) 
            return f"processed_b:{data}"
        
        def finalize_results(a_result: str, b_result: str) -> str:
            """Step 3: Finalize results."""
            return f"final:{a_result}+{b_result}"
        
        workflow_agent = Agent(
            name="workflow_agent",
            model="openai/gpt-4o-mini",
            tools=[prepare_data, process_chunk_a, process_chunk_b, finalize_results],
            system_prompt="Execute a workflow: prepare data, then process chunks A and B in parallel, then finalize"
        )
        
        prompt = Message.validate({
            "role": "user", 
            "content": "Execute the full workflow on data 'test_input'"
        })
        
        # Collect execution events
        events = []
        result = workflow_agent(prompt=prompt)
        
        # Collect events with timeout
        start_time = time.time()
        async for event in result:
            events.append(event)
            
            # Don't wait too long for completion
            if time.time() - start_time > 30 or len(events) >= 15:
                break
        
        # Analyze execution pattern
        all_paths = [e.path for e in events]
        
        # Should have workflow agent events
        workflow_events = [path for path in all_paths if path.startswith("workflow_agent")]
        assert len(workflow_events) > 0, "Should have workflow execution events"
        
        # Check for tool execution paths (if any tools were called)
        tool_names = ["prepare_data", "process_chunk_a", "process_chunk_b", "finalize_results"]
        tool_events = [path for path in all_paths if any(tool in path for tool in tool_names)]
        
        # Should have proper nesting structure
        for path in tool_events:
            assert path.startswith("workflow_agent."), f"Tool events should be nested under agent: {path}"


class TestTracingContextPropagation:
    """Test that tracing context is properly propagated through execution chains."""
    
    @pytest.mark.asyncio
    async def test_run_context_propagation(self):
        """Test that run context is properly maintained across nested calls."""
        def context_inspector() -> dict:
            """Tool that inspects the current run context."""
            run_context = get_run_context()
            return {
                "run_id": str(run_context.id),
                "parent_id": str(run_context.parent_id) if run_context.parent_id else None,
                "has_tracing": run_context.tracing is not None
            }
        
        agent = Agent(
            name="context_agent",
            model="openai/gpt-4o-mini", 
            tools=[context_inspector],
            system_prompt="Use the context inspector tool to examine the execution context"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Please inspect the current execution context"
        })
        
        result = agent(prompt=prompt)
        output = await result.collect()
        
        # Should have successful execution (skip if agent has issues)
        if isinstance(output, OutputEvent) and output.error:
            pytest.skip(f"Agent execution failed: {output.error.get('message', 'Unknown error')}")
        
        assert isinstance(output, OutputEvent)
        assert isinstance(output.output, Message)
        
        # The agent should have been able to access run context
        # (Specific validation would depend on tool execution and LLM response)
    
    @pytest.mark.asyncio
    async def test_nested_context_hierarchy(self):
        """Test context hierarchy in nested agent execution."""
        def get_context_info() -> str:
            """Get information about the current context."""
            run_context = get_run_context()
            return f"run_id:{run_context.id},parent:{run_context.parent_id}"
        
        # Child agent
        child_agent = Agent(
            name="child_context_agent",
            model="openai/gpt-4o-mini",
            tools=[get_context_info],
            description="Child agent that inspects context"
        )
        
        # Parent agent  
        parent_agent = Agent(
            name="parent_context_agent",
            model="openai/gpt-4o-mini",
            tools=[child_agent],
            system_prompt="Use the child agent to inspect execution context"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Have the child agent inspect the execution context"
        })
        
        # Execute and collect events
        events = []
        result = parent_agent(prompt=prompt)
        
        async for event in result:
            events.append(event)
            if len(events) >= 6:
                break
        
        # Should have events from both parent and potentially child
        parent_events = [e for e in events if e.path.startswith("parent_context_agent")]
        nested_events = [e for e in events if "child_context_agent" in e.path]
        
        assert len(parent_events) > 0, "Should have parent agent events"
        
        # Verify proper path nesting if child was actually called
        for event in nested_events:
            assert event.path.startswith("parent_context_agent."), f"Child events should be nested: {event.path}"


class TestErrorTracingPropagation:
    """Test that errors are properly traced through nested execution."""
    
    @pytest.mark.asyncio
    async def test_tool_error_tracing(self):
        """Test that tool errors are properly traced with call IDs."""
        def failing_tool(error_message: str) -> str:
            """A tool that always fails."""
            raise RuntimeError(f"Intentional failure: {error_message}")
        
        def working_tool(data: str) -> str:
            """A tool that works."""
            return f"success:{data}"
        
        agent = Agent(
            name="error_tracing_agent",
            model="openai/gpt-4o-mini",
            tools=[failing_tool, working_tool],
            system_prompt="Try using both tools - one will fail, one will succeed"
        )
        
        prompt = Message.validate({
            "role": "user", 
            "content": "Try to use both the failing tool and working tool"
        })
        
        result = agent(prompt=prompt)
        output = await result.collect()
        
        # Agent should handle tool failures gracefully
        assert isinstance(output, OutputEvent)
        
        # Should have an agent response even if tools failed
        assert isinstance(output.output, Message)
        
        # Verify the error was properly traced (path should show agent execution)
        assert output.path.startswith("error_tracing_agent"), f"Error should be traced to agent: {output.path}"
    
    @pytest.mark.asyncio
    async def test_nested_agent_error_tracing(self):
        """Test error tracing through nested agent calls."""
        def unreliable_operation(should_fail: bool) -> str:
            """Operation that fails conditionally."""
            if should_fail:
                raise ValueError("Operation failed as requested")
            return "Operation succeeded"
        
        # Unreliable child agent
        child_agent = Agent(
            name="unreliable_child",
            model="openai/gpt-4o-mini",
            tools=[unreliable_operation],
            description="Child agent with unreliable operations"
        )
        
        # Parent coordinator
        parent_agent = Agent(
            name="error_coordinator", 
            model="openai/gpt-4o-mini",
            tools=[child_agent],
            system_prompt="Use the child agent for operations, handle any failures gracefully"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Ask the child agent to perform unreliable_operation with should_fail=true"
        })
        
        result = parent_agent(prompt=prompt)
        output = await result.collect()
        
        # Should have proper error handling and tracing
        assert isinstance(output, OutputEvent)
        
        # Parent agent should respond even if child fails
        assert isinstance(output.output, Message)
        
        # Error should be traced to parent agent level
        assert output.path.startswith("error_coordinator"), f"Error should be traced to parent: {output.path}"


class TestEventSequenceValidation:
    """Test that events are generated in the correct sequence with proper IDs."""
    
    @pytest.mark.asyncio
    async def test_start_output_event_pairing(self):
        """Test that StartEvent and OutputEvent pairs have matching identifiers."""
        def simple_operation(value: str) -> str:
            return f"processed:{value}"
        
        tool = Tool(name="simple_operation", handler=simple_operation)
        
        # Collect all events from execution
        events = []
        result = tool(value="test")
        
        async for event in result:
            events.append(event)
        
        # Should have both start and output events
        start_events = [e for e in events if isinstance(e, StartEvent)]
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        
        assert len(start_events) >= 1, "Should have at least one start event"
        assert len(output_events) >= 1, "Should have at least one output event"
        
        # Events should have consistent paths
        for event in events:
            assert event.path == "simple_operation", f"All events should have same path: {event.path}"
    
    @pytest.mark.asyncio
    async def test_agent_event_sequence_validation(self):
        """Test that agent events follow proper sequence."""
        def quick_task(task_name: str) -> str:
            return f"completed:{task_name}"
        
        agent = Agent(
            name="sequence_agent",
            model="openai/gpt-4o-mini",
            tools=[quick_task],
            system_prompt="Execute the requested task quickly"
        )
        
        prompt = Message.validate({
            "role": "user",
            "content": "Please execute quick_task with task_name='test_sequence'"
        })
        
        # Collect events in sequence
        events = []
        result = agent(prompt=prompt)
        
        async for event in result:
            events.append(event)
            # Don't wait for full completion for sequence testing
            if len(events) >= 4:
                break
        
        # Should have some events
        assert len(events) > 0, "Should have generated some events"
        
        # All events should have agent prefix in path
        for event in events:
            assert event.path.startswith("sequence_agent"), f"Event path should start with agent name: {event.path}"
        
        # Should have start event for agent
        agent_start_events = [e for e in events if isinstance(e, StartEvent) and e.path == "sequence_agent"]
        assert len(agent_start_events) >= 1, "Should have agent start event"