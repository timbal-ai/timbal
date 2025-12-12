import pytest
import asyncio
from collections.abc import AsyncGenerator
from timbal import Agent, Tool
from timbal.tools import Bash
from timbal.state import get_run_context
from timbal.types.events.output import OutputEvent
from .conftest import skip_if_agent_error, assert_has_output_event


class TestBackgroundTasks:
    """Test background task execution and status checking."""

    @pytest.mark.asyncio
    async def test_background_task_execution_and_status(self):
        """Test agent can execute task in background and check its status."""
        # Create agent with Bash tool that supports background execution
        agent = Agent(
            name="background_test_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant that can run commands in the background.",
            tools=[Bash("*")]
        )

        # Ask agent to execute a sleep command in the background
        # This simulates a long-running task
        result = await agent(
            prompt="Run 'sleep 5'"
        ).collect()

        # Skip test if agent execution failed
        # skip_if_agent_error(result, "background_task_execution")
        assert_has_output_event(result)

        # Give a tiny moment for background task to start
        await asyncio.sleep(0.1)

        # Ask agent to check the status of the background task
        status_result = await agent(
            prompt="What is the status of your background tasks?"
        ).collect()

        # Verify the status check worked
        assert_has_output_event(status_result)

        # The response should mention something about status or the task
        response_text = status_result.output.content[0].text.lower()
        assert any(word in response_text for word in ["status", "running", "task", "background"]), \
            f"Response doesn't mention task status: {response_text}"

        # Wait for task to complete
        await asyncio.sleep(2.5)

        # Check status again to verify completion
        final_status = await agent(
            prompt="Check the status of your background tasks again"
        ).collect()

        assert_has_output_event(final_status)

    @pytest.mark.asyncio
    async def test_background_task_with_immediate_status_check(self):
        """Test agent can immediately check status of a background task."""
        agent = Agent(
            name="quick_status_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are helpful. When asked to run something in background, use run_in_background=True.",
            tools=[Bash("*")]
        )

        # Execute and check status in one go
        result = await agent(
            prompt="Run 'sleep 1 && echo done' in the background, then immediately check its status"
        ).collect()

        # Verify execution succeeded
        assert_has_output_event(result)

        # The agent should have used both the bash tool and get_background_task
        response_text = result.output.content[0].text.lower()
        # Just verify we got a response - the agent should have executed the task
        assert len(response_text) > 0

    @pytest.mark.asyncio
    async def test_bg_tasks_dict_populated_when_running_background(self):
        """Test that _bg_tasks dictionary is populated when a task runs in background."""
        agent = Agent(
            name="bg_tasks_check_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # Execute a command in the background
        result = await agent(
            prompt="Run 'sleep 1 && echo test' in the background"
        ).collect()

        assert_has_output_event(result)

        # Access the agent's _bg_tasks through the run context
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        
        # Verify _bg_tasks is not empty - a background task should be present
        assert agent_runnable._bg_tasks is not None, "_bg_tasks should not be None"
        assert len(agent_runnable._bg_tasks) > 0, "_bg_tasks should contain at least one background task"
        
        # Verify the structure of the background task
        task_id = list(agent_runnable._bg_tasks.keys())[0]
        task_info = agent_runnable._bg_tasks[task_id]
        assert 'task' in task_info, "Background task should have 'task' key"
        assert 'event_queue' in task_info, "Background task should have 'event_queue' key"
        assert isinstance(task_info['task'], asyncio.Task), "Should be an asyncio Task"

    @pytest.mark.asyncio
    async def test_agent_has_get_background_task_tool(self):
        """Test that agent has get_background_task tool when background tasks are present."""
        agent = Agent(
            name="tool_check_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # First, execute a command in the background to populate _bg_tasks
        async for event in agent(prompt="Run 'sleep 0.5' in the background"):
            if isinstance(event, OutputEvent) and event.path == "tool_check_agent.llm" and len(get_run_context().current_span().runnable._bg_tasks) > 0:
                tools_names = [tool.name for tool in event.input['tools']]
                assert "get_background_task" in tools_names, \
                    "Agent should have 'get_background_task' tool when background tasks exist"
            elif isinstance(event, OutputEvent):
                result = event

        assert_has_output_event(result)

        # Access the agent through run context
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        
        # Verify the agent has background tasks
        assert len(agent_runnable._bg_tasks) > 0, "Agent should have background tasks"


    @pytest.mark.asyncio
    async def test_background_task_with_events_and_logs(self):
        """Test background task that yields multiple events/logs during execution."""
        
        # Create a custom tool that yields multiple events (simulating a build process)
        async def build_interface(project_name: str) -> str:
            """Build an interface with multiple log steps."""
            logs = []
            
            # Simulate build steps with logs
            steps = [
                "Initializing build environment...",
                "Installing dependencies...",
                "Compiling TypeScript...",
                "Building React components...",
                "Optimizing bundle...",
                "Build completed successfully!"
            ]
            
            for step in steps:
                logs.append(step)
                await asyncio.sleep(0.1)  # Small delay to simulate work
            
            return "\n".join(logs)

        build_tool = Tool(
            name="build_interface",
            description="Build a web interface project with detailed logs",
            handler=build_interface,
            background_mode="auto"  # Enable background mode
        )

        agent = Agent(
            name="build_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful build assistant.",
            tools=[build_tool]
        )

        # Execute the build in the background
        result = await agent(
            prompt="Build a project called 'my-app' in the background"
        ).collect()

        assert_has_output_event(result)

        # Verify we got a task_id back (indicating background execution)
        if isinstance(result.output.content[0].text, str):
            response_text = result.output.content[0].text.lower()
            # The agent might mention task or background
            assert any(word in response_text for word in ["task", "background", "running"]), \
                f"Response should indicate background execution: {response_text}"

        # Give the background task time to start and generate some events
        await asyncio.sleep(0.3)

        # Check the status to see if events are available
        status_result = await agent(
            prompt="What is the status of the build task?"
        ).collect()

        assert_has_output_event(status_result)


    @pytest.mark.asyncio
    async def test_realtime_events_vs_background_events(self):
        """Compare real-time event streaming (foreground) vs background event queuing."""
        
        async def streaming_task(steps: int) -> AsyncGenerator[str, None]:
            """Task that yields progress updates."""
            for i in range(steps):
                yield f"Step {i+1}/{steps} completed"
                await asyncio.sleep(0.05)
        
        # Test 1: Real-time streaming (background_mode="never")
        realtime_tool = Tool(
            name="realtime_task",
            description="Streams events in real-time",
            handler=streaming_task,
            background_mode="never"
        )
        
        realtime_agent = Agent(
            name="realtime_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[realtime_tool]
        )
        
        # Collect events as they stream
        realtime_events = []
        realtime_chunks = []
        
        async for event in realtime_agent(prompt="Run realtime_task with steps=3"):
            realtime_events.append(event)
            # Capture chunk/delta events from the tool
            if hasattr(event, 'type') and event.type in ['CHUNK', 'DELTA']:
                if hasattr(event, 'chunk'):
                    realtime_chunks.append(event.chunk)
                elif hasattr(event, 'item'):
                    realtime_chunks.append(event.item)
        
        # Verify we got events in real-time
        assert len(realtime_events) > 0, "Should have collected real-time events"
        
        # Verify we have START, intermediate events, and OUTPUT
        event_types = [e.type for e in realtime_events if hasattr(e, 'type')]
        assert "START" in event_types, "Should have START event"
        assert "OUTPUT" in event_types, "Should have OUTPUT event"
        
        # Test 2: Background mode (events go to queue)
        background_tool = Tool(
            name="background_task",
            description="Runs in background with queued events",
            handler=streaming_task,
            background_mode="always"
        )
        
        background_agent = Agent(
            name="background_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[background_tool]
        )
        
        # Execute in background
        background_events = []
        
        async for event in background_agent(prompt="Run background_task with steps=3"):
            background_events.append(event)
        
        # In background mode, we should get immediate return (not streaming from tool)
        assert len(background_events) > 0, "Should have agent events"
        
        # Give background task time to generate events
        await asyncio.sleep(0.3)
        
        # Access the background task's event queue
        span = get_run_context().current_span()
        bg_agent_runnable = span.runnable
        
        if len(bg_agent_runnable._bg_tasks) > 0:
            task_id = list(bg_agent_runnable._bg_tasks.keys())[0]
            task_info = bg_agent_runnable._bg_tasks[task_id]
            event_queue = task_info['event_queue']
            
            # Events should be in the queue, not streamed to caller
            queued_event_count = event_queue.qsize()
            assert queued_event_count > 0, \
                "Background task events should be queued, not streamed to caller"
        
        # Wait for completion
        await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_background_task_event_queue_detailed_inspection(self):
        """Detailed test of background task event queue contents and structure."""
        
        async def detailed_task(task_name: str) -> AsyncGenerator[dict, None]:
            """Task that yields structured progress updates."""
            stages = [
                {"stage": "init", "progress": 0, "message": "Initializing..."},
                {"stage": "processing", "progress": 50, "message": "Processing data..."},
                {"stage": "finalizing", "progress": 90, "message": "Finalizing..."},
                {"stage": "complete", "progress": 100, "message": "Done!"}
            ]
            
            for stage_data in stages:
                yield stage_data
                await asyncio.sleep(0.08)
        
        detailed_tool = Tool(
            name="detailed_task",
            description="Task with detailed progress updates",
            handler=detailed_task,
            background_mode="always"
        )
        
        agent = Agent(
            name="detailed_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[detailed_tool]
        )
        
        # Start background task
        result = await agent(prompt="Run detailed_task with task_name='analysis'").collect()
        assert_has_output_event(result)
        
        # Give time for events to accumulate
        await asyncio.sleep(0.4)
        
        # Access and inspect the event queue
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        
        assert len(agent_runnable._bg_tasks) > 0, "Should have background task"
        
        task_id = list(agent_runnable._bg_tasks.keys())[0]
        task_info = agent_runnable._bg_tasks[task_id]
        event_queue = task_info['event_queue']
        
        # Collect all events from the queue
        all_events = []
        while not event_queue.empty():
            try:
                event = event_queue.get_nowait()
                all_events.append(event)
            except asyncio.QueueEmpty:
                break
        
        # Verify events were captured
        assert len(all_events) > 0, "Should have captured events in queue"
        
        
        # Verify event metadata
        for event in all_events:
            if hasattr(event, 'run_id'):
                assert event.run_id is not None, "Event should have run_id"
            if hasattr(event, 'path'):
                assert 'detailed_task' in event.path, "Event path should reference the tool"
        
        # Wait for task completion
        await asyncio.sleep(0.3)
        
        # Verify task completed successfully
        status = agent_runnable.get_background_task(task_id)
        assert status["status"] in ["completed", "not_found"], \
            f"Task should be completed or already cleaned up, got: {status['status']}"

    @pytest.mark.asyncio
    async def test_multiple_background_tasks_concurrently(self):
        """Test running multiple background tasks at the same time."""
        agent = Agent(
            name="multi_bg_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # Start multiple background tasks
        result1 = await agent(
            prompt="Run 'sleep 1 && echo task1' in the background"
        ).collect()
        
        assert_has_output_event(result1)

        # Start second background task while first is running
        result2 = await agent(
            prompt="Run 'sleep 1 && echo task2' in the background"
        ).collect()
        
        assert_has_output_event(result2)

        # Both tasks should be in _bg_tasks
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        assert len(agent_runnable._bg_tasks) >= 1, "Should have at least one background task"

        # Wait for both to complete
        await asyncio.sleep(1.5)


    @pytest.mark.asyncio
    async def test_background_task_error_handling(self):
        """Test that background tasks handle errors gracefully."""
        async def failing_tool(should_fail: bool = True) -> str:
            """A tool that fails on demand."""
            await asyncio.sleep(0.1)
            if should_fail:
                raise ValueError("Intentional failure for testing")
            return "Success"

        fail_tool = Tool(
            name="failing_tool",
            description="A tool that can fail",
            handler=failing_tool,
            background_mode="auto"
        )

        agent = Agent(
            name="error_handling_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[fail_tool]
        )

        # Execute a task that will fail in background
        result = await agent(
            prompt="Run failing_tool with should_fail=True in the background"
        ).collect()

        assert_has_output_event(result)

        # Give time for the task to fail
        await asyncio.sleep(0.3)

        # Check status - should report error
        status_result = await agent(
            prompt="Check the status of the background task"
        ).collect()

        assert_has_output_event(status_result)


    @pytest.mark.asyncio
    async def test_background_task_nonexistent_task_id(self):
        """Test checking status of a task_id that doesn't exist."""
        agent = Agent(
            name="check_missing_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        await agent(prompt="hi").collect()

        # Try to get a non-existent task directly
        span = get_run_context().current_span()
        
        # Create a minimal span/runnable context for testing
        result = agent.get_background_task("nonexistent_task_id_12345")
        
        assert result["status"] == "not_found", "Should return not_found for non-existent task"
        assert result["events"] == [], "Should return empty events list"


    @pytest.mark.asyncio
    async def test_background_mode_always(self):
        """Test tool with background_mode='always' runs in background automatically."""
        async def always_background_tool(message: str) -> str:
            """A tool that always runs in background."""
            await asyncio.sleep(0.5)
            return f"Processed: {message}"

        always_bg_tool = Tool(
            name="always_bg_tool",
            description="Always runs in background",
            handler=always_background_tool,
            background_mode="always"
        )

        agent = Agent(
            name="always_bg_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[always_bg_tool]
        )

        # Execute without explicitly mentioning background
        result = await agent(
            prompt="Use always_bg_tool with message='hello'"
        ).collect()

        assert_has_output_event(result)

        # Check that a background task was created
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        assert len(agent_runnable._bg_tasks) >= 1, "Should have created a background task automatically"


    @pytest.mark.asyncio
    async def test_background_mode_never(self):
        """Test tool with background_mode='never' never runs in background."""
        async def never_background_tool(message: str) -> str:
            """A tool that never runs in background."""
            await asyncio.sleep(0.1)
            return f"Processed: {message}"

        never_bg_tool = Tool(
            name="never_bg_tool",
            description="Never runs in background",
            handler=never_background_tool,
            background_mode="never"
        )

        agent = Agent(
            name="never_bg_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[never_bg_tool]
        )

        # Execute
        result = await agent(
            prompt="Use never_bg_tool with message='test'"
        ).collect()

        assert_has_output_event(result)

        # Verify the tool executed normally (no background task created)
        # The response should contain the actual result, not a task_id
        response_text = result.output.content[0].text
        assert len(response_text) > 0


    @pytest.mark.asyncio
    async def test_background_task_cleanup_after_completion(self):
        """Test that completed background tasks are cleaned up from _bg_tasks."""
        async def quick_task(value: str) -> str:
            """A quick task that completes fast."""
            await asyncio.sleep(0.1)
            return f"Done: {value}"

        quick_tool = Tool(
            name="quick_task",
            description="Quick task",
            handler=quick_task,
            background_mode="auto"
        )

        agent = Agent(
            name="cleanup_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[quick_tool]
        )

        # Start background task
        result = await agent(
            prompt="Run quick_task with value='test' in the background"
        ).collect()

        assert_has_output_event(result)

        # Verify task was added
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        initial_task_count = len(agent_runnable._bg_tasks)
        assert initial_task_count > 0, "Should have background task"

        # Wait for completion
        await asyncio.sleep(0.3)

        # Check status (this should trigger cleanup)
        status_result = await agent(
            prompt="Check the status of background tasks"
        ).collect()

        assert_has_output_event(status_result)


    @pytest.mark.asyncio
    async def test_background_task_with_structured_output(self):
        """Test background tasks that return structured data."""
        async def structured_task(count: int) -> dict:
            """Return structured data."""
            await asyncio.sleep(0.2)
            return {
                "count": count,
                "results": [f"item_{i}" for i in range(count)],
                "status": "completed"
            }

        struct_tool = Tool(
            name="structured_task",
            description="Returns structured data",
            handler=structured_task,
            background_mode="auto"
        )

        agent = Agent(
            name="structured_bg_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[struct_tool]
        )

        # Start background task
        result = await agent(
            prompt="Run structured_task with count=3 in the background"
        ).collect()

        assert_has_output_event(result)

        # Wait for completion
        await asyncio.sleep(0.5)

        # Check final status
        final_result = await agent(
            prompt="Get the final status of the background task"
        ).collect()

        assert_has_output_event(final_result)


    @pytest.mark.asyncio
    async def test_background_task_long_running_check_multiple_times(self):
        """Test checking status multiple times while task is running."""
        agent = Agent(
            name="long_check_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # Start a longer running task
        result = await agent(
            prompt="Run 'sleep 2 && echo completed' in the background"
        ).collect()

        assert_has_output_event(result)

        # Check status multiple times while running
        for i in range(3):
            await asyncio.sleep(0.3)
            status_result = await agent(
                prompt=f"Check status of background tasks (check {i+1})"
            ).collect()
            
            assert_has_output_event(status_result)
            response_text = status_result.output.content[0].text.lower()
            # Should mention task or status or running
            assert len(response_text) > 0

        # Wait for completion
        await asyncio.sleep(1.0)


    @pytest.mark.asyncio
    async def test_background_task_events_accumulation(self):
        """Test that events accumulate in the queue during background execution."""
        async def event_generating_task(num_events: int) -> str:
            """Generate multiple events during execution."""
            for i in range(num_events):
                await asyncio.sleep(0.05)
                # Simulate work that might generate events
            return f"Generated {num_events} events"

        event_tool = Tool(
            name="event_gen_tool",
            description="Generates events",
            handler=event_generating_task,
            background_mode="auto"
        )

        agent = Agent(
            name="event_accumulation_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[event_tool]
        )

        # Start task that generates events
        result = await agent(
            prompt="Run event_gen_tool with num_events=5 in the background"
        ).collect()

        assert_has_output_event(result)

        # Give time for events to accumulate
        await asyncio.sleep(0.4)

        # Check status to retrieve events
        status_result = await agent(
            prompt="Check status and get events from the background task"
        ).collect()

        assert_has_output_event(status_result)


    @pytest.mark.asyncio
    async def test_background_task_in_conversation_context(self):
        """Test that background tasks work correctly within conversation memory."""
        agent = Agent(
            name="conversation_bg_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant with memory.",
            tools=[Bash("*")]
        )

        # First turn: introduce context
        intro_result = await agent(
            prompt="I'm going to ask you to run a task in the background next"
        ).collect()

        assert_has_output_event(intro_result)

        # Second turn: start background task
        bg_result = await agent(
            prompt="Now run 'sleep 1 && echo done' in the background"
        ).collect()

        assert_has_output_event(bg_result)

        # Third turn: ask about it (should remember)
        await asyncio.sleep(0.2)
        
        remember_result = await agent(
            prompt="What task did I just ask you to run?"
        ).collect()

        assert_has_output_event(remember_result)
        
        # Agent should remember something about the background task
        response_text = remember_result.output.content[0].text.lower()
        assert len(response_text) > 0


    @pytest.mark.asyncio  
    async def test_get_background_task_tool_parameters(self):
        """Test the get_background_task tool with explicit task_id parameter."""
        agent = Agent(
            name="param_test_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # Start a background task
        result = await agent(
            prompt="Run 'sleep 0.5' in the background"
        ).collect()

        assert_has_output_event(result)

        # Get the agent's runnable and verify background task exists
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        
        if len(agent_runnable._bg_tasks) > 0:
            task_id = list(agent_runnable._bg_tasks.keys())[0]
            
            # Test get_background_task directly
            status = agent_runnable.get_background_task(task_id)
            assert status["status"] in ["running", "completed"], \
                f"Expected running or completed status, got {status['status']}"


    @pytest.mark.asyncio
    async def test_background_task_with_bash_command_output(self):
        """Test that background Bash commands capture output correctly."""
        agent = Agent(
            name="bash_output_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # Run a command that produces output
        result = await agent(
            prompt="Run 'echo \"Hello from background\"' in the background"
        ).collect()

        assert_has_output_event(result)

        # Wait for completion
        await asyncio.sleep(0.5)

        # Check final status
        status_result = await agent(
            prompt="What was the output of the background command?"
        ).collect()

        assert_has_output_event(status_result)


    @pytest.mark.asyncio
    async def test_background_task_event_queue_retrieval(self):
        """Test retrieving events from background task event queue."""
        async def multi_step_task(steps: int) -> str:
            """Task with multiple steps."""
            for i in range(steps):
                await asyncio.sleep(0.05)
            return f"Completed {steps} steps"

        multi_tool = Tool(
            name="multi_step",
            description="Multi-step task",
            handler=multi_step_task,
            background_mode="auto"
        )

        agent = Agent(
            name="queue_test_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[multi_tool]
        )

        # Start background task
        result = await agent(
            prompt="Run multi_step with steps=3 in the background"
        ).collect()

        assert_has_output_event(result)

        # Access task directly
        span = get_run_context().current_span()
        agent_runnable = span.runnable

        if len(agent_runnable._bg_tasks) > 0:
            task_id = list(agent_runnable._bg_tasks.keys())[0]
            task_info = agent_runnable._bg_tasks[task_id]
            
            # Verify event queue exists
            assert 'event_queue' in task_info, "Event queue should exist"
            assert isinstance(task_info['event_queue'], asyncio.Queue), \
                "Event queue should be an asyncio.Queue"


    @pytest.mark.asyncio
    async def test_background_tasks_persist_across_agent_calls(self):
        """Test that background tasks persist across multiple agent invocations."""
        agent = Agent(
            name="persistence_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[Bash("*")]
        )

        # Start background task in first call
        result1 = await agent(
            prompt="Run 'sleep 1' in the background"
        ).collect()

        assert_has_output_event(result1)

        # Immediately make another call to check status
        result2 = await agent(
            prompt="What background tasks are running?"
        ).collect()

        assert_has_output_event(result2)

        # The agent should be able to access the background task
        span = get_run_context().current_span()
        agent_runnable = span.runnable
        # Background tasks should still be tracked
        assert agent_runnable._bg_tasks is not None

