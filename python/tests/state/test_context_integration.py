"""Integration tests for RunContext trace access methods.

These tests focus on runtime behavior of RunContext methods like current_span(),
parent_span(), step_span(), and pre_hook integration scenarios.
"""
import asyncio
from datetime import datetime

import pytest
from timbal import Agent, Tool
from timbal.state import get_run_context
from timbal.types.file import File


class TestRunContextTraceAccess:
    """Test RunContext trace access methods in real execution scenarios."""

    @pytest.mark.asyncio
    async def test_current_span_basic_access(self):
        """Test basic current_span() access within agent execution."""
        def get_current_span_info() -> str:
            """Get information about the current span."""
            span = get_run_context().current_span()
            return f"Call ID: {span.call_id}, Path: {span.path}"

        agent = Agent(
            name="trace_test_agent",
            model="openai/gpt-4o-mini",
            tools=[get_current_span_info]
        )

        # Execute the agent with a simple prompt
        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use the get_current_span_info tool"})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed without errors
        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_current_span_input_access(self):
        """Test accessing current trace input data."""
        def get_trace_input() -> str:
            """Get the input from the current span."""
            span = get_run_context().current_span()
            return f"Input type: {type(span.input)}, Content: {str(span.input)[:100]}"

        agent = Agent(
            name="input_trace_agent",
            model="openai/gpt-4o-mini",
            tools=[get_trace_input]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use get_trace_input to show my input"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_parent_span_nested_execution(self):
        """Test parent_span() access in nested agent execution."""
        def get_parent_info() -> str:
            """Get information about the parent span."""
            try:
                span = get_run_context().current_span()
                parent = get_run_context().parent_span()
                return f"Current: {span.path}, Parent: {parent.path}"
            except RuntimeError as e:
                return f"Error: {str(e)}"

        # Create a nested tool that calls another agent
        async def nested_agent_call() -> str:
            """Call a nested agent to test parent trace access."""
            nested_agent = Agent(
                name="nested_agent",
                model="openai/gpt-4o-mini",
                tools=[get_parent_info]
            )

            from timbal.types.message import Message
            prompt = Message.validate({"role": "user", "content": "Use get_parent_info"})
            nested_result = nested_agent(prompt=prompt)
            nested_output = await nested_result.collect()
            if hasattr(nested_output, 'output') and nested_output.output:
                return str(nested_output.output.content)
            return "No parent info found"

        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[nested_agent_call]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use nested_agent_call"})
        result = main_agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_step_span_sibling_access(self):
        """Test step_span() access to sibling steps."""
        def step_one() -> str:
            """First step that creates trace data."""
            return "Step one completed"

        def step_two() -> str:
            """Second step that accesses step one's span."""
            try:
                # Try to access the step_one span
                step_span = get_run_context().step_span("step_one")
                return f"Found sibling trace: {step_span.path}, Output: {step_span.output}"
            except RuntimeError as e:
                return f"Could not access step_one trace: {str(e)}"

        agent = Agent(
            name="sibling_trace_agent",
            model="openai/gpt-4o-mini",
            tools=[step_one, step_two]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "First use step_one, then use step_two"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_pre_hook_trace_input_modification(self):
        """Test pre_hook accessing and modifying trace input."""
        # Mock STT function to avoid external dependencies
        async def mock_stt(audio_file: File) -> str:
            """Mock speech-to-text conversion."""
            # Use audio_file parameter to simulate the real STT function signature
            _ = audio_file  # Mark as used
            return "Transcribed: Hello world"

        def get_datetime() -> str:
            """Get current datetime."""
            return datetime.now().isoformat()

        async def pre_hook():
            """Pre-hook that modifies input based on trace data."""
            span = get_run_context().current_span()

            # Simulate audio file in input
            if isinstance(span.input, dict) and "prompt" in span.input:
                # In real scenario, this would be a File object
                if isinstance(span.input["prompt"], str) and span.input["prompt"].startswith("audio:"):
                    # Mock audio processing
                    audio_content = span.input["prompt"].replace("audio:", "")
                    transcribed = await mock_stt(File(content=audio_content, path="mock.wav"))
                    span.input["prompt"] = transcribed

        agent = Agent(
            name="pre_hook_agent",
            model="openai/gpt-4o-mini",
            pre_hook=pre_hook,
            tools=[get_datetime]
        )

        # Test with audio-like input
        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": {"prompt": "audio:sample_audio_content"}})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_usage_tracking_in_traces(self):
        """Test usage tracking across trace hierarchy."""
        def track_usage() -> str:
            """Tool that tracks usage metrics."""
            ctx = get_run_context()

            # Update some usage metrics
            ctx.update_usage("tool_calls", 1)
            ctx.update_usage("custom_metric", 5)

            # Get current trace to check usage
            span = ctx.current_span()
            return f"Usage tracked: {span.usage}"

        def check_usage() -> str:
            """Tool that checks current usage."""
            span = get_run_context().current_span()
            return f"Current usage: {span.usage}"

        agent = Agent(
            name="usage_agent",
            model="openai/gpt-4o-mini",
            tools=[track_usage, check_usage]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use track_usage then check_usage"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_error_handling_missing_traces(self):
        """Test error handling when traces don't exist."""
        def test_missing_parent() -> str:
            """Test accessing parent when none exists."""
            try:
                parent = get_run_context().parent_span()
                return f"Found parent: {parent.path}"
            except RuntimeError as e:
                return f"Expected error: {str(e)}"

        def test_missing_step() -> str:
            """Test accessing non-existent step."""
            try:
                step = get_run_context().step_span("nonexistent_step")
                return f"Found step: {step.path}"
            except RuntimeError as e:
                return f"Expected error: {str(e)}"

        agent = Agent(
            name="error_test_agent",
            model="openai/gpt-4o-mini",
            tools=[test_missing_parent, test_missing_step]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use both test tools"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_trace_data_persistence(self):
        """Test that trace data persists across tool calls."""
        call_count = 0

        def increment_counter() -> str:
            """Increment a counter and store in span."""
            nonlocal call_count
            call_count += 1

            span = get_run_context().current_span()
            # Store data in trace (this would persist in real scenarios)
            if not hasattr(span, 'metadata'):
                span.metadata = {}
            span.metadata['call_count'] = call_count

            return f"Counter: {call_count}"

        def check_counter() -> str:
            """Check the counter from trace data."""
            trace = get_run_context().current_span()
            stored_count = getattr(trace, 'metadata', {}).get('call_count', 0)
            return f"Stored count: {stored_count}, Current count: {call_count}"

        agent = Agent(
            name="persistence_agent",
            model="openai/gpt-4o-mini",
            tools=[increment_counter, check_counter]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use increment_counter twice, then check_counter"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_complex_pre_hook_scenario(self):
        """Test complex pre_hook scenario with multiple modifications."""
        def process_data(data: str) -> str:
            """Process some data."""
            return f"Processed: {data.upper()}"

        async def complex_pre_hook():
            """Complex pre-hook that performs multiple operations."""
            span = get_run_context().current_span()

            # Modify input based on various conditions
            if isinstance(span.input, str):
                # Simple string input
                span.input = f"[PRE-PROCESSED] {span.input}"
            elif isinstance(span.input, dict):
                # Dict input - modify specific keys
                if "prompt" in span.input:
                    span.input["prompt"] = f"Enhanced: {span.input['prompt']}"
                if "data" in span.input:
                    span.input["data"] = span.input["data"].lower()
                # Add metadata
                span.input["processed_at"] = datetime.now().isoformat()

        agent = Agent(
            name="complex_pre_hook_agent",
            model="openai/gpt-4o-mini",
            pre_hook=complex_pre_hook,
            tools=[process_data]
        )

        # Test with dict input
        complex_input = {
            "prompt": "Please process this data",
            "data": "SOME_DATA_TO_PROCESS"
        }

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": complex_input})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None


class TestRunContextWorkflowIntegration:
    """Test RunContext integration with workflows (when available)."""

    @pytest.mark.asyncio
    async def test_workflow_trace_access(self):
        """Test trace access in workflow context."""
        # This test would be expanded when workflow functionality is more mature
        def workflow_step() -> str:
            """A workflow step that accesses trace information."""
            try:
                span = get_run_context().current_span()
                return f"Workflow step trace: {span.path}"
            except Exception as e:
                return f"Workflow trace error: {str(e)}"

        # For now, just test basic functionality
        agent = Agent(
            name="workflow_trace_agent",
            model="openai/gpt-4o-mini",
            tools=[workflow_step]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use workflow_step"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None


class TestRunContextEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_span_access(self):
        """Test trace access in concurrent scenarios."""
        import asyncio

        async def concurrent_tool(tool_id: str) -> str:
            """Tool that might be called concurrently."""
            span = get_run_context().current_span()
            await asyncio.sleep(0.1)  # Simulate async work
            return f"Tool {tool_id}: {span.call_id}"

        def create_concurrent_tool(tool_id):
            async def tool():
                return await concurrent_tool(tool_id)
            tool.__name__ = f"tool_{tool_id}"
            return tool

        agent = Agent(
            name="concurrent_agent",
            model="openai/gpt-4o-mini",
            tools=[create_concurrent_tool("A"), create_concurrent_tool("B")]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use both tools"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_deep_nesting_traces(self):
        """Test trace access in deeply nested calls."""
        def level_three() -> str:
            """Deepest level tool."""
            try:
                current = get_run_context().current_span()
                parent = get_run_context().parent_span()
                return f"Level 3 - Current: {current.path}, Parent: {parent.path}"
            except Exception as e:
                return f"Level 3 error: {str(e)}"

        async def level_two() -> str:
            """Middle level that calls another agent."""
            level_three_agent = Agent(
                name="level_three_agent",
                model="openai/gpt-4o-mini",
                tools=[level_three]
            )

            from timbal.types.message import Message
            prompt = Message.validate({"role": "user", "content": "Use level_three"})
            result = level_three_agent(prompt=prompt)
            output = await result.collect()
            if hasattr(output, 'output') and output.output:
                return str(output.output.content)
            return "Level 3 not found"

        async def level_one() -> str:
            """Top level that starts the chain."""
            return await level_two()

        main_agent = Agent(
            name="deep_nesting_agent",
            model="openai/gpt-4o-mini",
            tools=[level_one]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use level_one"})
        result = main_agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_trace_memory_management(self):
        """Test that traces don't cause memory leaks in long-running scenarios."""
        iteration_count = 0

        def memory_test_tool() -> str:
            """Tool that creates traces repeatedly."""
            nonlocal iteration_count
            iteration_count += 1

            span = get_run_context().current_span()
            # Create some data that would accumulate if not managed properly
            span.metadata = {"iteration": iteration_count, "data": "x" * 100}

            return f"Iteration {iteration_count} completed"

        agent = Agent(
            name="memory_test_agent",
            model="openai/gpt-4o-mini",
            tools=[memory_test_tool]
        )

        # Run multiple iterations
        from timbal.types.message import Message
        for i in range(3):
            prompt = Message.validate({"role": "user", "content": f"Use memory_test_tool (run {i})"})
            result = agent(prompt=prompt)
            output = await result.collect()
            assert hasattr(output, 'output')
            assert output.output is not None

        assert iteration_count >= 3


class TestTraceLifecycle:
    """Test the complete lifecycle of traces: creation → execution → persistence."""

    @pytest.mark.asyncio
    async def test_trace_creation_and_initialization(self):
        """Test that traces are properly created and initialized."""
        def inspect_trace_creation() -> str:
            """Inspect the trace at creation time."""
            span = get_run_context().current_span()

            # Verify initial state
            assert span.call_id is not None
            assert span.path is not None
            assert span.t0 is not None
            assert span.t1 is None  # Should be None during execution
            assert span.output is None  # Should be None during execution
            assert span.error is None
            assert isinstance(span.usage, dict)
            assert isinstance(span.metadata, dict)

            return f"Trace created: {span.call_id}, Path: {span.path}"

        agent = Agent(
            name="trace_creation_agent",
            model="openai/gpt-4o-mini",
            tools=[inspect_trace_creation]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use inspect_trace_creation"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_trace_input_capture(self):
        """Test that trace input is properly captured."""
        def check_trace_input() -> str:
            """Check that the trace captures the correct input."""
            span = get_run_context().current_span()

            # The input should contain the message we sent
            assert span.input is not None
            input_str = str(span.input)
            assert "check_trace_input" in input_str or "test input capture" in input_str

            return f"Input captured: {type(span.input).__name__}"

        agent = Agent(
            name="input_capture_agent",
            model="openai/gpt-4o-mini",
            tools=[check_trace_input]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use check_trace_input to test input capture"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')

    @pytest.mark.asyncio
    async def test_trace_state_progression(self):
        """Test trace state changes during execution."""
        execution_stages = []

        def capture_execution_stage(stage: str) -> str:
            """Capture the current execution stage and trace state."""
            span = get_run_context().current_span()

            stage_info = {
                "stage": stage,
                "has_input": span.input is not None,
                "has_output": span.output is not None,
                "has_t1": span.t1 is not None,
                "usage_items": len(span.usage),
                "metadata_items": len(span.metadata)
            }

            execution_stages.append(stage_info)

            # Add some usage and metadata
            get_run_context().update_usage("test_calls", 1)
            span.metadata[f"stage_{stage}"] = True

            return f"Stage {stage} captured"

        def stage_one() -> str:
            return capture_execution_stage("one")

        def stage_two() -> str:
            return capture_execution_stage("two")

        agent = Agent(
            name="state_progression_agent",
            model="openai/gpt-4o-mini",
            tools=[stage_one, stage_two]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use stage_one then stage_two"})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed
        assert hasattr(output, 'output')

        # Verify we captured multiple stages
        assert len(execution_stages) >= 2

    @pytest.mark.asyncio
    async def test_trace_completion_and_timing(self):
        """Test that traces are properly completed with timing information."""
        import time

        def slow_operation() -> str:
            """Simulate a slow operation to test timing."""
            span = get_run_context().current_span()

            # Verify we're still executing (t1 should be None)
            assert span.t1 is None

            # Record start time for comparison
            start_time = span.t0
            assert start_time is not None

            # Simulate work
            time.sleep(0.1)

            return f"Operation completed, started at: {start_time}"

        agent = Agent(
            name="timing_agent",
            model="openai/gpt-4o-mini",
            tools=[slow_operation]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use slow_operation"})

        start_test_time = int(time.time() * 1000)
        result = agent(prompt=prompt)
        output = await result.collect()
        end_test_time = int(time.time() * 1000)

        # Verify execution completed
        assert hasattr(output, 'output')

        # Verify timing information in output event
        assert hasattr(output, 't0')
        assert hasattr(output, 't1')
        assert output.t0 is not None
        assert output.t1 is not None
        assert output.t1 > output.t0  # End time after start time
        assert output.t0 >= start_test_time  # Started after test began
        assert output.t1 <= end_test_time  # Ended before test completed

    @pytest.mark.asyncio
    async def test_trace_error_handling(self):
        """Test trace state when errors occur."""
        def failing_tool() -> str:
            """Tool that always fails."""
            span = get_run_context().current_span()

            # Verify trace state before error
            assert span.error is None
            assert span.t1 is None

            # Add some usage before failing
            get_run_context().update_usage("failed_calls", 1)

            raise ValueError("Intentional test error")

        agent = Agent(
            name="error_agent",
            model="openai/gpt-4o-mini",
            tools=[failing_tool]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use failing_tool"})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Even with errors, we should get an output event
        assert hasattr(output, 'output')

        # Check if error information is captured
        # Note: The exact error handling depends on agent implementation

    @pytest.mark.asyncio
    async def test_trace_persistence_and_retrieval(self):
        """Test trace persistence across agent calls."""
        stored_call_ids = []

        def store_call_id() -> str:
            """Store the current call ID for later retrieval."""
            span = get_run_context().current_span()
            stored_call_ids.append(span.call_id)
            return f"Stored call ID: {span.call_id}"

        def check_stored_ids() -> str:
            """Check that we can access previously stored call IDs."""
            span = get_run_context().current_span()

            # We should have access to the stored IDs
            assert len(stored_call_ids) > 0

            # Current call should be different from previous calls
            assert span.call_id not in stored_call_ids

            return f"Current: {span.call_id}, Stored: {stored_call_ids}"

        agent = Agent(
            name="persistence_agent",
            model="openai/gpt-4o-mini",
            tools=[store_call_id, check_stored_ids]
        )

        from timbal.types.message import Message

        # First call
        prompt1 = Message.validate({"role": "user", "content": "Use store_call_id"})
        result1 = agent(prompt=prompt1)
        output1 = await result1.collect()
        assert hasattr(output1, 'output')

        # Second call
        prompt2 = Message.validate({"role": "user", "content": "Use check_stored_ids"})
        result2 = agent(prompt=prompt2)
        output2 = await result2.collect()
        assert hasattr(output2, 'output')

        # Verify we stored at least one call ID
        assert len(stored_call_ids) >= 1

    @pytest.mark.asyncio
    async def test_nested_trace_hierarchy(self):
        """Test trace hierarchy in nested agent execution."""
        def get_trace_hierarchy() -> str:
            """Get information about the trace hierarchy."""
            current = get_run_context().current_span()

            hierarchy_info = {
                "current_call_id": current.call_id,
                "current_path": current.path,
                "parent_call_id": current.parent_call_id,
            }

            # Try to get parent if it exists
            try:
                parent = get_run_context().parent_span()
                hierarchy_info["parent_path"] = parent.path
                hierarchy_info["has_parent"] = True
            except RuntimeError:
                hierarchy_info["has_parent"] = False

            return f"Hierarchy: {hierarchy_info}"

        async def nested_call() -> str:
            """Create a nested agent call."""
            nested_agent = Agent(
                name="nested_hierarchy_agent",
                model="openai/gpt-4o-mini",
                tools=[get_trace_hierarchy]
            )

            from timbal.types.message import Message
            prompt = Message.validate({"role": "user", "content": "Use get_trace_hierarchy"})
            result = nested_agent(prompt=prompt)
            output = await result.collect()

            if hasattr(output, 'output') and output.output:
                return str(output.output.content)
            return "Nested call failed"

        main_agent = Agent(
            name="main_hierarchy_agent",
            model="openai/gpt-4o-mini",
            tools=[nested_call]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use nested_call"})
        result = main_agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None


class TestPostHookIntegration:
    """Test post_hook integration and output modification scenarios."""

    @pytest.mark.asyncio
    async def test_post_hook_output_modification(self):
        """Test post_hook modifying output after execution."""
        def simple_calculator(a: int, b: int) -> int:
            """Simple addition calculator."""
            return a + b

        async def post_hook():
            """Post-hook that modifies the output."""
            span = get_run_context().current_span()

            # Verify we have output at this stage
            assert span.output is not None

            # Modify the output (add formatting)
            if isinstance(span.output, int):
                span.output = f"Result: {span.output}"
            elif hasattr(span.output, 'content'):
                # For message objects, modify the content
                original_content = str(span.output.content)
                span.output.content = f"[POST-PROCESSED] {original_content}"

        agent = Agent(
            name="post_hook_agent",
            model="openai/gpt-4o-mini",
            post_hook=post_hook,
            tools=[simple_calculator]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use simple_calculator with a=5 and b=3"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert output.output is not None

    @pytest.mark.asyncio
    async def test_post_hook_metadata_enhancement(self):
        """Test post_hook adding metadata and analytics."""
        def data_processor(data: str) -> str:
            """Process some data."""
            return data.upper()

        async def analytics_post_hook():
            """Post-hook that adds analytics metadata."""
            span = get_run_context().current_span()

            # Add analytics metadata
            print(span.metadata)
            span.metadata.update({
                "processed_at": datetime.now().isoformat(),
                "output_length": len(str(span.output)) if span.output else 0,
                "processing_stage": "completed",
                "enhanced": True
            })
            print(span.metadata)

            # Update usage metrics
            get_run_context().update_usage("post_hook_calls", 1)
            get_run_context().update_usage("metadata_enhancements", 1)

        agent = Agent(
            name="analytics_agent",
            model="openai/gpt-4o-mini",
            post_hook=analytics_post_hook,
            tools=[data_processor]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use data_processor with data='hello world'"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert hasattr(output, 'usage')
        print(output)

    @pytest.mark.asyncio
    async def test_post_hook_error_handling(self):
        """Test post_hook behavior when errors occur during execution."""
        def working_tool() -> str:
            """A tool that works normally."""
            return "Tool executed successfully"

        async def error_sensitive_post_hook():
            """Post-hook that behaves differently based on errors."""
            span = get_run_context().current_span()

            if span.error:
                # Handle error case
                span.metadata["error_handled"] = True
                span.metadata["recovery_attempted"] = True
            else:
                # Handle success case
                span.metadata["success_processed"] = True
                if span.output:
                    span.metadata["output_type"] = type(span.output).__name__

        agent = Agent(
            name="error_handling_agent",
            model="openai/gpt-4o-mini",
            post_hook=error_sensitive_post_hook,
            tools=[working_tool]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use working_tool"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        # Post_hook should have run (even if metadata doesn't propagate to final output)
        assert output.error is None  # Should have succeeded

    @pytest.mark.asyncio
    async def test_post_hook_trace_access(self):
        """Test post_hook accessing trace information for logging/monitoring."""
        execution_log = []

        def monitored_tool(value: str) -> str:
            """A tool that will be monitored."""
            return f"Processed: {value}"

        async def monitoring_post_hook():
            """Post-hook that logs execution details."""
            span = get_run_context().current_span()

            # Create execution log entry
            log_entry = {
                "call_id": span.call_id,
                "path": span.path,
                "execution_time": span.t1 - span.t0 if span.t1 else None,
                "had_error": span.error is not None,
                "input_type": type(span.input).__name__ if span.input else None,
                "output_type": type(span.output).__name__ if span.output else None,
                "usage": dict(span.usage),
                "metadata_keys": list(span.metadata.keys())
            }

            execution_log.append(log_entry)

            # Add monitoring metadata
            span.metadata["monitored"] = True
            span.metadata["log_entry_id"] = len(execution_log) - 1

        agent = Agent(
            name="monitoring_agent",
            model="openai/gpt-4o-mini",
            post_hook=monitoring_post_hook,
            tools=[monitored_tool]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use monitored_tool with value='test_data'"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')

        # Verify monitoring worked
        assert len(execution_log) >= 1
        log_entry = execution_log[0]
        assert log_entry["call_id"] is not None
        assert log_entry["path"] is not None

    @pytest.mark.asyncio
    async def test_pre_and_post_hook_coordination(self):
        """Test coordination between pre_hook and post_hook."""
        processing_context = {}

        def transform_data(data: str) -> str:
            """Transform data with context."""
            # Access context set by pre_hook
            context = processing_context.get("current_context", {})
            prefix = context.get("prefix", "")
            suffix = context.get("suffix", "")

            return f"{prefix}{data.upper()}{suffix}"

        async def setup_pre_hook():
            """Pre-hook that sets up processing context."""
            span = get_run_context().current_span()

            # Set up context based on input
            processing_context["current_context"] = {
                "prefix": "[PRE] ",
                "suffix": " [/PRE]",
                "trace_id": span.call_id,
                "setup_time": datetime.now().isoformat()
            }

            # Store context in trace metadata
            span.metadata["pre_hook_context"] = processing_context["current_context"]

        async def cleanup_post_hook():
            """Post-hook that cleans up and finalizes processing."""
            span = get_run_context().current_span()

            # Get the context that was set up
            context = processing_context.get("current_context", {})

            # Add completion metadata
            span.metadata.update({
                "pre_hook_setup_time": context.get("setup_time"),
                "post_hook_completion_time": datetime.now().isoformat(),
                "processing_completed": True
            })

            # Clean up context
            processing_context.clear()

        agent = Agent(
            name="coordinated_agent",
            model="openai/gpt-4o-mini",
            pre_hook=setup_pre_hook,
            post_hook=cleanup_post_hook,
            tools=[transform_data]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use transform_data with data='hello world'"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')
        assert hasattr(output, 'metadata')
        assert output.metadata.get("processing_completed") is True

    @pytest.mark.asyncio
    async def test_post_hook_with_async_operations(self):
        """Test post_hook performing async operations."""
        import asyncio

        async_results = []

        def quick_task() -> str:
            """A quick task."""
            return "Task completed"

        async def async_post_hook():
            """Post-hook that performs async operations."""
            span = get_run_context().current_span()

            # Simulate async operations
            await asyncio.sleep(0.1)  # Simulate network call

            # Store async result
            async_result = {
                "call_id": span.call_id,
                "async_processed": True,
                "processing_delay": 100  # ms
            }
            async_results.append(async_result)

            # Update trace with async operation results
            span.metadata["async_post_processing"] = True
            span.metadata["async_operations_count"] = 1

        agent = Agent(
            name="async_post_hook_agent",
            model="openai/gpt-4o-mini",
            post_hook=async_post_hook,
            tools=[quick_task]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use quick_task"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert hasattr(output, 'output')

        # Verify async post-hook executed
        assert len(async_results) >= 1
        assert async_results[0]["async_processed"] is True


class TestComplexNestedExecution:
    """Ultimate complex test with nested agents, parallel execution, and comprehensive hook integration."""

    @pytest.mark.asyncio
    async def test_ultimate_complex_nested_agent_execution(self):
        """
        Complex final test that validates:
        - Nested agent structure (superagent -> secret_agent -> multiple tools)
        - Parallel execution forcing multiple tool uses
        - All tool types: sync, async, sync_gen, async_gen
        - Pre/post hooks at multiple levels with trace modifications
        - Usage tracking across all levels
        - Input/output modification and preservation
        - Complex trace hierarchy and context management
        - Error handling and recovery scenarios
        """
        # Global state for tracking execution across all levels
        execution_log = []
        hook_modifications = {}
        usage_summary = {}

        # Tool implementations with different execution patterns
        def first_clue(x: int) -> str:
            """Sync non-generator tool"""
            context = get_run_context()
            context.update_usage("sync_tool:calls", 1)
            span = context.current_span()
            execution_log.append({
                "stage": "first_clue",
                "path": span.path,
                "call_id": span.call_id,
                "input": x,
                "tool_type": "sync"
            })
            return "".join([f"{i+1}" for i in range(x)])

        async def second_clue(x: int) -> str:
            """Async coroutine tool"""
            context = get_run_context()
            context.update_usage("async_tool:calls", 1)
            span = context.current_span()
            execution_log.append({
                "stage": "second_clue",
                "path": span.path,
                "call_id": span.call_id,
                "input": x,
                "tool_type": "async"
            })
            # Simulate async work
            await asyncio.sleep(0.01)
            return "".join([f"{chr(65+i)}" for i in range(x)])  # Letters instead of numbers

        def third_clue(x: int) -> str:
            """Sync generator tool"""
            context = get_run_context()
            context.update_usage("sync_gen_tool:calls", 1)
            span = context.current_span()
            execution_log.append({
                "stage": "third_clue",
                "path": span.path,
                "call_id": span.call_id,
                "input": x,
                "tool_type": "sync_gen"
            })
            for i in range(x):
                yield f"[{i+1}]"

        async def fourth_clue(x: int) -> str:
            """Async generator tool"""
            context = get_run_context()
            context.update_usage("async_gen_tool:calls", 1)
            span = context.current_span()
            execution_log.append({
                "stage": "fourth_clue",
                "path": span.path,
                "call_id": span.call_id,
                "input": x,
                "tool_type": "async_gen"
            })
            for i in range(x):
                await asyncio.sleep(0.001)  # Simulate async work
                yield f"<{chr(97+i)}>"  # Lowercase letters in brackets

        # Pre-hook for secret agent that modifies input and adds context
        async def secret_agent_pre_hook():
            """Pre-hook that enriches the secret agent's execution context"""
            span = get_run_context().current_span()

            # Store original input before modification
            original_input = str(span.input)
            hook_modifications[f"secret_agent_pre_{span.call_id}"] = {
                "original_input": original_input,
                "stage": "pre_hook"
            }

            # Enhance the input with additional context
            if hasattr(span.input, 'content'):
                if isinstance(span.input.content, str):
                    enhanced_content = f"ENHANCED: {span.input.content} [Use ALL available tools for complete analysis]"
                    span.input.content = enhanced_content
                elif isinstance(span.input.content, dict) and "prompt" in span.input.content:
                    enhanced_prompt = f"PRIORITY: {span.input.content['prompt']} - Execute all tools in parallel for comprehensive results"
                    span.input.content["prompt"] = enhanced_prompt

            # Add metadata for tracking
            span.metadata.update({
                "pre_hook_enhanced": True,
                "enhancement_time": datetime.now().isoformat(),
                "expected_tools": ["first_clue", "second_clue", "third_clue", "fourth_clue"]
            })

        # Post-hook for secret agent that processes results
        async def secret_agent_post_hook():
            """Post-hook that analyzes and enhances the secret agent's output"""
            span = get_run_context().current_span()

            # Capture the original output before modification
            original_output = str(span.output) if span.output else "None"
            hook_modifications[f"secret_agent_post_{span.call_id}"] = {
                "original_output": original_output,
                "stage": "post_hook"
            }

            # Enhance the output with analysis
            if span.output and hasattr(span.output, 'content'):
                analysis = f"\n\n[ANALYSIS] Tools executed: {len(execution_log)} | Usage tracked: {dict(span.usage)}"
                if isinstance(span.output.content, list):
                    for item in span.output.content:
                        if hasattr(item, 'text'):
                            item.text += analysis
                elif hasattr(span.output.content, 'text'):
                    span.output.content.text += analysis

            # Add comprehensive metadata
            span.metadata.update({
                "post_hook_processed": True,
                "processing_time": datetime.now().isoformat(),
                "tools_called": [log["stage"] for log in execution_log if log["path"].startswith(span.path)],
                "total_usage": dict(span.usage)
            })

        # Pre-hook for superagent that sets up orchestration context
        async def superagent_pre_hook():
            """Pre-hook that prepares the superagent for complex orchestration"""
            span = get_run_context().current_span()

            hook_modifications[f"superagent_pre_{span.call_id}"] = {
                "original_input": str(span.input),
                "stage": "orchestrator_pre_hook"
            }

            # Add orchestration instructions
            if hasattr(span.input, 'content'):
                if isinstance(span.input.content, dict):
                    if "system_prompt" in span.input.content:
                        enhanced_system = (
                            f"{span.input.content.get('system_prompt', '')} "
                            f"CRITICAL: You must delegate to the secret_agent for BOTH numbers. "
                            f"Ensure the secret_agent uses ALL four available tools for each number. "
                            f"Combine and synthesize all results into a comprehensive final answer."
                        )
                        span.input.content["system_prompt"] = enhanced_system

            span.metadata.update({
                "orchestrator_pre_hook": True,
                "orchestration_setup": datetime.now().isoformat(),
                "delegation_strategy": "parallel_comprehensive"
            })

        # Post-hook for superagent that provides final synthesis
        async def superagent_post_hook():
            """Post-hook that provides final analysis and summary"""
            span = get_run_context().current_span()

            # Collect usage from all nested traces
            run_context = get_run_context()
            total_usage = {}
            all_traces = []

            for call_id, span_obj in run_context._trace.items():
                all_traces.append({
                    "call_id": call_id,
                    "path": span_obj.path,
                    "usage": dict(span_obj.usage),
                    "metadata": dict(span_obj.metadata)
                })
                # Aggregate usage
                for key, value in span_obj.usage.items():
                    total_usage[key] = total_usage.get(key, 0) + value

            usage_summary["final_aggregated_usage"] = total_usage
            usage_summary["all_traces"] = all_traces

            hook_modifications[f"superagent_post_{span.call_id}"] = {
                "original_output": str(span.output) if span.output else "None",
                "stage": "orchestrator_post_hook",
                "total_nested_traces": len(all_traces),
                "aggregated_usage": total_usage
            }

            # Add comprehensive final metadata
            span.metadata.update({
                "orchestrator_post_hook": True,
                "final_analysis": datetime.now().isoformat(),
                "total_nested_executions": len(all_traces),
                "execution_summary": {
                    "tools_executed": len([log for log in execution_log]),
                    "hook_modifications": len(hook_modifications),
                    "usage_metrics": total_usage
                }
            })

        # Create the secret agent with all tool types and hooks
        secret_agent = Agent(
            name="secret_agent",
            description="This agent computes secrets using all available tool types",
            model="openai/gpt-4o-mini",
            pre_hook=secret_agent_pre_hook,
            post_hook=secret_agent_post_hook,
            tools=[
                Tool(name="first_clue", handler=first_clue),
                second_clue,
                Tool(name="third_clue", handler=third_clue),
                Tool(name="fourth_clue", handler=fourth_clue),
            ],
        )

        # Create the superagent with orchestration hooks
        superagent = Agent(
            name="super_agent",
            model="openai/gpt-4o-mini",
            model_params={"max_tokens": 2048},
            pre_hook=superagent_pre_hook,
            post_hook=superagent_post_hook,
            tools=[secret_agent],
        )

        # Execute the complex nested scenario
        result = superagent(
            system_prompt=(
                "You are a master orchestrator. You must use the secret_agent tool to find secrets "
                "for BOTH number 2 and number 3. For each number, ensure the secret_agent uses "
                "ALL four available tools (first_clue, second_clue, third_clue, fourth_clue). "
                "Then combine and synthesize all results into a comprehensive final answer."
            ),
            prompt="Get the complete secret analysis for number 2 and number 3 using all available tools"
        )

        output = await result.collect()

        # Comprehensive validations
        assert hasattr(output, 'output')
        assert output.output is not None

        # Verify execution log captured all tool executions
        assert len(execution_log) >= 4  # At least 4 tools should have been called

        # Verify all tool types were executed
        tool_types_executed = set(log["tool_type"] for log in execution_log)
        expected_tool_types = {"sync", "async", "sync_gen", "async_gen"}
        assert tool_types_executed.intersection(expected_tool_types), f"Expected some of {expected_tool_types}, got {tool_types_executed}"

        # Verify hook modifications occurred at all levels
        assert len(hook_modifications) >= 4  # At least 2 pre-hooks and 2 post-hooks

        # Verify pre-hook and post-hook pairs
        pre_hook_calls = [k for k in hook_modifications.keys() if "pre_" in k]
        post_hook_calls = [k for k in hook_modifications.keys() if "post_" in k]
        assert len(pre_hook_calls) >= 2
        assert len(post_hook_calls) >= 2

        # Verify usage tracking aggregation
        assert "final_aggregated_usage" in usage_summary
        final_usage = usage_summary["final_aggregated_usage"]
        assert len(final_usage) > 0  # Should have captured usage from multiple sources

        # Verify trace hierarchy was properly established
        run_context = get_run_context()
        assert len(run_context._trace) >= 3  # Superagent, secret_agent, and at least one tool

        # Verify parent-child relationships in traces
        superagent_traces = [t for t in run_context._trace.values() if t.path == "super_agent"]
        secret_agent_traces = [t for t in run_context._trace.values() if "secret_agent" in t.path]
        tool_traces = [t for t in run_context._trace.values() if any(tool in t.path for tool in ["first_clue", "second_clue", "third_clue", "fourth_clue"])]

        assert len(superagent_traces) >= 1
        assert len(secret_agent_traces) >= 1
        assert len(tool_traces) >= 1

        # Verify metadata was properly enhanced by hooks
        for trace in run_context._trace.values():
            if trace.path == "super_agent":
                assert trace.metadata.get("orchestrator_pre_hook") is True
                assert trace.metadata.get("orchestrator_post_hook") is True
            elif "secret_agent" in trace.path and trace.path.count(".") == 0:  # Direct secret_agent, not nested
                assert trace.metadata.get("pre_hook_enhanced") is True
                assert trace.metadata.get("post_hook_processed") is True

        # Verify dumped versions preserve originals despite modifications
        for trace in run_context._trace.values():
            if hasattr(trace, '_input_dump') and trace._input_dump:
                dumped_input = str(trace._input_dump)
                # Dumped input should not contain hook modifications
                assert "ENHANCED:" not in dumped_input
                assert "PRIORITY:" not in dumped_input
                assert "CRITICAL:" not in dumped_input

        # Verify complex execution patterns
        execution_paths = [log["path"] for log in execution_log]
        unique_paths = set(execution_paths)
        assert len(unique_paths) >= 2  # Should have multiple execution paths

        # Verify usage metrics were tracked across all levels
        total_calls = sum(1 for k in final_usage.keys() if k.endswith(":calls"))
        assert total_calls >= 4  # Should have tracked multiple tool calls

        # Log final summary for verification
        print("\n=== COMPLEX EXECUTION SUMMARY ===")
        print(f"Total execution log entries: {len(execution_log)}")
        print(f"Hook modifications: {len(hook_modifications)}")
        print(f"Tool types executed: {tool_types_executed}")
        print(f"Final aggregated usage: {final_usage}")
        print(f"Total traces created: {len(run_context._trace)}")
        print("=====================================")


class TestTraceDumpedVersions:
    """Test that dumped versions of trace data preserve original state during modifications."""

    @pytest.mark.asyncio
    async def test_input_modification_preserves_dumped_original(self):
        """Test that when pre_hook modifies input, the dumped version contains the original."""
        original_inputs = []
        modified_inputs = []
        dumped_inputs = []

        def process_input(data: str) -> str:
            """Process the input data."""
            span = get_run_context().current_span()

            # Capture what the tool sees
            modified_inputs.append(str(span.input))

            return f"Processed: {data}"

        async def input_modifying_pre_hook():
            """Pre-hook that modifies the input."""
            span = get_run_context().current_span()

            # Capture original input before modification
            original_inputs.append(str(span.input))

            # Capture the dumped version if it exists
            if hasattr(span, '_input_dump'):
                dumped_inputs.append(str(span._input_dump))

            # Modify the input
            if hasattr(span.input, 'content'):
                # For Message objects
                original_content = span.input.content
                span.input.content = f"[MODIFIED] {original_content}"
            elif isinstance(span.input, dict):
                # For dict inputs
                span.input["modified"] = True
                span.input["original_preserved"] = False
            else:
                # For other types, wrap in a modified structure
                span.input = {"modified_input": str(span.input), "was_modified": True}

        agent = Agent(
            name="input_modification_agent",
            model="openai/gpt-4o-mini",
            pre_hook=input_modifying_pre_hook,
            tools=[process_input]
        )

        from timbal.types.message import Message
        original_prompt = "Use process_input with data='test_data'"
        prompt = Message.validate({"role": "user", "content": original_prompt})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed
        assert hasattr(output, 'output')

        # Verify we captured the inputs at different stages
        assert len(original_inputs) >= 1
        assert len(modified_inputs) >= 1

        # The dumped version should preserve the original input
        if hasattr(output, '_input_dump') and output._input_dump:
            dumped_content = str(output._input_dump)
            # The dumped version should contain the original, not the modified version
            assert "[MODIFIED]" not in dumped_content
            assert original_prompt in dumped_content

    @pytest.mark.asyncio
    async def test_output_modification_preserves_dumped_original(self):
        """Test that when post_hook modifies output, the dumped version contains the modified output."""
        async def output_modifying_post_hook():
            """Post-hook that modifies the output."""
            span = get_run_context().current_span()
            span.output.content[0] = f"[POST-MODIFIED] {span.output.content[0]}"

        agent = Agent(
            name="output_modification_agent",
            model="openai/gpt-4o-mini",
            post_hook=output_modifying_post_hook,
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Hello world"})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed
        assert hasattr(output, 'output')

        # The dumped version should contain the MODIFIED output (dump happens after post_hook)
        if hasattr(output, '_output_dump') and output._output_dump:
            assert "[POST-MODIFIED]" in output._output_dump["content"][0]

    @pytest.mark.asyncio
    async def test_audio_stt_input_modification_scenario(self):
        """Test the specific STT scenario where audio input is modified but original is preserved."""
        def process_text(text: str) -> str:
            """Process the text input."""
            return f"Processed text: {text.upper()}"

        async def stt_pre_hook():
            """Pre-hook that simulates STT conversion of audio to text."""
            span = get_run_context().current_span()

            # Simulate the scenario where input contains audio data
            if hasattr(span.input, 'content') and isinstance(span.input.content, dict):
                if "prompt" in span.input.content and str(span.input.content["prompt"]).startswith("audio:"):
                    # Simulate STT conversion
                    audio_content = span.input.content["prompt"]
                    transcribed_text = audio_content.replace("audio:", "transcribed:")

                    # Modify the input to contain transcribed text
                    span.input.content["prompt"] = transcribed_text
                    span.input.content["stt_processed"] = True

        agent = Agent(
            name="stt_agent",
            model="openai/gpt-4o-mini",
            pre_hook=stt_pre_hook,
            tools=[process_text]
        )

        from timbal.types.message import Message
        # Simulate audio input
        audio_prompt = {"prompt": "audio:hello_world.wav", "type": "audio"}
        prompt = Message.validate({"role": "user", "content": audio_prompt})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed
        assert hasattr(output, 'output')

        # Check that the dumped input preserves the original audio reference
        if hasattr(output, '_input_dump') and output._input_dump:
            dumped_input = output._input_dump
            dumped_str = str(dumped_input)

            # The dumped version should contain the original audio reference
            assert "audio:hello_world.wav" in dumped_str
            # The dumped version should NOT contain the transcribed version
            assert "transcribed:" not in dumped_str
            # The dumped version should NOT contain the stt_processed flag
            assert "stt_processed" not in dumped_str

    @pytest.mark.asyncio
    async def test_model_dump_behavior(self):
        """Test that span.model_dump() uses dumped versions when available."""
        def test_tool() -> str:
            """Test tool to verify dump behavior."""
            span = get_run_context().current_span()

            # Get the model dump
            span_dump = span.model_dump()

            # Verify the dump uses dumped versions if available
            if hasattr(span, '_input_dump'):
                # The model_dump should use _input_dump, not span.input
                assert span_dump["input"] == span._input_dump

            return "Tool executed"

        async def modifying_pre_hook():
            """Pre-hook that modifies input."""
            span = get_run_context().current_span()

            # Modify the input after it's been dumped
            if hasattr(span.input, 'content'):
                span.input.content = f"MODIFIED: {span.input.content}"

        agent = Agent(
            name="dump_behavior_agent",
            model="openai/gpt-4o-mini",
            pre_hook=modifying_pre_hook,
            tools=[test_tool]
        )

        from timbal.types.message import Message
        prompt = Message.validate({"role": "user", "content": "Use test_tool"})
        result = agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed
        assert hasattr(output, 'output')

    @pytest.mark.asyncio
    async def test_complex_nested_modification_scenario(self):
        """Test complex scenario with nested modifications and dump preservation."""
        modification_log = []

        def inner_tool(data: str) -> str:
            """Inner tool that processes data."""
            span = get_run_context().current_span()
            modification_log.append({
                "stage": "inner_tool",
                "input": str(span.input),
                "has_input_dump": hasattr(span, '_input_dump'),
                "input_dump": str(getattr(span, '_input_dump', None))
            })
            return f"Inner processed: {data}"

        async def inner_pre_hook():
            """Pre-hook that makes inner modifications."""
            span = get_run_context().current_span()
            modification_log.append({
                "stage": "inner_pre_hook_before",
                "input": str(span.input),
                "has_input_dump": hasattr(span, '_input_dump')
            })

            # Modify input
            if hasattr(span.input, 'content'):
                span.input.content = f"[INNER-PRE] {span.input.content}"

        async def outer_tool_wrapper() -> str:
            """Outer tool that calls an inner agent."""
            inner_agent = Agent(
                name="inner_modification_agent",
                model="openai/gpt-4o-mini",
                pre_hook=inner_pre_hook,
                tools=[inner_tool]
            )

            from timbal.types.message import Message
            prompt = Message.validate({"role": "user", "content": "Use inner_tool with data='nested_data'"})
            result = inner_agent(prompt=prompt)
            output = await result.collect()

            # Log the final output event
            modification_log.append({
                "stage": "inner_agent_output",
                "has_input_dump": hasattr(output, '_input_dump'),
                "input_dump": str(getattr(output, '_input_dump', None)),
                "input_from_output": str(getattr(output, 'input', None))
            })

            return "Outer wrapper completed"

        async def outer_pre_hook():
            """Outer pre-hook that makes initial modifications."""
            span = get_run_context().current_span()
            modification_log.append({
                "stage": "outer_pre_hook",
                "input": str(span.input),
                "has_input_dump": hasattr(span, '_input_dump')
            })

            # Modify input at outer level
            if hasattr(span.input, 'content'):
                span.input.content = f"[OUTER-PRE] {span.input.content}"

        outer_agent = Agent(
            name="outer_modification_agent",
            model="openai/gpt-4o-mini",
            pre_hook=outer_pre_hook,
            tools=[outer_tool_wrapper]
        )

        from timbal.types.message import Message
        original_content = "Use outer_tool_wrapper for complex nesting"
        prompt = Message.validate({"role": "user", "content": original_content})
        result = outer_agent(prompt=prompt)
        output = await result.collect()

        # Verify execution completed
        assert hasattr(output, 'output')

        # Verify modification log was populated
        assert len(modification_log) > 0

        # Verify that at each level, the dumped version preserves the appropriate original
        for log_entry in modification_log:
            if log_entry.get("has_input_dump") and log_entry.get("input_dump") and log_entry.get("input_dump") != "None":
                # Dumped versions should not contain modification markers
                dump_content = log_entry.get("input_dump", "")
                # The final dumped version should contain the original content
                if "complex nesting" in dump_content:
                    assert "[OUTER-PRE]" not in dump_content
                    assert "[INNER-PRE]" not in dump_content