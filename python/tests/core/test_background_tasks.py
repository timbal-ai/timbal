import asyncio
from collections.abc import AsyncGenerator

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import get_run_context
from timbal.types.content import ToolUseContent
from timbal.types.events.output import OutputEvent
from timbal.types.message import Message

from ..conftest import assert_has_output_event


def _tool_call(tool_name: str, input: dict, *, id: str = "c1", run_in_background: bool = False) -> Message:
    """Return a TestModel response that calls one tool."""
    actual_input = {**input, **({"run_in_background": True} if run_in_background else {})}
    return Message(
        role="assistant",
        content=[ToolUseContent(id=id, name=tool_name, input=actual_input)],
        stop_reason="tool_use",
    )


class TestBackgroundTasks:
    """Test background task execution and status checking."""

    @pytest.mark.asyncio
    async def test_background_task_execution_and_status(self):
        """Background task is registered in _bg_tasks and get_background_task reports its status."""

        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        bg_tool = Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="bg_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"duration": 0.2}, run_in_background=True),
                "Task started in the background.",
            ]),
            tools=[bg_tool],
        )

        result = await agent(prompt="run slow task").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        agent_runnable = span.runnable
        assert len(agent_runnable._bg_tasks) > 0

        await asyncio.sleep(0.3)  # wait for task to finish

    @pytest.mark.asyncio
    async def test_background_task_with_immediate_status_check(self):
        """Agent can start a background task and immediately check its status."""

        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        bg_tool = Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="bg_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                "The task is running.",
            ]),
            tools=[bg_tool],
        )

        result = await agent(prompt="run and check").collect()
        assert_has_output_event(result)

        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_bg_tasks_dict_populated_when_running_background(self):
        """_bg_tasks dictionary is populated when a tool runs in background."""

        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        bg_tool = Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="bg_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                "Started.",
            ]),
            tools=[bg_tool],
        )

        result = await agent(prompt="run in background").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        agent_runnable = span.runnable
        assert agent_runnable._bg_tasks is not None
        assert len(agent_runnable._bg_tasks) > 0

        task_id = list(agent_runnable._bg_tasks.keys())[0]
        task_info = agent_runnable._bg_tasks[task_id]
        assert "task" in task_info
        assert "event_queue" in task_info
        assert isinstance(task_info["task"], asyncio.Task)

        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_agent_has_get_background_task_tool(self):
        """get_background_task tool appears in the tool list once bg tasks exist."""

        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        bg_tool = Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="bg_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                "Started.",
            ]),
            tools=[bg_tool],
        )

        tool_names_seen = []
        async for event in agent(prompt="run in background"):
            if (
                isinstance(event, OutputEvent)
                and event.path == "bg_agent.llm"
                and len(get_run_context().current_span().runnable._bg_tasks) > 0
            ):
                tool_names_seen.extend(tool.name for tool in event.input["tools"])

        assert "get_background_task" in tool_names_seen

        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_background_task_with_events_and_logs(self):
        """Background tool that yields multiple log steps completes successfully."""

        async def build_interface(project_name: str) -> str:
            steps = [
                "Initializing build environment...",
                "Installing dependencies...",
                "Compiling TypeScript...",
                "Building React components...",
                "Optimizing bundle...",
                "Build completed successfully!",
            ]
            for step in steps:
                await asyncio.sleep(0.02)
            return "\n".join(steps)

        build_tool = Tool(
            name="build_interface",
            description="Build a web interface project",
            handler=build_interface,
            background_mode="auto",
        )

        agent = Agent(
            name="build_agent",
            model=TestModel(responses=[
                _tool_call("build_interface", {"project_name": "my-app"}, run_in_background=True),
                "Build started in the background.",
            ]),
            tools=[build_tool],
        )

        result = await agent(prompt="build my-app in background").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        assert len(span.runnable._bg_tasks) > 0

        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_realtime_events_vs_background_events(self):
        """Foreground tools stream events; background tools queue them."""

        async def streaming_task(steps: int) -> AsyncGenerator[str, None]:
            for i in range(steps):
                yield f"Step {i + 1}/{steps} completed"
                await asyncio.sleep(0.02)

        realtime_tool = Tool(
            name="realtime_task",
            description="Streams events in real-time",
            handler=streaming_task,
            background_mode="never",
        )

        realtime_agent = Agent(
            name="realtime_agent",
            model=TestModel(responses=[
                _tool_call("realtime_task", {"steps": 3}),
                "Done.",
            ]),
            tools=[realtime_tool],
        )

        realtime_events = []
        async for event in realtime_agent(prompt="run realtime task"):
            realtime_events.append(event)

        event_types = [e.type for e in realtime_events if hasattr(e, "type")]
        assert "START" in event_types
        assert "OUTPUT" in event_types

        # Background mode: events go to queue
        background_tool = Tool(
            name="background_task",
            description="Runs in background",
            handler=streaming_task,
            background_mode="always",
        )

        background_agent = Agent(
            name="background_agent",
            model=TestModel(responses=[
                _tool_call("background_task", {"steps": 3}),
                "Started in background.",
            ]),
            tools=[background_tool],
        )

        bg_events = []
        async for event in background_agent(prompt="run background task"):
            bg_events.append(event)

        assert len(bg_events) > 0

        await asyncio.sleep(0.1)

        span = get_run_context().root_span()
        bg_agent_runnable = span.runnable
        if bg_agent_runnable._bg_tasks:
            task_id = list(bg_agent_runnable._bg_tasks.keys())[0]
            event_queue = bg_agent_runnable._bg_tasks[task_id]["event_queue"]
            assert event_queue.qsize() > 0

        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_background_task_event_queue_detailed_inspection(self):
        """Events from a background tool are queued and have the correct metadata."""

        async def detailed_task(task_name: str) -> AsyncGenerator[dict, None]:
            stages = [
                {"stage": "init", "progress": 0},
                {"stage": "processing", "progress": 50},
                {"stage": "finalizing", "progress": 90},
                {"stage": "complete", "progress": 100},
            ]
            for stage_data in stages:
                yield stage_data
                await asyncio.sleep(0.02)

        detailed_tool = Tool(
            name="detailed_task",
            description="Task with detailed progress updates",
            handler=detailed_task,
            background_mode="always",
        )

        agent = Agent(
            name="detailed_agent",
            model=TestModel(responses=[
                _tool_call("detailed_task", {"task_name": "analysis"}),
                "Task queued.",
            ]),
            tools=[detailed_tool],
        )

        result = await agent(prompt="run detailed_task").collect()
        assert_has_output_event(result)

        await asyncio.sleep(0.1)

        span = get_run_context().root_span()
        agent_runnable = span.runnable
        assert len(agent_runnable._bg_tasks) > 0

        task_id = list(agent_runnable._bg_tasks.keys())[0]
        task_info = agent_runnable._bg_tasks[task_id]
        event_queue = task_info["event_queue"]

        all_events = []
        while not event_queue.empty():
            try:
                all_events.append(event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        assert len(all_events) > 0
        for event in all_events:
            if hasattr(event, "path"):
                assert "detailed_task" in event.path

        await asyncio.sleep(0.1)

        status = agent_runnable.get_background_task(task_id)
        assert status["status"] in ["completed", "not_found"]

    @pytest.mark.asyncio
    async def test_multiple_background_tasks_concurrently(self):
        """Multiple background tasks can run at the same time."""

        async def slow_task(tag: str) -> str:
            await asyncio.sleep(0.2)
            return f"done:{tag}"

        bg_tool = Tool(name="slow_task", description="Slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="multi_bg_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"tag": "task1"}, id="c1", run_in_background=True),
                "Task 1 started.",
                _tool_call("slow_task", {"tag": "task2"}, id="c2", run_in_background=True),
                "Task 2 started.",
            ]),
            tools=[bg_tool],
        )

        result1 = await agent(prompt="start task 1").collect()
        assert_has_output_event(result1)

        result2 = await agent(prompt="start task 2").collect()
        assert_has_output_event(result2)

        span = get_run_context().root_span()
        assert len(span.runnable._bg_tasks) >= 1

        await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_background_task_error_handling(self):
        """Background tasks that raise are handled gracefully."""

        async def failing_tool(should_fail: bool = True) -> str:
            await asyncio.sleep(0.05)
            if should_fail:
                raise ValueError("Intentional failure for testing")
            return "Success"

        fail_tool = Tool(
            name="failing_tool",
            description="A tool that can fail",
            handler=failing_tool,
            background_mode="auto",
        )

        agent = Agent(
            name="error_handling_agent",
            model=TestModel(responses=[
                _tool_call("failing_tool", {"should_fail": True}, run_in_background=True),
                "Task started.",
            ]),
            tools=[fail_tool],
        )

        result = await agent(prompt="run failing tool").collect()
        assert_has_output_event(result)

        await asyncio.sleep(0.2)

        span = get_run_context().root_span()
        agent_runnable = span.runnable
        if agent_runnable._bg_tasks:
            task_id = list(agent_runnable._bg_tasks.keys())[0]
            task_info = agent_runnable._bg_tasks[task_id]
            task = task_info["task"]
            assert task.done()

    @pytest.mark.asyncio
    async def test_background_task_nonexistent_task_id(self):
        """get_background_task returns not_found for an unknown task_id."""
        agent = Agent(name="check_missing_agent", model=TestModel())

        await agent(prompt="hi").collect()

        result = agent.get_background_task("nonexistent_task_id_12345")
        assert result["status"] == "not_found"
        assert result["events"] == []

    @pytest.mark.asyncio
    async def test_background_mode_always(self):
        """Tool with background_mode='always' creates a bg task automatically."""

        async def always_background_tool(message: str) -> str:
            await asyncio.sleep(0.1)
            return f"Processed: {message}"

        always_bg_tool = Tool(
            name="always_bg_tool",
            description="Always runs in background",
            handler=always_background_tool,
            background_mode="always",
        )

        agent = Agent(
            name="always_bg_agent",
            model=TestModel(responses=[
                _tool_call("always_bg_tool", {"message": "hello"}),
                "Done.",
            ]),
            tools=[always_bg_tool],
        )

        result = await agent(prompt="use always_bg_tool").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        assert len(span.runnable._bg_tasks) >= 1

        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_background_mode_never(self):
        """Tool with background_mode='never' runs inline and creates no bg task."""

        async def never_background_tool(message: str) -> str:
            await asyncio.sleep(0.05)
            return f"Processed: {message}"

        never_bg_tool = Tool(
            name="never_bg_tool",
            description="Never runs in background",
            handler=never_background_tool,
            background_mode="never",
        )

        agent = Agent(
            name="never_bg_agent",
            model=TestModel(responses=[
                _tool_call("never_bg_tool", {"message": "test"}),
                "Done.",
            ]),
            tools=[never_bg_tool],
        )

        result = await agent(prompt="use never_bg_tool").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        assert len(span.runnable._bg_tasks) == 0

    @pytest.mark.asyncio
    async def test_background_task_cleanup_after_completion(self):
        """Completed background task info is accessible until cleaned up."""

        async def quick_task(value: str) -> str:
            await asyncio.sleep(0.05)
            return f"Done: {value}"

        quick_tool = Tool(
            name="quick_task",
            description="Quick task",
            handler=quick_task,
            background_mode="auto",
        )

        agent = Agent(
            name="cleanup_agent",
            model=TestModel(responses=[
                _tool_call("quick_task", {"value": "test"}, run_in_background=True),
                "Started.",
            ]),
            tools=[quick_tool],
        )

        result = await agent(prompt="run quick task in background").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        agent_runnable = span.runnable
        assert len(agent_runnable._bg_tasks) > 0

        await asyncio.sleep(0.2)

        task_id = list(agent_runnable._bg_tasks.keys())[0]
        status = agent_runnable.get_background_task(task_id)
        assert status["status"] in ["completed", "not_found"]

    @pytest.mark.asyncio
    async def test_background_task_with_structured_output(self):
        """Background tool that returns structured data completes without error."""

        async def structured_task(count: int) -> dict:
            await asyncio.sleep(0.05)
            return {"count": count, "results": [f"item_{i}" for i in range(count)], "status": "completed"}

        struct_tool = Tool(
            name="structured_task",
            description="Returns structured data",
            handler=structured_task,
            background_mode="auto",
        )

        agent = Agent(
            name="structured_bg_agent",
            model=TestModel(responses=[
                _tool_call("structured_task", {"count": 3}, run_in_background=True),
                "Task started.",
            ]),
            tools=[struct_tool],
        )

        result = await agent(prompt="run structured_task").collect()
        assert_has_output_event(result)

        await asyncio.sleep(0.2)

        span = get_run_context().root_span()
        task_id = list(span.runnable._bg_tasks.keys())[0]
        status = span.runnable.get_background_task(task_id)
        assert status["status"] in ["completed", "not_found"]

    @pytest.mark.asyncio
    async def test_background_task_event_queue_retrieval(self):
        """Event queue exists and is an asyncio.Queue for every background task."""

        async def multi_step_task(steps: int) -> str:
            for _ in range(steps):
                await asyncio.sleep(0.02)
            return f"Completed {steps} steps"

        multi_tool = Tool(
            name="multi_step",
            description="Multi-step task",
            handler=multi_step_task,
            background_mode="auto",
        )

        agent = Agent(
            name="queue_test_agent",
            model=TestModel(responses=[
                _tool_call("multi_step", {"steps": 3}, run_in_background=True),
                "Started.",
            ]),
            tools=[multi_tool],
        )

        result = await agent(prompt="run multi_step").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        agent_runnable = span.runnable
        assert len(agent_runnable._bg_tasks) > 0

        task_id = list(agent_runnable._bg_tasks.keys())[0]
        task_info = agent_runnable._bg_tasks[task_id]
        assert "event_queue" in task_info
        assert isinstance(task_info["event_queue"], asyncio.Queue)

        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_background_task_events_accumulation(self):
        """Events generated by a background tool accumulate in the queue."""

        async def event_generating_task(num_events: int) -> str:
            for _ in range(num_events):
                await asyncio.sleep(0.02)
            return f"Generated {num_events} events"

        event_tool = Tool(
            name="event_gen_tool",
            description="Generates events",
            handler=event_generating_task,
            background_mode="auto",
        )

        agent = Agent(
            name="event_accumulation_agent",
            model=TestModel(responses=[
                _tool_call("event_gen_tool", {"num_events": 5}, run_in_background=True),
                "Started.",
            ]),
            tools=[event_tool],
        )

        result = await agent(prompt="run event_gen_tool").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        assert len(span.runnable._bg_tasks) > 0

        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_background_tasks_persist_across_agent_calls(self):
        """Background tasks started in one call are still tracked in the next."""

        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        bg_tool = Tool(name="slow_task", description="Slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="persistence_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                "Task started.",
                "Still tracking it.",
            ]),
            tools=[bg_tool],
        )

        result1 = await agent(prompt="start task").collect()
        assert_has_output_event(result1)

        result2 = await agent(prompt="check tasks").collect()
        assert_has_output_event(result2)

        span = get_run_context().root_span()
        assert span.runnable._bg_tasks is not None

        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_get_background_task_tool_parameters(self):
        """get_background_task can be called directly with a task_id."""

        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        bg_tool = Tool(name="slow_task", description="Slow task", handler=slow_task, background_mode="auto")

        agent = Agent(
            name="param_test_agent",
            model=TestModel(responses=[
                _tool_call("slow_task", {"duration": 0.2}, run_in_background=True),
                "Started.",
            ]),
            tools=[bg_tool],
        )

        result = await agent(prompt="run slow task").collect()
        assert_has_output_event(result)

        span = get_run_context().root_span()
        agent_runnable = span.runnable

        if agent_runnable._bg_tasks:
            task_id = list(agent_runnable._bg_tasks.keys())[0]
            status = agent_runnable.get_background_task(task_id)
            assert status["status"] in ["running", "completed"]

        await asyncio.sleep(0.3)
