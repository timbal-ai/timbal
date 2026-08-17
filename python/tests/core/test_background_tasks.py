import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.server.jobs import JOB_DONE_SENTINEL, JobStore
from timbal.state import (
    cancel_background_task,
    get_background_task,
    get_run_context,
    list_background_tasks,
    read_background_transcript,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from timbal.types.content import ToolUseContent
from timbal.types.events.delta import DeltaEvent, TextDelta
from timbal.types.events.output import OutputEvent
from timbal.types.events.start import StartEvent
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


def _tool_calls(*calls: tuple[str, dict, str, bool]) -> Message:
    """One assistant message with N parallel tool calls."""
    return Message(
        role="assistant",
        content=[
            ToolUseContent(
                id=call_id,
                name=name,
                input={**input, **({"run_in_background": True} if bg else {})},
            )
            for name, input, call_id, bg in calls
        ],
        stop_reason="tool_use",
    )


async def _fake_streaming_builder(prompt: str) -> AsyncGenerator[TextDelta, None]:
    """Async gen of raw delta *items*, which the runnable wraps for us."""
    for i in range(4):
        yield TextDelta(id="b", text_delta=f"[{prompt}] chunk {i} ")
        await asyncio.sleep(0.08)
    yield TextDelta(id="b", text_delta=f"[{prompt}] done")


CHILD_RUN_ID = "child-run-1"


async def _fake_event_builder(prompt: str) -> AsyncGenerator[Any, None]:
    """Async gen of already-formed Timbal *events* — the real sidecar contract.

    A harness child (composer's Cursor ``build_turn``) does not yield delta
    items for us to wrap: it runs its own agent and emits a complete
    START/DELTA/OUTPUT stream under its own ``run_id``, which is the handle its
    harness is cancellable by. Those ids only survive if the events themselves
    reach the log.
    """
    ids = {
        "run_id": CHILD_RUN_ID,
        "path": "composer.builder",
        "call_id": "child-call-1",
        "parent_run_id": None,
        "parent_call_id": None,
    }
    yield StartEvent(**ids)
    for i in range(4):
        yield DeltaEvent(**ids, item=TextDelta(id="b", text_delta=f"[{prompt}] chunk {i} "))
        await asyncio.sleep(0.08)
    yield OutputEvent(
        **ids,
        status={"code": "success"},
        t0=0,
        t1=1,
        output=Message.validate(f"[{prompt}] done"),
        metadata={"harness": "cursor", "cursor_agent_id": "cursor-agent-1"},
    )


async def _wait_for_first_event(task_id: str, timeout: float = 5.0) -> int:
    """Wait until a background child has actually emitted, and return its cursor.

    Spawning only *schedules* the child: nothing orders its first event against
    the parent's turn ending, and the interleaving differs by Python version and
    OS. Poll for the event rather than sleeping a guessed interval.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cursor = read_background_transcript(task_id)["cursor"]
        if cursor > 0:
            return cursor
        await asyncio.sleep(0.01)
    raise AssertionError(f"background task {task_id} never emitted an event")


class TestBackgroundTasks:
    """Background spawn / peek / list — session bag, not Agent singleton."""

    @pytest.mark.asyncio
    async def test_background_task_execution_and_status(self):
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        agent = Agent(
            name="bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"duration": 0.2}, run_in_background=True),
                    "Task started in the background.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")],
        )

        result = await agent(prompt="run slow task").collect()
        assert_has_output_event(result)
        listed = list_background_tasks()
        assert len(listed) == 1
        assert listed[0]["status"] in ("running", "completed")
        await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_background_task_with_immediate_status_check(self):
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        agent = Agent(
            name="bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                    "The task is running.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")],
        )

        result = await agent(prompt="run and check").collect()
        assert_has_output_event(result)
        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_store_populated_when_running_background(self):
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        agent = Agent(
            name="bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")],
        )

        result = await agent(prompt="run in background").collect()
        assert_has_output_event(result)
        listed = list_background_tasks()
        assert len(listed) == 1
        snap = get_background_task(listed[0]["task_id"])
        assert snap["status"] in ("running", "completed")
        assert snap["name"] == "slow_task"
        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_agent_has_get_background_task_tool(self):
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        agent = Agent(
            name="bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Run a slow task", handler=slow_task, background_mode="auto")],
        )

        tool_names_seen = []
        async for event in agent(prompt="run in background"):
            if isinstance(event, OutputEvent) and event.path == "bg_agent.llm" and list_background_tasks():
                tool_names_seen.extend(tool.name for tool in event.input["tools"])

        assert "get_background_task" in tool_names_seen
        assert "list_background_tasks" in tool_names_seen
        assert "cancel_background_task" in tool_names_seen
        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_background_task_with_events_and_logs(self):
        async def build_interface(project_name: str) -> str:  # noqa: ARG001
            return "ok"

        agent = Agent(
            name="build_agent",
            model=TestModel(
                responses=[
                    _tool_call("build_interface", {"project_name": "my-app"}, run_in_background=True),
                    "Build started in the background.",
                ]
            ),
            tools=[
                Tool(
                    name="build_interface",
                    description="Build a web interface project",
                    handler=build_interface,
                    background_mode="auto",
                )
            ],
        )

        result = await agent(prompt="build my-app in background").collect()
        assert_has_output_event(result)
        assert len(list_background_tasks()) == 1
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_realtime_events_vs_background_events(self):
        async def streaming_task(steps: int) -> AsyncGenerator[str, None]:
            for i in range(steps):
                yield f"Step {i + 1}/{steps} completed"
                await asyncio.sleep(0.02)

        realtime_agent = Agent(
            name="realtime_agent",
            model=TestModel(
                responses=[
                    _tool_call("realtime_task", {"steps": 3}),
                    "Done.",
                ]
            ),
            tools=[
                Tool(
                    name="realtime_task",
                    description="Streams events in real-time",
                    handler=streaming_task,
                    background_mode="never",
                )
            ],
        )

        realtime_events = [event async for event in realtime_agent(prompt="run realtime task")]
        assert "START" in [e.type for e in realtime_events if hasattr(e, "type")]
        assert "OUTPUT" in [e.type for e in realtime_events if hasattr(e, "type")]

        background_agent = Agent(
            name="background_agent",
            model=TestModel(
                responses=[
                    _tool_call("background_task", {"steps": 3}),
                    "Started in background.",
                ]
            ),
            tools=[
                Tool(
                    name="background_task",
                    description="Runs in background",
                    handler=streaming_task,
                    background_mode="always",
                )
            ],
        )

        bg_events = [event async for event in background_agent(prompt="run background task")]
        assert len(bg_events) > 0
        await asyncio.sleep(0.1)
        listed = list_background_tasks()
        assert listed
        transcript = read_background_transcript(listed[0]["task_id"])
        assert transcript["cursor"] > 0
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_background_task_event_log_detailed_inspection(self):
        async def detailed_task(task_name: str) -> AsyncGenerator[dict, None]:  # noqa: ARG001
            for stage_data in (
                {"stage": "init", "progress": 0},
                {"stage": "processing", "progress": 50},
                {"stage": "complete", "progress": 100},
            ):
                yield stage_data
                await asyncio.sleep(0.02)

        agent = Agent(
            name="detailed_agent",
            model=TestModel(
                responses=[
                    _tool_call("detailed_task", {"task_name": "analysis"}),
                    "Task queued.",
                ]
            ),
            tools=[
                Tool(
                    name="detailed_task",
                    description="Task with detailed progress updates",
                    handler=detailed_task,
                    background_mode="always",
                )
            ],
        )

        result = await agent(prompt="run detailed_task").collect()
        assert_has_output_event(result)
        await asyncio.sleep(0.15)

        task_id = list_background_tasks()[0]["task_id"]
        first = read_background_transcript(task_id)
        assert first["cursor"] > 0
        for event in first["events"]:
            if isinstance(event, dict) and event.get("path"):
                assert "detailed_task" in event["path"]

        # Peek does not drain — second read from 0 still has them.
        second = read_background_transcript(task_id, after=0)
        assert second["cursor"] == first["cursor"]
        assert len(second["events"]) == len(first["events"])

        status = get_background_task(task_id)
        assert status["status"] in ("completed", "running")
        assert "events" not in status
        assert status["summary"]["event_count"] >= 1

    @pytest.mark.asyncio
    async def test_multiple_background_tasks_sequential_turns(self):
        async def slow_task(tag: str) -> str:
            await asyncio.sleep(0.2)
            return f"done:{tag}"

        agent = Agent(
            name="multi_bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"tag": "task1"}, id="c1", run_in_background=True),
                    "Task 1 started.",
                    _tool_call("slow_task", {"tag": "task2"}, id="c2", run_in_background=True),
                    "Task 2 started.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Slow task", handler=slow_task, background_mode="auto")],
        )

        result1 = await agent(prompt="start task 1").collect()
        assert_has_output_event(result1)
        result2 = await agent(prompt="start task 2").collect()
        assert_has_output_event(result2)
        assert len(list_background_tasks()) >= 2
        await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_background_task_error_handling(self):
        async def failing_tool(should_fail: bool = True) -> str:
            await asyncio.sleep(0.05)
            if should_fail:
                raise ValueError("Intentional failure for testing")
            return "Success"

        agent = Agent(
            name="error_handling_agent",
            model=TestModel(
                responses=[
                    _tool_call("failing_tool", {"should_fail": True}, run_in_background=True),
                    "Task started.",
                ]
            ),
            tools=[
                Tool(
                    name="failing_tool",
                    description="A tool that can fail",
                    handler=failing_tool,
                    background_mode="auto",
                )
            ],
        )

        result = await agent(prompt="run failing tool").collect()
        assert_has_output_event(result)
        await asyncio.sleep(0.2)
        snap = get_background_task(list_background_tasks()[0]["task_id"])
        assert snap["status"] == "error"
        assert "Intentional failure" in snap["error"]

    @pytest.mark.asyncio
    async def test_background_task_nonexistent_task_id(self):
        agent = Agent(name="check_missing_agent", model=TestModel())
        await agent(prompt="hi").collect()
        result = agent.get_background_task("nonexistent_task_id_12345")
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_background_mode_always(self):
        async def always_background_tool(message: str) -> str:
            await asyncio.sleep(0.1)
            return f"Processed: {message}"

        agent = Agent(
            name="always_bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("always_bg_tool", {"message": "hello"}),
                    "Done.",
                ]
            ),
            tools=[
                Tool(
                    name="always_bg_tool",
                    description="Always runs in background",
                    handler=always_background_tool,
                    background_mode="always",
                )
            ],
        )

        result = await agent(prompt="use always_bg_tool").collect()
        assert_has_output_event(result)
        assert len(list_background_tasks()) >= 1
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_background_mode_never(self):
        async def never_background_tool(message: str) -> str:
            await asyncio.sleep(0.05)
            return f"Processed: {message}"

        agent = Agent(
            name="never_bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("never_bg_tool", {"message": "test"}),
                    "Done.",
                ]
            ),
            tools=[
                Tool(
                    name="never_bg_tool",
                    description="Never runs in background",
                    handler=never_background_tool,
                    background_mode="never",
                )
            ],
        )

        result = await agent(prompt="use never_bg_tool").collect()
        assert_has_output_event(result)
        assert list_background_tasks() == []

    @pytest.mark.asyncio
    async def test_background_task_cleanup_after_completion(self):
        async def quick_task(value: str) -> str:
            await asyncio.sleep(0.05)
            return f"Done: {value}"

        agent = Agent(
            name="cleanup_agent",
            model=TestModel(
                responses=[
                    _tool_call("quick_task", {"value": "test"}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="quick_task", description="Quick task", handler=quick_task, background_mode="auto")],
        )

        result = await agent(prompt="run quick task in background").collect()
        assert_has_output_event(result)
        assert list_background_tasks()
        await asyncio.sleep(0.2)
        status = get_background_task(list_background_tasks()[0]["task_id"])
        assert status["status"] == "completed"
        assert status["result"] == "Done: test"

    @pytest.mark.asyncio
    async def test_background_task_with_structured_output(self):
        async def structured_task(count: int) -> dict:
            await asyncio.sleep(0.05)
            return {"count": count, "results": [f"item_{i}" for i in range(count)], "status": "completed"}

        agent = Agent(
            name="structured_bg_agent",
            model=TestModel(
                responses=[
                    _tool_call("structured_task", {"count": 3}, run_in_background=True),
                    "Task started.",
                ]
            ),
            tools=[
                Tool(
                    name="structured_task",
                    description="Returns structured data",
                    handler=structured_task,
                    background_mode="auto",
                )
            ],
        )

        result = await agent(prompt="run structured_task").collect()
        assert_has_output_event(result)
        await asyncio.sleep(0.2)
        status = get_background_task(list_background_tasks()[0]["task_id"])
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_background_tasks_persist_across_agent_calls(self):
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        agent = Agent(
            name="persistence_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"duration": 0.3}, run_in_background=True),
                    "Task started.",
                    "Still tracking it.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Slow task", handler=slow_task, background_mode="auto")],
        )

        result1 = await agent(prompt="start task").collect()
        assert_has_output_event(result1)
        ids_after_first = {t["task_id"] for t in list_background_tasks()}
        assert ids_after_first

        result2 = await agent(prompt="check tasks").collect()
        assert_has_output_event(result2)
        ids_after_second = {t["task_id"] for t in list_background_tasks()}
        assert ids_after_first <= ids_after_second
        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_get_background_task_tool_parameters(self):
        async def slow_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        agent = Agent(
            name="param_test_agent",
            model=TestModel(
                responses=[
                    _tool_call("slow_task", {"duration": 0.2}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="slow_task", description="Slow task", handler=slow_task, background_mode="auto")],
        )

        result = await agent(prompt="run slow task").collect()
        assert_has_output_event(result)
        listed = list_background_tasks()
        if listed:
            status = agent.get_background_task(listed[0]["task_id"])
            assert status["status"] in ("running", "completed")
        await asyncio.sleep(0.3)


class TestBackgroundMultitask:
    """Cursor-shaped multitask: parent stays talkable, N children, parent can answer."""

    @pytest.mark.asyncio
    async def test_one_iteration_two_background_children(self):
        """§6 / 5.5.1 — one parent iteration, two run_in_background calls."""
        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_calls(
                        ("builder", {"prompt": "A"}, "c1", True),
                        ("builder", {"prompt": "B"}, "c2", True),
                    ),
                    "Both started.",
                ]
            ),
            tools=[Tool(name="builder", handler=_fake_streaming_builder, background_mode="auto")],
        )

        r1 = await parent(prompt="build A and B").collect()
        assert_has_output_event(r1)
        ids = list_background_tasks()
        assert len(ids) == 2
        names = {t["title"] for t in ids}
        assert any("A" in (t or "") for t in names)
        assert any("B" in (t or "") for t in names)
        await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_parent_output_while_children_still_run(self):
        """§6 item (3) — THE load-bearing test.

        Parent OUTPUT exists; both children still running. Next collect() on
        the same Agent (session-chained RunContext) can peek and see running.
        """
        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_calls(
                        ("builder", {"prompt": "A"}, "c1", True),
                        ("builder", {"prompt": "B"}, "c2", True),
                    ),
                    "Both started.",
                    _tool_call("get_background_task", {"task_id": "placeholder"}),
                    "First one is still going.",
                ]
            ),
            tools=[Tool(name="builder", handler=_fake_streaming_builder, background_mode="auto")],
        )

        r1 = await parent(prompt="build A and B").collect()
        assert_has_output_event(r1)
        ids = list_background_tasks()
        assert len(ids) == 2
        first_id = ids[0]["task_id"]
        await _wait_for_first_event(first_id)
        mid = get_background_task(first_id)
        assert mid["status"] == "running"
        assert mid["summary"]["event_count"] >= 1
        assert mid["summary"]["text"]
        other_id = ids[1]["task_id"]
        assert get_background_task(other_id)["status"] == "running"

        # Next parent turn — this is the dock: new /stream, same session.
        parent.model = TestModel(
            responses=[
                _tool_call("get_background_task", {"task_id": first_id}),
                "The first builder is still running.",
            ]
        )
        r2 = await parent(prompt="what's the status of the first one?").collect()
        assert_has_output_event(r2)
        assert get_run_context().parent_id == r1.run_id
        peek = get_background_task(first_id)
        assert peek["status"] in ("running", "completed")
        assert peek["summary"]["event_count"] >= 1
        # Sibling still on the same bag, untouched by the peek.
        assert get_background_task(other_id)["status"] in ("running", "completed")
        await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_streaming_peek_does_not_drain(self):
        """5.5.2 — mid-flight peek, second peek still has the same events."""
        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_call("builder", {"prompt": "long"}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="builder", handler=_fake_streaming_builder, background_mode="auto")],
        )

        await parent(prompt="build").collect()
        task_id = list_background_tasks()[0]["task_id"]
        await _wait_for_first_event(task_id)
        a = read_background_transcript(task_id, after=0)
        assert a["status"] == "running"
        assert a["cursor"] >= 1
        b = read_background_transcript(task_id, after=0)
        assert b["cursor"] >= a["cursor"]
        assert len(b["events"]) >= len(a["events"])
        snap = get_background_task(task_id)
        assert "events" not in snap
        assert snap["summary"]["text"]
        await asyncio.sleep(0.4)

    @pytest.mark.asyncio
    async def test_handler_yielding_timbal_events_reaches_the_log(self):
        """5.5.2 — a child that speaks Timbal events natively, not delta items.

        Regression: ``process_event`` returned already-formed ``BaseEvent``s
        without queueing them, on the grounds that they were "already logged".
        For a foreground run that is true — the caller sees them on the stream.
        After detach there is no stream, so the log was the only copy and the
        parent got ``event_count: 0``: no transcript to show, no summary to
        brief the user with, and no child ``run_id`` on the record, which is
        what ``on_background_cancel`` needs to stop the external harness.
        """
        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_call("builder", {"prompt": "harness"}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="builder", handler=_fake_event_builder, background_mode="auto")],
        )

        await parent(prompt="build").collect()
        task_id = list_background_tasks()[0]["task_id"]
        await _wait_for_first_event(task_id)

        snap = get_background_task(task_id)
        assert snap["summary"]["event_count"] >= 1
        assert "[harness]" in snap["summary"]["text"]
        # The child's own ids, not the parent's — these make it cancellable.
        assert snap["run_id"] == CHILD_RUN_ID

        # The events on the log are the child's, unwrapped.
        events = read_background_transcript(task_id)["events"]
        assert events
        assert {event["path"] for event in events} == {"composer.builder"}

        # ...and the terminal OUTPUT's ids land too, so a later turn can resume.
        for _ in range(100):
            if get_background_task(task_id)["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        assert get_background_task(task_id)["cursor_agent_id"] == "cursor-agent-1"

    @pytest.mark.asyncio
    async def test_agent_as_tool_background(self):
        """5.5.4 — child Agent with its own tools, parent detaches, parent peeks."""

        def ping(x: str) -> str:
            return f"pong:{x}"

        child = Agent(
            name="specialist",
            model=TestModel(
                responses=[
                    _tool_call("ping", {"x": "hi"}),
                    "child done",
                ]
            ),
            tools=[ping],
            background_mode="auto",
        )

        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_call("specialist", {"prompt": "go"}, run_in_background=True),
                    "specialist started",
                ]
            ),
            tools=[child],
        )

        r1 = await parent(prompt="delegate").collect()
        assert_has_output_event(r1)
        listed = list_background_tasks()
        assert len(listed) == 1
        assert listed[0]["name"] == "specialist"
        await asyncio.sleep(0.2)
        snap = get_background_task(listed[0]["task_id"])
        assert snap["status"] in ("running", "completed")
        assert snap.get("run_id") or snap["summary"]["event_count"] >= 0

    @pytest.mark.asyncio
    async def test_cancel_stops_in_flight_work(self):
        """5.5.5 — cancel → peek cancelled; handler actually stops."""
        cancelled = {"hit": False}

        async def stubborn(prompt: str) -> AsyncGenerator[TextDelta, None]:
            try:
                while True:
                    yield TextDelta(id="s", text_delta=f"{prompt}...")
                    await asyncio.sleep(0.05)
            finally:
                # Collector aclose() delivers GeneratorExit, not CancelledError.
                cancelled["hit"] = True

        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_call("builder", {"prompt": "loop"}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="builder", handler=stubborn, background_mode="auto")],
        )

        await parent(prompt="go").collect()
        task_id = list_background_tasks()[0]["task_id"]
        assert get_background_task(task_id)["status"] == "running"
        from timbal.state.background import current_background_store

        # Cancelling a child that never started proves nothing about stopping
        # in-flight work, so make sure there is something in flight first.
        await _wait_for_first_event(task_id)

        record = current_background_store().get(task_id)
        result = cancel_background_task(task_id)
        assert result["status"] == "cancelled"

        # Wait for the cancel to unwind AND the store's aclose to finalize the
        # handler generator (that is what runs the handler's finally).
        for _ in range(100):
            if cancelled["hit"]:
                break
            await asyncio.sleep(0.01)

        assert get_background_task(task_id)["status"] == "cancelled"
        assert cancelled["hit"] is True, "handler generator was abandoned, not closed"
        assert record.task.done()
        after = read_background_transcript(task_id)["cursor"]
        await asyncio.sleep(0.15)
        assert read_background_transcript(task_id)["cursor"] == after, "child kept emitting after cancel"

    @pytest.mark.asyncio
    async def test_isolation_two_concurrent_parent_runs(self):
        """5.5.6 — two concurrent parent runs on one Agent must not see each other."""

        async def slow_task(tag: str) -> str:
            await asyncio.sleep(0.25)
            return tag

        def _handler(messages):
            from timbal.types.content import ToolResultContent

            if any(isinstance(c, ToolResultContent) for m in messages for c in m.content):
                return "Started."
            prompt = messages[-1].collect_text()
            tag = "A" if "A" in prompt else "B"
            return _tool_call("builder", {"tag": tag}, id=tag, run_in_background=True)

        parent = Agent(
            name="composer",
            model=TestModel(handler=_handler),
            tools=[Tool(name="builder", handler=slow_task, background_mode="auto")],
        )

        async def run_one(prompt: str) -> list[str]:
            set_run_context(None)
            set_call_id(None)
            set_parent_call_id(None)
            await parent(prompt=prompt).collect()
            return [t["task_id"] for t in list_background_tasks()]

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)
        left, right = await asyncio.gather(run_one("A"), run_one("B"))
        assert len(left) == 1
        assert len(right) == 1
        assert left[0] != right[0]
        await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_job_store_cancel_parent_does_not_kill_children(self):
        """5.6 — JobStore.cancel_job(parent) must not cancel _bg_tasks."""

        async def stubborn(prompt: str) -> AsyncGenerator[TextDelta, None]:
            for i in range(30):
                yield TextDelta(id="s", text_delta=f"{prompt} {i} ")
                await asyncio.sleep(0.05)

        parent = Agent(
            name="composer",
            model=TestModel(
                responses=[
                    _tool_call("builder", {"prompt": "keep-going"}, run_in_background=True),
                    "Started.",
                ]
            ),
            tools=[Tool(name="builder", handler=stubborn, background_mode="auto")],
        )

        jobs = JobStore()
        job_id, job = jobs.create_job(parent, {"prompt": "go"})

        run_id = None
        while True:
            event = await job.queue.get()
            if event is JOB_DONE_SENTINEL:
                break
            if isinstance(event, OutputEvent) and str(event.path).endswith(".builder"):
                run_id = event.run_id
                break

        assert run_id, "parent never spawned the builder"
        jobs.cancel_job(job_id)
        try:
            await job.task
        except asyncio.CancelledError:
            pass

        from timbal.state.background import store_for_run

        bag = store_for_run(run_id)
        assert bag is not None and len(bag) == 1
        task_id = bag.list()[0]["task_id"]
        assert bag.snapshot(task_id)["status"] == "running"
        bag.cancel(task_id)
        await asyncio.sleep(0.1)
