"""``Runnable.timeout`` — a wall-clock deadline on one foreground call."""

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.errors import RunTimeout
from timbal.types.content import ToolResultContent, ToolUseContent
from timbal.types.events.delta import TextDelta
from timbal.types.message import Message


def _tool_call(tool_name: str, input: dict, *, id: str = "c1") -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=id, name=tool_name, input=input)],
        stop_reason="tool_use",
    )


async def _slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "slept"


def _stuck_agent(*, timeout: float | None, name: str = "worker") -> Agent:
    return Agent(
        name=name,
        model=TestModel(responses=[_tool_call("slow", {"seconds": 5.0}), "Done."]),
        tools=[Tool(name="slow", handler=_slow)],
        timeout=timeout,
    )


class TestAgentTimeout:
    async def test_times_out_mid_tool(self):
        agent = _stuck_agent(timeout=0.2)

        t0 = time.monotonic()
        result = await agent(prompt="go").collect()

        assert time.monotonic() - t0 < 2.0
        assert result.status.code == "timeout"
        assert result.status.reason == "timeout"
        assert result.error["type"] == "RunTimeout"
        assert result.error["timeout"] == 0.2
        assert "timed out after 0.2s" in result.error["message"]

    async def test_finishes_inside_the_deadline(self):
        agent = Agent(
            name="quick",
            model=TestModel(responses=[_tool_call("slow", {"seconds": 0.01}), "Done."]),
            tools=[Tool(name="slow", handler=_slow)],
            timeout=5.0,
        )

        result = await agent(prompt="go").collect()

        assert result.status.code == "success"
        assert result.error is None
        assert result.output.collect_text() == "Done."

    @pytest.mark.parametrize("timeout", [None, 0, -1])
    async def test_unset_or_non_positive_means_no_deadline(self, timeout):
        agent = Agent(
            name="unbounded",
            model=TestModel(responses=[_tool_call("slow", {"seconds": 0.05}), "Done."]),
            tools=[Tool(name="slow", handler=_slow)],
            timeout=timeout,
        )

        result = await agent(prompt="go").collect()

        assert result.status.code == "success"

    async def test_timeout_is_mutable_config(self):
        agent = _stuck_agent(timeout=None)
        agent.timeout = 0.1

        result = await agent(prompt="go").collect()

        assert result.status.code == "timeout"

    async def test_next_turn_runs_after_a_timeout(self):
        """The interrupted tool_use must not poison the chained memory."""
        seen: list[Message] = []

        def model(messages: list[Message]) -> Message | str:
            seen[:] = messages
            if len(messages) == 1:
                return _tool_call("slow", {"seconds": 5.0})
            return "Back again."

        agent = Agent(
            name="resumable",
            model=TestModel(handler=model),
            tools=[Tool(name="slow", handler=_slow)],
            timeout=0.2,
        )

        first = await agent(prompt="go").collect()
        assert first.status.code == "timeout"

        second = await agent(prompt="try again").collect()
        assert second.status.code == "success"
        assert second.output.collect_text() == "Back again."
        results = [c for m in seen for c in m.content if isinstance(c, ToolResultContent)]
        assert [r.id for r in results] == ["c1"]

    async def test_parallel_tool_calls_are_all_stopped(self):
        agent = Agent(
            name="fanout",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id="a", name="slow", input={"seconds": 5.0}),
                            ToolUseContent(id="b", name="slow", input={"seconds": 5.0}),
                        ],
                        stop_reason="tool_use",
                    ),
                    "Done.",
                ]
            ),
            tools=[Tool(name="slow", handler=_slow)],
            timeout=0.2,
        )

        t0 = time.monotonic()
        result = await agent(prompt="go").collect()

        assert time.monotonic() - t0 < 2.0
        assert result.status.code == "timeout"
        leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        assert leftover == []


class TestSubAgentTimeout:
    async def test_parent_sees_an_error_tool_result_and_continues(self):
        seen: list[Message] = []

        def parent_model(messages: list[Message]) -> Message | str:
            seen[:] = messages
            if len(messages) == 1:
                return _tool_call("worker", {"prompt": "do the slow thing"})
            return "Worker timed out; answering without it."

        parent = Agent(
            name="parent",
            model=TestModel(handler=parent_model),
            tools=[_stuck_agent(timeout=0.2)],
        )

        result = await parent(prompt="delegate").collect()

        assert result.status.code == "success"
        assert result.output.collect_text() == "Worker timed out; answering without it."
        tool_results = [c for m in seen for c in m.content if isinstance(c, ToolResultContent)]
        assert len(tool_results) == 1
        assert "timed out after 0.2s" in str(tool_results[0].content)

    async def test_parent_timeout_bounds_the_whole_turn(self):
        parent = Agent(
            name="parent",
            model=TestModel(responses=[_tool_call("worker", {"prompt": "go"}), "Done."]),
            tools=[_stuck_agent(timeout=None)],
            timeout=0.2,
        )

        t0 = time.monotonic()
        result = await parent(prompt="delegate").collect()

        assert time.monotonic() - t0 < 2.0
        assert result.status.code == "timeout"
        assert result.error["message"].startswith("parent ")


class TestToolTimeout:
    async def test_coroutine_handler(self):
        tool = Tool(name="slow", handler=_slow, timeout=0.1)

        result = await tool(seconds=5.0).collect()

        assert result.status.code == "timeout"
        assert result.error["type"] == "RunTimeout"

    async def test_streaming_handler_keeps_partial_output(self):
        async def ticker() -> AsyncGenerator[TextDelta, None]:
            for i in range(100):
                yield TextDelta(id="t", text_delta=f"{i} ")
                await asyncio.sleep(0.02)

        tool = Tool(name="ticker", handler=ticker, timeout=0.15)

        result = await tool().collect()

        assert result.status.code == "timeout"
        assert result.output, "partial stream should survive the timeout"

    async def test_a_handlers_own_timeout_error_stays_an_error(self):
        async def flaky() -> str:
            raise TimeoutError("upstream read timed out")

        tool = Tool(name="flaky", handler=flaky, timeout=5.0)

        result = await tool().collect()

        assert result.status.code == "error"
        assert result.error["type"] == "TimeoutError"

    def test_run_timeout_is_a_timeout_error(self):
        err = RunTimeout("a.b", 1.5)
        assert isinstance(err, TimeoutError)
        assert str(err) == "a.b timed out after 1.5s"


class TestWorkflowStepTimeout:
    async def test_step_timeout_fails_the_step(self):
        workflow = Workflow(name="pipeline").step(Tool(name="slow", handler=_slow, timeout=0.1), seconds=5.0)

        result = await workflow().collect()

        assert result.status.code == "error"
        assert "timed out after 0.1s" in str(result.error)
