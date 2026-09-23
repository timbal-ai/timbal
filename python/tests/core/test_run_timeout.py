"""``Runnable.timeout`` — a wall-clock deadline on one foreground call."""

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.errors import RunTimeout
from timbal.state import get_run_context
from timbal.types.content import ToolResultContent, ToolUseContent
from timbal.types.events import DeltaEvent, OutputEvent
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

    @pytest.mark.parametrize("cancellation", ["caught", "nested_tool"])
    async def test_parallel_tools_do_not_resume_after_swallowed_cancellation(self, cancellation):
        """A final delta after cancellation must not resume the handler again."""
        started: set[str] = set()
        closed: set[str] = set()
        side_effects: list[str] = []
        waiting = asyncio.Event()
        inner = Tool(name="inner", handler=_slow)

        async def streaming(label: str) -> AsyncGenerator[TextDelta, None]:
            started.add(label)
            try:
                yield TextDelta(id=label, text_delta="ready")
                if cancellation == "nested_tool":
                    # Runnable turns CancelledError into an interrupted result.
                    # The enclosing handler must still stop at its next yield.
                    await inner(seconds=5.0).collect()
                else:
                    try:
                        await waiting.wait()
                    except asyncio.CancelledError:
                        pass
                yield TextDelta(id=label, text_delta="interrupted")
                side_effects.append(label)
            finally:
                # Cleanup can itself await and must finish before root OUTPUT.
                await asyncio.sleep(0)
                closed.add(label)

        agent = Agent(
            name="fanout",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id=label, name="streaming", input={"label": label}) for label in ("a", "b")
                        ],
                        stop_reason="tool_use",
                    ),
                    "Done.",
                ]
            ),
            tools=[streaming],
            timeout=0.1,
        )

        async def consume():
            result = None
            stream = agent(prompt="go")
            try:
                async for event in stream:
                    if isinstance(event, OutputEvent) and event.path == "fanout":
                        assert closed == {"a", "b"}, "timeout was reported before child cleanup"
                        result = event
            finally:
                await stream.aclose()
            return result

        result = await asyncio.wait_for(consume(), timeout=2.0)

        assert result is not None and result.status.code == "timeout"
        assert started == {"a", "b"}
        assert side_effects == [], "cancelled handlers resumed after yielding their final delta"

    @pytest.mark.parametrize("wrapper", ["direct", "agent", "workflow"])
    async def test_parallel_tools_are_closed_before_timeout_output_after_slow_consumer(self, wrapper):
        """Expiry between events must join child tasks before reporting timeout."""
        release = asyncio.Event()
        both_started = asyncio.Event()
        started: set[str] = set()
        closed: set[str] = set()
        side_effects: list[str] = []
        tool_tasks: set[asyncio.Task] = set()

        async def streaming(label: str) -> AsyncGenerator[TextDelta, None]:
            task = asyncio.current_task()
            assert task is not None
            tool_tasks.add(task)
            started.add(label)
            if len(started) == 2:
                both_started.set()
            try:
                yield TextDelta(id=label, text_delta="ready")
                await release.wait()
                side_effects.append(label)
            finally:
                closed.add(label)

        agent = Agent(
            name="fanout" if wrapper == "direct" else "worker",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id="a", name="streaming", input={"label": "a"}),
                            ToolUseContent(id="b", name="streaming", input={"label": "b"}),
                        ],
                        stop_reason="tool_use",
                    ),
                    "Done.",
                ]
            ),
            tools=[streaming],
            timeout=0.2 if wrapper == "direct" else None,
        )
        if wrapper != "direct":
            child = Workflow(name="worker").step(agent, prompt="go") if wrapper == "workflow" else agent
            child_input = {} if wrapper == "workflow" else {"prompt": "go"}
            agent = Agent(
                name="fanout",
                model=TestModel(responses=[_tool_call("worker", child_input), "Done."]),
                tools=[child],
                timeout=0.2,
            )
        stream = agent(prompt="go")
        delayed = False
        closed_at_timeout = None
        pending_at_timeout = None
        try:
            async for event in stream:
                if not delayed and isinstance(event, DeltaEvent) and event.path.endswith(".streaming"):
                    await asyncio.wait_for(both_started.wait(), timeout=1.0)
                    delayed = True
                    # The deadline expires while the consumer holds an event,
                    # not while the handler is being awaited.
                    await asyncio.sleep(0.25)
                if isinstance(event, OutputEvent) and event.path == "fanout":
                    assert event.status.code == "timeout"
                    closed_at_timeout = set(closed)
                    pending_at_timeout = [task for task in tool_tasks if not task.done()]

            assert delayed, "the test must reach the slow-consumer path"
            # Any work still alive after the terminal event can now run a side
            # effect. A correct timeout has already cancelled and joined it.
            release.set()
            await asyncio.wait_for(asyncio.gather(*tool_tasks, return_exceptions=True), timeout=1.0)
            assert closed_at_timeout == {"a", "b"}, (
                f"timeout was reported before child cleanup; post-timeout side effects: {side_effects}"
            )
            assert pending_at_timeout == []
            assert side_effects == []
        finally:
            # Keep a failed regression from leaking tasks
            # into later tests, even if an earlier assertion fails.
            for task in tool_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tool_tasks, return_exceptions=True)
            await stream.aclose()


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

    @pytest.mark.parametrize("hook_name", ["pre_hook", "post_hook"])
    async def test_foreground_timeout_includes_hooks(self, hook_name):
        """A slow hook must not outlive the whole-call deadline and succeed."""
        hook_completed = False
        handler_calls = 0

        async def slow_hook() -> None:
            nonlocal hook_completed
            await asyncio.sleep(0.2)
            hook_completed = True

        async def handler() -> str:
            nonlocal handler_calls
            handler_calls += 1
            return "Done."

        tool = Tool(name="hooked", handler=handler, timeout=0.05, **{hook_name: slow_hook})
        result = await tool().collect()

        assert result.status.code == "timeout", f"{hook_name} bypassed the foreground deadline"
        assert result.error["type"] == "RunTimeout"
        assert not hook_completed
        assert handler_calls == (0 if hook_name == "pre_hook" else 1)

    async def test_hooks_and_handler_share_one_deadline(self):
        """Each stage fits alone, but their combined work exceeds the budget."""
        post_started = False
        post_completed = False

        async def pre_hook() -> None:
            await asyncio.sleep(0.06)

        async def handler() -> str:
            await asyncio.sleep(0.06)
            return "Done."

        async def post_hook() -> None:
            nonlocal post_started, post_completed
            post_started = True
            await asyncio.sleep(0.14)
            post_completed = True

        tool = Tool(name="hooked", handler=handler, pre_hook=pre_hook, post_hook=post_hook, timeout=0.2)
        result = await tool().collect()

        assert result.status.code == "timeout"
        assert post_started
        assert not post_completed
        assert result.output == "Done."
        assert result._output_dump == "Done."

    async def test_post_hook_timeout_keeps_mutated_message_dump_consistent(self):
        async def post_hook() -> None:
            output = get_run_context().current_span().output
            output.content[0].text = "Updated by hook."
            await asyncio.sleep(0.2)

        agent = Agent(name="hooked", model=TestModel(responses=["Original."]), post_hook=post_hook, timeout=0.05)
        result = await agent(prompt="go").collect()

        assert result.status.code == "timeout"
        assert result.output.collect_text() == "Updated by hook."
        assert result._output_dump["content"][0]["text"] == "Updated by hook."


class TestWorkflowStepTimeout:
    async def test_step_timeout_fails_the_step(self):
        workflow = Workflow(name="pipeline").step(Tool(name="slow", handler=_slow, timeout=0.1), seconds=5.0)

        result = await workflow().collect()

        assert result.status.code == "error"
        assert "timed out after 0.1s" in str(result.error)


class TestNestedWorkflowTimeoutRegressions:
    @pytest.mark.parametrize("child_kind", ["tool", "agent"])
    @pytest.mark.parametrize("execution", ["standalone", "linear", "parallel"])
    async def test_agent_can_recover_from_child_timeout_inside_workflow(self, child_kind, execution):
        """A descendant error must not preempt the enclosing agent's recovery."""
        seen: list[Message] = []
        if child_kind == "tool":
            child = Tool(name="helper", handler=_slow, timeout=0.1)
            child_input = {"seconds": 5.0}
        else:
            child = _stuck_agent(name="helper", timeout=0.1)
            child_input = {"prompt": "go"}

        def model(messages: list[Message]) -> Message | str:
            seen[:] = messages
            if len(messages) == 1:
                return _tool_call("helper", child_input)
            return "Recovered without the child."

        worker = Agent(name="worker", model=TestModel(handler=model), tools=[child])

        async def independent() -> str:
            return "independent completed"

        if execution == "standalone":
            root = worker
            inputs = {"prompt": "go"}
            worker_path = "worker"
        else:
            root = Workflow(name="pipeline").step(worker, prompt="go")
            if execution == "parallel":
                root.step(independent)
            inputs = {}
            worker_path = "pipeline.worker"

        outputs: list[OutputEvent] = []

        async def consume():
            stream = root(**inputs)
            try:
                async for event in stream:
                    if isinstance(event, OutputEvent):
                        outputs.append(event)
            finally:
                await stream.aclose()

        await asyncio.wait_for(consume(), timeout=2.0)

        child_output = next(event for event in outputs if event.path == f"{worker_path}.helper")
        assert child_output.status.code == "timeout"
        assert child_output.error["type"] == "RunTimeout"
        assert outputs[-1].status.code == "success", (
            "the workflow consumed a descendant error before the agent could recover: "
            f"{[(event.path, event.status.code) for event in outputs]}"
        )
        worker_output = next(event for event in outputs if event.path == worker_path)
        assert worker_output.output.collect_text() == "Recovered without the child."
        tool_results = [
            content for message in seen for content in message.content if isinstance(content, ToolResultContent)
        ]
        assert len(tool_results) == 1
        assert "timed out after 0.1s" in str(tool_results[0].content)

    @pytest.mark.parametrize("step_kind", ["tool", "agent"])
    @pytest.mark.parametrize("execution", ["linear", "parallel"])
    async def test_timeout_prevents_dependent_and_transitive_steps(self, step_kind, execution):
        """Dependents must not produce side effects after their prerequisite fails."""
        effects: list[str] = []

        async def dependent() -> str:
            effects.append("dependent")
            return "should not run"

        async def transitive() -> str:
            effects.append("transitive")
            return "should not run either"

        async def independent() -> str:
            return "independent completed"

        if step_kind == "tool":
            step = Tool(name="slow", handler=_slow, timeout=0.1)
            inputs = {"seconds": 5.0}
        else:
            step = _stuck_agent(name="slow", timeout=0.1)
            inputs = {"prompt": "go"}
        workflow = (
            Workflow(name="pipeline")
            .step(step, **inputs)
            .step(dependent, depends_on=["slow"])
            .step(transitive, depends_on=["dependent"])
        )
        if execution == "parallel":
            workflow.step(independent)

        result = await asyncio.wait_for(workflow().collect(), timeout=2.0)

        assert result.status.code == "error"
        assert result.error["type"] == "RunTimeout"
        assert effects == [], f"steps ran after their prerequisite timed out: {effects}"

    @pytest.mark.parametrize("wrapper", ["direct", "agent", "workflow"])
    @pytest.mark.parametrize("expiry", ["awaiting_handler", "slow_consumer"])
    @pytest.mark.parametrize("cancellation", ["caught", "nested_tool"])
    async def test_parallel_workflow_stops_handlers_after_swallowed_cancellation(self, wrapper, expiry, cancellation):
        """Closing a timed-out workflow must not resume cancelled step handlers."""
        started: set[str] = set()
        cancelled: set[str] = set()
        closed: set[str] = set()
        effects: list[str] = []
        tasks: set[asyncio.Task] = set()
        both_started = asyncio.Event()
        never_release = asyncio.Event()
        inner = Tool(name="inner", handler=_slow)

        async def streaming(label: str) -> AsyncGenerator[TextDelta, None]:
            task = asyncio.current_task()
            assert task is not None
            tasks.add(task)
            started.add(label)
            if len(started) == 2:
                both_started.set()
            try:
                yield TextDelta(id=label, text_delta="ready")
                if cancellation == "nested_tool":
                    result = await inner(seconds=5.0).collect()
                    assert result.status.code == "cancelled"
                else:
                    try:
                        await never_release.wait()
                    except asyncio.CancelledError:
                        pass
                cancelled.add(label)
                yield TextDelta(id=label, text_delta="interrupted")
                effects.append(label)
            finally:
                await asyncio.sleep(0)
                closed.add(label)

        workflow = (
            Workflow(name="root" if wrapper == "direct" else "fanout", timeout=0.1 if wrapper == "direct" else None)
            .step(Tool(name="a", handler=streaming), label="a")
            .step(Tool(name="b", handler=streaming), label="b")
        )
        inputs = {}
        if wrapper == "agent":
            root = Agent(
                name="root",
                model=TestModel(responses=[_tool_call("fanout", {}), "Done."]),
                tools=[workflow],
                timeout=0.1,
            )
            inputs = {"prompt": "go"}
        elif wrapper == "workflow":
            root = Workflow(name="root", timeout=0.1).step(workflow)
        else:
            root = workflow
        stream = root(**inputs)
        delayed = False
        timeout_snapshot = None

        async def consume():
            nonlocal delayed, timeout_snapshot
            async for event in stream:
                if expiry == "slow_consumer" and not delayed and isinstance(event, DeltaEvent):
                    await asyncio.wait_for(both_started.wait(), timeout=1.0)
                    delayed = True
                    await asyncio.sleep(0.15)
                if isinstance(event, OutputEvent) and event.path == "root":
                    assert event.status.code == "timeout"
                    timeout_snapshot = (set(closed), [task for task in tasks if not task.done()], list(effects))

        try:
            await asyncio.wait_for(consume(), timeout=2.0)
            assert started == cancelled == {"a", "b"}, "both children must reach the cancellation path"
            assert delayed == (expiry == "slow_consumer")
            assert timeout_snapshot is not None
            closed_at_timeout, pending_at_timeout, effects_at_timeout = timeout_snapshot
            assert closed_at_timeout == {"a", "b"}, "timeout was reported before asynchronous child cleanup"
            assert pending_at_timeout == [], "child tasks outlived the root timeout event"
            assert effects_at_timeout == effects == [], f"cancelled workflow handlers resumed after yielding: {effects}"
        finally:
            # Leave no tasks or suspended root generators behind on failure.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await stream.aclose()
