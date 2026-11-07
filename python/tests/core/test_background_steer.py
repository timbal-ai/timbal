"""Steering background sub-agents: messages a running background agent reads between its steps."""

import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, list_background_tasks, message_background_task
from timbal.state.background import store_for_run
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def _call(name: str, input: dict, id: str = "c1") -> Message:
    return Message(role="assistant", content=[ToolUseContent(id=id, name=name, input=input)], stop_reason="tool_use")


def _user_texts(messages: list[Message]) -> list[str]:
    return [m.collect_text() for m in messages if m.role == "user" and not m.is_runtime() and m.collect_text()]


class Child:
    """A researcher that runs one slow step, then answers with every instruction it has."""

    def __init__(self, steps: int = 1):
        self.steps = steps
        self.calls = 0
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.seen: list[list[Message]] = []

    async def slow(self) -> str:
        self.started.set()
        await self.gate.wait()
        self.gate.clear()
        return "step done"

    def model(self, messages: list[Message]) -> Message | str:
        self.seen.append(list(messages))
        self.calls += 1
        if self.calls <= self.steps:
            return _call("slow", {}, id=f"s{self.calls}")
        return "Did: " + " | ".join(_user_texts(messages))

    def agent(self, **kwargs) -> Agent:
        return Agent(
            name="researcher",
            model=TestModel(handler=self.model),
            tools=[Tool(name="slow", handler=self.slow)],
            background_mode="always",
            **kwargs,
        )


async def _spawn(child: Child, parent_responses: list, **child_kwargs) -> tuple[Agent, str, str]:
    parent = Agent(
        name="lead",
        model=TestModel(responses=[_call("researcher", {"prompt": "research cats"}), *parent_responses]),
        tools=[child.agent(**child_kwargs)],
    )
    result = await parent(prompt="go").collect()
    assert result.status.code == "success"
    (task,) = list_background_tasks()
    await asyncio.wait_for(child.started.wait(), 2)
    return parent, result.run_id, task["task_id"]


async def _finish(run_id: str, task_id: str) -> dict:
    store = store_for_run(run_id)
    await asyncio.wait_for(asyncio.shield(store.get(task_id).task), 5)
    return store.snapshot(task_id)


class TestMessageTask:
    @pytest.mark.asyncio
    async def test_the_running_agent_reads_it_after_its_step(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        assert store.list()[0]["takes_messages"] is True

        receipt = store.message_task(task_id, "focus on tigers")
        assert receipt.state == "pending"
        child.gate.set()
        snap = await _finish(run_id, task_id)

        assert receipt.state == "delivered"
        assert snap["status"] == "completed"
        assert _answer(snap) == "Did: research cats | focus on tigers"
        # It came in after the tool result, as a user message, before the answer.
        last = child.seen[-1]
        assert last[-1].role == "user" and last[-1].metadata.get("steered") is True
        assert last[-1].metadata["sources"] == ["app"]
        assert last[-2].content[0].type == "tool_result"

    @pytest.mark.asyncio
    async def test_several_messages_join_as_one_and_restore_the_budget(self):
        child = Child(steps=2)
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        first = store.message_task(task_id, "tigers")
        second = store.message_task(task_id, "and lions", source="user")
        child.gate.set()
        await asyncio.wait_for(_until(lambda: first.state == "delivered"), 2)
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert second.state == "delivered"
        assert _answer(snap) == "Did: research cats | tigers\n\nand lions"
        span = _task_span(run_id, task_id)
        assert span.metadata["task_messages"] == [first.id, second.id]

    @pytest.mark.asyncio
    async def test_a_message_the_task_never_read_is_reported_with_its_completion(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        record = store.get(task_id)
        take = record.take_messages
        receipts = []

        def take_messages():
            taken = take()
            if child.calls > child.steps:  # its last look, before the final answer: the message lands after it
                receipts.append(store.message_task(task_id, "one more thing"))
            return taken

        record.take_messages = take_messages
        child.gate.set()
        await _finish(run_id, task_id)
        (receipt,) = receipts
        assert receipt.state == "undelivered"
        (note,) = store.pending_completions()
        assert note["unread_messages"] == ["one more thing"]
        from timbal.state.background import format_background_completion_notice

        text = format_background_completion_notice([note]).collect_text()
        assert "finished before reading your message: one more thing" in text

    @pytest.mark.asyncio
    async def test_rejections(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        with pytest.raises(LookupError):
            store.message_task("nope", "x")
        with pytest.raises(ValueError, match="non-empty"):
            store.message_task(task_id, "  ")
        with pytest.raises(ValueError, match="limited"):
            store.message_task(task_id, "x" * 4001)
        child.gate.set()
        await _finish(run_id, task_id)
        with pytest.raises(ValueError, match="has finished"):
            store.message_task(task_id, "too late")

    @pytest.mark.asyncio
    async def test_a_tool_is_not_steerable(self):
        gate = asyncio.Event()

        async def build() -> str:
            await gate.wait()
            return "built"

        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("build", {}), "Started."]),
            tools=[Tool(name="build", handler=build, background_mode="always")],
        )
        result = await parent(prompt="go").collect()
        store = store_for_run(result.run_id)
        (task,) = store.list()
        assert "takes_messages" not in task
        with pytest.raises(ValueError, match="not an agent"):
            store.message_task(task["task_id"], "faster")
        assert message_background_task(task["task_id"], "faster")["status"] == "rejected"
        gate.set()

    @pytest.mark.asyncio
    async def test_steering_false_opts_an_agent_out(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."], steering=False)
        store = store_for_run(run_id)
        with pytest.raises(ValueError, match="not an agent"):
            store.message_task(task_id, "x")
        child.gate.set()
        await _finish(run_id, task_id)

    @pytest.mark.asyncio
    async def test_a_nested_agent_inside_the_task_does_not_take_its_messages(self):
        inner_seen: list[list[Message]] = []
        inner_gate = asyncio.Event()

        async def wait() -> str:
            await inner_gate.wait()
            return "waited"

        inner_calls = 0

        def inner_model(messages):
            nonlocal inner_calls
            inner_calls += 1
            inner_seen.append(list(messages))
            return _call("wait", {}, id="w1") if inner_calls == 1 else "inner done"

        inner = Agent(name="helper", model=TestModel(handler=inner_model), tools=[Tool(name="wait", handler=wait)])
        outer_calls = 0
        outer_seen: list[list[Message]] = []

        def outer_model(messages):
            nonlocal outer_calls
            outer_calls += 1
            outer_seen.append(list(messages))
            if outer_calls == 1:
                return _call("helper", {"prompt": "help"}, id="h1")
            return "Did: " + " | ".join(_user_texts(messages))

        outer = Agent(name="researcher", model=TestModel(handler=outer_model), tools=[inner], background_mode="always")
        parent = Agent(
            name="lead", model=TestModel(responses=[_call("researcher", {"prompt": "go"}), "Started."]), tools=[outer]
        )
        result = await parent(prompt="go").collect()
        store = store_for_run(result.run_id)
        task_id = store.list()[0]["task_id"]
        await asyncio.wait_for(_until(lambda: inner_calls == 1), 2)
        receipt = store.message_task(task_id, "tigers")
        inner_gate.set()
        snap = await _finish(result.run_id, task_id)
        assert receipt.state == "delivered"
        assert all("tigers" not in " ".join(_user_texts(seen)) for seen in inner_seen)
        assert _answer(snap) == "Did: go | tigers"

    @pytest.mark.asyncio
    async def test_the_root_now_cut_does_not_reach_the_child(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        ctx = _task_context(run_id, task_id)
        ctx._cut = asyncio.Event()
        ctx._cut.set()
        store_for_run(run_id).message_task(task_id, "tigers")
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert _answer(snap) == "Did: research cats | tigers"
        assert ctx._cut.is_set()  # a child's drain must not clear the root turn's cut


class TestParentTool:
    @pytest.mark.asyncio
    async def test_the_parent_messages_its_child_on_a_later_turn(self):
        child = Child()
        parent, run_id, task_id = await _spawn(child, ["Started."])
        calls = 0

        def lead(_messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _call("message_background_task", {"task_id": task_id, "message": "only tigers"}, id="m1")
            return "Told it."

        parent.model = TestModel(handler=lead)
        result = await parent(prompt="tell it: only tigers", parent_id=run_id).collect()
        assert result.status.code == "success"
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert _answer(snap) == "Did: research cats | only tigers"
        steered = child.seen[-1][-1]
        assert steered.metadata["sources"] == ["parent"]

    @pytest.mark.asyncio
    async def test_the_tool_is_not_offered_for_plain_tools(self):
        gate = asyncio.Event()

        async def build() -> str:
            await gate.wait()
            return "built"

        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("build", {}), "Started."]),
            tools=[Tool(name="build", handler=build, background_mode="always")],
        )
        await parent(prompt="go").collect()
        tools, _ = await _resolve(parent)
        names = {tool.name for tool in tools}
        assert "cancel_background_task" in names and "message_background_task" not in names
        gate.set()

    @pytest.mark.asyncio
    async def test_the_tool_is_offered_once_a_task_takes_messages(self):
        child = Child()
        parent, _, _ = await _spawn(child, ["Started."])
        tools, _ = await _resolve(parent)
        assert "message_background_task" in {tool.name for tool in tools}
        child.gate.set()


class TestChatSession:
    @pytest.mark.asyncio
    async def test_the_user_messages_a_running_sub_agent(self):
        from timbal.core.chat_session import ChatSession

        child = Child()
        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("researcher", {"prompt": "research cats"}), "Started."]),
            tools=[child.agent()],
        )
        async with ChatSession(parent) as session:
            session.send("go")
            await asyncio.wait_for(child.started.wait(), 2)
            (task,) = session.store.list()
            receipt = session.message_task(task["task_id"], "tigers")
            child.gate.set()
            assert await asyncio.wait_for(receipt.settled(), 2) == "delivered"
            await asyncio.wait_for(asyncio.shield(session.store.get(task["task_id"]).task), 2)
            assert _answer(session.store.snapshot(task["task_id"])) == "Did: research cats | tigers"
            assert child.seen[-1][-1].metadata["sources"] == ["user"]
            with pytest.raises(ValueError, match="has finished"):
                session.message_task(task["task_id"], "again")


async def _until(predicate, interval: float = 0.01) -> None:
    while not predicate():
        await asyncio.sleep(interval)


async def _resolve(agent: Agent):
    return await agent._resolve_tools(0)


def _task_context(run_id: str, task_id: str):
    store = store_for_run(run_id)
    record = store.get(task_id)
    ctx = get_run_context()
    assert ctx is not None and ctx._trace.get(record.call_id) is not None
    return ctx


def _task_span(run_id: str, task_id: str):
    record = store_for_run(run_id).get(task_id)
    return _task_context(run_id, task_id)._trace.get(record.call_id)


def _answer(snap: dict) -> str:
    result = snap["result"]
    return getattr(result, "output", result).collect_text()
