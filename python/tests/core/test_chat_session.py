"""ChatSession and steering: messages sent while a turn runs, and what each mode does with them."""

import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.session_scheduler import take_followup, user_batch
from timbal.core.test_model import TestModel
from timbal.guardrails import Verdict, guardrail
from timbal.state import post_session_event, set_call_id, set_parent_call_id, set_run_context
from timbal.state.background import BackgroundTaskStore
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import USER_MESSAGE_KIND, Message


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def tool_call(name, call_id=None):
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id or f"call-{name}", name=name, input={})],
        stop_reason="tool_use",
    )


def last_user_text(messages):
    for message in reversed(messages):
        if message.role == "user" and not message.is_runtime():
            return message.collect_text()
    return ""


def echo(messages):
    """``start`` / ``slow`` call a tool; anything else is echoed; a tool result is answered ``done``."""
    last = messages[-1]
    if last.role == "tool":
        return "done"
    text = last_user_text(messages)
    if text in ("start", "slow"):
        return tool_call("work" if text == "start" else "forever", call_id=f"call-{len(messages)}")
    return f"echo: {text}"


class Gate:
    """A tool that waits until released, so a test can act while a turn is mid-tool."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def work(self) -> str:
        self.entered.set()
        await self.release.wait()
        return "worked"


def make_agent(gate, *, model=None, **kwargs):
    async def forever() -> str:
        gate.entered.set()
        await asyncio.Event().wait()
        return "never"

    return Agent(
        name="agent",
        model=model or TestModel(handler=echo),
        tools=[Tool(name="work", handler=gate.work), Tool(name="forever", handler=forever)],
        tracing_provider=InMemoryTracingProvider,
        **kwargs,
    )


async def records(session, kind, count, timeout=3.0):
    """The first ``count`` SESSION records of ``kind`` in the log (waits for them)."""
    found = []

    async def collect():
        async for item in session.events():
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), timeout)
    return found


def root_outputs(session):
    events, _ = session.log.peek()
    return [
        event
        for event in events
        if isinstance(event, OutputEvent) and event.parent_call_id is None and event.path == "agent"
    ]


async def stored_memory(output):
    trace = await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(output.run_id)
    return [Message.validate(message) for message in trace[output.call_id].memory]


class TestPostUserMessage:
    def test_validation(self):
        store = BackgroundTaskStore()
        with pytest.raises(ValueError, match="mode="):
            store.post_event("x", source="user", lane="next")
        with pytest.raises(ValueError, match="key"):
            store.post_event("x", source="user", key="k")
        with pytest.raises(ValueError, match="mode"):
            store.post_event("x", source="user", mode="later")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="user messages"):
            store.post_event("x", mode="steer")
        with pytest.raises(ValueError, match="lane"):
            store.post_event("x", lane="soon")  # type: ignore[arg-type]
        assert store.pending_events() == []

    def test_mode_picks_the_lane_and_long_text_is_fine(self):
        store = BackgroundTaskStore()
        steer = store.post_event("y" * 10_000, source="user", mode="steer")
        followup = store.post_event("z", source="user")
        assert (steer.lane, steer.mode) == ("next", "steer")
        assert (followup.lane, followup.mode) == ("later", "followup")
        assert followup.receipt()["mode"] == "followup"

    def test_on_settle_runs_synchronously_once(self):
        store = BackgroundTaskStore()
        event = store.post_event("x", source="user")
        seen = []
        event.on_settle(lambda e: seen.append((e.state, e.run_id)))
        store.drain_events(run_id="run-a")
        assert seen == [("delivered", "run-a")]
        event.on_settle(lambda e: seen.append(("late", e.state)))
        assert seen[-1] == ("late", "delivered")

    def test_user_batch(self):
        store = BackgroundTaskStore()
        a = store.post_event("a", source="user", mode="steer")
        b = store.post_event("b", source="user", mode="collect")
        c = store.post_event("c", source="user", mode="followup")
        d = store.post_event("d", source="user", mode="steer")
        app = store.post_event("e")
        assert user_batch(store.pending_events()) == [a, b]
        assert user_batch([c, d, app]) == [c]
        assert user_batch([app]) == []
        urgent = store.post_event("now", source="user", announce="interrupt")
        assert user_batch(store.pending_events()) == [urgent]

    def test_take_followup_starts_with_the_user_and_leaves_the_rest(self):
        store = BackgroundTaskStore()
        app = store.post_event("deploy finished")
        first = store.post_event("hello", source="user", mode="steer")
        second = store.post_event("and this", source="user", mode="collect")
        later = store.post_event("later", source="user", mode="followup")
        store._completion_inbox.append({"task_id": "t1", "name": "w", "status": "completed"})
        followup = take_followup(store)
        assert followup.events == [first, second] and followup.completions == []
        assert followup.prompt.collect_text() == "hello\n\nand this"
        assert followup.prompt.metadata["kind"] == USER_MESSAGE_KIND and not followup.prompt.is_runtime()
        assert followup.trigger["kind"] == "user_message"
        assert store.pending_events() == [app, later]
        assert len(store.pending_completions()) == 1


class TestSteering:
    async def test_steer_joins_the_running_turn_after_its_tool_batch(self):
        gate = Gate()
        model = TestModel(handler=echo)
        async with ChatSession(make_agent(gate, model=model)) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            steer = session.send("also check the logs")
            assert session.busy
            gate.release.set()
            finished = await records(session, "turn_finished", 1)
            assert await asyncio.wait_for(steer.settled(), 1) == "delivered"
            assert steer.run_id == finished[0]["run_id"]
            assert session.turns == 1 and model.call_count == 2
            (output,) = root_outputs(session)
            assert output.output.collect_text() == "echo: also check the logs"
            memory = await stored_memory(output)
            assert [m.role for m in memory] == ["user", "assistant", "tool", "user", "assistant"]
            assert memory[3].metadata["steered"] is True
            assert output.metadata["steered_events"] == [steer.id]
            # The delivery is on the log before the answer to it.
            log, _ = session.log.peek()
            settled_at = next(
                i
                for i, e in enumerate(log)
                if isinstance(e, dict) and e.get("event_id") == steer.id and e["kind"] == "message_settled"
            )
            answer_at = next(
                i
                for i, e in enumerate(log)
                if isinstance(e, OutputEvent)
                and e.path == "agent.llm"
                and i > settled_at - 1
                and e.output.collect_text() == "echo: also check the logs"
            )
            assert settled_at < answer_at

    async def test_a_message_that_arrives_during_the_final_answer_keeps_the_turn_going(self):
        entered, opened = asyncio.Event(), asyncio.Event()

        class Gated(TestModel):
            async def stream(self, messages, **kwargs):
                if self.call_count == 0:
                    entered.set()
                    await opened.wait()
                async for event in super().stream(messages, **kwargs):
                    yield event

        model = Gated(handler=echo)
        async with ChatSession(make_agent(Gate(), model=model)) as session:
            session.send("hi")
            await asyncio.wait_for(entered.wait(), 2)
            session.send("one more thing")
            opened.set()
            await records(session, "turn_finished", 1)
            assert session.turns == 1 and model.call_count == 2
            (output,) = root_outputs(session)
            assert output.output.collect_text() == "echo: one more thing"

    async def test_without_steering_a_steer_starts_the_next_turn(self):
        gate = Gate()
        async with ChatSession(make_agent(gate, steering=False)) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("first")
            session.send("second")
            gate.release.set()
            finished = await records(session, "turn_finished", 2)
            assert session.turns == 2
            outputs = root_outputs(session)
            assert [o.output.collect_text() for o in outputs] == ["done", "echo: first\n\nsecond"]
            started = await records(session, "turn_started", 2)
            assert started[1]["trigger"]["kind"] == "user_message"
            assert started[1]["parent_id"] == finished[0]["run_id"]

    async def test_steered_text_is_screened_like_a_prompt(self):
        gate = Gate()
        rail = guardrail(lambda text: Verdict.block("nope") if "forbidden" in text else True, stages=["input"])
        async with ChatSession(make_agent(gate, guardrails=[rail])) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("something forbidden")
            gate.release.set()
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["status"]["code"] == "blocked"

    async def test_a_plain_agent_run_leaves_user_messages_pending(self):
        agent = make_agent(Gate())
        first = await agent(prompt="hi").collect()
        message = post_session_event(first.run_id, "queued", source="user", mode="steer", create=True)
        app = post_session_event(first.run_id, "deploy finished")
        second = await agent(prompt="next", parent_id=first.run_id).collect()
        assert app.state == "delivered" and message.state == "pending"
        memory = await stored_memory(second)
        assert all("queued" not in m.collect_text() for m in memory)


class TestModes:
    async def test_followups_each_get_a_turn(self):
        gate = Gate()
        async with ChatSession(make_agent(gate)) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("q1", mode="followup")
            session.send("q2", mode="followup")
            gate.release.set()
            await records(session, "turn_finished", 3)
            assert [o.output.collect_text() for o in root_outputs(session)] == ["done", "echo: q1", "echo: q2"]

    async def test_collected_messages_share_one_turn_after_the_quiet_window(self):
        gate = Gate()
        async with ChatSession(make_agent(gate), collect_window=0.05) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("a", mode="collect")
            session.send("b", mode="collect")
            gate.release.set()
            await records(session, "turn_finished", 2)
            assert [o.output.collect_text() for o in root_outputs(session)] == ["done", "echo: a\n\nb"]

    async def test_interrupt_cuts_the_turn_and_goes_first(self):
        gate = Gate()
        async with ChatSession(make_agent(gate)) as session:
            session.send("slow")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("queued", mode="followup")
            session.send("stop, do this instead", mode="interrupt")
            finished = await records(session, "turn_finished", 3)
            assert finished[0]["status"] == {"code": "cancelled", "reason": "interrupted", "message": ""}
            texts = [o.output.collect_text() if o.output else None for o in root_outputs(session)[1:]]
            assert texts == ["echo: stop, do this instead", "echo: queued"]
            # The interrupted turn's history carries on.
            assert finished[1]["run_id"] != finished[0]["run_id"]
            started = await records(session, "turn_started", 2)
            assert started[1]["parent_id"] == finished[0]["run_id"]

    async def test_interrupt_with_nothing_running_is_a_plain_message(self):
        async with ChatSession(make_agent(Gate())) as session:
            assert session.interrupt() is False
            session.send("hello", mode="interrupt")
            await records(session, "turn_finished", 1)
            assert root_outputs(session)[0].output.collect_text() == "echo: hello"


class TestSessionLifecycle:
    async def test_turns_chain_and_history_carries(self):
        async with ChatSession(make_agent(Gate())) as session:
            session.send("one")
            first = await records(session, "turn_finished", 1)
            session.send("two")
            finished = await records(session, "turn_finished", 2)
            assert session.head == finished[1]["run_id"] != first[0]["run_id"]
            memory = await stored_memory(root_outputs(session)[1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["one", "two"]

    async def test_a_parked_turn_waits_for_resume(self):
        def charge() -> str:
            return "charged"

        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=charge, requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("pay")
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["parked"] and session.parked
            approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
            assert session.state()["pending_approvals"][0]["approval_id"] == approval.approval_id
            waiting = session.send("hello?", mode="followup")
            await asyncio.sleep(0.05)
            assert waiting.state == "pending" and session.turns == 1
            session.resume({approval.approval_id: True})
            await records(session, "turn_finished", 3)
            texts = [o.output.collect_text() for o in root_outputs(session)[1:]]
            assert texts == ["paid", "echo: hello?"]

    async def test_constructor_checks(self):
        with pytest.raises(ValueError, match="tracing provider"):
            ChatSession(Agent(name="a", model=TestModel(), tracing_provider=None))
        with pytest.raises(TypeError, match="send"):
            ChatSession(make_agent(Gate()), prompt="hi")
        with pytest.raises(ValueError, match="busy_mode"):
            ChatSession(make_agent(Gate()), busy_mode="later")  # type: ignore[arg-type]

    async def test_close_cancels_the_turn_and_stops_the_log(self):
        gate = Gate()
        session = ChatSession(make_agent(gate))
        session.send("slow")
        await asyncio.wait_for(gate.entered.wait(), 2)
        await session.close()
        assert not session.busy and session.closed
        events, _ = session.log.peek()
        assert events[-1]["kind"] == "closed"
        with pytest.raises(RuntimeError, match="closed"):
            session.send("x")
