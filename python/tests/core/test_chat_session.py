"""ChatSession and steering: messages sent while a turn runs, and what each mode does with them."""

import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession, _ToolPaths, _TurnOptions
from timbal.core.session_scheduler import take_followup, user_batch
from timbal.core.test_model import TestModel
from timbal.guardrails import Verdict, guardrail
from timbal.state import get_run_context, post_session_event, set_call_id, set_parent_call_id, set_run_context
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

    async def test_a_blocked_steer_is_dropped_and_the_turn_goes_on(self):
        gate = Gate()
        rail = guardrail(lambda text: Verdict.block("nope") if "forbidden" in text else True, stages=["input"])
        async with ChatSession(make_agent(gate, guardrails=[rail])) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            blocked = session.send("something forbidden")
            allowed = session.send("and something fine")
            gate.release.set()
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["status"]["code"] == "success"
            assert blocked.state == allowed.state == "delivered"
            (output,) = root_outputs(session)
            assert output.output.collect_text() == "echo: and something fine"
            memory = await stored_memory(output)
            assert all("forbidden" not in m.collect_text() for m in memory)
            notices = [m for m in memory if m.metadata.get("kind") == "steer_blocked"]
            assert len(notices) == 1 and "guardrail" in notices[0].collect_text()
            assert output.metadata["guardrails"]
            # The tool's result is kept: the work done before the steer is not thrown away.
            assert [m.role for m in memory][:3] == ["user", "assistant", "tool"]

    async def test_a_steer_a_guardrail_rewrites_is_kept_rewritten(self):
        gate = Gate()
        rail = guardrail(
            lambda text: text.replace("secret", "[redacted]") if "secret" in text else True,
            stages=["input"],
            action="redact",
        )
        async with ChatSession(make_agent(gate, guardrails=[rail])) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("my secret is 42")
            gate.release.set()
            await records(session, "turn_finished", 1)
            assert root_outputs(session)[0].output.collect_text() == "echo: my [redacted] is 42"

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


def wake_agent(sent, *, reply="NO_REPLY", extra_tools=(), model=None):
    """``wake`` prompts try to email (hidden when read-only) then answer ``reply``; users get echoes."""

    def send_email(to: str) -> str:
        sent.append(to)
        return "sent"

    def lookup(q: str) -> str:
        return f"found {q}"

    def handler(messages):
        last = messages[-1]
        prompt = next(m for m in reversed(messages) if m.role == "user")
        if last.role == "tool":
            return reply
        if prompt.is_runtime() and prompt.metadata.get("kind") == "wake":
            return Message(
                role="assistant",
                content=[ToolUseContent(id=f"call-{len(messages)}", name="send_email", input={"to": "boss@acme.test"})],
                stop_reason="tool_use",
            )
        if prompt.is_runtime():
            return reply
        return f"echo: {last_user_text(messages)}"

    return Agent(
        name="agent",
        model=model or TestModel(handler=handler),
        tools=[
            Tool(name="lookup", handler=lookup, read_only=True),
            Tool(name="send_email", handler=send_email),
            *extra_tools,
        ],
        tracing_provider=InMemoryTracingProvider,
    )


class TestWakeAndSilence:
    async def test_wake_runs_read_only_and_may_stay_silent(self):
        sent = []
        async with ChatSession(wake_agent(sent)) as session:
            session.wake("Check the inbox for anything urgent.")
            (finished,) = await records(session, "turn_finished", 1)
            (started,) = await records(session, "turn_started", 1)
            assert started["trigger"]["kind"] == "wake" and started["trigger"]["tool_policy"] == "read_only"
            assert finished["silent"] is True and finished["status"]["code"] == "success"
            assert sent == []  # send_email was hidden from the wake turn
            memory = await stored_memory(root_outputs(session)[0])
            prompt = memory[0]
            assert prompt.is_runtime() and prompt.metadata["kind"] == "wake"
            assert "NO_REPLY" in prompt.collect_text() and "Check the inbox" in prompt.collect_text()
            assert "This turn can only read" in prompt.collect_text()

    async def test_wake_with_all_tools_and_no_silence(self):
        sent = []
        async with ChatSession(wake_agent(sent, reply="Emailed them.")) as session:
            session.wake("Send the daily digest.", tool_policy=None, silent_ok=False)
            (finished,) = await records(session, "turn_finished", 1)
            assert sent == ["boss@acme.test"]
            assert finished["silent"] is False
            prompt = (await stored_memory(root_outputs(session)[0]))[0]
            assert "NO_REPLY" not in prompt.collect_text()

    async def test_user_messages_go_before_wakes_and_wakes_never_join_user_turns(self):
        gate = Gate()
        sent = []

        async def work() -> str:
            return await gate.work()

        agent = wake_agent(sent, extra_tools=[Tool(name="work", handler=work)])
        base_handler = agent.model.handler
        agent.model.handler = lambda m: (
            tool_call("work") if last_user_text(m) == "start" and m[-1].role == "user" else base_handler(m)
        )
        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.wake("periodic check")
            session.send("hi", mode="followup")
            gate.release.set()
            await records(session, "turn_finished", 3)
            kinds = [r["trigger"]["kind"] for r in await records(session, "turn_started", 3)]
            assert kinds == ["user_message", "user_message", "wake"]
            user_turn = await stored_memory(root_outputs(session)[1])
            assert all("periodic check" not in m.collect_text() for m in user_turn)

    async def test_a_newer_wake_with_the_same_key_replaces_a_pending_one(self):
        gate = Gate()
        sent = []
        agent = wake_agent(sent, extra_tools=[Tool(name="work", handler=gate.work)])
        base = agent.model.handler
        agent.model.handler = lambda m: (
            tool_call("work") if last_user_text(m) == "start" and m[-1].role == "user" else base(m)
        )
        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            first = session.wake("heartbeat 1", key="heartbeat")
            second = session.wake("heartbeat 2", key="heartbeat")
            assert [w["wake_id"] for w in session.state()["wakes"]] == [second]
            gate.release.set()
            await records(session, "turn_finished", 2)
            (superseded,) = await records(session, "wake_superseded", 1)
            assert superseded["wake_id"] == first
            started = await records(session, "turn_started", 2)
            assert started[1]["trigger"]["wake_id"] == second

    async def test_a_wake_is_not_held_back_by_a_running_background_task(self):
        release = asyncio.Event()

        async def build() -> str:
            await release.wait()
            return "built"

        def handler(messages):
            prompt = next(m for m in reversed(messages) if m.role == "user")
            if messages[-1].role == "tool":
                return "Started the build."
            if prompt.is_runtime() and prompt.metadata.get("kind") == "wake":
                return "Still building."
            return tool_call("build")

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="build", handler=build, background_mode="always")],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("build it")
            await records(session, "turn_finished", 1)
            assert session.store.running_count() == 1
            session.wake("How is the build going?")
            finished = await records(session, "turn_finished", 2, timeout=1.5)
            assert root_outputs(session)[1].output.collect_text() == "Still building."
            assert finished[1]["silent"] is False
            release.set()
            # The finished build then gets a turn of its own, which may stay silent.
            done = await records(session, "turn_finished", 3)
            started = await records(session, "turn_started", 3)
            assert started[2]["trigger"]["kind"] == "background_completion"
            assert "silent" in done[2]

    async def test_event_turns_may_stay_silent_but_user_turns_may_not(self):
        sent = []
        async with ChatSession(wake_agent(sent)) as session:
            session.store.post_event("Deploy 42 finished.")
            (event_turn,) = await records(session, "turn_finished", 1)
            assert event_turn["silent"] is True
            prompt = (await stored_memory(root_outputs(session)[0]))[0]
            assert "NO_REPLY" in prompt.collect_text()

        def says_no_reply(_messages):
            return "NO_REPLY"

        agent = Agent(name="agent", model=TestModel(handler=says_no_reply), tracing_provider=InMemoryTracingProvider)
        async with ChatSession(agent) as session:
            session.send("hello?")
            (user_turn,) = await records(session, "turn_finished", 1)
            assert user_turn["silent"] is False

    async def test_allow_silence_off(self):
        sent = []
        async with ChatSession(wake_agent(sent), allow_silence=False) as session:
            session.store.post_event("Deploy 42 finished.")
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["silent"] is False
            prompt = (await stored_memory(root_outputs(session)[0]))[0]
            assert "NO_REPLY" not in prompt.collect_text()

    async def test_a_resumed_wake_keeps_its_tool_policy(self):
        policies = []

        def charge() -> str:
            policies.append(get_run_context()._tool_policy)
            return "charged"

        def handler(messages):
            if messages[-1].role == "tool":
                return "Charged."
            return tool_call("charge")

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=charge, requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.wake("Pay the invoice.", tool_policy=["charge"])
            (parked,) = await records(session, "turn_finished", 1)
            assert parked["parked"]
            approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
            session.resume({approval.approval_id: True})
            await records(session, "turn_finished", 2)
            assert policies == [frozenset({"charge"})]

    async def test_wake_validation(self):
        session = ChatSession(make_agent(Gate()))
        with pytest.raises(ValueError, match="non-empty"):
            session.wake("   ")
        with pytest.raises(ValueError, match="tool_policy"):
            session.wake("x", tool_policy="read")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="strings"):
            session.wake("x", tool_policy=[1])  # type: ignore[list-item]
        await session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.wake("x")


class ConcurrencyProbe(TestModel):
    """Echoes like ``echo`` and records how many model calls overlap (a session runs one turn at a time)."""

    def __init__(self, delay=0.01):
        super().__init__(handler=echo)
        self.active = 0
        self.max_active = 0
        self.delay = delay

    async def stream(self, messages, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            async for event in super().stream(messages, **kwargs):
                yield event
        finally:
            self.active -= 1


class TestRobustness:
    async def test_two_subscribers_see_the_same_log(self):
        """Two tabs on one session: both follow every record, in the same order."""
        async with ChatSession(make_agent(Gate())) as session:
            seen = [[], []]

            async def follow(i):
                async for item in session.events():
                    seen[i].append(item)
                    if isinstance(item, dict) and item.get("kind") == "turn_finished":
                        return

            tabs = [asyncio.create_task(follow(0)), asyncio.create_task(follow(1))]
            await asyncio.sleep(0)
            session.send("hello")
            await asyncio.wait_for(asyncio.gather(*tabs), 3)
            assert seen[0] == seen[1] and len(seen[0]) > 5

    async def test_reconnecting_mid_turn_replays_from_the_cursor(self):
        gate = Gate()
        async with ChatSession(make_agent(gate)) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            cursor = session.log.cursor_end
            before, _ = session.log.peek()
            gate.release.set()
            await records(session, "turn_finished", 1)
            tail = []
            async for item in session.events(after=cursor):
                tail.append(item)
                if isinstance(item, dict) and item.get("kind") == "turn_finished":
                    break
            everything, _ = session.log.peek()
            assert before + tail == everything[: len(before) + len(tail)]

    async def test_a_foreign_turn_waits_for_the_running_one_and_the_session_continues_after_it(self):
        gate = Gate()
        agent = make_agent(gate)
        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            running = session.run_id
            # Another client continues the same conversation directly (e.g. POST /stream with this parent).
            foreign = asyncio.create_task(agent(prompt="from elsewhere", parent_id=running).collect())
            await asyncio.sleep(0.05)
            assert not foreign.done()  # queued behind the session's running turn
            gate.release.set()
            other = await asyncio.wait_for(foreign, 3)
            assert other.output.collect_text() == "echo: from elsewhere"
            assert session.head == other.run_id
            session.send("and now?")
            finished = await records(session, "turn_finished", 2)
            started = await records(session, "turn_started", 2)
            assert started[1]["parent_id"] == other.run_id
            memory = await stored_memory(root_outputs(session)[1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["start", "from elsewhere", "and now?"]
            assert finished[1]["status"]["code"] == "success"

    async def test_messages_wait_while_a_foreign_run_of_the_session_is_parked(self):
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
            session.send("hi")
            await records(session, "turn_finished", 1)
            # Another client continues the conversation directly and parks on an approval.
            events = [e async for e in agent(prompt="pay", parent_id=session.head)]
            approval = next(e for e in events if isinstance(e, ApprovalEvent))
            parked = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == "agent")
            assert parked.status.reason == "approval_required" and session.head == parked.run_id
            waiting = session.send("new question")
            await asyncio.sleep(0.1)
            assert waiting.state == "pending" and session.turns == 1
            # That client answers its approval; the session then goes on from its resume.
            resumed = await agent(prompt="pay", parent_id=parked.run_id, resume={approval.approval_id: True}).collect()
            assert resumed.output.collect_text() == "paid"
            await records(session, "turn_finished", 2)
            memory = await stored_memory(root_outputs(session)[1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["hi", "pay", "new question"]

    async def test_a_steered_user_message_gets_its_tools_back_after_the_denial_breaker(self):
        holder = {}

        def act() -> str:
            return "acted"

        def handler(messages):
            if last_user_text(messages) == "and this?":
                return "here you go"
            if messages[-1].role == "user" and last_user_text(messages) == "start":
                holder["session"].send("and this?")  # arrives while the turn runs: steered in
                return tool_call("act")
            return "stopping"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[
                Tool(
                    name="act",
                    handler=act,
                    guardrails=[guardrail(lambda _t: Verdict.block("no"), stages=["tool_args"])],
                )
            ],
            max_consecutive_denials=1,
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            holder["session"] = session
            session.send("start")
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["status"]["code"] == "success"
            trace = await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(finished["run_id"])
            llm_calls = [span for span in trace.values() if span.path == "agent.llm"]
            assert llm_calls[-1].input.get("tools"), "the steered user request was answered without tools"

    @pytest.mark.parametrize("steering", [None, True])
    async def test_a_user_message_does_not_join_a_running_wake(self, steering):
        gate = Gate()

        def handler(messages):
            last = messages[-1]
            prompt = next(m for m in reversed(messages) if m.role == "user")
            if last.role == "tool":
                return "NO_REPLY"
            if prompt.is_runtime() and prompt.metadata.get("kind") == "wake":
                return tool_call("peek", call_id=f"call-{len(messages)}")
            return f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="peek", handler=gate.work, read_only=True)],
            steering=steering,
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.wake("Check the inbox.")
            await asyncio.wait_for(gate.entered.wait(), 2)
            session.send("hello")  # default mode: steer
            gate.release.set()
            finished = await records(session, "turn_finished", 2)
            started = await records(session, "turn_started", 2)
            assert [s["trigger"]["kind"] for s in started] == ["wake", "user_message"]
            assert finished[0]["silent"] is True and finished[1]["silent"] is False
            wake_memory = await stored_memory(root_outputs(session)[0])
            assert "hello" not in [m.collect_text() for m in wake_memory if m.role == "user"]
            assert root_outputs(session)[1].output.collect_text() == "echo: hello"

    async def test_a_foreign_run_parking_during_the_collect_window_holds_the_turn(self):
        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent, collect_window=0.3) as session:
            session.send("hi")
            await records(session, "turn_finished", 1)
            waiting = session.send("later", mode="collect")
            await asyncio.sleep(0.05)  # the session is inside its quiet window now
            events = [e async for e in agent(prompt="pay", parent_id=session.head)]
            approval = next(e for e in events if isinstance(e, ApprovalEvent))
            parked = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == "agent")
            await asyncio.sleep(0.5)  # past the window
            assert waiting.state == "pending" and session.turns == 1
            await agent(prompt="pay", parent_id=parked.run_id, resume={approval.approval_id: True}).collect()
            await records(session, "turn_finished", 2)
            memory = await stored_memory(root_outputs(session)[1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["hi", "pay", "later"]

    async def test_a_turn_failing_before_it_starts_backs_off_then_waits_for_new_work(self, monkeypatch):
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        attempts = []
        async with ChatSession(make_agent(Gate())) as session:

            async def broken_pump(_context, _kwargs, followup, trigger):
                attempts.append(trigger["kind"])
                if followup is not None:
                    followup.release()
                raise RuntimeError("model setup is broken")

            monkeypatch.setattr(session, "_pump", broken_pump)
            waiting = session.send("hi")
            failed = await records(session, "turn_failed", 3)
            await asyncio.sleep(0.2)
            assert len(attempts) == 3, "kept retrying the same failing work"
            assert [f["attempt"] for f in failed] == [1, 2, 3]
            assert waiting.state == "pending"
            session.send("again")  # new work: one more try
            await records(session, "turn_failed", 4)
            await asyncio.sleep(0.2)
            assert len(attempts) == 4

    async def test_work_arriving_during_the_last_failed_attempt_gets_its_try(self, monkeypatch):
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        attempts = []
        async with ChatSession(make_agent(Gate())) as session:

            async def broken_pump(_context, _kwargs, followup, trigger):
                attempts.append(trigger["kind"])
                if len(attempts) == 3:
                    session.send("also this")  # new work while the third attempt runs
                if followup is not None:
                    followup.release()
                raise RuntimeError("model setup is broken")

            monkeypatch.setattr(session, "_pump", broken_pump)
            session.send("hi")
            await records(session, "turn_failed", 4)
            await asyncio.sleep(0.2)
            assert len(attempts) == 4

    async def test_another_client_answering_the_parked_turn_unparks_the_session(self):
        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("pay")
            await records(session, "turn_finished", 1)
            assert session.parked
            (approval,) = session.state()["pending_approvals"]
            resumed = await agent(
                prompt="pay", parent_id=session.head, resume={approval["approval_id"]: True}
            ).collect()
            assert resumed.status.code == "success"
            assert not session.parked and session.head == resumed.run_id
            session.send("next")
            await records(session, "turn_finished", 2)
            assert root_outputs(session)[-1].output.collect_text() == "echo: next"
            memory = await stored_memory(root_outputs(session)[-1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["pay", "next"]

    @pytest.mark.parametrize("enter", [True, False])
    async def test_a_session_continuing_a_parked_run_starts_parked(self, enter):
        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        initial = await agent(prompt="pay").collect()
        assert initial.status.reason == "approval_required"
        session = ChatSession(agent, parent_id=initial.run_id)
        try:
            if enter:
                await session.__aenter__()
                assert session.parked
                assert session.state()["pending_approvals"]
            waiting = session.send("new question")
            await asyncio.sleep(0.2)
            assert waiting.state == "pending" and session.turns == 0 and session.parked
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            (finished,) = await records(session, "turn_finished", 1)  # the resumed turn takes it (steer)
            assert finished["status"]["code"] == "success"
            assert root_outputs(session)[-1].output.collect_text() == "echo: new question"
            memory = await stored_memory(root_outputs(session)[-1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["pay", "new question"]
        finally:
            await session.close()

    async def test_a_restored_parked_wake_resumes_under_its_own_options(self):
        def handler(messages):
            last = messages[-1]
            prompt = next(m for m in reversed(messages) if m.role == "user")
            if last.role == "tool":
                return "NO_REPLY"
            if prompt.is_runtime() and prompt.metadata.get("kind") == "wake":
                return tool_call("peek")
            return f"echo: {last_user_text(messages)}"

        def build():
            return Agent(
                name="agent",
                model=TestModel(handler=handler),
                tools=[
                    Tool(name="peek", handler=lambda: "inbox", read_only=True, requires_approval=True),
                    Tool(name="send_email", handler=lambda: "sent"),
                ],
                tracing_provider=InMemoryTracingProvider,
            )

        async with ChatSession(build()) as first:
            first.wake("Check the inbox.")
            (finished,) = await records(first, "turn_finished", 1)
            assert finished["parked"]
            parked_run = first.head
        async with ChatSession(build(), parent_id=parked_run) as session:
            assert session.parked
            assert session._last_options == _TurnOptions(tool_policy="read_only", silence=True, steer=False)
            session.send("hello")
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            finished = await records(session, "turn_finished", 2)
            assert finished[0]["silent"] is True and finished[1]["silent"] is False
            wake_memory = await stored_memory(root_outputs(session)[0])
            assert "hello" not in [m.collect_text() for m in wake_memory if m.role == "user"]
            assert root_outputs(session)[1].output.collect_text() == "echo: hello"

    async def test_a_restored_wake_with_a_callable_policy_gets_no_tool_it_did_not_have(self):
        secrets_read = []

        def secret() -> str:
            secrets_read.append("payroll")
            return "payroll"

        def handler(messages):
            last = messages[-1]
            prompt = next(m for m in reversed(messages) if m.role == "user")
            if last.role == "tool":
                return "NO_REPLY" if len(messages) > 4 else tool_call("secret", call_id="call-2")
            if prompt.is_runtime() and prompt.metadata.get("kind") == "wake":
                return tool_call("peek")
            return f"echo: {last_user_text(messages)}"

        def build():
            return Agent(
                name="agent",
                model=TestModel(handler=handler),
                tools=[
                    Tool(name="peek", handler=lambda: "inbox", read_only=True, requires_approval=True),
                    Tool(name="secret", handler=secret, read_only=True),
                ],
                tracing_provider=InMemoryTracingProvider,
            )

        async with ChatSession(build()) as first:
            first.wake("Check the inbox.", tool_policy=lambda tool: tool.name == "peek")
            (finished,) = await records(first, "turn_finished", 1)
            assert finished["parked"]
            parked_run = first.head
        async with ChatSession(build(), parent_id=parked_run) as session:
            assert session._last_options.tool_policy == _ToolPaths(frozenset({"agent.peek"}))
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["status"]["code"] == "success"
        assert secrets_read == []

    async def test_a_restored_callable_policy_keeps_each_agents_own_tools(self):
        writes = []

        def root_handler(messages):
            results = sum(1 for m in messages if m.role == "tool")
            if results == 0:
                return tool_call("lookup")
            if results == 1:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="call-child", name="child", input={"prompt": "go"})],
                    stop_reason="tool_use",
                )
            return "NO_REPLY"

        def child_handler(messages):
            return "done" if messages[-1].role == "tool" else tool_call("lookup", call_id="call-private")

        def build():
            child = Agent(
                name="child",
                model=TestModel(handler=child_handler),
                tools=[Tool(name="lookup", handler=lambda: writes.append("private lookup executed") or "private")],
            )
            return Agent(
                name="agent",
                model=TestModel(handler=root_handler),
                tools=[
                    Tool(name="lookup", handler=lambda: writes.append("root lookup") or "root", requires_approval=True),
                    child,
                ],
                tracing_provider=InMemoryTracingProvider,
            )

        allowed = {"agent.lookup", "agent.child"}
        async with ChatSession(build()) as first:
            first.wake("Look it up.", tool_policy=lambda tool: tool._path in allowed)
            (finished,) = await records(first, "turn_finished", 1)
            assert finished["parked"]
            parked_run = first.head
        trace = await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(parked_run)
        # Stored on agent spans, which every provider returns (the platform's resume read skips LLM spans).
        noted = {
            p
            for span in trace.values()
            if span.metadata.get("type") != "LLM"
            for p in span.metadata.get("tool_policy_allowed", ())
        }
        assert noted == allowed
        async with ChatSession(build(), parent_id=parked_run) as session:
            assert session._last_options.tool_policy == _ToolPaths(frozenset(allowed))
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["status"]["code"] == "success"
        assert writes == ["root lookup"]

    @pytest.mark.parametrize("enter", [True, False])
    async def test_a_failed_parent_lookup_leaves_the_session_unresolved(self, enter, monkeypatch):
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        failures = {"left": 0}

        class FlakyProvider(InMemoryTracingProvider):
            @classmethod
            async def get(cls, run_context):
                if failures["left"]:
                    failures["left"] -= 1
                    raise ConnectionError("trace store down")
                return await super().get(run_context)

        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=FlakyProvider,
        )
        initial = await agent(prompt="pay").collect()
        failures["left"] = 1
        session = ChatSession(agent, parent_id=initial.run_id)
        try:
            if enter:
                with pytest.raises(ConnectionError):
                    await session.__aenter__()
                assert not session.parked
                await session.restore()  # the next call tries again
                assert session.parked
            waiting = session.send("new question")
            await asyncio.sleep(0.2)
            assert waiting.state == "pending" and session.parked and session.turns == 0
            (failed,) = await records(session, "restore_failed", 1)
            assert failed["run_id"] == initial.run_id
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            await records(session, "turn_finished", 1)
            memory = await stored_memory(root_outputs(session)[-1])
            assert [m.collect_text() for m in memory if m.role == "user"] == ["pay", "new question"]
        finally:
            await session.close()

    @pytest.mark.parametrize("parked", [True, False])
    async def test_restore_works_with_the_default_tracing_setting(self, parked):
        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(  # no tracing_provider: TRACING_UNSET, resolved by the run context
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
        )
        initial = await agent(prompt="pay" if parked else "hi").collect()
        async with ChatSession(agent, parent_id=initial.run_id) as session:
            assert session.parked is parked
            if not parked:
                session.send("again")
                (finished,) = await records(session, "turn_finished", 1)
                assert finished["status"]["code"] == "success"

    async def test_a_restored_callable_policy_does_not_bring_back_a_revoked_tool(self):
        state = {"revoked": False}
        secrets_read = []

        def revoke() -> str:
            state["revoked"] = True
            return "revoked"

        def secret() -> str:
            secrets_read.append("payroll")
            return "payroll"

        def policy(tool):
            return tool.name in {"revoke", "peek"} or (tool.name == "secret" and not state["revoked"])

        def handler(messages):
            results = sum(1 for m in messages if m.role == "tool")
            return (
                [tool_call("revoke"), tool_call("peek", call_id="c2"), tool_call("secret", call_id="c3")][results]
                if results < 3
                else "NO_REPLY"
            )

        def build():
            return Agent(
                name="agent",
                model=TestModel(handler=handler),
                tools=[
                    Tool(name="revoke", handler=revoke),
                    Tool(name="peek", handler=lambda: "inbox", requires_approval=True),
                    Tool(name="secret", handler=secret),
                ],
                tracing_provider=InMemoryTracingProvider,
            )

        async with ChatSession(build()) as first:
            first.wake("Check.", tool_policy=policy)
            (finished,) = await records(first, "turn_finished", 1)
            assert finished["parked"]
            parked_run = first.head
        async with ChatSession(build(), parent_id=parked_run) as session:
            assert session._last_options.tool_policy == _ToolPaths(frozenset({"agent.revoke", "agent.peek"}))
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            await records(session, "turn_finished", 1)
        assert secrets_read == []

    def test_restored_paths_take_each_agents_latest_span(self):
        from types import SimpleNamespace

        from timbal.core.chat_session import _allowed_tool_paths

        def span(path, t0, allowed):
            return SimpleNamespace(path=path, t0=t0, metadata={"tool_policy_allowed": allowed})

        trace = {
            "1": span("agent", 1, ["agent.child"]),
            "2": span("agent.child", 2, ["agent.child.lookup", "agent.child.secret"]),
            "3": span("agent.child", 3, ["agent.child.lookup"]),  # a later call took "secret" back
            "4": SimpleNamespace(path="agent.child.lookup", t0=4, metadata={}),
        }
        assert _allowed_tool_paths(trace) == {"agent.child", "agent.child.lookup"}

    async def test_answers_sent_before_the_worker_takes_them_go_together(self):
        def handler(messages):
            if messages[-1].role == "tool":
                return "both done"
            return Message(
                role="assistant",
                content=[
                    ToolUseContent(id="call-a", name="first", input={}),
                    ToolUseContent(id="call-b", name="second", input={}),
                ],
                stop_reason="tool_use",
            )

        ran = []
        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[
                Tool(name="first", handler=lambda: ran.append("first") or "1", requires_approval=True),
                Tool(name="second", handler=lambda: ran.append("second") or "2", requires_approval=True),
            ],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("go")
            await records(session, "turn_finished", 1)
            approvals = {a["path"].rsplit(".", 1)[-1]: a["approval_id"] for a in session.state()["pending_approvals"]}
            assert set(approvals) == {"first", "second"}
            session.resume({approvals["first"]: True})
            with pytest.raises(ValueError, match="Already answered differently"):
                session.resume({approvals["first"]: False})
            session.resume({approvals["first"]: True, approvals["second"]: True})  # the same answer again is fine
            finished = await records(session, "turn_finished", 2)
            assert finished[1]["status"]["code"] == "success" and not finished[1]["parked"]
            assert sorted(ran) == ["first", "second"]

    async def test_an_answer_while_a_turn_runs_is_refused(self):
        gate = Gate()
        async with ChatSession(make_agent(gate)) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            with pytest.raises(RuntimeError, match="running"):
                session.resume({"any": True})
            gate.release.set()
            await records(session, "turn_finished", 1)

    async def test_a_jsonl_read_error_does_not_unpark_a_restored_session(self, tmp_path, monkeypatch):
        import pathlib

        import timbal.core.chat_session as chat_session_module
        from timbal.state.tracing.providers import JsonlTracingProvider

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        provider = JsonlTracingProvider.configured(path=tmp_path / "traces.jsonl")

        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=provider,
        )
        initial = await agent(prompt="pay").collect()
        real_open, failures = pathlib.Path.open, {"left": 1}

        def flaky_open(self, *args, **kwargs):
            if self == provider._path and failures["left"]:
                failures["left"] -= 1
                raise OSError("transient read error")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", flaky_open)
        session = ChatSession(agent, parent_id=initial.run_id)
        try:
            waiting = session.send("new question")
            await asyncio.sleep(0.3)
            assert session.parked and waiting.state == "pending" and session.turns == 0
            (approval,) = session.state()["pending_approvals"]
            session.resume({approval["approval_id"]: True})
            await records(session, "turn_finished", 1)
            output = root_outputs(session)[-1]
            trace = await RunContext(tracing_provider=provider)._get_parent_trace(output.run_id)
            memory = [Message.validate(message) for message in trace[output.call_id].memory]
            assert [m.collect_text() for m in memory if m.role == "user"] == ["pay", "new question"]
        finally:
            await session.close()

    @pytest.mark.parametrize("parked", [True, False])
    async def test_an_answer_sent_before_restore_waits_for_it(self, parked):
        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        events = [e async for e in agent(prompt="pay" if parked else "hi")]
        initial = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == "agent")
        approval_id = next((e.approval_id for e in events if isinstance(e, ApprovalEvent)), "stale")
        session = ChatSession(agent, parent_id=initial.run_id)
        try:
            session.resume({approval_id: True})  # no restore() yet
            if parked:
                (finished,) = await records(session, "turn_finished", 1)
                assert finished["status"]["code"] == "success"
                assert root_outputs(session)[-1].output.collect_text() == "paid"
            else:
                (dropped,) = await records(session, "resume_dropped", 1)
                await asyncio.sleep(0.1)
                assert session.turns == 0  # the old prompt was not re-sent as a new turn
        finally:
            await session.close()

    async def test_an_interrupt_keeps_the_answers_already_given(self):
        def handler(messages):
            if messages[-1].role == "tool":
                return "over"
            if last_user_text(messages) != "go":
                return f"echo: {last_user_text(messages)}"
            return Message(
                role="assistant",
                content=[
                    ToolUseContent(id="call-a", name="first", input={}),
                    ToolUseContent(id="call-b", name="second", input={}),
                ],
                stop_reason="tool_use",
            )

        ran = []
        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[
                Tool(name="first", handler=lambda: ran.append("first") or "1", requires_approval=True),
                Tool(name="second", handler=lambda: ran.append("second") or "2", requires_approval=True),
            ],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("go")
            await records(session, "turn_finished", 1)
            approvals = {a["path"].rsplit(".", 1)[-1]: a["approval_id"] for a in session.state()["pending_approvals"]}
            session.resume({approvals["first"]: True})
            assert session.interrupt() is True
            (interrupt,) = await records(session, "interrupt", 1)
            assert interrupt["abandoned"] == [approvals["second"]]
            await records(session, "turn_finished", 2)
            assert ran == ["first"]

    async def test_an_interrupt_sent_before_restore_cancels_the_parked_run(self):
        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        initial = await agent(prompt="pay").collect()
        session = ChatSession(agent, parent_id=initial.run_id)
        try:
            session.send("forget it", mode="interrupt")  # no restore() yet
            finished = await records(session, "turn_finished", 2)
            assert finished[0]["status"]["code"] == "cancelled"
            (interrupt,) = await records(session, "interrupt", 1)
            assert interrupt["run_id"] == initial.run_id and interrupt["abandoned"]
            assert root_outputs(session)[-1].output.collect_text() == "echo: forget it"
        finally:
            await session.close()

    async def test_interrupting_a_tool_runs_its_cleanup_and_the_next_turn_has_a_result_for_it(self):
        cleaned = asyncio.Event()

        async def forever() -> str:
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return "never"

        agent = Agent(
            name="agent",
            model=TestModel(handler=echo),
            tools=[Tool(name="forever", handler=forever)],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("slow")
            await asyncio.sleep(0.1)
            session.send("stop", mode="interrupt")
            await records(session, "turn_finished", 2)
            assert cleaned.is_set()
            memory = await stored_memory(root_outputs(session)[1])
            # Every tool_use is answered (a synthetic result for the cut call), so the provider accepts it.
            uses = [c.id for m in memory if m.role == "assistant" for c in m.content if c.type == "tool_use"]
            results = [c.id for m in memory if m.role == "tool" for c in m.content if c.type == "tool_result"]
            assert uses and set(uses) <= set(results)

    async def test_a_burst_of_mixed_messages_is_each_delivered_exactly_once(self):
        model = ConcurrencyProbe()
        gate = Gate()
        async with ChatSession(make_agent(gate, model=model), collect_window=0.01) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 2)
            modes = ["steer", "followup", "collect", "steer", "followup"] * 10
            events = [session.send(f"m{i}", mode=mode) for i, mode in enumerate(modes)]
            gate.release.set()
            await asyncio.wait_for(asyncio.gather(*(event.settled() for event in events)), 5)
            await asyncio.sleep(0.1)
            while session.busy or session.pending_messages():
                await asyncio.sleep(0.02)
            assert model.max_active == 1
            assert all(event.state == "delivered" for event in events)
            # Each turn's memory holds the conversation so far; take the messages each turn added.
            delivered: list[str] = []
            for output in root_outputs(session):
                ids = [
                    event_id
                    for message in await stored_memory(output)
                    if message.metadata.get("kind") == "user_message"
                    for event_id in message.metadata["event_ids"]
                ]
                assert ids[: len(delivered)] == delivered  # history is append-only
                delivered = ids
            sent = {event.id for event in events}
            burst = [event_id for event_id in delivered if event_id in sent]
            assert sorted(burst) == sorted(sent) and len(burst) == len(sent)  # exactly once
            # Steers jump ahead of queued follow-ups by design; within a mode the order holds.
            for mode in ("steer", "followup", "collect"):
                expected = [event.id for event, m in zip(events, modes, strict=True) if m == mode]
                assert [event_id for event_id in burst if event_id in set(expected)] == expected

    async def test_an_interrupt_storm_ends_with_every_message_answered(self):
        gate = Gate()
        async with ChatSession(make_agent(gate)) as session:
            session.send("slow")
            await asyncio.wait_for(gate.entered.wait(), 2)
            events = [session.send(f"cut {i}", mode="interrupt") for i in range(5)]
            await asyncio.wait_for(asyncio.gather(*(event.settled() for event in events)), 5)
            while session.busy:
                await asyncio.sleep(0.02)
            assert {event.state for event in events} == {"delivered"}
            last = root_outputs(session)[-1]
            assert last.status.code == "success"
            assert "cut 4" in last.output.collect_text()

    async def test_close_with_pending_work_and_a_parked_turn(self):
        def handler(messages):
            return "ok" if messages[-1].role == "tool" else tool_call("charge")

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: "x", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        session = ChatSession(agent)
        session.send("pay")
        await records(session, "turn_finished", 1)
        assert session.parked
        waiting = session.send("hello?", mode="followup")
        session.wake("check")
        await session.close()
        assert waiting.state == "pending" and session.closed
        with pytest.raises(RuntimeError):
            session.resume({"x": True})


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

    async def test_interrupt_abandons_a_parked_turn_and_runs_the_new_message(self):
        charged = []

        def handler(messages):
            if messages[-1].role == "tool":
                return "paid"
            return tool_call("charge") if last_user_text(messages) == "pay" else f"echo: {last_user_text(messages)}"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: charged.append(1) or "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(agent) as session:
            session.send("pay")
            await records(session, "turn_finished", 1)
            assert session.parked
            session.send("actually, forget it", mode="interrupt")
            finished = await records(session, "turn_finished", 3)
            assert finished[1]["status"]["code"] == "cancelled"
            assert finished[2]["status"]["code"] == "success" and not session.parked
            assert charged == []
            assert root_outputs(session)[-1].output.collect_text() == "echo: actually, forget it"
            (cut,) = await records(session, "interrupt", 1)
            assert cut["abandoned"]
            # The next turn's history answers the cancelled call, so the provider accepts it.
            memory = await stored_memory(root_outputs(session)[-1])
            uses = [c.id for m in memory if m.role == "assistant" for c in m.content if c.type == "tool_use"]
            results = [c.id for m in memory if m.role == "tool" for c in m.content if c.type == "tool_result"]
            assert set(uses) <= set(results)

    async def test_stop_while_parked_cancels_the_question(self):
        def handler(messages):
            if messages[-1].role == "tool":
                return "thanks"
            return Message(
                role="assistant",
                content=[ToolUseContent(id="ask-1", name="ask_user", input={"question": "Which city?"})],
                stop_reason="tool_use",
            )

        from timbal.tools.interaction import ask_user

        agent = Agent(
            name="agent", model=TestModel(handler=handler), tools=[ask_user], tracing_provider=InMemoryTracingProvider
        )
        async with ChatSession(agent) as session:
            session.send("ask me")
            await records(session, "turn_finished", 1)
            assert session.parked and session.state()["pending_interactions"]
            assert session.interrupt() is True
            finished = await records(session, "turn_finished", 2)
            assert finished[1]["status"]["code"] == "cancelled" and not session.parked
            assert session.interrupt() is False

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


class TestLogCursors:
    async def test_numbered_events_keep_their_own_cursors_across_a_trim(self):
        from timbal.state.background import BACKGROUND_LOG_GAPPED, BackgroundEventLog

        log = BackgroundEventLog(max_events=2)
        for item in "abc":
            log.put_nowait(item)
        stream = log.subscribe(0, numbered=True)
        assert await stream.__anext__() == (1, BACKGROUND_LOG_GAPPED)  # "a" was dropped; the replay is b, c
        log.put_nowait("d")
        log.put_nowait("e")  # trims b and c meanwhile
        got = [await stream.__anext__() for _ in range(4)]
        assert got == [(2, "b"), (3, "c"), (4, "d"), (5, "e")]
        assert log.cursor_end == 5
        await stream.aclose()

    async def test_unnumbered_subscribe_is_unchanged(self):
        from timbal.state.background import BACKGROUND_LOG_GAPPED, BackgroundEventLog

        log = BackgroundEventLog(max_events=2)
        for item in "abc":
            log.put_nowait(item)
        stream = log.subscribe(0)
        assert [await stream.__anext__() for _ in range(3)] == [BACKGROUND_LOG_GAPPED, "b", "c"]
        await stream.aclose()
