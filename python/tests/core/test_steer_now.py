"""ChatSession.send(mode="now"): cut the answer being streamed or the interruptible tools in flight."""

import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.agent import DEFAULT_STEERED_TOOL_RESULT
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolResultContent, ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from ..guardrails.conftest import StreamingTestModel, text_stream


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def calls(*names):
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"call-{name}", name=name, input={}) for name in names],
        stop_reason="tool_use",
    )


def user_texts(messages):
    return [m.collect_text() for m in messages if m.role == "user" and not m.is_runtime()]


async def records(session, kind, count, timeout=5.0):
    found = []

    async def collect():
        async for item in session.events():
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), timeout)
    return found


async def memory_of(session):
    events, _ = session.log.peek()
    root = [e for e in events if isinstance(e, OutputEvent) and e.parent_call_id is None and e.path == "agent"][-1]
    trace = await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(root.run_id)
    return root, [Message.validate(m) for m in trace[root.call_id].memory]


def tool_results(memory):
    return {
        c.id: c.content[0].text if c.content and hasattr(c.content[0], "text") else c.content
        for m in memory
        if m.role == "tool"
        for c in m.content
        if isinstance(c, ToolResultContent)
    }


class Probe:
    """Tools that block until released (or forever), recording how they ended."""

    def __init__(self):
        self.started: dict[str, asyncio.Event] = {}
        self.release = asyncio.Event()
        self.ended: dict[str, str] = {}

    def tool(self, name, *, forever=False, **flags):
        self.started[name] = asyncio.Event()

        async def handler() -> str:
            self.started[name].set()
            try:
                if forever:
                    await asyncio.Event().wait()
                await self.release.wait()
                self.ended[name] = "finished"
                return f"{name} done"
            except asyncio.CancelledError:
                self.ended[name] = "cancelled"
                raise

        return Tool(name=name, handler=handler, **flags)


def batch_model(*names):
    """Calls ``names`` in one batch for 'start'; any later step answers the last user message."""

    def handler(messages):
        last_user = user_texts(messages)[-1]
        if messages[-1].role == "user" and last_user == "start":
            return calls(*names)
        return f"answer to: {last_user}"

    return TestModel(handler=handler)


def agent(model, tools=(), **kwargs):
    return Agent(name="agent", model=model, tools=list(tools), tracing_provider=InMemoryTracingProvider, **kwargs)


class TestToolCut:
    async def test_cuts_a_read_only_tool_and_answers_in_the_same_turn(self):
        probe = Probe()
        a = agent(batch_model("search"), [probe.tool("search", forever=True, read_only=True)])
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(probe.started["search"].wait(), 2)
            session.send("never mind, just say hi", mode="now")
            (finished,) = await records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "success"
        assert probe.ended["search"] == "cancelled"
        root, memory = await memory_of(session)
        assert tool_results(memory)["call-search"] == DEFAULT_STEERED_TOOL_RESULT
        assert user_texts(memory) == ["start", "never mind, just say hi"]
        assert root.output.collect_text() == "answer to: never mind, just say hi"
        assert root.metadata["steer_now"] == [{"cut": "tool", "tool": "search"}]

    async def test_waits_for_a_tool_that_acts(self):
        probe = Probe()
        a = agent(batch_model("charge"), [probe.tool("charge")])
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(probe.started["charge"].wait(), 2)
            session.send("also say hi", mode="now")
            await asyncio.sleep(0.3)
            assert "charge" not in probe.ended  # still running: not cancelled
            probe.release.set()
            await records(session, "turn_finished", 1)
        assert probe.ended["charge"] == "finished"
        root, memory = await memory_of(session)
        assert tool_results(memory)["call-charge"] == "charge done"
        assert root.output.collect_text() == "answer to: also say hi"
        assert "steer_now" not in root.metadata

    async def test_mixed_batch_cuts_only_the_interruptible_one(self):
        probe = Probe()
        a = agent(
            batch_model("search", "charge"),
            [probe.tool("search", forever=True, read_only=True), probe.tool("charge")],
        )
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(probe.started["search"].wait(), 2)
            await asyncio.wait_for(probe.started["charge"].wait(), 2)
            session.send("stop searching", mode="now")
            for _ in range(50):
                if "search" in probe.ended:
                    break
                await asyncio.sleep(0.02)
            assert probe.ended == {"search": "cancelled"}  # cut at once; charge still running
            probe.release.set()
            await records(session, "turn_finished", 1)
        assert probe.ended == {"search": "cancelled", "charge": "finished"}
        _, memory = await memory_of(session)
        results = tool_results(memory)
        assert results == {"call-search": DEFAULT_STEERED_TOOL_RESULT, "call-charge": "charge done"}

    @pytest.mark.parametrize(
        "flags, cut", [({"read_only": True, "interruptible": False}, False), ({"interruptible": True}, True)]
    )
    async def test_interruptible_overrides_read_only(self, flags, cut):
        probe = Probe()
        a = agent(batch_model("work"), [probe.tool("work", **flags)])
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(probe.started["work"].wait(), 2)
            session.send("hi", mode="now")
            await asyncio.sleep(0.3)
            assert ("work" in probe.ended) is cut
            probe.release.set()
            await records(session, "turn_finished", 1)
        assert probe.ended["work"] == ("cancelled" if cut else "finished")

    async def test_cut_tool_card_closes_on_its_own_span(self):
        probe = Probe()
        a = agent(batch_model("search"), [probe.tool("search", forever=True, read_only=True)])
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(probe.started["search"].wait(), 2)
            session.send("hi", mode="now")
            await records(session, "turn_finished", 1)
        events, _ = session.log.peek()
        start = next(e for e in events if getattr(e, "type", None) == "START" and e.path == "agent.search")
        outs = [e for e in events if isinstance(e, OutputEvent) and e.path == "agent.search"]
        assert [(o.call_id, o.status.reason) for o in outs] == [(start.call_id, "steered")]


class SlowStream(StreamingTestModel):
    """Streams its scripts a chunk at a time so a test can act mid-answer."""

    __test__ = False

    def __init__(self, scripts, delay=0.03):
        super().__init__(scripts)
        self.delay = delay
        self.streaming = asyncio.Event()

    async def stream(self, messages, **kwargs):
        async for chunk in super().stream(messages, **kwargs):
            self.streaming.set()
            yield chunk
            await asyncio.sleep(self.delay)


class TestAnswerCut:
    async def test_cuts_the_answer_keeps_its_text_and_answers_again(self):
        long = "Once upon a time there was a lighthouse keeper who " * 6
        model = SlowStream([text_stream(long), text_stream("Sure: BANANA.")])
        async with ChatSession(agent(model)) as session:
            session.send("tell me a long story")
            await asyncio.wait_for(model.streaming.wait(), 2)
            await asyncio.sleep(0.1)
            session.send("stop, just say BANANA", mode="now")
            (finished,) = await records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "success"
        root, memory = await memory_of(session)
        assert [m.role for m in memory] == ["user", "assistant", "user", "assistant"]
        partial = memory[1]
        assert partial.metadata.get("interrupted") == "steer_now"
        assert 0 < len(partial.collect_text()) < len(long)
        assert long.startswith(partial.collect_text())
        assert root.output.collect_text() == "Sure: BANANA."
        (record,) = root.metadata["steer_now"]
        assert record["cut"] == "answer" and record["kept_chars"] == len(partial.collect_text())

    async def test_a_burst_of_now_messages_makes_one_cut(self):
        model = SlowStream([text_stream("word " * 200), text_stream("ok")])
        async with ChatSession(agent(model)) as session:
            session.send("go")
            await asyncio.wait_for(model.streaming.wait(), 2)
            for text in ("one", "two", "three"):
                session.send(text, mode="now")
                await asyncio.sleep(0.05)
            await records(session, "turn_finished", 1)
        root, memory = await memory_of(session)
        assert user_texts(memory) == ["go", "one", "two", "three"]
        assert len(root.metadata["steer_now"]) == 1
        assert model.call_count == 2


class TestEdges:
    async def test_now_while_idle_starts_a_turn(self):
        async with ChatSession(agent(batch_model())) as session:
            session.send("hello", mode="now")
            (finished,) = await records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "success"
        root, _ = await memory_of(session)
        assert root.output.collect_text() == "answer to: hello"

    async def test_agent_without_steering_gets_it_as_a_followup(self):
        probe = Probe()
        a = agent(batch_model("search"), [probe.tool("search", read_only=True)], steering=False)
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(probe.started["search"].wait(), 2)
            session.send("hi", mode="now")
            await asyncio.sleep(0.3)
            assert "search" not in probe.ended  # no cut without steering
            probe.release.set()
            finished = await records(session, "turn_finished", 2)
        assert [f["status"]["code"] for f in finished] == ["success", "success"]
        assert probe.ended["search"] == "finished"

    async def test_message_record_says_now(self):
        async with ChatSession(agent(batch_model())) as session:
            session.send("hello", mode="now")
            (message,) = await records(session, "message", 1)
            await records(session, "turn_finished", 1)
        assert message["mode"] == "now"
