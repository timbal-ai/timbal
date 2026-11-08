"""delivery="tool": replies go out through send_message; plain text is a draft."""

import asyncio

from timbal import Agent, Tool
from timbal.core.agent import SEND_MESSAGE_NAME
from timbal.core.test_model import TestModel
from timbal.guardrails import Verdict, guardrail
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.types.content import ToolUseContent
from timbal.types.events.delta import DeltaEvent
from timbal.types.message import Message


def calls(*items: tuple[str, dict]) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"c{i}-{name}", name=name, input=input) for i, (name, input) in enumerate(items)],
        stop_reason="tool_use",
    )


def send(text: str, end_turn: bool = False) -> tuple[str, dict]:
    return SEND_MESSAGE_NAME, {"text": text, "end_turn": end_turn}


def fresh():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


async def run(agent: Agent, prompt: str = "go", **kwargs):
    events = [event async for event in agent(prompt=prompt, **kwargs)]
    messages = [
        e.item.data
        for e in events
        if isinstance(e, DeltaEvent) and e.item.type == "custom" and e.item.data.get("kind") == "message"
    ]
    return events[-1], messages


def work() -> str:
    return "worked"


class TestSendMessage:
    async def test_several_messages_and_end_turn_skips_the_last_call(self):
        fresh()
        model = TestModel(
            responses=[
                calls(send("On it."), ("work", {})),
                calls(send("Done: worked.", end_turn=True)),
                "this response must never be requested",
            ]
        )
        agent = Agent(name="bot", model=model, tools=[Tool(name="work", handler=work)], delivery="tool")
        out, messages = await run(agent)
        assert out.status.code == "success", out.error
        assert model.call_count == 2  # end_turn: no call after the last batch
        assert [m["text"] for m in messages] == ["On it.", "Done: worked."]
        assert messages[-1]["end_turn"] is True
        assert out.output.collect_text() == "On it.\n\nDone: worked."

    async def test_a_turn_that_sent_nothing_delivers_its_draft(self):
        fresh()
        agent = Agent(name="bot", model=TestModel(responses=["Plain answer."]), delivery="tool")
        out, messages = await run(agent)
        assert out.output.collect_text() == "Plain answer."
        assert messages == [
            {"kind": "message", "text": "Plain answer.", "index": 0, "end_turn": True, "fallback": True}
        ]

    async def test_text_after_sending_stays_a_draft(self):
        fresh()
        model = TestModel(responses=[calls(send("Here you go.")), "note to self: user seemed happy"])
        agent = Agent(name="bot", model=model, delivery="tool")
        out, messages = await run(agent)
        assert [m["text"] for m in messages] == ["Here you go."]
        assert out.output.collect_text() == "Here you go."

    async def test_silence_sends_nothing(self):
        fresh()
        ctx = RunContext()
        ctx._silence_token = "NO_REPLY"
        set_run_context(ctx)
        agent = Agent(name="bot", model=TestModel(responses=["NO_REPLY"]), delivery="tool")
        out, messages = await run(agent)
        assert messages == []
        assert out.status.code == "success"

    async def test_the_silent_reply_sent_as_a_message_is_not_delivered(self):
        fresh()
        ctx = RunContext()
        ctx._silence_token = "NO_REPLY"
        set_run_context(ctx)
        model = TestModel(responses=[calls(send("NO_REPLY")), "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool")
        out, messages = await run(agent)
        assert messages == [] and model.call_count == 1  # silent, and the turn ended there
        assert out.status.code == "success" and out.output.collect_text() == "NO_REPLY"  # same as a silent text turn

    async def test_the_model_output_guardrails_screen_what_is_sent(self):
        fresh()
        no_secrets = guardrail(
            lambda text: Verdict.block("secret") if "42" in text else True, stages=["model_output"], name="no_secrets"
        )
        model = TestModel(responses=[calls(send("The code is 42.")), calls(send("I can't share that.", end_turn=True))])
        agent = Agent(name="bot", model=model, delivery="tool", guardrails=[no_secrets])
        out, messages = await run(agent)
        assert [m["text"] for m in messages] == ["I can't share that."]
        assert out.output.collect_text() == "I can't share that."

    async def test_a_subagent_answers_with_text(self):
        fresh()
        inner_seen = []

        def inner(messages):
            inner_seen.append(messages)
            return "inner answer"

        inner_agent = Agent(name="helper", model=TestModel(handler=inner), delivery="tool")
        outer = Agent(
            name="bot",
            model=TestModel(responses=[calls(("helper", {"prompt": "x"})), calls(send("Got it.", end_turn=True))]),
            tools=[inner_agent],
            delivery="tool",
        )
        out, messages = await run(outer)
        assert [m["text"] for m in messages] == ["Got it."]  # the helper's text was not delivered
        assert out.output.collect_text() == "Got it."

    async def test_off_by_default(self):
        fresh()
        agent = Agent(name="bot", model=TestModel(responses=["hi"]))
        out, messages = await run(agent)
        assert messages == [] and out.output.collect_text() == "hi"

    def test_renamed_when_an_agent_tool_has_the_name(self):
        def send_message(x: str) -> str:
            return x

        agent = Agent(name="bot", model=TestModel(responses=["x"]), tools=[send_message], delivery="tool")
        assert agent._send_tool_name() == "send_to_user"


def memory_of(out):
    from timbal.state import get_run_context

    return [Message.validate(m) for m in get_run_context()._trace[out.call_id].memory]


class TestCompaction:
    async def test_compacting_tool_results_keeps_what_the_user_was_told_as_text(self):
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.types.content import TextContent, ToolResultContent

        fresh()
        rail = guardrail(lambda text: text.replace("4111", "[card]"), stages=["model_output"], action="redact")
        model = TestModel(responses=[calls(("work", {}), send("Card 4111 charged.")), calls(send("Done.", True))])
        agent = Agent(
            name="bot", model=model, tools=[Tool(name="work", handler=work)], delivery="tool", guardrails=[rail]
        )
        out, _ = await run(agent)
        memory = memory_of(out)
        results = [c for m in memory if m.role == "tool" for c in m.content if isinstance(c, ToolResultContent)]
        assert [c.delivered for c in results] == [None, "Card [card] charged.", "Done."]  # what the user saw

        for compact in (compact_tool_results(), compact_tool_results(replacement="[{tool_name} output removed]")):
            compacted = compact(memory)
            assistant = [
                c.text for m in compacted if m.role == "assistant" for c in m.content if isinstance(c, TextContent)
            ]
            assert assistant == ["Card [card] charged.", "Done."]
            assert not any(
                isinstance(c, ToolUseContent) and c.name == SEND_MESSAGE_NAME for m in compacted for c in m.content
            )
            ids = {c.id for m in compacted for c in m.content if isinstance(c, ToolResultContent | ToolUseContent)}
            assert all(not c.delivered for m in compacted for c in m.content if isinstance(c, ToolResultContent))
            assert ids <= {"c0-work"}  # only the real tool's pair may remain (as a placeholder)

    async def test_a_failed_send_or_an_ordinary_send_message_tool_is_compacted_like_any_tool(self):
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.types.content import ToolResultContent

        fresh()
        # delivery off: an app's own "send_message" tool (Slack, email) is tool output, not the conversation.
        model = TestModel(responses=[calls(send("hi")), "ok"])
        agent = Agent(name="bot", model=model, tools=[Tool(name=SEND_MESSAGE_NAME, handler=lambda **_: "posted")])
        out, _ = await run(agent)
        memory = memory_of(out)
        assert not any(c.delivered for m in memory for c in m.content if isinstance(c, ToolResultContent))
        assert all(m.role != "tool" for m in compact_tool_results()(memory))

        fresh()
        model = TestModel(responses=[calls(send("   ")), calls(send("fine", True))])  # an empty message fails
        out, _ = await run(Agent(name="bot", model=model, delivery="tool"))
        delivered = [c.delivered for m in memory_of(out) for c in m.content if isinstance(c, ToolResultContent)]
        assert delivered == [None, "fine"]


class TestWithChatSession:
    async def test_a_message_steered_in_during_the_last_batch_is_answered(self):
        from timbal.core.chat_session import ChatSession

        fresh()
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow() -> str:
            entered.set()
            await release.wait()
            return "slow done"

        def handler(messages):
            said = [m.collect_text() for m in messages if m.role == "user" and not m.is_runtime()]
            if said[-1] == "start" and not any(m.role == "tool" for m in messages):
                return calls(("slow", {}), send("Started, that's all.", end_turn=True))
            return calls(send(f"Also handled: {said[-1]}", end_turn=True))

        agent = Agent(
            name="bot", model=TestModel(handler=handler), tools=[Tool(name="slow", handler=slow)], delivery="tool"
        )
        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(entered.wait(), 2)
            session.send("and one more thing", mode="steer")
            await asyncio.sleep(0.05)
            release.set()
            texts = []
            async for item in session.events():
                data = getattr(getattr(item, "item", None), "data", None)
                if isinstance(data, dict) and data.get("kind") == "message":
                    texts.append(data["text"])
                if isinstance(item, dict) and item.get("kind") == "turn_finished":
                    break
        assert texts == ["Started, that's all.", "Also handled: and one more thing"]
