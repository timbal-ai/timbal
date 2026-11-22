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


def llm_calls() -> list[tuple[str | None, list[str]]]:
    """Each model call of the run that just ended: its system prompt and the names of the tools it was offered."""
    from timbal.state import get_run_context

    return [
        (
            span.input.get("system_prompt"),
            [tool["name"] if isinstance(tool, dict) else tool.name for tool in span.input.get("tools") or []],
        )
        for span in get_run_context()._trace.values()
        if span.path.endswith(".llm")
    ]


def llm_inputs(path: str = "bot.llm") -> list[dict]:
    """The input of each model call at ``path`` in the run that just ended: what the call's inputs became."""
    from timbal.state import get_run_context

    return [span.input for span in get_run_context()._trace.values() if span.path == path]


def results(out) -> list[str]:
    """The text of every tool result in the run's memory, in order."""
    from timbal.types.content import ToolResultContent

    return [
        "".join(item.text for item in c.content)
        for m in memory_of(out)
        for c in m.content
        if isinstance(c, ToolResultContent)
    ]


async def finished_turn(session) -> tuple[list[str], dict]:
    """What the session's first turn to finish sent the user, and its ``turn_finished`` record."""
    texts = []
    async for item in session.events():
        data = getattr(getattr(item, "item", None), "data", None)
        if isinstance(data, dict) and data.get("kind") == "message":
            texts.append(data["text"])
        if isinstance(item, dict) and item.get("kind") == "turn_finished":
            return texts, item
    raise AssertionError("the session closed before a turn finished")


async def turn_texts(session) -> list[str]:
    return (await finished_turn(session))[0]


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
        model = TestModel(
            responses=[
                calls(send("The code is 42.")),
                calls(send("I can't share that.", end_turn=True)),
                "must not be requested",
            ]
        )
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
            model=TestModel(
                responses=[
                    calls(("helper", {"prompt": "x"})),
                    calls(send("Got it.", end_turn=True)),
                    "must not be requested",
                ]
            ),
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


def sends_and_text(memory) -> tuple[list[str], list[str]]:
    """What a history holds of the agent's own words: the messages it sent (the calls, by their text) and
    its plain text."""
    from timbal.types.content import TextContent

    blocks = [c for m in memory if m.role == "assistant" for c in m.content]
    sends = [c.input["text"] for c in blocks if isinstance(c, ToolUseContent) and c.name == SEND_MESSAGE_NAME]
    return sends, [c.text for c in blocks if isinstance(c, TextContent)]


def pairs(memory) -> tuple[set[str], set[str]]:
    """The ids of the tool calls and of the tool results in a history: providers want them to match."""
    from timbal.types.content import ToolResultContent

    blocks = [c for m in memory for c in m.content]
    uses = {c.id for c in blocks if isinstance(c, ToolUseContent)}
    return uses, {c.id for c in blocks if isinstance(c, ToolResultContent)}


class TestCompaction:
    """What the agent told the user is the conversation: ``compact_tool_results`` keeps those calls, as calls."""

    async def test_compacting_tool_results_keeps_what_the_user_was_told(self):
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.types.content import ToolResultContent

        fresh()
        rail = guardrail(lambda text: text.replace("4111", "[card]"), stages=["model_output"], action="redact")
        model = TestModel(
            responses=[calls(("work", {}), send("Card 4111 charged.")), calls(send("Done.", True)), "not requested"]
        )
        agent = Agent(
            name="bot", model=model, tools=[Tool(name="work", handler=work)], delivery="tool", guardrails=[rail]
        )
        out, _ = await run(agent)
        memory = memory_of(out)
        results = [c for m in memory if m.role == "tool" for c in m.content if isinstance(c, ToolResultContent)]
        assert [c.delivered for c in results] == [None, "Card [card] charged.", "Done."]  # what the user saw

        for compact, work_left in (
            (compact_tool_results(), False),
            (compact_tool_results(replacement="[{tool_name} output removed]"), True),
        ):
            compacted = compact(memory)
            sends, plain = sends_and_text(compacted)
            assert sends == ["Card 4111 charged.", "Done."] and plain == []  # still calls, in order
            kept = [c for m in compacted if m.role == "tool" for c in m.content if isinstance(c, ToolResultContent)]
            # Their results too, which say what went out after the guardrail.
            assert [c.delivered for c in kept if c.delivered] == ["Card [card] charged.", "Done."]
            uses, answered = pairs(compacted)
            assert uses == answered and ("c0-work" in uses) is work_left  # the real tool's pair is compacted

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
        # An empty message fails. (Its own id, as a provider gives every call.)
        failed = Message(
            role="assistant",
            content=[ToolUseContent(id="failed", name=SEND_MESSAGE_NAME, input={"text": "   "})],
            stop_reason="tool_use",
        )
        model = TestModel(responses=[failed, calls(send("fine", True)), "must not be requested"])
        out, _ = await run(Agent(name="bot", model=model, delivery="tool"))
        memory = memory_of(out)
        delivered = [c.delivered for m in memory for c in m.content if isinstance(c, ToolResultContent)]
        assert delivered == [None, "fine"]
        assert sends_and_text(compact_tool_results()(memory))[0] == ["fine"]  # the failed one goes

    async def test_a_draft_is_never_turned_into_what_was_said(self):
        """The model's own text is a draft the user never read: next to a message in the same step, or in
        a message of its own. Whatever is compacted, it must stay told apart from what was sent."""
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.types.content import TextContent

        fresh()
        first = Message(
            role="assistant",
            content=[
                TextContent(text="private: the user sounds annoyed"),
                ToolUseContent(id="c0", name=SEND_MESSAGE_NAME, input={"text": "Here you go."}),
                ToolUseContent(id="c1", name="work", input={}),
            ],
            stop_reason="tool_use",
        )
        model = TestModel(responses=[first, calls(send("One more thing.")), "note to self: follow up tomorrow"])
        agent = Agent(name="bot", model=model, tools=[Tool(name="work", handler=work)], delivery="tool")
        out, messages = await run(agent)
        received = [m["text"] for m in messages]
        assert received == ["Here you go.", "One more thing."]
        for compact in (
            compact_tool_results(),
            compact_tool_results(replacement="[{tool_name} output removed]"),
            compact_tool_results(keep_last_n=1),
        ):
            compacted = compact(memory_of(out))
            sends, plain = sends_and_text(compacted)
            assert sends == received  # what was said: the calls
            assert plain == ["private: the user sounds annoyed", "note to self: follow up tomorrow"]  # what was not
            uses, answered = pairs(compacted)
            assert uses == answered

    async def test_what_was_said_survives_passes_and_folds_inside_a_turn(self, monkeypatch):
        """A pass inside a turn can leave the history ending on the assistant's side (a draft whose call was
        dropped), and that message is folded into the next step's. A later pass over the folded message
        must lose nothing the user was told, and must not make a draft of it either."""
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.types.content import TextContent, ToolResultContent

        # A small window: each fetched payload crosses the ratio, so the turn compacts after each one.
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000)

        def fetch(n: int) -> str:
            return f"payload-{n}: " + "x" * 4_000

        def step(number: int, *blocks) -> Message:
            # Every call with an id of its own, as a provider gives them.
            content = [
                TextContent(text=block)
                if isinstance(block, str)
                else ToolUseContent(id=f"s{number}-{i}", name=block[0], input=block[1])
                for i, block in enumerate(blocks)
            ]
            return Message(role="assistant", content=content, stop_reason="tool_use")

        steps = [
            step(1, send("Looking into it.")),
            step(2, "draft: fetch the first page", ("fetch", {"n": 1})),
            step(3, "draft: tell them, fetch more", send("Found it, fixing."), ("fetch", {"n": 2})),
            step(4, ("fetch", {"n": 3})),
            step(5, "draft: one more page", ("fetch", {"n": 4})),
            step(6, send("All done.", end_turn=True)),
        ]
        asked = []

        def handler(_messages):
            asked.append(1)
            return steps[len(asked) - 1] if len(asked) <= len(steps) else "must not be requested"

        fresh()
        agent = Agent(
            name="bot",
            model=TestModel(handler=handler),
            tools=[Tool(name="fetch", handler=fetch)],
            delivery="tool",
            memory_compaction=compact_tool_results(),
            memory_compaction_ratio=0.75,
        )
        out, messages = await run(agent)
        assert out.status.code == "success", out.error
        received = [m["text"] for m in messages]
        assert received == ["Looking into it.", "Found it, fixing.", "All done."]
        assert out.metadata["compaction"]["passes"] >= 2
        memory = memory_of(out)
        # A fold happened: no step wrote two texts, so a message holding two is two steps' messages in one.
        texts_per_message = [
            len([c for c in m.content if isinstance(c, TextContent)]) for m in memory if m.role == "assistant"
        ]
        assert max(texts_per_message) >= 2, texts_per_message
        sends, plain = sends_and_text(memory)
        assert sends == received  # every one of them, once, in order, still a call
        assert all(text.startswith("draft:") for text in plain)  # and nothing said became a draft
        blocks = [c for m in memory for c in m.content]
        uses = sorted(c.id for c in blocks if isinstance(c, ToolUseContent))
        answered = sorted(c.id for c in blocks if isinstance(c, ToolResultContent))
        assert uses == answered  # as lists: each call has its one result


class TestTheToolsName:
    async def test_a_tool_set_that_has_the_name(self):
        """An MCP server's own ``send_message`` is known only once resolved: two tools of one name is a 400."""
        from timbal.core.tool_set import ToolSet

        class Messenger(ToolSet):
            async def resolve(self):
                return [Tool(name=SEND_MESSAGE_NAME, handler=lambda to, body: f"posted {body} to {to}")]

        fresh()
        model = TestModel(
            responses=[
                calls(
                    (SEND_MESSAGE_NAME, {"to": "ops", "body": "hi"}),
                    ("send_to_user", {"text": "Told ops.", "end_turn": True}),
                ),
                "must not be requested",
            ]
        )
        agent = Agent(name="bot", model=model, tools=[Messenger()], delivery="tool")
        out, messages = await run(agent)
        assert out.status.code == "success", out.error
        assert model.call_count == 1
        ((prompt, tools),) = llm_calls()
        assert tools == [SEND_MESSAGE_NAME, "send_to_user"]
        assert "send_to_user" in prompt  # the note names the tool the model was given
        assert [m["text"] for m in messages] == ["Told ops."]
        assert results(out) == ["posted hi to ops", "Sent. The turn ends after this step."]

    def test_every_other_name_is_taken_too(self):
        names = (SEND_MESSAGE_NAME, "send_to_user", "reply_to_user")
        tools = [Tool(name=name, handler=lambda: "x") for name in names]
        agent = Agent(name="bot", model=TestModel(responses=["x"]), tools=tools, delivery="tool")
        assert agent._send_tool_name() == "send_message_2"

    async def test_a_name_a_tool_policy_hides_is_not_taken(self):
        """The policy narrows what a run may call, not what a name means."""
        fresh()
        ctx = RunContext()
        ctx._tool_policy = []
        set_run_context(ctx)
        model = TestModel(responses=[calls(("send_to_user", {"text": "hi", "end_turn": True})), "not requested"])
        own = Tool(name=SEND_MESSAGE_NAME, handler=lambda to: f"posted to {to}")
        out, messages = await run(Agent(name="bot", model=model, tools=[own], delivery="tool"))
        assert [tools for _, tools in llm_calls()] == [["send_to_user"]]  # the reply is not a tool the policy hides
        assert [m["text"] for m in messages] == ["hi"]

    async def test_with_tool_search(self):
        """Never deferred: offered on every step next to ``search_tools``, and reachable through ``call_tool``."""
        for mode, first, text in (
            ("client", calls(send("direct")), "direct"),
            (
                "dispatch",
                calls(("call_tool", {"name": SEND_MESSAGE_NAME, "arguments": {"text": "dispatched"}})),
                "dispatched",
            ),
        ):
            fresh()
            model = TestModel(responses=[first, calls(send("bye", end_turn=True)), "must not be requested"])
            weather = Tool(
                name="get_weather",
                description="Weather for a city.",
                handler=lambda city: f"sunny in {city}",
                defer_loading=True,
            )
            agent = Agent(
                name="bot",
                model=model,
                tools=[weather, Tool(name="work", handler=work)],
                delivery="tool",
                tool_search=mode,
            )
            out, messages = await run(agent)
            assert out.status.code == "success", out.error
            offered = ["work", SEND_MESSAGE_NAME, "search_tools"] + (["call_tool"] if mode == "dispatch" else [])
            assert [tools for _, tools in llm_calls()] == [offered, offered]
            assert [m["text"] for m in messages] == [text, "bye"]

    async def test_only_the_delivery_tool_marks_a_result_delivered(self):
        from timbal.state import get_run_context
        from timbal.types.content import ToolResultContent

        def mark() -> str:
            get_run_context().current_span().metadata["delivered"] = "words the assistant never said"
            return "ok"

        fresh()
        model = TestModel(responses=[calls(("mark", {}), send("hi", True)), "must not be requested"])
        out, _ = await run(Agent(name="bot", model=model, tools=[Tool(name="mark", handler=mark)], delivery="tool"))
        delivered = [c.delivered for m in memory_of(out) for c in m.content if isinstance(c, ToolResultContent)]
        assert delivered == [None, "hi"]  # compaction would turn the first into assistant text


class TestWhatCountsAsSent:
    async def test_a_turn_that_sent_a_message_is_not_silent(self):
        for responses in (
            [calls(send("Standup in 5.")), "NO_REPLY"],  # the silent reply as its last text
            [calls(send("Standup in 5."), send("NO_REPLY")), "must not be requested"],  # or as a message
        ):
            fresh()
            ctx = RunContext()
            ctx._silence_token = "NO_REPLY"
            set_run_context(ctx)
            out, messages = await run(Agent(name="bot", model=TestModel(responses=responses), delivery="tool"))
            assert [m["text"] for m in messages] == ["Standup in 5."]
            assert out.output.collect_text() == "Standup in 5."
            assert "silent" not in out.metadata  # a channel hides a silent turn

    async def test_a_message_a_guardrail_leaves_nothing_of_is_not_sent(self):
        fresh()
        wipe = guardrail(
            lambda text: "" if "secret" in text else text, stages=["model_output"], action="redact", name="wipe"
        )
        model = TestModel(
            responses=[calls(send("secret secret")), calls(send("I can't say.", end_turn=True)), "not requested"]
        )
        out, messages = await run(Agent(name="bot", model=model, delivery="tool", guardrails=[wipe]))
        assert [m["text"] for m in messages] == ["I can't say."]  # no empty message
        assert out.output.collect_text() == "I can't say."
        assert "removed all of this message" in results(out)[0]

    async def test_the_receipt_says_what_a_guardrail_changed(self):
        fresh()
        rail = guardrail(lambda text: text.replace("4111", "[card]"), stages=["model_output"], action="redact")
        model = TestModel(responses=[calls(send("Card 4111 charged.")), calls(send("Done.", True)), "not requested"])
        out, messages = await run(Agent(name="bot", model=model, delivery="tool", guardrails=[rail]))
        assert [m["text"] for m in messages] == ["Card [card] charged.", "Done."]
        assert results(out) == [
            "Sent, after a guardrail changed it. The user received: Card [card] charged.",
            "Sent. The turn ends after this step.",
        ]

    async def test_an_empty_answer_delivers_nothing(self):
        """An empty answer is not kept in memory, so the last message there is the user's prompt, or a
        runtime notice: neither is the agent's reply."""
        fresh()
        agent = Agent(name="bot", model=TestModel(responses=[""]), delivery="tool")
        out, messages = await run(agent, "what is my balance?")
        assert messages == [] and "delivery" not in out.metadata

        fresh()
        model = TestModel(responses=[calls(("work", {})), ""])  # the budget notice is last in memory
        agent = Agent(name="bot", model=model, tools=[Tool(name="work", handler=work)], delivery="tool", max_iter=1)
        out, messages = await run(agent)
        assert messages == [] and "delivery" not in out.metadata

    async def test_a_message_sent_does_not_reset_the_denial_count(self):
        """Telling the user "trying again" is not a call that went through."""
        refused = guardrail(lambda _args: Verdict.block("no"), stages=["tool_args"])

        def agent(**limits):
            model = TestModel(
                responses=[
                    calls(send("Trying."), ("act", {})),  # refused once, next to a message
                    calls(send("Trying again.")),  # a step that only tells the user
                    calls(("act", {})),  # refused twice: nothing went through in between
                    "I could not do that.",
                    "must not be requested",
                ]
            )
            tools = [Tool(name="act", handler=lambda: "acted", guardrails=[refused])]
            return Agent(name="bot", model=model, tools=tools, delivery="tool", **limits)

        for limits, record in (
            ({"max_consecutive_denials": 2}, {"denials": 2}),
            ({"max_consecutive_denials": None, "max_denials_window": (2, 2)}, {"denials": 2, "window": 2}),
        ):
            fresh()
            out, messages = await run(agent(**limits))
            assert out.metadata.get("denial_limit") == [record], limits
            offered = ["act", SEND_MESSAGE_NAME]
            assert [tools for _, tools in llm_calls()] == [offered, offered, offered, []]
            # The step the limit left without tools answers in text, which is the reply.
            assert [m["text"] for m in messages] == ["Trying.", "Trying again.", "I could not do that."]

    async def test_a_draft_is_screened_before_it_is_delivered(self):
        """The fallback is the turn's final answer: the model-output guardrails have seen it by then."""
        fresh()
        no_secrets = guardrail(
            lambda text: Verdict.block("secret") if "42" in text else True, stages=["model_output"], name="no_secrets"
        )
        agent = Agent(
            name="bot", model=TestModel(responses=["The code is 42."]), delivery="tool", guardrails=[no_secrets]
        )
        out, messages = await run(agent)
        assert out.status.code == "blocked" and messages == []

        fresh()
        redact = guardrail(lambda text: text.replace("42", "[code]"), stages=["model_output"], action="redact")
        agent = Agent(name="bot", model=TestModel(responses=["The code is 42."]), delivery="tool", guardrails=[redact])
        out, messages = await run(agent)
        assert [m["text"] for m in messages] == ["The code is [code]."]
        assert out.output.collect_text() == "The code is [code]."


class TestAStepWithoutTheTool:
    """A step with no tools cannot send: its text is the reply, also after messages the turn sent."""

    async def test_the_answer_at_max_iter(self):
        fresh()
        model = TestModel(responses=[calls(send("On it."), ("work", {})), "All done: worked."])
        agent = Agent(name="bot", model=model, tools=[Tool(name="work", handler=work)], delivery="tool", max_iter=1)
        out, messages = await run(agent)
        assert [tools for _, tools in llm_calls()] == [["work", SEND_MESSAGE_NAME], []]
        assert [m["text"] for m in messages] == ["On it.", "All done: worked."]
        assert messages[-1] == {
            "kind": "message",
            "text": "All done: worked.",
            "index": 1,
            "end_turn": True,
            "fallback": True,
        }
        assert out.output.collect_text() == "On it.\n\nAll done: worked."
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 2, "fallback": True}

    async def test_the_question_after_a_denial_limit(self):
        fresh()
        refused = guardrail(lambda _args: Verdict.block("no"), stages=["tool_args"])
        model = TestModel(responses=[calls(send("Trying."), ("act", {})), "I could not do that. Another way?"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Tool(name="act", handler=lambda: "acted", guardrails=[refused])],
            delivery="tool",
            max_consecutive_denials=1,
        )
        out, messages = await run(agent)
        assert [tools for _, tools in llm_calls()] == [["act", SEND_MESSAGE_NAME], []]
        assert [m["text"] for m in messages] == ["Trying.", "I could not do that. Another way?"]

    async def test_the_reply_of_on_max_iter(self):
        from timbal.core.agent import DEFAULT_MAX_ITER_STOP_MESSAGE

        stopped = DEFAULT_MAX_ITER_STOP_MESSAGE.format(max_iter=1)
        for first, sent in (
            (calls(send("On it."), ("work", {})), ["On it.", stopped]),
            (calls(("work", {})), [stopped]),
        ):
            fresh()
            model = TestModel(responses=[first, "must not be requested"])
            agent = Agent(
                name="bot",
                model=model,
                tools=[Tool(name="work", handler=work)],
                delivery="tool",
                max_iter=1,
                on_max_iter="stop",
            )
            out, messages = await run(agent)
            assert model.call_count == 1 and out.status.reason == "max_iter"
            assert [m["text"] for m in messages] == sent
            assert out.output.collect_text() == "\n\n".join(sent)

    async def test_a_slash_command_reaches_the_user(self):
        """A command's output is the turn's answer and no model runs: nothing else could send it."""

        def ping() -> str:
            return "pong"

        fresh()
        model = TestModel(responses=["must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Tool(handler=ping, command="ping")], delivery="tool")
        out, messages = await run(agent, "/ping")
        assert model.call_count == 0
        assert messages == [{"kind": "message", "text": "pong", "index": 0, "end_turn": True, "fallback": True}]

    async def test_a_slash_command_whose_output_is_not_a_string(self):
        """What the user reads is the output's text, not the Python repr of a message or a dict."""

        def balance() -> dict:
            return {"balance": 12}

        helper = Agent(name="helper", model=TestModel(responses=["helper says hi"]), command="helper")
        for command, tool, text in (
            ("/balance", Tool(handler=balance, command="balance"), '{"balance": 12}'),
            ("/helper hello", helper, "helper says hi"),
        ):
            fresh()
            agent = Agent(
                name="bot", model=TestModel(responses=["must not be requested"]), tools=[tool], delivery="tool"
            )
            out, messages = await run(agent, command)
            assert [m["text"] for m in messages] == [text]  # once: not the helper's own model call too


class TestAcrossAPause:
    """A turn paused for an approval goes on in the run that resumes it: what it sent still counts."""

    def _agent(self, handler, charged):
        from timbal.state.tracing.providers import InMemoryTracingProvider

        return Agent(
            name="bot",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: charged.append(1) or "charged", requires_approval=True)],
            delivery="tool",
            tracing_provider=InMemoryTracingProvider,
        )

    async def _pause_and_resume(self, agent, sent=("Charging your card now.",), silence=None, resumed_silence=None):
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.events import ApprovalEvent

        def context(parent_id=None):
            ctx = RunContext(parent_id=parent_id, tracing_provider=InMemoryTracingProvider)
            ctx._silence_token = resumed_silence if parent_id else silence
            set_run_context(ctx)

        fresh()
        context()
        events = [event async for event in agent(prompt="pay")]
        parked = events[-1]
        assert parked.status.reason == "approval_required"
        texts = [e.item.data["text"] for e in events if isinstance(e, DeltaEvent) and e.item.type == "custom"]
        assert texts == list(sent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        context(parked.run_id)
        return await run(agent, "pay", resume={approval.approval_id: True})

    async def test_a_draft_after_the_resume_stays_private(self):
        charged = []

        def handler(messages):
            if not any(m.role == "tool" for m in messages):
                return calls(send("Charging your card now."), ("charge", {}))
            return "note to self: it went through"

        out, messages = await self._pause_and_resume(self._agent(handler, charged))
        assert out.status.code == "success", out.error
        assert charged == [1]
        assert messages == []  # the turn had answered: the note is a draft, not the reply of a turn that said nothing
        assert out.output.collect_text() == ""
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "before_pause": 1}

    async def test_end_turn_asked_before_the_pause_ends_the_turn_after_it(self):
        charged = []
        asked = []

        def handler(messages):
            asked.append(len(messages))
            if len(asked) > 1:
                return "must not be requested"
            return calls(send("Charging your card now.", end_turn=True), ("charge", {}))

        out, messages = await self._pause_and_resume(self._agent(handler, charged))
        assert out.status.code == "success", out.error
        assert charged == [1]
        assert len(asked) == 1  # no model call after the batch it ended its turn with
        assert messages == []
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "before_pause": 1, "end_turn": True}

    async def test_staying_silent_before_the_pause_ends_the_turn_silently_after_it(self):
        # Resumed by the channel that allowed the silence, or by a plain call that carries no token.
        for resumed_silence, output in (("NO_REPLY", "NO_REPLY"), (None, "")):
            charged = []
            asked = []

            def handler(messages, asked=asked):
                asked.append(len(messages))
                if len(asked) > 1:
                    return "must not be requested"
                return calls(send("NO_REPLY"), ("charge", {}))

            out, messages = await self._pause_and_resume(
                self._agent(handler, charged), sent=(), silence="NO_REPLY", resumed_silence=resumed_silence
            )
            assert out.status.code == "success", out.error
            assert charged == [1]
            assert len(asked) == 1 and messages == []  # no model call whose text would go out as the reply
            assert out.metadata.get("silent") is True and out.output.collect_text() == output
            assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "silent": True, "end_turn": True}

    async def test_a_turn_that_spoke_and_then_stayed_silent_is_not_a_silent_turn(self):
        """Silent means the turn told the user nothing: the same with a pause in it as without one."""
        charged = []
        asked = []

        def handler(messages):
            asked.append(len(messages))
            if len(asked) == 1:
                return calls(send("Charging your card now."))
            if len(asked) == 2:
                return calls(send("NO_REPLY"), ("charge", {}))
            return "must not be requested"

        out, messages = await self._pause_and_resume(
            self._agent(handler, charged), silence="NO_REPLY", resumed_silence="NO_REPLY"
        )
        assert out.status.code == "success", out.error
        assert charged == [1] and len(asked) == 2 and messages == []
        assert "silent" not in out.metadata
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "before_pause": 1, "end_turn": True}


class TestPerCall:
    async def test_a_call_can_ask_for_text(self):
        fresh()
        agent = Agent(name="bot", model=TestModel(responses=["Plain answer."]), delivery="tool")
        out, messages = await run(agent, delivery="text")
        assert messages == [] and out.output.collect_text() == "Plain answer."
        assert llm_calls() == [(None, [])]  # no tool to send with, no note about one
        assert "delivery" not in out.metadata

    async def test_only_text_and_only_for_an_agent_that_delivers_by_tool(self):
        """``delivery`` may be an application's own input: nothing else about it changes."""
        fresh()
        plain = Agent(name="bot", model=TestModel(responses=["Plain answer."]))
        out, messages = await run(plain, delivery="tool")  # not a way to turn it on
        assert messages == [] and llm_calls() == [(None, [])]
        assert out.input["delivery"] == "tool"  # still there for the application's own code

        fresh()
        agent = Agent(name="bot", model=TestModel(responses=["Plain answer."]), delivery="tool")
        out, messages = await run(agent, delivery="courier")  # any other value is not an override
        assert out.status.code == "success", out.error
        assert [m["text"] for m in messages] == ["Plain answer."] and messages[0]["fallback"] is True

    async def test_a_structured_answer_is_the_answer(self):
        """With ``output_model`` the turn ends on the model's first answer and runs no tool call of it."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str

        fresh()
        model = TestModel(responses=['{"answer": "x"}'])
        out, messages = await run(Agent(name="bot", model=model, output_model=Answer, delivery="tool"))
        assert out.output == Answer(answer="x") and messages == []
        assert [tools for _, tools in llm_calls()] == [[]]  # no tool it could call and never see run
        assert "send_message" not in (llm_calls()[0][0] or "")

    async def test_any_other_value_goes_where_the_other_inputs_go(self):
        """Only ``"text"`` is the agent's to take. An application's own ``delivery`` input reaches the model
        call's input like its other inputs, as it does for an agent that does not deliver by tool."""
        for value in ("courier", "tool", "TEXT", 7, {"slot": "am"}):
            fresh()
            agent = Agent(name="bot", model=TestModel(responses=["Plain answer."]), delivery="tool")
            out, messages = await run(agent, delivery=value, user="ada")
            assert out.status.code == "success", out.error
            (llm_input,) = llm_inputs()
            assert llm_input["delivery"] == value and llm_input["user"] == "ada"
            assert [m["text"] for m in messages] == ["Plain answer."]  # and it is not an override either

        fresh()
        agent = Agent(name="bot", model=TestModel(responses=["Plain answer."]), delivery="tool")
        out, messages = await run(agent, delivery="text", user="ada")
        (llm_input,) = llm_inputs()
        assert "delivery" not in llm_input and llm_input["user"] == "ada"  # the override is the agent's
        assert messages == []

    async def test_an_agent_with_nothing_to_override_keeps_an_input_of_that_name(self):
        """A subagent and an agent with an output model answer in text whatever is passed: for them every
        value, ``"text"`` too, is the application's, as it is for an agent not set to deliver by tool."""
        from pydantic import BaseModel

        for value in ("courier", "text"):
            fresh()
            helper = Agent(name="helper", model=TestModel(responses=["inner answer"]), delivery="tool")
            model = TestModel(responses=[calls(("helper", {"prompt": "x", "delivery": value})), "done"])
            out, _ = await run(Agent(name="bot", model=model, tools=[helper]))
            assert out.status.code == "success", out.error
            (llm_input,) = llm_inputs("bot.helper.llm")
            assert llm_input["delivery"] == value

        class Answer(BaseModel):
            answer: str

        fresh()
        model = TestModel(responses=['{"answer": "x"}'])
        out, _ = await run(Agent(name="bot", model=model, output_model=Answer, delivery="tool"), delivery="text")
        assert out.output == Answer(answer="x")
        (llm_input,) = llm_inputs()
        assert llm_input["delivery"] == "text"


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
            await asyncio.wait_for(entered.wait(), 30)
            session.send("and one more thing", mode="steer")  # queued by the time send() returns
            release.set()
            texts = await asyncio.wait_for(turn_texts(session), 30)
        assert texts == ["Started, that's all.", "Also handled: and one more thing"]

    async def test_end_turn_does_not_end_a_turn_that_an_event_joined(self):
        """Like a final answer in text: what joined the turn's memory is answered before the turn ends."""
        from timbal.core.chat_session import ChatSession

        fresh()
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow() -> str:
            entered.set()
            await release.wait()
            return "slow done"

        seen = []

        def handler(messages):
            seen.append([m.collect_text() for m in messages if m.is_runtime()])
            if not any(m.role == "tool" for m in messages):
                return calls(("slow", {}), send("Started, that's all.", end_turn=True))
            return calls(send("And the deploy finished.", end_turn=True))

        agent = Agent(
            name="bot", model=TestModel(handler=handler), tools=[Tool(name="slow", handler=slow)], delivery="tool"
        )
        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(entered.wait(), 30)
            event = session.store.post_event("the deploy finished", lane="next")  # the application's
            release.set()
            texts = await asyncio.wait_for(turn_texts(session), 30)
        assert event.state == "delivered"
        # Taken by the turn, so shown to its model before it ends: nothing else would ever answer it.
        assert texts == ["Started, that's all.", "And the deploy finished."]
        assert any("the deploy finished" in text for text in seen[-1])

    async def test_a_turn_that_goes_on_after_staying_silent_can_still_answer(self):
        """Silence was its last word until the user wrote: what it says next is not a silent turn's draft."""
        from timbal.core.chat_session import ChatSession

        fresh()
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow() -> str:
            entered.set()
            await release.wait()
            return "slow done"

        def handler(messages):
            if not any(m.role == "tool" for m in messages):
                return calls(("slow", {}), send("NO_REPLY"))
            return "It finished a minute ago."

        agent = Agent(
            name="bot", model=TestModel(handler=handler), tools=[Tool(name="slow", handler=slow)], delivery="tool"
        )
        async with ChatSession(agent) as session:
            session.store.post_event("the deploy finished")  # the application's: its turn may stay silent
            await asyncio.wait_for(entered.wait(), 30)
            session.send("wait, what happened?", mode="steer")
            release.set()
            texts, finished = await asyncio.wait_for(finished_turn(session), 30)
        assert texts == ["It finished a minute ago."]
        assert not finished.get("silent")
