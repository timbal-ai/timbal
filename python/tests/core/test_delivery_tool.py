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
        if isinstance(e, DeltaEvent)
        and e.item.type == "custom"
        and isinstance(e.item.data, dict)  # (a generator tool's chunks are custom items too)
        and e.item.data.get("kind") == "message"
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

    async def test_the_agent_s_tool_args_guardrails_run_on_a_send(self):
        """To the agent's ``tool_args`` guardrails a send is a call like the others: an application that
        gates calls at that stage gates this one too. A send it blocks emits no message, and the model is
        told, so it can write another."""
        seen = []

        def gate(_text: str, ctx):
            seen.append(ctx.tool_name)
            if ctx.tool_name == SEND_MESSAGE_NAME and "secret" in ctx.tool_args["text"]:
                return Verdict.block("not for users")
            return True

        fresh()
        model = TestModel(
            responses=[
                calls(("work", {})),
                calls(send("the secret plan", end_turn=True)),
                calls(send("a public note", end_turn=True)),
                "must not be requested",
            ]
        )
        agent = Agent(
            name="bot",
            model=model,
            tools=[work],
            delivery="tool",
            guardrails=[guardrail(gate, stages=["tool_args"], name="gate")],
        )
        out, messages = await run(agent)
        assert seen == ["work", SEND_MESSAGE_NAME, SEND_MESSAGE_NAME]
        assert [m["text"] for m in messages] == ["a public note"] and model.call_count == 3
        assert out.output.collect_text() == "a public note"
        assert any("[Blocked by guardrail]" in result for result in results(out))

    async def test_a_tool_args_guardrail_can_ask_a_person_before_a_send(self):
        """An escalation at that stage is an approval gate: nothing is sent until a person lets it go."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.events import ApprovalEvent

        def ask(_text: str, ctx):
            return Verdict.escalate("a person checks what goes out") if ctx.tool_name == SEND_MESSAGE_NAME else True

        fresh()
        set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
        model = TestModel(responses=[calls(send("Your refund is approved.", end_turn=True)), "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            delivery="tool",
            guardrails=[guardrail(ask, stages=["tool_args"], name="ask")],
            tracing_provider=InMemoryTracingProvider,
        )
        events = [event async for event in in_time(agent(prompt="go"))]
        parked = events[-1]
        assert parked.status.reason == "approval_required" and message_events(events) == []
        approval = next(event for event in events if isinstance(event, ApprovalEvent))
        set_run_context(RunContext(parent_id=parked.run_id, tracing_provider=InMemoryTracingProvider))
        out, messages = await run(agent, resume={approval.approval_id: True})
        assert [m["text"] for m in messages] == ["Your refund is approved."] and model.call_count == 1
        assert out.output.collect_text() == "Your refund is approved."

    async def test_a_draft_does_not_get_round_the_gate_on_sends(self):
        """A draft that is delivered stands for a send the model did not make, so the same ``tool_args``
        guardrails decide on it: what they refuse does not reach the user as a fallback, in a message or in
        the run's output, whether the model tried a send first or never called the tool."""

        def gate(_text: str, ctx):
            if ctx.tool_name == SEND_MESSAGE_NAME and "secret" in ctx.tool_args["text"]:
                return Verdict.block("not for users")
            return True

        rail = guardrail(gate, stages=["tool_args"], name="gate")
        for responses, limits, sent in (
            ([calls(send("the secret plan", True)), "the secret plan"], {}, []),  # refused, then said in text
            (["the secret plan"], {}, []),  # never sent: the fallback is the only way out
            ([calls(send("the secret plan", True)), "Nothing to share yet."], {}, ["Nothing to share yet."]),
            (["Nothing to share yet."], {}, ["Nothing to share yet."]),
            # A step without the tool (the denial limit took it away, the budget is spent) answers in text.
            ([calls(send("the secret plan")), "the secret plan"], {"max_consecutive_denials": 1}, []),
            ([calls(send("the secret plan")), "the secret plan"], {"max_iter": 1}, []),
        ):
            fresh()
            model = TestModel(responses=[*responses, "must not be requested"])
            agent = Agent(name="bot", model=model, delivery="tool", guardrails=[rail], **limits)
            out, messages = await run(agent)
            assert [m["text"] for m in messages] == sent, (responses, limits)
            assert out.output.collect_text() == "".join(sent), (responses, limits)
            assert "secret" not in repr(out.output)
            if not sent:
                assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "withheld": True}

    async def test_a_draft_is_withheld_when_a_send_needs_a_person(self):
        """An approval cannot be asked for a call the model did not make. Where the gate would ask, the
        draft is not delivered: nothing goes out until a person lets it go."""

        def ask(_text: str, ctx):
            return Verdict.escalate("a person checks what goes out") if ctx.tool_name == SEND_MESSAGE_NAME else True

        fresh()
        model = TestModel(responses=["Your refund is approved.", "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool", guardrails=[guardrail(ask, stages=["tool_args"])])
        out, messages = await run(agent)
        assert messages == [] and out.output.collect_text() == "" and out.status.code == "success"
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "withheld": True}

    async def test_a_draft_a_tool_args_guardrail_rewrites_is_delivered_as_rewritten(self):
        fresh()
        rail = guardrail(lambda text: text.replace("4111", "[card]"), stages=["tool_args"], action="redact")
        model = TestModel(responses=["Card 4111 charged.", "must not be requested"])
        out, messages = await run(Agent(name="bot", model=model, delivery="tool", guardrails=[rail]))
        assert [m["text"] for m in messages] == ["Card [card] charged."]
        assert out.output.collect_text() == "Card [card] charged."  # the output is what was delivered

    async def test_a_draft_is_withheld_when_a_guardrail_s_rewrite_is_not_a_send(self):
        """A rail that answers with arguments the send tool would not take (no text, a text that is not
        one) makes a send fail. The draft it stands for is not delivered as the model wrote it."""

        def rewriting(replacement):
            return guardrail(lambda _text: Verdict.replace(replacement), stages=["tool_args"], name="rewrite")

        for replacement in ({}, {"text": None}, {"text": 5}, {"message": "clean"}, {"text": "  "}):
            fresh()
            model = TestModel(responses=["Card 4111 charged.", "must not be requested"])
            agent = Agent(name="bot", model=model, delivery="tool", guardrails=[rewriting(replacement)])
            out, messages = await run(agent)
            assert messages == [] and out.output.collect_text() == "", replacement
            assert out.metadata["delivery"] == {"mode": "tool", "messages": 0, "withheld": True}
        # A rewrite that is one is what goes out.
        fresh()
        model = TestModel(responses=["Card 4111 charged.", "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool", guardrails=[rewriting({"text": "Card charged."})])
        out, messages = await run(agent)
        assert [m["text"] for m in messages] == ["Card charged."] and out.output.collect_text() == "Card charged."

    async def test_a_draft_a_tool_args_guardrail_rewrote_is_screened_like_a_send(self):
        """A send passes the ``tool_args`` guardrails and then, in its handler, the ``model_output`` ones. A
        draft the first rewrite is checked by the second like the send it stands for, not delivered as the
        first left it. (The ``model_output`` guardrails have seen the draft once before, as the model wrote
        it: a final answer is screened as it ends, whatever becomes of it.)"""
        renames = guardrail(
            lambda text: text.replace("invoice_123", "order_123"), stages=["tool_args"], action="redact", name="a"
        )
        formats = guardrail(
            lambda text: text.replace("order_123", "Order 123"), stages=["model_output"], action="redact", name="b"
        )
        refuses = guardrail(
            lambda text: Verdict.block("an order") if "order_123" in text else True, stages=["model_output"], name="c"
        )
        for second, sent in ((formats, ["See Order 123."]), (refuses, [])):
            for first in (calls(send("See invoice_123.", True)), "See invoice_123."):
                fresh()
                # (A send that is refused is answered to the model, which here has nothing more to say.)
                model = TestModel(responses=[first, "nothing more"])
                agent = Agent(name="bot", model=model, delivery="tool", guardrails=[renames, second], max_iter=2)
                out, messages = await run(agent)
                delivered = [m["text"] for m in messages if m["text"] != "nothing more"]
                assert delivered == sent, (second.name, first)
                assert "order_123" not in repr(out.output) and "invoice_123" not in repr(out.output)

    async def test_history_keeps_a_draft_as_it_was_delivered(self, tmp_path):
        """The gate rewrote the draft: the user read the rewrite, so the next turn's model reads it too."""
        from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider

        rail = guardrail(lambda text: text.replace("Monday", "Tuesday"), stages=["tool_args"], action="redact")
        for provider in (InMemoryTracingProvider, JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")):
            read = []

            def model(messages, read=read):
                read.append(list(messages))
                return "Delivery is Monday."

            fresh()
            set_run_context(RunContext(tracing_provider=provider))
            agent = Agent(
                name="bot",
                model=TestModel(handler=model),
                delivery="tool",
                guardrails=[rail],
                tracing_provider=provider,
            )
            out, messages = await run(agent, "when?")
            assert [m["text"] for m in messages] == ["Delivery is Tuesday."]
            set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
            await run(agent, "sure?")
            assert [m.collect_text() for m in read[-1] if m.role == "assistant"] == ["Delivery is Tuesday."]

    async def test_a_rewritten_draft_keeps_what_came_with_it(self, tmp_path):
        """The final answer may hold more than text. A rail that edits the text leaves the rest in the output."""
        from timbal.types.content import FileContent, TextContent
        from timbal.types.file import File

        path = tmp_path / "chart.txt"
        path.write_text("1,2,3")
        answer = Message(
            role="assistant",
            content=[TextContent(text="Card 4111 charged."), FileContent(file=File.validate(str(path)))],
            stop_reason="end_turn",
        )
        edit = guardrail(lambda text: text.replace("4111", "[card]"), stages=["tool_args"], action="redact")
        for rails, text in (([], "Card 4111 charged."), ([edit], "Card [card] charged.")):
            fresh()
            model = TestModel(responses=[answer, "must not be requested"])
            out, messages = await run(Agent(name="bot", model=model, delivery="tool", guardrails=rails))
            assert [m["text"] for m in messages] == [text] and out.output.collect_text() == text
            assert len(files_of(out.output.content)) == 1, text

    async def test_a_draft_is_not_screened_twice_where_nothing_changed_it(self):
        """The ``model_output`` guardrails have seen a final answer. They see it again only as another text."""
        shown = []
        watches = guardrail(lambda text: shown.append(text) or True, stages=["model_output"], name="watches")
        lets_through = guardrail(lambda _text: True, stages=["tool_args"], name="lets_through")
        renames = guardrail(
            lambda text: text.replace("Monday", "Tuesday"), stages=["tool_args"], action="redact", name="renames"
        )
        for first, expected in (
            (lets_through, ["Delivery is Monday."]),
            (renames, ["Delivery is Monday.", "Delivery is Tuesday."]),
        ):
            shown.clear()
            fresh()
            model = TestModel(responses=["Delivery is Monday.", "must not be requested"])
            await run(Agent(name="bot", model=model, delivery="tool", guardrails=[first, watches]))
            assert shown == expected

    async def test_a_rewritten_draft_leaves_the_model_call_s_own_output_alone(self):
        """History says what the user read. The model call's span and its event say what the model produced."""
        from timbal.state import get_run_context

        fresh()
        rail = guardrail(lambda text: text.replace("Monday", "Tuesday"), stages=["tool_args"], action="redact")
        model = TestModel(responses=["Delivery is Monday.", "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool", guardrails=[rail])
        events = [event async for event in agent(prompt="when?")]
        assert events[-1].output.collect_text() == "Delivery is Tuesday."
        (call,) = [span for span in get_run_context()._trace.values() if span.path == "bot.llm"]
        assert call.output.collect_text() == "Delivery is Monday."
        produced = [e.output for e in events if getattr(e, "path", None) == "bot.llm" and e.type == "OUTPUT"]
        assert [message.collect_text() for message in produced] == ["Delivery is Monday."]

    async def test_a_draft_keeps_its_shape_when_rewritten_and_nothing_of_it_when_refused(self, tmp_path):
        """Rewritten, the output is the final message with its text replaced where it was. Refused, nothing
        of the message goes out: not the file its text came with."""
        from timbal.types.content import FileContent, TextContent
        from timbal.types.file import File

        path = tmp_path / "chart.txt"
        path.write_text("1,2,3")
        answer = Message(
            role="assistant",
            content=[FileContent(file=File.validate(str(path))), TextContent(text="Card 4111 charged.")],
            stop_reason="end_turn",
        )
        edit = guardrail(lambda text: text.replace("4111", "[card]"), stages=["tool_args"], action="redact")
        refuse = guardrail(lambda _text: Verdict.block("not for users"), stages=["tool_args"], name="refuse")
        for rail, kinds, text in ((edit, ["file", "text"], "Card [card] charged."), (refuse, ["text"], "")):
            fresh()
            model = TestModel(responses=[answer, "must not be requested"])
            out, _ = await run(Agent(name="bot", model=model, delivery="tool", guardrails=[rail]))
            assert [item.type for item in out.output.content] == kinds and out.output.collect_text() == text

    async def test_the_reply_of_an_on_max_iter_hook_is_screened_with_what_it_holds(self, tmp_path):
        """The hook's reply is shown to the ``model_output`` guardrails as the message it is, so one that
        reads the message decides on its file, with or without a text next to it. Refused, nothing of the
        reply goes out. And a reply stored without its empty text block is still found in the history."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.content import FileContent, TextContent
        from timbal.types.file import File

        path = tmp_path / "report.txt"
        path.write_text("1,2,3")

        def no_files(_text: str, ctx):
            held = any(isinstance(item, FileContent) for item in ctx.payload.content)
            return Verdict.block("an attachment") if held else True

        for caption in ("Here is what I have.", ""):

            def hook(_context, caption=caption) -> Message:
                content = [TextContent(text=caption)] if caption else []
                return Message(role="assistant", content=[*content, FileContent(file=File.validate(str(path)))])

            fresh()
            model = TestModel(responses=[calls(("work", {})), "must not be requested"])
            agent = Agent(
                name="bot",
                model=model,
                tools=[Tool(name="work", handler=work)],
                delivery="tool",
                max_iter=1,
                on_max_iter=hook,
                guardrails=[guardrail(no_files, stages=["model_output"], name="no_files")],
            )
            out, messages = await run(agent)
            assert messages == [] and files_of(out.output.content) == [], caption
            assert out.output.collect_text() == ""

        # A reply whose first text block is empty is stored without it: found all the same.
        read = []

        def model_fn(messages):
            read.append(list(messages))
            return calls(("work", {})) if len(read) == 1 else "ok"

        fresh()
        set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
        agent = Agent(
            name="bot",
            model=TestModel(handler=model_fn),
            tools=[Tool(name="work", handler=work)],
            delivery="tool",
            max_iter=1,
            on_max_iter=lambda _context: Message(
                role="assistant", content=[TextContent(text=""), TextContent(text="Delivery is Monday.")]
            ),
            guardrails=[
                guardrail(lambda text: text.replace("Monday", "Tuesday"), stages=["tool_args"], action="redact")
            ],
            tracing_provider=InMemoryTracingProvider,
        )
        out, messages = await run(agent)
        assert [m["text"] for m in messages] == ["Delivery is Tuesday."]
        set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
        await run(agent, "sure?", delivery="text")
        assert [m.collect_text() for m in read[-1] if m.role == "assistant" and m.collect_text()] == [
            "Delivery is Tuesday."
        ]

    async def test_a_withheld_draft_is_not_in_the_next_turn_s_history(self, tmp_path):
        """The user got nothing of a draft the checks kept back, so the next turn's model does not read it
        as the reply it gave. The model call's own output still holds what the model wrote."""
        from timbal.state import get_run_context
        from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider

        refuse = guardrail(
            lambda _text, ctx: Verdict.block("not yet") if "Monday" in ctx.tool_args.get("text", "") else True,
            stages=["tool_args"],
            name="refuse",
        )
        cases = (
            # A draft that would have been the reply.
            ({}, lambda: ["Delivery is Monday."], []),
            # The answer of a step without the tool, after a message the turn sent.
            ({"max_iter": 1}, lambda: [calls(send("Checking."), ("work", {})), "Delivery is Monday."], ["Checking."]),
            # The answer after a denial limit took the tools away.
            (
                {"max_consecutive_denials": 1},
                lambda: [calls(send("Delivery is Monday.")), "Delivery is Monday."],
                [],
            ),
            # Both at one step: the budget is spent and the denial limit is reached. Two notices go.
            (
                {"max_iter": 1, "max_consecutive_denials": 1},
                lambda: [calls(send("Delivery is Monday.")), "Delivery is Monday."],
                [],
            ),
            # on_max_iter's own reply.
            (
                {
                    "max_iter": 1,
                    "on_max_iter": lambda _c: Message.validate({"role": "assistant", "content": "Monday."}),
                },
                lambda: [calls(("work", {}))],
                [],
            ),
        )
        for number, (options, script, sent) in enumerate(cases):
            for stored in (False, True):
                provider = InMemoryTracingProvider
                if stored:
                    provider = JsonlTracingProvider.configured(_path=tmp_path / f"traces{number}.jsonl")
                read = []
                replies = list(script())

                def model(messages, read=read, replies=replies):
                    read.append(list(messages))
                    return replies.pop(0) if replies else "ok"

                fresh()
                set_run_context(RunContext(tracing_provider=provider))
                agent = Agent(
                    name="bot",
                    model=TestModel(handler=model),
                    tools=[Tool(name="work", handler=work)],
                    delivery="tool",
                    guardrails=[refuse],
                    tracing_provider=provider,
                    **options,
                )
                out, messages = await run(agent, "when?")
                assert [m["text"] for m in messages] == sent, (number, stored)
                assert "Monday" not in repr(out.output)
                calls_made = [span for span in get_run_context()._trace.values() if span.path == "bot.llm"]
                if "on_max_iter" not in options:
                    assert "Monday" in calls_made[-1].output.collect_text()  # what the model wrote, as it wrote it
                set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
                await run(agent, "and?", delivery="text")
                # No reply of that turn says it. (A refused send stays a call, with the result that refused it.)
                replies = [c.text for m in read[-1] if m.role == "assistant" for c in m.content if c.type == "text"]
                assert not any("Monday" in reply for reply in replies), (number, stored)
                # Nor the notice that asked for that answer: this turn has its tools.
                kinds = [m.metadata.get("kind") for m in read[-1]]
                assert not {"max_iter", "max_iter_notice", "denial_limit"} & set(kinds), (number, stored, kinds)

    async def test_the_answer_of_a_step_without_the_tool_keeps_its_file_after_a_message(self, tmp_path):
        """Past the budget the model answers without the tool, and that answer is delivered, file and all,
        also after a progress message: the output is what was sent, then the answer's text and files."""
        from timbal.types.content import FileContent, TextContent, ThinkingContent
        from timbal.types.file import File

        path = tmp_path / "report.txt"
        path.write_text("1,2,3")
        refuse = guardrail(
            lambda _text, ctx: Verdict.block("not yet") if "draft" in ctx.tool_args.get("text", "") else True,
            stages=["tool_args"],
            name="refuse",
        )
        for progress in (True, False):
            for caption, rails, sent_after, files in (
                ("Here is the report.", [], ["Here is the report."], 1),
                ("", [], [], 1),
                ("The draft report.", [refuse], [], 0),  # refused: nothing of the answer, its file neither
            ):
                content = [TextContent(text=caption)] if caption else []
                thought = ThinkingContent(thinking="They want the report.")  # the model's own, never delivered
                answer = Message(
                    role="assistant",
                    content=[thought, *content, FileContent(file=File.validate(str(path)))],
                    stop_reason="end_turn",
                )
                first = calls(send("Preparing the report."), ("work", {})) if progress else calls(("work", {}))
                fresh()
                model = TestModel(responses=[first, answer, "must not be requested"])
                agent = Agent(
                    name="bot",
                    model=model,
                    tools=[Tool(name="work", handler=work)],
                    delivery="tool",
                    max_iter=1,
                    guardrails=rails,
                )
                out, messages = await run(agent)
                sent = (["Preparing the report."] if progress else []) + sent_after
                assert [m["text"] for m in messages] == sent, (progress, caption)
                assert out.output.collect_text() == "\n\n".join(sent), (progress, caption)
                assert len(files_of(out.output.content)) == files, (progress, caption)
                if progress:  # the output is built from what went out: no thinking in it
                    assert not any(isinstance(item, ThinkingContent) for item in out.output.content)

    async def test_a_rewritten_draft_keeps_its_phase(self):
        """A text block carries more than its text: the OpenAI Responses phase of an assistant message goes
        back to the provider with it. A rewrite changes the text and keeps the rest, at either stage."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.content import TextContent

        at_tool_args = guardrail(lambda text: text.replace("Monday", "Tuesday"), stages=["tool_args"], action="redact")
        at_output = guardrail(lambda text: text.replace("Monday", "Tuesday"), stages=["model_output"], action="redact")
        for delivery, rails in (("tool", [at_tool_args]), ("tool", [at_output]), ("text", [at_output])):
            read = []

            def model(messages, read=read):
                read.append(list(messages))
                content = [TextContent(text="Delivery is Monday.", phase="final_answer")]
                return Message(role="assistant", content=content, stop_reason="end_turn")

            fresh()
            set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
            agent = Agent(
                name="bot",
                model=TestModel(handler=model),
                delivery=delivery,
                guardrails=rails,
                tracing_provider=InMemoryTracingProvider,
            )
            out, _ = await run(agent, "when?")
            set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
            await run(agent, "sure?", delivery="text")
            said = [c for m in read[-1] if m.role == "assistant" for c in m.content if isinstance(c, TextContent)]
            assert [(c.text, c.phase) for c in said] == [("Delivery is Tuesday.", "final_answer")], (delivery, rails)

    def test_a_rewrite_keeps_the_phase_only_where_the_text_blocks_agree_on_one(self):
        """A rewrite makes the message's text one block. It keeps the phase all of its text blocks had; a
        message that held a preamble and an answer gets none, as before, not the preamble's for the whole."""
        from timbal.guardrails.apply import replace_message_text
        from timbal.types.content import TextContent

        def message(*phases) -> Message:
            texts = [TextContent(text=f"part {n}. ", phase=phase) for n, phase in enumerate(phases)]
            return Message(role="assistant", content=texts)

        for phases, kept in (
            (("final_answer",), "final_answer"),
            (("commentary",), "commentary"),
            (("final_answer", "final_answer"), "final_answer"),
            (("commentary", "final_answer"), None),
            ((None, "final_answer"), None),
            ((None,), None),
        ):
            original = message(*phases)
            rewritten = Agent._reworded(original, "new")
            assert [c.phase for c in original.content] == list(phases)  # a copy: the original is as it was
            assert [(c.text, c.phase) for c in rewritten.content] == [("new", kept)], phases
            replace_message_text(original, "new")
            assert [(c.text, c.phase) for c in original.content] == [("new", kept)], phases

    async def test_an_on_max_iter_reply_keeps_what_it_holds_but_its_thinking(self):
        """A hook's reply goes out whole, but for the reasoning a user never receives: with or without a
        message the turn sent before, and as a guardrail left its text."""
        from timbal.types.content import CustomContent, TextContent, ThinkingContent

        def hook(_context) -> Message:
            content = [ThinkingContent(thinking="hm"), TextContent(text="Here."), CustomContent(value={"card": 1})]
            return Message(role="assistant", content=content)

        def thinking_only(_context) -> Message:
            return Message(role="assistant", content=[ThinkingContent(thinking="hm")])

        rewrite = guardrail(lambda text: text.replace("Here", "There"), stages=["tool_args"], action="redact")
        for progress in (True, False):
            for rails, said in (([], "Here."), ([rewrite], "There.")):
                fresh()
                first = calls(send("Preparing."), ("work", {})) if progress else calls(("work", {}))
                model = TestModel(responses=[first, "must not be requested"])
                agent = Agent(
                    name="bot",
                    model=model,
                    tools=[Tool(name="work", handler=work)],
                    delivery="tool",
                    max_iter=1,
                    on_max_iter=hook,
                    guardrails=rails,
                )
                out, messages = await run(agent)
                sent = (["Preparing."] if progress else []) + [said]
                assert [m["text"] for m in messages] == sent, (progress, said)
                assert [item.type for item in out.output.content] == ["text", "custom"], (progress, said)
                assert out.output.collect_text() == "\n\n".join(sent), (progress, said)
        # A reply that is only reasoning: nothing of it goes out.
        fresh()
        model = TestModel(responses=[calls(("work", {})), "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Tool(name="work", handler=work)],
            delivery="tool",
            max_iter=1,
            on_max_iter=thinking_only,
        )
        out, messages = await run(agent)
        assert messages == [] and [(item.type, item.text) for item in out.output.content] == [("text", "")]

    async def test_the_reply_of_on_max_iter_passes_the_gate_on_sends_too(self):
        def gate(_text: str, ctx):
            return Verdict.block("not for users") if "budget" in ctx.tool_args.get("text", "") else True

        for first, sent in ((calls(send("On it."), ("work", {})), ["On it."]), (calls(("work", {})), [])):
            fresh()
            model = TestModel(responses=[first, "must not be requested"])
            agent = Agent(
                name="bot",
                model=model,
                tools=[Tool(name="work", handler=work)],
                delivery="tool",
                max_iter=1,
                on_max_iter="stop",  # its reply names the budget
                guardrails=[guardrail(gate, stages=["tool_args"], name="gate")],
            )
            out, messages = await run(agent)
            assert [m["text"] for m in messages] == sent and out.output.collect_text() == "".join(sent)

    async def test_the_reply_of_on_max_iter_is_screened_and_kept_as_delivered(self):
        """No guardrail has seen ``on_max_iter``'s own reply. Delivered, it gets what a send gets: the
        ``model_output`` guardrails too, and the history holds it as the user read it."""
        from timbal.state.tracing.providers import InMemoryTracingProvider

        softens = guardrail(
            lambda text: text.replace("exhausted", "used up"), stages=["model_output"], action="redact", name="softens"
        )
        refuses = guardrail(
            lambda text: Verdict.block("internal") if "budget" in text else True,
            stages=["model_output"],
            name="refuses",
        )
        for rail, said in ((softens, "used up"), (refuses, None)):
            read = []

            def model(messages, read=read):
                read.append(list(messages))
                return calls(("work", {})) if len(read) == 1 else "ok"

            fresh()
            set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
            agent = Agent(
                name="bot",
                model=TestModel(handler=model),
                tools=[Tool(name="work", handler=work)],
                delivery="tool",
                max_iter=1,
                on_max_iter="stop",
                guardrails=[rail],
                tracing_provider=InMemoryTracingProvider,
            )
            out, messages = await run(agent)
            assert out.status.reason == "max_iter" and "exhausted" not in repr(out.output)
            if said is None:
                assert messages == [] and out.output.collect_text() == ""
                continue
            assert len(messages) == 1 and said in messages[0]["text"]
            set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
            await run(agent, "and then?", delivery="text")
            assert [m.collect_text() for m in read[-1] if m.role == "assistant" and m.collect_text()] == [
                messages[0]["text"]
            ]

    async def test_the_receipt_says_what_a_person_changed_when_approving(self):
        """A person approving a send may edit it. What went out is then not what the model wrote, and the
        receipt says so, as after a guardrail's rewrite."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.events import ApprovalEvent

        def ask(_text: str, ctx):
            return Verdict.escalate("a person checks what goes out") if ctx.tool_name == SEND_MESSAGE_NAME else True

        fresh()
        set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
        model = TestModel(
            responses=[calls(send("Your refund of 500 is approved.", end_turn=True)), "must not be requested"]
        )
        agent = Agent(
            name="bot",
            model=model,
            delivery="tool",
            guardrails=[guardrail(ask, stages=["tool_args"], name="ask")],
            tracing_provider=InMemoryTracingProvider,
        )
        events = [event async for event in in_time(agent(prompt="go"))]
        approval = next(event for event in events if isinstance(event, ApprovalEvent))
        # "Don't ask again" for this tool would be "never ask again for anything the agent says".
        assert approval.grantable is False
        set_run_context(RunContext(parent_id=events[-1].run_id, tracing_provider=InMemoryTracingProvider))
        edit = {"approved": True, "override_input": {"text": "Your refund of 50 is approved."}}
        out, messages = await run(agent, resume={approval.approval_id: edit})
        assert [m["text"] for m in messages] == ["Your refund of 50 is approved."]
        assert results(out) == [
            "Sent, after a guardrail changed it. The user received: Your refund of 50 is approved. "
            "The turn ends after this step."
        ]

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


async def in_time(stream, seconds: float = 60):
    """``stream``'s events, each awaited with a deadline: a batch that waits on itself fails here."""
    iterator = stream.__aiter__()
    while True:
        try:
            yield await asyncio.wait_for(iterator.__anext__(), seconds)
        except StopAsyncIteration:
            return


def message_events(events) -> list[tuple[str, str]]:
    """The ``message`` events of a stream in the order they arrived: (text, the call that sent it)."""
    return [
        (e.item.data["text"], e.call_id)
        for e in events
        if isinstance(e, DeltaEvent)
        and e.item.type == "custom"
        and isinstance(e.item.data, dict)
        and e.item.data.get("kind") == "message"
    ]


def assert_within_their_calls(events) -> None:
    """Every ``message`` event arrives after its own call's START and before that call's OUTPUT."""
    position = {id(e): n for n, e in enumerate(events)}
    for e in events:
        data = getattr(getattr(e, "item", None), "data", None)
        if not (isinstance(data, dict) and data.get("kind") == "message"):
            continue
        own = [x for x in events if x.call_id == e.call_id]
        start = next(x for x in own if x.type == "START")
        output = next(x for x in own if x.type == "OUTPUT")
        assert position[id(start)] < position[id(e)] < position[id(output)], data


class TestOrderInABatch:
    """Several sends in one step go out in the order the model wrote them, whatever screening each one takes."""

    def _slow(self, slow: str, others: int):
        """An output guardrail that takes longer over ``slow`` than over the batch's ``others`` messages:
        it lets that one through only once every other one has been screened."""
        screened = []
        all_others = asyncio.Event()

        async def rail(text: str) -> bool:
            if text == slow:
                await asyncio.wait_for(all_others.wait(), 30)
            else:
                screened.append(text)
                if len(screened) >= others:
                    all_others.set()
            return True

        return guardrail(rail, stages=["model_output"], name="slow")

    async def test_a_slow_guardrail_does_not_move_a_later_message_ahead(self):
        fresh()
        model = TestModel(responses=[calls(send("First."), send("Last.", end_turn=True)), "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool", guardrails=[self._slow("First.", others=1)])
        events = [event async for event in in_time(agent(prompt="go"))]
        out = events[-1]
        assert out.status.code == "success", out.error
        messages = [e.item.data for e in events if isinstance(e, DeltaEvent) and e.item.type == "custom"]
        assert [(m["text"], m["index"], m["end_turn"]) for m in messages] == [("First.", 0, False), ("Last.", 1, True)]
        assert out.output.collect_text() == "First.\n\nLast."
        assert model.call_count == 1
        assert_within_their_calls(events)

    async def test_a_slow_trace_save_does_not_move_a_later_message_ahead_either(self, monkeypatch):
        """A call's emitted event reaches the stream once the call's span has been stored. The next send
        must not go out while the first one's message waits behind that save."""
        from timbal.core.agent import _SendOrder
        from timbal.state.tracing.providers import InMemoryTracingProvider

        held, second_at_its_turn = [], asyncio.Event()
        original = _SendOrder.turn

        async def turn(self, call_id):
            if call_id == "c1-send_message":
                second_at_its_turn.set()  # screened, and asking to publish
            await original(self, call_id)

        monkeypatch.setattr(_SendOrder, "turn", turn)

        class HoldsTheFirstSave(InMemoryTracingProvider):
            @classmethod
            async def _store(cls, run_context):
                sent = run_context.root_span().metadata.get("messages_sent") or ()
                if not held and "First." in sent:  # the save that follows the first send's handler
                    held.append(1)
                    await asyncio.wait_for(second_at_its_turn.wait(), 30)
                await super()._store(run_context)

        fresh()
        model = TestModel(responses=[calls(send("First."), send("Last.", end_turn=True)), "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool", tracing_provider=HoldsTheFirstSave)
        events = [event async for event in in_time(agent(prompt="go"))]
        assert held == [1]
        assert [text for text, _ in message_events(events)] == ["First.", "Last."]
        assert_within_their_calls(events)

    async def test_three_messages_with_the_middle_one_slow(self):
        fresh()
        batch = calls(send("One."), send("Two."), send("Three.", end_turn=True))
        model = TestModel(responses=[batch, "must not be requested"])
        agent = Agent(name="bot", model=model, delivery="tool", guardrails=[self._slow("Two.", others=2)])
        out, messages = await asyncio.wait_for(run(agent), 60)
        assert [m["text"] for m in messages] == ["One.", "Two.", "Three."]
        assert out.output.collect_text() == "One.\n\nTwo.\n\nThree."

    async def test_a_message_that_is_not_sent_does_not_hold_the_ones_after_it(self):
        """Refused before its handler runs (no text), by a guardrail, or empty: the next ones still go out."""
        no_secrets = guardrail(
            lambda text: Verdict.block("secret") if "42" in text else True, stages=["model_output"], name="no_secrets"
        )
        for first in ((SEND_MESSAGE_NAME, {"end_turn": False}), send("The code is 42."), send("   ")):
            fresh()
            model = TestModel(responses=[calls(first, send("Second."), send("Third.", end_turn=True)), "not requested"])
            agent = Agent(name="bot", model=model, delivery="tool", guardrails=[no_secrets])
            out, messages = await asyncio.wait_for(run(agent), 60)
            assert out.status.code == "success", out.error
            assert [(m["text"], m["index"]) for m in messages] == [("Second.", 0), ("Third.", 1)]

    async def test_in_dispatch_mode(self):
        fresh()
        batch = calls(
            ("call_tool", {"name": SEND_MESSAGE_NAME, "arguments": {"text": "First."}}),
            send("Last.", end_turn=True),
        )
        weather = Tool(
            name="get_weather", description="Weather.", handler=lambda city: f"sunny in {city}", defer_loading=True
        )
        agent = Agent(
            name="bot",
            model=TestModel(responses=[batch, "must not be requested"]),
            tools=[weather],
            delivery="tool",
            tool_search="dispatch",
            guardrails=[self._slow("First.", others=1)],
        )
        out, messages = await asyncio.wait_for(run(agent), 60)
        assert [m["text"] for m in messages] == ["First.", "Last."]


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

    async def test_the_receipt_says_what_a_tool_args_guardrail_changed(self):
        """The agent's ``tool_args`` guardrails rewrite the call before the tool runs. What went out is then
        not what the model wrote either, and the receipt says so."""
        fresh()
        rail = guardrail(lambda text: text.replace("4111", "[card]"), stages=["tool_args"], action="redact")
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


def files_of(content) -> list:
    from timbal.types.content import FileContent

    return [c for c in content if isinstance(c, FileContent)]


def files_in(history) -> list[str]:
    """Where each file of a history is: the role of its message, and whether a tool result holds it."""
    from timbal.types.content import ToolResultContent

    found = []
    for message in history:
        found += [message.role for _ in files_of(message.content)]
        for c in message.content:
            if isinstance(c, ToolResultContent):
                found += [f"{message.role} result" for _ in files_of(c.content)]
    return found


async def chart_then_a_turn(folder, command: str, rails: list, stored: bool):
    """Run a slash command that answers with a caption and a file (``delivery="tool"``), then one more
    turn. Returns the command's output, the messages it sent, and the history the next turn's model read."""
    from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider
    from timbal.types.content import FileContent, TextContent
    from timbal.types.file import File

    folder.mkdir()
    path = folder / "chart.txt"
    path.write_text("1,2,3")

    def chart(kind: str = "") -> Message:
        caption = {"plain": "Here.", "secret": "The draft."}.get(kind)  # (the prompt does not hold the caption)
        content = [TextContent(text=caption)] if caption else []
        return Message(role="assistant", content=[*content, FileContent(file=File.validate(str(path)))])

    provider = JsonlTracingProvider.configured(_path=folder / "traces.jsonl") if stored else InMemoryTracingProvider
    read = []

    def model(messages):
        read.append(list(messages))
        return "ok"

    fresh()
    set_run_context(RunContext(tracing_provider=provider))
    agent = Agent(
        name="bot",
        model=TestModel(handler=model),
        tools=[Tool(handler=chart, command="chart")],
        delivery="tool",
        guardrails=rails,
        tracing_provider=provider,
    )
    out, messages = await run(agent, command)
    set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
    await run(agent, "what is in it?", delivery="text")
    (history,) = read
    return out, messages, history


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

    async def test_the_reply_of_an_on_max_iter_hook_keeps_its_file_and_what_was_sent(self, tmp_path):
        """The turn's output is all it told the user and what the hook's reply holds besides text: the file
        is not lost to the text of the messages, nor the messages to a reply that is only a file."""
        from timbal.types.content import FileContent, TextContent
        from timbal.types.file import File

        path = tmp_path / "report.txt"
        path.write_text("1,2,3")
        for before, caption in (
            ("Preparing the report.", "Here is what I have."),
            ("Preparing the report.", ""),
            (None, "Here is what I have."),
            (None, ""),
        ):

            def hook(_context, caption=caption) -> Message:
                content = [TextContent(text=caption)] if caption else []
                return Message(role="assistant", content=[*content, FileContent(file=File.validate(str(path)))])

            fresh()
            first = calls(send(before), ("work", {})) if before else calls(("work", {}))
            model = TestModel(responses=[first, "must not be requested"])
            agent = Agent(
                name="bot",
                model=model,
                tools=[Tool(name="work", handler=work)],
                delivery="tool",
                max_iter=1,
                on_max_iter=hook,
            )
            out, messages = await run(agent)
            sent = [text for text in (before, caption) if text]
            assert model.call_count == 1 and out.status.reason == "max_iter"
            assert [m["text"] for m in messages] == sent, (before, caption)
            assert out.output.collect_text() == "\n\n".join(sent), (before, caption)
            assert len(files_of(out.output.content)) == 1, (before, caption)

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
        assert out.output.collect_text() == "pong"  # the turn's output is what was delivered
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 1, "fallback": True}

    async def test_a_slash_command_whose_output_is_not_a_string(self):
        """What the user reads is the tool result's text, as a model would read it: never the Python repr of
        a message."""

        def balance() -> dict:
            return {"balance": 12}

        helper = Agent(name="helper", model=TestModel(responses=["helper says hi"]), command="helper")
        for command, tool, text in (
            ("/balance", Tool(handler=balance, command="balance"), "{'balance': 12}"),
            ("/helper hello", helper, "helper says hi"),
        ):
            fresh()
            model = TestModel(responses=["must not be requested"])
            agent = Agent(name="bot", model=model, tools=[tool], delivery="tool")
            out, messages = await run(agent, command)
            assert [m["text"] for m in messages] == [text]  # once: not the helper's own model call too

    async def test_a_slash_command_s_result_is_delivered_as_its_guardrails_left_it(self):
        """The command's result is the turn's answer. What goes to the user is the result the tool-result
        guardrails screened, not the tool's raw output."""

        def status() -> str:
            return "status: internal draft"

        def rewrite(text: str) -> str:
            return text.replace("internal draft", "public summary")

        on_agent = guardrail(rewrite, stages=["tool_result"], action="redact", name="rewrite")
        on_tool = guardrail(rewrite, stages=["tool_result"], action="redact", name="rewrite")
        for agent_rails, tool_rails in (([on_agent], []), ([], [on_tool])):
            fresh()
            tool = Tool(handler=status, command="status", guardrails=tool_rails)
            model = TestModel(responses=["must not be requested"])
            agent = Agent(name="bot", model=model, tools=[tool], delivery="tool", guardrails=agent_rails)
            out, messages = await run(agent, "/status")
            assert model.call_count == 0
            assert [m["text"] for m in messages] == ["status: public summary"]
            assert out.output.collect_text() == "status: public summary"  # and the run's own output

    async def test_a_slash_command_s_result_a_guardrail_blocks_is_not_delivered(self):
        def status() -> str:
            return "the key is hunter2"

        fresh()
        no_keys = guardrail(
            lambda text: Verdict.block("a credential") if "hunter2" in text else True,
            stages=["tool_result"],
            name="no_keys",
            blocked_message="That result cannot be shown.",
        )
        model = TestModel(responses=["must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Tool(handler=status, command="status")],
            delivery="tool",
            guardrails=[no_keys],
        )
        out, messages = await run(agent, "/status")
        # The user is told it was withheld, in the rail's words for a user: not the result, not the rail's reason.
        assert [m["text"] for m in messages] == ["That result cannot be shown."]
        assert out.output.collect_text() == "That result cannot be shown."

    async def test_a_slash_command_s_result_a_guardrail_leaves_nothing_of(self):
        """Nothing is left to deliver, and the run's output is not the raw result either."""

        def status() -> str:
            return "status: internal draft"

        fresh()
        wipe = guardrail(lambda _text: "", stages=["tool_result"], action="redact", name="wipe")
        model = TestModel(responses=["must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Tool(handler=status, command="status")], delivery="tool", guardrails=[wipe]
        )
        out, messages = await run(agent, "/status")
        assert messages == []
        assert out.output.collect_text() == ""
        assert out.metadata["delivery"] == {"mode": "tool", "messages": 0}

    async def test_a_slash_command_s_history_is_what_the_user_was_told(self, tmp_path):
        """The next turn's model reads the command's answer as the user got it: not what a guardrail took out
        of it, whichever store the history comes back from."""
        from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider

        def status() -> str:
            return "status: internal draft"

        rewrite = guardrail(
            lambda text: text.replace("internal draft", "public summary"),
            stages=["tool_result"],
            action="redact",
            name="rewrite",
        )
        block = guardrail(
            lambda text: Verdict.block("a draft") if "internal draft" in text else True,
            stages=["tool_result"],
            name="no_drafts",
            blocked_message="That result cannot be shown.",
        )
        wipe = guardrail(lambda _text: "", stages=["tool_result"], action="redact", name="wipe")
        for number, (rail, told) in enumerate(
            ((rewrite, "status: public summary"), (block, "That result cannot be shown."), (wipe, None))
        ):
            stored = JsonlTracingProvider.configured(_path=tmp_path / f"traces{number}.jsonl")
            for provider in (InMemoryTracingProvider, stored):
                read = []

                def handler(messages, read=read):
                    read.append(list(messages))
                    return "must not be requested" if not read[1:] else "ok"

                fresh()
                set_run_context(RunContext(tracing_provider=provider))
                agent = Agent(
                    name="bot",
                    model=TestModel(handler=handler),
                    tools=[Tool(handler=status, command="status")],
                    delivery="tool",
                    guardrails=[rail],
                    tracing_provider=provider,
                )
                out, messages = await run(agent, "/status")
                assert not read and [m["text"] for m in messages] == ([told] if told else [])
                set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
                await run(agent, "and then?", delivery="text")
                (history,) = read
                assert "internal draft" not in repr(history)
                if provider is InMemoryTracingProvider:
                    # Kept as the assistant's reply: exactly what was delivered. With nothing delivered there
                    # is no reply: the command is its call and what was left of its result.
                    said = [c.text for m in history if m.role == "assistant" for c in m.content if c.type == "text"]
                    assert said == ([told] if told else [])
                    assert [m.role for m in history] == (
                        ["user", "assistant", "user"] if told else ["user", "assistant", "tool", "user"]
                    )

    async def test_a_slash_command_s_history_has_no_output_of_the_calls_inside_it(self):
        """A command that is an agent runs calls of its own. Their raw outputs are not replies of this agent."""
        from timbal.state.tracing.providers import InMemoryTracingProvider

        def lookup() -> str:
            return "a raw record"

        read = []

        def handler(messages):
            read.append(list(messages))
            return "ok"

        fresh()
        set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
        helper = Agent(
            name="helper",
            model=TestModel(responses=[calls(("lookup", {})), "helper says hi"]),
            tools=[lookup],
            command="helper",
        )
        agent = Agent(
            name="bot",
            model=TestModel(handler=handler),
            tools=[helper],
            delivery="tool",
            tracing_provider=InMemoryTracingProvider,
        )
        out, messages = await run(agent, "/helper hello")
        assert [m["text"] for m in messages] == ["helper says hi"]
        set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
        await run(agent, "and then?", delivery="text")
        (history,) = read
        assert [(m.role, m.collect_text()) for m in history] == [
            ("user", "/helper hello"),
            ("assistant", "helper says hi"),
            ("user", "and then?"),
        ]

    async def test_a_slash_command_s_list_keeps_its_items_apart(self):
        """A list a command returns is several values. Glued, [12, 34] reads as another number: each item
        is a line of the message, of the output, and of what the result's guardrails check and rewrite.
        What a generator yields is one stream, and a message's text blocks one text: joined as they are."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.content import TextContent

        def counts() -> list:
            return [12, 34]

        def mixed() -> list:
            return [[1, 2], {"a": 1}, "x"]

        def people() -> list:
            return ["ann@x.io", "bob@y.io"]

        def table() -> list:  # rows that end their own line: a blank line between them breaks the table
            return ["| item | qty |\n", "|---|---|\n", "| bolts | 12 |\n"]

        def config() -> list:  # lines that begin with their indentation are lines all the same
            return "server:\n  host: db1\n\n  port: 5432".splitlines()

        def note() -> Message:
            return Message(role="assistant", content=[TextContent(text="Hello, "), TextContent(text="world.")])

        async def report():  # a stream whose first item is not text is collected as a list of its chunks
            yield {"progress": 0.5}
            for chunk in ("The bu", "ild pass", "ed", ".", " See https://x.io/", "runs/", "42"):
                yield chunk

        def totals():  # values a generator yields are values all the same
            yield 12
            yield 34

        def last_step(first: str) -> list:
            return [12, 34] if first == "worked" else [first]

        from timbal import Workflow
        from timbal.state import get_run_context

        steps = (  # a workflow's handler is a generator too: its output is its last step's, not chunks
            Workflow(name="steps", command="steps")
            .step(work)
            .step(last_step, first=lambda: get_run_context().step_span("work").output)
        )

        checked = []
        spy = guardrail(lambda text: checked.append(text) or True, stages=["tool_result"], name="spy")
        redact = guardrail(
            lambda text: text.replace("ann@x.io", "[email]"), stages=["tool_result"], action="redact", name="redact"
        )
        for handler, rails, text in (
            (counts, [], "12\n34"),
            (mixed, [], "[1, 2]\n{'a': 1}\nx"),
            (people, [spy], "ann@x.io\nbob@y.io"),
            (people, [redact], "[email]\nbob@y.io"),
            (table, [], "| item | qty |\n|---|---|\n| bolts | 12 |"),
            (config, [], "server:\n  host: db1\n\n  port: 5432"),
            (note, [], "Hello, world."),
            (report, [spy], "{'progress': 0.5}\nThe build passed. See https://x.io/runs/42"),
            (totals, [], "12\n34"),
            (steps, [], "12\n34"),
        ):
            read = []

            def model(messages, read=read):
                read.append(list(messages))
                return "ok"

            fresh()
            set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
            command = getattr(handler, "__name__", None) or handler.name
            agent = Agent(
                name="bot",
                model=TestModel(handler=model),
                tools=[handler if handler is steps else Tool(handler=handler, command=command)],
                delivery="tool",
                guardrails=rails,
                tracing_provider=InMemoryTracingProvider,
            )
            out, messages = await run(agent, f"/{command}")
            assert [m["text"] for m in messages] == [text]
            assert out.output.collect_text() == text
            set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
            await run(agent, "and then?", delivery="text")
            assert [m.collect_text() for m in read[0] if m.role == "assistant"] == [text]
        # What the guardrail read is what the user would.
        assert checked == ["ann@x.io\nbob@y.io", "{'progress': 0.5}\nThe build passed. See https://x.io/runs/42"]

    def test_a_list_s_text_items_as_one_text(self):
        from timbal.core.agent import _one_text
        from timbal.types.content import TextContent

        def text(values: list, chunks: bool = False) -> str:
            (one,) = _one_text(values, [TextContent(text=str(value)) for value in values], chunks)
            return one.text

        assert text([12, 34]) == "12\n34" and text(["only"]) == "only" and _one_text([], [], False) == []
        assert text(["a\n", "b\n", "c"]) == "a\nb\nc"  # no second break after an item that ends its line
        assert text(["a", "", "b"]) == "a\n\nb" and text(["a", "  b", "\tc"]) == "a\n  b\n\tc"
        assert text(["Total: ", "12"]) == "Total: \n12"
        # What a generator yielded: strings that follow one another are one stream; a value has its line.
        assert text(["Hel", "lo", ",", " world"], chunks=True) == "Hello, world"
        assert text([{"n": 1}, "Hel", "lo", 7, 8, "!", "?"], chunks=True) == "{'n': 1}\nHello\n7\n8\n!?"
        assert text([12, 34], chunks=True) == "12\n34"

    async def test_a_slash_command_with_no_result_leaves_no_reply_in_history(self):
        """A command that timed out or was denied told the user nothing. What it had produced until then,
        which no guardrail screened, is not the assistant's reply on the next turn: the history has the
        call and what became of it, as a store that keeps calls has."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.content import ToolResultContent, ToolUseContent
        from timbal.types.events import ApprovalEvent

        async def slow():
            yield "internal draft, "
            await asyncio.Event().wait()  # never: the timeout ends it

        def charge() -> str:
            return "charged"

        for tool, prompt, notice in (
            (Tool(handler=slow, command="slow", timeout=0.2), "/slow", "RunTimeout"),
            (Tool(handler=charge, command="charge", requires_approval=True), "/charge", "[Approval denied]"),
        ):
            read = []

            def model(messages, read=read):
                read.append(list(messages))
                return "ok"

            fresh()
            set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
            wipe = guardrail(lambda _text: "", stages=["tool_result"], action="redact", name="wipe")
            agent = Agent(
                name="bot",
                model=TestModel(handler=model),
                tools=[tool],
                delivery="tool",
                guardrails=[wipe],
                tracing_provider=InMemoryTracingProvider,
            )
            events = [e async for e in in_time(agent(prompt=prompt))]
            out = events[-1]
            if prompt == "/charge":
                assert out.status.reason == "approval_required"
                approval = next(e for e in events if isinstance(e, ApprovalEvent))
                set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
                out, messages = await run(agent, prompt, resume={approval.approval_id: False})
                assert messages == [] and out.status.reason == "approval_denied"
            else:
                assert out.status.code == "timeout", out.status
            set_run_context(RunContext(parent_id=out.run_id, tracing_provider=InMemoryTracingProvider))
            await run(agent, "and then?", delivery="text")
            (history,) = read
            assert [m.role for m in history] == ["user", "assistant", "tool", "user"]
            (call,), (result,) = history[1].content, history[2].content
            assert isinstance(call, ToolUseContent) and isinstance(result, ToolResultContent)
            assert call.id == result.id and result.delivered is None
            assert notice in repr(result) and "internal draft" not in repr(history)

    async def test_a_slash_command_s_answer_survives_compaction_like_a_message_sent(self, tmp_path, monkeypatch):
        """With a store that keeps the history as calls, the answer is a tool result. It is what the user
        was told: dropping old tool results must not take it."""
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.state.tracing.providers import JsonlTracingProvider
        from timbal.types.content import ToolResultContent

        def status() -> str:
            return "Order 4411 ships on Friday."

        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        read = []

        def model(messages):
            read.append(list(messages))
            return "an answer long enough to fill the small window " * 6

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000_000)
        fresh()
        set_run_context(RunContext(tracing_provider=provider))
        agent = Agent(
            name="bot",
            model=TestModel(handler=model),
            tools=[Tool(handler=status, command="status"), work],
            delivery="tool",
            memory_compaction=compact_tool_results(),
            memory_compaction_ratio=0.5,
            tracing_provider=provider,
        )
        out, messages = await run(agent, "/status")
        assert [m["text"] for m in messages] == ["Order 4411 ships on Friday."]
        set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
        out, _ = await run(agent, "thanks", delivery="text")
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)
        set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
        out, _ = await run(agent, "when does it ship again?", delivery="text")
        assert out.metadata["compaction"]["triggered"] is True
        told = [c for m in read[-1] for c in m.content if isinstance(c, ToolResultContent)]
        assert [c.delivered for c in told] == ["Order 4411 ships on Friday."]
        assert "Order 4411 ships on Friday." in repr(read[-1])

    async def test_a_slash_command_s_attachment_is_in_the_next_turn_s_history(self, tmp_path):
        """A command's file is part of its answer: the next turn's model can look at what the user received,
        whichever store the history comes back from."""
        rewrite = guardrail(
            lambda text: text.replace("draft", "summary"), stages=["tool_result"], action="redact", name="rewrite"
        )
        for number, (command, rails, text) in enumerate(
            (
                ("/chart plain", [], "Here."),
                ("/chart", [], None),
                ("/chart secret", [rewrite], "The summary."),
                ("/chart", [rewrite], None),  # a rail that rewrites text has none to rewrite: the file stays as it is
            )
        ):
            for stored in (False, True):
                out, messages, history = await chart_then_a_turn(tmp_path / f"{number}{stored}", command, rails, stored)
                assert [m["text"] for m in messages] == ([text] if text else [])
                assert len(files_of(out.output.content)) == 1
                # In a tool result: an assistant message cannot carry a file to the providers.
                assert files_in(history) == ["tool result"], (command, stored)
                (result,) = [c for m in history if m.role == "tool" for c in m.content]
                assert len(result.content) == (2 if text else 1)  # no empty text put next to the file
                assert "draft" not in repr(history)
                if not stored and text:
                    assert [m.collect_text() for m in history if m.role == "assistant" and m.collect_text()] == [text]

    async def test_a_blocked_slash_command_result_leaves_no_attachment(self, tmp_path):
        """The user got the refusal and nothing of what the result held. Neither does the next turn's model:
        not the text, and not the file that came with it."""
        block = guardrail(
            lambda text: Verdict.block("not for users") if "draft" in text else True,
            stages=["tool_result"],
            name="private",
            blocked_message="That result cannot be shown.",
        )
        for stored in (False, True):
            out, messages, history = await chart_then_a_turn(tmp_path / str(stored), "/chart secret", [block], stored)
            assert [m["text"] for m in messages] == ["That result cannot be shown."]
            assert files_of(out.output.content) == [] and files_in(history) == [], stored
            assert "draft" not in repr(history)

    async def test_a_guardrail_reads_a_slash_command_s_file_without_a_caption(self, tmp_path):
        """A rail that looks at the result itself (``ctx.payload``) decides on its attachments. A result
        that is only a file has no text to check, and is still the rail's to refuse."""
        from timbal.types.content import FileContent

        def no_files(_text: str, ctx):
            held = any(isinstance(item, FileContent) for item in ctx.payload.content)
            return Verdict.block("an attachment") if held else True

        for command in ("/chart plain", "/chart"):
            for stored in (False, True):
                rail = guardrail(
                    no_files, stages=["tool_result"], name="no_files", blocked_message="That file cannot be shown."
                )
                out, messages, history = await chart_then_a_turn(
                    tmp_path / f"{command[-1]}{stored}", command, [rail], stored
                )
                assert [m["text"] for m in messages] == ["That file cannot be shown."], (command, stored)
                assert files_of(out.output.content) == [] and files_in(history) == [], (command, stored)

    async def test_a_slash_command_s_file_survives_compaction_without_a_caption(self, tmp_path, monkeypatch):
        """The user received the file: dropping old tool results must not take it, with or without a text
        next to it."""
        from timbal.core.memory_compaction import compact_tool_results
        from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider
        from timbal.types.content import FileContent, TextContent
        from timbal.types.file import File

        path = tmp_path / "chart.txt"
        path.write_text("1,2,3")

        def chart(caption: str = "") -> Message:
            content = [TextContent(text=caption)] if caption else []
            return Message(role="assistant", content=[*content, FileContent(file=File.validate(str(path)))])

        number = 0
        for command in ("/chart Here.", "/chart"):
            for stored in (False, True):
                number += 1
                provider = InMemoryTracingProvider
                if stored:
                    provider = JsonlTracingProvider.configured(_path=tmp_path / f"traces{number}.jsonl")
                read = []

                def model(messages, read=read):
                    read.append(list(messages))
                    return "an answer long enough to fill the small window " * 6

                monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000_000)
                fresh()
                set_run_context(RunContext(tracing_provider=provider))
                agent = Agent(
                    name="bot",
                    model=TestModel(handler=model),
                    tools=[Tool(handler=chart, command="chart"), work],
                    delivery="tool",
                    memory_compaction=compact_tool_results(),
                    memory_compaction_ratio=0.5,
                    tracing_provider=provider,
                )
                out, _ = await run(agent, command)
                assert len(files_of(out.output.content)) == 1
                set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
                out, _ = await run(agent, "thanks", delivery="text")
                monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)
                set_run_context(RunContext(parent_id=out.run_id, tracing_provider=provider))
                out, _ = await run(agent, "what was in the file?", delivery="text")
                assert out.metadata["compaction"]["triggered"] is True
                assert files_in(read[-1]) == ["tool result"], (command, stored)

    async def test_a_slash_command_resumed_after_an_approval_is_delivered_like_any(self, tmp_path):
        """A command that paused resumes as a command, whichever store its history comes back from: its answer
        is delivered, screened and kept as for one that never paused, and no model is called for it."""
        from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider
        from timbal.types.content import FileContent, TextContent
        from timbal.types.events import ApprovalEvent
        from timbal.types.file import File

        path = tmp_path / "chart.txt"
        path.write_text("1,2,3")

        def chart(caption: str = "") -> Message:
            content = [TextContent(text=caption)] if caption else []
            return Message(role="assistant", content=[*content, FileContent(file=File.validate(str(path)))])

        def no_files(_text: str, ctx):
            held = any(isinstance(item, FileContent) for item in ctx.payload.content)
            return Verdict.block("an attachment") if held else True

        number = 0
        for command, refuse, sent, files in (
            ("/chart Here.", False, ["Here."], 1),
            ("/chart", False, [], 1),
            ("/chart Here.", True, ["That file cannot be shown."], 0),
            ("/chart", True, ["That file cannot be shown."], 0),
        ):
            for stored in (False, True):
                number += 1
                provider = InMemoryTracingProvider
                if stored:
                    provider = JsonlTracingProvider.configured(_path=tmp_path / f"traces{number}.jsonl")
                rails = []
                if refuse:
                    rails = [
                        guardrail(
                            no_files,
                            stages=["tool_result"],
                            name="no_files",
                            blocked_message="That file cannot be shown.",
                        )
                    ]
                fresh()
                set_run_context(RunContext(tracing_provider=provider))
                model = TestModel(responses=["must not be requested"])
                agent = Agent(
                    name="bot",
                    model=model,
                    tools=[Tool(handler=chart, command="chart", requires_approval=True)],
                    delivery="tool",
                    guardrails=rails,
                    tracing_provider=provider,
                )
                events = [event async for event in in_time(agent(prompt=command))]
                parked = events[-1]
                assert parked.status.reason == "approval_required" and message_events(events) == []
                approval = next(event for event in events if isinstance(event, ApprovalEvent))
                set_run_context(RunContext(parent_id=parked.run_id, tracing_provider=provider))
                out, messages = await run(agent, command, resume={approval.approval_id: True})
                assert [m["text"] for m in messages] == sent, (command, refuse, stored)
                assert model.call_count == 0 and len(files_of(out.output.content)) == files, (command, refuse, stored)

    async def test_a_slash_command_that_returns_a_file(self, tmp_path):
        """A message event carries text. The result's files are in the run's output, as in text mode."""
        from timbal.types.content import FileContent, TextContent
        from timbal.types.file import File

        path = tmp_path / "chart.txt"
        path.write_text("1,2,3")

        def chart(caption: str = "") -> Message:
            content = [TextContent(text=caption)] if caption else []
            return Message(role="assistant", content=[*content, FileContent(file=File.validate(str(path)))])

        for command, texts in (("/chart", []), ("/chart Here.", ["Here."])):
            fresh()
            model = TestModel(responses=["must not be requested"])
            agent = Agent(name="bot", model=model, tools=[Tool(handler=chart, command="chart")], delivery="tool")
            out, messages = await run(agent, command)
            assert out.status.code == "success", out.error
            assert [m["text"] for m in messages] == texts
            assert out.output.collect_text() == "".join(texts)
            assert len([c for c in out.output.content if isinstance(c, FileContent)]) == 1

    async def test_a_slash_command_that_is_denied_delivers_no_notice(self):
        """A denial is not a result: what a model would be told about it is not a message for the user."""
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.types.events import ApprovalEvent

        def charge() -> str:
            return "charged"

        fresh()
        set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
        model = TestModel(responses=["must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Tool(handler=charge, command="charge", requires_approval=True)],
            delivery="tool",
            tracing_provider=InMemoryTracingProvider,
        )
        events, messages = [e async for e in agent(prompt="/charge")], []
        parked = events[-1]
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        set_run_context(RunContext(parent_id=parked.run_id, tracing_provider=InMemoryTracingProvider))
        out, messages = await run(agent, "/charge", resume={approval.approval_id: False})
        assert messages == []
        assert out.status.reason == "approval_denied" and "delivery" not in out.metadata

    async def test_a_slash_command_s_long_result_is_delivered_whole(self):
        """A result limit reduces what goes into the model's context. The user asked for the output."""
        from timbal.core import ToolResultLimit, Truncate

        report = "line " * 800

        def long() -> str:
            return report

        fresh()
        model = TestModel(responses=["must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Tool(handler=long, command="long")],
            delivery="tool",
            tool_result_limit=ToolResultLimit(threshold=100, action=Truncate(strategy="head", max_chars=50)),
        )
        out, messages = await run(agent, "/long")
        assert [m["text"] for m in messages] == [report.strip()]
        # Stored, the result is the reduction, and its mark as delivered is no second copy of the report.
        from timbal.state import get_run_context

        root = next(span for span in get_run_context()._trace.values() if span.path == "bot")
        (stored,) = root._memory_dump[-1]["content"]
        assert stored["delivered"] == report[:2_000] and report.strip() not in str(stored)


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

    async def test_a_now_message_does_not_cut_a_send_that_was_published(self):
        """A send acts: once it has recorded its message, a ``now`` message must not cancel the call before
        the message is out, or the run would say something went out that the user never got."""
        from timbal.core.chat_session import ChatSession
        from timbal.state.tracing.providers import InMemoryTracingProvider

        held, holding, release = [], asyncio.Event(), asyncio.Event()

        class HoldsTheFirstSave(InMemoryTracingProvider):
            @classmethod
            async def _store(cls, run_context):
                root = run_context.root_span()
                sent = (root.metadata.get("messages_sent") or ()) if root is not None else ()
                if not held and "S0" in sent:  # the save between the first send's handler and its event
                    held.append(1)
                    holding.set()
                    await asyncio.wait_for(release.wait(), 30)
                await super()._store(run_context)

        def handler(messages):
            said = [m.collect_text() for m in messages if m.role == "user" and not m.is_runtime()]
            if said[-1] == "start":
                return calls(send("S0"), send("S1", end_turn=True))
            return calls(send("Answering the new message.", end_turn=True))

        fresh()
        agent = Agent(name="bot", model=TestModel(handler=handler), delivery="tool", tracing_provider=HoldsTheFirstSave)
        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(holding.wait(), 30)
            session.send("a new question", mode="now")  # cuts what may be cut
            release.set()
            texts, finished = await asyncio.wait_for(finished_turn(session), 60)
        assert texts == ["S0", "S1", "Answering the new message."]
        trace = await RunContext(tracing_provider=HoldsTheFirstSave)._get_parent_trace(finished["run_id"])
        root = next(span for span in trace.values() if span.parent_call_id is None)
        assert root.metadata["messages_sent"] == texts  # what the run says it sent is what went out
