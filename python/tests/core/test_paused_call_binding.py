"""A call that paused (an approval, a question) runs, when its turn resumes, on the tool it was made for.

The agent adds tools of its own under names it picks at run time (the delivery tool, the salience tool,
``search_tools`` / ``call_tool``) or under fixed names an application's tool may already have
(``read_offloaded``). Between the pause and the resume the tools an application offers can change (a tool
set's catalog). A pending call is matched by name, so a name freed meanwhile could hand an application's
call to a tool of the agent's own, which would take the user's answer and report success. The names the
agent's own tools had when the turn paused are kept for the turn; a call whose tool is gone is answered as
such and not run.
"""

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.core.tool_set import ToolSet
from timbal.state import set_call_id, set_parent_call_id, set_run_context, suspend
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent
from timbal.types.events import ApprovalEvent, DeltaEvent, InteractionEvent, OutputEvent
from timbal.types.message import Message

# What the application's tool set offers right now (a module global: a field would be copied per instance).
CURRENT: list = []


class Catalog(ToolSet):
    """An application's tools that may change between runs (an MCP server's, say)."""

    async def resolve(self):
        return list(CURRENT)


def fresh(parent_id=None):
    set_run_context(RunContext(parent_id=parent_id, tracing_provider=InMemoryTracingProvider))
    set_call_id(None)
    set_parent_call_id(None)


def call(tool, **input):
    return Message(
        role="assistant", content=[ToolUseContent(id=f"c-{tool}", name=tool, input=input)], stop_reason="tool_use"
    )


def results(memory) -> list[str]:
    return [
        "".join(getattr(item, "text", "") for item in c.content) if isinstance(c.content, list) else str(c.content)
        for m in memory
        for c in m.content
        if isinstance(c, ToolResultContent)
    ]


async def run(agent, prompt="go", **kwargs):
    events = [event async for event in agent(prompt=prompt, **kwargs)]
    out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
    return events, out


def memory_of(out):
    """The memory and root span of the run ``out`` ended."""
    assert out.parent_call_id is None
    from timbal.state import get_run_context

    root = next(s for s in get_run_context()._trace.values() if s.parent_call_id is None)
    return root.memory, root


@pytest.fixture(autouse=True)
def reset():
    CURRENT.clear()
    yield
    set_run_context(None)


class TestAnApplicationsCallIsNotTakenByTheAgentsOwnTool:
    async def test_a_salience_tool_does_not_take_a_question_that_lost_its_tool(self):
        """The reported case: an application's ``mark_salient`` asked the user which note to annotate; while
        the turn was parked it left its tool set. The built-in took the free name, the pending call and the
        user's answer, recorded a memory mark and reported success."""

        def annotate(note: str) -> str:
            return f"annotated {note} with {suspend({'question': 'Which note?'})}"

        notes = Catalog()
        CURRENT[:] = [Tool(name="mark_salient", handler=annotate)]
        model = TestModel(responses=[call("mark_salient", note="draft"), "done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[notes], salience_tool=True, tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "input_required"
        question = next(e for e in events if isinstance(e, InteractionEvent))
        CURRENT.clear()  # gone from its tool set while the turn waited
        fresh(parked.run_id)
        _, out = await run(agent, resume={question.interaction_id: "note 3"})
        memory, root = memory_of(out)
        assert "memory" not in root.metadata  # no mark was recorded
        (result,) = results(memory)
        assert "this call was made for" in result and "mark_salient" in result

    async def test_a_send_tool_does_not_deliver_an_application_call(self):
        """An application's ``send_message`` (posting to a team channel) waited for approval; it is gone at the
        resume. The delivery tool, free to take the name, would have sent its text to the user."""
        posted = []
        channel = Catalog()
        CURRENT[:] = [
            Tool(name="send_message", handler=lambda text: posted.append(text) or "posted", requires_approval=True)
        ]
        model = TestModel(responses=[call("send_message", text="Internal: refund the customer"), "done", "x"])
        agent = Agent(
            name="bot", model=model, tools=[channel], delivery="tool", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        fresh(parked.run_id)
        events, out = await run(agent, resume={approval.approval_id: True})
        delivered = [
            e.item.data
            for e in events
            if isinstance(e, DeltaEvent)
            and e.item.type == "custom"
            and isinstance(e.item.data, dict)
            and e.item.data.get("kind") == "message"
        ]
        assert not any("refund" in m["text"] for m in delivered) and posted == []
        memory, _ = memory_of(out)
        assert any("this call was made for" in r for r in results(memory))

    async def test_a_dispatcher_does_not_take_an_application_call(self):
        """With ``tool_search="dispatch"`` the agent's ``call_tool`` was renamed (an application tool had the
        name). That tool is gone at the resume: the dispatcher, under the name now, would have run its target."""
        charged = []
        catalog = Catalog()
        CURRENT[:] = [
            Tool(
                name="call_tool",
                handler=lambda name, arguments: f"queued {name} with {arguments}",
                requires_approval=True,
            ),
            Tool(name="charge", handler=lambda amount: charged.append(amount) or "charged", defer_loading=True),
        ]
        model = TestModel(
            responses=[call("call_tool", name="charge", arguments={"amount": 500}), "done", "must not be requested"]
        )
        agent = Agent(
            name="bot", model=model, tools=[catalog], tool_search="dispatch", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT[:] = [CURRENT[1]]  # the application's call_tool is gone
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert charged == []
        memory, _ = memory_of(out)
        assert any("this call was made for" in r for r in results(memory))

    async def test_a_fixed_name_of_the_agents_does_not_take_an_application_call(self, tmp_path):
        """``read_offloaded`` has a fixed name: an application's tool of that name had it at the pause."""
        from timbal.core import LocalOffloadStore, Spill, ToolResultLimit

        catalog = Catalog()
        CURRENT[:] = [Tool(name="read_offloaded", handler=lambda handle: f"opened {handle}", requires_approval=True)]
        model = TestModel(responses=[call("read_offloaded", handle="ticket-9"), "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[catalog],
            tool_result_limit=ToolResultLimit(threshold=10_000, action=Spill(), store=LocalOffloadStore(tmp_path)),
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        (result,) = results(memory)
        assert "this call was made for" in result

    async def test_a_background_tool_of_the_agents_does_not_take_an_application_call(self):
        """The background-task tools have fixed names too: an application's ``get_background_task`` waited for
        approval, and at the resume the agent's own (the session has a task now) would have taken it."""

        async def build() -> str:
            return "built"

        catalog = Catalog()
        CURRENT[:] = [
            Tool(name="get_background_task", handler=lambda task_id: f"ticket {task_id}", requires_approval=True)
        ]
        model = TestModel(
            responses=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="c-build", name="build", input={}),
                        ToolUseContent(id="c-get", name="get_background_task", input={"task_id": "T-1"}),
                    ],
                    stop_reason="tool_use",
                ),
                "done",
                "must not be requested",
            ]
        )
        agent = Agent(
            name="bot",
            model=model,
            tools=[catalog, Tool(name="build", handler=build, background_mode="always")],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, root = memory_of(out)
        assert any("this call was made for" in r for r in results(memory))
        ((record),) = root.metadata["calls_not_resumed"]
        assert record["tool_call_id"] == "c-get" and record["tool"] == "get_background_task"
        assert record["made_for"].startswith("tool@")  # a tool set's tool, by its provider


class TestTheAgentsOwnPausedCallKeepsItsTool:
    async def test_a_mark_that_waited_for_approval_is_still_a_mark(self):
        """The other way round: the built-in's call waited for approval, and meanwhile an application's tool
        took the name. The built-in keeps its name for the turn, and its call is its own."""
        from timbal.guardrails import Verdict, guardrail

        def ask(_text, ctx):
            return Verdict.escalate("a person checks marks") if ctx.tool_name == "mark_salient" else True

        app_ran = []
        catalog = Catalog()
        model = TestModel(responses=[call("mark_salient", why="User is vegetarian."), "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[catalog],
            salience_tool=True,
            guardrails=[guardrail(ask, stages=["tool_args"], name="ask")],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT[:] = [Tool(name="mark_salient", handler=lambda why: app_ran.append(why) or "app")]
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        _, root = memory_of(out)
        assert root.metadata["memory"]["salient"] == ["User is vegetarian."] and app_ran == []

    async def test_a_send_that_waited_for_approval_is_still_a_send(self):
        """The delivery tool's send waited for a person; an application's ``send_message`` came meanwhile. The
        send goes to the user, once approved, and the system prompt still names the tool the turn had."""
        from timbal.guardrails import Verdict, guardrail

        def ask(_text, ctx):
            return Verdict.escalate("a person checks sends") if ctx.tool_name == "send_message" else True

        posted = []
        catalog = Catalog()
        model = TestModel(responses=[call("send_message", text="Your refund is on its way.", end_turn=True), "x", "x"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[catalog],
            delivery="tool",
            guardrails=[guardrail(ask, stages=["tool_args"], name="ask")],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT[:] = [Tool(name="send_message", handler=lambda text: posted.append(text) or "posted")]
        fresh(parked.run_id)
        events, out = await run(agent, resume={approval.approval_id: True})
        delivered = [
            e.item.data["text"]
            for e in events
            if isinstance(e, DeltaEvent) and e.item.type == "custom" and e.item.data.get("kind") == "message"
        ]
        assert delivered == ["Your refund is on its way."] and posted == []
        assert model.call_count == 1  # end_turn: no model call after the resumed send

    async def test_a_dispatch_that_waited_for_approval_still_runs_its_tool(self):
        """``call_tool`` ran a deferred tool that waited for approval; an application's ``call_tool`` came
        meanwhile. The dispatcher keeps the name for the turn and runs the approved call."""
        charged, app_ran = [], []
        catalog = Catalog()
        charge = Tool(
            name="charge",
            handler=lambda amount: charged.append(amount) or "charged",
            defer_loading=True,
            requires_approval=True,
        )
        CURRENT[:] = [charge]
        model = TestModel(
            responses=[call("call_tool", name="charge", arguments={"amount": 5}), "done", "must not be requested"]
        )
        agent = Agent(
            name="bot", model=model, tools=[catalog], tool_search="dispatch", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT[:] = [
            charge,
            Tool(name="call_tool", handler=lambda name, arguments: app_ran.append((name, arguments)) or "app"),
        ]
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert out.status.code == "success"
        assert charged == [5] and app_ran == []

    async def test_the_search_keeps_its_name_after_the_resume(self):
        """Client tool search: the turn paused on another call of the batch that searched. At the resume an
        application's tool has the search's name: the search keeps it for the turn (the model's next call of
        it searches), and the application's tool is not offered."""
        app_ran = []
        catalog = Catalog()
        deferred = Tool(name="refund", description="Refund an order", handler=lambda order: order, defer_loading=True)
        CURRENT[:] = [deferred, Tool(name="hold", handler=lambda: "held", requires_approval=True)]
        model = TestModel(
            responses=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="c-search", name="search_tools", input={"query": "nothing like it"}),
                        ToolUseContent(id="c-hold", name="hold", input={}),
                    ],
                    stop_reason="tool_use",
                ),
                call("search_tools", query="refund"),
                "done",
                "must not be requested",
            ]
        )
        agent = Agent(
            name="bot", model=model, tools=[catalog], tool_search="client", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.append(Tool(name="search_tools", handler=lambda query: app_ran.append(query) or "app"))
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        assert '"loaded"' in results(memory)[-1] and app_ran == []


class TestARunPausedBeforeTheRecord:
    async def test_resumes_as_before(self):
        """A run paused by an earlier version has no record of its calls' tools: its calls are replayed by
        name, as before."""
        ran = []
        model = TestModel(responses=[call("refund", order="A-1"), "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Tool(name="refund", handler=lambda order: ran.append(order) or "ok", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        from timbal.state import get_run_context

        root = next(s for s in get_run_context()._trace.values() if s.parent_call_id is None)
        assert root.metadata.pop("paused_calls")["bindings"] == {"c-refund": "tool"}
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert out.status.code == "success" and ran == ["A-1"]


async def stop_at_the_card(agent, close=True, **kwargs):
    """Read the run up to its approval or question, then stop reading (a client that shows the card and
    lets the stream go): the run stays waiting. ``close=False``: a plain ``break``, which leaves the stream to
    be closed later, by the event loop."""
    stream = agent(prompt="go", **kwargs)
    events = []
    async for event in stream:
        events.append(event)
        if isinstance(event, ApprovalEvent | InteractionEvent):
            break
    if close:
        await stream.aclose()
    return events


class TestACallerThatStopsAtTheCard:
    """The pause is the same when the caller stops reading at the card instead of reading to the end."""

    async def test_a_question_that_lost_its_tool_is_not_taken(self):
        def annotate(note: str) -> str:
            return f"annotated {note} with {suspend({'question': 'Which note?'})}"

        CURRENT[:] = [Tool(name="mark_salient", handler=annotate)]
        model = TestModel(responses=[call("mark_salient", note="draft"), "done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog()], salience_tool=True, tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events = await stop_at_the_card(agent)
        question = events[-1]
        assert isinstance(question, InteractionEvent)
        CURRENT.clear()
        fresh(question.run_id)
        _, out = await run(agent, resume={question.interaction_id: "note 3"})
        memory, root = memory_of(out)
        assert "memory" not in root.metadata
        (result,) = results(memory)
        assert "this call was made for" in result

    async def test_a_resume_right_after_a_plain_break_finds_the_record(self):
        """The stream is not closed yet when the resume reads the run (the same process, a live trace): the
        record was made as the card went out, not on the way out of the turn."""

        def annotate(note: str) -> str:
            return f"annotated {note} with {suspend({'question': 'Which note?'})}"

        CURRENT[:] = [Tool(name="mark_salient", handler=annotate)]
        model = TestModel(responses=[call("mark_salient", note="draft"), "done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog()], salience_tool=True, tracing_provider=InMemoryTracingProvider
        )
        fresh()
        question = (await stop_at_the_card(agent, close=False))[-1]
        CURRENT.clear()
        fresh(question.run_id)
        _, out = await run(agent, resume={question.interaction_id: "note 3"})
        memory, root = memory_of(out)
        assert "memory" not in root.metadata
        (result,) = results(memory)
        assert "this call was made for" in result

    async def test_an_application_send_is_not_delivered(self):
        posted = []
        CURRENT[:] = [
            Tool(name="send_message", handler=lambda text: posted.append(text) or "posted", requires_approval=True)
        ]
        model = TestModel(responses=[call("send_message", text="Internal: refund the customer"), "done", "x"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog()], delivery="tool", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        approval = (await stop_at_the_card(agent))[-1]
        CURRENT.clear()
        fresh(approval.run_id)
        events, _ = await run(agent, resume={approval.approval_id: True})
        delivered = [
            e.item.data["text"]
            for e in events
            if isinstance(e, DeltaEvent) and e.item.type == "custom" and e.item.data.get("kind") == "message"
        ]
        assert not any("refund" in text for text in delivered) and posted == []

    async def test_the_agents_own_mark_keeps_its_tool(self):
        from timbal.guardrails import Verdict, guardrail

        def ask(_text, ctx):
            return Verdict.escalate("a person checks marks") if ctx.tool_name == "mark_salient" else True

        app_ran = []
        model = TestModel(responses=[call("mark_salient", why="User is vegetarian."), "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Catalog()],
            salience_tool=True,
            guardrails=[guardrail(ask, stages=["tool_args"], name="ask")],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        approval = (await stop_at_the_card(agent))[-1]
        CURRENT[:] = [Tool(name="mark_salient", handler=lambda why: app_ran.append(why) or "app")]
        fresh(approval.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        _, root = memory_of(out)
        assert root.metadata["memory"]["salient"] == ["User is vegetarian."] and app_ran == []


class TestARefusedCall:
    async def test_a_stop_for_it_ends_the_run(self):
        """The user stopped the question (a Cancel, as ``ChatSession.interrupt()`` sends) whose tool is gone:
        the run ends cancelled, and the model is not called again (it might call tools after the stop)."""
        CURRENT[:] = [Tool(name="refund", handler=lambda order: f"refunded {order}", requires_approval=True)]
        model = TestModel(responses=[call("refund", order="A-1"), "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Catalog()], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        fresh(parked.run_id)
        from timbal.types.approval import Cancel

        _, out = await run(agent, resume={approval.approval_id: Cancel(reason="the user moved on")})
        assert out.status.code == "cancelled" and model.call_count == 1

    async def test_a_dispatch_names_the_tool_it_was_to_run(self):
        """The dispatcher is still there (other deferred tools): the refusal names the tool that is gone, not
        ``call_tool``, which the model may go on using."""
        charge = Tool(name="charge", handler=lambda amount: amount, defer_loading=True, requires_approval=True)
        refund = Tool(name="refund", handler=lambda order: order, defer_loading=True)
        CURRENT[:] = [charge, refund]
        model = TestModel(
            responses=[call("call_tool", name="charge", arguments={"amount": 5}), "done", "must not be requested"]
        )
        agent = Agent(
            name="bot", model=model, tools=[Catalog()], tool_search="dispatch", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT[:] = [refund]
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        (result,) = results(memory)
        assert "The tool 'charge' this call was made for" in result

    async def test_the_dispatchers_name_stays_the_agents_with_nothing_deferred(self):
        """At the resume nothing is deferred (no dispatcher on any step), and an application's tool has the
        dispatcher's name: it is not offered for the rest of the turn, where the history's calls of that name
        are the dispatcher's."""
        app_ran = []
        CURRENT[:] = [Tool(name="charge", handler=lambda amount: amount, defer_loading=True, requires_approval=True)]
        model = TestModel(
            responses=[
                call("call_tool", name="charge", arguments={"amount": 5}),
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c-again", name="call_tool", input={"name": "charge", "arguments": {}})],
                    stop_reason="tool_use",
                ),
                "done",
                "must not be requested",
            ]
        )
        agent = Agent(
            name="bot", model=model, tools=[Catalog()], tool_search="dispatch", tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT[:] = [Tool(name="call_tool", handler=lambda name, arguments: app_ran.append((name, arguments)))]
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        assert app_ran == [] and len(results(memory)) == 2


class TestASlashCommandThatPauses:
    @pytest.mark.parametrize("close", [True, False], ids=["read to the end", "plain break"])
    async def test_its_call_is_recorded_and_it_resumes(self, close):
        """A command's call is not in the turn's memory while it runs: its record is taken from the command's
        own tool, and the turn still pauses on it and runs it once approved."""
        charged = []
        charge = Tool(name="charge", handler=lambda amount: charged.append(amount) or "charged", requires_approval=True)
        charge.command = "charge"
        model = TestModel(responses=["must not be requested"])
        agent = Agent(name="bot", model=model, tools=[charge], tracing_provider=InMemoryTracingProvider)
        fresh()
        stream = agent(prompt="/charge 5")
        approval = None
        async for event in stream:
            if isinstance(event, ApprovalEvent):
                approval = event
                if not close:
                    break
        if close:
            from timbal.state import get_run_context

            root = next(s for s in get_run_context()._trace.values() if s.parent_call_id is None)
            assert root.status.reason == "approval_required"
            (binding,) = root.metadata["paused_calls"]["bindings"].values()
            assert binding == "tool"
        fresh(approval.run_id)
        _, out = await run(agent, prompt="/charge 5", resume={approval.approval_id: True})
        assert out.status.code == "success" and charged == ["5"] and model.call_count == 0


# A second provider of tools, with a catalog of its own.
OTHER: list = []


class OtherCatalog(ToolSet):
    async def resolve(self):
        return list(OTHER)


@pytest.fixture(autouse=True)
def reset_other():
    OTHER.clear()
    yield
    OTHER.clear()


class TestAnotherProvidersToolOfTheSameName:
    """Two tool sets offer a tool of one name. The call was made for the first one's: the second one's does not
    run it, with the approval given for the first one's."""

    @pytest.mark.parametrize("dispatch", [False, True], ids=["called", "dispatched"])
    async def test_is_not_run_with_the_first_ones_approval(self, dispatch):
        ran = []
        first = Tool(name="lookup", handler=lambda key: f"first {key}", requires_approval=True)
        second = Tool(name="lookup", handler=lambda key: ran.append(key) or "second", requires_approval=True)
        if dispatch:
            first.defer_loading = second.defer_loading = True
            asked = call("call_tool", name="lookup", arguments={"key": "K"})
        else:
            asked = call("lookup", key="K")
        CURRENT[:] = [first]
        model = TestModel(responses=[asked, "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[Catalog(), OtherCatalog()],
            tool_search="dispatch" if dispatch else "auto",
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        OTHER[:] = [second]
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        assert ran == []
        assert any("this call was made for" in r for r in results(memory))

    async def test_a_tool_two_sets_share_is_the_first_ones(self):
        """One tool object offered by two sets is the first set's (the agent keeps that one), and it runs on the
        resume while that set still offers it, whatever the second set does."""
        ran = []
        shared = Tool(name="lookup", handler=lambda key: ran.append(key) or "found", requires_approval=True)
        CURRENT[:] = [shared]
        OTHER[:] = [shared]
        model = TestModel(responses=[call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog(), OtherCatalog()], tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        OTHER.clear()
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert out.status.code == "success" and ran == ["K"]


class TestAStopThatIsNotForThisRun:
    async def test_a_cancel_for_another_card_does_not_end_the_run(self):
        """The resume carries an approval for the waiting call (whose tool is gone) and a stale Cancel for a
        card this run never waited on: the stale one is not the user's word about this turn."""
        from timbal.types.approval import Cancel

        CURRENT[:] = [Tool(name="refund", handler=lambda order: f"refunded {order}", requires_approval=True)]
        model = TestModel(responses=[call("refund", order="A-1"), "done", "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Catalog()], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True, "some-old-card": Cancel(reason="stale")})
        assert out.status.code == "success" and model.call_count == 2


class TestTheAgentsMetadata:
    async def test_does_not_change_which_tools_a_turn_has(self):
        """A span starts with the agent's own metadata: keys of that name there are the application's, not a
        record of a paused turn, and a first turn keeps every tool it has."""
        ran = []
        lookup = Tool(name="lookup", handler=lambda key: ran.append(key) or "found")
        model = TestModel(responses=[call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[lookup],
            tool_search="off",
            metadata={
                "tool_names": {"search_tools": "lookup"},
                "paused_calls": {"tool_names": {"search_tools": "lookup"}, "bindings": {}, "waiting_on": []},
            },
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        _, out = await run(agent)
        assert out.status.code == "success" and ran == ["K"]


class TestARecordTheRunDidNotWrite:
    async def test_a_pause_at_the_agents_own_gate_reads_none(self):
        """The turn paused at the agent's own approval, before its handler ran: its span holds only what the
        agent's metadata brought, which is not a record of a paused turn."""
        ran = []
        lookup = Tool(name="lookup", handler=lambda key: ran.append(key) or "found")
        model = TestModel(responses=[call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(
            name="bot",
            model=model,
            tools=[lookup],
            tool_search="off",
            requires_approval=True,
            metadata={"paused_calls": {"tool_names": {"search_tools": "lookup"}, "bindings": {}, "waiting_on": []}},
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert out.status.code == "success" and ran == ["K"]


class TestTheProviderOfAResumedCall:
    """The rest of the provider check: a slash command run again, a resume given its history by the caller, and a
    provider whose identity changed or cannot be told apart."""

    async def test_a_slash_command_is_not_run_by_another_providers_tool(self):
        ran = []
        first = Tool(name="charge", handler=lambda amount: f"first {amount}", requires_approval=True)
        first.command = "charge"
        second = Tool(name="charge", handler=lambda amount: ran.append(amount) or "second", requires_approval=True)
        second.command = "charge"
        CURRENT[:] = [first]
        model = TestModel(responses=["done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog(), OtherCatalog()], tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent, prompt="/charge 5")
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        OTHER[:] = [second]
        fresh(parked.run_id)
        _, out = await run(agent, prompt="/charge 5", resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        assert ran == []
        assert any("this call was made for" in r for r in results(memory))

    async def test_a_resume_with_the_callers_history_is_checked_too(self):
        ran = []
        first = Tool(name="lookup", handler=lambda key: f"first {key}", requires_approval=True)
        second = Tool(name="lookup", handler=lambda key: ran.append(key) or "second", requires_approval=True)
        CURRENT[:] = [first]
        model = TestModel(responses=[call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog(), OtherCatalog()], tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        history, _ = memory_of(parked)
        CURRENT.clear()
        OTHER[:] = [second]
        fresh(parked.run_id)
        events = [e async for e in agent(messages=list(history), resume={approval.approval_id: True})]
        out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
        memory, _ = memory_of(out)
        assert ran == []
        assert any("this call was made for" in r for r in results(memory))

    def test_a_named_mcp_server_is_its_endpoint_too(self):
        from timbal.core import MCPServer

        def identity(**kwargs):
            return MCPServer(name="crm", transport="http", **kwargs)._provider_identity()

        assert identity(url="https://a.example/mcp") != identity(url="https://b.example/mcp")
        assert identity(url="https://a.example/mcp") == identity(url="https://a.example/mcp")
        # Two tenants of one endpoint are two backends.
        tenant = {"url": "https://a.example/mcp"}
        assert identity(**tenant, headers={"X-Tenant": "a"}) != identity(**tenant, headers={"X-Tenant": "b"})

    def test_a_mcp_servers_auth_is_who_it_reaches_the_server_as(self):
        """``auth`` (an ``httpx.Auth``) is a credential like a header: two tenants that differ only by it are two
        backends. What it does as it runs (a digest's nonce count) is not who it is."""
        import httpx
        from timbal.core import MCPServer

        def identity(auth):
            return MCPServer(name="crm", transport="http", url="https://a.example/mcp", auth=auth)._provider_identity()

        assert identity(httpx.BasicAuth("a", "x")) != identity(httpx.BasicAuth("b", "y"))
        assert identity(httpx.BasicAuth("a", "x")) == identity(httpx.BasicAuth("a", "x"))
        assert identity(httpx.DigestAuth("a", "x")) != identity(httpx.DigestAuth("b", "y"))
        assert identity(httpx.BasicAuth("a", "x")) != identity(None)

        class Bearer(httpx.Auth):
            def __init__(self, token):
                self.token = token
                self.calls = 0

            def auth_flow(self, request):
                self.calls += 1
                request.headers["Authorization"] = f"Bearer {self.token}"
                yield request

        used = Bearer("t-1")
        before = identity(used)
        used.calls = 7
        assert identity(used) == before and identity(Bearer("t-2")) != before

        # Credentials kept in slots: written as one name, or declared by a base class.
        class OneSlot(httpx.Auth):
            __slots__ = "token"

            def __init__(self, token):
                self.token = token

        class Base(httpx.Auth):
            __slots__ = ("token",)

            def __init__(self, token):
                self.token = token

        class Child(Base):
            __slots__ = ("extra",)

        assert identity(OneSlot("a")) != identity(OneSlot("b"))
        assert identity(Child("a")) != identity(Child("b"))

    def test_credentials_an_auth_keeps_inside_other_objects_count_too(self):
        """``SnowflakeAuth`` (the auth documented for ``MCPServer``) keeps its account, user, keys and integration
        inside an object of its own: two tenants of one endpoint are told apart by them, and the same
        configuration built again is the same."""
        import httpx
        from pydantic import SecretStr
        from timbal.core import MCPServer
        from timbal.tools.snowflake import SnowflakeAuth

        def identity(auth):
            return MCPServer(name="sf", transport="http", url="https://x.example/mcp", auth=auth)._provider_identity()

        assert identity(SnowflakeAuth(account="a", user="u", token="t")) != identity(
            SnowflakeAuth(account="b", user="u", token="t")
        )
        assert identity(SnowflakeAuth(integration="int-1")) != identity(SnowflakeAuth(integration="int-2"))
        assert identity(SnowflakeAuth(account="a", private_key=SecretStr("k1"))) != identity(
            SnowflakeAuth(account="a", private_key=SecretStr("k2"))
        )
        assert identity(SnowflakeAuth(account="a", user="u", token="t")) == identity(
            SnowflakeAuth(account="a", user="u", token="t")
        )

        class Nested(httpx.Auth):
            def __init__(self, tenant):
                self.config = {"tenant": {"id": tenant}, "retries": 3}
                self.me = self  # a cycle

        assert identity(Nested("a")) != identity(Nested("b"))
        assert identity(Nested("a")) == identity(Nested("a"))

        # What SnowflakeAuth resolves as it runs (prepare()) is not who it is.
        prepared = SnowflakeAuth(account="a", user="u", token="t")
        before = identity(prepared)
        prepared._settings = {"account": "a", "token": "minted-at-run-time"}
        assert identity(prepared) == before

    def test_an_auths_identity_is_the_same_in_every_process(self):
        """Taken from what it holds in an order every process agrees on: not a set's iteration order (it follows
        the hash seed), not an object's ``str()`` (it holds an address)."""
        import os
        import subprocess
        import sys

        script = (
            "import hashlib, json, uuid, httpx\n"
            "from pydantic import BaseModel, PrivateAttr\n"
            "from timbal.core.mcp import _auth_identity\n"
            "class Tag: pass\n"
            "class Settings(BaseModel):\n"
            "    api_key: str\n"
            "    _session: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)\n"
            "class Auth(httpx.Auth):\n"
            "    def __init__(self):\n"
            "        self.settings = Settings(api_key='tenant-A')\n"
            "        self.by_tag = {Tag(): 'x', 'tenant': 'a'}\n"
            "        self.scopes = {f'scope:{i}' for i in range(100)}\n"
            "print(hashlib.sha256(json.dumps(_auth_identity(Auth()), default=str).encode()).hexdigest())\n"
        )
        digests = {
            subprocess.run(
                [sys.executable, "-c", script],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.strip()
            for seed in ("1", "2", "3")
        }
        assert len(digests) == 1

    def test_an_auth_that_raises_when_looked_at_does_not_stop_the_agent(self):
        """A proxy whose ``__dict__`` raises outside its context, a property that raises, a key whose ``str()``
        raises: what cannot be read is left out, and the identity is still taken."""
        import httpx
        from timbal.core.mcp import _auth_identity

        class Proxy:
            @property
            def __dict__(self):
                raise RuntimeError("working outside of application context")

        class Slotted:
            __slots__ = ("token",)

        class Loud(Slotted):
            @property
            def token(self):
                raise RuntimeError("no token here")

        class BadKey:
            def __str__(self):
                raise RuntimeError("no str")

        class Auth(httpx.Auth):
            def __init__(self):
                self.proxy = Proxy()
                self.loud = Loud()
                self.keyed = {BadKey(): "x"}
                self.tenant = "a"

        held = _auth_identity(Auth())[2]
        assert (".tenant", "a") in held

    def test_a_models_fields_count_and_its_private_state_does_not(self):
        import httpx
        from pydantic import BaseModel, PrivateAttr
        from timbal.core.mcp import _auth_identity

        class Settings(BaseModel):
            api_key: str
            _calls: list = PrivateAttr(default_factory=list)

        class Meta(type):
            def __getattr__(cls, name):
                raise AttributeError(name)

        class Holder(metaclass=Meta):
            def __init__(self, token):
                self.token = token

        class Auth(httpx.Auth):
            def __init__(self, key, token="t"):
                self.settings = Settings(api_key=key)
                self.holder = Holder(token)

        assert _auth_identity(Auth("tenant-A")) != _auth_identity(Auth("tenant-B"))
        used = Auth("tenant-A")
        before = _auth_identity(used)
        used.settings._calls.append("ran")
        assert _auth_identity(used) == before
        # A class whose metaclass answers lookups is still read (only the instance's own lookups count).
        assert _auth_identity(Auth("tenant-A", "t1")) != _auth_identity(Auth("tenant-A", "t2"))

    async def test_sets_of_one_class_with_another_configuration_are_two_providers(self):
        """``SqlTools(db="a")`` and ``SqlTools(db="b")``: their configuration tells them apart, so their swapping
        places is a provider change, and a call that paused on the first is not run on the second."""
        ran = []

        class SqlTools(ToolSet):
            db: str
            offered: list

            async def resolve(self):
                return list(self.offered)

        def query(db):
            return Tool(name="query", handler=lambda q: ran.append((db, q)) or db, requires_approval=True)

        a, b = SqlTools(db="a", offered=[query("a")]), SqlTools(db="b", offered=[query("b")])
        assert a._provider_identity() != b._provider_identity()
        model = TestModel(responses=[call("query", q="drop"), "done", "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[a, b], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        agent.tools[:] = [b, a]
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        memory, _ = memory_of(out)
        assert ran == [] and any("this call was made for" in r for r in results(memory))

    async def test_sets_alike_resume_while_nothing_changed(self):
        """Two sets that are the same by every measure the agent has (the same class, no name) are told apart by
        their order. Nothing changed between the pause and the resume: the question gets its answer (a refusal
        here would ask again, and refuse again, for ever)."""
        got = []

        def ask(note: str) -> str:
            answer = suspend({"question": "Which?"})
            got.append(answer)
            return f"{note}: {answer}"

        class Static(ToolSet):
            offered: list

            async def resolve(self):
                return list(self.offered)

        agent = Agent(
            name="bot",
            model=TestModel(responses=[call("ask_user", note="n"), "done", "must not be requested"]),
            tools=[
                Static(offered=[Tool(name="ask_user", handler=ask)]),
                Static(offered=[Tool(name="refund", handler=lambda o: o)]),
            ],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        question = next(e for e in events if isinstance(e, InteractionEvent))
        fresh(parked.run_id)
        _, out = await run(agent, resume={question.interaction_id: "Ada"})
        assert out.status.code == "success" and got == ["Ada"]

    async def test_sets_alike_whose_layout_changed_need_a_fresh_decision(self):
        """The two sets (alike: the same class and configuration) swapped places between the pause and the resume,
        and both offer the tool: the call is not run on the old answer. The model calls again; that call is bound
        to the layout as it is now, and runs once approved (a fresh decision that sticks)."""
        ran = []

        class Static(ToolSet):
            offered: list

            async def resolve(self):
                return list(self.offered)

        def lookup(where):
            return Tool(name="lookup", handler=lambda key: ran.append((where, key)) or where, requires_approval=True)

        def only(where):
            return Tool(name=f"{where}_only", handler=lambda: where)

        first = Static(offered=[lookup("first"), only("first")])
        second = Static(offered=[lookup("second"), only("second")])
        model = TestModel(
            responses=[call("lookup", key="K"), call("lookup", key="K2"), "done", "must not be requested"]
        )
        agent = Agent(name="bot", model=model, tools=[first, second], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        agent.tools[:] = [second, first]
        fresh(parked.run_id)
        events, parked = await run(agent, resume={approval.approval_id: True})
        assert parked.status.reason == "approval_required" and ran == []
        memory, _ = memory_of(parked)
        assert any("may not be the one" in r for r in results(memory))
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert out.status.code == "success" and ran == [("second", "K2")]


class TestAProvidersIdentityHoldsWhileItRuns:
    async def test_state_a_set_keeps_in_its_fields_does_not_change_it(self):
        """A set may keep what it does in its fields (a skill the references it read): the identity is taken the
        first time the agent sees the set, so the call that waited is still the set's when it resumes."""
        ran = []

        class Remembering(ToolSet):
            seen: list = []

            async def resolve(self):
                self.seen = [*self.seen, len(self.seen)]
                return [Tool(name="refund", handler=lambda order: ran.append(order) or "ok", requires_approval=True)]

        model = TestModel(responses=[call("refund", order="A-1"), "done", "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Remembering()], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        fresh(parked.run_id)
        _, out = await run(agent, resume={approval.approval_id: True})
        assert out.status.code == "success" and ran == ["A-1"]

    def test_a_skill_is_not_its_references(self, tmp_path):
        from timbal.core.skill import Skill

        (tmp_path / "SKILL.md").write_text("---\nname: billing\ndescription: Billing.\n---\nRefunds.\n")
        skill = Skill(path=tmp_path)
        before = skill._provider_identity()
        skill.references["policy.md"] = "Refund within 30 days."
        assert skill._provider_identity() == before

    async def test_a_siblings_changing_description_does_not_move_a_call(self):
        """Two sets alike (the same class and configuration); the second's tool says something new on every
        resolve. The first one's question still gets its answer."""
        got, n = [], [0]

        def ask(note: str) -> str:
            got.append(suspend({"question": "Which?"}))
            return note

        class Static(ToolSet):
            which: int = 0

            async def resolve(self):
                if self.__dict__.get("_asks"):
                    return [Tool(name="ask_user", handler=ask)]
                n[0] += 1
                return [Tool(name="status", description=f"Open tickets: {n[0]}", handler=lambda: "ok")]

        first, second = Static(), Static()
        first.__dict__["_asks"] = True
        agent = Agent(
            name="bot",
            model=TestModel(responses=[call("ask_user", note="n"), "done", "must not be requested"]),
            tools=[first, second],
            tracing_provider=InMemoryTracingProvider,
        )
        fresh()
        events, parked = await run(agent)
        question = next(e for e in events if isinstance(e, InteractionEvent))
        fresh(parked.run_id)
        _, out = await run(agent, resume={question.interaction_id: "Ada"})
        assert out.status.code == "success" and got == ["Ada"]


class TestARefusedCallsAnswer:
    """Review of 3e93f2f5: what the refusal of a paused call leaves behind, and when a resume can be checked."""

    async def test_is_not_used_by_the_same_call_made_again(self):
        """The call was refused (another provider has the tool now). The model makes it again, with the same
        input: that is a new decision on the provider there now, not the old answer (whose approval id, the path
        and the input, is the same)."""
        ran = []
        first = Tool(name="lookup", handler=lambda key: f"first {key}", requires_approval=True)
        second = Tool(name="lookup", handler=lambda key: ran.append(key) or "second", requires_approval=True)
        CURRENT[:] = [first]
        model = TestModel(responses=[call("lookup", key="K"), call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(
            name="bot", model=model, tools=[Catalog(), OtherCatalog()], tracing_provider=InMemoryTracingProvider
        )
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        CURRENT.clear()
        OTHER[:] = [second]
        fresh(parked.run_id)
        events, out = await run(agent, resume={approval.approval_id: True})
        assert ran == []
        assert out.status.reason == "approval_required"  # asked again, for the provider there now
        assert any(isinstance(e, ApprovalEvent) for e in events)

    async def test_a_history_whose_turn_cannot_be_read_is_not_replayed(self):
        """The caller passes the history (``messages=``) and the run it continues cannot be read: nothing says
        what its waiting calls were made for, so none runs on the answers given."""
        ran = []
        lookup = Tool(name="lookup", handler=lambda key: ran.append(key) or "found", requires_approval=True)
        CURRENT[:] = [lookup]
        model = TestModel(responses=[call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Catalog()], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        history, _ = memory_of(parked)
        fresh("a-run-nobody-stored")
        events = [e async for e in agent(messages=list(history), resume={approval.approval_id: True})]
        out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
        memory, _ = memory_of(out)
        assert ran == []
        assert any("could not be checked" in r for r in results(memory))

    async def test_a_stop_with_a_history_that_cannot_be_checked_ends_the_run(self):
        """The run it continues cannot be read, so which card a Cancel answers is not known either: the caller
        said stop, and the run ends there (no model call, no tool), rather than the Cancel being set aside with
        the other answers while the turn goes on."""
        from timbal.types.approval import Cancel

        ran = []
        lookup = Tool(name="lookup", handler=lambda key: ran.append(key) or "found", requires_approval=True)
        CURRENT[:] = [lookup]
        model = TestModel(responses=[call("lookup", key="K"), "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Catalog()], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        history, _ = memory_of(parked)
        fresh("a-run-nobody-stored")
        events = [e async for e in agent(messages=list(history), resume={approval.approval_id: Cancel(reason="stop")})]
        out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
        assert out.status.code == "cancelled" and ran == [] and model.call_count == 1

    async def test_a_history_call_the_paused_turn_did_not_record_is_not_replayed(self):
        """The caller's history holds a waiting call the paused turn has no record of (another id): it is not
        one the answers were given for."""
        ran = []
        lookup = Tool(name="lookup", handler=lambda key: ran.append(key) or "found", requires_approval=True)
        CURRENT[:] = [lookup]
        model = TestModel(responses=[call("lookup", key="K"), "done", "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[Catalog()], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        history, _ = memory_of(parked)
        edited = [
            *history[:-1],
            Message(
                role="assistant",
                content=[ToolUseContent(id="c-other", name="lookup", input={"key": "K"})],
                stop_reason="tool_use",
            ),
        ]
        fresh(parked.run_id)
        events = [e async for e in agent(messages=edited, resume={approval.approval_id: True})]
        out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
        memory, _ = memory_of(out)
        assert ran == []
        assert any("could not be checked" in r for r in results(memory))


class TestAProvidersIdentityFollowsItsConfiguration:
    def test_a_mcp_server_reconfigured_while_running_is_another_provider(self):
        """An MCP server's identity is all configuration: one whose address or credentials change while the
        process runs is another backend from then on (it is not kept from the first time the agent saw it)."""
        from timbal.core import MCPServer
        from timbal.core.agent import _provider_key

        server = MCPServer(name="crm", transport="http", url="https://a.example/mcp")
        before = _provider_key(server)
        server.url = "https://b.example/mcp"
        assert _provider_key(server) != before
        server.url = "https://a.example/mcp"
        assert _provider_key(server) == before

    def test_what_a_servers_auth_does_as_it_runs_does_not_change_it(self):
        """The server's own fields are read as they are now; its auth is read the first time it is seen (a digest
        auth keeps the server's last challenge, an OAuth provider its tokens: what it does, not who it is). A new
        auth object is a reconfiguration, and another identity."""
        import httpx
        from timbal.core import MCPServer
        from timbal.core.agent import _provider_key

        auth = httpx.DigestAuth("user", "pass")
        server = MCPServer(name="crm", transport="http", url="https://a.example/mcp", auth=auth)
        before = _provider_key(server)
        auth._last_challenge = ("realm", "nonce-1")  # a first 401
        assert _provider_key(server) == before
        auth._last_challenge = ("realm", "nonce-2")  # a new nonce
        assert _provider_key(server) == before
        server.auth = httpx.DigestAuth("other", "pass")
        assert _provider_key(server) != before

    def test_equal_auth_mappings_are_one_identity_whatever_their_order(self):
        import httpx
        from timbal.core.mcp import _auth_identity

        class Auth(httpx.Auth):
            def __init__(self, config):
                self.config = config

        assert _auth_identity(Auth({"tenant": "a", "region": "eu"})) == _auth_identity(
            Auth({"region": "eu", "tenant": "a"})
        )
        assert _auth_identity(Auth({"tenant": "a"})) != _auth_identity(Auth({"tenant": "b"}))


class TestReview6065154730:
    """Review of d5e1eaae."""

    async def test_a_structured_output_turn_is_not_offered_the_salience_tool(self):
        """A turn with an ``output_model`` ends on the model's first answer, before any tool of it runs: a mark
        it chose would be neither recorded nor the declared output. Like the delivery tool, it is not offered."""
        from pydantic import BaseModel
        from timbal.state import get_run_context

        class Answer(BaseModel):
            text: str

        model = TestModel(responses=['{"text": "ok"}'])
        agent = Agent(
            name="bot", model=model, salience_tool=True, output_model=Answer, tracing_provider=InMemoryTracingProvider
        )
        fresh()
        _, out = await run(agent)
        llm = next(s for s in get_run_context()._trace.values() if s.path == "bot.llm")
        names = [t["name"] if isinstance(t, dict) else t.name for t in (llm.input or {}).get("tools") or ()]
        assert "mark_salient" not in names
        assert isinstance(out.output, Answer)

    async def test_a_callers_history_with_no_answers_runs_its_calls_with_tracing_off(self):
        """With tracing off, a caller keeps the history: a turn that ends in a tool call it wants run (no
        approval, no answers given) runs it, as before. The check is for answers given to a paused turn."""
        ran = []

        def multiply(a: int, b: int) -> int:
            ran.append((a, b))
            return a * b

        model = TestModel(responses=["done", "must not be requested"])
        agent = Agent(name="bot", model=model, tools=[multiply], tracing_provider=None)
        history = [
            Message(role="user", content=[TextContent(text="6 times 7?")]),
            Message(
                role="assistant",
                content=[ToolUseContent(id="c-mul", name="multiply", input={"a": 6, "b": 7})],
                stop_reason="tool_use",
            ),
        ]
        set_run_context(RunContext(parent_id="an-earlier-turn", tracing_provider=None))
        set_call_id(None)
        set_parent_call_id(None)
        events = [e async for e in agent(messages=history)]
        out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
        assert out.status.code == "success" and ran == [(6, 7)]
