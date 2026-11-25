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
from timbal.types.content import ToolResultContent, ToolUseContent
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
        assert root.metadata["calls_not_resumed"] == [
            {"tool_call_id": "c-get", "tool": "get_background_task", "made_for": "tool"}
        ]


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
        assert root.metadata.pop("tool_bindings") == {"c-refund": "tool"}
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
            (binding,) = root.metadata["tool_bindings"].values()
            assert binding == "tool"
        fresh(approval.run_id)
        _, out = await run(agent, prompt="/charge 5", resume={approval.approval_id: True})
        assert out.status.code == "success" and charged == ["5"] and model.call_count == 0
