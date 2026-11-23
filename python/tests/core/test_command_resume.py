"""A slash command that paused (an approval, a question) and is resumed.

The caller answers with the same prompt, ``parent_id`` and ``resume``. A command is the user's own
request, run without the model: resumed, it is still that. The call the user approved runs once, a call
they denied does not run, and no model answers in the command's place, whatever the turn before left in
memory and wherever the run was stored.

How: the agent stores a made-up call for the command before it runs, marked as a command's. A command
that pauses has no result, so the paused turn's history ends with that call, in memory as in a store.
The turn that continues it starts like any turn continuing a pending call (nothing new taken in), then
takes the call back out and runs the command again from the user's message, letting the command's
stream finish before it pauses again. A first run is as it was: it stops at the first pause.
"""

import asyncio
import json

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.chat_session import ChatSession
from timbal.core.runnable import _approval_id_for
from timbal.core.test_model import TestModel
from timbal.core.tool_set import ToolSet
from timbal.guardrails import guardrail
from timbal.state import (
    get_run_context,
    post_session_event,
    set_call_id,
    set_parent_call_id,
    set_run_context,
    suspend,
)
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.providers.sqlite import SqliteTracingProvider
from timbal.types.approval import Cancel
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, InteractionEvent, OutputEvent
from timbal.types.message import Message

# A hang guard, not a timing check: these runs take milliseconds.
DEADLINE_S = 30

# This process's memory, and two stores that keep the history as it is serialized (as calls).
KINDS = ["memory", "jsonl", "sqlite"]
# Stores that keep approval claims (the in-memory provider keeps none: two resumes of one run both run).
CLAIMING = ["jsonl", "sqlite"]


async def _run(agent: Agent, **kwargs) -> list:
    """Every event of one run."""

    async def events() -> list:
        return [event async for event in agent(**kwargs)]

    return await asyncio.wait_for(events(), DEADLINE_S)


async def _run_releasing(agent: Agent, release: asyncio.Event, on: str, **kwargs) -> list:
    """Every event of one run, setting ``release`` as soon as a call of it ends with the status reason
    ``on`` (a pause, a lost claim): what a handler waits for to be sure that event came while it ran."""

    async def events() -> list:
        seen = []
        async for event in agent(**kwargs):
            seen.append(event)
            if isinstance(event, OutputEvent) and event.status.reason == on:
                release.set()
        return seen

    return await asyncio.wait_for(events(), DEADLINE_S)


async def _run_until_asked(agent: Agent, **kwargs) -> ApprovalEvent:
    """Read a run up to its approval card and stop reading there, as a caller that breaks out of its loop."""

    async def read() -> ApprovalEvent:
        stream = agent(**kwargs)
        try:
            async for event in stream:
                if isinstance(event, ApprovalEvent):
                    return event
        finally:
            await stream.aclose()
        raise AssertionError("no approval was asked for")

    return await asyncio.wait_for(read(), DEADLINE_S)


def _new_conversation() -> None:
    """A run started after this one continues nothing: without it, a call with no ``parent_id`` made from
    the task that ran the last one would continue that one."""
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def _approval(events: list) -> ApprovalEvent:
    return next(event for event in events if isinstance(event, ApprovalEvent))


def _asked(events: list) -> dict[str, str]:
    """The approval id of each card a run showed, by the amount it asks about."""
    return {event.input["amount"]: event.approval_id for event in events if isinstance(event, ApprovalEvent)}


def _final(events: list) -> OutputEvent:
    return next(event for event in reversed(events) if isinstance(event, OutputEvent))


def _provider(kind: str, tmp_path):
    if kind == "memory":
        return InMemoryTracingProvider
    if kind == "jsonl":
        return JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
    return SqliteTracingProvider.configured(_path=tmp_path / "traces.db")


def _outer_model() -> TestModel:
    """The model of the agent the command is typed to. A command never reaches it."""
    return TestModel(responses=["must not be requested"])


def _tool_call(name: str, **arguments) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"call-{name}", name=name, input=arguments)],
        stop_reason="tool_use",
    )


def _two_charges() -> Message:
    """One step of a model that asks for two gated calls at once."""
    return Message(
        role="assistant",
        content=[
            ToolUseContent(id="call-a", name="charge", input={"amount": "5"}),
            ToolUseContent(id="call-b", name="charge", input={"amount": "7"}),
        ],
        stop_reason="tool_use",
    )


async def _history(provider, run_id: str, path: str = "outer") -> list[Message]:
    """What a run left for the turn after it, read back the way that turn reads it."""
    trace = await asyncio.wait_for(RunContext(tracing_provider=provider)._get_parent_trace(run_id), DEADLINE_S)
    (span,) = trace.get_path(path)
    return [Message.validate(message) for message in span.memory]


def _shape(messages: list[Message]) -> list[tuple[str, list[str]]]:
    return [(message.role, [content.type for content in message.content]) for message in messages]


class _Charges:
    """A tool that must not run before it is approved, and then once."""

    def __init__(self):
        self.started: list[str] = []
        self.made: list[str] = []
        # Set by the reader of a run (``_run_releasing``) when the event a test is about has reached it.
        self.release = asyncio.Event()

    def charge(self, amount: str) -> str:
        self.started.append(amount)
        self.made.append(amount)
        return f"charged {amount}"

    async def charge_when_released(self, amount: str) -> str:
        """A call still running when the call next to it pauses or loses its gate: it ends only once that
        event has reached the reader of the run."""
        self.started.append(amount)
        await self.release.wait()
        self.made.append(amount)
        return f"charged {amount}"

    def tool(self, *, when_released: bool = False, name: str = "charge", **kwargs) -> Tool:
        handler = self.charge_when_released if when_released else self.charge
        return Tool(name=name, handler=handler, requires_approval=True, **kwargs)


class TestNestedPause:
    """The command is an agent or a workflow, and the pause comes from a step of its own: other steps
    have finished by then."""

    @pytest.mark.parametrize("kind", KINDS)
    async def test_an_approved_call_inside_an_agent_command_runs_on_resume(self, kind, tmp_path):
        charges = _Charges()
        helper = Agent(
            name="helper",
            command="helper",
            model=TestModel(responses=[_tool_call("charge", amount="5"), "paid"]),
            tools=[charges.tool()],
        )
        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[helper], tracing_provider=_provider(kind, tmp_path))

        first = await _run(outer, prompt="/helper pay")
        paused = _final(first)
        assert (paused.path, paused.status.reason) == ("outer", "approval_required")
        assert charges.made == []

        resumed = _final(
            await _run(
                outer,
                prompt="/helper pay",
                parent_id=paused.run_id,
                resume={_approval(first).approval_id: True},
            )
        )

        assert charges.made == ["5"], "the approved call must run, once"
        assert model.call_count == 0, "a command is not answered by the model"
        assert resumed.status.code == "success"
        assert resumed.output.collect_text() == "paid", "the command's own answer is the turn's"

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_gated_step_of_a_workflow_command_runs_on_resume(self, kind, tmp_path):
        charges = _Charges()

        def prepare(amount: str) -> str:
            return f"{amount} EUR"

        pay = (
            Workflow(name="pay", command="pay")
            .step(prepare)
            .step(charges.tool(), amount=lambda: get_run_context().step_span("prepare").output)
        )
        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[pay], tracing_provider=_provider(kind, tmp_path))

        first = await _run(outer, prompt="/pay 5")
        paused = _final(first)
        assert (paused.path, paused.status.reason) == ("outer", "approval_required")
        assert charges.made == []

        resumed = _final(
            await _run(outer, prompt="/pay 5", parent_id=paused.run_id, resume={_approval(first).approval_id: True})
        )

        assert charges.made == ["5 EUR"], "the approved step must run, once"
        assert model.call_count == 0, "a command is not answered by the model"
        assert (resumed.status.code, resumed.output) == ("success", "charged 5 EUR")

    @pytest.mark.parametrize("kind", KINDS)
    async def test_memory_holds_what_is_stored_when_a_command_pauses(self, kind, tmp_path):
        helper = Agent(
            name="helper",
            command="helper",
            model=TestModel(responses=[_tool_call("charge", amount="5"), "paid"]),
            tools=[_Charges().tool()],
        )
        provider = _provider(kind, tmp_path)
        outer = Agent(name="outer", model=_outer_model(), tools=[helper], tracing_provider=provider)

        paused = _final(await _run(outer, prompt="/helper pay"))

        # The user's message and the command's call, marked as one. Not the helper's first answer, which
        # text mode appends to memory as each step finishes: the resume would read it as the last message.
        asked, call = await _history(provider, paused.run_id)
        assert (asked.role, asked.collect_text()) == ("user", "/helper pay")
        assert _shape([call]) == [("assistant", ["tool_use"])]
        assert (call.content[0].name, call.metadata.get("kind")) == ("helper", "command")


class TestCallsInFlight:
    """A command with several calls in flight when one of them pauses.

    Run again by a resume, it is given its stream to the end before the turn pauses or stops on a lost
    claim, as a batch of pending calls is: a call in flight may be one the user has just approved. A first
    run stops at the pause, as it always did: nothing is approved yet, and a call that finished there would
    take effect a second time when the resume runs the command again.
    """

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_call_approved_next_to_one_still_waiting_runs(self, kind, tmp_path):
        charges = _Charges()
        helper = Agent(
            name="helper",
            command="helper",
            model=TestModel(responses=[_two_charges(), "paid"]),
            tools=[charges.tool(when_released=True)],
        )
        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[helper], tracing_provider=_provider(kind, tmp_path))

        first = await _run(outer, prompt="/helper pay")
        amount, approval_id = sorted(_asked(first).items())[0]  # whichever card the first run got to show

        resumed = _final(
            await _run_releasing(
                outer,
                charges.release,
                "approval_required",
                prompt="/helper pay",
                parent_id=_final(first).run_id,
                resume={approval_id: True},
            )
        )

        assert charges.made == [amount], "the approved call runs to its end while the other one waits"
        assert (resumed.status.code, resumed.status.reason) == ("cancelled", "approval_required")
        assert model.call_count == 0

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_step_approved_next_to_one_still_waiting_runs(self, kind, tmp_path):
        charges = _Charges()
        pay = (
            Workflow(name="pay", command="pay")
            .step(charges.tool(when_released=True, name="rent"), amount="5")
            .step(charges.tool(when_released=True, name="bills"), amount="7")
        )
        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[pay], tracing_provider=_provider(kind, tmp_path))

        first = await _run(outer, prompt="/pay")
        amount, approval_id = sorted(_asked(first).items())[0]

        resumed = _final(
            await _run_releasing(
                outer,
                charges.release,
                "approval_required",
                prompt="/pay",
                parent_id=_final(first).run_id,
                resume={approval_id: True},
            )
        )

        assert charges.made == [amount]
        assert (resumed.status.code, resumed.status.reason) == ("cancelled", "approval_required")
        assert model.call_count == 0

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_call_next_to_the_gate_takes_effect_once(self, kind, tmp_path):
        """A call that needs no approval, asked for in the same step as one that does. The first run stops
        at the pause and cuts it; the resume runs the command again, and it runs then."""
        noted: list[str] = []
        charges = _Charges()

        async def note(text: str) -> str:
            await charges.release.wait()
            noted.append(text)
            return "noted"

        both = Message(
            role="assistant",
            content=[
                ToolUseContent(id="call-note", name="note", input={"text": "rent"}),
                ToolUseContent(id="call-charge", name="charge", input={"amount": "5"}),
            ],
            stop_reason="tool_use",
        )
        helper = Agent(
            name="helper",
            command="helper",
            model=TestModel(responses=[both, "paid"]),
            tools=[Tool(name="note", handler=note), charges.tool()],
        )
        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[helper], tracing_provider=_provider(kind, tmp_path))

        first = await _run(outer, prompt="/helper pay")  # ends although ``note`` is still waiting
        paused = _final(first)
        assert (paused.status.code, paused.status.reason) == ("cancelled", "approval_required")
        assert noted == [], "nothing is approved yet: the run stops at the pause"

        charges.release.set()
        resumed = _final(
            await _run(
                outer, prompt="/helper pay", parent_id=paused.run_id, resume={_approval(first).approval_id: True}
            )
        )

        assert (noted, charges.made) == (["rent"], ["5"])
        assert resumed.status.code == "success" and model.call_count == 0

    @pytest.mark.parametrize("kind", CLAIMING)
    async def test_a_run_that_loses_a_gate_finishes_the_call_it_won(self, kind, tmp_path):
        """Two runs continue one paused command, and each wins one of its two gates. The run that finds a
        gate claimed stops, but only once the call it did claim is done: cut there, that call would be
        claimed and never made."""
        charges = _Charges()
        helper = Agent(
            name="helper",
            command="helper",
            model=TestModel(responses=[_two_charges(), "paid"]),
            tools=[charges.tool(when_released=True)],
        )
        provider = _provider(kind, tmp_path)
        outer = Agent(name="outer", model=_outer_model(), tools=[helper], tracing_provider=provider)
        paused = _final(await _run(outer, prompt="/helper pay"))
        five, seven = (_approval_id_for("outer.helper.charge", {"amount": amount}) for amount in ("5", "7"))
        # The other run got to the second gate first.
        assert await asyncio.wait_for(provider.claim_approval(paused.run_id, seven, "another-run"), DEADLINE_S)

        stopped = _final(
            await _run_releasing(
                outer,
                charges.release,
                "approval_already_claimed",
                prompt="/helper pay",
                parent_id=paused.run_id,
                resume={five: True, seven: True},
            )
        )

        assert (charges.started, charges.made) == (["5"], ["5"])
        assert (stopped.status.code, stopped.status.reason) == ("cancelled", "approval_already_claimed")


class TestStoredHistory:
    """The same answer to the same command, whether the paused run is read back from this process's
    memory or from a store that keeps the history as calls."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("approved", [True, False], ids=["approved", "denied"])
    async def test_a_resumed_command_is_still_a_command(self, kind, approved, tmp_path):
        charges = _Charges()
        model = _outer_model()
        outer = Agent(
            name="outer",
            model=model,
            tools=[charges.tool(command="charge")],
            tracing_provider=_provider(kind, tmp_path),
        )

        first = await _run(outer, prompt="/charge 5")
        paused = _final(first)
        assert (paused.path, paused.status.reason) == ("outer", "approval_required")
        assert charges.made == []

        resumed = _final(
            await _run(
                outer,
                prompt="/charge 5",
                parent_id=paused.run_id,
                resume={_approval(first).approval_id: approved},
            )
        )

        assert model.call_count == 0, "a command is not answered by the model"
        if approved:
            assert charges.made == ["5"]
            assert (resumed.status.code, resumed.output) == ("success", "charged 5")
        else:
            assert charges.made == []
            assert (resumed.status.code, resumed.status.reason) == ("cancelled", "approval_denied")

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_question_a_command_asks_is_answered_once(self, kind, tmp_path):
        answers: list[str] = []

        def confirm(target: str) -> str:
            answer = suspend({"question": f"Delete {target}?"})
            answers.append(answer)
            return f"{target}: {answer}"

        model = _outer_model()
        outer = Agent(
            name="outer",
            model=model,
            tools=[Tool(name="confirm", handler=confirm, command="confirm")],
            tracing_provider=_provider(kind, tmp_path),
        )

        first = await _run(outer, prompt="/confirm prod")
        paused = _final(first)
        assert (paused.path, paused.status.reason) == ("outer", "input_required")
        question = next(event for event in first if isinstance(event, InteractionEvent))

        resumed = _final(
            await _run(
                outer,
                prompt="/confirm prod",
                parent_id=paused.run_id,
                resume={question.interaction_id: "yes"},
            )
        )

        assert answers == ["yes"], "the answer must reach the command that asked, the first time"
        assert model.call_count == 0, "a command is not answered by the model"
        assert (resumed.status.code, resumed.output) == ("success", "prod: yes")

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_stop_in_answer_to_a_command_runs_nothing(self, kind, tmp_path):
        """A guard, true before the fix as well: ``Cancel`` ends the run, by either path."""
        charges = _Charges()
        model = _outer_model()
        outer = Agent(
            name="outer",
            model=model,
            tools=[charges.tool(command="charge")],
            tracing_provider=_provider(kind, tmp_path),
        )

        first = await _run(outer, prompt="/charge 5")
        resumed = _final(
            await _run(
                outer,
                prompt="/charge 5",
                parent_id=_final(first).run_id,
                resume={_approval(first).approval_id: Cancel(reason="never mind")},
            )
        )

        assert (resumed.status.code, resumed.status.reason) == ("cancelled", "cancelled")
        assert charges.made == [] and model.call_count == 0

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("command", ["tool", "agent"])
    async def test_a_caller_that_stops_reading_at_the_approval(self, kind, command, tmp_path):
        """The stream is closed on the approval card, before the run's own last events. The run is stored
        as waiting all the same, and its answer must still run the command."""
        charges = _Charges()
        if command == "tool":
            runnable, prompt = charges.tool(command="charge"), "/charge 5"
        else:
            runnable = Agent(
                name="helper",
                command="helper",
                model=TestModel(responses=[_tool_call("charge", amount="5"), "paid"]),
                tools=[charges.tool()],
            )
            prompt = "/helper pay"
        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[runnable], tracing_provider=_provider(kind, tmp_path))

        asked = await _run_until_asked(outer, prompt=prompt)
        resumed = _final(await _run(outer, prompt=prompt, parent_id=asked.run_id, resume={asked.approval_id: True}))

        assert charges.made == ["5"], "the approved call must run, once"
        assert model.call_count == 0, "a command is not answered by the model"
        assert resumed.status.code == "success"

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_resumed_command_leaves_what_one_that_never_paused_leaves(self, kind, tmp_path):
        provider = _provider(kind, tmp_path)
        seen: list[list[Message]] = []

        def answer(messages: list[Message]) -> str:
            seen.append(list(messages))
            return "you are welcome"

        def agent(tool: Tool) -> Agent:
            return Agent(name="outer", model=TestModel(handler=answer), tools=[tool], tracing_provider=provider)

        charges = _Charges()
        gated = agent(charges.tool(command="charge"))
        first = await _run(gated, prompt="/charge 5")
        paused = _final(first)
        resumed = _final(
            await _run(gated, prompt="/charge 5", parent_id=paused.run_id, resume={_approval(first).approval_id: True})
        )
        _new_conversation()
        plain = _final(
            await _run(agent(Tool(name="charge", handler=charges.charge, command="charge")), prompt="/charge 5")
        )
        assert charges.made == ["5", "5"]
        assert _shape(await _history(provider, resumed.run_id)) == _shape(await _history(provider, plain.run_id))

        # And the turn after it reads the command and its result, then the new message.
        assert seen == []
        await _run(gated, prompt="thanks", parent_id=resumed.run_id)
        (messages,) = seen
        assert messages[0].collect_text() == "/charge 5" and messages[-1].collect_text() == "thanks"
        assert "charged 5" in "".join(str(content) for message in messages for content in message.content)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("now", [None, "pay"], ids=["no_longer_a_command", "renamed"])
    async def test_a_command_that_is_none_any_more_still_runs_the_approved_call(self, kind, now, tmp_path):
        """The tool is still there when the answer comes, but ``/charge`` no longer names it (a deploy while
        the card was up). The stored call is a call of that tool and the approval is for it: it runs as the
        pending call it is, and the model is called after it, as a resume always did."""
        provider = _provider(kind, tmp_path)
        charges = _Charges()
        before = Agent(
            name="outer", model=_outer_model(), tools=[charges.tool(command="charge")], tracing_provider=provider
        )
        first = await _run(before, prompt="/charge 5")

        model = TestModel(responses=["charged, as you approved"])
        after = Agent(name="outer", model=model, tools=[charges.tool(command=now)], tracing_provider=provider)
        resumed = _final(
            await _run(
                after, prompt="/charge 5", parent_id=_final(first).run_id, resume={_approval(first).approval_id: True}
            )
        )

        assert charges.made == ["5"], "the approval must not be dropped"
        assert (resumed.status.code, model.call_count) == ("success", 1)

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_command_name_given_to_another_tool_does_not_run_that_tool(self, kind, tmp_path):
        """The approval was for a call of ``charge``. If ``/charge`` names another tool by the time the
        answer comes, running the command again would run that one, unasked: the stored call runs instead."""
        provider = _provider(kind, tmp_path)
        charges = _Charges()
        refunded: list[str] = []

        def refund(amount: str) -> str:
            refunded.append(amount)
            return f"refunded {amount}"

        before = Agent(
            name="outer", model=_outer_model(), tools=[charges.tool(command="charge")], tracing_provider=provider
        )
        first = await _run(before, prompt="/charge 5")

        model = TestModel(responses=["charged, as you approved"])
        after = Agent(
            name="outer",
            model=model,
            tools=[charges.tool(), Tool(name="refund", handler=refund, command="charge")],
            tracing_provider=provider,
        )
        resumed = _final(
            await _run(
                after, prompt="/charge 5", parent_id=_final(first).run_id, resume={_approval(first).approval_id: True}
            )
        )

        assert (charges.made, refunded) == (["5"], [])
        assert (resumed.status.code, model.call_count) == ("success", 1)

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_resume_that_fails_before_its_step_starts_keeps_the_call(self, kind, tmp_path):
        """The tools cannot be resolved for the resume's first step (a tool server restarting). The call was
        taken out of the history by then: the failed run must store it all the same, or a turn after it
        would read the user's ``/charge 5`` with no sign that a call was waiting."""
        provider = _provider(kind, tmp_path)
        charges = _Charges()
        source = {"down": False}

        class Tools(ToolSet):
            async def resolve(self) -> list:
                if source["down"]:
                    raise ConnectionError("the tool source is down")
                return [charges.tool(command="charge")]

        model = _outer_model()
        outer = Agent(name="outer", model=model, tools=[Tools()], tracing_provider=provider)
        first = await _run(outer, prompt="/charge 5")
        paused = _final(first)
        answer = {_approval(first).approval_id: True}

        source["down"] = True
        failed = _final(await _run(outer, prompt="/charge 5", parent_id=paused.run_id, resume=answer))
        assert failed.status.code == "error" and charges.made == []
        assert _shape(await _history(provider, failed.run_id)) == [("user", ["text"]), ("assistant", ["tool_use"])]

        # The paused run is as it was: answered again once the tools are back, the command runs.
        source["down"] = False
        resumed = _final(await _run(outer, prompt="/charge 5", parent_id=paused.run_id, resume=answer))
        assert (charges.made, resumed.status.code, model.call_count) == (["5"], "success", 0)

    async def test_a_marked_call_away_from_its_command_runs_as_a_pending_call(self):
        """Only a call that directly follows the user's ``/command`` message is run again as a command. One
        that does not (a compaction took the message, here a caller's own ``messages``) has nothing to be
        run again from: it runs as the pending call it is."""
        charges = _Charges()
        model = TestModel(responses=["paid"])
        outer = Agent(
            name="outer",
            model=model,
            tools=[Tool(name="charge", handler=charges.charge, command="charge")],
            tracing_provider=InMemoryTracingProvider,
        )
        history = [
            Message(role="user", content="pay the invoice"),
            Message(
                role="assistant",
                content=[ToolUseContent(id="call-charge", name="charge", input={"amount": "5"})],
                metadata={"kind": "command"},
            ),
        ]

        answered = _final(await _run(outer, messages=history))

        assert charges.made == ["5"]
        assert (answered.status.code, model.call_count) == ("success", 1)

    async def test_a_run_paused_by_an_earlier_version_still_resumes(self, tmp_path):
        """A store may hold runs an earlier version paused: the made-up call that ends their history has
        no mark. They resume as they did, the call run as a pending one and the model called after it, so
        an approval given across an upgrade still runs its tool, once."""
        path = tmp_path / "traces.jsonl"
        charges = _Charges()
        model = TestModel(responses=["the model, after the call: what a resume did before"])
        outer = Agent(
            name="outer",
            model=model,
            tools=[charges.tool(command="charge")],
            tracing_provider=JsonlTracingProvider.configured(_path=path),
        )
        first = await _run(outer, prompt="/charge 5")
        paused = _final(first)

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        (record,) = [record for record in records if record["run_id"] == paused.run_id]
        (root,) = [span for span in record["spans"] if span["path"] == "outer"]
        assert root["memory"][-1].pop("metadata") == {"kind": "command"}
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

        resumed = _final(
            await _run(outer, prompt="/charge 5", parent_id=paused.run_id, resume={_approval(first).approval_id: True})
        )

        assert charges.made == ["5"]
        assert resumed.status.code == "success"
        assert model.call_count == 1, "the path this run was paused on, not the command's"


class TestNothingNewIsTakenIn:
    """A resume continues a turn that has started: what arrived while the command waited is for a later
    turn, and what the turn was given has been screened. As when a model's call waits."""

    @pytest.mark.parametrize("kind", KINDS)
    async def test_an_event_posted_while_a_command_waits_is_left_for_a_turn_that_reads_it(self, kind, tmp_path):
        charges = _Charges()
        model = _outer_model()
        outer = Agent(
            name="outer",
            model=model,
            tools=[charges.tool(command="charge")],
            tracing_provider=_provider(kind, tmp_path),
        )
        first = await _run(outer, prompt="/charge 5")
        paused = _final(first)

        receipt = post_session_event(paused.run_id, "the nightly build failed", create=True)
        await _run(outer, prompt="/charge 5", parent_id=paused.run_id, resume={_approval(first).approval_id: True})

        assert charges.made == ["5"] and model.call_count == 0
        assert receipt.state == "pending", "no model ran in that turn: the event would have been read by none"

    @pytest.mark.parametrize("kind", KINDS)
    async def test_the_command_is_screened_once(self, kind, tmp_path):
        screened: list[str] = []

        def rail(text: str) -> bool:
            screened.append(text)
            return True

        charges = _Charges()
        outer = Agent(
            name="outer",
            model=_outer_model(),
            tools=[charges.tool(command="charge")],
            guardrails=[guardrail(rail, stages=["input"], name="counter")],
            tracing_provider=_provider(kind, tmp_path),
        )
        first = await _run(outer, prompt="/charge 5")
        await _run(
            outer, prompt="/charge 5", parent_id=_final(first).run_id, resume={_approval(first).approval_id: True}
        )

        assert charges.made == ["5"]
        assert screened == ["/charge 5"]


async def _records(session: ChatSession, kind: str, count: int) -> list[dict]:
    """The first ``count`` session records of ``kind`` (waits for them)."""
    found: list[dict] = []

    async def collect() -> None:
        async for item in session.events():
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), DEADLINE_S)
    return found


def _card(session: ChatSession) -> ApprovalEvent:
    return next(event for event in reversed(session.log.peek()[0]) if isinstance(event, ApprovalEvent))


class TestChatSession:
    """The same through a session, which answers a parked turn with ``resume()``."""

    @pytest.mark.parametrize("kind", ["memory", "sqlite"])
    @pytest.mark.parametrize("approved", [True, False], ids=["approved", "denied"])
    async def test_a_parked_command_is_answered_as_a_command(self, kind, approved, tmp_path):
        charges = _Charges()
        model = _outer_model()
        agent = Agent(
            name="agent",
            model=model,
            tools=[charges.tool(command="charge")],
            tracing_provider=_provider(kind, tmp_path),
        )

        async with ChatSession(agent) as session:
            session.send("/charge 5")
            (parked,) = await _records(session, "turn_finished", 1)
            assert parked["parked"] and charges.made == []

            session.resume({_card(session).approval_id: approved})
            _, answered = await _records(session, "turn_finished", 2)

        assert model.call_count == 0, "a command is not answered by the model"
        assert not answered["parked"]
        if approved:
            assert charges.made == ["5"] and answered["status"]["code"] == "success"
        else:
            assert charges.made == []
            assert (answered["status"]["code"], answered["status"]["reason"]) == ("cancelled", "approval_denied")

    @pytest.mark.parametrize("kind", ["memory", "sqlite"])
    async def test_an_event_posted_while_a_command_is_parked_gets_a_turn_of_its_own(self, kind, tmp_path):
        charges = _Charges()
        read: list[str] = []

        def answer(messages: list[Message]) -> str:
            read.append(messages[-1].collect_text())
            return "noted"

        agent = Agent(
            name="agent",
            model=TestModel(handler=answer),
            tools=[charges.tool(command="charge")],
            tracing_provider=_provider(kind, tmp_path),
        )

        async with ChatSession(agent) as session:
            session.send("/charge 5")
            (parked,) = await _records(session, "turn_finished", 1)
            post_session_event(parked["run_id"], "the nightly build failed", create=True)

            session.resume({_card(session).approval_id: True})
            started = await _records(session, "turn_started", 3)
            await _records(session, "turn_finished", 3)

        assert charges.made == ["5"]
        assert [record["trigger"]["kind"] for record in started] == ["user_message", "resume", "session_event"]
        (told,) = read
        assert "the nightly build failed" in told

    @pytest.mark.parametrize("kind", ["memory", "sqlite"])
    async def test_a_command_steered_into_a_turn_is_resumed_as_a_command(self, kind, tmp_path):
        """The command arrives while a tool of the turn runs, and pauses. The model's call for that tool is
        behind it in the history: the resume must not take it for a call still waiting to be run."""
        ran: list[str] = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def work() -> str:
            ran.append("work")
            entered.set()
            await release.wait()
            return "worked"

        def charge(amount: str) -> str:
            ran.append(f"charge {amount}")
            return f"charged {amount}"

        def answer(messages: list[Message]) -> Message | str:
            last = messages[-1]
            if last.role == "user" and last.collect_text() == "start":
                return _tool_call("work")
            return "must not be requested"

        model = TestModel(handler=answer)
        agent = Agent(
            name="agent",
            model=model,
            tools=[
                Tool(name="work", handler=work),
                Tool(name="charge", handler=charge, command="charge", requires_approval=True),
            ],
            tracing_provider=_provider(kind, tmp_path),
        )

        async with ChatSession(agent) as session:
            session.send("start")
            await asyncio.wait_for(entered.wait(), DEADLINE_S)
            session.send("/charge 5", mode="steer")
            release.set()
            (parked,) = await _records(session, "turn_finished", 1)
            assert parked["parked"] and ran == ["work"]
            asked_the_model = model.call_count

            session.resume({_card(session).approval_id: True})
            _, answered = await _records(session, "turn_finished", 2)

        assert ran == ["work", "charge 5"], "the approved command runs once, and nothing before it runs again"
        assert model.call_count == asked_the_model, "a command is not answered by the model"
        assert answered["status"]["code"] == "success"
