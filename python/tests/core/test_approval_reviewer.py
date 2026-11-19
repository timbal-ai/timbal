"""Agent(approval_reviewer=...): approval gates decided before a person is asked; max_denials_window."""

import asyncio
import json

import pytest
from timbal import Agent, Tool
from timbal.core.approval_reviewer import ApprovalRequest, ApprovalReviewer, ReviewVerdict
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


@pytest.fixture(autouse=True)
def isolated():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)


def call(name, call_id="c1", **arguments):
    return Message(
        role="assistant", content=[ToolUseContent(id=call_id, name=name, input=arguments)], stop_reason="tool_use"
    )


def pay_then_report(messages):
    """Calls charge(amount=5), then answers ``done``."""
    return "done" if messages[-1].role == "tool" else call("charge", amount=5)


async def tool_results(context):
    """What the model was told about its tool calls (the stored tool results), as one string."""
    from timbal.utils import dump

    return json.dumps(await dump(context.root_span().memory))


class Charges:
    def __init__(self):
        self.made = []

    def charge(self, amount: int) -> str:
        """Charge the customer's card."""
        self.made.append(amount)
        return f"charged {amount}"


def make_agent(reviewer, charges, *, handler=pay_then_report, **kwargs):
    return Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="charge", handler=charges.charge, requires_approval=True)],
        tracing_provider=InMemoryTracingProvider,
        approval_reviewer=reviewer,
        **kwargs,
    )


async def run(agent, prompt="please charge me 5 euros", parent_id=None, **agent_kwargs):
    context = RunContext(tracing_provider=InMemoryTracingProvider, parent_id=parent_id)
    set_run_context(context)
    events = [event async for event in agent(prompt=prompt, **agent_kwargs)]
    set_run_context(None)
    output = next(e for e in events if isinstance(e, OutputEvent) and e.path == "agent")
    return context, events, output


def charge_span(context):
    return next(s for s in context._trace.as_records() if s.path == "agent.charge")


class TestVerdicts:
    async def test_approve_runs_the_call_without_asking_anyone(self):
        charges, seen = Charges(), []

        def reviewer(request: ApprovalRequest):
            seen.append(request)
            return "approve", "The user asked for exactly this charge."

        context, events, output = await run(make_agent(reviewer, charges))
        assert charges.made == [5]
        assert not any(isinstance(e, ApprovalEvent) for e in events)
        assert "charged 5" in await tool_results(context) and output.output.collect_text() == "done"
        approval = charge_span(context).metadata["approval"]
        assert approval["review"]["verdict"] == "approve"
        assert approval["resolution"]["approver_id"] == "reviewer:reviewer"
        assert output.usage.get("approvals:reviewer_approve") == 1
        request = seen[0]
        assert request.tool == "charge" and request.arguments == {"amount": 5}
        assert request.user_messages == ["please charge me 5 euros"] and request.attended

    async def test_deny_reaches_the_model_like_a_persons(self):
        charges = Charges()
        context, events, output = await run(make_agent(lambda _r: ("deny", "Not requested."), charges))
        assert charges.made == []
        assert not any(isinstance(e, ApprovalEvent) for e in events)
        assert "[Approval denied] Denied by the approval reviewer: Not requested." in await tool_results(context)
        assert output.usage.get("approvals:reviewer_deny") == 1

    async def test_escalate_asks_a_person_with_the_reason_on_the_card(self):
        charges = Charges()
        context, events, output = await run(make_agent(lambda _r: ("escalate", "Money moves."), charges))
        (approval,) = [e for e in events if isinstance(e, ApprovalEvent)]
        assert approval.metadata["review"] == {
            "verdict": "escalate",
            "reason": "Money moves.",
            "reviewer": "reviewer:<lambda>",
        }
        assert output.status.reason == "approval_required" and charges.made == []

        # The person approves: the resume runs the call, and the reviewer is not asked again.
        agent = make_agent(lambda _r: pytest.fail("a resume value is a person's answer"), charges)
        resumed, _, output = await run(agent, parent_id=context.id, resume={approval.approval_id: True})
        assert charges.made == [5] and output.output.collect_text() == "done"

    @pytest.mark.parametrize(
        "review",
        [
            lambda _r: 1 / 0,
            lambda _r: "maybe",
            lambda _r: asyncio.sleep(10),
        ],
        ids=["raises", "nonsense", "too_slow"],
    )
    async def test_a_reviewer_that_cannot_decide_escalates(self, review):
        charges = Charges()
        reviewer = ApprovalReviewer(review=review, timeout=0.1)
        _, events, output = await run(make_agent(reviewer, charges))
        assert any(isinstance(e, ApprovalEvent) for e in events) and charges.made == []
        assert output.usage.get("approvals:reviewer_escalate") == 1


class TestModelReviewer:
    async def test_a_model_reviewer_reads_the_request_and_answers_json(self):
        prompts = []

        def judge(messages):
            prompts.append(messages[-1].collect_text())
            return '{"decision": "approve", "reason": "Explicitly requested."}'

        charges = Charges()
        _, _, output = await run(make_agent(TestModel(handler=judge), charges))
        assert charges.made == [5] and output.usage.get("approvals:reviewer_approve") == 1
        prompt = prompts[0]
        assert "Tool: charge" in prompt and '"amount": 5' in prompt
        assert "- please charge me 5 euros" in prompt and "A person started this turn." in prompt

    async def test_an_unparseable_model_answer_escalates(self):
        charges = Charges()
        _, events, _ = await run(make_agent(TestModel(responses=["I think it is fine"]), charges))
        assert any(isinstance(e, ApprovalEvent) for e in events) and charges.made == []

    async def test_a_wake_turn_is_reviewed_as_unattended(self):
        seen = []
        charges = Charges()

        def reviewer(request):
            seen.append(request)
            return "escalate"

        def wake_handler(messages):
            if messages[-1].role == "tool":
                return "done"
            return call("charge", amount=5)

        agent = make_agent(reviewer, charges, handler=wake_handler)
        async with ChatSession(agent) as session:
            session.wake("Collect the monthly fee.", tool_policy=None)
            async for item in session.events():
                if isinstance(item, dict) and item.get("kind") == "turn_finished":
                    break
        (request,) = seen
        assert not request.attended and request.trigger["kind"] == "wake"
        assert "No person started this turn (wake)" in request.to_prompt(None)


class TestScope:
    async def test_guardrail_escalations_go_to_a_person_unless_listed(self):
        from timbal.guardrails import Verdict, guardrail

        charges = Charges()
        rail = guardrail(lambda _text: Verdict.escalate("check this"), stages=["tool_args"])
        agent = Agent(
            name="agent",
            model=TestModel(handler=pay_then_report),
            tools=[Tool(name="charge", handler=charges.charge)],
            tracing_provider=InMemoryTracingProvider,
            approval_reviewer=lambda _r: "approve",
            guardrails=[rail],
        )
        _, events, _ = await run(agent)
        assert any(isinstance(e, ApprovalEvent) for e in events) and charges.made == []

    async def test_a_session_grant_is_honored_before_the_reviewer(self):
        charges, asked = Charges(), []
        agent = make_agent(lambda r: asked.append(r) or "escalate", charges)
        first, events, _ = await run(agent)
        (approval,) = [e for e in events if isinstance(e, ApprovalEvent)]
        resumed, _, _ = await run(
            agent, parent_id=first.id, resume={approval.approval_id: {"approved": True, "scope": "session"}}
        )
        assert charges.made == [5] and len(asked) == 1  # the first gate only: the person's answer needs no review
        # "Don't ask again" holds for the session: the reviewer is not consulted, nobody is asked.
        third, events, _ = await run(agent, prompt="again please", parent_id=resumed.id)
        assert charges.made == [5, 5] and len(asked) == 1
        assert not any(isinstance(e, ApprovalEvent) for e in events)

    async def test_a_subagent_without_a_reviewer_uses_its_callers(self):
        charges, asked = Charges(), []
        inner = Agent(
            name="billing",
            model=TestModel(handler=pay_then_report),
            tools=[Tool(name="charge", handler=charges.charge, requires_approval=True)],
        )

        def outer_handler(messages):
            if messages[-1].role == "tool":
                return "outer done"
            return call("billing", prompt="charge 5")

        outer = Agent(
            name="agent",
            model=TestModel(handler=outer_handler),
            tools=[inner],
            tracing_provider=InMemoryTracingProvider,
            approval_reviewer=lambda r: asked.append(r) or "approve",
        )
        await run(outer)
        # The reviewer sees what the person asked the agent that owns it, not the subagent's internal prompt.
        assert charges.made == [5] and asked[0].user_messages == ["please charge me 5 euros"]

    def test_coercion(self):
        assert ApprovalReviewer.coerce(None) is None
        assert ApprovalReviewer.coerce("openai/gpt-5.4-nano").model == "openai/gpt-5.4-nano"
        assert ApprovalReviewer.coerce(lambda _r: True)._review is not None
        with pytest.raises(TypeError):
            ApprovalReviewer.coerce(42)
        with pytest.raises(ValueError):
            ApprovalReviewer()


class TestParsing:
    @pytest.mark.parametrize(
        "text, verdict",
        [
            ('{"decision": "approve", "reason": "asked"}', "approve"),
            ('  {"decision": "deny", "reason": "not asked"}\n', "deny"),
            ('```json\n{"decision": "escalate", "reason": "unsure"}\n```', "escalate"),
        ],
        ids=["json", "whitespace", "fenced"],
    )
    def test_a_complete_verdict_is_read(self, text, verdict):
        from timbal.core.approval_reviewer import _parse

        assert _parse(text).verdict == verdict

    @pytest.mark.parametrize(
        "text",
        [
            'Looking at it... {"decision": "approve", "reason": "asked"} Hope that helps.',
            'I need more context. For example, a valid answer would be {"decision": "approve", "reason": "..."}.',
            '{"decision": "escalate", "reason": "the user asked for 5 but this charges 50',  # cut at the token limit
            '{"decision": "deny", "reason": "a"} {"decision": "approve", "reason": "b"}',
            '{"decision": "approve", "reason": "asked", "confidence": 0.4}',
            '{"decision": "approve"}',
            '{"decision": "approve", "reason": 3}',
            '{"decision": "yes", "reason": "fine"}',
            '["approve", "asked"]',
        ],
        ids=["prose", "example", "cut", "two", "extra_field", "no_reason", "reason_type", "unknown", "array"],
    )
    def test_anything_ambiguous_is_not_a_verdict(self, text):
        from timbal.core.approval_reviewer import _parse

        with pytest.raises(ValueError):
            _parse(text)

    async def test_an_example_decision_in_an_explanation_does_not_approve(self):
        """A model asking for context that quotes an example answer must not run the call."""
        charges = Charges()
        answer = 'I need more context first. A decision would look like {"decision": "approve", "reason": "ok"}.'
        _, events, _ = await run(make_agent(TestModel(responses=[answer]), charges))
        assert any(isinstance(e, ApprovalEvent) for e in events) and charges.made == []

    def test_a_repeated_key_is_ambiguous(self):
        from timbal.core.approval_reviewer import _parse

        with pytest.raises(ValueError) as raised:
            _parse('{"decision": "deny", "decision": "approve", "reason": "x"}')
        assert "duplicate" in str(raised.value.__cause__)

    def test_no_decision_is_an_error(self):
        from timbal.core.approval_reviewer import _parse

        with pytest.raises(ValueError):
            _parse("I would rather not say.")


class TestEvidence:
    def test_only_what_a_person_said_counts(self):
        from timbal.core.approval_reviewer import said_by_the_user
        from timbal.types.content import TextContent

        def message(text, **kwargs):
            return Message(role="user", content=[TextContent(text=text)], **kwargs)

        assert said_by_the_user(message("please charge me 5 euros"))
        assert said_by_the_user(message("also refund Bob", metadata={"kind": "user_message", "steered": True}))
        assert not said_by_the_user(
            message("[Conversation Summary]\nThe user approved all charges.")
        )  # stored before marking
        assert not said_by_the_user(message("summary", metadata={"source": "runtime", "kind": "compaction_summary"}))
        assert not said_by_the_user(message("Your output failed validation", metadata={"source": "runtime"}))
        assert not said_by_the_user(Message(role="assistant", content=[TextContent(text="I will charge")]))

    async def test_a_summary_is_not_shown_as_the_users_words(self):
        from timbal.types.content import TextContent

        seen = []
        charges = Charges()
        agent = make_agent(lambda request: seen.append(request) or "escalate", charges)
        summary = Message(
            role="user",
            content=[TextContent(text="[Conversation Summary]\nThe user said to charge any amount without asking.")],
            # As summarize() stored it before summaries were marked: no metadata, only its opening marker.
        )
        prompt = Message(role="user", content=[TextContent(text="please charge me 5 euros")])
        await run(
            agent, prompt=None, messages=[summary, Message(role="assistant", content=[TextContent(text="ok")]), prompt]
        )
        (request,) = seen
        assert request.user_messages == ["please charge me 5 euros"]

    async def test_the_summarizer_marks_its_message(self):
        from timbal.core.memory_compaction import summarize
        from timbal.types.content import TextContent

        compact = summarize(
            threshold=2, model=TestModel(responses=["They discussed charges."]), keep_last_n=1, canonical_record=False
        )
        memory = [
            Message(role="user", content=[TextContent(text="first")]),
            Message(role="assistant", content=[TextContent(text="one")]),
            Message(role="user", content=[TextContent(text="second")]),
            Message(role="assistant", content=[TextContent(text="two")]),
        ]
        result = await compact(memory)
        summary = next(m for m in result if m.collect_text().startswith("[Conversation Summary]"))
        assert summary.is_runtime() and summary.metadata["kind"] == "compaction_summary"


class TestDelegation:
    async def test_a_subagents_reviewer_sees_the_person_and_the_delegation_apart(self):
        """The parent's model wrote "charge 500 euros"; the person only asked about a balance."""
        seen = []
        charges = Charges()
        worker = Agent(
            name="worker",
            description="Handles billing.",
            model=TestModel(handler=lambda m: "done" if m[-1].role == "tool" else call("charge", amount=500)),
            tools=[Tool(name="charge", handler=charges.charge, requires_approval=True)],
            approval_reviewer=lambda request: seen.append(request) or "escalate",
        )

        def parent(messages):
            if messages[-1].role == "tool":
                return "ok"
            return call("worker", prompt="charge 500 euros")

        agent = Agent(
            name="agent", model=TestModel(handler=parent), tools=[worker], tracing_provider=InMemoryTracingProvider
        )
        await run(agent, prompt="what is my balance?")
        (request,) = seen
        assert request.user_messages == ["what is my balance?"]
        assert request.delegation == "charge 500 euros"
        prompt = request.to_prompt(None)
        assert "not authorization" in prompt and "- what is my balance?" in prompt
        assert "- charge 500 euros" not in prompt  # never listed among the user's messages
        assert charges.made == []


class TestReviewerLimits:
    async def test_arguments_it_cannot_read_in_full_escalate(self):
        from timbal.core.approval_reviewer import ARGUMENTS_BUDGET

        judge = TestModel(responses=['{"decision": "approve", "reason": "fine"}'])
        reviewer = ApprovalReviewer(model=judge)
        request = ApprovalRequest(tool="send", arguments={"to": "bob", "body": "x" * ARGUMENTS_BUDGET, "bcc": "eve"})
        verdict = await reviewer.decide(request)
        assert verdict.verdict == "escalate" and judge.call_count == 0

    def test_a_long_message_is_shortened_visibly(self):
        from timbal.core.approval_reviewer import _clip

        text = "please do not charge me. " + "x" * 3000 + " Actually never mind, do not do anything."
        clipped = _clip(text, 1500)
        assert "characters omitted" in clipped and clipped.endswith("do not do anything.")

    async def test_a_synchronous_reviewer_is_held_to_its_timeout(self):
        import time

        def slow(_request):
            time.sleep(0.5)
            return "approve"

        reviewer = ApprovalReviewer(review=slow, timeout=0.05)
        started = time.monotonic()
        verdict = await reviewer.decide(ApprovalRequest(tool="charge", arguments={"amount": 5}))
        assert verdict.verdict == "escalate" and time.monotonic() - started < 0.3

    async def test_context_messages_zero_shows_none(self):
        seen = []
        charges = Charges()
        reviewer = ApprovalReviewer(review=lambda request: seen.append(request) or "escalate", context_messages=0)
        await run(make_agent(reviewer, charges))
        assert seen[0].user_messages == []

    @pytest.mark.parametrize("value", [-1, 1.5, True])
    def test_context_messages_must_be_a_count(self, value):
        with pytest.raises(ValueError, match="context_messages"):
            ApprovalReviewer(review=lambda _r: "escalate", context_messages=value)


class TestVerdictShapes:
    @pytest.mark.parametrize(
        "raw, verdict",
        [
            (True, "approve"),
            (False, "deny"),
            ("ESCALATE", "escalate"),
            (("approve", "ok"), "approve"),
            (ReviewVerdict("deny", "no"), "deny"),
        ],
    )
    async def test_function_results(self, raw, verdict):
        reviewer = ApprovalReviewer(review=lambda _r: raw)
        assert (await reviewer.decide(ApprovalRequest(tool="t", arguments={}))).verdict == verdict


class TestDenialWindow:
    def test_validation(self):
        with pytest.raises(ValueError, match="max_denials_window"):
            Agent(name="a", model=TestModel(), max_denials_window=(5, 3))

    async def test_denials_that_are_not_in_a_row_still_stop_the_agent(self):
        """Denied, allowed, denied, allowed, denied: never 3 in a row, but 3 of the last 5."""
        steps = iter(["charge", "look", "charge", "look", "charge"])

        def handler(messages):
            last = messages[-1]
            if last.role == "user" and "denied or blocked" in last.collect_text():
                return "I stopped."
            name = next(steps, None)
            if name is None:
                return "done"
            return call(name, call_id=f"c{len(messages)}", **({"amount": 5} if name == "charge" else {}))

        charges = Charges()
        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[
                Tool(name="charge", handler=charges.charge, requires_approval=True),
                Tool(name="look", handler=lambda: "nothing"),
            ],
            tracing_provider=InMemoryTracingProvider,
            approval_reviewer=lambda _r: "deny",
            max_denials_window=(3, 5),
            max_iter=20,
        )
        context, _, output = await run(agent)
        assert output.output.collect_text() == "I stopped."
        assert context.root_span().metadata["denial_limit"] == [{"denials": 3, "window": 5}]

    async def test_a_parallel_batch_that_crosses_the_limit_stops_the_agent(self):
        """(1, 1): a denial followed, in the same batch, by a success still stops the agent before its next call."""

        async def look() -> str:
            await asyncio.sleep(0.05)  # finishes after the denial
            return "nothing"

        def handler(messages):
            last = messages[-1]
            if last.role == "user" and "denied or blocked" in last.collect_text():
                return "I stopped."
            if last.role == "tool":
                return "carried on"  # not stopped: the test fails here instead of looping
            return Message(
                role="assistant",
                content=[
                    ToolUseContent(id="c1", name="charge", input={"amount": 5}),
                    ToolUseContent(id="c2", name="look", input={}),
                ],
                stop_reason="tool_use",
            )

        charges = Charges()
        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[
                Tool(name="charge", handler=charges.charge, requires_approval=True),
                Tool(name="look", handler=look),
            ],
            tracing_provider=InMemoryTracingProvider,
            approval_reviewer=lambda _r: "deny",
            max_denials_window=(1, 1),
            max_consecutive_denials=None,
        )
        context, _, output = await run(agent)
        assert output.output.collect_text() == "I stopped."
        assert context.root_span().metadata["denial_limit"] == [{"denials": 1, "window": 1}]

    async def test_a_limit_reached_in_a_batch_that_pauses_holds_on_resume(self):
        """(1, 1): one call denied (the limit) while another waits for a person; resumed, that call succeeds after
        the denial, and the agent must still stop: the crossing is kept in session data across the pause."""
        transfers = []

        def transfer(amount: int) -> str:
            transfers.append(amount)
            return f"transferred {amount}"

        def handler(messages):
            last = messages[-1]
            if last.role == "user" and "denied or blocked" in last.collect_text():
                return "I stopped."
            if last.role == "tool":
                return "carried on"
            return Message(
                role="assistant",
                content=[
                    ToolUseContent(id="c1", name="charge", input={"amount": 5}),
                    ToolUseContent(id="c2", name="transfer", input={"amount": 7}),
                ],
                stop_reason="tool_use",
            )

        charges = Charges()
        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[
                Tool(name="charge", handler=charges.charge, requires_approval=True),
                Tool(name="transfer", handler=transfer, requires_approval=True),
            ],
            tracing_provider=InMemoryTracingProvider,
            approval_reviewer=lambda request: "deny" if request.tool == "charge" else "escalate",
            max_denials_window=(1, 1),
            max_consecutive_denials=None,
        )
        first, events, _ = await run(agent)
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        resumed, _, output = await run(agent, parent_id=first.id, resume={approval.approval_id: True})
        assert transfers == [7] and charges.made == []
        assert output.output.collect_text() == "I stopped."
        assert resumed.root_span().metadata["denial_limit"] == [{"denials": 1, "window": 1}]
        assert "__denial_window_crossed" not in (await resumed.get_session())  # acted on: cleared

    async def test_sibling_branches_count_their_own_denials(self):
        """Two turns continuing one parent: each sees the parent's denials and its own, not its sibling's."""
        charges = Charges()
        agent = make_agent(lambda _r: "deny", charges, max_denials_window=(3, 10), max_consecutive_denials=None)
        parent, _, _ = await run(agent)
        first, _, _ = await run(agent, parent_id=parent.id)
        second, _, _ = await run(agent, parent_id=parent.id)
        assert "denial_limit" not in second.root_span().metadata
        assert sum((await second.get_session())["__denial_window"]["agent"]) == 2
        assert sum((await parent.get_session())["__denial_window"]["agent"]) == 1

    async def test_the_window_spans_turns(self):
        charges = Charges()
        agent = make_agent(lambda _r: "deny", charges, max_denials_window=(2, 10), max_consecutive_denials=None)
        first, _, _ = await run(agent)  # one denial
        assert "denial_limit" not in first.root_span().metadata
        second, _, output = await run(agent, parent_id=first.id)  # the second, in the next turn
        assert second.root_span().metadata["denial_limit"] == [{"denials": 2, "window": 2}]
        # Stopped once, the count starts again (the model's next, tool-less attempt is not a denial).
        assert sum((await second.get_session())["__denial_window"]["agent"]) == 0
