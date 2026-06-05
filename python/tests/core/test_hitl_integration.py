"""Real-LLM integration tests for human-in-the-loop.

The rest of the HITL suite (test_approval*, test_suspend*) pins the *mechanics*
with a scripted ``TestModel``: pause, persist, resume, redact, claim, enumerate.
What a fake model cannot prove is the part that only emerges when a real model
reasons over the loop:

    1. the model actually *decides* to call a gated / asking tool,
    2. after a denial is fed back as a tool_result, the model *adapts* and the
       agent run completes (instead of looping or crashing),
    3. after a suspend() resume, the supplied value flows back into the handler
       and the model continues coherently.

These tests are marked ``integration`` and are deselected by default
(``-m "not integration"`` in pyproject). Run explicitly with real credentials:

    uv run pytest python/tests/core/test_hitl_integration.py -m integration

Assertions favour structural facts (paused?, event emitted?, handler call
count, value-flow) over brittle text matching, since model output is
nondeterministic. A couple of soft text checks are kept loose on purpose.
"""

import pytest
from timbal import Agent
from timbal.state import suspend
from timbal.types.approval import ApprovalResolution
from timbal.types.events import ApprovalEvent, InteractionEvent, OutputEvent
from timbal.types.message import Message

# A small spread across the three tool-calling dialects most likely to diverge:
# OpenAI function calling, Anthropic tools, and Google function calling.
MODELS = [
    pytest.param("openai/gpt-4o-mini", None, id="openai"),
    pytest.param("anthropic/claude-haiku-4-5", 1024, id="anthropic"),
    pytest.param("google/gemini-3.1-flash-lite", None, id="google"),
]


def _first(events, kind) -> object | None:
    return next((e for e in events if isinstance(e, kind)), None)


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


# ---------------------------------------------------------------------------
# Approval gate — real model decides to call the gated tool
# ---------------------------------------------------------------------------


class TestApprovalGateRealModel:
    @pytest.mark.integration
    @pytest.mark.parametrize("model,max_tokens", MODELS)
    async def test_model_triggers_gate_then_approve_executes_once(self, model, max_tokens):
        calls: list[dict] = []

        def transfer_funds(amount: int, recipient: str) -> str:
            """Transfer money to a recipient."""
            calls.append({"amount": amount, "recipient": recipient})
            return f"Transferred ${amount} to {recipient}."

        from timbal.core.tool import Tool

        tool = Tool(
            handler=transfer_funds,
            requires_approval=True,
            approval_prompt=lambda amount, recipient: f"Approve transfer of ${amount} to {recipient}?",
        )
        agent = Agent(
            name="banker",
            model=model,
            max_tokens=max_tokens,
            tools=[tool],
            system_prompt="You are a banking assistant. To move money you MUST call the transfer_funds tool.",
        )

        prompt = "Please transfer $500 to Alice."
        events1 = [e async for e in agent(prompt=prompt)]
        approval = _first(events1, ApprovalEvent)
        out1 = _final_output(events1)

        assert approval is not None, "real model did not call the gated tool"
        assert out1.status.reason == "approval_required"
        assert calls == [], "handler must not run before approval"

        out2 = await agent(
            prompt=prompt,
            parent_id=out1.run_id,
            resume={approval.approval_id: True},
        ).collect()

        assert out2.status.code == "success", out2.error
        assert len(calls) == 1, "approved handler must execute exactly once"

    @pytest.mark.integration
    @pytest.mark.parametrize("model,max_tokens", MODELS)
    async def test_deny_is_fed_back_and_agent_adapts(self, model, max_tokens):
        """The key real-model behaviour: a denial becomes a tool_result, and the
        model must continue to a normal completion rather than loop/crash."""
        calls: list[dict] = []

        def delete_account(user_id: str) -> str:
            """Permanently delete a user account."""
            calls.append({"user_id": user_id})
            return f"Deleted account {user_id}."

        from timbal.core.tool import Tool

        tool = Tool(
            handler=delete_account,
            requires_approval=True,
            approval_prompt="Approve permanent account deletion?",
        )
        agent = Agent(
            name="admin",
            model=model,
            max_tokens=max_tokens,
            tools=[tool],
            system_prompt="You are an admin assistant. To delete an account you MUST call the delete_account tool.",
        )

        prompt = "Delete the account for user_id 'u_123'."
        events1 = [e async for e in agent(prompt=prompt)]
        approval = _first(events1, ApprovalEvent)
        out1 = _final_output(events1)
        assert approval is not None, "real model did not call the gated tool"
        assert out1.status.reason == "approval_required"

        out2 = await agent(
            prompt=prompt,
            parent_id=out1.run_id,
            resume={
                approval.approval_id: ApprovalResolution(
                    approved=False,
                    reason="Account deletion is not permitted in this session.",
                )
            },
        ).collect()

        # Agent denial must NOT crash and must NOT run the handler; the model
        # gets the denial as a tool_result and produces a final assistant turn.
        assert out2.status.code == "success", out2.error
        assert calls == [], "denied handler must never execute"
        assert isinstance(out2.output, Message)
        assert out2.output.collect_text().strip(), "model should produce a closing message after denial"


# ---------------------------------------------------------------------------
# suspend() — real model asks the user, answer flows back into the handler
# ---------------------------------------------------------------------------


async def _drive_suspends(agent, prompt, answer_for, max_rounds: int = 4) -> tuple[OutputEvent, int]:
    """Run ``agent`` and keep resuming every ``ask_user`` suspend with the value
    returned by ``answer_for(question)`` until the run finishes (or we hit
    ``max_rounds``). Returns the final OutputEvent and the number of questions asked.

    A real model may ask more than one clarifying question, and each distinct
    question is a distinct suspension id (the id is derived from the payload), so
    a single-shot resume is inherently racy. Draining the loop is the robust
    contract: "however many times you ask, we can answer and you finish".
    """
    parent_id = None
    resume = None
    questions = 0
    out: OutputEvent | None = None
    for _ in range(max_rounds):
        kwargs: dict = {"prompt": prompt}
        if parent_id is not None:
            kwargs["parent_id"] = parent_id
        if resume is not None:
            kwargs["resume"] = resume
        events = [e async for e in agent(**kwargs)]
        out = _final_output(events)
        if out.status.reason != "input_required":
            break
        interaction = _first(events, InteractionEvent)
        assert interaction is not None and interaction.kind == "ask_user"
        questions += 1
        parent_id = out.run_id
        resume = {interaction.interaction_id: answer_for(interaction.payload.get("question", ""))}
    return out, questions


class TestSuspendRealModel:
    @pytest.mark.integration
    @pytest.mark.parametrize("model,max_tokens", MODELS)
    async def test_model_asks_user_then_resumes_with_answer(self, model, max_tokens):
        received: list[str] = []

        def ask_user(question: str) -> str:
            """Ask the user a clarifying question. Use ONLY when blocked on a needed detail."""
            answer = suspend({"question": question}, kind="ask_user")
            received.append(answer)  # only runs once suspend() returns on resume
            return answer

        agent = Agent(
            name="assistant",
            model=model,
            max_tokens=max_tokens,
            tools=[ask_user],
            system_prompt=(
                "You help set up databases. The user has NOT said which engine to use, so you "
                "MUST call the ask_user tool to ask (never ask in plain text). "
                "Ask EXACTLY ONE question — the database engine — then proceed and confirm. "
                "Never ask a second question."
            ),
        )

        prompt = "Set up a database for my new app."

        # First pass must pause on input_required and must NOT run the post-suspend body.
        events1 = [e async for e in agent(prompt=prompt)]
        interaction = _first(events1, InteractionEvent)
        out1 = _final_output(events1)
        assert interaction is not None, "real model did not call ask_user"
        assert interaction.kind == "ask_user"
        assert out1.status.reason == "input_required"
        assert received == [], "handler body after suspend() must not run before resume"

        # Drive the whole loop, always answering PostgreSQL. Robust to the model
        # asking one or several clarifying questions.
        out, questions = await _drive_suspends(agent, prompt, answer_for=lambda _q: "PostgreSQL")

        assert questions >= 1
        assert out.status.code == "success", out.error
        assert received and all(a == "PostgreSQL" for a in received), (
            "every resume value must flow back into the handler"
        )
        assert "postgres" in out.output.collect_text().lower()

    @pytest.mark.integration
    @pytest.mark.parametrize("model,max_tokens", MODELS)
    async def test_user_dismisses_question_and_agent_continues(self, model, max_tokens):
        """A user can *dismiss* an ask_user instead of answering. There is no
        special "cancel" primitive — dismissal is just a resume value the handler
        recognises and translates into guidance for the model. The model must then
        adapt (pick a sensible default) and finish, rather than loop forever."""
        outcomes: list[str] = []

        def ask_user(question: str) -> str:
            """Ask the user a clarifying question. Use ONLY when blocked on a needed detail."""
            answer = suspend({"question": question}, kind="ask_user")
            if answer == "__DISMISSED__":
                outcomes.append("dismissed")
                return (
                    "The user declined to answer and wants you to proceed. "
                    "Choose a sensible default yourself and continue."
                )
            outcomes.append("answered")
            return answer

        agent = Agent(
            name="assistant",
            model=model,
            max_tokens=max_tokens,
            tools=[ask_user],
            system_prompt=(
                "You help set up databases. The user has NOT said which engine to use, so you "
                "MUST call the ask_user tool to ask (never ask in plain text). "
                "If the user declines to answer, pick a sensible default database engine and "
                "proceed to confirm the setup — do NOT keep asking."
            ),
        )

        prompt = "Set up a database for my new app."

        # The user dismisses every question instead of answering it.
        out, questions = await _drive_suspends(agent, prompt, answer_for=lambda _q: "__DISMISSED__")

        assert questions >= 1, "real model did not ask anything to dismiss"
        assert out.status.code == "success", out.error
        assert "dismissed" in outcomes, "the dismissal value must reach the handler"
        # The model adapted to the dismissal and produced a closing turn.
        assert isinstance(out.output, Message)
        assert out.output.collect_text().strip(), "model should produce a closing message after dismissal"
