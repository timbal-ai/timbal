"""Tests for the suspend()/resume= human-in-the-loop substrate.

suspend() generalizes the approval gate: instead of resuming with a bool, the
run resumes with an arbitrary value supplied via ``resume={interaction_id: ...}``.
"""

from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, suspend
from timbal.tools import ask_user as builtin_ask_user
from timbal.tools import ask_user_multi as builtin_ask_user_multi
from timbal.tools import confirm as builtin_confirm
from timbal.types.events import InteractionEvent, OutputEvent
from timbal.types.message import Message

# --- direct tool suspend/resume --------------------------------------------


def ask_color() -> str:
    """Suspend asking for a color."""
    return suspend({"question": "favorite color?"}, kind="ask_user")


class TestToolSuspend:
    async def test_tool_suspends_and_emits_interaction_event(self):
        tool = Tool(handler=ask_color)

        events = [e async for e in tool()]
        interactions = [e for e in events if isinstance(e, InteractionEvent)]
        outputs = [e for e in events if isinstance(e, OutputEvent)]

        assert len(interactions) == 1
        ev = interactions[0]
        assert ev.kind == "ask_user"
        assert ev.payload == {"question": "favorite color?"}
        assert ev.runnable_name == "ask_color"

        assert outputs[-1].status.code == "cancelled"
        assert outputs[-1].status.reason == "input_required"
        assert outputs[-1].output["suspension_id"] == ev.interaction_id

    async def test_tool_resumes_with_value(self):
        tool = Tool(handler=ask_color)

        # First pass: capture the interaction id.
        first = await tool().collect()
        assert first.status.reason == "input_required"
        interaction_id = first.output["suspension_id"]

        # Resume: suspend() returns the supplied value, handler completes.
        result = await tool(parent_id=first.run_id, resume={interaction_id: "blue"}).collect()
        assert result.status.code == "success"
        assert result.output == "blue"

    async def test_unrecognized_resume_value_is_ignored(self):
        tool = Tool(handler=ask_color)
        # A resume value that matches no suspend() id: the tool still suspends.
        result = await tool(resume={"does-not-exist": "x"}).collect()
        assert result.status.reason == "input_required"


# --- agent loop suspend/resume ---------------------------------------------


def ask_user(question: str) -> str:
    """Ask the user a question."""
    return suspend({"question": question}, kind="ask_user")


class TestAgentSuspend:
    async def test_agent_pauses_on_tool_suspend_then_resumes(self):
        # Model: first turn calls ask_user; after resume, replies with the answer.
        tool_call = Message.validate({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "ask_user", "input": {"question": "name?"}}],
        })
        model = TestModel(responses=[tool_call, "Nice to meet you, Ada!"])
        agent = Agent(name="assistant", model=model, tools=[ask_user])

        # First call: agent calls the tool, which suspends.
        pending: list[InteractionEvent] = []
        final = None
        async for event in agent(prompt="hi"):
            if isinstance(event, InteractionEvent):
                pending.append(event)
            if isinstance(event, OutputEvent) and event.path == "assistant":
                final = event

        assert len(pending) == 1
        assert pending[0].kind == "ask_user"
        assert final is not None
        assert final.status.reason == "input_required"

        interaction_id = pending[0].interaction_id

        # Resume with the user's answer.
        result = await agent(
            prompt="hi",
            parent_id=final.run_id,
            resume={interaction_id: "Ada"},
        ).collect()

        assert result.status.code == "success"
        assert result.output.collect_text() == "Nice to meet you, Ada!"

    async def test_agent_pauses_on_two_parallel_tool_suspends_then_resumes(self):
        # Model fires two ask_user tool_uses in a single turn; the agent runs them
        # concurrently, both suspend, and we resume with both answers at once.
        tool_calls = Message.validate({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_a", "name": "ask_user", "input": {"question": "first name?"}},
                {"type": "tool_use", "id": "call_b", "name": "ask_user", "input": {"question": "last name?"}},
            ],
        })
        model = TestModel(responses=[tool_calls, "Hello, Ada Lovelace!"])
        agent = Agent(name="assistant", model=model, tools=[ask_user])

        pending: list[InteractionEvent] = []
        final = None
        async for event in agent(prompt="hi"):
            if isinstance(event, InteractionEvent):
                pending.append(event)
            if isinstance(event, OutputEvent) and event.path == "assistant":
                final = event

        # Both questions surfaced before the run paused.
        assert len(pending) == 2
        assert {e.payload["question"] for e in pending} == {"first name?", "last name?"}
        # Distinct payloads -> distinct interaction ids.
        assert len({e.interaction_id for e in pending}) == 2
        # Each interaction correlates back to its originating LLM tool_use block.
        assert {e.tool_call_id for e in pending} == {"call_a", "call_b"}

        assert final is not None
        assert final.status.reason == "input_required"

        by_question = {e.payload["question"]: e.interaction_id for e in pending}
        resume = {
            by_question["first name?"]: "Ada",
            by_question["last name?"]: "Lovelace",
        }

        result = await agent(
            prompt="hi",
            parent_id=final.run_id,
            resume=resume,
        ).collect()

        assert result.status.code == "success"
        assert result.output.collect_text() == "Hello, Ada Lovelace!"

    async def test_agent_partial_resume_keeps_other_question_pending(self):
        # Resuming only one of two parallel suspends must leave the other pending
        # (its tool_use stays unresolved and re-fires on the next resume).
        tool_calls = Message.validate({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_a", "name": "ask_user", "input": {"question": "first name?"}},
                {"type": "tool_use", "id": "call_b", "name": "ask_user", "input": {"question": "last name?"}},
            ],
        })
        model = TestModel(responses=[tool_calls, "Hello, Ada Lovelace!"])
        agent = Agent(name="assistant", model=model, tools=[ask_user])

        first = await agent(prompt="hi").collect()
        first_pending = {p["payload"]["question"]: p["interaction_id"] for p in first.metadata["pending_interactions"]}

        # Resume only the first question.
        second = await agent(
            prompt="hi",
            parent_id=first.run_id,
            resume={first_pending["first name?"]: "Ada"},
        ).collect()

        assert second.status.reason == "input_required"
        still_pending = {p["payload"]["question"] for p in second.metadata["pending_interactions"]}
        assert still_pending == {"last name?"}

        # Resume the remaining one -> run completes.
        second_pending = {p["payload"]["question"]: p["interaction_id"] for p in second.metadata["pending_interactions"]}
        third = await agent(
            prompt="hi",
            parent_id=second.run_id,
            resume={second_pending["last name?"]: "Lovelace"},
        ).collect()

        assert third.status.code == "success"
        assert third.output.collect_text() == "Hello, Ada Lovelace!"

    async def test_agent_identical_parallel_questions_get_distinct_ids(self):
        # Two tool_use blocks calling the same tool with the SAME payload must not
        # collide: the LLM tool_call_id is folded into the suspension id so each
        # gets its own answer instead of silently sharing one.
        tool_calls = Message.validate({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_a", "name": "ask_user", "input": {"question": "value?"}},
                {"type": "tool_use", "id": "call_b", "name": "ask_user", "input": {"question": "value?"}},
            ],
        })
        model = TestModel(responses=[tool_calls, "done"])
        agent = Agent(name="assistant", model=model, tools=[ask_user])

        pending: list[InteractionEvent] = []
        final = None
        async for event in agent(prompt="hi"):
            if isinstance(event, InteractionEvent):
                pending.append(event)
            if isinstance(event, OutputEvent) and event.path == "assistant":
                final = event

        # Identical payloads, yet two distinct interactions keyed by tool_call_id.
        assert len(pending) == 2
        assert {e.payload["question"] for e in pending} == {"value?"}
        assert len({e.interaction_id for e in pending}) == 2

        by_call = {e.tool_call_id: e.interaction_id for e in pending}
        result = await agent(
            prompt="hi",
            parent_id=final.run_id,
            resume={by_call["call_a"]: "X", by_call["call_b"]: "Y"},
        ).collect()
        assert result.status.code == "success"


# --- workflow step suspend/resume ------------------------------------------


def needs_input() -> str:
    """A workflow step that suspends for input."""
    return suspend({"question": "proceed value?"}, kind="ask_user")


def echo(value: str) -> str:
    return f"got: {value}"


class TestWorkflowSuspend:
    async def test_workflow_step_suspends_and_resumes(self):
        workflow = (
            Workflow(name="wf")
            .step(needs_input)
            .step(echo, value=lambda: get_run_context().step_span("needs_input").output)
        )

        first = await workflow().collect()
        assert first.status.reason == "input_required"

        # Find the interaction id from the suspended step output.
        interaction_id = None
        async for event in workflow():
            if isinstance(event, InteractionEvent):
                interaction_id = event.interaction_id
                break
        assert interaction_id is not None

        result = await workflow(parent_id=first.run_id, resume={interaction_id: "hello"}).collect()
        assert result.status.code == "success"
        assert result.output == "got: hello"

    async def test_workflow_step_cancel_aborts_run(self):
        from timbal.types.approval import Cancel

        ran: list[str] = []

        def downstream(value: str) -> str:
            ran.append(value)
            return f"got: {value}"

        workflow = (
            Workflow(name="wf")
            .step(needs_input)
            .step(downstream, value=lambda: get_run_context().step_span("needs_input").output)
        )

        first = await workflow().collect()
        interaction_id = first.output["suspension_id"] if isinstance(first.output, dict) else None
        if interaction_id is None:
            # Workflow output wraps step outputs; pull the pending id from metadata.
            interaction_id = first.metadata["pending_interactions"][0]["interaction_id"]

        result = await workflow(
            parent_id=first.run_id,
            resume={interaction_id: Cancel(reason="abort the pipeline")},
        ).collect()
        assert result.status.code == "cancelled"
        assert result.status.reason == "cancelled"
        assert ran == [], "downstream step must not run after a cancelled step"


# --- coexistence with approvals --------------------------------------------


class TestApprovalStillWorks:
    async def test_approval_gate_unaffected(self):
        def refund(amount: int) -> str:
            return f"refunded {amount}"

        tool = Tool(
            handler=refund,
            requires_approval=True,
            approval_prompt="approve refund?",
        )
        first = await tool(amount=100).collect()
        assert first.status.reason == "approval_required"
        approval_id = first.output["approval_id"]

        result = await tool(amount=100, parent_id=first.run_id, resume={approval_id: True}).collect()
        assert result.status.code == "success"
        assert result.output == "refunded 100"


# --- enumeration / DX helpers ----------------------------------------------


class TestPendingInteractionsDX:
    async def test_metadata_lists_pending_interactions(self):
        tool = Tool(handler=ask_color)
        result = await tool().collect()

        pending = (result.metadata or {}).get("pending_interactions")
        assert pending is not None
        assert len(pending) == 1
        assert pending[0]["interaction_id"] == result.output["suspension_id"]
        assert pending[0]["kind"] == "ask_user"
        assert pending[0]["payload"] == {"question": "favorite color?"}
        # tool_call_id/response_schema must be surfaced in collect() metadata too.
        assert pending[0]["tool_call_id"] is None  # direct call, no LLM tool_use
        assert "response_schema" in pending[0]

    async def test_run_context_enumerates_pending_interactions(self):
        captured: dict = {}

        def ask_two() -> str:
            captured["ctx"] = get_run_context()
            return suspend({"question": "x?"}, kind="ask_user")

        tool = Tool(handler=ask_two)
        result = await tool().collect()

        pending = captured["ctx"].pending_interactions()
        assert len(pending) == 1
        assert pending[0]["interaction_id"] == result.output["suspension_id"]


# --- built-in interaction tools --------------------------------------------


class TestBuiltinInteractionTools:
    async def test_ask_user_round_trip(self):
        tool_call = Message.validate({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "ask_user",
                "input": {"question": "db?"},
            }],
        })
        model = TestModel(responses=[tool_call, "Using postgres."])
        agent = Agent(name="assistant", model=model, tools=[builtin_ask_user])

        first = await agent(prompt="set up the db").collect()
        assert first.status.reason == "input_required"
        interaction_id = first.metadata["pending_interactions"][0]["interaction_id"]

        result = await agent(
            prompt="set up the db",
            parent_id=first.run_id,
            resume={interaction_id: "postgres"},
        ).collect()
        assert result.status.code == "success"
        assert result.output.collect_text() == "Using postgres."

    async def test_ask_user_multi_round_trip(self):
        tool_call = Message.validate({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "ask_user_multi",
                "input": {
                    "question": "Which integrations?",
                    "options": ["slack", "gmail", "jira"],
                },
            }],
        })
        model = TestModel(responses=[tool_call, "Enabled slack and gmail."])
        agent = Agent(name="assistant", model=model, tools=[builtin_ask_user_multi])

        first = await agent(prompt="set up integrations").collect()
        assert first.status.reason == "input_required"
        pending = first.metadata["pending_interactions"][0]
        assert pending["kind"] == "ask_user_multi"
        assert pending["response_schema"] == {
            "type": "array",
            "items": {"type": "string", "enum": ["slack", "gmail", "jira"]},
            "minItems": 1,
        }

        result = await agent(
            prompt="set up integrations",
            parent_id=first.run_id,
            resume={pending["interaction_id"]: ["slack", "gmail"]},
        ).collect()
        assert result.status.code == "success"
        assert result.output.collect_text() == "Enabled slack and gmail."

    async def test_confirm_returns_bool(self):
        tool = Tool(handler=builtin_confirm)
        first = await tool(action="delete prod").collect()
        assert first.status.reason == "input_required"
        assert first.output["kind"] == "confirm"

        interaction_id = first.output["suspension_id"]
        result = await tool(action="delete prod", parent_id=first.run_id, resume={interaction_id: True}).collect()
        assert result.status.code == "success"
        assert result.output is True


# --- response_schema (client-side validation of the resume value) -----------


def ask_typed() -> dict:
    """Ask with a declared response schema."""
    return suspend(
        {"question": "how old?"},
        kind="ask_user",
        response_schema={"type": "integer", "minimum": 0},
    )


class TestInteractionResponseSchema:
    async def test_schema_surfaces_on_event_and_pending(self):
        tool = Tool(handler=ask_typed)
        events = [e async for e in tool()]
        ev = next(e for e in events if isinstance(e, InteractionEvent))
        assert ev.response_schema == {"type": "integer", "minimum": 0}

        ctx = get_run_context()
        pending = ctx.pending_interactions()
        assert pending[0]["response_schema"] == {"type": "integer", "minimum": 0}

    async def test_schema_defaults_to_none(self):
        tool = Tool(handler=ask_color)
        events = [e async for e in tool()]
        ev = next(e for e in events if isinstance(e, InteractionEvent))
        assert ev.response_schema is None


# --- tool_call_id correlation ----------------------------------------------


class TestInteractionToolCallId:
    async def test_interaction_carries_originating_tool_call_id(self):
        tool_call = Message.validate({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "ask_user",
                "input": {"question": "name?"},
            }],
        })
        model = TestModel(responses=[tool_call, "hi"])
        agent = Agent(name="assistant", model=model, tools=[ask_user])

        pending: list[InteractionEvent] = []
        async for event in agent(prompt="hi"):
            if isinstance(event, InteractionEvent):
                pending.append(event)
        assert pending[0].tool_call_id == "toolu_abc123"

    async def test_direct_call_has_no_tool_call_id(self):
        tool = Tool(handler=ask_color)
        events = [e async for e in tool()]
        ev = next(e for e in events if isinstance(e, InteractionEvent))
        assert ev.tool_call_id is None


# --- Cancel on the resume channel aborts the run ----------------------------


class TestSuspendCancel:
    async def test_cancel_terminates_tool_run(self):
        from timbal.types.approval import Cancel

        tool = Tool(handler=ask_color)
        first = await tool().collect()
        interaction_id = first.output["suspension_id"]

        result = await tool(
            parent_id=first.run_id,
            resume={interaction_id: Cancel(reason="user closed dialog")},
        ).collect()
        assert result.status.code == "cancelled"
        assert result.status.reason == "cancelled"
        assert result.status.message == "user closed dialog"

    async def test_cancel_via_json_tagged_dict(self):
        """The HTTP/JSON form (no Python object) must be coerced to a Cancel."""
        tool = Tool(handler=ask_color)
        first = await tool().collect()
        interaction_id = first.output["suspension_id"]

        result = await tool(
            parent_id=first.run_id,
            resume={interaction_id: {"type": "timbal.cancel", "reason": "closed"}},
        ).collect()
        assert result.status.code == "cancelled"
        assert result.status.reason == "cancelled"
        assert result.status.message == "closed"

    async def test_cancel_terminates_agent_run_without_feedback(self):
        from timbal.types.approval import Cancel

        tool_call = Message.validate({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "ask_user", "input": {"question": "name?"}}],
        })
        # Second response would only be reached if the agent kept going; cancel
        # must stop the run before it is consumed.
        model = TestModel(responses=[tool_call, "should never be returned"])
        agent = Agent(name="assistant", model=model, tools=[ask_user])

        first = await agent(prompt="hi").collect()
        interaction_id = first.metadata["pending_interactions"][0]["interaction_id"]

        result = await agent(
            prompt="hi",
            parent_id=first.run_id,
            resume={interaction_id: Cancel()},
        ).collect()
        assert result.status.code == "cancelled"
        assert result.status.reason == "cancelled"
