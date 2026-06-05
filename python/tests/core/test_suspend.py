"""Tests for the suspend()/resume= human-in-the-loop substrate.

suspend() generalizes the approval gate: instead of resuming with a bool, the
run resumes with an arbitrary value supplied via ``resume={interaction_id: ...}``.
"""

from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, suspend
from timbal.tools import ask_user as builtin_ask_user
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

    async def test_confirm_returns_bool(self):
        tool = Tool(handler=builtin_confirm)
        first = await tool(action="delete prod").collect()
        assert first.status.reason == "input_required"
        assert first.output["kind"] == "confirm"

        interaction_id = first.output["suspension_id"]
        result = await tool(action="delete prod", parent_id=first.run_id, resume={interaction_id: True}).collect()
        assert result.status.code == "success"
        assert result.output is True
