"""Tests verifying that a Workflow with a single Agent node behaves identically to a standalone Agent.

The codegen pipeline (get_flow) wraps a standalone Agent into a single-node flow.
These tests ensure the two execution paths produce equivalent results.
"""

import pytest
from timbal import Agent, Workflow
from timbal.state import get_run_context
from timbal.types.events import OutputEvent
from timbal.types.events.start import StartEvent
from timbal.types.message import Message

from ..conftest import assert_has_output_event, assert_no_errors, skip_if_agent_error


# ==============================================================================
# Shared tools
# ==============================================================================

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def agent():
    return Agent(
        name="test_agent",
        model="openai/gpt-4o-mini",
        tools=[add, multiply],
        system_prompt="You are a helpful calculator. Use your tools for any math.",
    )


@pytest.fixture
def workflow_with_agent(agent):
    wf = Workflow(name="test_workflow")
    wf.step(agent)
    return wf


# ==============================================================================
# Tests
# ==============================================================================

class TestOutputEquivalence:
    """The final output of a workflow-wrapped agent should match a standalone agent."""

    @pytest.mark.asyncio
    async def test_both_return_message(self, agent, workflow_with_agent):
        """Both should return a Message as the final output."""
        prompt = Message.validate({"role": "user", "content": "What is 3 + 4?"})

        agent_output = await agent(prompt=prompt).collect()
        skip_if_agent_error(agent_output, "standalone_agent")
        assert isinstance(agent_output.output, Message)
        assert agent_output.output.role == "assistant"

        workflow_output = await workflow_with_agent(prompt=prompt).collect()
        skip_if_agent_error(workflow_output, "workflow_agent")
        assert isinstance(workflow_output.output, Message)
        assert workflow_output.output.role == "assistant"

    @pytest.mark.asyncio
    async def test_tool_calling_works_in_both(self, agent, workflow_with_agent):
        """Both should be able to call tools and use the results."""
        prompt = Message.validate({"role": "user", "content": "What is 6 multiplied by 7? Use the multiply tool."})

        agent_output = await agent(prompt=prompt).collect()
        skip_if_agent_error(agent_output, "standalone_agent_tool")
        content = str(agent_output.output.content).lower()
        assert "42" in content

        workflow_output = await workflow_with_agent(prompt=prompt).collect()
        skip_if_agent_error(workflow_output, "workflow_agent_tool")
        content = str(workflow_output.output.content).lower()
        assert "42" in content

    @pytest.mark.asyncio
    async def test_no_errors_in_either(self, agent, workflow_with_agent):
        """Neither execution path should produce errors for a simple request."""
        prompt = Message.validate({"role": "user", "content": "Hello!"})

        agent_output = await agent(prompt=prompt).collect()
        assert_has_output_event(agent_output)
        assert_no_errors(agent_output)

        workflow_output = await workflow_with_agent(prompt=prompt).collect()
        assert_has_output_event(workflow_output)
        assert_no_errors(workflow_output)

    @pytest.mark.asyncio
    async def test_output_event_status_matches(self, agent, workflow_with_agent):
        """Both should produce OutputEvents with successful status."""
        prompt = Message.validate({"role": "user", "content": "Say hello."})

        agent_output = await agent(prompt=prompt).collect()
        skip_if_agent_error(agent_output, "agent_status")
        workflow_output = await workflow_with_agent(prompt=prompt).collect()
        skip_if_agent_error(workflow_output, "workflow_status")

        assert agent_output.status.code == workflow_output.status.code


class TestEventStreaming:
    """Event streams should be structurally similar."""

    @pytest.mark.asyncio
    async def test_both_emit_start_and_output_events(self, agent, workflow_with_agent):
        """Both should emit START and OUTPUT events for the agent."""
        prompt = Message.validate({"role": "user", "content": "What is 2 + 2?"})

        agent_events = []
        async for event in agent(prompt=prompt):
            agent_events.append(event)

        workflow_events = []
        async for event in workflow_with_agent(prompt=prompt):
            workflow_events.append(event)

        # Agent should have START and OUTPUT events
        agent_start_events = [e for e in agent_events if isinstance(e, StartEvent)]
        agent_output_events = [e for e in agent_events if isinstance(e, OutputEvent)]
        assert len(agent_start_events) >= 1
        assert len(agent_output_events) >= 1

        # Workflow should also have START and OUTPUT events (from the nested agent)
        wf_start_events = [e for e in workflow_events if isinstance(e, StartEvent)]
        wf_output_events = [e for e in workflow_events if isinstance(e, OutputEvent)]
        assert len(wf_start_events) >= 1
        assert len(wf_output_events) >= 1

    @pytest.mark.asyncio
    async def test_workflow_agent_events_have_nested_paths(self):
        """Workflow events should have paths nested under the workflow name."""
        prompt = Message.validate({"role": "user", "content": "Hello!"})

        # Standalone agent: paths start with agent name
        standalone = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
        )
        agent_events = []
        async for event in standalone(prompt=prompt):
            agent_events.append(event)

        for event in agent_events:
            assert event.path.startswith("test_agent"), f"Agent event path should start with 'test_agent', got '{event.path}'"

        # Workflow-wrapped agent: paths start with workflow name
        wrapped = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
        )
        wf = Workflow(name="test_workflow")
        wf.step(wrapped)

        wf_events = []
        async for event in wf(prompt=prompt):
            wf_events.append(event)

        for event in wf_events:
            assert event.path.startswith("test_workflow"), (
                f"Workflow event path should start with 'test_workflow', got '{event.path}'"
            )

        # Agent-level events should be nested under workflow
        agent_events_in_wf = [e for e in wf_events if "test_agent" in e.path]
        assert len(agent_events_in_wf) > 0, "Should have agent events nested under workflow"
        for event in agent_events_in_wf:
            assert event.path.startswith("test_workflow.test_agent"), (
                f"Agent event in workflow should start with 'test_workflow.test_agent', got '{event.path}'"
            )

    @pytest.mark.asyncio
    async def test_same_event_types_emitted(self, agent, workflow_with_agent):
        """Both should emit the same set of event types (ignoring path differences)."""
        prompt = Message.validate({"role": "user", "content": "Hello!"})

        agent_event_types = set()
        async for event in agent(prompt=prompt):
            agent_event_types.add(event.type)

        wf_event_types = set()
        async for event in workflow_with_agent(prompt=prompt):
            wf_event_types.add(event.type)

        # The workflow should emit at least all the event types the agent does
        assert agent_event_types.issubset(wf_event_types), (
            f"Agent event types {agent_event_types} not a subset of workflow event types {wf_event_types}"
        )


class TestParameterPassing:
    """Parameters should flow correctly through the workflow to the agent."""

    @pytest.mark.asyncio
    async def test_prompt_param_passes_through(self):
        """Workflow should pass the prompt parameter to the agent step."""
        agent = Agent(
            name="param_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You must respond with exactly: RECEIVED",
        )
        wf = Workflow(name="param_wf").step(agent)

        prompt = Message.validate({"role": "user", "content": "Test"})
        output = await wf(prompt=prompt).collect()
        skip_if_agent_error(output, "param_passing")

        assert isinstance(output.output, Message)

    @pytest.mark.asyncio
    async def test_system_prompt_preserved_in_workflow(self):
        """The agent's system_prompt should be preserved when run inside a workflow."""
        agent = Agent(
            name="sys_prompt_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You must always respond with exactly the word 'PINEAPPLE' and nothing else.",
        )

        # Standalone
        prompt = Message.validate({"role": "user", "content": "Reply now."})
        standalone_output = await agent(prompt=prompt).collect()
        skip_if_agent_error(standalone_output, "standalone_sys_prompt")

        # Wrapped in workflow
        wf = Workflow(name="sys_wf").step(agent)
        wf_output = await wf(prompt=prompt).collect()
        skip_if_agent_error(wf_output, "workflow_sys_prompt")

        # Both should follow the system prompt
        assert "pineapple" in str(standalone_output.output.content).lower()
        assert "pineapple" in str(wf_output.output.content).lower()

    @pytest.mark.asyncio
    async def test_workflow_can_override_agent_params(self):
        """Workflow kwargs should override agent step defaults."""
        agent = Agent(
            name="override_agent",
            model="openai/gpt-4o-mini",
        )
        wf = Workflow(name="override_wf").step(agent)

        # Override system_prompt from workflow level
        prompt = Message.validate({"role": "user", "content": "Reply now."})
        output = await wf(
            prompt=prompt,
            system_prompt="You must always respond with exactly the word 'OVERRIDE' and nothing else.",
        ).collect()
        skip_if_agent_error(output, "workflow_override")

        assert "override" in str(output.output.content).lower()


class TestTracing:
    """Tracing should be consistent with nesting expectations."""

    @pytest.mark.asyncio
    async def test_standalone_agent_trace_structure(self):
        """Standalone agent should have a flat trace: agent -> llm."""
        agent = Agent(
            name="trace_agent",
            model="openai/gpt-4o-mini",
        )
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        output = await agent(prompt=prompt).collect()
        skip_if_agent_error(output, "agent_trace")

        records = get_run_context()._trace.as_records()
        paths = [r.path for r in records]
        assert "trace_agent" in paths
        assert "trace_agent.llm" in paths

    @pytest.mark.asyncio
    async def test_workflow_agent_trace_structure(self):
        """Workflow with single agent should have: workflow -> agent -> llm."""
        agent = Agent(
            name="trace_agent",
            model="openai/gpt-4o-mini",
        )
        wf = Workflow(name="trace_wf").step(agent)

        prompt = Message.validate({"role": "user", "content": "Hello!"})
        output = await wf(prompt=prompt).collect()
        skip_if_agent_error(output, "workflow_trace")

        records = get_run_context()._trace.as_records()
        paths = [r.path for r in records]
        assert "trace_wf" in paths
        assert "trace_wf.trace_agent" in paths
        assert "trace_wf.trace_agent.llm" in paths

    @pytest.mark.asyncio
    async def test_tool_trace_paths_nested_correctly(self):
        """Tool calls in both modes should have correct nested paths in the trace."""
        agent = Agent(
            name="tool_trace_agent",
            model="openai/gpt-4o-mini",
            tools=[add],
            system_prompt="Always use the add tool for any question. Call add(1, 2).",
        )

        # Standalone
        prompt = Message.validate({"role": "user", "content": "Add 1 and 2."})
        output = await agent(prompt=prompt).collect()
        skip_if_agent_error(output, "tool_trace_standalone")

        records = get_run_context()._trace.as_records()
        tool_paths = [r.path for r in records if "add" in r.path]
        if tool_paths:
            assert any(p == "tool_trace_agent.add" for p in tool_paths)

        # Workflow
        wf = Workflow(name="tool_trace_wf").step(agent)
        output = await wf(prompt=prompt).collect()
        skip_if_agent_error(output, "tool_trace_workflow")

        records = get_run_context()._trace.as_records()
        tool_paths = [r.path for r in records if "add" in r.path]
        if tool_paths:
            assert any(p == "tool_trace_wf.tool_trace_agent.add" for p in tool_paths)


    @pytest.mark.asyncio
    async def test_skill_tool_trace_paths_in_workflow(self, tmp_path):
        """Skill tools inside a workflow-wrapped agent should have correct nested trace paths."""
        # Create a skill directory structure following the docs pattern:
        # skills/payment_processing/SKILL.md, tools/process_refund.py, fraud_detection.md
        skill_dir = tmp_path / "skills" / "payment_processing"
        tools_dir = skill_dir / "tools"
        tools_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: payment_processing\n"
            "description: Complete payment processing including payments, refunds, and fraud detection\n"
            "---\n\n"
            "## Overview\n"
            "There are different operations for e-commerce orders.\n\n"
            "### Payment Policy\n"
            "- Payments must be made within 30 days of order placement\n"
            "- See `fraud_detection.md` for security guidelines\n\n"
            "### Usage Guidelines\n"
            "Always verify the order exists before processing payments or refunds.\n"
        )
        (skill_dir / "fraud_detection.md").write_text(
            "# Fraud Detection\n\nFlag orders over $10,000 for manual review.\n"
        )
        (tools_dir / "process_refund.py").write_text(
            "from timbal import Tool\n\n"
            'async def process_refund(order_id: str, amount: float, reason: str) -> str:\n'
            '    """Process a customer refund."""\n'
            '    return f"Refund of ${amount} processed for order {order_id}"\n\n'
            "process_refund_tool = Tool(\n"
            '    name="process_refund",\n'
            '    description="Process a refund for a customer order",\n'
            "    handler=process_refund,\n"
            ")\n"
        )
        (tools_dir / "check_status.py").write_text(
            "from timbal import Tool\n\n"
            'async def check_status(order_id: str) -> str:\n'
            '    """Check the status of an order."""\n'
            '    return f"Order {order_id} status: delivered"\n\n'
            "check_status_tool = Tool(\n"
            '    name="check_status",\n'
            '    description="Check the status of a customer order",\n'
            "    handler=check_status,\n"
            ")\n"
        )

        agent = Agent(
            name="support_agent",
            model="openai/gpt-4o-mini",
            skills_path=str(tmp_path / "skills"),
            system_prompt=(
                "You are a customer support agent. "
                "First call read_skill with name='payment_processing' to load the skill, "
                "then use check_status to check the order and process_refund to issue the refund."
            ),
        )

        # Standalone: verify trace paths for skill tools.
        prompt = Message.validate({
            "role": "user",
            "content": "Check order ORD-123 and refund $50 for a damaged item.",
        })
        output = await agent(prompt=prompt).collect()
        skip_if_agent_error(output, "skill_standalone")

        records = get_run_context()._trace.as_records()
        read_skill_paths = [r.path for r in records if "read_skill" in r.path]
        if read_skill_paths:
            assert any(p == "support_agent.read_skill" for p in read_skill_paths)

        refund_paths = [r.path for r in records if "process_refund" in r.path]
        if refund_paths:
            assert any(p == "support_agent.process_refund" for p in refund_paths)

        status_paths = [r.path for r in records if "check_status" in r.path]
        if status_paths:
            assert any(p == "support_agent.check_status" for p in status_paths)

        # Workflow-wrapped: paths should be prefixed with workflow name.
        wf = Workflow(name="support_wf").step(agent)
        output = await wf(prompt=prompt).collect()
        skip_if_agent_error(output, "skill_workflow")

        records = get_run_context()._trace.as_records()
        read_skill_paths = [r.path for r in records if "read_skill" in r.path]
        if read_skill_paths:
            assert any(p == "support_wf.support_agent.read_skill" for p in read_skill_paths)

        refund_paths = [r.path for r in records if "process_refund" in r.path]
        if refund_paths:
            assert any(p == "support_wf.support_agent.process_refund" for p in refund_paths)

        status_paths = [r.path for r in records if "check_status" in r.path]
        if status_paths:
            assert any(p == "support_wf.support_agent.check_status" for p in status_paths)


class TestEdgeCases:
    """Edge cases and error handling should be consistent."""

    @pytest.mark.asyncio
    async def test_agent_without_tools(self):
        """Both modes should work with an agent that has no tools."""
        agent = Agent(
            name="no_tools_agent",
            model="openai/gpt-4o-mini",
        )
        wf = Workflow(name="no_tools_wf").step(agent)

        prompt = Message.validate({"role": "user", "content": "Hello!"})

        agent_output = await agent(prompt=prompt).collect()
        assert_has_output_event(agent_output)
        assert_no_errors(agent_output)

        wf_output = await wf(prompt=prompt).collect()
        assert_has_output_event(wf_output)
        assert_no_errors(wf_output)

    @pytest.mark.asyncio
    async def test_messages_param_works_in_both(self):
        """Both modes should accept the messages parameter instead of prompt."""
        agent = Agent(
            name="messages_agent",
            model="openai/gpt-4o-mini",
        )
        wf = Workflow(name="messages_wf").step(agent)

        messages = [
            Message.validate({"role": "user", "content": "Hello!"}),
        ]

        agent_output = await agent(messages=messages).collect()
        assert_has_output_event(agent_output)
        assert_no_errors(agent_output)

        wf_output = await wf(messages=messages).collect()
        assert_has_output_event(wf_output)
        assert_no_errors(wf_output)

    @pytest.mark.asyncio
    async def test_multi_tool_agent_in_workflow(self):
        """An agent with multiple tools should work the same in both modes."""
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        agent = Agent(
            name="multi_agent",
            model="openai/gpt-4o-mini",
            tools=[add, multiply, greet],
            system_prompt="You are a helpful assistant with math and greeting tools.",
        )
        wf = Workflow(name="multi_wf").step(agent)

        prompt = Message.validate({"role": "user", "content": "Greet Alice"})

        agent_output = await agent(prompt=prompt).collect()
        skip_if_agent_error(agent_output, "multi_tool_standalone")
        assert "alice" in str(agent_output.output.content).lower()

        wf_output = await wf(prompt=prompt).collect()
        skip_if_agent_error(wf_output, "multi_tool_workflow")
        assert "alice" in str(wf_output.output.content).lower()

    @pytest.mark.asyncio
    async def test_max_iter_respected_in_workflow(self):
        """max_iter should be respected when the agent runs inside a workflow."""
        def loop_tool(x: str) -> str:
            return f"call me again with: {x}"

        agent = Agent(
            name="iter_agent",
            model="openai/gpt-4o-mini",
            tools=[loop_tool],
            max_iter=2,
        )
        wf = Workflow(name="iter_wf").step(agent)

        prompt = Message.validate({"role": "user", "content": "Keep calling loop_tool"})

        # Both should complete without hanging
        agent_output = await agent(prompt=prompt).collect()
        assert isinstance(agent_output, OutputEvent)
        assert isinstance(agent_output.output, Message)

        wf_output = await wf(prompt=prompt).collect()
        assert isinstance(wf_output, OutputEvent)
        assert isinstance(wf_output.output, Message)
