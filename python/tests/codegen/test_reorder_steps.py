import textwrap
from pathlib import Path

import pytest
from timbal.codegen.transformers import apply_operation, reorder_step_calls

WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def wf_workspace(tmp_path):
    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


def _step_order(code: str) -> list[str]:
    """Extract the order of step names from workflow.step() calls."""
    import re

    return re.findall(r"workflow\.step\(\s*(\w+)", code)


class TestReorderStepCalls:
    def test_no_reorder_when_already_ordered(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        result = reorder_step_calls(code, "workflow")
        assert _step_order(result) == ["agent_a", "agent_b"]

    def test_reorder_when_dependency_after_dependent(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b, depends_on=["agent_a"])
        workflow.step(agent_a)
        """)
        result = reorder_step_calls(code, "workflow")
        assert _step_order(result) == ["agent_a", "agent_b"]

    def test_reorder_with_step_span_dependency(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        workflow.step(agent_a)
        """)
        result = reorder_step_calls(code, "workflow")
        assert _step_order(result) == ["agent_a", "agent_b"]

    def test_reorder_chain_of_three(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
        agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_c, depends_on=["agent_b"])
        workflow.step(agent_b, depends_on=["agent_a"])
        workflow.step(agent_a)
        """)
        result = reorder_step_calls(code, "workflow")
        assert _step_order(result) == ["agent_a", "agent_b", "agent_c"]

    def test_preserves_order_for_independent_steps(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
        agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b)
        workflow.step(agent_a)
        workflow.step(agent_c, depends_on=["agent_a"])
        """)
        result = reorder_step_calls(code, "workflow")
        order = _step_order(result)
        # agent_c must come after agent_a, but agent_b is independent
        assert order.index("agent_a") < order.index("agent_c")
        # agent_b and agent_a have no constraint — original order preserved
        assert order.index("agent_b") < order.index("agent_a")

    def test_single_step_no_reorder(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        """)
        result = reorder_step_calls(code, "workflow")
        assert result == code

    def test_non_step_code_not_affected(self):
        code = textwrap.dedent("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b, depends_on=["agent_a"])
        workflow.step(agent_a)
        """)
        result = reorder_step_calls(code, "workflow")
        # Variable assignments should still be before the step calls
        assert result.index("agent_a = Agent") < result.index("workflow.step")


class TestReorderViaAddEdge:
    """Test that add-edge triggers reordering end-to-end."""

    def test_add_edge_reorders_steps(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b)
        workflow.step(agent_a)
        """)
        output = apply_operation(ws, "add_edge", source="agent_a", target="agent_b", when=None)
        assert _step_order(output) == ["agent_a", "agent_b"]


class TestReorderViaSetParam:
    """Test that set-param type=map triggers reordering end-to-end."""

    def test_set_param_map_reorders_steps(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b)
        workflow.step(agent_a)
        """)
        output = apply_operation(
            ws, "set_param",
            target="agent_b", name="prompt", param_type="map",
            source="agent_a", key=None, value=None,
        )
        assert _step_order(output) == ["agent_a", "agent_b"]

    def test_set_param_value_does_not_reorder(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_b)
        workflow.step(agent_a)
        """)
        output = apply_operation(
            ws, "set_param",
            target="agent_b", name="prompt", param_type="value",
            source=None, key=None, value='"hello"',
        )
        # No dependency introduced — order should stay as-is
        assert _step_order(output) == ["agent_b", "agent_a"]
