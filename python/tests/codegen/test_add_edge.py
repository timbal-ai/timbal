import re
import textwrap
from pathlib import Path

import pytest
from timbal.codegen.transformers import apply_operation

WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def wf_workspace(tmp_path):
    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


def _run(ws: Path, **kwargs) -> str:
    return apply_operation(ws, "add_edge", **kwargs)


class TestOrderingEdge:
    def test_pure_ordering(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run(ws, source="agent_a", target="agent_b", when=None)
        normalized = " ".join(output.split())
        assert 'depends_on=["agent_a"]' in normalized

    def test_merge_depends_on(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
        agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        workflow.step(agent_c, depends_on=["agent_a"])
        """)
        output = _run(ws, source="agent_b", target="agent_c", when=None)
        normalized = " ".join(output.split())
        assert '"agent_a"' in normalized
        assert '"agent_b"' in normalized

    def test_deduplicate_depends_on(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        output = _run(ws, source="agent_a", target="agent_b", when=None)
        # Extract the depends_on list — should contain agent_a only once.
        match = re.search(r'depends_on=\[([^\]]*)\]', " ".join(output.split()))
        assert match is not None
        assert match.group(1).count('"agent_a"') == 1


    def test_source_variable_name_resolved_to_runtime_name(self, wf_workspace):
        """When source is a variable name that differs from the step's runtime name,
        depends_on should use the runtime name (the name= kwarg)."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="Agent 1", model="openai/gpt-4o-mini")
        agent_b = Agent(name="Agent 2", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        # Pass the variable name as source — codegen should resolve it to "Agent 1"
        output = _run(ws, source="agent_a", target="agent_b", when=None)
        normalized = " ".join(output.split())
        assert 'depends_on=["Agent 1"]' in normalized

    def test_target_variable_name_resolved_to_runtime_name(self, wf_workspace):
        """When target is a variable name that differs from the step's runtime name,
        the target step should still be found and updated."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="Agent 1", model="openai/gpt-4o-mini")
        agent_b = Agent(name="Agent 2", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        # Pass the runtime name as source, variable name as target
        output = _run(ws, source="Agent 1", target="agent_b", when=None)
        normalized = " ".join(output.split())
        assert 'depends_on=["Agent 1"]' in normalized


class TestConditionalEdge:
    def test_when_condition(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run(
            ws, source="agent_a", target="agent_b",
            when="lambda: True",
        )
        assert "when=lambda: True" in output
        # Should also add depends_on
        normalized = " ".join(output.split())
        assert 'depends_on=["agent_a"]' in normalized

    def test_when_with_get_run_context(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run(
            ws, source="agent_a", target="agent_b",
            when='lambda: get_run_context().step_span("agent_a").output is not None',
        )
        assert "when=lambda:" in output
        assert "from timbal.state import get_run_context" in output


class TestValidation:
    def test_rejects_non_workflow(self, tmp_path):
        (tmp_path / "agent.py").write_text(textwrap.dedent("""\
        from timbal import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """))
        (tmp_path / "timbal.yaml").write_text('fqn: "agent.py::agent"\n')
        with pytest.raises(ValueError, match="Workflow"):
            apply_operation(tmp_path, "add_edge", source="a", target="b", when=None)
