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
    return apply_operation(ws, "remove_edge", **kwargs)


class TestRemoveOrderingEdge:
    def test_remove_depends_on(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        output = _run(ws, source="agent_a", target="agent_b")
        assert "depends_on" not in output

    def test_remove_one_from_multiple_depends_on(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
        agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        workflow.step(agent_c, depends_on=["agent_a", "agent_b"])
        """)
        output = _run(ws, source="agent_a", target="agent_c")
        normalized = " ".join(output.split())
        assert '"agent_b"' in normalized
        assert '"agent_a"' not in normalized.split("depends_on")[1] if "depends_on" in normalized else True


class TestRemoveDataFlowEdge:
    def test_remove_param_lambda(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """)
        output = _run(ws, source="agent_a", target="agent_b")
        assert "step_span" not in output


class TestRemoveConditionalEdge:
    def test_remove_when_referencing_source(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, when=lambda: get_run_context().step_span("agent_a").output is not None)
        """)
        output = _run(ws, source="agent_a", target="agent_b")
        assert "when" not in output


class TestValidation:
    def test_rejects_non_workflow(self, tmp_path):
        (tmp_path / "agent.py").write_text(textwrap.dedent("""\
        from timbal import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """))
        (tmp_path / "timbal.yaml").write_text('fqn: "agent.py::agent"\n')
        with pytest.raises(ValueError, match="Workflow"):
            apply_operation(tmp_path, "remove_edge", source="a", target="b")
