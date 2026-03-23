import json
import subprocess
import textwrap
from pathlib import Path

import pytest

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'
WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def workspace(tmp_path):
    def _write(source: str) -> Path:
        (tmp_path / "agent.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
        return tmp_path

    return _write


@pytest.fixture
def wf_workspace(tmp_path):
    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


def _run_dry(workspace_path: Path, *cli_args: str) -> str:
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "set-position", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_fail(workspace_path: Path, *cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "set-position", *cli_args],
        capture_output=True,
        text=True,
    )


def _exec_agent(code: str) -> dict:
    ns = {}
    exec(code, ns)
    return ns


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


class TestAgentPosition:
    def test_set_position_no_metadata(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws, "--x", "100", "--y", "200")
        ns = _exec_agent(output)
        assert ns["agent"].metadata["position"] == {"x": 100.0, "y": 200.0}

    def test_set_position_existing_metadata(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", metadata={"custom": "value"})
        """)
        output = _run_dry(ws, "--x", "50", "--y", "75")
        ns = _exec_agent(output)
        assert ns["agent"].metadata["position"] == {"x": 50.0, "y": 75.0}
        assert ns["agent"].metadata["custom"] == "value"

    def test_update_existing_position(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", metadata={"position": {"x": 0, "y": 0}})
        """)
        output = _run_dry(ws, "--x", "300", "--y", "400")
        ns = _exec_agent(output)
        assert ns["agent"].metadata["position"] == {"x": 300.0, "y": 400.0}

    def test_preserves_other_kwargs(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", system_prompt="Be helpful.")
        """)
        output = _run_dry(ws, "--x", "10", "--y", "20")
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == "Be helpful."
        assert ns["agent"].metadata["position"] == {"x": 10.0, "y": 20.0}

    def test_integer_coords(self, workspace):
        """Integer-like floats (100.0) work fine."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws, "--x", "0", "--y", "0")
        ns = _exec_agent(output)
        assert ns["agent"].metadata["position"] == {"x": 0.0, "y": 0.0}

    def test_rejects_name_for_agent(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        result = _run_dry_fail(ws, "--name", "something", "--x", "0", "--y", "0")
        assert result.returncode != 0
        assert "--name is only supported" in result.stderr


# ---------------------------------------------------------------------------
# Workflow step
# ---------------------------------------------------------------------------


class TestWorkflowStepPosition:
    def test_set_step_position(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "--name", "agent_a", "--x", "150", "--y", "250")
        ns = _exec_agent(output)
        assert ns["agent_a"].metadata["position"] == {"x": 150.0, "y": 250.0}

    def test_set_step_position_preserves_metadata(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"custom": "data"})

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "--name", "agent_a", "--x", "50", "--y", "60")
        ns = _exec_agent(output)
        assert ns["agent_a"].metadata["position"] == {"x": 50.0, "y": 60.0}
        assert ns["agent_a"].metadata["custom"] == "data"

    def test_update_step_position(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 0, "y": 0}})

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "--name", "agent_a", "--x", "999", "--y", "888")
        ns = _exec_agent(output)
        assert ns["agent_a"].metadata["position"] == {"x": 999.0, "y": 888.0}

    def test_requires_name_for_workflow(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        result = _run_dry_fail(ws, "--x", "0", "--y", "0")
        assert result.returncode != 0
        assert "requires a step --name" in result.stderr

    def test_multiple_steps_only_target_affected(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        output = _run_dry(ws, "--name", "agent_a", "--x", "100", "--y", "200")
        ns = _exec_agent(output)
        assert ns["agent_a"].metadata["position"] == {"x": 100.0, "y": 200.0}
        assert "position" not in ns["agent_b"].metadata


# ---------------------------------------------------------------------------
# get-flow integration
# ---------------------------------------------------------------------------


class TestGetFlowPosition:
    def test_position_as_top_level_key(self, workspace):
        """Position appears as a top-level node key next to id, type, data."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", metadata={"position": {"x": 42, "y": 84}})
        """)
        from timbal.codegen.flow import get_flow

        flow = get_flow(ws)
        node = flow["nodes"][0]
        assert node["position"] == {"x": 42, "y": 84}

    def test_position_defaults_to_origin(self, workspace):
        """Without position in metadata, defaults to {x: 0, y: 0}."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        from timbal.codegen.flow import get_flow

        flow = get_flow(ws)
        node = flow["nodes"][0]
        assert node["position"] == {"x": 0, "y": 0}

    def test_position_in_workflow_flow(self, wf_workspace):
        """Position appears as top-level node key for workflow steps."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 10, "y": 20}})

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        from timbal.codegen.flow import get_flow

        flow = get_flow(ws)
        node = flow["nodes"][0]
        assert node["position"] == {"x": 10, "y": 20}
