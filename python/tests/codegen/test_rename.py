import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'
AGENT_YAML = 'fqn: "agent.py::agent"\n'


@pytest.fixture
def wf_workspace(tmp_path):
    """Write a workflow source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


@pytest.fixture
def agent_workspace(tmp_path):
    """Write an agent source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "agent.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(AGENT_YAML)
        return tmp_path

    return _write


def _run_dry(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen rename with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "rename", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_expect_error(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen rename with --dry-run, expecting failure. Returns stderr."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "rename", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    return result.stderr


class TestRenameStep:
    def test_rename_step_variable_and_name(self, wf_workspace):
        """Rename a step: variable, name kwarg, and .step() reference."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "agent_a", "--to", "agent_b")
        assert 'agent_b = Agent(name="agent_b"' in output
        assert "workflow.step(agent_b)" in output
        assert "agent_a" not in output

    def test_rename_step_updates_depends_on(self, wf_workspace):
        """Rename a step updates depends_on string references."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        output = _run_dry(ws, "agent_a", "--to", "first_agent")
        assert 'first_agent = Agent(name="first_agent"' in output
        assert "workflow.step(first_agent)" in output
        assert 'depends_on=["first_agent"]' in output
        assert '"agent_a"' not in output

    def test_rename_step_updates_step_span(self, wf_workspace):
        """Rename a step updates step_span string references."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """)
        output = _run_dry(ws, "agent_a", "--to", "first_agent")
        assert 'step_span("first_agent")' in output
        assert 'step_span("agent_a")' not in output


class TestRenameTool:
    def test_rename_tool_on_agent(self, agent_workspace):
        """Rename a tool: variable, name kwarg, and tools list reference."""
        ws = agent_workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        output = _run_dry(ws, "web_search", "--to", "my_search")
        assert "my_search = WebSearch(" in output
        assert "tools=[my_search]" in output
        assert "web_search" not in output


class TestRenameEdgeCases:
    def test_rejects_rename_entry_point(self, wf_workspace):
        """Cannot rename the entry point variable."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        stderr = _run_dry_expect_error(ws, "workflow", "--to", "my_workflow")
        assert "entry point" in stderr.lower()

    def test_rejects_nonexistent_name(self, wf_workspace):
        """Cannot rename a name that doesn't exist."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        stderr = _run_dry_expect_error(ws, "nonexistent", "--to", "something")
        assert "found" in stderr.lower()

    def test_rename_preserves_other_config(self, wf_workspace):
        """Renaming preserves all other constructor kwargs."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", system_prompt="Be helpful.")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "agent_a", "--to", "agent_b")
        assert 'name="agent_b"' in output
        assert 'model="openai/gpt-4o-mini"' in output
        assert 'system_prompt="Be helpful."' in output
