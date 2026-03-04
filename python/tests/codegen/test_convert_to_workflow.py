import subprocess
import textwrap
from pathlib import Path

import pytest

TIMBAL_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def workspace(tmp_path):
    """Write a source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
        return tmp_path

    return _write


def _run_dry(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen convert-to-workflow with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "convert-to-workflow", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_expect_error(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen convert-to-workflow with --dry-run, expecting failure. Returns stderr."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "convert-to-workflow", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    return result.stderr


class TestConvertToWorkflow:
    def test_basic_conversion(self, workspace):
        """Convert a simple Agent into a Workflow with one step."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(name="agent_a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws)
        assert "from timbal import Agent" in output
        assert "Workflow" in output
        assert 'agent_a = Agent(name="agent_a"' in output
        assert 'workflow = Workflow(name="workflow")' in output
        assert "workflow.step(agent_a)" in output

    def test_preserves_tools(self, workspace):
        """Agent tools should be preserved on the Agent instance."""
        ws = workspace("""\
        from timbal import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        workflow = Agent(name="agent_a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        output = _run_dry(ws)
        assert "tools=[web_search]" in output
        assert "workflow.step(agent_a)" in output
        assert "web_search = WebSearch()" in output

    def test_custom_workflow_name(self, workspace):
        """--name overrides the Workflow name kwarg."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(name="agent_a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws, "--name", "my_pipeline")
        assert 'Workflow(name="my_pipeline")' in output

    def test_agent_without_name_kwarg(self, workspace):
        """Agent without a name= kwarg should fall back to 'agent' as variable name."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws)
        assert "agent = Agent(" in output
        assert "workflow.step(agent)" in output

    def test_rejects_workflow_entry_point(self, workspace):
        """Should reject a Workflow entry point — it's already a Workflow."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        _run_dry_expect_error(ws)

    def test_name_collision_with_entry_point(self, workspace):
        """When agent name matches the entry point variable, disambiguate with _step suffix."""
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini")
        """)
        ws_yaml = ws / "timbal.yaml"
        ws_yaml.write_text('fqn: "workflow.py::agent"\n')
        output = _run_dry(ws)
        assert 'agent_step = Agent(name="agent"' in output
        assert "agent = Workflow(" in output
        assert "agent.step(agent_step)" in output

    def test_workflow_import_merged(self, workspace):
        """If Agent is already imported from timbal, Workflow import is added."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(name="agent_a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws)
        # Ruff merges imports: "from timbal import Agent, Workflow"
        assert "Agent" in output
        assert "Workflow" in output
        assert "from timbal import" in output
