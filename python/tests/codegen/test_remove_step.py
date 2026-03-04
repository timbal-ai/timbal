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


def _run_dry(workspace_path: Path, step_name: str) -> str:
    """Run codegen remove-step with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "remove-step", step_name],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_expect_error(workspace_path: Path, step_name: str) -> str:
    """Run codegen remove-step with --dry-run, expecting failure. Returns stderr."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "remove-step", step_name],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    return result.stderr


class TestRemoveAgentStep:
    def test_removes_agent_step(self, workspace):
        """Remove an Agent step from the workflow."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step(agent_a)" not in output
        # Variable and import should be cleaned up by remove_unused_code
        assert "agent_a = Agent" not in output

    def test_removes_one_keeps_other(self, workspace):
        """Remove one step but keep the other intact."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step(agent_a)" not in output
        assert "workflow.step(agent_b)" in output
        assert 'Agent(name="agent_b"' in output

    def test_removes_step_with_kwargs(self, workspace):
        """Remove a step that has kwargs (depends_on, lambdas)."""
        ws = workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """)
        output = _run_dry(ws, "agent_b")
        assert "workflow.step(agent_b" not in output
        assert "workflow.step(agent_a)" in output


class TestRemoveCustomStep:
    def test_removes_custom_function_step(self, workspace):
        """Remove a custom function step and clean up the function definition."""
        ws = workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        output = _run_dry(ws, "process")
        assert "workflow.step(process)" not in output
        # Function should be cleaned up by remove_unused_code
        assert "def process" not in output


class TestRemoveFrameworkToolStep:
    def test_removes_framework_tool_step(self, workspace):
        """Remove a framework tool step and clean up the variable."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.tools import WebSearch

        web_search = WebSearch()

        workflow = Workflow(name="my_workflow")
        workflow.step(web_search)
        """)
        output = _run_dry(ws, "web_search")
        assert "workflow.step(web_search)" not in output
        assert "web_search = WebSearch()" not in output


class TestRemoveEdgeCases:
    def test_noop_when_step_not_found(self, workspace):
        """Removing a step that doesn't exist should be a no-op."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "nonexistent")
        assert "workflow.step(agent_a)" in output

    def test_removes_last_step(self, workspace):
        """Removing the only step should leave a clean workflow."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step" not in output
        assert 'Workflow(name="my_workflow")' in output

    def test_rejects_agent_entry_point(self, workspace):
        """remove-step should reject an Agent entry point."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        _run_dry_expect_error(ws, "some_step")

    def test_remove_then_add_roundtrip(self, workspace):
        """Remove a step then add it back."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        # Remove agent_b (writes to file)
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "remove-step", "agent_b"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"remove failed:\n{result.stderr}"

        # Add it back with --dry-run
        result = subprocess.run(
            [
                "python", "-m", "timbal.codegen", "--path", str(ws), "--dry-run",
                "add-step", "--type", "Agent",
                "--config", '{"name": "agent_b", "model": "openai/gpt-4o-mini"}',
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"add failed:\n{result.stderr}"
        assert "workflow.step(agent_a)" in result.stdout
        assert "workflow.step(agent_b)" in result.stdout
