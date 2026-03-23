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
    """Run codegen add-step with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "add-step", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_expect_error(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen add-step with --dry-run, expecting failure. Returns stderr."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "add-step", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    return result.stderr


class TestAgentStep:
    def test_adds_agent_step(self, workspace):
        """Add a standalone Agent step to a workflow."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "agent_a", "model": "openai/gpt-4o-mini"}')
        assert "from timbal import Agent" in output or "import Agent" in output
        assert 'Agent(name="agent_a"' in output
        assert "workflow.step(agent_a)" in output

    def test_rejects_agent_without_name(self, workspace):
        """Agent steps must have a name."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        _run_dry_expect_error(ws, "--type", "Agent", "--config", '{"model": "openai/gpt-4o-mini"}')

    def test_validates_agent_config_fields(self, workspace):
        """Unknown config fields should be rejected."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        _run_dry_expect_error(
            ws, "--type", "Agent",
            "--config", '{"name": "a", "model": "openai/gpt-4o-mini", "bogus_field": true}',
        )

    def test_step_inserted_after_last_step_call(self, workspace):
        """New step should be appended after existing .step() calls."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "agent_b", "model": "openai/gpt-4o-mini"}')
        a_idx = output.index("workflow.step(agent_a)")
        b_idx = output.index("workflow.step(agent_b)")
        assert b_idx > a_idx

    def test_idempotent_agent_step(self, workspace):
        """Re-adding an existing Agent step updates it rather than duplicating."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "agent_a", "model": "openai/gpt-4o"}')
        assert output.count("workflow.step(agent_a)") == 1
        assert 'model="openai/gpt-4o"' in output


class TestCustomStep:
    def test_adds_custom_function_step(self, workspace):
        """Custom steps wrap the function in a Tool() and add a .step() call."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        definition = "def process(x: str) -> str:\n    return x.upper()"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition)
        assert "from timbal.core import Tool" in output
        assert "def process_fn(x: str) -> str:" in output
        assert 'process = Tool(name="process", handler=process_fn)' in output
        assert "workflow.step(process)" in output

    def test_custom_step_requires_definition(self, workspace):
        """--definition is required for Custom steps."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        _run_dry_expect_error(ws, "--type", "Custom")

    def test_custom_step_idempotent(self, workspace):
        """Re-adding a custom step updates the function body and Tool wrapper."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.core import Tool

        def process_fn(x: str) -> str:
            return x.upper()

        process = Tool(name="process", handler=process_fn)

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        definition = "def process(x: str) -> str:\n    return x.lower()"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition)
        assert "return x.lower()" in output
        assert output.count("workflow.step(process)") == 1
        assert 'process = Tool(name="process", handler=process_fn)' in output


class TestFrameworkToolStep:
    def test_adds_web_search_step(self, workspace):
        """Add a framework tool as a workflow step."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        assert "from timbal.tools import WebSearch" in output
        assert "web_search = WebSearch()" in output
        assert "workflow.step(web_search)" in output


class TestEntryPointValidation:
    def test_rejects_agent_entry_point(self, workspace):
        """add-step should reject an Agent entry point."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        _run_dry_expect_error(ws, "--type", "Agent", "--config", '{"name": "a", "model": "openai/gpt-4o-mini"}')
