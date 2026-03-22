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
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "remove-step", "--name", step_name],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_expect_error(workspace_path: Path, step_name: str) -> str:
    """Run codegen remove-step with --dry-run, expecting failure. Returns stderr."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "remove-step", "--name", step_name],
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


class TestRemoveStepCleansEdges:
    def test_removes_depends_on_reference(self, workspace):
        """Removing a step should clean up depends_on references in other steps."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step(agent_a)" not in output
        assert "workflow.step(agent_b" in output
        # depends_on should be cleaned up
        assert "depends_on" not in output

    def test_removes_step_span_param_reference(self, workspace):
        """Removing a step should clean up step_span() param references in other steps."""
        ws = workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step(agent_a)" not in output
        assert "workflow.step(agent_b" in output
        # step_span("agent_a") param should be cleaned up
        assert 'step_span("agent_a")' not in output

    def test_removes_when_reference(self, workspace):
        """Removing a step should clean up when= references in other steps."""
        ws = workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"], when=lambda: get_run_context().step_span("agent_a").output.content != "")
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step(agent_a)" not in output
        assert "workflow.step(agent_b" in output
        assert "depends_on" not in output
        assert "when=" not in output

    def test_preserves_other_depends_on(self, workspace):
        """Removing a step should only remove that step from depends_on, keeping others."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
        agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(agent_b)
        workflow.step(agent_c, depends_on=["agent_a", "agent_b"])
        """)
        output = _run_dry(ws, "agent_a")
        assert "workflow.step(agent_a)" not in output
        assert "workflow.step(agent_b)" in output
        assert "workflow.step(agent_c" in output
        # Should keep agent_b in depends_on but remove agent_a
        assert '"agent_b"' in output
        assert '"agent_a"' not in output


class TestRemoveChainedStep:
    """Test removing steps from chained .step() syntax: workflow = Workflow(...).step(A).step(B)."""

    def test_removes_last_chained_step(self, workspace):
        """Remove the last step in a chain."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.state import get_run_context

        def fetch_data() -> list:
            return [1, 2, 3]

        def process(data: list) -> str:
            return str(data)

        workflow = Workflow(name="w").step(fetch_data).step(process, data=lambda: get_run_context().step_span("fetch_data").output)
        """)
        output = _run_dry(ws, "process")
        assert "step(process" not in output
        assert "step(fetch_data)" in output
        assert "def process" not in output

    def test_removes_middle_chained_step(self, workspace):
        """Remove a step from the middle of a chain."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.state import get_run_context

        def step_a() -> str:
            return "a"

        def step_b(x: str) -> str:
            return x + "b"

        def step_c(x: str) -> str:
            return x + "c"

        workflow = (
            Workflow(name="w")
            .step(step_a)
            .step(step_b, x=lambda: get_run_context().step_span("step_a").output)
            .step(step_c, x=lambda: get_run_context().step_span("step_b").output)
        )
        """)
        output = _run_dry(ws, "step_b")
        assert "step(step_a)" in output
        assert "step(step_b" not in output
        assert "step(step_c" in output
        # step_c's reference to step_b should be cleaned up
        assert 'step_span("step_b")' not in output
        # step_b's function should be cleaned up
        assert "def step_b" not in output

    def test_removes_first_chained_step(self, workspace):
        """Remove the first step in a chain."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.state import get_run_context

        def step_a() -> str:
            return "a"

        def step_b(x: str) -> str:
            return x + "b"

        workflow = (
            Workflow(name="w")
            .step(step_a)
            .step(step_b, x=lambda: get_run_context().step_span("step_a").output)
        )
        """)
        output = _run_dry(ws, "step_a")
        assert "step(step_a" not in output
        assert "step(step_b" in output
        # step_b's reference to step_a should be cleaned up
        assert 'step_span("step_a")' not in output
        assert "def step_a" not in output

    def test_removes_chained_step_with_standalone_after(self, workspace):
        """Remove a chained step when there's also a standalone .step() call."""
        ws = workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        def fetch_data() -> list:
            return [1, 2, 3]

        def prettify(data: list) -> str:
            return str(data)

        agent = Agent(name="agent", model="openai/gpt-4o-mini")

        workflow = (
            Workflow(name="w")
            .step(fetch_data)
            .step(prettify, data=lambda: get_run_context().step_span("fetch_data").output)
        )
        workflow.step(agent)
        """)
        output = _run_dry(ws, "prettify")
        assert "step(prettify" not in output
        assert "step(fetch_data)" in output
        assert "workflow.step(agent)" in output
        assert "def prettify" not in output

    def test_removes_only_chained_step(self, workspace):
        """Remove the only step in a chain, leaving bare Workflow(...)."""
        ws = workspace("""\
        from timbal import Workflow

        def step_a() -> str:
            return "a"

        workflow = Workflow(name="w").step(step_a)
        """)
        output = _run_dry(ws, "step_a")
        assert ".step(" not in output
        assert 'Workflow(name="w")' in output

    def test_cleans_edges_in_chained_steps(self, workspace):
        """Removing a step should clean up depends_on in other chained steps."""
        ws = workspace("""\
        from timbal import Workflow

        def step_a() -> str:
            return "a"

        def step_b() -> str:
            return "b"

        def step_c() -> str:
            return "c"

        workflow = (
            Workflow(name="w")
            .step(step_a)
            .step(step_b)
            .step(step_c, depends_on=["step_a", "step_b"])
        )
        """)
        output = _run_dry(ws, "step_a")
        assert "step(step_a" not in output
        assert "step(step_b)" in output
        assert "step(step_c" in output
        assert '"step_a"' not in output
        assert '"step_b"' in output


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
            ["python", "-m", "timbal.codegen", "--path", str(ws), "remove-step", "--name", "agent_b"],
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
