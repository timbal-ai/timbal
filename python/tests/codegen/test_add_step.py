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


def _normalize(s: str) -> str:
    """Collapse whitespace so assertions are insensitive to formatting."""
    return " ".join(s.split())


class TestAgentStep:
    def test_adds_agent_step(self, workspace):
        """Add a standalone Agent step to a workflow."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "agent_a", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        assert "from timbal import Agent" in output or "import Agent" in output
        assert 'name="agent_a"' in norm
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
        norm = _normalize(output)
        assert "from timbal.core import Tool" in output
        assert "def process_fn(x: str) -> str:" in output
        assert 'name="process"' in norm
        assert "handler=process_fn" in norm
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
        norm = _normalize(output)
        assert "return x.lower()" in output
        assert output.count("workflow.step(process)") == 1
        assert 'name="process"' in norm
        assert "handler=process_fn" in norm


class TestFrameworkToolStep:
    def test_adds_web_search_step(self, workspace):
        """Add a framework tool as a workflow step."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        norm = _normalize(output)
        assert "from timbal.tools import WebSearch" in output
        assert "web_search = WebSearch(" in output
        assert "metadata=" in norm
        assert "workflow.step(web_search)" in output


class TestEntryPointValidation:
    def test_rejects_agent_entry_point(self, workspace):
        """add-step should reject an Agent entry point."""
        ws = workspace("""\
        from timbal import Agent

        workflow = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        _run_dry_expect_error(ws, "--type", "Agent", "--config", '{"name": "a", "model": "openai/gpt-4o-mini"}')


class TestAutoPosition:
    """Tests for automatic node positioning when adding steps."""

    def test_first_step_at_origin(self, workspace):
        """First step in an empty workflow gets position (100, 100)."""
        ws = workspace("""\
        from timbal import Workflow

        workflow = Workflow(name="my_workflow")
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "a", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        assert '"x": 100' in norm
        assert '"y": 100' in norm

    def test_second_step_offset_right(self, workspace):
        """Second step placed 360px to the right of the first."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "b", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        # agent_b should be at x=460 (100+360), y=100
        assert '"x": 460' in norm
        assert '"y": 100' in norm

    def test_centers_y_across_rightmost_column(self, workspace):
        """When multiple nodes share the max x, new node's y is their average."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        a = Agent(name="a", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 100}})
        b = Agent(name="b", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 240}})
        c = Agent(name="c", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 380}})

        workflow = Workflow(name="my_workflow")
        workflow.step(a)
        workflow.step(b)
        workflow.step(c)
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "d", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        # x = 460 + 360 = 820, y = avg(100, 240, 380) = 240
        assert '"x": 820' in norm
        assert '"y": 240' in norm

    def test_ignores_non_rightmost_for_y(self, workspace):
        """Nodes not in the rightmost column don't affect the new node's y."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        a = Agent(name="a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 500}})
        b = Agent(name="b", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 200}})

        workflow = Workflow(name="my_workflow")
        workflow.step(a)
        workflow.step(b)
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "c", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        # x = 460 + 360 = 820, y = 200 (only b matters, a is not rightmost)
        assert '"x": 820' in norm
        assert '"y": 200' in norm

    def test_explicit_position_overrides(self, workspace):
        """--x and --y override the auto-computed position."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        a = Agent(name="a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})

        workflow = Workflow(name="my_workflow")
        workflow.step(a)
        """)
        output = _run_dry(
            ws, "--type", "Agent",
            "--config", '{"name": "b", "model": "openai/gpt-4o-mini"}',
            "--x", "999", "--y", "777",
        )
        norm = _normalize(output)
        assert '"x": 999' in norm
        assert '"y": 777' in norm

    def test_chained_steps_with_bare_functions(self, workspace):
        """Bare function steps count as nodes at the default x column."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.core import Tool
        from timbal.state import get_run_context

        def fetch_data() -> list:
            return []

        def filter_data(data: list) -> dict:
            return {}

        delay = Tool(
            name="delay",
            handler=lambda: None,
            metadata={"position": {"x": 400, "y": 500}},
        )
        delay_2 = Tool(
            name="delay_2",
            handler=lambda: None,
            metadata={"position": {"x": 180, "y": 700}},
        )

        workflow = (
            Workflow(name="workflow")
            .step(fetch_data)
            .step(filter_data, data=lambda: get_run_context().step_span("fetch_data").output)
        )
        workflow.step(delay, depends_on=["filter_data"])
        workflow.step(delay_2, depends_on=["delay"])
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "new_agent", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        # 2 bare funcs at x=100 (virtual y=100, y=240)
        # delay at x=400, delay_2 at x=180
        # Rightmost column: x=400 (only delay) => new step x=760, y=500
        assert '"x": 760' in norm
        assert '"y": 500' in norm

    def test_bare_functions_only(self, workspace):
        """When all steps are bare functions, new node goes one column right, centered."""
        ws = workspace("""\
        from timbal import Workflow
        from timbal.state import get_run_context

        def step_a() -> str:
            return "a"

        def step_b(a: str) -> str:
            return "b"

        def step_c(b: str) -> str:
            return "c"

        workflow = (
            Workflow(name="workflow")
            .step(step_a)
            .step(step_b, a=lambda: get_run_context().step_span("step_a").output)
            .step(step_c, b=lambda: get_run_context().step_span("step_b").output)
        )
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "d", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        # 3 bare funcs at x=100 => virtual y: 100, 240, 380 => avg_y=240
        # New step: x=100+360=460, y=240
        assert '"x": 460' in norm
        assert '"y": 240' in norm

    def test_unpositioned_tool_treated_as_default_x(self, workspace):
        """A Tool step without position metadata counts as a node at default x."""
        ws = workspace("""\
        from timbal import Agent, Workflow

        a = Agent(name="a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(a)
        """)
        output = _run_dry(ws, "--type", "Agent", "--config", '{"name": "b", "model": "openai/gpt-4o-mini"}')
        norm = _normalize(output)
        # a has no position => 1 unpositioned step at virtual (100, 100)
        # New step: x=100+360=460, y=100
        b_section = output[output.index('name="b"'):]
        b_norm = _normalize(b_section)
        assert '"x": 460' in b_norm
        assert '"y": 100' in b_norm
