"""Tests for automatic wrapping of bare Python functions in Tool when codegen
operations (set-position, set-config) target them as workflow steps."""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def wf_workspace(tmp_path):
    """Write a workflow source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


def _run_dry(workspace_path: Path, operation: str, *cli_args: str) -> str:
    """Run a codegen operation with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", operation, *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_fail(workspace_path: Path, operation: str, *cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", operation, *cli_args],
        capture_output=True,
        text=True,
    )


def _exec_code(code: str) -> dict:
    ns = {}
    exec(code, ns)
    return ns


# ---------------------------------------------------------------------------
# set-position on bare function steps
# ---------------------------------------------------------------------------


class TestSetPositionBareFunction:
    """set-position should auto-wrap a bare function step in Tool."""

    def test_basic_bare_function(self, wf_workspace):
        """A bare function step gets wrapped in Tool with position metadata."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "100", "--y", "200")
        ns = _exec_code(output)
        assert ns["process"].metadata["position"] == {"x": 100.0, "y": 200.0}
        assert ns["process"].name == "process"
        # Function was renamed to process_fn and wrapped in Tool.
        assert "def process_fn" in output
        assert 'name="process"' in output
        assert "handler=process_fn" in output
        assert "Tool(" in output

    def test_bare_function_with_existing_steps(self, wf_workspace):
        """Wrapping targets only the bare function, not other steps."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(process, depends_on=["agent_a"])
        """)
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "50", "--y", "75")
        ns = _exec_code(output)
        assert ns["process"].metadata["position"] == {"x": 50.0, "y": 75.0}
        # Agent step untouched.
        assert "position" not in ns["agent_a"].metadata

    def test_bare_function_preserves_step_call_kwargs(self, wf_workspace):
        """The .step() call kwargs (depends_on, etc.) are preserved after wrapping."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(process, depends_on=["agent_a"])
        """)
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "10", "--y", "20")
        assert 'depends_on=["agent_a"]' in output

    def test_bare_function_with_complex_body(self, wf_workspace):
        """Bare function with multi-line body is properly wrapped."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def transform(data: dict) -> dict:
            result = {}
            for key, value in data.items():
                result[key.upper()] = str(value)
            return result

        workflow = Workflow(name="my_workflow")
        workflow.step(transform)
        """)
        output = _run_dry(ws, "set-position", "--name", "transform", "--x", "300", "--y", "400")
        ns = _exec_code(output)
        assert ns["transform"].metadata["position"] == {"x": 300.0, "y": 400.0}
        assert ns["transform"].name == "transform"

    def test_already_wrapped_function_not_double_wrapped(self, wf_workspace):
        """A function already wrapped in Tool is handled normally, not re-wrapped."""
        ws = wf_workspace("""\
        from timbal import Workflow
        from timbal.core import Tool

        def process_fn(x: str) -> str:
            return x.upper()

        process = Tool(name="process", handler=process_fn)

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "100", "--y", "200")
        ns = _exec_code(output)
        assert ns["process"].metadata["position"] == {"x": 100.0, "y": 200.0}
        # Should NOT add another Tool wrapper.
        assert output.count("Tool(") == 1

    def test_idempotent_after_wrapping(self, wf_workspace):
        """Running set-position twice works: first wraps, second updates."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        # First run: wraps and sets position.
        output1 = _run_dry(ws, "set-position", "--name", "process", "--x", "100", "--y", "200")
        # Write the result back and run again.
        (ws / "workflow.py").write_text(output1)
        output2 = _run_dry(ws, "set-position", "--name", "process", "--x", "300", "--y", "400")
        ns = _exec_code(output2)
        assert ns["process"].metadata["position"] == {"x": 300.0, "y": 400.0}
        assert output2.count("Tool(") == 1

    def test_bare_async_function(self, wf_workspace):
        """Async bare functions are wrapped correctly."""
        ws = wf_workspace("""\
        from timbal import Workflow

        async def fetch(url: str) -> str:
            return url

        workflow = Workflow(name="my_workflow")
        workflow.step(fetch)
        """)
        output = _run_dry(ws, "set-position", "--name", "fetch", "--x", "50", "--y", "60")
        ns = _exec_code(output)
        assert ns["fetch"].metadata["position"] == {"x": 50.0, "y": 60.0}
        assert "async def fetch_fn" in output

    def test_tool_import_not_duplicated(self, wf_workspace):
        """If Tool is already imported, don't add a second import."""
        ws = wf_workspace("""\
        from timbal import Workflow
        from timbal.core import Tool

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "10", "--y", "20")
        assert output.count("from timbal.core import Tool") == 1


# ---------------------------------------------------------------------------
# set-config on bare function steps
# ---------------------------------------------------------------------------


class TestSetConfigBareFunction:
    """set-config should auto-wrap a bare function step in Tool."""

    def test_set_name_on_bare_function(self, wf_workspace):
        """Setting name on a bare function wraps it and applies the name."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        config = json.dumps({"name": "my_processor"})
        output = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        ns = _exec_code(output)
        assert ns["process"].name == "my_processor"
        assert "def process_fn" in output
        assert "Tool(" in output

    def test_set_description_on_bare_function(self, wf_workspace):
        """Setting description on a bare function wraps it and applies the description."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        config = json.dumps({"description": "Processes text"})
        output = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        ns = _exec_code(output)
        assert ns["process"].description == "Processes text"

    def test_set_config_preserves_depends_on(self, wf_workspace):
        """Wrapping for set-config preserves .step() kwargs."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(process, depends_on=["agent_a"])
        """)
        config = json.dumps({"description": "Processes text"})
        output = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        assert 'depends_on=["agent_a"]' in output

    def test_rename_bare_function_updates_references(self, wf_workspace):
        """Renaming a wrapped bare function updates depends_on and step_span refs."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        def process(x: str) -> str:
            return x.upper()

        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        workflow.step(agent_b, depends_on=["process"], prompt=lambda: get_run_context().step_span("process").output)
        """)
        config = json.dumps({"name": "preprocessor"})
        output = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        assert 'name="preprocessor"' in output
        assert 'depends_on=["preprocessor"]' in output
        assert 'step_span("preprocessor")' in output
        # Old name should be gone from references.
        assert 'depends_on=["process"]' not in output
        assert 'step_span("process")' not in output

    def test_already_wrapped_not_double_wrapped(self, wf_workspace):
        """A Tool-wrapped function is handled normally by set-config."""
        ws = wf_workspace("""\
        from timbal import Workflow
        from timbal.core import Tool

        def process_fn(x: str) -> str:
            return x.upper()

        process = Tool(name="process", handler=process_fn)

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        config = json.dumps({"description": "Processes text"})
        output = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        ns = _exec_code(output)
        assert ns["process"].description == "Processes text"
        assert output.count("Tool(") == 1


# ---------------------------------------------------------------------------
# Combined operations (set-position then set-config, etc.)
# ---------------------------------------------------------------------------


class TestSequentialOperations:
    """Codegen operations applied sequentially on bare functions."""

    def test_set_position_then_set_config(self, wf_workspace):
        """First set-position wraps, then set-config modifies the Tool."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        output1 = _run_dry(ws, "set-position", "--name", "process", "--x", "100", "--y", "200")
        (ws / "workflow.py").write_text(output1)

        config = json.dumps({"description": "A processor"})
        output2 = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        ns = _exec_code(output2)
        assert ns["process"].metadata["position"] == {"x": 100.0, "y": 200.0}
        assert ns["process"].description == "A processor"

    def test_set_config_then_set_position(self, wf_workspace):
        """First set-config wraps, then set-position modifies the Tool."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        config = json.dumps({"description": "A processor"})
        output1 = _run_dry(ws, "set-config", "--name", "process", "--config", config)
        (ws / "workflow.py").write_text(output1)

        output2 = _run_dry(ws, "set-position", "--name", "process", "--x", "100", "--y", "200")
        ns = _exec_code(output2)
        assert ns["process"].metadata["position"] == {"x": 100.0, "y": 200.0}
        assert ns["process"].description == "A processor"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBareFunctionEdgeCases:
    """Edge cases for bare function wrapping."""

    def test_multiple_bare_functions_only_target_wrapped(self, wf_workspace):
        """Only the target bare function is wrapped, not others."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def step_a(x: str) -> str:
            return x.upper()

        def step_b(y: str) -> str:
            return y.lower()

        workflow = Workflow(name="my_workflow")
        workflow.step(step_a)
        workflow.step(step_b, depends_on=["step_a"])
        """)
        output = _run_dry(ws, "set-position", "--name", "step_a", "--x", "10", "--y", "20")
        ns = _exec_code(output)
        assert ns["step_a"].metadata["position"] == {"x": 10.0, "y": 20.0}
        # step_b should still be a bare function (not wrapped).
        assert "def step_b(y: str)" in output
        # Only step_a was renamed.
        assert "def step_a_fn(x: str)" in output

    def test_bare_function_with_lambda_params(self, wf_workspace):
        """Bare function step with lambda param mappings in .step() call."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        def process(text: str) -> str:
            return text.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        workflow.step(process, text=lambda: get_run_context().step_span("agent_a").output, depends_on=["agent_a"])
        """)
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "200", "--y", "300")
        ns = _exec_code(output)
        assert ns["process"].metadata["position"] == {"x": 200.0, "y": 300.0}
        # Lambda param mapping should be preserved.
        assert 'step_span("agent_a")' in output

    def test_helper_function_not_treated_as_bare_step(self, wf_workspace):
        """A helper function not used in .step() is not wrapped."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        def helper(x):
            return x * 2

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        # Trying to set position on "helper" — it's not a step, so no wrapping.
        # The helper may be removed by dead-code cleanup (it's unused), but
        # crucially it must NOT have been wrapped in Tool.
        output = _run_dry(ws, "set-position", "--name", "helper", "--x", "10", "--y", "20")
        assert "Tool(" not in output

    def test_bare_function_step_with_real_world_pattern(self, wf_workspace):
        """Reproduce the pattern from the user's licitacion example."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        greeter = Agent(
            name="greeter",
            model="openai/gpt-4o-mini",
            metadata={"position": {"x": 239.5, "y": 62.0}},
            system_prompt="You are a helpful assistant.",
            max_iter=1,
        )

        def licitacion_user_message(licitacion: dict):
            return {"role": "user", "content": str(licitacion)}

        workflow = Workflow(name="licitacion_pipeline")
        workflow.step(licitacion_user_message)
        workflow.step(
            greeter,
            depends_on=["licitacion_user_message"],
            prompt=lambda: (
                get_run_context().step_span("licitacion_user_message").output
            ),
        )
        """)
        output = _run_dry(
            ws, "set-position",
            "--name", "licitacion_user_message",
            "--x", "100", "--y", "50",
        )
        ns = _exec_code(output)
        assert ns["licitacion_user_message"].metadata["position"] == {"x": 100.0, "y": 50.0}
        assert ns["licitacion_user_message"].name == "licitacion_user_message"
        # The greeter agent should be untouched.
        assert ns["greeter"].metadata["position"] == {"x": 239.5, "y": 62.0}
        # References to the bare function step should be preserved.
        assert 'depends_on=["licitacion_user_message"]' in output
        assert 'step_span("licitacion_user_message")' in output


# ---------------------------------------------------------------------------
# get-flow integration with bare functions
# ---------------------------------------------------------------------------


class TestGetFlowBareFunction:
    """get-flow should work after wrapping bare functions."""

    def test_get_flow_after_wrapping(self, wf_workspace):
        """get-flow returns correct positions after wrapping."""
        ws = wf_workspace("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="my_workflow")
        workflow.step(process)
        """)
        # First, wrap via set-position.
        output = _run_dry(ws, "set-position", "--name", "process", "--x", "100", "--y", "200")
        (ws / "workflow.py").write_text(output)

        from timbal.codegen.flow import get_flow

        flow = get_flow(ws)
        node = flow["nodes"][0]
        assert node["position"] == {"x": 100.0, "y": 200.0}


# ---------------------------------------------------------------------------
# Unit test: cst_utils helpers
# ---------------------------------------------------------------------------


class TestIsBareFunctionStep:
    """Direct tests for the is_bare_function_step utility."""

    def test_detects_bare_function(self):
        import libcst as cst
        from timbal.codegen.cst_utils import collect_assignments, is_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="wf")
        workflow.step(process)
        """)
        tree = cst.parse_module(code)
        assignments = collect_assignments(tree)
        assert is_bare_function_step(tree, "workflow", "process", assignments) is True

    def test_rejects_already_wrapped(self):
        import libcst as cst
        from timbal.codegen.cst_utils import collect_assignments, is_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow
        from timbal.core import Tool

        def process_fn(x: str) -> str:
            return x.upper()

        process = Tool(name="process", handler=process_fn)

        workflow = Workflow(name="wf")
        workflow.step(process)
        """)
        tree = cst.parse_module(code)
        assignments = collect_assignments(tree)
        assert is_bare_function_step(tree, "workflow", "process", assignments) is False

    def test_rejects_helper_not_in_step(self):
        import libcst as cst
        from timbal.codegen.cst_utils import collect_assignments, is_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow

        def helper(x):
            return x * 2

        workflow = Workflow(name="wf")
        """)
        tree = cst.parse_module(code)
        assignments = collect_assignments(tree)
        assert is_bare_function_step(tree, "workflow", "helper", assignments) is False

    def test_rejects_nonexistent_function(self):
        import libcst as cst
        from timbal.codegen.cst_utils import collect_assignments, is_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow

        workflow = Workflow(name="wf")
        """)
        tree = cst.parse_module(code)
        assignments = collect_assignments(tree)
        assert is_bare_function_step(tree, "workflow", "nonexistent", assignments) is False


class TestWrapBareFunctionStep:
    """Direct tests for the wrap_bare_function_step utility."""

    def test_basic_wrap(self):
        import libcst as cst
        from timbal.codegen.cst_utils import wrap_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="wf")
        workflow.step(process)
        """)
        tree = cst.parse_module(code)
        new_tree = wrap_bare_function_step(tree, "workflow", "process")
        result = new_tree.code

        assert "def process_fn(x: str)" in result
        assert 'process = Tool(name="process", handler=process_fn)' in result
        assert "from timbal.core import Tool" in result

    def test_wrap_preserves_imports(self):
        import libcst as cst
        from timbal.codegen.cst_utils import wrap_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow
        from timbal.core import Tool

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="wf")
        workflow.step(process)
        """)
        tree = cst.parse_module(code)
        new_tree = wrap_bare_function_step(tree, "workflow", "process")
        result = new_tree.code

        # Should not duplicate the Tool import.
        assert result.count("from timbal.core import Tool") == 1

    def test_wrap_inserts_before_entry_point(self):
        import libcst as cst
        from timbal.codegen.cst_utils import wrap_bare_function_step

        code = textwrap.dedent("""\
        from timbal import Workflow

        def process(x: str) -> str:
            return x.upper()

        workflow = Workflow(name="wf")
        workflow.step(process)
        """)
        tree = cst.parse_module(code)
        new_tree = wrap_bare_function_step(tree, "workflow", "process")
        result = new_tree.code

        # Tool assignment should appear before the workflow assignment.
        tool_idx = result.index('process = Tool(')
        workflow_idx = result.index('workflow = Workflow(')
        assert tool_idx < workflow_idx
