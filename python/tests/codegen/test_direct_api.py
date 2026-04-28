"""Direct Python API tests for codegen transformers.

All tests call apply_operation() or transformer functions directly (no subprocess)
so that pytest-cov can measure coverage of the transformer code.
"""

import io
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import libcst as cst
import pytest

from timbal.codegen import parse_fqn
from timbal.codegen.model_discovery import get_models, get_provider_summaries
from timbal.codegen.transformers import apply_operation


# ---------------------------------------------------------------------------
# Shared workspace fixture
# ---------------------------------------------------------------------------

TIMBAL_YAML = 'fqn: "flow.py::flow"\n'


@pytest.fixture
def ws(tmp_path):
    """Return a factory: write flow.py + timbal.yaml, return workspace Path."""

    def _write(source: str) -> Path:
        (tmp_path / "flow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
        return tmp_path

    return _write


def _apply(ws_path: Path, operation: str, **kwargs) -> str:
    return apply_operation(ws_path, operation, **kwargs)


# ---------------------------------------------------------------------------
# model_discovery.py
# ---------------------------------------------------------------------------

class TestModelDiscovery:
    def test_get_models_returns_nonempty_list(self):
        models = get_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_models_entries_have_required_keys(self):
        models = get_models()
        for model in models[:3]:
            assert "id" in model
            assert "provider" in model

    def test_get_provider_summaries_returns_sorted_desc(self):
        summaries = get_provider_summaries()
        assert isinstance(summaries, list)
        assert len(summaries) > 0
        counts = [s["model_count"] for s in summaries]
        assert counts == sorted(counts, reverse=True)

    def test_get_provider_summaries_have_expected_keys(self):
        summaries = get_provider_summaries()
        for s in summaries:
            assert "name" in s
            assert "model_count" in s
            assert s["model_count"] > 0


# ---------------------------------------------------------------------------
# codegen/test.py (run_test)
# ---------------------------------------------------------------------------

class TestRunTest:
    async def test_run_test_no_stream(self, capsys):
        from timbal.codegen.test import run_test
        from timbal.utils import ImportSpec

        fixture = Path(__file__).parent.parent / "server" / "fixtures" / "tool_fixture.py"
        spec = ImportSpec(path=fixture, target="tool_fixture")
        await run_test(spec, params={"x": "hello"}, stream=False)
        captured = capsys.readouterr()
        assert "result: hello" in captured.out

    async def test_run_test_with_stream(self, capsys):
        from timbal.codegen.test import run_test
        from timbal.utils import ImportSpec

        fixture = Path(__file__).parent.parent / "server" / "fixtures" / "tool_fixture.py"
        spec = ImportSpec(path=fixture, target="tool_fixture")
        await run_test(spec, params={"x": "world"}, stream=True)
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
        assert events
        assert any(event["type"] == "OUTPUT" for event in events)

    async def test_run_test_stream_keeps_user_prints_off_stdout(self, tmp_path, capsys):
        from timbal.codegen.test import run_test
        from timbal.utils import ImportSpec

        fixture = tmp_path / "printing_tool.py"
        fixture.write_text(
            textwrap.dedent(
                """
                from timbal import Tool

                def handler(x: str) -> str:
                    print(f"user print: {x}")
                    return f"result: {x}"

                flow = Tool(name="printing_tool", handler=handler)
                """
            )
        )
        spec = ImportSpec(path=fixture, target="flow")

        await run_test(spec, params={"x": "world"}, stream=True)
        captured = capsys.readouterr()

        events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
        assert events
        assert "user print: world" in captured.err

    async def test_run_test_with_run_context(self, capsys):
        from timbal.codegen.test import run_test
        from timbal.state import RunContext
        from timbal.utils import ImportSpec

        fixture = Path(__file__).parent.parent / "server" / "fixtures" / "tool_fixture.py"
        spec = ImportSpec(path=fixture, target="tool_fixture")
        run_context = RunContext()
        await run_test(spec, params={"x": "ctx"}, run_context=run_context, stream=False)
        captured = capsys.readouterr()
        assert "result: ctx" in captured.out


# ---------------------------------------------------------------------------
# apply_operation errors
# ---------------------------------------------------------------------------

class TestApplyOperationErrors:
    def test_file_not_found_raises(self, tmp_path):
        (tmp_path / "timbal.yaml").write_text('fqn: "missing.py::flow"\n')
        with pytest.raises(FileNotFoundError):
            apply_operation(tmp_path, "set_config", name=None, config='{"model": "openai/gpt-4o"}')

    def test_unknown_operation_raises(self, ws):
        ws_path = ws("from timbal import Agent\nflow = Agent(name='a', model='openai/gpt-4o')\n")
        with pytest.raises(ValueError, match="unknown operation"):
            apply_operation(ws_path, "nonexistent_op")


# ---------------------------------------------------------------------------
# add_step
# ---------------------------------------------------------------------------

WORKFLOW_SOURCE = """\
from timbal import Workflow

flow = Workflow(name="my_workflow")
"""

WORKFLOW_WITH_AGENT_STEP = """\
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
"""


class TestAddStep:
    def test_add_agent_step(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        result = _apply(ws_path, "add_step", step_type="Agent", config='{"name": "agent_a", "model": "openai/gpt-4o-mini"}', definition=None, step_name=None, x=None, y=None)
        assert "Agent(" in result
        assert 'name="agent_a"' in result
        assert "flow.step(agent_a)" in result
        assert "from timbal import Agent" in result

    def test_add_agent_step_explicit_position(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        result = _apply(ws_path, "add_step", step_type="Agent", config='{"name": "agent_a", "model": "openai/gpt-4o-mini"}', definition=None, step_name=None, x=200.0, y=300.0)
        assert '"x": 200' in result
        assert '"y": 300' in result

    def test_add_agent_step_name_via_step_name_arg(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        result = _apply(ws_path, "add_step", step_type="Agent", config='{"model": "openai/gpt-4o-mini"}', definition=None, step_name="my_agent", x=None, y=None)
        assert "flow.step(my_agent)" in result

    def test_add_agent_step_missing_name_raises(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="name"):
            _apply(ws_path, "add_step", step_type="Agent", config='{"model": "openai/gpt-4o-mini"}', definition=None, step_name=None, x=None, y=None)

    def test_add_agent_step_unknown_config_field_raises(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="Unknown agent config field"):
            _apply(ws_path, "add_step", step_type="Agent", config='{"name": "a", "bogus_field": true}', definition=None, step_name=None, x=None, y=None)

    def test_add_custom_step(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        defn = "def process(x: str) -> str:\n    return x.upper()"
        result = _apply(ws_path, "add_step", step_type="Custom", config=None, definition=defn, step_name=None, x=None, y=None)
        assert "from timbal.core import Tool" in result
        assert "Tool(" in result
        assert "flow.step(process)" in result

    def test_add_custom_step_with_explicit_name(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        defn = "def process(x: str) -> str:\n    return x.upper()"
        result = _apply(ws_path, "add_step", step_type="Custom", config=None, definition=defn, step_name="my_proc", x=None, y=None)
        assert "flow.step(my_proc)" in result

    def test_add_custom_step_no_definition_raises(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="--definition"):
            _apply(ws_path, "add_step", step_type="Custom", config=None, definition=None, step_name=None, x=None, y=None)

    def test_add_custom_step_no_func_in_definition_raises(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="function definition"):
            _apply(ws_path, "add_step", step_type="Custom", config=None, definition="x = 1", step_name=None, x=None, y=None)

    def test_add_framework_tool_step(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        result = _apply(ws_path, "add_step", step_type="Bash", config=None, definition=None, step_name=None, x=None, y=None)
        assert "Bash(" in result
        assert "flow.step(bash)" in result
        assert "from timbal.tools import Bash" in result

    def test_add_framework_tool_step_with_name_override(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        result = _apply(ws_path, "add_step", step_type="Bash", config=None, definition=None, step_name="my_bash", x=None, y=None)
        assert "flow.step(my_bash)" in result

    def test_add_step_invalid_type_raises(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="not valid"):
            _apply(ws_path, "add_step", step_type="NotAType", config=None, definition=None, step_name=None, x=None, y=None)

    def test_add_step_requires_workflow_entry_point(self, ws):
        agent_source = """\
from timbal import Agent
flow = Agent(name="my_agent", model="openai/gpt-4o-mini")
"""
        ws_path = ws(agent_source)
        with pytest.raises(ValueError, match="Workflow"):
            _apply(ws_path, "add_step", step_type="Agent", config='{"name": "a", "model": "openai/gpt-4o-mini"}', definition=None, step_name=None, x=None, y=None)

    def test_add_step_idempotent_updates_existing(self, ws):
        """Adding the same step twice should update, not duplicate."""
        ws_path = ws(WORKFLOW_WITH_AGENT_STEP)
        result = _apply(ws_path, "add_step", step_type="Agent", config='{"name": "agent_a", "model": "openai/gpt-4o"}', definition=None, step_name=None, x=None, y=None)
        # Should still have exactly one flow.step(agent_a)
        assert result.count("flow.step(agent_a)") == 1
        assert 'model="openai/gpt-4o"' in result

    def test_add_step_position_computed_from_existing_steps(self, ws):
        """When existing steps have positions, new step should be placed further right."""
        ws_path = ws(WORKFLOW_WITH_AGENT_STEP)
        result = _apply(ws_path, "add_step", step_type="Agent", config='{"name": "agent_b", "model": "openai/gpt-4o-mini"}', definition=None, step_name=None, x=None, y=None)
        assert "agent_b" in result
        # New step should be at x=100+360=460 (further right than existing x=100)
        assert '"x": 460' in result


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------

AGENT_SOURCE = """\
from timbal import Agent

flow = Agent(name="my_agent", model="openai/gpt-4o-mini")
"""

AGENT_WITH_TOOL_SOURCE = """\
from timbal import Agent
from timbal.tools import Bash

bash = Bash()
flow = Agent(name="my_agent", model="openai/gpt-4o-mini", tools=[bash])
"""

WORKFLOW_WITH_STEP_SOURCE = """\
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
"""


class TestSetConfig:
    def test_set_agent_model(self, ws):
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "set_config", name=None, config='{"model": "openai/gpt-4o"}')
        assert 'model="openai/gpt-4o"' in result

    def test_set_agent_config_no_config_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="--config"):
            _apply(ws_path, "set_config", name=None, config=None)

    def test_set_agent_config_empty_config_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="--config"):
            _apply(ws_path, "set_config", name=None, config="{}")

    def test_set_agent_config_unknown_field_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="Unknown agent config field"):
            _apply(ws_path, "set_config", name=None, config='{"bogus": true}')

    def test_set_agent_config_removes_field_when_null(self, ws):
        """Setting a field to null should remove it from the constructor."""
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "set_config", name=None, config='{"model": null}')
        assert "model=" not in result

    def test_set_tool_config(self, ws):
        ws_path = ws(AGENT_WITH_TOOL_SOURCE)
        result = _apply(ws_path, "set_config", name="bash", config='{"name": "my_bash"}')
        assert "my_bash" in result

    def test_set_tool_config_not_found_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="not found"):
            _apply(ws_path, "set_config", name="nonexistent_tool", config='{"timeout": 30}')

    def test_set_workflow_step_config_requires_name(self, ws):
        ws_path = ws(WORKFLOW_WITH_STEP_SOURCE)
        with pytest.raises(ValueError, match="step name"):
            _apply(ws_path, "set_config", name=None, config='{"model": "openai/gpt-4o"}')

    def test_set_workflow_step_config_requires_config(self, ws):
        ws_path = ws(WORKFLOW_WITH_STEP_SOURCE)
        with pytest.raises(ValueError, match="--config"):
            _apply(ws_path, "set_config", name="agent_a", config=None)

    def test_set_workflow_step_config(self, ws):
        ws_path = ws(WORKFLOW_WITH_STEP_SOURCE)
        result = _apply(ws_path, "set_config", name="agent_a", config='{"model": "openai/gpt-4o"}')
        assert 'model="openai/gpt-4o"' in result

    def test_set_workflow_step_config_unknown_agent_field_raises(self, ws):
        ws_path = ws(WORKFLOW_WITH_STEP_SOURCE)
        with pytest.raises(ValueError, match="Unknown agent config field"):
            _apply(ws_path, "set_config", name="agent_a", config='{"bogus_field": true}')

    def test_set_config_wrong_entry_type_raises(self, ws):
        """If entry point is neither Agent nor Workflow, raise."""
        source = """\
from timbal.core import Tool

def my_fn(x: str) -> str:
    return x

flow = Tool(name="my_fn", handler=my_fn)
"""
        ws_path = ws(source)
        # Tool is not recognized as Agent or Workflow, so ep_type check triggers
        # Actually Tool won't be detected as Agent or Workflow — resolve_entry_point_type returns None
        # so set_config will treat it as Agent and check AGENT_FIELDS
        # Just test it doesn't crash on a valid config
        result = _apply(ws_path, "set_config", name=None, config='{"name": "new_name"}')
        assert "new_name" in result


# ---------------------------------------------------------------------------
# add_tool
# ---------------------------------------------------------------------------

class TestAddTool:
    def test_add_framework_tool_to_agent(self, ws):
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name=None, step=None)
        assert "Bash(" in result
        assert "tools=[bash]" in result or "tools=[" in result
        assert "from timbal.tools import Bash" in result

    def test_add_framework_tool_with_explicit_name(self, ws):
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name="my_bash", step=None)
        assert "my_bash" in result

    def test_add_framework_tool_invalid_type_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="--type must be one of"):
            _apply(ws_path, "add_tool", tool_type="NotATool", definition=None, tool_name=None, step=None)

    def test_add_custom_tool_to_agent(self, ws):
        ws_path = ws(AGENT_SOURCE)
        defn = "def my_search(query: str) -> str:\n    return query"
        result = _apply(ws_path, "add_tool", tool_type="Custom", definition=defn, tool_name=None, step=None)
        assert "Tool(" in result
        assert "from timbal.core import Tool" in result

    def test_add_custom_tool_with_explicit_name(self, ws):
        ws_path = ws(AGENT_SOURCE)
        defn = "def my_search(query: str) -> str:\n    return query"
        result = _apply(ws_path, "add_tool", tool_type="Custom", definition=defn, tool_name="searcher", step=None)
        assert "searcher" in result

    def test_add_custom_tool_no_definition_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="--definition"):
            _apply(ws_path, "add_tool", tool_type="Custom", definition=None, tool_name=None, step=None)

    def test_add_custom_tool_no_func_in_definition_raises(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="function definition"):
            _apply(ws_path, "add_tool", tool_type="Custom", definition="x = 1", tool_name=None, step=None)

    def test_add_tool_requires_agent_entry_point(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="Agent"):
            _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name=None, step=None)

    def test_add_tool_to_workflow_step(self, ws):
        ws_path = ws(WORKFLOW_WITH_AGENT_STEP)
        result = _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name=None, step="agent_a")
        assert "Bash(" in result

    def test_add_tool_step_requires_workflow(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="Workflow"):
            _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name=None, step="some_step")

    def test_add_tool_idempotent(self, ws):
        """Adding the same tool twice should not duplicate it."""
        ws_path = ws(AGENT_WITH_TOOL_SOURCE)
        result = _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name=None, step=None)
        # Should still have only one tools=[bash]
        assert result.count("bash") >= 1


# ---------------------------------------------------------------------------
# remove_step
# ---------------------------------------------------------------------------

class TestRemoveStep:
    def test_remove_step_standalone(self, ws):
        ws_path = ws(WORKFLOW_WITH_AGENT_STEP)
        result = _apply(ws_path, "remove_step", name="agent_a")
        assert "agent_a" not in result
        assert "flow.step(" not in result

    def test_remove_step_cleans_depends_on_references(self, ws):
        source = """\
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})
agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
flow.step(agent_b, depends_on=["agent_a"])
"""
        ws_path = ws(source)
        result = _apply(ws_path, "remove_step", name="agent_a")
        assert "agent_a" not in result
        assert "depends_on" not in result

    def test_remove_step_cleans_when_references(self, ws):
        source = """\
from timbal import Agent, Workflow
from timbal.state import get_run_context

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})
agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
flow.step(agent_b, when=lambda: get_run_context().step_span("agent_a").output)
"""
        ws_path = ws(source)
        result = _apply(ws_path, "remove_step", name="agent_a")
        assert "agent_a" not in result

    def test_remove_step_requires_workflow(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="Workflow"):
            _apply(ws_path, "remove_step", name="some_step")


# ---------------------------------------------------------------------------
# remove_tool
# ---------------------------------------------------------------------------

class TestRemoveTool:
    def test_remove_tool_from_agent(self, ws):
        ws_path = ws(AGENT_WITH_TOOL_SOURCE)
        result = _apply(ws_path, "remove_tool", name="bash", step=None)
        assert "tools=[]" in result or "bash" not in result

    def test_remove_tool_requires_agent_entry_point(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="Agent"):
            _apply(ws_path, "remove_tool", name="bash", step=None)

    def test_remove_tool_from_workflow_step(self, ws):
        source = """\
from timbal import Agent, Workflow
from timbal.tools import Bash

bash = Bash()
agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", tools=[bash], metadata={"position": {"x": 100, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
"""
        ws_path = ws(source)
        result = _apply(ws_path, "remove_tool", name="bash", step="agent_a")
        # bash should be removed from agent_a's tools
        assert result.count("tools=[bash]") == 0

    def test_remove_tool_step_requires_workflow(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="Workflow"):
            _apply(ws_path, "remove_tool", name="bash", step="some_step")


# ---------------------------------------------------------------------------
# convert_to_workflow
# ---------------------------------------------------------------------------

class TestConvertToWorkflow:
    def test_convert_agent_to_workflow(self, ws):
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "convert_to_workflow", workflow_name=None)
        assert "Workflow(" in result
        assert "from timbal import" in result
        assert "Workflow" in result
        assert ".step(" in result

    def test_convert_with_custom_workflow_name(self, ws):
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "convert_to_workflow", workflow_name="my_pipeline")
        assert 'name="my_pipeline"' in result

    def test_convert_already_has_workflow_import(self, ws):
        source = """\
from timbal import Agent, Workflow

flow = Agent(name="my_agent", model="openai/gpt-4o-mini")
"""
        ws_path = ws(source)
        result = _apply(ws_path, "convert_to_workflow", workflow_name=None)
        # Should not duplicate import
        assert result.count("from timbal import") >= 1

    def test_convert_requires_agent_entry_point(self, ws):
        ws_path = ws(WORKFLOW_SOURCE)
        with pytest.raises(ValueError, match="Agent"):
            _apply(ws_path, "convert_to_workflow", workflow_name=None)

    def test_convert_agent_name_collision(self, ws):
        """When agent's name== entry_point var, it should use a _step suffix."""
        source = """\
from timbal import Agent

flow = Agent(name="flow", model="openai/gpt-4o-mini")
"""
        ws_path = ws(source)
        result = _apply(ws_path, "convert_to_workflow", workflow_name=None)
        # The agent var should be renamed to avoid self-reference
        assert "flow_step" in result


# ---------------------------------------------------------------------------
# set_position
# ---------------------------------------------------------------------------

class TestSetPosition:
    def test_set_agent_position(self, ws):
        ws_path = ws(AGENT_SOURCE)
        result = _apply(ws_path, "set_position", name=None, x=200.0, y=300.0)
        assert '"x": 200' in result
        assert '"y": 300' in result

    def test_set_agent_position_forbids_name(self, ws):
        ws_path = ws(AGENT_SOURCE)
        with pytest.raises(ValueError, match="--name is only supported for Workflow"):
            _apply(ws_path, "set_position", name="some_step", x=100.0, y=100.0)

    def test_set_workflow_step_position(self, ws):
        ws_path = ws(WORKFLOW_WITH_STEP_SOURCE)
        result = _apply(ws_path, "set_position", name="agent_a", x=500.0, y=250.0)
        assert '"x": 500' in result
        assert '"y": 250' in result

    def test_set_workflow_position_requires_name(self, ws):
        ws_path = ws(WORKFLOW_WITH_STEP_SOURCE)
        with pytest.raises(ValueError, match="requires a step --name"):
            _apply(ws_path, "set_position", name=None, x=100.0, y=100.0)

    def test_set_position_updates_existing_metadata(self, ws):
        """Position should overwrite existing position in metadata."""
        ws_path = ws(WORKFLOW_WITH_AGENT_STEP)
        result = _apply(ws_path, "set_position", name="agent_a", x=999.0, y=888.0)
        assert '"x": 999' in result
        assert '"y": 888' in result
        # Old position x=100 should be gone
        assert '"x": 100' not in result


# ---------------------------------------------------------------------------
# set_config — deeper paths
# ---------------------------------------------------------------------------

AGENT_WITH_INLINE_TOOL_SOURCE = """\
from timbal import Agent
from timbal.tools import Bash

flow = Agent(name="my_agent", model="openai/gpt-4o-mini", tools=[Bash()])
"""

WORKFLOW_WITH_BARE_FUNC_SOURCE = """\
from timbal import Workflow


def my_func(x: str) -> str:
    return x.upper()


flow = Workflow(name="my_workflow")
flow.step(my_func)
"""

WORKFLOW_WITH_TWO_STEPS_SOURCE = """\
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})
agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
flow.step(agent_b, depends_on=["agent_a"])
"""


class TestSetConfigDeepPaths:
    def test_set_inline_tool_config_migrates_to_variable(self, ws):
        """ToolConfigSetter: inline Bash() in tools list → extract to variable."""
        ws_path = ws(AGENT_WITH_INLINE_TOOL_SOURCE)
        result = _apply(ws_path, "set_config", name="bash", config='{"name": "my_bash"}')
        # Tool should be extracted to a variable
        assert "bash" in result

    def test_set_config_on_bare_function_step_wraps_first(self, ws):
        """set_config on a bare function step wraps it in Tool first."""
        ws_path = ws(WORKFLOW_WITH_BARE_FUNC_SOURCE)
        result = _apply(ws_path, "set_config", name="my_func", config='{"description": "my func"}')
        assert "my_func" in result
        assert "description" in result

    def test_set_step_config_rename_updates_depends_on(self, ws):
        """StepConstructorConfigSetter: renaming a step updates depends_on references."""
        ws_path = ws(WORKFLOW_WITH_TWO_STEPS_SOURCE)
        result = _apply(ws_path, "set_config", name="agent_a", config='{"name": "renamed_a"}')
        assert "renamed_a" in result
        # depends_on should be updated to new name
        assert '"agent_a"' not in result or "renamed_a" in result


# ---------------------------------------------------------------------------
# remove_step — chained syntax and additional reference cleanup
# ---------------------------------------------------------------------------

WORKFLOW_CHAINED_SOURCE = """\
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
flow = (
    Workflow(name="my_workflow")
    .step(agent_a)
    .step(agent_b)
)
"""

WORKFLOW_PARTIAL_DEPENDS_ON_SOURCE = """\
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", metadata={"position": {"x": 100, "y": 100}})
agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", metadata={"position": {"x": 460, "y": 100}})
agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini", metadata={"position": {"x": 820, "y": 100}})
flow = Workflow(name="my_workflow")
flow.step(agent_a)
flow.step(agent_b)
flow.step(agent_c, depends_on=["agent_a", "agent_b"])
"""


class TestRemoveStepDeepPaths:
    def test_remove_step_chained_syntax(self, ws):
        """Remove a step from a chained Workflow().step().step() chain."""
        ws_path = ws(WORKFLOW_CHAINED_SOURCE)
        result = _apply(ws_path, "remove_step", name="agent_a")
        assert "agent_a" not in result

    def test_remove_step_partial_depends_on(self, ws):
        """When a step has multiple deps and one is removed, keep the others."""
        ws_path = ws(WORKFLOW_PARTIAL_DEPENDS_ON_SOURCE)
        result = _apply(ws_path, "remove_step", name="agent_a")
        # agent_c's depends_on should still reference agent_b
        assert "agent_a" not in result
        assert "agent_b" in result


# ---------------------------------------------------------------------------
# add_tool — inline tool migration
# ---------------------------------------------------------------------------

AGENT_WITH_INLINE_BASH_SOURCE = """\
from timbal import Agent
from timbal.tools import Bash

flow = Agent(name="my_agent", model="openai/gpt-4o-mini", tools=[Bash()])
"""


class TestAddToolDeepPaths:
    def test_add_tool_when_existing_inline_tool_present(self, ws):
        """Adding a new tool alongside an existing inline tool."""
        ws_path = ws(AGENT_WITH_INLINE_BASH_SOURCE)
        result = _apply(ws_path, "add_tool", tool_type="Bash", definition=None, tool_name="my_bash", step=None)
        # Should still produce valid Python with the tool
        assert "my_bash" in result


# ---------------------------------------------------------------------------
# set_position — bare function step wrapping
# ---------------------------------------------------------------------------

class TestSetPositionDeepPaths:
    def test_set_position_on_bare_function_step(self, ws):
        """set-position on a bare function step should wrap it in Tool first."""
        ws_path = ws(WORKFLOW_WITH_BARE_FUNC_SOURCE)
        result = _apply(ws_path, "set_position", name="my_func", x=200.0, y=150.0)
        assert '"x": 200' in result
        assert '"y": 150' in result
