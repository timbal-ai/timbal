"""End-to-end tests that chain every codegen operation and validate each intermediate state.

TestFullLifecycle:
    Basic lifecycle: agent → add/remove tools → convert to workflow → add/remove steps → rename.

TestComplexWorkflow:
    Multi-step workflow with custom function steps, framework tool steps, chained params
    with key indexing via set-param, ordering edges, tool management on individual steps,
    renames that propagate across step_span and depends_on references, and sequential
    removal that validates cleanup.
"""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'

INITIAL_SOURCE = """\
from timbal.core import Agent

agent = Agent(name="agent", model="openai/gpt-4o-mini")
"""


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with a bare agent and return the directory."""
    (tmp_path / "agent.py").write_text(textwrap.dedent(INITIAL_SOURCE))
    (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
    return tmp_path


def _run(workspace_path: Path, operation: str, *cli_args: str) -> None:
    """Run a codegen operation that writes to disk. Asserts success."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), operation, *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{operation} failed:\n{result.stderr}"


def _run_err(workspace_path: Path, operation: str, *cli_args: str) -> str:
    """Run a codegen operation expected to fail. Returns stderr."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), operation, *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"{operation} should have failed but succeeded"
    return result.stderr


def _dry_run(workspace_path: Path, operation: str, *cli_args: str) -> str:
    """Run a codegen operation with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", operation, *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{operation} --dry-run failed:\n{result.stderr}"
    return result.stdout


def _read_source(workspace_path: Path) -> str:
    """Read the current agent source file."""
    return (workspace_path / "agent.py").read_text()


def _exec_code(code: str) -> dict:
    """Exec Python code and return the resulting namespace."""
    ns = {}
    exec(code, ns)
    return ns


def _tool_names(agent) -> list[str]:
    """Extract tool runtime names from an agent object."""
    return [t.name for t in agent.tools]


def _count_step_calls(source: str, entry_point: str = "agent") -> int:
    """Count the number of .step() calls on the entry point in source."""
    return source.count(f"{entry_point}.step(")


def _norm(source: str) -> str:
    """Collapse whitespace for assertions that span multiple lines."""
    return " ".join(source.split())


def _has_step_call(source: str, step_name: str, entry_point: str = "agent") -> bool:
    """Check if source contains entry_point.step(step_name ...) allowing ruff formatting."""
    import re
    return bool(re.search(rf"{re.escape(entry_point)}\.step\(\s*{re.escape(step_name)}\b", source))


class TestFullLifecycle:
    def test_full_lifecycle(self, workspace):
        ws = workspace

        # ---------------------------------------------------------------
        # Step 1: Verify initial state — bare agent, no tools
        # ---------------------------------------------------------------
        source = _read_source(ws)
        ns = _exec_code(source)
        assert ns["agent"].name == "agent"
        assert ns["agent"].model == "openai/gpt-4o-mini"
        assert len(ns["agent"].tools) == 0

        # ---------------------------------------------------------------
        # Step 2: Add a framework tool (WebSearch)
        # ---------------------------------------------------------------
        _run(ws, "add-tool", "--type", "WebSearch")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "web_search" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 1
        assert "from timbal.tools import WebSearch" in source

        # ---------------------------------------------------------------
        # Step 3: Add a custom tool
        # ---------------------------------------------------------------
        definition = "def summarize(text: str) -> str:\n    return text[:100]"
        _run(ws, "add-tool", "--type", "Custom", "--definition", definition)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "summarize" in _tool_names(ns["agent"])
        assert "web_search" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 2
        assert "def summarize(text: str) -> str:" in source

        # ---------------------------------------------------------------
        # Step 4: Set config on the agent (model, system_prompt)
        # ---------------------------------------------------------------
        config = json.dumps({"model": "openai/gpt-4o", "system_prompt": "You are a helpful assistant."})
        _run(ws, "set-config", "--config", config)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].system_prompt == "You are a helpful assistant."
        # Tools should be preserved.
        assert len(ns["agent"].tools) == 2

        # ---------------------------------------------------------------
        # Step 5: Set config on a tool (WebSearch allowed_domains)
        # ---------------------------------------------------------------
        tool_config = json.dumps({"allowed_domains": ["example.com", "docs.python.org"]})
        _run(ws, "set-config", "--name", "web_search", "--config", tool_config)
        source = _read_source(ws)
        ns = _exec_code(source)

        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.allowed_domains == ["example.com", "docs.python.org"]
        # Other tools and agent config preserved.
        assert ns["agent"].model == "openai/gpt-4o"
        assert "summarize" in _tool_names(ns["agent"])

        # ---------------------------------------------------------------
        # Step 6: Remove the custom tool
        # ---------------------------------------------------------------
        _run(ws, "remove-tool", "--name", "summarize")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "summarize" not in _tool_names(ns["agent"])
        assert "web_search" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 1
        # The function definition should be cleaned up.
        assert "def summarize" not in source

        # ---------------------------------------------------------------
        # Step 7: Add the custom tool back
        # ---------------------------------------------------------------
        _run(ws, "add-tool", "--type", "Custom", "--definition", definition)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "summarize" in _tool_names(ns["agent"])
        assert "web_search" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 2

        # ---------------------------------------------------------------
        # Step 8: Convert to workflow
        # ---------------------------------------------------------------
        _run(ws, "convert-to-workflow", "--name", "my_workflow")
        source = _read_source(ws)
        ns = _exec_code(source)

        from timbal.core.workflow import Workflow

        assert isinstance(ns["agent"], Workflow)
        assert ns["agent"].name == "my_workflow"
        assert ".step(" in source
        assert "from timbal import" in source or "from timbal.core import" in source

        # ---------------------------------------------------------------
        # Step 9: Add a new Agent step to the workflow
        # ---------------------------------------------------------------
        step_config = json.dumps({"name": "agent_b", "model": "openai/gpt-4o-mini"})
        _run(ws, "add-step", "--type", "Agent", "--config", step_config)
        source = _read_source(ws)

        assert "agent_b = Agent(" in source
        assert 'name="agent_b"' in source
        assert _has_step_call(source, "agent_b")

        # ---------------------------------------------------------------
        # Step 10: Add a tool to the new step (--step flag)
        # ---------------------------------------------------------------
        _run(ws, "add-tool", "--type", "Edit", "--step", "agent_b")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "Edit" in source
        assert "edit" in [t.name for t in ns["agent_b"].tools]

        # ---------------------------------------------------------------
        # Step 11: Set config on the new step (model)
        # ---------------------------------------------------------------
        step_model_config = json.dumps({"model": "openai/gpt-4o"})
        _run(ws, "set-config", "--name", "agent_b", "--config", step_model_config)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert ns["agent_b"].model == "openai/gpt-4o"
        assert "edit" in [t.name for t in ns["agent_b"].tools]

        # ---------------------------------------------------------------
        # Step 12: Set step params (wire agent_b's prompt)
        # ---------------------------------------------------------------
        _run(ws, "set-param", "--target", "agent_b", "--name", "prompt", "--type", "map", "--source", "agent")
        source = _read_source(ws)

        assert 'step_span("agent")' in source
        assert "get_run_context" in source

        # ---------------------------------------------------------------
        # Step 13: Remove the tool from the step
        # ---------------------------------------------------------------
        _run(ws, "remove-tool", "--name", "edit", "--step", "agent_b")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "edit" not in [t.name for t in ns["agent_b"].tools]
        assert ns["agent_b"].model == "openai/gpt-4o"

        # ---------------------------------------------------------------
        # Step 14: Remove the step entirely
        # ---------------------------------------------------------------
        _run(ws, "remove-step", "--name", "agent_b")
        source = _read_source(ws)

        assert "agent_b" not in source
        assert ".step(" in source

        # ---------------------------------------------------------------
        # Step 15: Rename the original agent step
        # ---------------------------------------------------------------
        assert 'name="agent"' in source

        _run(ws, "rename", "--old-name", "agent", "--to", "primary_agent")
        source = _read_source(ws)

        assert "agent_step" not in source
        assert "primary_agent" in source
        assert 'name="primary_agent"' in source
        assert _has_step_call(source, "primary_agent")

        # ---------------------------------------------------------------
        # Final validation
        # ---------------------------------------------------------------
        ns = _exec_code(source)
        assert isinstance(ns["agent"], Workflow)
        assert ns["agent"].name == "my_workflow"


class TestComplexWorkflow:
    """Build a multi-step workflow with custom functions, chained params,
    depends_on ordering, tool management, renames, and removals."""

    def test_multi_step_pipeline(self, workspace):
        ws = workspace

        # ===============================================================
        # Phase 1: Build up the agent with multiple tools
        # ===============================================================

        # -- 1a. Add two custom tools with multi-line bodies --
        extract_def = (
            "def extract_keywords(text: str) -> list:\n"
            "    words = text.lower().split()\n"
            "    stopwords = {'the', 'a', 'an', 'is', 'are', 'was'}\n"
            "    return [w for w in words if w not in stopwords]"
        )
        _run(ws, "add-tool", "--type", "Custom", "--definition", extract_def)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "extract_keywords" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 1
        # Verify the function actually works at runtime.
        fn = ns["extract_keywords"]
        assert fn("The quick brown fox") == ["quick", "brown", "fox"]

        format_def = (
            "def format_output(items: list, separator: str = ', ') -> str:\n"
            "    return separator.join(str(i) for i in items)"
        )
        _run(ws, "add-tool", "--type", "Custom", "--definition", format_def)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "format_output" in _tool_names(ns["agent"])
        assert "extract_keywords" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 2
        # Verify the function works.
        fn = ns["format_output"]
        assert fn(["a", "b", "c"]) == "a, b, c"
        assert fn(["x", "y"], separator=" | ") == "x | y"

        # -- 1b. Add a framework tool (WebSearch) with a custom name --
        _run(ws, "add-tool", "--type", "WebSearch", "--name", "search")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "search" in _tool_names(ns["agent"])
        assert len(ns["agent"].tools) == 3
        assert "from timbal.tools import WebSearch" in source

        # -- 1c. Set agent config --
        config = json.dumps({
            "model": "openai/gpt-4o",
            "system_prompt": "You are a data processing pipeline.",
            "max_iter": 10,
            "temperature": 0.7,
        })
        _run(ws, "set-config", "--config", config)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].system_prompt == "You are a data processing pipeline."
        assert ns["agent"].max_iter == 10
        assert ns["agent"].temperature == 0.7
        # All 3 tools preserved.
        assert len(ns["agent"].tools) == 3

        # -- 1d. Configure the WebSearch tool --
        tool_config = json.dumps({
            "allowed_domains": ["wikipedia.org", "arxiv.org"],
        })
        _run(ws, "set-config", "--name", "search", "--config", tool_config)
        source = _read_source(ws)
        ns = _exec_code(source)

        search_tool = next(t for t in ns["agent"].tools if t.name == "search")
        assert search_tool.allowed_domains == ["wikipedia.org", "arxiv.org"]

        # ===============================================================
        # Phase 2: Convert to workflow
        # ===============================================================
        _run(ws, "convert-to-workflow", "--name", "pipeline")
        source = _read_source(ws)
        ns = _exec_code(source)

        from timbal.core.workflow import Workflow

        assert isinstance(ns["agent"], Workflow)
        assert ns["agent"].name == "pipeline"
        # Original agent (name="agent") becomes agent_step variable.
        assert "agent_step" in source
        assert 'name="agent"' in source
        assert _has_step_call(source, "agent_step")
        assert _count_step_calls(source) == 1

        # Verify the original agent step preserved all config and tools.
        ns = _exec_code(source)
        assert ns["agent_step"].model == "openai/gpt-4o"
        assert ns["agent_step"].system_prompt == "You are a data processing pipeline."
        assert ns["agent_step"].max_iter == 10
        assert len(ns["agent_step"].tools) == 3

        # ===============================================================
        # Phase 3: Add multiple steps to the workflow
        # ===============================================================

        # -- 3a. Add a custom function step (preprocessor) --
        preprocess_def = (
            "def preprocessor(text: str) -> dict:\n"
            "    cleaned = text.strip().lower()\n"
            "    word_count = len(cleaned.split())\n"
            "    return {'cleaned': cleaned, 'word_count': word_count}"
        )
        _run(ws, "add-step", "--type", "Custom", "--definition", preprocess_def)
        source = _read_source(ws)

        assert "def preprocessor_fn(text: str) -> dict:" in source
        assert 'preprocessor = Tool(name="preprocessor", handler=preprocessor_fn)' in source
        assert _has_step_call(source, "preprocessor")
        assert _count_step_calls(source) == 2

        # Verify the function works.
        ns = _exec_code(source)
        result = ns["preprocessor_fn"]("  Hello World  ")
        assert result == {"cleaned": "hello world", "word_count": 2}

        # -- 3b. Add another custom function step (postprocessor) --
        postprocess_def = (
            "def postprocessor(data: dict, prefix: str = 'Result') -> str:\n"
            "    parts = [f'{prefix}: {k}={v}' for k, v in data.items()]\n"
            "    return '\\n'.join(parts)"
        )
        _run(ws, "add-step", "--type", "Custom", "--definition", postprocess_def)
        source = _read_source(ws)

        assert "def postprocessor_fn(data: dict" in source
        assert _has_step_call(source, "postprocessor")
        assert _count_step_calls(source) == 3

        # Verify the function works.
        ns = _exec_code(source)
        result = ns["postprocessor_fn"]({"a": 1, "b": 2})
        assert result == "Result: a=1\nResult: b=2"

        # -- 3c. Add a second Agent step (reviewer) --
        reviewer_config = json.dumps({
            "name": "reviewer",
            "model": "openai/gpt-4o-mini",
            "system_prompt": "You review and validate outputs.",
            "max_iter": 3,
        })
        _run(ws, "add-step", "--type", "Agent", "--config", reviewer_config)
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "reviewer = Agent(" in source
        assert 'name="reviewer"' in source
        assert _has_step_call(source, "reviewer")
        assert _count_step_calls(source) == 4
        assert ns["reviewer"].model == "openai/gpt-4o-mini"
        assert ns["reviewer"].system_prompt == "You review and validate outputs."
        assert ns["reviewer"].max_iter == 3
        assert len(ns["reviewer"].tools) == 0

        # -- 3d. Add a third Agent step (summarizer_agent) --
        summarizer_config = json.dumps({
            "name": "summarizer_agent",
            "model": "openai/gpt-4o",
        })
        _run(ws, "add-step", "--type", "Agent", "--config", summarizer_config)
        source = _read_source(ws)

        assert "summarizer_agent = Agent(" in source
        assert 'name="summarizer_agent"' in source
        assert _has_step_call(source, "summarizer_agent")
        assert _count_step_calls(source) == 5

        # ===============================================================
        # Phase 4: Add tools to individual steps
        # ===============================================================

        # -- 4a. Add WebSearch to reviewer --
        _run(ws, "add-tool", "--type", "WebSearch", "--step", "reviewer")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "web_search" in [t.name for t in ns["reviewer"].tools]
        assert len(ns["reviewer"].tools) == 1
        # Original agent_step tools should be untouched.
        assert len(ns["agent_step"].tools) == 3

        # -- 4b. Add Read tool to reviewer --
        _run(ws, "add-tool", "--type", "Read", "--step", "reviewer")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "read" in [t.name for t in ns["reviewer"].tools]
        assert "web_search" in [t.name for t in ns["reviewer"].tools]
        assert len(ns["reviewer"].tools) == 2

        # -- 4c. Add a custom tool to summarizer_agent --
        count_def = "def count_words(text: str) -> int:\n    return len(text.split())"
        _run(ws, "add-tool", "--type", "Custom", "--definition", count_def, "--step", "summarizer_agent")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "count_words" in [t.name for t in ns["summarizer_agent"].tools]
        assert len(ns["summarizer_agent"].tools) == 1
        # Verify the function is defined.
        assert "def count_words(text: str) -> int:" in source

        # ===============================================================
        # Phase 5: Wire steps with params, key indexing, and depends_on
        # ===============================================================

        # -- 5a. Wire reviewer's prompt to preprocessor's output with key --
        _run(ws, "set-param", "--target", "reviewer", "--name", "prompt", "--type", "map", "--source", "preprocessor", "--key", "output.cleaned")
        source = _read_source(ws)

        assert 'step_span("preprocessor")' in source
        assert ".output.cleaned" in source
        assert "get_run_context" in source
        assert "from timbal.state import get_run_context" in source

        # -- 5b. Wire summarizer_agent with multiple params --
        _run(ws, "set-param", "--target", "summarizer_agent", "--name", "prompt", "--type", "map", "--source", "reviewer")
        _run(ws, "set-param", "--target", "summarizer_agent", "--name", "context", "--type", "map", "--source", "preprocessor", "--key", "output.word_count")
        source = _read_source(ws)

        assert 'step_span("reviewer").output' in source
        assert 'step_span("preprocessor").output.word_count' in source

        # -- 5c. Set postprocessor depends_on preprocessor (ordering only) --
        _run(ws, "add-edge", "--source", "preprocessor", "--target", "postprocessor")
        source = _read_source(ws)
        normalized = " ".join(source.split())

        assert 'depends_on=["preprocessor"]' in normalized

        # ===============================================================
        # Phase 6: Update step configs (combined config + params)
        # ===============================================================

        # -- 6a. Update reviewer: change model, then update params --
        new_config = json.dumps({"model": "openai/gpt-4o", "max_iter": 5})
        _run(ws, "set-config", "--name", "reviewer", "--config", new_config)
        _run(ws, "set-param", "--target", "reviewer", "--name", "prompt", "--type", "map", "--source", "preprocessor")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert ns["reviewer"].model == "openai/gpt-4o"
        assert ns["reviewer"].max_iter == 5
        # Params updated: no more key indexing.
        assert 'step_span("preprocessor").output' in source
        # Tools should still be there.
        assert len(ns["reviewer"].tools) == 2
        # system_prompt should be preserved.
        assert ns["reviewer"].system_prompt == "You review and validate outputs."

        # ===============================================================
        # Phase 7: Dry-run verification
        # ===============================================================

        # -- 7a. Dry-run a rename and verify it doesn't write to disk --
        source_before = _read_source(ws)
        dry_output = _dry_run(ws, "rename", "--old-name", "reviewer", "--to", "validator")
        source_after = _read_source(ws)

        assert source_before == source_after, "dry-run should not modify files"
        assert 'name="validator"' in dry_output
        assert "validator" in dry_output

        # ===============================================================
        # Phase 8: Rename a step and verify reference propagation
        # ===============================================================

        # Rename reviewer → validator. This should update:
        # - Variable name
        # - name= kwarg
        # - step_span("reviewer") references in summarizer_agent's params
        # - depends_on=["reviewer"] references in summarizer_agent
        _run(ws, "rename", "--old-name", "reviewer", "--to", "validator")
        source = _read_source(ws)
        ns = _exec_code(source)

        # Old name completely gone.
        assert "reviewer" not in source
        # New name present.
        assert "validator = Agent(" in source
        assert 'name="validator"' in source
        assert _has_step_call(source, "validator")
        assert ns["validator"].model == "openai/gpt-4o"
        assert ns["validator"].max_iter == 5
        assert ns["validator"].system_prompt == "You review and validate outputs."
        assert len(ns["validator"].tools) == 2

        # step_span and depends_on references updated.
        assert 'step_span("validator")' in source
        assert '"reviewer"' not in source
        normalized = " ".join(source.split())
        assert '"validator"' in normalized

        # ===============================================================
        # Phase 9: Remove tools from a step selectively
        # ===============================================================

        # -- 9a. Remove WebSearch from validator, keep Read --
        _run(ws, "remove-tool", "--name", "web_search", "--step", "validator")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "web_search" not in [t.name for t in ns["validator"].tools]
        assert "read" in [t.name for t in ns["validator"].tools]
        assert len(ns["validator"].tools) == 1
        # validator config preserved.
        assert ns["validator"].model == "openai/gpt-4o"

        # -- 9b. Remove Read from validator --
        _run(ws, "remove-tool", "--name", "read", "--step", "validator")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert len(ns["validator"].tools) == 0
        assert ns["validator"].model == "openai/gpt-4o"

        # -- 9c. Add a tool back to validator --
        _run(ws, "add-tool", "--type", "Write", "--step", "validator")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "write" in [t.name for t in ns["validator"].tools]
        assert len(ns["validator"].tools) == 1

        # ===============================================================
        # Phase 10: Remove steps and verify cleanup
        # ===============================================================

        assert _count_step_calls(source) == 5

        # -- 10a. Remove postprocessor --
        _run(ws, "remove-step", "--name", "postprocessor")
        source = _read_source(ws)

        assert "postprocessor" not in source
        assert "def postprocessor" not in source
        assert _count_step_calls(source) == 4

        # All other steps still intact.
        assert "preprocessor" in source
        assert "validator" in source
        assert "summarizer_agent" in source
        assert "agent_step" in source

        # -- 10b. Remove summarizer_agent --
        _run(ws, "remove-step", "--name", "summarizer_agent")
        source = _read_source(ws)

        assert "summarizer_agent" not in source
        assert _count_step_calls(source) == 3
        # count_words custom function should be cleaned up (it was only on summarizer_agent).
        assert "def count_words" not in source

        # Remaining steps still intact.
        ns = _exec_code(source)
        assert ns["validator"].model == "openai/gpt-4o"
        assert "write" in [t.name for t in ns["validator"].tools]

        # -- 10c. Remove preprocessor --
        _run(ws, "remove-step", "--name", "preprocessor")
        source = _read_source(ws)

        assert "def preprocessor_fn" not in source
        assert "preprocessor" not in source
        assert _count_step_calls(source) == 2

        # -- 10d. Remove validator --
        _run(ws, "remove-step", "--name", "validator")
        source = _read_source(ws)

        assert "validator" not in source
        assert _count_step_calls(source) == 1
        # Only the original agent_step should remain.
        assert _has_step_call(source, "agent_step")

        # ===============================================================
        # Phase 11: Verify the original agent step survived intact
        # ===============================================================
        ns = _exec_code(source)
        assert isinstance(ns["agent"], Workflow)
        assert ns["agent"].name == "pipeline"
        assert ns["agent_step"].model == "openai/gpt-4o"
        assert ns["agent_step"].system_prompt == "You are a data processing pipeline."
        assert ns["agent_step"].max_iter == 10
        # Original agent_step should still have its 3 tools.
        tool_names = _tool_names(ns["agent_step"])
        assert "extract_keywords" in tool_names
        assert "format_output" in tool_names
        assert "search" in tool_names
        assert len(tool_names) == 3

        # ===============================================================
        # Phase 12: Rebuild — add steps back and verify fresh state
        # ===============================================================

        # -- 12a. Re-add preprocessor with updated body --
        new_preprocess_def = (
            "def preprocessor(text: str) -> dict:\n"
            "    cleaned = ' '.join(text.strip().lower().split())\n"
            "    return {'cleaned': cleaned, 'length': len(cleaned)}"
        )
        _run(ws, "add-step", "--type", "Custom", "--definition", new_preprocess_def)
        source = _read_source(ws)

        assert "def preprocessor_fn(text: str) -> dict:" in source
        assert _has_step_call(source, "preprocessor")
        assert _count_step_calls(source) == 2

        # Verify the updated function body.
        ns = _exec_code(source)
        result = ns["preprocessor_fn"]("  Hello   World  ")
        assert result == {"cleaned": "hello world", "length": 11}

        # -- 12b. Add a new agent step, wire edges separately --
        final_config = json.dumps({"name": "final_agent", "model": "openai/gpt-4o"})
        _run(ws, "add-step", "--type", "Agent", "--config", final_config)
        _run(ws, "set-param", "--target", "final_agent", "--name", "prompt", "--type", "map", "--source", "preprocessor", "--key", "output.cleaned")
        _run(ws, "add-edge", "--source", "agent", "--target", "final_agent")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert ns["final_agent"].model == "openai/gpt-4o"
        assert 'step_span("preprocessor").output.cleaned' in source
        assert _count_step_calls(source) == 3

        # ===============================================================
        # Phase 13: Rename the original agent step
        # ===============================================================

        _run(ws, "rename", "--old-name", "agent", "--to", "intake_agent")
        source = _read_source(ws)
        ns = _exec_code(source)

        assert "agent_step" not in source
        assert "intake_agent = Agent(" in source
        assert 'name="intake_agent"' in source
        assert _has_step_call(source, "intake_agent")
        assert ns["intake_agent"].model == "openai/gpt-4o"
        assert ns["intake_agent"].system_prompt == "You are a data processing pipeline."
        assert len(ns["intake_agent"].tools) == 3

        # References in depends_on should be updated too.
        # (The depends_on had "agent" which is now "intake_agent".)
        normalized = " ".join(source.split())
        assert '"intake_agent"' in normalized

        # ===============================================================
        # Phase 14: Error cases — operations that should fail
        # ===============================================================

        # -- 14a. Cannot rename the entry point --
        err = _run_err(ws, "rename", "--old-name", "pipeline", "--to", "new_pipeline")
        assert "entry point" in err.lower() or "not found" in err.lower()

        # -- 14b. Cannot add-step on non-existent type --
        err = _run_err(ws, "add-step", "--type", "NonExistentType")
        assert "must be one of" in err.lower() or "nonexistenttype" in err.lower()

        # ===============================================================
        # Final validation
        # ===============================================================
        source = _read_source(ws)
        ns = _exec_code(source)
        assert isinstance(ns["agent"], Workflow)
        assert ns["agent"].name == "pipeline"
        assert _count_step_calls(source) == 3
