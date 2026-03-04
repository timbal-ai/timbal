import subprocess
import textwrap
from pathlib import Path

import pytest

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'


@pytest.fixture
def workspace(tmp_path):
    """Write a source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "agent.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
        return tmp_path

    return _write


def _run_dry(workspace_path: Path, tool_name: str) -> str:
    """Run codegen remove-tool with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "remove-tool", tool_name],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _exec_agent(code: str) -> dict:
    """Exec the generated code and return its globals."""
    ns = {}
    exec(code, ns)
    return ns


class TestRemoveFrameworkTool:
    def test_removes_inline_web_search(self, workspace):
        """Remove WebSearch() from tools=[WebSearch()]."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(ws, "web_search")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_one_keeps_others(self, workspace):
        """Remove WebSearch but keep other tools intact."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch, Edit

        def my_func(x: int) -> int:
            return x + 1

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_func, WebSearch(), Edit()])
        """)
        output = _run_dry(ws, "web_search")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" not in tool_names
        assert "my_func" in tool_names
        assert "edit" in tool_names

    def test_removes_framework_tool_cleans_import(self, workspace):
        """After removing the only usage of WebSearch, ruff should clean the import."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(ws, "web_search")
        assert "from timbal.tools import WebSearch" not in output


class TestRemoveCustomTool:
    def test_removes_bare_function_reference(self, workspace):
        """Remove a bare function name from tools list."""
        ws = workspace("""\
        from timbal.core import Agent

        def my_search(query: str) -> str:
            return query.upper()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        output = _run_dry(ws, "my_search")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_tool_wrapper_by_resolved_name(self, workspace):
        """Remove tool identified by its name= kwarg in Tool(name=..., handler=...)."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def search_impl(query: str) -> str:
            return query.upper()

        searcher = Tool(name="my_search", handler=search_impl)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[searcher])
        """)
        output = _run_dry(ws, "my_search")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_correct_tool_among_mixed(self, workspace):
        """Remove one custom tool from a mixed list of framework + custom tools."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core import Tool
        from timbal.tools import WebSearch

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        calculator = Tool(name="calc", handler=multiply)

        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            tools=[add, WebSearch(), calculator],
        )
        """)
        output = _run_dry(ws, "calc")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "calc" not in tool_names
        assert "add" in tool_names
        assert "web_search" in tool_names

    def test_removes_by_handler_name_fallback(self, workspace):
        """Tool(handler=foo) without name= kwarg resolves to handler name 'foo'."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def foo(x: str) -> str:
            return x

        wrapper = Tool(handler=foo)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[wrapper])
        """)
        output = _run_dry(ws, "foo")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0


class TestRemoveEdgeCases:
    def test_noop_when_tool_not_found(self, workspace):
        """Removing a tool that doesn't exist should be a no-op."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(ws, "nonexistent_tool")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" in tool_names

    def test_noop_when_no_tools_kwarg(self, workspace):
        """Removing from an agent with no tools kwarg should be a no-op."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws, "whatever")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_last_tool_leaves_empty_list(self, workspace):
        """Removing the only tool should leave tools=[]."""
        ws = workspace("""\
        from timbal.core import Agent

        def only_tool():
            pass

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[only_tool])
        """)
        output = _run_dry(ws, "only_tool")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_remove_then_add_roundtrip(self, workspace):
        """Remove a tool then add it back — should end up with just that tool."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch, Edit

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch(), Edit()])
        """)
        # First remove WebSearch.
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "remove-tool", "web_search"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"remove failed:\n{result.stderr}"

        # Then add it back with --dry-run to inspect.
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "--dry-run", "add-tool", "--type", "WebSearch"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"add failed:\n{result.stderr}"
        ns = _exec_agent(result.stdout)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" in tool_names
        assert "edit" in tool_names

    def test_remove_multiple_sequentially(self, workspace):
        """Remove two tools in sequence, verify only the third remains."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch, Edit

        def custom():
            pass

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[custom, WebSearch(), Edit()])
        """)

        # Remove WebSearch (writes to file).
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "remove-tool", "web_search"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Remove custom (writes to file).
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "remove-tool", "custom"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Dry-run to inspect final state.
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "--dry-run", "remove-tool", "nonexistent"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        ns = _exec_agent(result.stdout)
        tool_names = [t.name for t in ns["agent"].tools]
        assert tool_names == ["edit"]


# ---------------------------------------------------------------------------
# --step flag (Workflow entry point)
# ---------------------------------------------------------------------------

WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def wf_workspace(tmp_path):
    """Write a workflow source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


def _run_dry_wf(workspace_path: Path, tool_name: str, *extra_args: str) -> str:
    """Run codegen remove-tool with --dry-run on a workflow workspace."""
    result = subprocess.run(
        [
            "python", "-m", "timbal.codegen", "--path", str(workspace_path),
            "--dry-run", "remove-tool", tool_name, *extra_args,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


class TestRemoveToolStep:
    def test_removes_tool_from_workflow_step(self, wf_workspace):
        """Remove a tool from a specific Agent step in a Workflow."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.tools import WebSearch

        web_search = WebSearch()
        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", tools=[web_search])

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry_wf(ws, "web_search", "--step", "agent_a")
        assert "tools=[]" in output or "tools" not in output

    def test_removes_one_tool_keeps_others_on_step(self, wf_workspace):
        """Remove one tool from a step that has multiple tools."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.tools import WebSearch, Edit

        web_search = WebSearch()
        edit = Edit()
        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", tools=[web_search, edit])

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry_wf(ws, "web_search", "--step", "agent_a")
        assert "edit" in output
        assert "tools=[edit]" in output

    def test_rejects_step_on_agent_entry_point(self, workspace):
        """--step should fail when the entry point is an Agent."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        result = subprocess.run(
            [
                "python", "-m", "timbal.codegen", "--path", str(ws), "--dry-run",
                "remove-tool", "web_search", "--step", "agent_a",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--step requires a Workflow" in result.stderr
