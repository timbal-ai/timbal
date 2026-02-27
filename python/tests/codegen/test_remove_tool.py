import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def source_file(tmp_path):
    """Write a source file and return its path."""

    def _write(source: str) -> Path:
        p = tmp_path / "agent.py"
        p.write_text(textwrap.dedent(source))
        return p

    return _write


def _run_dry(source_path: Path, tool_name: str) -> str:
    """Run codegen remove-tool with --dry-run and return stdout."""
    fqn = f"{source_path}::agent"
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", fqn, "--dry-run", "remove-tool", tool_name],
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
    def test_removes_inline_web_search(self, source_file):
        """Remove WebSearch() from tools=[WebSearch()]."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(p, "web_search")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_one_keeps_others(self, source_file):
        """Remove WebSearch but keep other tools intact."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch, Edit

        def my_func(x: int) -> int:
            return x + 1

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_func, WebSearch(), Edit()])
        """)
        output = _run_dry(p, "web_search")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" not in tool_names
        assert "my_func" in tool_names
        assert "edit" in tool_names

    def test_removes_framework_tool_cleans_import(self, source_file):
        """After removing the only usage of WebSearch, ruff should clean the import."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(p, "web_search")
        assert "from timbal.tools import WebSearch" not in output


class TestRemoveCustomTool:
    def test_removes_bare_function_reference(self, source_file):
        """Remove a bare function name from tools list."""
        p = source_file("""\
        from timbal.core import Agent

        def my_search(query: str) -> str:
            return query.upper()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        output = _run_dry(p, "my_search")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_tool_wrapper_by_resolved_name(self, source_file):
        """Remove tool identified by its name= kwarg in Tool(name=..., handler=...)."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def search_impl(query: str) -> str:
            return query.upper()

        searcher = Tool(name="my_search", handler=search_impl)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[searcher])
        """)
        output = _run_dry(p, "my_search")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_correct_tool_among_mixed(self, source_file):
        """Remove one custom tool from a mixed list of framework + custom tools."""
        p = source_file("""\
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
        output = _run_dry(p, "calc")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "calc" not in tool_names
        assert "add" in tool_names
        assert "web_search" in tool_names

    def test_removes_by_handler_name_fallback(self, source_file):
        """Tool(handler=foo) without name= kwarg resolves to handler name 'foo'."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def foo(x: str) -> str:
            return x

        wrapper = Tool(handler=foo)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[wrapper])
        """)
        output = _run_dry(p, "foo")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0


class TestRemoveEdgeCases:
    def test_noop_when_tool_not_found(self, source_file):
        """Removing a tool that doesn't exist should be a no-op."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(p, "nonexistent_tool")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" in tool_names

    def test_noop_when_no_tools_kwarg(self, source_file):
        """Removing from an agent with no tools kwarg should be a no-op."""
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(p, "whatever")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_removes_last_tool_leaves_empty_list(self, source_file):
        """Removing the only tool should leave tools=[]."""
        p = source_file("""\
        from timbal.core import Agent

        def only_tool():
            pass

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[only_tool])
        """)
        output = _run_dry(p, "only_tool")
        ns = _exec_agent(output)
        assert len(ns["agent"].tools) == 0

    def test_remove_then_add_roundtrip(self, source_file):
        """Remove a tool then add it back — should end up with just that tool."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch, Edit

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch(), Edit()])
        """)
        # First remove WebSearch.
        fqn = f"{p}::agent"
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "remove-tool", "web_search"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"remove failed:\n{result.stderr}"

        # Then add it back with --dry-run to inspect.
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "--dry-run", "add-tool", "--type", "WebSearch"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"add failed:\n{result.stderr}"
        ns = _exec_agent(result.stdout)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" in tool_names
        assert "edit" in tool_names

    def test_remove_multiple_sequentially(self, source_file):
        """Remove two tools in sequence, verify only the third remains."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch, Edit

        def custom():
            pass

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[custom, WebSearch(), Edit()])
        """)
        fqn = f"{p}::agent"

        # Remove WebSearch (writes to file).
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "remove-tool", "web_search"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Remove custom (writes to file).
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "remove-tool", "custom"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Dry-run to inspect final state.
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "--dry-run", "remove-tool", "nonexistent"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        ns = _exec_agent(result.stdout)
        tool_names = [t.name for t in ns["agent"].tools]
        assert tool_names == ["edit"]
