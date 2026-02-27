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


def _run_dry(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen add-tool with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "add-tool", *cli_args],
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


class TestFrameworkTool:
    def test_adds_web_search_as_variable(self, workspace):
        """WebSearch is assigned to a variable, referenced by name in tools list."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        assert "web_search = WebSearch()" in output
        assert "tools=[web_search]" in output
        ns = _exec_agent(output)
        assert "web_search" in [t.name for t in ns["agent"].tools]

    def test_no_tools_kwarg(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        ns = _exec_agent(output)
        assert "web_search" in [t.name for t in ns["agent"].tools]

    def test_idempotent(self, workspace):
        """Re-adding an existing variable-style tool doesn't duplicate it."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("web_search") == 1

    def test_migrates_inline_to_variable(self, workspace):
        """An inline WebSearch() in tools list is migrated to variable style."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        assert "web_search = WebSearch()" in output
        assert "tools=[web_search]" in output
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("web_search") == 1


class TestCustomTool:
    def test_adds_custom_function_with_tool_wrapper(self, workspace):
        """Custom tools get a Tool() wrapper variable."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.upper()"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition)
        assert "from timbal.core import Tool" in output
        assert 'Tool(name="my_search", handler=my_search)' in output
        ns = _exec_agent(output)
        assert "my_search" in [t.name for t in ns["agent"].tools]

    def test_idempotent(self, workspace):
        """Re-adding an existing custom tool updates the definition."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def my_search(query: str) -> str:
            return query.upper()

        my_search = Tool(name="my_search", handler=my_search)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("my_search") == 1

    def test_updates_handler_in_tool_wrapper(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def my_search(query: str) -> str:
            return query.upper()

        my_search = Tool(name="my_search", handler=my_search)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        assert "my_search" in [t.name for t in ns["agent"].tools]

    def test_definition_required(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", "--path", str(ws), "--dry-run", "add-tool", "--type", "Custom"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
