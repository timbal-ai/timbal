import json
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


def _run_dry(source_path: Path, *cli_args: str) -> str:
    """Run codegen add-tool with --dry-run and return stdout."""
    fqn = f"{source_path}::agent"
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", fqn, "--dry-run", "add-tool", *cli_args],
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
    def test_adds_web_search_as_variable(self, source_file):
        """WebSearch is assigned to a variable, referenced by name in tools list."""
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        assert "web_search = WebSearch()" in output
        assert "tools=[web_search]" in output
        ns = _exec_agent(output)
        assert "web_search" in [t.name for t in ns["agent"].tools]

    def test_no_tools_kwarg(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        assert "web_search" in [t.name for t in ns["agent"].tools]

    def test_idempotent(self, source_file):
        """Re-adding an existing variable-style tool doesn't duplicate it."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("web_search") == 1

    def test_migrates_inline_to_variable(self, source_file):
        """An inline WebSearch() in tools list is migrated to variable style."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        assert "web_search = WebSearch()" in output
        assert "tools=[web_search]" in output
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("web_search") == 1

    def test_custom_name(self, source_file):
        """--name sets the variable name for framework tools."""
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(p, "--type", "WebSearch", "--name", "my_ws")
        assert "my_ws = WebSearch()" in output
        assert "tools=[my_ws]" in output
        ns = _exec_agent(output)
        assert "web_search" in [t.name for t in ns["agent"].tools]


class TestToolConfig:
    def test_adds_web_search_with_config(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config = json.dumps({"allowed_domains": ["example.com"], "blocked_domains": ["spam.com"]})
        output = _run_dry(p, "--type", "WebSearch", "--config", config)
        ns = _exec_agent(output)
        ws = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws.config.allowed_domains == ["example.com"]
        assert ws.config.blocked_domains == ["spam.com"]

    def test_updates_existing_tool_config(self, source_file):
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["old.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"allowed_domains": ["new.com"]})
        output = _run_dry(p, "--type", "WebSearch", "--config", config)
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("web_search") == 1
        ws = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws.config.allowed_domains == ["new.com"]

    def test_clears_config_when_no_config_passed(self, source_file):
        """Re-adding a tool without --config strips previously set params."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["old.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        ws = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws.config.allowed_domains is None

    def test_rejects_unknown_config_fields(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config = json.dumps({"invalid_field": "value"})
        fqn = f"{p}::agent"
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "--dry-run", "add-tool", "--type", "WebSearch", "--config", config],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Unknown config field(s)" in result.stderr

    def test_rejects_config_for_tool_without_config_model(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config = json.dumps({"some_param": "value"})
        fqn = f"{p}::agent"
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "--dry-run", "add-tool", "--type", "Edit", "--config", config],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not supported" in result.stderr


class TestDescription:
    def test_framework_tool_with_description(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(p, "--type", "WebSearch", "--description", "Search the web")
        assert 'description="Search the web"' in output
        ns = _exec_agent(output)
        ws = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws.description == "Search the web"

    def test_custom_tool_with_description(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.upper()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition, "--description", "My custom search")
        assert 'description="My custom search"' in output
        ns = _exec_agent(output)
        tool = next(t for t in ns["agent"].tools if t.name == "my_search")
        assert tool.description == "My custom search"


class TestCustomTool:
    def test_adds_custom_function_with_tool_wrapper(self, source_file):
        """Custom tools get a Tool() wrapper variable."""
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.upper()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        assert "from timbal.core import Tool" in output
        assert 'Tool(name="my_search", handler=my_search)' in output
        ns = _exec_agent(output)
        assert "my_search" in [t.name for t in ns["agent"].tools]

    def test_custom_tool_with_name(self, source_file):
        """--name sets both the variable name and runtime name for custom tools."""
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def search_impl(query: str) -> str:\n    return query.upper()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition, "--name", "search")
        assert 'search = Tool(name="search", handler=search_impl)' in output
        assert "tools=[search]" in output
        ns = _exec_agent(output)
        assert "search" in [t.name for t in ns["agent"].tools]

    def test_idempotent(self, source_file):
        """Re-adding an existing custom tool updates the definition."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def my_search(query: str) -> str:
            return query.upper()

        my_search = Tool(name="my_search", handler=my_search)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        assert [t.name for t in ns["agent"].tools].count("my_search") == 1

    def test_updates_handler_in_tool_wrapper(self, source_file):
        p = source_file("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def my_search(query: str) -> str:
            return query.upper()

        my_search = Tool(name="my_search", handler=my_search)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        assert "my_search" in [t.name for t in ns["agent"].tools]

    def test_updates_handler_renamed_function(self, source_file):
        p = source_file("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def old_search(query: str) -> str:
            return query.upper()

        searcher = Tool(name="my_search", handler=old_search)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[searcher])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        assert "my_search" in [t.name for t in ns["agent"].tools]
        assert "old_search" not in ns

    def test_definition_required(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        fqn = f"{p}::agent"
        result = subprocess.run(
            ["python", "-m", "timbal.codegen", fqn, "--dry-run", "add-tool", "--type", "Custom"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
