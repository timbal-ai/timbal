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


class TestWebSearch:
    def test_adds_web_search(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert "web_search" in tool_names

    def test_no_tools_kwarg_web_search(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert "web_search" in tool_names

    def test_idempotent_web_search(self, source_file):
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert tool_names.count("web_search") == 1


class TestToolConfig:
    def test_adds_web_search_with_config(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config = json.dumps({"allowed_domains": ["example.com"], "blocked_domains": ["spam.com"]})
        output = _run_dry(p, "--type", "WebSearch", "--config", config)
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert "web_search" in tool_names
        ws = next(t for t in agent.tools if t.name == "web_search")
        assert ws.config.allowed_domains == ["example.com"]
        assert ws.config.blocked_domains == ["spam.com"]

    def test_updates_existing_tool_config(self, source_file):
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch(allowed_domains=["old.com"])])
        """)
        config = json.dumps({"allowed_domains": ["new.com"]})
        output = _run_dry(p, "--type", "WebSearch", "--config", config)
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert tool_names.count("web_search") == 1
        ws = next(t for t in agent.tools if t.name == "web_search")
        assert ws.config.allowed_domains == ["new.com"]

    def test_adds_tool_with_no_config(self, source_file):
        """Adding a tool without --config still works as before."""
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        agent = ns["agent"]
        ws = next(t for t in agent.tools if t.name == "web_search")
        assert ws.config.allowed_domains is None

    def test_clears_config_when_no_config_passed(self, source_file):
        """Re-adding a tool without --config strips previously set params."""
        p = source_file("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch(allowed_domains=["old.com"])])
        """)
        output = _run_dry(p, "--type", "WebSearch")
        ns = _exec_agent(output)
        agent = ns["agent"]
        assert [t.name for t in agent.tools].count("web_search") == 1
        ws = next(t for t in agent.tools if t.name == "web_search")
        assert ws.config.allowed_domains is None

    def test_rejects_unknown_config_fields(self, source_file):
        """Passing unknown config fields raises an error."""
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
        """Passing --config for a tool that doesn't support it raises an error."""
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


class TestCustomTool:
    def test_adds_custom_function(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.upper()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert "my_search" in tool_names
        assert ns["my_search"]("hello") == "HELLO"

    def test_idempotent(self, source_file):
        p = source_file("""\
        from timbal.core import Agent

        def my_search(query: str) -> str:
            return query.upper()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_search])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert tool_names.count("my_search") == 1
        # Definition is replaced with the new body.
        assert ns["my_search"]("Hello") == "hello"

    def test_updates_handler_in_tool_wrapper(self, source_file):
        p = source_file("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def my_search(query: str) -> str:
            return query.upper()

        searcher = Tool(name="my_search", handler=my_search)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[searcher])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.lower()"
        output = _run_dry(p, "--type", "Custom", "--definition", definition)
        ns = _exec_agent(output)
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert "my_search" in tool_names
        # The handler's underlying function is updated.
        assert ns["my_search"]("Hello") == "hello"

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
        agent = ns["agent"]
        tool_names = [t.name for t in agent.tools]
        assert "my_search" in tool_names
        # handler= is updated to the new function, old_search is removed by ruff.
        assert "old_search" not in ns
        assert ns["my_search"]("Hello") == "hello"

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
