import json
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
    """Run codegen set-config with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "set-config", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_fail(workspace_path: Path, *cli_args: str) -> subprocess.CompletedProcess:
    """Run codegen set-config with --dry-run and expect failure."""
    return subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "set-config", *cli_args],
        capture_output=True,
        text=True,
    )


def _exec_agent(code: str) -> dict:
    """Exec the generated code and return its globals."""
    ns = {}
    exec(code, ns)
    return ns


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_set_model(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"model": "openai/gpt-4o"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"

    def test_set_system_prompt(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"system_prompt": "You are helpful."})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == "You are helpful."

    def test_set_name(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"name": "my_agent"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].name == "my_agent"

    def test_set_description(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"description": "A helpful agent"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].description == "A helpful agent"

    def test_set_max_iter(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"max_iter": 5})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].max_iter == 5

    def test_set_model_params(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"model_params": {"temperature": 0.7, "max_tokens": 1024}})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model_params == {"temperature": 0.7, "max_tokens": 1024}

    def test_set_skills_path(self, workspace):
        """skills_path kwarg is set in the generated source."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"skills_path": "./skills"})
        output = _run_dry(ws, "--config", config)
        assert 'skills_path="./skills"' in output

    def test_remove_field_with_null(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", system_prompt="old prompt")
        """)
        config = json.dumps({"system_prompt": None})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt is None

    def test_multiple_fields_at_once(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"model": "openai/gpt-4o", "max_iter": 20, "system_prompt": "Be concise."})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].max_iter == 20
        assert ns["agent"].system_prompt == "Be concise."

    def test_omitted_fields_unchanged(self, workspace):
        """Fields not in the config JSON are left unchanged."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", system_prompt="keep me")
        """)
        config = json.dumps({"max_iter": 3})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == "keep me"
        assert ns["agent"].max_iter == 3

    def test_rejects_unknown_fields(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"invalid_field": "value"})
        result = _run_dry_fail(ws, "--config", config)
        assert result.returncode != 0
        assert "Unknown agent config field(s)" in result.stderr


# ---------------------------------------------------------------------------
# Tool config
# ---------------------------------------------------------------------------


class TestToolConfig:
    def test_set_web_search_config(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"allowed_domains": ["example.com"], "blocked_domains": ["spam.com"]})
        output = _run_dry(ws, "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.config.allowed_domains == ["example.com"]
        assert ws_tool.config.blocked_domains == ["spam.com"]

    def test_updates_existing_config(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["old.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"allowed_domains": ["new.com"]})
        output = _run_dry(ws, "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.config.allowed_domains == ["new.com"]

    def test_full_replace_semantics(self, workspace):
        """Config fully replaces — old fields not in new config are removed."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["old.com"], blocked_domains=["spam.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"allowed_domains": ["new.com"]})
        output = _run_dry(ws, "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.config.allowed_domains == ["new.com"]
        assert ws_tool.config.blocked_domains is None

    def test_rejects_unknown_config_fields(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"invalid_field": "value"})
        result = _run_dry_fail(ws, "web_search", "--config", config)
        assert result.returncode != 0
        assert "Unknown config field(s)" in result.stderr

    def test_rejects_config_for_unconfigurable_tool(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import Edit

        edit = Edit()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[edit])
        """)
        config = json.dumps({"some_param": "value"})
        result = _run_dry_fail(ws, "edit", "--config", config)
        assert result.returncode != 0
        assert "not supported" in result.stderr

    def test_tool_not_found(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config = json.dumps({"allowed_domains": ["example.com"]})
        result = _run_dry_fail(ws, "web_search", "--config", config)
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_inline_tool_migration(self, workspace):
        """Inline WebSearch() is migrated to variable with config applied."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[WebSearch()])
        """)
        config = json.dumps({"allowed_domains": ["example.com"]})
        output = _run_dry(ws, "web_search", "--config", config)
        assert "tools=[web_search]" in output
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.config.allowed_domains == ["example.com"]
