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

    def test_set_max_tokens(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"max_tokens": 1024})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].max_tokens == 1024

    def test_set_model_params_backward_compat(self, workspace):
        """model_params still works but values get denormalized into individual fields."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"model_params": {"cache_control": {"type": "ephemeral"}, "max_tokens": 1024}})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].max_tokens == 1024
        assert ns["agent"].model_params == {"cache_control": {"type": "ephemeral"}}

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
        output = _run_dry(ws, "--name", "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.allowed_domains == ["example.com"]
        assert ws_tool.blocked_domains == ["spam.com"]

    def test_updates_existing_config(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["old.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"allowed_domains": ["new.com"]})
        output = _run_dry(ws, "--name", "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.allowed_domains == ["new.com"]

    def test_omitted_fields_unchanged(self, workspace):
        """Fields not in the config JSON are left unchanged (partial update semantics)."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["old.com"], blocked_domains=["spam.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"allowed_domains": ["new.com"]})
        output = _run_dry(ws, "--name", "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.allowed_domains == ["new.com"]
        assert ws_tool.blocked_domains == ["spam.com"]

    def test_rejects_unknown_config_fields(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"invalid_field": "value"})
        result = _run_dry_fail(ws, "--name", "web_search", "--config", config)
        assert result.returncode != 0
        assert "Unknown config field(s)" in result.stderr

    def test_rejects_unknown_config_field_for_tool(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import Edit

        edit = Edit()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[edit])
        """)
        config = json.dumps({"some_param": "value"})
        result = _run_dry_fail(ws, "--name", "edit", "--config", config)
        assert result.returncode != 0
        assert "Unknown config field(s)" in result.stderr

    def test_tool_not_found(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config = json.dumps({"allowed_domains": ["example.com"]})
        result = _run_dry_fail(ws, "--name", "web_search", "--config", config)
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
        output = _run_dry(ws, "--name", "web_search", "--config", config)
        assert "tools=[web_search]" in output
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.allowed_domains == ["example.com"]


# ---------------------------------------------------------------------------
# Tool name and description
# ---------------------------------------------------------------------------


class TestToolNameAndDescription:
    def test_set_web_search_name(self, workspace):
        """Setting name on WebSearch preserves other config."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["example.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"name": "my_search"})
        output = _run_dry(ws, "--name", "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = ns["web_search"]
        assert ws_tool.name == "my_search"
        assert ws_tool.allowed_domains == ["example.com"]

    def test_set_web_search_description(self, workspace):
        """Setting description on WebSearch preserves other config."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["example.com"])

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        config = json.dumps({"description": "Custom search tool"})
        output = _run_dry(ws, "--name", "web_search", "--config", config)
        ns = _exec_agent(output)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.description == "Custom search tool"
        assert ws_tool.allowed_domains == ["example.com"]

    def test_set_cala_search_name(self, workspace):
        """Setting name on CalaSearch preserves other config."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import CalaSearch

        cala_search = CalaSearch(api_key="test-key", base_url="https://custom.api/v1")

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[cala_search])
        """)
        config = json.dumps({"name": "my_cala"})
        output = _run_dry(ws, "--name", "cala_search", "--config", config)
        ns = _exec_agent(output)
        cs_tool = ns["cala_search"]
        assert cs_tool.name == "my_cala"
        assert cs_tool.api_key.get_secret_value() == "test-key"
        assert cs_tool.base_url == "https://custom.api/v1"

    def test_set_cala_search_description(self, workspace):
        """Setting description on CalaSearch preserves other config."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import CalaSearch

        cala_search = CalaSearch(api_key="test-key")

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[cala_search])
        """)
        config = json.dumps({"description": "Custom knowledge search"})
        output = _run_dry(ws, "--name", "cala_search", "--config", config)
        ns = _exec_agent(output)
        cs_tool = next(t for t in ns["agent"].tools if t.name == "cala_search")
        assert cs_tool.description == "Custom knowledge search"
        assert cs_tool.api_key.get_secret_value() == "test-key"

    def test_set_custom_tool_name(self, workspace):
        """Setting name on a custom Tool preserves handler and description."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core.tool import Tool

        def my_func():
            return "hello"

        my_tool = Tool(handler=my_func, name="my_tool", description="Original desc")

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_tool])
        """)
        config = json.dumps({"name": "renamed_tool"})
        output = _run_dry(ws, "--name", "my_tool", "--config", config)
        ns = _exec_agent(output)
        tool = ns["my_tool"]
        assert tool.name == "renamed_tool"
        assert tool.description == "Original desc"

    def test_set_custom_tool_description(self, workspace):
        """Setting description on a custom Tool preserves handler and name."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core.tool import Tool

        def my_func():
            return "hello"

        my_tool = Tool(handler=my_func, name="my_tool")

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[my_tool])
        """)
        config = json.dumps({"description": "Does something useful"})
        output = _run_dry(ws, "--name", "my_tool", "--config", config)
        ns = _exec_agent(output)
        tool = ns["my_tool"]
        assert tool.name == "my_tool"
        assert tool.description == "Does something useful"


# ---------------------------------------------------------------------------
# Workflow step config
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


def _run_dry_wf(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen set-config with --dry-run and return stdout."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "set-config", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _run_dry_wf_fail(workspace_path: Path, *cli_args: str) -> subprocess.CompletedProcess:
    """Run codegen set-config with --dry-run and expect failure."""
    return subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "set-config", *cli_args],
        capture_output=True,
        text=True,
    )


class TestStepConstructorConfig:
    def test_set_step_model(self, wf_workspace):
        """Update a step's constructor config (e.g. model)."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="my_workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry_wf(ws, "--name", "agent_a", "--config", '{"model": "openai/gpt-4o"}')
        assert 'model="openai/gpt-4o"' in output
