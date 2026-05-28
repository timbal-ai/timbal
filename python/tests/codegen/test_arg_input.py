"""Tests for the @path/- arg_input convention used by codegen CLI args."""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from .conftest import codegen_cmd

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'


@pytest.fixture
def workspace(tmp_path):
    def _write(source: str) -> Path:
        (tmp_path / "agent.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
        return tmp_path

    return _write


def _run(workspace_path: Path, *cli_args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        codegen_cmd("--path", str(workspace_path), "--dry-run", *cli_args),
        input=stdin,
        capture_output=True,
        text=True,
    )


def _exec_agent(code: str) -> dict:
    ns: dict = {}
    exec(code, ns)
    return ns


# ---------------------------------------------------------------------------
# @path file redirection
# ---------------------------------------------------------------------------


class TestAtPath:
    def test_set_config_reads_config_from_file(self, workspace, tmp_path):
        """`--config @file.json` slurps JSON from disk, sidestepping shell quoting."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        # Use a payload with parens, embedded \n, and quotes — the kind of
        # blob that breaks under bash single-quoting.
        config = {"system_prompt": "You are helpful.\n\nAlways answer (succinctly)."}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = _run(ws, "set-config", "--config", f"@{config_file}")
        assert result.returncode == 0, result.stderr
        ns = _exec_agent(result.stdout)
        assert ns["agent"].system_prompt == "You are helpful.\n\nAlways answer (succinctly)."

    def test_add_tool_reads_config_from_file(self, workspace, tmp_path):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        config_file = tmp_path / "tool.json"
        config_file.write_text(json.dumps({"allowed_domains": ["example.com"]}))

        result = _run(ws, "add-tool", "--type", "WebSearch", "--config", f"@{config_file}")
        assert result.returncode == 0, result.stderr
        ns = _exec_agent(result.stdout)
        ws_tool = next(t for t in ns["agent"].tools if t.name == "web_search")
        assert ws_tool.allowed_domains == ["example.com"]

    def test_add_tool_reads_definition_from_file(self, workspace, tmp_path):
        """`--definition @file.py` for Custom tools — the worst quoting case."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        def_file = tmp_path / "fn.py"
        def_file.write_text("def my_search(query: str) -> str:\n    return query.upper()\n")

        result = _run(ws, "add-tool", "--type", "Custom", "--definition", f"@{def_file}")
        assert result.returncode == 0, result.stderr
        ns = _exec_agent(result.stdout)
        assert "my_search" in [t.name for t in ns["agent"].tools]

    def test_missing_file_reports_clear_error(self, workspace, tmp_path):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        missing = tmp_path / "does-not-exist.json"
        result = _run(ws, "set-config", "--config", f"@{missing}")
        assert result.returncode != 0
        assert "cannot read" in result.stderr or "No such file" in result.stderr


# ---------------------------------------------------------------------------
# - stdin redirection
# ---------------------------------------------------------------------------


class TestStdin:
    def test_set_config_reads_from_stdin(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        config = json.dumps({"model": "openai/gpt-4o"})
        result = _run(ws, "set-config", "--config", "-", stdin=config)
        assert result.returncode == 0, result.stderr
        ns = _exec_agent(result.stdout)
        assert ns["agent"].model == "openai/gpt-4o"

    def test_add_tool_reads_definition_from_stdin(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.upper()\n"
        result = _run(
            ws, "add-tool", "--type", "Custom", "--definition", "-", stdin=definition
        )
        assert result.returncode == 0, result.stderr
        ns = _exec_agent(result.stdout)
        assert "my_search" in [t.name for t in ns["agent"].tools]


# ---------------------------------------------------------------------------
# Plain literal still works
# ---------------------------------------------------------------------------


class TestLiteralPassthrough:
    def test_inline_json_unchanged(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        result = _run(ws, "set-config", "--config", '{"model": "openai/gpt-4o"}')
        assert result.returncode == 0, result.stderr
        ns = _exec_agent(result.stdout)
        assert ns["agent"].model == "openai/gpt-4o"
