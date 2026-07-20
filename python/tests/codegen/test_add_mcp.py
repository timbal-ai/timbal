import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from .conftest import codegen_cmd

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'


@pytest.fixture
def workspace(tmp_path):
    """Write a source file + timbal.yaml and return the workspace directory."""

    def _write(source: str) -> Path:
        (tmp_path / "agent.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(TIMBAL_YAML)
        return tmp_path

    return _write


AGENT_SOURCE = """\
from timbal.core import Agent

agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
"""


def _run(workspace_path: Path, *cli_args: str, dry_run: bool = True, expect_error: bool = False) -> str:
    argv = ["--path", str(workspace_path)]
    if dry_run:
        argv.append("--dry-run")
    result = subprocess.run(
        codegen_cmd(*argv, "add-mcp", *cli_args),
        capture_output=True,
        text=True,
    )
    if expect_error:
        assert result.returncode != 0, f"expected failure but succeeded:\n{result.stdout}"
        return result.stderr
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


def _exec_agent(code: str) -> dict:
    ns = {}
    exec(code, ns)
    return ns


class TestAddStdioServer:
    def test_basic(self, workspace):
        ws = workspace(AGENT_SOURCE)
        output = _run(
            ws,
            "--name", "fs",
            "--command", "npx",
            "--args", '["-y", "@modelcontextprotocol/server-filesystem", "."]',
        )
        # Ruff may merge this into the existing `from timbal.core import ...` import.
        assert "MCPServer" in output.split("\n\n")[0]
        assert "tools=[fs]" in output

        ns = _exec_agent(output)
        server = ns["agent"].tools[0]
        assert server.name == "fs"
        assert server.transport == "stdio"
        assert server.args == ["-y", "@modelcontextprotocol/server-filesystem", "."]

    def test_transport_inferred_from_command(self, workspace):
        ws = workspace(AGENT_SOURCE)
        output = _run(ws, "--name", "fs", "--command", "npx")
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].transport == "stdio"

    def test_env_placeholder_becomes_environ_lookup(self, workspace, monkeypatch):
        ws = workspace(AGENT_SOURCE)
        output = _run(
            ws,
            "--name", "gh",
            "--command", "npx",
            "--env", '{"GITHUB_TOKEN": "$MY_GH_TOKEN"}',
        )
        assert 'os.environ["MY_GH_TOKEN"]' in output
        assert "import os" in output

        monkeypatch.setenv("MY_GH_TOKEN", "secret-token")
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].env == {"GITHUB_TOKEN": "secret-token"}


class TestAddHttpServer:
    def test_basic(self, workspace):
        ws = workspace(AGENT_SOURCE)
        output = _run(ws, "--name", "timbal", "--url", "https://api.timbal.ai/mcp")
        assert "tools=[timbal]" in output
        ns = _exec_agent(output)
        server = ns["agent"].tools[0]
        assert server.transport == "http"
        assert server.url == "https://api.timbal.ai/mcp"

    def test_header_with_embedded_placeholder_becomes_fstring(self, workspace, monkeypatch):
        ws = workspace(AGENT_SOURCE)
        output = _run(
            ws,
            "--name", "timbal",
            "--url", "https://api.timbal.ai/mcp",
            "--headers", '{"Authorization": "Bearer $TIMBAL_API_KEY"}',
        )
        # The secret must never be hardcoded — only the env lookup.
        assert "os.environ['TIMBAL_API_KEY']" in output
        assert 'f"Bearer {' in output

        monkeypatch.setenv("TIMBAL_API_KEY", "sk-123")
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].headers == {"Authorization": "Bearer sk-123"}

    def test_braced_placeholder(self, workspace, monkeypatch):
        ws = workspace(AGENT_SOURCE)
        output = _run(
            ws,
            "--name", "timbal",
            "--url", "https://api.timbal.ai/mcp",
            "--headers", '{"x-key": "${API_KEY}"}',
        )
        assert 'os.environ["API_KEY"]' in output
        monkeypatch.setenv("API_KEY", "k1")
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].headers == {"x-key": "k1"}


class TestFromJson:
    def test_adds_all_servers(self, workspace, tmp_path, monkeypatch):
        ws = workspace(AGENT_SOURCE)
        config = {
            "mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
                "timbal": {"url": "https://api.timbal.ai/mcp", "headers": {"Authorization": "Bearer $TIMBAL_API_KEY"}},
            }
        }
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps(config))

        output = _run(ws, "--from-json", f"@{config_path}")
        assert "tools=[fs, timbal]" in output

        monkeypatch.setenv("TIMBAL_API_KEY", "sk-123")
        ns = _exec_agent(output)
        servers = {s.name: s for s in ns["agent"].tools}
        assert servers["fs"].transport == "stdio"
        assert servers["timbal"].transport == "http"
        assert servers["timbal"].headers == {"Authorization": "Bearer sk-123"}

    def test_type_alias_streamable_http(self, workspace):
        ws = workspace(AGENT_SOURCE)
        config = json.dumps({"mcpServers": {"remote": {"type": "streamable-http", "url": "https://x.dev/mcp"}}})
        output = _run(ws, "--from-json", config)
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].transport == "http"

    def test_mutually_exclusive_with_flags(self, workspace):
        ws = workspace(AGENT_SOURCE)
        config = json.dumps({"mcpServers": {"fs": {"command": "npx"}}})
        stderr = _run(ws, "--from-json", config, "--name", "fs", expect_error=True)
        assert "mutually exclusive" in stderr


class TestIdempotency:
    def test_rerun_does_not_duplicate(self, workspace):
        ws = workspace(AGENT_SOURCE)
        _run(ws, "--name", "fs", "--command", "npx", dry_run=False)
        output = _run(ws, "--name", "fs", "--command", "npx")
        assert output.count("MCPServer(") == 1
        assert output.count("tools=[fs]") == 1

    def test_rerun_replaces_spec(self, workspace):
        ws = workspace(AGENT_SOURCE)
        _run(ws, "--name", "timbal", "--url", "https://api.dev.timbal.ai/mcp", dry_run=False)
        output = _run(ws, "--name", "timbal", "--url", "https://api.timbal.ai/mcp")
        assert output.count("MCPServer(") == 1
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].url == "https://api.timbal.ai/mcp"


class TestRemoval:
    def test_remove_tool_cleans_up_server(self, workspace):
        ws = workspace(AGENT_SOURCE)
        _run(ws, "--name", "fs", "--command", "npx", dry_run=False)

        result = subprocess.run(
            codegen_cmd("--path", str(ws), "--dry-run", "remove-tool", "--name", "fs"),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        output = result.stdout
        assert "MCPServer" not in output  # assignment and import both cleaned up
        assert "tools=[]" in output


class TestNameCollisions:
    """A same-named non-MCP runnable must never be replaced or rewired by add-mcp."""

    def test_tool_assignment_with_same_name_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool

        def list_files(path: str) -> str:
            \"\"\"List files.\"\"\"
            return path

        fs = Tool(name="fs", handler=list_files)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[fs])
        """)
        stderr = _run(ws, "--name", "fs", "--command", "npx", expect_error=True)
        assert "already used by a non-MCP tool" in stderr
        # Source untouched.
        assert 'Tool(name="fs"' in (ws / "agent.py").read_text()

    def test_inline_tool_with_same_name_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool

        def list_files(path: str) -> str:
            \"\"\"List files.\"\"\"
            return path

        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            tools=[Tool(name="fs", handler=list_files)],
        )
        """)
        stderr = _run(ws, "--name", "fs", "--command", "npx", expect_error=True)
        assert "already used by a non-MCP tool" in stderr

    def test_bare_function_with_same_name_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        def fs(path: str) -> str:
            \"\"\"List files.\"\"\"
            return path

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[fs])
        """)
        stderr = _run(ws, "--name", "fs", "--command", "npx", expect_error=True)
        assert "already exists" in stderr

    def test_unrelated_variable_shadowing_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool

        def list_files(path: str) -> str:
            \"\"\"List files.\"\"\"
            return path

        fs = Tool(name="lister", handler=list_files)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[fs])
        """)
        stderr = _run(ws, "--name", "fs", "--command", "npx", expect_error=True)
        assert "already exists and is not MCP server" in stderr

    def test_existing_mcp_server_still_updates(self, workspace):
        """The legit idempotent path must keep working after the collision guard."""
        ws = workspace("""\
        from timbal.core import Agent, MCPServer

        fs = MCPServer(name="fs", transport="stdio", command="old-command")

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[fs])
        """)
        output = _run(ws, "--name", "fs", "--command", "npx")
        assert output.count("MCPServer(") == 1
        ns = _exec_agent(output)
        assert ns["agent"].tools[0].command == "npx"


class TestErrors:
    def test_name_required(self, workspace):
        ws = workspace(AGENT_SOURCE)
        stderr = _run(ws, "--command", "npx", expect_error=True)
        assert "--name is required" in stderr

    def test_no_transport_inferable(self, workspace):
        ws = workspace(AGENT_SOURCE)
        stderr = _run(ws, "--name", "x", expect_error=True)
        assert "Cannot infer transport" in stderr

    def test_command_and_url_ambiguous(self, workspace):
        ws = workspace(AGENT_SOURCE)
        stderr = _run(ws, "--name", "x", "--command", "npx", "--url", "https://x.dev/mcp", expect_error=True)
        assert "disambiguate" in stderr

    def test_invalid_config_rejected(self, workspace):
        ws = workspace(AGENT_SOURCE)
        # transport=stdio without command is invalid at the MCPServer model level.
        stderr = _run(ws, "--name", "x", "--transport", "stdio", "--url", "https://x.dev/mcp", expect_error=True)
        assert "Invalid MCP server config" in stderr

    def test_workflow_entry_point_rejected(self, tmp_path):
        (tmp_path / "agent.py").write_text(
            textwrap.dedent("""\
            from timbal.core import Workflow

            def a() -> str:
                return "a"

            workflow = Workflow(name="wf").step(a)
            """)
        )
        (tmp_path / "timbal.yaml").write_text('fqn: "agent.py::workflow"\n')
        stderr = _run(tmp_path, "--name", "x", "--command", "npx", expect_error=True)
        assert "requires an Agent entry point" in stderr
