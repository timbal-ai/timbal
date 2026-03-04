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
        assert "import Agent, Tool" in output or "import Tool" in output
        assert "Tool(handler=my_search)" in output
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

    def test_adds_custom_function_with_explicit_name(self, workspace):
        """Custom tool with --name includes name= kwarg."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_search(query: str) -> str:\n    return query.upper()"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition, "--name", "custom_search")
        assert 'Tool(name="custom_search", handler=my_search)' in output
        ns = _exec_agent(output)
        assert "custom_search" in [t.name for t in ns["agent"].tools]

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


# ---------------------------------------------------------------------------
# Explicit --name flag
# ---------------------------------------------------------------------------


class TestExplicitName:
    def test_framework_tool_with_name(self, workspace):
        """Framework tool with --name includes name= kwarg."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(ws, "--type", "WebSearch", "--name", "my_search")
        assert 'WebSearch(name="my_search")' in output
        ns = _exec_agent(output)
        assert "my_search" in [t.name for t in ns["agent"].tools]

    def test_framework_tool_without_name(self, workspace):
        """Framework tool without --name uses class default."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        output = _run_dry(ws, "--type", "WebSearch")
        assert "WebSearch()" in output
        ns = _exec_agent(output)
        assert "web_search" in [t.name for t in ns["agent"].tools]

    def test_custom_tool_without_name(self, workspace):
        """Custom tool without --name derives name from handler."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[])
        """)
        definition = "def my_func():\n    return 42"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition)
        assert "Tool(handler=my_func)" in output
        ns = _exec_agent(output)
        assert "my_func" in [t.name for t in ns["agent"].tools]

    def test_two_custom_tools_same_handler_different_names(self, workspace):
        """Two custom tools from the same handler get distinct variables when using --name."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.core import Tool

        def hello():
            print("hello")

        hello_v1 = Tool(name="hello_v1", handler=hello)

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[hello_v1])
        """)
        definition = "def hello():\n    print('hello')"
        output = _run_dry(ws, "--type", "Custom", "--definition", definition, "--name", "hello_v2")
        assert 'Tool(name="hello_v2", handler=hello)' in output
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "hello_v1" in tool_names
        assert "hello_v2" in tool_names

    def test_two_web_search_tools_different_names(self, workspace):
        """Two WebSearch tools with different --name coexist."""
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()

        agent = Agent(name="a", model="openai/gpt-4o-mini", tools=[web_search])
        """)
        output = _run_dry(ws, "--type", "WebSearch", "--name", "second_search")
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "web_search" in tool_names
        assert "second_search" in tool_names


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


def _run_dry_wf(workspace_path: Path, *cli_args: str) -> str:
    """Run codegen add-tool with --dry-run on a workflow workspace."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), "--dry-run", "add-tool", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result.stdout


class TestStepFlag:
    def test_adds_tool_to_workflow_step(self, wf_workspace):
        """Add a framework tool to a specific Agent step in a Workflow."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry_wf(ws, "--type", "WebSearch", "--step", "agent_a")
        assert "web_search = WebSearch()" in output
        assert "tools=[web_search]" in output
        # The tool should be on agent_a, not on the workflow
        assert "agent_a = Agent(" in output

    def test_adds_tool_to_step_no_existing_tools(self, wf_workspace):
        """Add a tool to a step that has no tools kwarg yet."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry_wf(ws, "--type", "WebSearch", "--step", "agent_a")
        assert "tools=[web_search]" in output

    def test_adds_tool_to_step_with_existing_tools(self, wf_workspace):
        """Add a tool to a step that already has tools."""
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.tools import Edit

        edit = Edit()
        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", tools=[edit])

        workflow = Workflow(name="workflow")
        workflow.step(agent_a)
        """)
        output = _run_dry_wf(ws, "--type", "WebSearch", "--step", "agent_a")
        assert "web_search" in output
        assert "edit" in output

    def test_rejects_step_on_agent_entry_point(self, workspace):
        """--step should fail when the entry point is an Agent."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        result = subprocess.run(
            [
                "python", "-m", "timbal.codegen", "--path", str(ws), "--dry-run",
                "add-tool", "--type", "WebSearch", "--step", "agent_a",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--step requires a Workflow" in result.stderr
