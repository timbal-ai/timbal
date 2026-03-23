import textwrap
from pathlib import Path

import pytest
from timbal.codegen.flow import get_flow

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'


@pytest.fixture
def workspace(tmp_path):
    """Write a source file + timbal.yaml and return the workspace directory."""

    def _write(source: str, *, filename: str = "agent.py", fqn: str | None = None) -> Path:
        (tmp_path / filename).write_text(textwrap.dedent(source))
        yaml = fqn if fqn else f'fqn: "{filename}::agent"\n'
        (tmp_path / "timbal.yaml").write_text(yaml)
        return tmp_path

    return _write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flow(workspace_path: Path) -> dict:
    return get_flow(workspace_path)


def _single_node(flow: dict) -> dict:
    assert len(flow["nodes"]) == 1
    return flow["nodes"][0]


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------


class TestTopLevel:
    def test_has_version_nodes_edges(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        flow = _flow(ws)
        assert "_version" in flow
        assert isinstance(flow["_version"], str)
        assert "nodes" in flow
        assert "edges" in flow

    def test_agent_entry_point_single_node(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        flow = _flow(ws)
        assert len(flow["nodes"]) == 1
        assert flow["edges"] == []


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


class TestAgentNode:
    def test_node_type_and_id(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="my_agent", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        node = _single_node(_flow(ws))
        assert node["type"] == "agent"
        assert node["id"] == "my_agent"

    def test_config_has_name_and_description(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(
            name="my_agent",
            description="A test agent",
            model="openai/gpt-4o-mini",
            max_tokens=128,
        )
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["name"]["value"] == "my_agent"
        assert config["name"]["type"] == "string"
        assert config["description"]["value"] == "A test agent"

    def test_config_has_agent_fields(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(
            name="a",
            model="anthropic/claude-haiku-4-5",
            max_tokens=512,
            system_prompt="You are helpful.",
            max_iter=5,
        )
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["model"]["value"] == "anthropic/claude-haiku-4-5"
        assert config["system_prompt"]["value"] == "You are helpful."
        assert config["max_iter"]["value"] == 5
        assert config["max_iter"]["type"] == "integer"
        assert config["max_tokens"]["value"] == 512

    def test_system_prompt_callable(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        def get_prompt():
            return "dynamic prompt"

        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            system_prompt=get_prompt,
        )
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["system_prompt"]["value"] == "<get_prompt>"
        # Schema should include callable as a possible type
        any_of_types = [v.get("type") for v in config["system_prompt"]["anyOf"]]
        assert "callable" in any_of_types
        assert "string" in any_of_types

    def test_system_prompt_none(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["system_prompt"]["value"] is None

    def test_has_params_and_return(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        data = _single_node(_flow(ws))["data"]
        assert "params" in data
        assert "return" in data
        assert data["params"]["type"] == "object"

    def test_has_metadata(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        data = _single_node(_flow(ws))["data"]
        assert data["metadata"]["type"] == "Agent"


# ---------------------------------------------------------------------------
# Tools inside agent
# ---------------------------------------------------------------------------


class TestAgentTools:
    def test_tools_in_config(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()
        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[web_search],
        )
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert "tools" in config
        assert len(config["tools"]) == 1

    def test_tool_node_structure(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()
        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[web_search],
        )
        """)
        tool = _single_node(_flow(ws))["data"]["config"]["tools"][0]
        assert tool["type"] == "tool"
        assert tool["id"] == "a.web_search"
        assert tool["data"]["config"]["name"]["value"] == "web_search"

    def test_web_search_config(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch(allowed_domains=["example.com", "test.com"])
        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[web_search],
        )
        """)
        tool_config = _single_node(_flow(ws))["data"]["config"]["tools"][0]["data"]["config"]
        assert tool_config["allowed_domains"]["value"] == ["example.com", "test.com"]
        assert tool_config["blocked_domains"]["value"] is None

    def test_custom_tool(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool

        def greet(name: str) -> str:
            return f"Hello {name}"

        greet_tool = Tool(handler=greet)
        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[greet_tool],
        )
        """)
        tool = _single_node(_flow(ws))["data"]["config"]["tools"][0]
        assert tool["type"] == "tool"
        assert tool["data"]["config"]["name"]["value"] == "greet"
        # Custom tool should have params from handler signature
        assert "name" in tool["data"]["params"]["properties"]

    def test_multiple_tools(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool
        from timbal.tools import WebSearch

        def my_func(query: str) -> str:
            return query

        my_func_tool = Tool(handler=my_func)
        web_search = WebSearch()
        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[my_func_tool, web_search],
        )
        """)
        tools = _single_node(_flow(ws))["data"]["config"]["tools"]
        assert len(tools) == 2
        names = [t["data"]["config"]["name"]["value"] for t in tools]
        assert "my_func" in names
        assert "web_search" in names

    def test_agent_as_tool_no_recursion(self, workspace):
        """An agent used as a tool of another agent should appear as a node
        but its own tools should NOT be included (no recursion)."""
        ws = workspace("""\
        from timbal.core import Agent

        inner = Agent(
            name="inner",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[],
        )
        agent = Agent(
            name="outer",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[inner],
        )
        """)
        tools = _single_node(_flow(ws))["data"]["config"]["tools"]
        assert len(tools) == 1
        inner_tool = tools[0]
        assert inner_tool["type"] == "agent"
        assert inner_tool["id"] == "outer.inner"
        # No recursion — inner agent's tools should NOT be present
        assert "tools" not in inner_tool["data"]["config"]

    def test_no_tools(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        tools = _single_node(_flow(ws))["data"]["config"]["tools"]
        assert tools == []


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class TestWorkflow:
    def test_workflow_steps_as_nodes(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def step_a() -> str:
            return "a"

        def step_b() -> str:
            return "b"

        agent = Workflow(name="wf")
        agent.step(Tool(handler=step_a))
        agent.step(Tool(handler=step_b), depends_on=["step_a"])
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        assert len(flow["nodes"]) == 2
        node_names = [n["data"]["config"]["name"]["value"] for n in flow["nodes"]]
        assert "step_a" in node_names
        assert "step_b" in node_names

    def test_workflow_edges(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def step_a() -> str:
            return "a"

        def step_b() -> str:
            return "b"

        agent = Workflow(name="wf")
        agent.step(Tool(handler=step_a))
        agent.step(Tool(handler=step_b), depends_on=["step_a"])
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        assert len(flow["edges"]) == 1
        edge = flow["edges"][0]
        assert edge["source"] == "wf.step_a"
        assert edge["target"] == "wf.step_b"

    def test_workflow_agent_step_includes_tools(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow
        from timbal.tools import WebSearch

        web_search = WebSearch()
        inner_agent = Agent(
            name="summarizer",
            model="openai/gpt-4o-mini",
            max_tokens=128,
            tools=[web_search],
        )

        agent = Workflow(name="wf")
        agent.step(inner_agent)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        assert len(flow["nodes"]) == 1
        node = flow["nodes"][0]
        assert node["type"] == "agent"
        tools = node["data"]["config"]["tools"]
        assert len(tools) == 1
        assert tools[0]["data"]["config"]["name"]["value"] == "web_search"

    def test_workflow_no_edges_for_independent_steps(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def step_a() -> str:
            return "a"

        def step_b() -> str:
            return "b"

        agent = Workflow(name="wf")
        agent.step(Tool(handler=step_a))
        agent.step(Tool(handler=step_b))
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        assert len(flow["nodes"]) == 2
        assert flow["edges"] == []


# ---------------------------------------------------------------------------
# Config schema annotations
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_string_field_has_type(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["name"]["type"] == "string"

    def test_nullable_field_has_anyof(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        # description is str | None
        assert "anyOf" in config["description"]
        types = [v.get("type") for v in config["description"]["anyOf"]]
        assert "string" in types
        assert "null" in types

    def test_integer_field(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128, max_iter=7)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["max_iter"]["type"] == "integer"
        assert config["max_iter"]["value"] == 7

    def test_default_is_preserved(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["max_iter"]["default"] == 10

    def test_model_enum(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        # model enum is compacted — full Literal list is stripped, replaced with a ref marker
        assert config["model"]["x-timbal-ref"] == "models"
        assert config["model"]["type"] == "string"
        assert "enum" not in config["model"]
        assert "anyOf" not in config["model"]

    def test_callable_type_in_schema(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        any_of_types = [v.get("type") for v in config["system_prompt"]["anyOf"]]
        assert "callable" in any_of_types


# ---------------------------------------------------------------------------
# Param defaults in schema
# ---------------------------------------------------------------------------


class TestParamDefaults:
    def test_map_param_in_schema(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        # Find agent_b node
        agent_b_node = [n for n in flow["nodes"] if n["id"] == "wf.agent_b"][0]
        prompt_prop = agent_b_node["data"]["params"]["properties"]["prompt"]
        assert prompt_prop["value"] == {"type": "map", "source": "agent_a"}

    def test_map_param_with_bracket_key_in_schema(self, workspace):
        """Backward compat: bracket notation output["cleaned"] should round-trip."""
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output["cleaned"])
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        agent_b_node = [n for n in flow["nodes"] if n["id"] == "wf.agent_b"][0]
        prompt_prop = agent_b_node["data"]["params"]["properties"]["prompt"]
        assert prompt_prop["value"] == {"type": "map", "source": "agent_a", "key": "output.cleaned"}

    def test_map_param_with_dot_key_in_schema(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output.cleaned)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        agent_b_node = [n for n in flow["nodes"] if n["id"] == "wf.agent_b"][0]
        prompt_prop = agent_b_node["data"]["params"]["properties"]["prompt"]
        assert prompt_prop["value"] == {"type": "map", "source": "agent_a", "key": "output.cleaned"}

    def test_map_param_with_nested_key_in_schema(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output[0].items.name)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        agent_b_node = [n for n in flow["nodes"] if n["id"] == "wf.agent_b"][0]
        prompt_prop = agent_b_node["data"]["params"]["properties"]["prompt"]
        assert prompt_prop["value"] == {"type": "map", "source": "agent_a", "key": "output.0.items.name"}

    def test_value_param_in_schema(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a, prompt="Hello world")
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        node = flow["nodes"][0]
        prompt_prop = node["data"]["params"]["properties"]["prompt"]
        assert prompt_prop["value"] == {"type": "value", "value": "Hello world"}

    def test_no_default_param(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        node = flow["nodes"][0]
        prompt_prop = node["data"]["params"]["properties"]["prompt"]
        # No value set — should not have our custom value field
        assert "value" not in prompt_prop


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_timbal_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_flow(tmp_path)

    def test_missing_source_file(self, tmp_path):
        (tmp_path / "timbal.yaml").write_text('fqn: "nonexistent.py::agent"\n')
        with pytest.raises(FileNotFoundError):
            get_flow(tmp_path)
