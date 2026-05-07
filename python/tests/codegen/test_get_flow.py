import textwrap
from pathlib import Path

import pytest
from timbal.codegen.flow import format_compact, get_flow

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

    def test_fallback_model_is_json_serialisable(self, workspace):
        """Regression: FallbackModel as agent.model used to crash json.dumps
        with `TypeError: Object of type FallbackModel is not JSON serializable`
        because get_config wrote the raw object into config['model']['value'].

        Also verifies the new shape:
          - `model` is always a string (the primary model only).
          - `fallbacks` is an array of objects, one per non-primary entry,
            carrying every ModelEntry logic field.
        """
        import json

        ws = workspace("""\
        from timbal.core import Agent, FallbackModel

        agent = Agent(
            name="a",
            model=FallbackModel(
                "openai/gpt-4o-mini",
                "anthropic/claude-haiku-4-5",
                "google/gemini-2.5-flash",
            ),
            max_tokens=128,
        )
        """)
        flow = _flow(ws)
        json.dumps(flow)
        config = _single_node(flow)["data"]["config"]
        assert config["model"]["value"] == "openai/gpt-4o-mini"

        fb = config["fallbacks"]
        assert fb["type"] == "array"
        assert fb["items"]["type"] == "object"
        assert set(fb["items"]["properties"].keys()) == {
            "model",
            "max_retries",
            "retry_delay",
            "api_key",
            "base_url",
        }
        assert fb["items"]["required"] == ["model"]
        assert fb["value"] == [
            {
                "model": "anthropic/claude-haiku-4-5",
                "max_retries": 2,
                "retry_delay": 1.0,
                "api_key": None,
                "base_url": None,
            },
            {
                "model": "google/gemini-2.5-flash",
                "max_retries": 2,
                "retry_delay": 1.0,
                "api_key": None,
                "base_url": None,
            },
        ]

    def test_fallback_model_entry_fields_round_trip(self, workspace):
        """Custom max_retries / retry_delay / base_url / api_key on a
        ModelEntry must be preserved (api_key masked)."""
        import json

        ws = workspace("""\
        from timbal.core import Agent, FallbackModel, ModelEntry

        agent = Agent(
            name="a",
            model=FallbackModel(
                "openai/gpt-4o-mini",
                ModelEntry(
                    model="anthropic/claude-haiku-4-5",
                    max_retries=5,
                    retry_delay=2.5,
                    api_key="hunter2",
                    base_url="https://example.com",
                ),
            ),
            max_tokens=128,
        )
        """)
        flow = _flow(ws)
        json.dumps(flow)
        config = _single_node(flow)["data"]["config"]
        assert config["fallbacks"]["value"] == [
            {
                "model": "anthropic/claude-haiku-4-5",
                "max_retries": 5,
                "retry_delay": 2.5,
                "api_key": "**********",
                "base_url": "https://example.com",
            },
        ]

    def test_no_fallback_model_has_empty_fallbacks(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        config = _single_node(_flow(ws))["data"]["config"]
        assert config["fallbacks"]["value"] == []

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

    def test_helper_lambda_param_is_callable_not_phantom_map(self, workspace):
        """Regression: a runtime lambda whose source doesn't reference
        step_span(...) (helper-fn call) used to crash the compact formatter
        because it produced {'type': 'map', 'source': None}.

        Now it should be recorded as a 'callable' entry and never as a
        map with source=None."""
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow

        def _build_prompt():
            return "hello"

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, depends_on=["agent_a"], prompt=lambda: _build_prompt())
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        agent_b_node = [n for n in flow["nodes"] if n["id"] == "wf.agent_b"][0]
        prompt_prop = agent_b_node["data"]["params"]["properties"]["prompt"]
        val = prompt_prop["value"]
        assert val["type"] == "callable"
        assert "expr" in val and val["expr"]
        # No phantom map edge with null source.
        assert val != {"type": "map", "source": None}

    def test_no_phantom_map_with_null_source_anywhere(self, workspace):
        """Stronger guarantee: no node in the graph has a value of
        {'type': 'map', 'source': None}."""
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow

        def _build_prompt():
            return "x"

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, depends_on=["agent_a"], prompt=lambda: _build_prompt())
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        flow = _flow(ws)
        for node in flow["nodes"]:
            for prop in node["data"]["params"].get("properties", {}).values():
                val = prop.get("value")
                if isinstance(val, dict) and val.get("type") == "map":
                    assert val.get("source"), f"phantom map with null source: {val}"


# ---------------------------------------------------------------------------
# format_compact
# ---------------------------------------------------------------------------


def _compact(workspace_path: Path) -> str:
    return format_compact(get_flow(workspace_path))


class TestFormatCompact:
    def test_empty_flow(self):
        assert format_compact({"nodes": [], "edges": []}) == "(empty flow)"

    def test_standalone_agent_header(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="my_agent", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        out = _compact(ws)
        assert out.startswith("AGENT my_agent")
        assert "openai/gpt-4o-mini" in out

    def test_standalone_agent_max_iter(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=256, max_iter=3)
        """)
        out = _compact(ws)
        assert "max_iter=3" in out
        assert "max_tokens=256" in out

    def test_standalone_agent_system_prompt(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128, system_prompt="Be helpful.")
        """)
        out = _compact(ws)
        assert 'system_prompt: "Be helpful."' in out

    def test_standalone_agent_system_prompt_truncated(self, workspace):
        long_prompt = "x" * 200
        ws = workspace(f"""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128, system_prompt="{long_prompt}")
        """)
        out = _compact(ws)
        assert "…" in out
        # Should not include the full prompt
        assert "x" * 121 not in out

    def test_standalone_agent_no_system_prompt(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        out = _compact(ws)
        assert "system_prompt" not in out

    def test_standalone_agent_tools_listed(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128, tools=[WebSearch()])
        """)
        out = _compact(ws)
        assert "web_search" in out

    def test_standalone_agent_fallbacks_rendered(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, FallbackModel

        agent = Agent(
            name="a",
            model=FallbackModel("openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"),
            max_tokens=128,
        )
        """)
        out = _compact(ws)
        assert "openai/gpt-4o-mini" in out  # primary still in header
        assert "fallbacks: anthropic/claude-haiku-4-5" in out

    def test_standalone_agent_fallbacks_render_non_default_fields(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, FallbackModel, ModelEntry

        agent = Agent(
            name="a",
            model=FallbackModel(
                "openai/gpt-4o-mini",
                ModelEntry(model="anthropic/claude-haiku-4-5", max_retries=5, retry_delay=2.5),
            ),
            max_tokens=128,
        )
        """)
        out = _compact(ws)
        assert "anthropic/claude-haiku-4-5(retries=5, delay=2.5)" in out

    def test_standalone_agent_no_fallbacks_line_when_empty(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        out = _compact(ws)
        assert "fallbacks:" not in out

    def test_standalone_agent_no_tools(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini", max_tokens=128)
        """)
        out = _compact(ws)
        assert "tools: none" in out

    def test_workflow_header(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def step_a() -> str:
            return "a"

        agent = Workflow(name="my_wf")
        agent.step(Tool(handler=step_a))
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert out.startswith("WORKFLOW my_wf")
        assert "STEPS" in out

    def test_workflow_tool_step_signature(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def greet(name: str, loud: bool = False) -> str:
            return f"Hello {name}"

        agent = Workflow(name="wf")
        agent.step(Tool(handler=greet))
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "tool   greet(name:string, loud?:boolean)" in out

    def test_workflow_tool_return_type(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def produce() -> dict:
            return {}

        agent = Workflow(name="wf")
        agent.step(Tool(handler=produce))
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "→ dict" in out

    def test_workflow_agent_step(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow

        inner = Agent(name="summarizer", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(inner)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "agent  summarizer" in out
        assert "openai/gpt-4o-mini" in out

    def test_workflow_map_param_shown(self, workspace):
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
        out = _compact(ws)
        assert "prompt ← agent_a" in out

    def test_workflow_map_param_with_key(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output[0].text)
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "prompt ← agent_a.output.0.text" in out

    def test_workflow_edges_section(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def step_a() -> str:
            return "a"

        def step_b(x: str) -> str:
            return x

        agent = Workflow(name="wf")
        agent.step(Tool(handler=step_a))
        agent.step(Tool(handler=step_b), depends_on=["step_a"])
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "EDGES" in out
        assert "step_a → step_b" in out

    def test_workflow_no_edges_section_when_empty(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def step_a() -> str:
            return "a"

        agent = Workflow(name="wf")
        agent.step(Tool(handler=step_a))
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "EDGES" not in out

    def test_compact_with_helper_lambda_does_not_crash(self, workspace):
        """Regression for AttributeError: 'NoneType' object has no attribute
        'split' in _short(). A lambda that doesn't statically reference
        step_span(...) used to produce source=None and crash the formatter."""
        ws = workspace(
            """\
        from timbal.core import Agent, Workflow

        def _build_prompt():
            return "go"

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini", max_tokens=128)
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini", max_tokens=128)

        agent = Workflow(name="wf")
        agent.step(agent_a)
        agent.step(agent_b, depends_on=["agent_a"], prompt=lambda: _build_prompt())
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        # Should render the callable on the prompt line without crashing.
        assert "agent_b" in out
        assert "prompt ← " in out
        # And must not emit a "?" placeholder for that line (we have a real expr).
        assert "prompt ← ?" not in out

    def test_compact_with_helper_lambda_on_tool_step(self, workspace):
        """Same regression but on a Tool step (different code path in
        _fmt_node_lines)."""
        ws = workspace(
            """\
        from timbal.core import Tool, Workflow

        def _build_x():
            return "y"

        def consumer(x: str) -> str:
            return x

        agent = Workflow(name="wf")
        agent.step(Tool(handler=consumer), x=lambda: _build_x())
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "tool   consumer(" in out
        assert "x:←" in out

    def test_workflow_fan_out_edges(self, workspace):
        ws = workspace(
            """\
        from timbal.core import Workflow, Tool

        def root() -> str:
            return "r"

        def branch_a() -> str:
            return "a"

        def branch_b() -> str:
            return "b"

        agent = Workflow(name="wf")
        agent.step(Tool(handler=root))
        agent.step(Tool(handler=branch_a), depends_on=["root"])
        agent.step(Tool(handler=branch_b), depends_on=["root"])
        """,
            fqn='fqn: "agent.py::agent"\n',
        )
        out = _compact(ws)
        assert "root → branch_a, branch_b" in out


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
