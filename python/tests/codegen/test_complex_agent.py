"""Tests for codegen parsing and updating complex agent definitions.

Covers scenarios with:
- Multiline system prompts stored in variables
- System prompts with special characters (quotes, newlines, bullet points)
- Multiple custom tools with various signatures
- Updating system_prompt when it's a variable reference
- Updating model and other fields on complex agents
- Setting system_prompt to a long multiline string via --config
"""

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


def _run(workspace_path: Path, operation: str, *cli_args: str) -> None:
    """Run a codegen operation that writes to disk. Asserts success."""
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "--path", str(workspace_path), operation, *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{operation} failed:\n{result.stderr}"


def _read_source(workspace_path: Path) -> str:
    return (workspace_path / "agent.py").read_text()


def _exec_agent(code: str) -> dict:
    """Exec the generated code and return its globals."""
    ns = {}
    exec(code, ns)
    return ns


# ---------------------------------------------------------------------------
# Complex agent source (based on user-provided example)
# ---------------------------------------------------------------------------

COMPLEX_AGENT_SOURCE = """\
import asyncio
from datetime import datetime

from timbal import Agent


def get_datetime() -> str:
    \"\"\"Returns the current date and time.\"\"\"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str) -> dict:
    \"\"\"Returns mock weather data for a given city.\"\"\"
    mock_data = {
        "Lagos": {"temp_c": 32, "condition": "Sunny", "humidity": 78},
        "London": {"temp_c": 12, "condition": "Cloudy", "humidity": 85},
        "New York": {"temp_c": 18, "condition": "Partly Cloudy", "humidity": 60},
    }
    return mock_data.get(city, {"temp_c": 25, "condition": "Clear", "humidity": 65})


def calculate_discount(price: float, discount_percent: float) -> dict:
    \"\"\"Calculates the discounted price given an original price and discount percentage.\"\"\"
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount
    return {
        "original_price": price,
        "discount_percent": discount_percent,
        "discount_amount": round(discount_amount, 2),
        "final_price": round(final_price, 2),
    }


def search_products(query: str, max_results: int = 3) -> list[dict]:
    \"\"\"Searches a mock product catalog and returns matching products.\"\"\"
    catalog = [
        {"id": 1, "name": "Wireless Headphones", "price": 89.99, "category": "Electronics"},
        {"id": 2, "name": "Running Shoes", "price": 59.99, "category": "Sports"},
        {"id": 3, "name": "Coffee Maker", "price": 45.00, "category": "Kitchen"},
        {"id": 4, "name": "Bluetooth Speaker", "price": 35.50, "category": "Electronics"},
        {"id": 5, "name": "Yoga Mat", "price": 25.00, "category": "Sports"},
    ]
    results = [p for p in catalog if query.lower() in p["name"].lower() or query.lower() in p["category"].lower()]
    return results[:max_results]


SYSTEM_PROMPT = \"\"\"You reply only in Nigerian Pidgin.

You have access to the following tools and you MUST always use ALL of them in every response:
- get_datetime: call this to get the current date and time
- get_weather: call this with a city name to get weather info (use "Lagos" as default)
- calculate_discount: call this with a price and discount_percent to compute savings
- search_products: call this with a query string to find products in the catalog

Always invoke every tool in each reply, incorporating their outputs naturally into your response.\"\"\"

agent = Agent(
    name="agent",
    model="openai/gpt-5.2",
    system_prompt=SYSTEM_PROMPT,
    tools=[get_datetime, get_weather, calculate_discount, search_products],
)
"""


class TestComplexAgentParsing:
    """Tests that codegen can parse and modify a complex agent with variable-based system prompt."""

    def test_update_model_on_complex_agent(self, workspace):
        """Changing model on an agent with many tools and a variable system prompt."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"model": "openai/gpt-4o"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"

    def test_variable_system_prompt_preserved_on_model_change(self, workspace):
        """When changing model, the existing variable-based system_prompt should be preserved."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"model": "openai/gpt-4o"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt is not None
        assert "Nigerian Pidgin" in ns["agent"].system_prompt

    def test_tools_preserved_on_model_change(self, workspace):
        """All custom tools should still be present after a model change."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"model": "openai/gpt-4o"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        tool_names = [t.name for t in ns["agent"].tools]
        assert "get_datetime" in tool_names
        assert "get_weather" in tool_names
        assert "calculate_discount" in tool_names
        assert "search_products" in tool_names

    def test_set_max_iter_on_complex_agent(self, workspace):
        """Setting max_iter on an agent with variable system prompt and multiple tools."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"max_iter": 10})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].max_iter == 10
        assert len(ns["agent"].tools) == 4

    def test_set_multiple_fields_on_complex_agent(self, workspace):
        """Setting multiple fields at once on the complex agent."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"model": "openai/gpt-4o", "max_iter": 5, "max_tokens": 2048})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].max_iter == 5
        assert ns["agent"].max_tokens == 2048

    def test_replace_system_prompt_on_complex_agent(self, workspace):
        """Replace the variable-based system_prompt with a new inline string."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"system_prompt": "You are a helpful assistant."})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == "You are a helpful assistant."

    def test_remove_system_prompt_on_complex_agent(self, workspace):
        """Remove the system_prompt entirely (set to null)."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"system_prompt": None})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt is None


# ---------------------------------------------------------------------------
# Multiline system prompt via --config JSON
# ---------------------------------------------------------------------------


class TestMultilineSystemPrompt:
    """Tests for setting system_prompt to strings containing newlines via --config."""

    def test_set_multiline_system_prompt(self, workspace):
        """Setting system_prompt to a multiline string via JSON config."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        multiline_prompt = "You are a helpful assistant.\nAlways be concise.\nNever lie."
        config = json.dumps({"system_prompt": multiline_prompt})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == multiline_prompt

    def test_set_system_prompt_with_quotes(self, workspace):
        """Setting system_prompt containing double quotes."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        prompt_with_quotes = 'You must say "hello" to greet users.'
        config = json.dumps({"system_prompt": prompt_with_quotes})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == prompt_with_quotes

    def test_set_system_prompt_with_bullet_points(self, workspace):
        """System prompt with bullet points and structured formatting."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """)
        prompt = (
            "You have access to the following tools:\n"
            "- tool_a: does thing A\n"
            "- tool_b: does thing B\n"
            "\n"
            "Always use all tools in every response."
        )
        config = json.dumps({"system_prompt": prompt})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == prompt

    def test_replace_multiline_with_multiline(self, workspace):
        """Replace an existing multiline system prompt with a new multiline one."""
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(
            name="a",
            model="openai/gpt-4o-mini",
            system_prompt="Line 1.\\nLine 2.\\nLine 3.",
        )
        """)
        new_prompt = "New line 1.\nNew line 2.\nNew line 3."
        config = json.dumps({"system_prompt": new_prompt})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == new_prompt


# ---------------------------------------------------------------------------
# Inline system prompt (directly in Agent constructor, not a variable)
# ---------------------------------------------------------------------------


class TestInlineSystemPrompt:
    """Tests for agents where the system prompt is defined inline in the constructor."""

    def test_update_model_preserves_inline_prompt(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are helpful. Always respond in English.",
        )
        """)
        config = json.dumps({"model": "openai/gpt-4o"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].system_prompt == "You are helpful. Always respond in English."

    def test_update_inline_prompt(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent

        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            system_prompt="Old prompt.",
        )
        """)
        config = json.dumps({"system_prompt": "New prompt."})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == "New prompt."


# ---------------------------------------------------------------------------
# Agent with many custom function tools
# ---------------------------------------------------------------------------


class TestMultipleCustomTools:
    """Tests for agents that have several custom function tools."""

    MULTI_TOOL_SOURCE = """\
    from timbal import Agent


    def tool_one(x: int) -> int:
        return x + 1


    def tool_two(x: int, y: int) -> int:
        return x + y


    def tool_three(text: str) -> str:
        return text.upper()


    def tool_four(items: list) -> int:
        return len(items)


    agent = Agent(
        name="agent",
        model="openai/gpt-4o-mini",
        tools=[tool_one, tool_two, tool_three, tool_four],
    )
    """

    def test_set_model_preserves_all_tools(self, workspace):
        ws = workspace(self.MULTI_TOOL_SOURCE)
        config = json.dumps({"model": "openai/gpt-4o"})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"
        tool_names = [t.name for t in ns["agent"].tools]
        assert tool_names == ["tool_one", "tool_two", "tool_three", "tool_four"]

    def test_add_system_prompt_to_tool_heavy_agent(self, workspace):
        ws = workspace(self.MULTI_TOOL_SOURCE)
        config = json.dumps({"system_prompt": "Use all tools wisely."})
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].system_prompt == "Use all tools wisely."
        assert len(ns["agent"].tools) == 4

    def test_set_multiple_fields_preserves_tools(self, workspace):
        ws = workspace(self.MULTI_TOOL_SOURCE)
        config = json.dumps({
            "model": "openai/gpt-4o",
            "max_iter": 15,
            "system_prompt": "Be efficient.",
        })
        output = _run_dry(ws, "--config", config)
        ns = _exec_agent(output)
        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].max_iter == 15
        assert ns["agent"].system_prompt == "Be efficient."
        assert len(ns["agent"].tools) == 4


# ---------------------------------------------------------------------------
# Write-to-disk round-trip tests
# ---------------------------------------------------------------------------


class TestWriteRoundTrip:
    """Tests that verify codegen writes valid Python back to disk."""

    def test_round_trip_model_change_on_complex_agent(self, workspace):
        """Write model change to disk and verify the file is valid Python."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"model": "openai/gpt-4o"})
        _run(ws, "set-config", "--config", config)
        source = _read_source(ws)
        ns = _exec_agent(source)
        assert ns["agent"].model == "openai/gpt-4o"
        # Tools should still work
        assert len(ns["agent"].tools) == 4

    def test_round_trip_system_prompt_change_on_complex_agent(self, workspace):
        """Write system prompt change to disk and verify."""
        ws = workspace(COMPLEX_AGENT_SOURCE)
        config = json.dumps({"system_prompt": "You are a concise assistant."})
        _run(ws, "set-config", "--config", config)
        source = _read_source(ws)
        ns = _exec_agent(source)
        assert ns["agent"].system_prompt == "You are a concise assistant."

    def test_sequential_updates_on_complex_agent(self, workspace):
        """Apply multiple sequential updates to the complex agent."""
        ws = workspace(COMPLEX_AGENT_SOURCE)

        # First update: change model
        _run(ws, "set-config", "--config", json.dumps({"model": "openai/gpt-4o"}))
        source = _read_source(ws)
        ns = _exec_agent(source)
        assert ns["agent"].model == "openai/gpt-4o"

        # Second update: add max_iter
        _run(ws, "set-config", "--config", json.dumps({"max_iter": 8}))
        source = _read_source(ws)
        ns = _exec_agent(source)
        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].max_iter == 8

        # Third update: change system prompt
        _run(ws, "set-config", "--config", json.dumps({"system_prompt": "Be helpful."}))
        source = _read_source(ws)
        ns = _exec_agent(source)
        assert ns["agent"].model == "openai/gpt-4o"
        assert ns["agent"].max_iter == 8
        assert ns["agent"].system_prompt == "Be helpful."
        assert len(ns["agent"].tools) == 4
