"""Org tool library ops: extract-tool and add-library-tool."""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from timbal.codegen.library import extract_tool

from .conftest import codegen_cmd

TIMBAL_YAML = 'fqn: "agent.py::agent"\n'

PRODUCER_SOURCE = """\
import os

import httpx
from timbal.core import Agent, Tool

API_BASE = "https://api.example.com"
TIMEOUT = 10


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {os.getenv('EXAMPLE_API_KEY')}"}


def search_products(query: str, limit: int = 5) -> str:
    \"\"\"Search the product catalog.

    Longer docs here.
    \"\"\"
    resp = httpx.get(
        f"{API_BASE}/search",
        params={"q": query, "limit": limit},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    return resp.text


def get_datetime() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


search_tool = Tool(handler=search_products, description="Product catalog search.")

agent = Agent(
    name="agent",
    model="openai/gpt-5.2",
    tools=[search_tool, get_datetime],
)
"""


@pytest.fixture
def workspace(tmp_path):
    """Write a source file + timbal.yaml (+ optional pyproject) and return the dir."""

    def _write(source: str, pyproject: str | None = None, subdir: str | None = None) -> Path:
        ws = tmp_path / subdir if subdir else tmp_path
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "agent.py").write_text(textwrap.dedent(source))
        (ws / "timbal.yaml").write_text(TIMBAL_YAML)
        if pyproject is not None:
            (ws / "pyproject.toml").write_text(textwrap.dedent(pyproject))
        return ws

    return _write


def _run_cli(*cli_args: str, expect_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(codegen_cmd(*cli_args), capture_output=True, text=True)
    if expect_failure:
        assert result.returncode != 0, f"expected failure, got:\n{result.stdout}"
    else:
        assert result.returncode == 0, f"codegen failed:\n{result.stderr}"
    return result


class TestExtractClosure:
    def test_wrapper_tool_closure(self, workspace):
        """The closure has the handler, helpers, constants, and imports — nothing else."""
        ws = workspace(
            PRODUCER_SOURCE,
            pyproject="""\
            [project]
            name = "demo"
            dependencies = ["timbal[all]", "httpx>=0.27"]
            """,
        )
        manifest = extract_tool(ws, "search_products")

        assert manifest["name"] == "search_products"
        assert manifest["binding"] == "search_tool"
        source = manifest["source"]
        assert "def search_products" in source
        assert "def _auth_headers" in source
        assert "API_BASE" in source
        assert "TIMEOUT" in source
        assert "import httpx" in source
        # Unrelated definitions and the agent must not leak in.
        assert "get_datetime" not in source
        assert "Agent(" not in source

        # The module must be importable and expose the binding.
        ns: dict = {}
        exec(compile(source, "<tool>", "exec"), ns)
        assert ns["search_tool"].name == "search_products"

    def test_manifest_metadata(self, workspace):
        ws = workspace(
            PRODUCER_SOURCE,
            pyproject="""\
            [project]
            name = "demo"
            dependencies = ["timbal[all]", "httpx>=0.27"]
            """,
        )
        manifest = extract_tool(ws, "search_products")

        assert manifest["description"] == "Product catalog search."
        assert manifest["requirements"] == ["httpx>=0.27"]
        assert manifest["env_vars"] == ["EXAMPLE_API_KEY"]
        assert [p["name"] for p in manifest["params"]] == ["query", "limit"]
        assert manifest["params"][0]["annotation"] == "str"
        assert manifest["params"][1]["default"] == "5"

    def test_requirements_fall_back_to_import_name(self, workspace):
        """Imports missing from pyproject still show up, dist-mapped."""
        ws = workspace("""\
        import yaml
        from timbal.core import Agent, Tool


        def load(doc: str) -> str:
            return str(yaml.safe_load(doc))


        load_tool = Tool(handler=load)
        agent = Agent(name="a", model="openai/gpt-5.2", tools=[load_tool])
        """)
        manifest = extract_tool(ws, "load")
        assert manifest["requirements"] == ["pyyaml"]

    def test_bare_function_tool(self, workspace):
        ws = workspace(PRODUCER_SOURCE)
        manifest = extract_tool(ws, "get_datetime")
        assert manifest["binding"] == "get_datetime"
        assert "def get_datetime" in manifest["source"]
        assert "search_products" not in manifest["source"]
        # Docstring absent → description falls back to None.
        assert manifest["description"] is None

    def test_inline_tool_call_synthesizes_binding(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool


        def shout(text: str) -> str:
            return text.upper()


        agent = Agent(name="a", model="openai/gpt-5.2", tools=[Tool(handler=shout)])
        """)
        manifest = extract_tool(ws, "shout")
        assert manifest["binding"] == "shout_tool"
        assert "shout_tool = Tool(handler=shout)" in manifest["source"]
        ns: dict = {}
        exec(compile(manifest["source"], "<tool>", "exec"), ns)
        assert ns["shout_tool"].name == "shout"

    def test_integration_annotation_detected(self, workspace):
        ws = workspace("""\
        from typing import Annotated

        from timbal.core import Agent, Tool
        from timbal.platform.integrations import Integration


        def send(to: str, integration: Annotated[str, Integration("gmail")] | None = None) -> str:
            return to


        send_tool = Tool(handler=send)
        agent = Agent(name="a", model="openai/gpt-5.2", tools=[send_tool])
        """)
        manifest = extract_tool(ws, "send")
        assert manifest["integrations"] == ["gmail"]


class TestExtractRejections:
    def test_framework_tool_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent
        from timbal.tools import WebSearch

        web_search = WebSearch()
        agent = Agent(name="a", model="openai/gpt-5.2", tools=[web_search])
        """)
        with pytest.raises(ValueError, match="Only custom tools"):
            extract_tool(ws, "web_search")

    def test_entry_point_reference_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool


        def ask_self(q: str) -> str:
            return str(agent)


        ask_tool = Tool(handler=ask_self)
        agent = Agent(name="a", model="openai/gpt-5.2", tools=[ask_tool])
        """)
        with pytest.raises(ValueError, match="entry point"):
            extract_tool(ws, "ask_self")

    def test_local_module_import_rejected(self, workspace):
        ws = workspace("""\
        import helpers
        from timbal.core import Agent, Tool


        def run(q: str) -> str:
            return helpers.go(q)


        run_tool = Tool(handler=run)
        agent = Agent(name="a", model="openai/gpt-5.2", tools=[run_tool])
        """)
        (ws / "helpers.py").write_text("def go(q):\n    return q\n")
        with pytest.raises(ValueError, match="local module 'helpers'"):
            extract_tool(ws, "run")

    def test_unknown_tool_lists_available(self, workspace):
        ws = workspace(PRODUCER_SOURCE)
        with pytest.raises(ValueError, match="Available tools"):
            extract_tool(ws, "nope")

    def test_loop_bound_dependency_rejected(self, workspace):
        ws = workspace("""\
        from timbal.core import Agent, Tool

        for _prefix in ["a", "b"]:
            PREFIX = _prefix


        def tag(text: str) -> str:
            return PREFIX + text


        tag_tool = Tool(handler=tag)
        agent = Agent(name="a", model="openai/gpt-5.2", tools=[tag_tool])
        """)
        with pytest.raises(ValueError, match="unsupported statements"):
            extract_tool(ws, "tag")


class TestAddLibraryTool:
    LIB_MODULE = """\
    from timbal.core import Tool


    def shout(text: str) -> str:
        \"\"\"Uppercase the text.\"\"\"
        return text.upper()


    shout_tool = Tool(handler=shout)
    """

    CONSUMER = """\
    from timbal.core import Agent

    agent = Agent(name="agent", model="openai/gpt-5.2", tools=[])
    """

    def _write_lib(self, tmp_path: Path) -> Path:
        lib = tmp_path / "lib_module.py"
        lib.write_text(textwrap.dedent(self.LIB_MODULE))
        return lib

    def test_vendors_wires_and_stamps_provenance(self, workspace, tmp_path):
        ws = workspace(self.CONSUMER, subdir="consumer")
        lib = self._write_lib(tmp_path)
        result = _run_cli(
            "--path", str(ws), "add-library-tool",
            "--tool", "shout", "--source", f"@{lib}", "--provenance", "shout@abc1234",
        )
        info = json.loads(result.stdout)
        assert info["binding"] == "shout_tool"
        assert info["name"] == "shout"

        vendored = (ws / "tools" / "shout.py").read_text()
        assert vendored.startswith("# timbal-tool: shout@abc1234\n")
        agent_src = (ws / "agent.py").read_text()
        assert "from tools.shout import shout_tool" in agent_src
        assert "tools=[shout_tool]" in agent_src

    def test_idempotent_re_add(self, workspace, tmp_path):
        ws = workspace(self.CONSUMER, subdir="consumer")
        lib = self._write_lib(tmp_path)
        for _ in range(2):
            _run_cli(
                "--path", str(ws), "add-library-tool",
                "--tool", "shout", "--source", f"@{lib}", "--provenance", "shout@abc1234",
            )
        agent_src = (ws / "agent.py").read_text()
        assert agent_src.count("shout_tool") == 2  # one import + one tools entry

    def test_dry_run_writes_nothing(self, workspace, tmp_path):
        ws = workspace(self.CONSUMER, subdir="consumer")
        lib = self._write_lib(tmp_path)
        original = (ws / "agent.py").read_text()
        result = _run_cli(
            "--path", str(ws), "--dry-run", "add-library-tool",
            "--tool", "shout", "--source", f"@{lib}",
        )
        assert "shout_tool" in result.stdout
        assert (ws / "agent.py").read_text() == original
        assert not (ws / "tools").exists()

    def test_runtime_name_conflict_rejected(self, workspace, tmp_path):
        ws = workspace("""\
        from timbal.core import Agent


        def shout(text: str) -> str:
            return text


        agent = Agent(name="agent", model="openai/gpt-5.2", tools=[shout])
        """, subdir="consumer")
        lib = self._write_lib(tmp_path)
        result = _run_cli(
            "--path", str(ws), "add-library-tool",
            "--tool", "org_shout", "--source", f"@{lib}",
            expect_failure=True,
        )
        assert "already exists" in result.stderr

    def test_symbol_collision_gets_aliased_import(self, workspace, tmp_path):
        ws = workspace("""\
        from timbal.core import Agent


        def shout_tool(text: str) -> str:
            return text


        agent = Agent(name="agent", model="openai/gpt-5.2", tools=[])
        """, subdir="consumer")
        lib = self._write_lib(tmp_path)
        result = _run_cli(
            "--path", str(ws), "add-library-tool",
            "--tool", "org_shout", "--source", f"@{lib}",
        )
        info = json.loads(result.stdout)
        assert info["local_name"] == "org_shout"
        agent_src = (ws / "agent.py").read_text()
        assert "from tools.org_shout import shout_tool as org_shout" in agent_src
        assert "tools=[org_shout]" in agent_src

    def test_bare_function_module_binding_inferred(self, workspace, tmp_path):
        ws = workspace(self.CONSUMER, subdir="consumer")
        lib = tmp_path / "bare.py"
        lib.write_text("def stamp() -> str:\n    return 'now'\n")
        result = _run_cli(
            "--path", str(ws), "add-library-tool",
            "--tool", "stamp", "--source", f"@{lib}",
        )
        info = json.loads(result.stdout)
        assert info["binding"] == "stamp"
        assert "from tools.stamp import stamp" in (ws / "agent.py").read_text()


class TestRoundTrip:
    def test_extract_then_add_produces_working_agent(self, workspace, tmp_path):
        producer = workspace("""\
        from timbal.core import Agent, Tool

        SUFFIX = "!"


        def _decorate(text: str) -> str:
            return text + SUFFIX


        def shout(text: str) -> str:
            \"\"\"Uppercase and decorate.\"\"\"
            return _decorate(text.upper())


        shout_tool = Tool(handler=shout)
        agent = Agent(name="agent", model="openai/gpt-5.2", tools=[shout_tool])
        """, subdir="producer")
        manifest = extract_tool(producer, "shout")

        lib = tmp_path / "lib.py"
        lib.write_text(manifest["source"])

        consumer = workspace("""\
        from timbal.core import Agent

        agent = Agent(name="agent", model="openai/gpt-5.2", tools=[])
        """, subdir="consumer")
        _run_cli(
            "--path", str(consumer), "add-library-tool",
            "--tool", "shout", "--source", f"@{lib}",
            "--binding", manifest["binding"],
            "--provenance", "shout@1234abc",
        )

        # Load the consumer agent with the member dir on sys.path (mirrors
        # how ImportSpec.load runs it) and check the tool is live.
        import subprocess as sp
        import sys

        code = (
            "import sys; sys.path.insert(0, sys.argv[1])\n"
            "import agent\n"
            "tool = next(t for t in agent.agent.tools if t.name == 'shout')\n"
            "import asyncio\n"
            "print(asyncio.run(tool(text='hey').collect()).output)\n"
        )
        result = sp.run(
            [sys.executable, "-c", code, str(consumer)],
            capture_output=True, text=True, cwd=str(consumer),
        )
        assert result.returncode == 0, result.stderr
        assert "HEY!" in result.stdout
