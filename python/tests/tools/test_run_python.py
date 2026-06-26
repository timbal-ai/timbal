"""Tests for RunPython tool.

Cases ported from public competitor test suites:
- pydantic/monty: crates/monty-python/tests/test_basic.py, test_exceptions.py,
  test_external.py, test_print.py, test_limits.py
- pydantic/mcp-run-python: tests/test_sandbox.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import timbal as _timbal_pkg
from timbal import Agent
from timbal.tools.run_python import RunPython, _detect_imports, _resolve_dependencies

EXECUTOR = "venv"


def _repo_root() -> Path:
    """Locate the repo root (dir containing pyproject.toml) from the package."""
    start = Path(_timbal_pkg.__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate timbal repo root")


@pytest.fixture
def run_python() -> RunPython:
    return RunPython(executor=EXECUTOR)


async def collect(tool: RunPython, code: str, **kwargs: Any) -> dict[str, Any]:
    result = await tool(code=code, **kwargs).collect()
    assert result.error is None, result.error
    return result.output


def double(x: int) -> int:
    """Double an integer."""
    return x * 2


def noop() -> str:
    """No-op external function."""
    return "called"


def take_positional(a: int, b: int, c: int) -> str:
    """Accept positional args (Monty test_external_function_positional_args)."""
    assert (a, b, c) == (1, 2, 3)
    return "ok"


def take_kwargs(**kwargs: Any) -> str:
    """Accept keyword args."""
    assert kwargs == {"a": 1, "b": "two"}
    return "ok"


def take_mixed(a: int, b: int, **kwargs: Any) -> str:
    """Accept positional and keyword args."""
    assert (a, b) == (1, 2)
    assert kwargs == {"x": "hello", "y": True}
    return "ok"


def fail_tool() -> None:
    """Raise a ValueError like Monty external-function error tests."""
    raise ValueError("intentional error")


class TestRunPythonInitialization:
    def test_tool_creation(self):
        tool = RunPython()
        assert tool.name == "run_python"
        assert tool.handler is not None
        assert tool.timeout == 60.0

    def test_tool_with_exposed_tools(self):
        tool = RunPython(tools=[double])
        assert "double" in tool.description

    def test_detect_imports(self):
        code = "import json\nfrom pathlib import Path\nimport tabulate"
        imports = _detect_imports(code)
        assert "json" in imports
        assert "pathlib" in imports
        assert "tabulate" in imports

    def test_resolve_dependencies_auto(self):
        code = "import tabulate\nimport yaml"
        deps = _resolve_dependencies(code, explicit=None, default=None)
        assert "tabulate" in deps
        assert "pyyaml" in deps

    def test_resolve_dependencies_explicit(self):
        code = "x = 1"
        deps = _resolve_dependencies(code, explicit=["requests>=2.0"], default=["httpx"])
        assert "requests>=2.0" in deps
        assert "httpx" in deps


class TestRunPythonMontyBasic:
    """Ported from pydantic/monty test_basic.py."""

    @pytest.mark.asyncio
    async def test_simple_expression(self, run_python: RunPython):
        out = await collect(run_python, "1 + 2")
        assert out["return_value"] == 3

    @pytest.mark.asyncio
    async def test_arithmetic(self, run_python: RunPython):
        out = await collect(run_python, "10 * 5 - 3")
        assert out["return_value"] == 47

    @pytest.mark.asyncio
    async def test_string_concatenation(self, run_python: RunPython):
        out = await collect(run_python, '"hello" + " " + "world"')
        assert out["return_value"] == "hello world"

    @pytest.mark.asyncio
    async def test_multiple_runs_same_code(self, run_python: RunPython):
        for x, expected in [(5, 10), (10, 20), (-3, -6)]:
            code = f"x = {x}\nx * 2"
            out = await collect(run_python, code)
            assert out["return_value"] == expected

    @pytest.mark.asyncio
    async def test_multiline_code(self, run_python: RunPython):
        code = "x = 1\ny = 2\nx + y"
        out = await collect(run_python, code)
        assert out["return_value"] == 3

    @pytest.mark.asyncio
    async def test_function_definition_and_call(self, run_python: RunPython):
        code = """
def add(a, b):
    return a + b

add(3, 4)
"""
        out = await collect(run_python, code)
        assert out["return_value"] == 7


class TestRunPythonMontyExceptions:
    """Ported from pydantic/monty test_exceptions.py."""

    @pytest.mark.asyncio
    async def test_zero_division_error(self, run_python: RunPython):
        out = await collect(run_python, "1 / 0")
        assert out["status"] == "error"
        assert out["error"]["type"] == "ZeroDivisionError"

    @pytest.mark.asyncio
    async def test_value_error(self, run_python: RunPython):
        out = await collect(run_python, "raise ValueError('bad value')")
        assert out["status"] == "error"
        assert out["error"]["type"] == "ValueError"
        assert "bad value" in out["error"]["message"]

    @pytest.mark.asyncio
    async def test_type_error(self, run_python: RunPython):
        out = await collect(run_python, "'string' + 1")
        assert out["status"] == "error"
        assert out["error"]["type"] == "TypeError"

    @pytest.mark.asyncio
    async def test_index_error(self, run_python: RunPython):
        out = await collect(run_python, "[1, 2, 3][10]")
        assert out["status"] == "error"
        assert out["error"]["type"] == "IndexError"

    @pytest.mark.asyncio
    async def test_key_error(self, run_python: RunPython):
        out = await collect(run_python, "{'a': 1}['b']")
        assert out["status"] == "error"
        assert out["error"]["type"] == "KeyError"

    @pytest.mark.asyncio
    async def test_name_error(self, run_python: RunPython):
        out = await collect(run_python, "undefined_variable")
        assert out["status"] == "error"
        assert out["error"]["type"] == "NameError"

    @pytest.mark.asyncio
    async def test_assertion_error(self, run_python: RunPython):
        out = await collect(run_python, "assert False")
        assert out["status"] == "error"
        assert out["error"]["type"] == "AssertionError"

    @pytest.mark.asyncio
    async def test_assertion_error_with_message(self, run_python: RunPython):
        out = await collect(run_python, "assert False, 'custom message'")
        assert out["status"] == "error"
        assert out["error"]["type"] == "AssertionError"
        assert "custom message" in out["error"]["message"]

    @pytest.mark.asyncio
    async def test_nested_traceback(self, run_python: RunPython):
        """Ported from mcp-run-python test_sandbox traceback case."""
        code = """
def foo():
    1 / 0

def bar():
    foo()

def baz():
    bar()

baz()
"""
        out = await collect(run_python, code)
        assert out["status"] == "error"
        assert out["error"]["type"] == "ZeroDivisionError"
        assert "traceback" in out["error"]
        assert "foo" in out["error"]["traceback"]


class TestRunPythonMontyPrint:
    """Ported from pydantic/monty test_print.py and mcp-run-python print cases."""

    @pytest.mark.asyncio
    async def test_print_basic(self, run_python: RunPython):
        out = await collect(run_python, 'print("hello")')
        assert "hello" in out["stdout"]
        assert out["return_value"] is None

    @pytest.mark.asyncio
    async def test_print_multiple_lines(self, run_python: RunPython):
        code = 'print("line 1")\nprint("line 2")'
        out = await collect(run_python, code)
        assert "line 1" in out["stdout"]
        assert "line 2" in out["stdout"]

    @pytest.mark.asyncio
    async def test_print_with_values(self, run_python: RunPython):
        out = await collect(run_python, "print(1, 2, 3)")
        assert "1 2 3" in out["stdout"]

    @pytest.mark.asyncio
    async def test_print_with_sep(self, run_python: RunPython):
        out = await collect(run_python, 'print(1, 2, 3, sep="-")')
        assert "1-2-3" in out["stdout"]

    @pytest.mark.asyncio
    async def test_print_with_end(self, run_python: RunPython):
        out = await collect(run_python, 'print("hello", end="!")')
        assert "hello!" in out["stdout"]

    @pytest.mark.asyncio
    async def test_print_then_expression(self, run_python: RunPython):
        """Ported from mcp-run-python: print output + last expression return value."""
        out = await collect(run_python, "print(1)\n1")
        assert "1" in out["stdout"]
        assert out["return_value"] == 1


class TestRunPythonMcpSandbox:
    """Ported from pydantic/mcp-run-python tests/test_sandbox.py."""

    @pytest.mark.asyncio
    async def test_return_value_success(self, run_python: RunPython):
        out = await collect(run_python, "a = 1\na + 1")
        assert out["return_value"] == 2

    @pytest.mark.asyncio
    async def test_return_string(self, run_python: RunPython):
        out = await collect(run_python, '"foobar"')
        assert out["return_value"] == "foobar"

    @pytest.mark.asyncio
    async def test_inline_globals_equivalent(self, run_python: RunPython):
        """mcp-run-python injects globals; we inline equivalent assignments."""
        out = await collect(run_python, "a = [1, 2, 3]\na")
        assert out["return_value"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_multiple_globals(self, run_python: RunPython):
        out = await collect(run_python, "a = 4\nb = 5\na + b")
        assert out["return_value"] == 9

    @pytest.mark.asyncio
    async def test_return_complex_dataclass(self, run_python: RunPython):
        code = """
from dataclasses import dataclass

@dataclass
class Foobar:
    a: int
    b: str
    c: bytes

f = Foobar(1, "2", b"3")
{"a": f.a, "b": f.b, "c": f.c.decode()}
"""
        out = await collect(run_python, code)
        assert out["return_value"] == {"a": 1, "b": "2", "c": "3"}

    @pytest.mark.asyncio
    async def test_print_error_name_error(self, run_python: RunPython):
        out = await collect(run_python, "print(unknown)")
        assert out["status"] == "error"
        assert out["error"]["type"] == "NameError"

    @pytest.mark.asyncio
    async def test_multiple_sequential_runs(self, run_python: RunPython):
        """Ported from mcp-run-python test_multiple_commands."""
        for expected in (1, 2, 3):
            out = await collect(run_python, f"print({expected})\n{expected}")
            assert str(expected) in out["stdout"]
            assert out["return_value"] == expected

    @pytest.mark.asyncio
    async def test_missing_package_without_deps(self, run_python: RunPython):
        """Ported from mcp-run-python ModuleNotFoundError when deps not installed."""
        out = await collect(run_python, "__import__('timbal_nonexistent_module_xyz')")
        assert out["status"] == "error"
        assert out["error"]["type"] in {"ModuleNotFoundError", "ImportError"}

    @pytest.mark.asyncio
    async def test_explicit_deps_install_package(self, run_python: RunPython):
        """Explicit dependencies make third-party imports available (mcp-run-python deps param)."""
        out = await collect(
            run_python,
            "import tabulate\ntabulate.tabulate([[1]], headers=['x'])",
            dependencies=["tabulate"],
        )
        assert out["status"] == "success"

    @pytest.mark.asyncio
    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not available")
    async def test_numpy_with_dependencies(self):
        """Ported from mcp-run-python return-numpy-success."""
        tool = RunPython(executor="uv")
        code = "import numpy\nnumpy.array([1, 2, 3]).tolist()"
        out = await collect(tool, code, dependencies=["numpy"])
        assert out["return_value"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_python_version_available(self, run_python: RunPython):
        """Ported from mcp-run-python python-version case."""
        code = "import sys\nlist(sys.version_info[:3])"
        out = await collect(run_python, code)
        assert isinstance(out["return_value"], list)
        assert len(out["return_value"]) == 3


class TestRunPythonExecution:
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Ported from pydantic/monty test_limits.py ResourceLimits(max_duration_secs)."""
        tool = RunPython(timeout=1.0, executor=EXECUTOR)
        out = await collect(tool, "import time\ntime.sleep(5)")
        assert out["status"] == "error"
        assert out["error"]["type"] == "TimeoutError"

    @pytest.mark.asyncio
    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not available")
    async def test_auto_dependency_install_uv(self):
        """Ported from mcp-run-python auto dependency detection."""
        tool = RunPython(executor="uv")
        code = """
import tabulate
tabulate.tabulate([["a", 1], ["b", 2]], headers=["name", "value"])
"""
        out = await collect(tool, code)
        assert out["status"] == "success"
        assert "a" in out["return_value"]

    @pytest.mark.asyncio
    async def test_explicit_dependencies_venv(self, run_python: RunPython):
        code = """
import tabulate
tabulate.tabulate([["x", 1]], headers=["k", "v"])
"""
        out = await collect(run_python, code, dependencies=["tabulate"])
        assert out["status"] == "success"


class TestRunPythonMontyExternal:
    """Ported from pydantic/monty test_external.py (code mode / external_functions)."""

    @pytest.mark.asyncio
    async def test_external_function_no_args(self):
        tool = RunPython(tools=[noop], executor=EXECUTOR)
        out = await collect(tool, "noop()")
        assert out["return_value"] == "called"

    @pytest.mark.asyncio
    async def test_external_function_positional_args(self):
        tool = RunPython(tools=[take_positional], executor=EXECUTOR)
        out = await collect(tool, "take_positional(1, 2, 3)")
        assert out["return_value"] == "ok"

    @pytest.mark.asyncio
    async def test_external_function_kwargs_only(self):
        tool = RunPython(tools=[take_kwargs], executor=EXECUTOR)
        out = await collect(tool, 'take_kwargs(a=1, b="two")')
        assert out["return_value"] == "ok"

    @pytest.mark.asyncio
    async def test_external_function_mixed_args_kwargs(self):
        tool = RunPython(tools=[take_mixed], executor=EXECUTOR)
        out = await collect(tool, 'take_mixed(1, 2, x="hello", y=True)')
        assert out["return_value"] == "ok"

    @pytest.mark.asyncio
    async def test_external_function_complex_types(self):
        def take_complex(items: list[int], mapping: dict[str, str]) -> str:
            assert items == [1, 2]
            assert mapping == {"key": "value"}
            return "ok"

        tool = RunPython(tools=[take_complex], executor=EXECUTOR)
        out = await collect(tool, 'take_complex([1, 2], {"key": "value"})')
        assert out["return_value"] == "ok"

    @pytest.mark.asyncio
    async def test_external_function_raises_exception(self):
        tool = RunPython(tools=[fail_tool], executor=EXECUTOR)
        out = await collect(tool, "fail_tool()")
        assert out["status"] == "error"
        assert out["error"]["type"] == "ValueError"
        assert "intentional error" in out["error"]["message"]

    @pytest.mark.asyncio
    async def test_external_function_caught_inside_script(self):
        """Ported from pydantic/monty test_external.py except-in-script cases."""
        tool = RunPython(tools=[fail_tool], executor=EXECUTOR)
        code = """
try:
    fail_tool()
    result = 'no error'
except ValueError:
    result = 'caught'
result
"""
        out = await collect(tool, code)
        assert out["return_value"] == "caught"

    @pytest.mark.asyncio
    async def test_undeclared_external_function_name_error(self, run_python: RunPython):
        """Ported from pydantic/monty: unknown external function -> NameError."""
        out = await collect(run_python, "missing_fn()")
        assert out["status"] == "error"
        assert out["error"]["type"] == "NameError"

    @pytest.mark.asyncio
    async def test_double_external_function(self):
        tool = RunPython(tools=[double], executor=EXECUTOR)
        out = await collect(tool, "double(5)")
        assert out["return_value"] == 10

    @pytest.mark.asyncio
    async def test_external_function_in_loop(self):
        tool = RunPython(tools=[double], executor=EXECUTOR)
        code = """
values = [double(1), double(2), double(3)]
sum(values)
"""
        out = await collect(tool, code)
        assert out["return_value"] == 12


class TestRunPythonWithAgent:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_uses_run_python(self):
        agent = Agent(
            name="python_runner",
            model="openai/gpt-4o-mini",
            system_prompt=(
                "You solve math by calling run_python with Python code. "
                "Always use run_python for calculations. Return only the numeric result."
            ),
            tools=[RunPython(executor=EXECUTOR)],
        )
        response = await agent(prompt="What is 17 * 23? Use run_python.").collect()
        assert response.error is None
        assert response.output is not None


class TestRunPythonHardening:
    """Hardening behaviour inspired by E2B / Monty top-performer guidance."""

    @pytest.mark.asyncio
    async def test_secrets_not_leaked_to_child(self, monkeypatch: pytest.MonkeyPatch):
        """Host secrets must never be visible to executed code."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecret")
        monkeypatch.setenv("TIMBAL_API_KEY", "tk-supersecret")
        tool = RunPython(executor=EXECUTOR)
        code = "import os\n[os.environ.get('OPENAI_API_KEY'), os.environ.get('TIMBAL_API_KEY')]"
        out = await collect(tool, code)
        assert out["return_value"] == [None, None]

    @pytest.mark.asyncio
    async def test_env_passthrough_allowlist(self, monkeypatch: pytest.MonkeyPatch):
        """Explicitly allowlisted variables are forwarded to the child."""
        monkeypatch.setenv("MY_PUBLIC_FLAG", "enabled")
        tool = RunPython(executor=EXECUTOR, env_passthrough=["MY_PUBLIC_FLAG"])
        out = await collect(tool, "import os\nos.environ.get('MY_PUBLIC_FLAG')")
        assert out["return_value"] == "enabled"

    @pytest.mark.asyncio
    async def test_stdout_truncation(self):
        tool = RunPython(executor=EXECUTOR, max_output_chars=200)
        out = await collect(tool, "print('A' * 5000)")
        assert len(out["stdout"]) < 500
        assert "characters truncated" in out["stdout"]

    @pytest.mark.asyncio
    async def test_code_size_limit(self, run_python: RunPython):
        oversized = "x = 1\n" + ("# pad\n" * 40000)
        result = await run_python(code=oversized).collect()
        out = result.output
        assert out["status"] == "error"
        assert out["error"]["type"] == "ValueError"
        assert "maximum size" in out["error"]["message"]

    @pytest.mark.asyncio
    async def test_result_size_limit(self, run_python: RunPython):
        result = await run_python(code="'X' * 2_000_000").collect()
        out = result.output
        assert out["status"] == "error"
        assert out["error"]["type"] == "ValueError"
        assert "maximum size" in out["error"]["message"]

    @pytest.mark.asyncio
    async def test_timeout_kills_runaway(self):
        tool = RunPython(executor=EXECUTOR, timeout=2.0)
        result = await tool(code="while True:\n    pass").collect()
        out = result.output
        assert out["status"] == "error"
        assert out["error"]["type"] == "TimeoutError"


class TestRunPythonAsync:
    """Top-level await support (so agent-written async code just works)."""

    @pytest.mark.asyncio
    async def test_top_level_await_return_value(self, run_python: RunPython):
        code = """
import asyncio
await asyncio.sleep(0)
total = 0
for i in range(5):
    await asyncio.sleep(0)
    total += i
total
"""
        out = await collect(run_python, code)
        assert out["return_value"] == 10

    @pytest.mark.asyncio
    async def test_top_level_await_with_gather(self, run_python: RunPython):
        code = """
import asyncio

async def work(n):
    await asyncio.sleep(0)
    return n * n

results = await asyncio.gather(*[work(i) for i in range(4)])
sum(results)
"""
        out = await collect(run_python, code)
        assert out["return_value"] == 14

    @pytest.mark.asyncio
    async def test_await_in_codemode(self):
        """Async user code can still call exposed tools synchronously."""
        tool = RunPython(tools=[double], executor=EXECUTOR)
        code = """
import asyncio
await asyncio.sleep(0)
double(21)
"""
        out = await collect(tool, code)
        assert out["return_value"] == 42


class TestRunPythonRunner:
    """Unit tests for the stdlib-only child runner module (no subprocess)."""

    def test_execute_user_code_sync_last_expr(self):
        from timbal.tools._run_python_runner import execute_user_code

        out = execute_user_code("1 + 2", {})
        assert out["error"] is None
        assert out["return_value"] == 3

    def test_execute_user_code_sync_exec_only(self):
        from timbal.tools._run_python_runner import execute_user_code

        out = execute_user_code("x = 10", {})
        assert out["error"] is None
        assert out["return_value"] is None

    def test_execute_user_code_catches_exception(self):
        from timbal.tools._run_python_runner import execute_user_code

        out = execute_user_code("1 / 0", {})
        assert out["return_value"] is None
        assert out["error"]["type"] == "ZeroDivisionError"

    def test_execute_user_code_non_json_serializable_return(self):
        from timbal.tools._run_python_runner import execute_user_code

        out = execute_user_code("object()", {})
        assert out["error"] is None
        assert isinstance(out["return_value"], str)
        assert "object" in out["return_value"]

    def test_has_top_level_await_detects_await(self):
        import ast

        from timbal.tools._run_python_runner import has_top_level_await

        tree = ast.parse("import asyncio\nawait asyncio.sleep(0)")
        assert has_top_level_await(tree) is True

    def test_has_top_level_await_ignores_nested(self):
        import ast

        from timbal.tools._run_python_runner import has_top_level_await

        tree = ast.parse("async def f():\n    await asyncio.sleep(0)")
        assert has_top_level_await(tree) is False

    def test_execute_user_code_top_level_await(self):
        from timbal.tools._run_python_runner import execute_user_code

        code = """
import asyncio
await asyncio.sleep(0)
sum(range(4))
"""
        out = execute_user_code(code, {})
        assert out["error"] is None
        assert out["return_value"] == 6

    def test_build_proxies_calls_call_tool(self, monkeypatch: pytest.MonkeyPatch):
        from timbal.tools._run_python_runner import build_proxies

        calls: list[tuple[str, list[Any], dict[str, Any]]] = []

        def fake_call_tool(name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
            calls.append((name, args, kwargs))
            return 42

        monkeypatch.setattr("timbal.tools._run_python_runner.call_tool", fake_call_tool)
        proxies = build_proxies(["double"])
        assert proxies["double"](21) == 42
        assert calls == [("double", [21], {})]


class TestRunPythonRpcTransport:
    @pytest.mark.asyncio
    async def test_code_mode_uses_tcp_when_unix_server_unavailable(self, monkeypatch):
        """Regression: Windows lacks asyncio.start_unix_server."""
        import asyncio

        if not hasattr(asyncio, "start_unix_server"):
            pytest.skip("platform already lacks unix server")

        monkeypatch.delattr(asyncio, "start_unix_server", raising=False)
        tool = RunPython(tools=[noop], executor=EXECUTOR)
        out = await collect(tool, "noop()")
        assert out["return_value"] == "called"


class TestTimbalInTimbal:
    """Install timbal inside the sandbox and run an Agent within an Agent.

    This is the headline use case: an outer agent generates Python that spins up
    a fully isolated inner Timbal agent (its own installed timbal, its own LLM
    call) and returns the result. Requires ANTHROPIC_API_KEY and uv.
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY required"
    )
    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv required")
    async def test_inner_agent_runs(self, monkeypatch: pytest.MonkeyPatch):
        # hatch-vcs derives the version from git, which may be unavailable in the
        # isolated build; pin a pretend version so the source build never fails.
        monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")

        repo_root = _repo_root()
        tool = RunPython(
            dependencies=[f"timbal @ file://{repo_root}"],
            env_passthrough=["ANTHROPIC_API_KEY", "SETUPTOOLS_SCM_PRETEND_VERSION"],
            timeout=900,
            executor="uv",
        )

        code = """
from timbal import Agent

inner = Agent(
    name="inner_agent",
    model="anthropic/claude-haiku-4-5",
    tools=[],
    max_tokens=64,
    system_prompt="You reply with exactly one lowercase word and nothing else.",
)

result = await inner(prompt="Reply with the single word: pong").collect()
text = result.output.collect_text().strip().lower()
{"text": text, "status": result.status.code}
"""

        result = await tool(code=code).collect()
        assert result.error is None, result.error
        out = result.output
        assert out["status"] == "success", out
        assert out["return_value"]["status"] == "success", out["return_value"]
        assert "pong" in out["return_value"]["text"], out["return_value"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY required"
    )
    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv required")
    async def test_outer_agent_delegates_to_inner_agent(self, monkeypatch: pytest.MonkeyPatch):
        """End-to-end: an outer Agent uses run_python to launch an inner Agent."""
        monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
        repo_root = _repo_root()

        run_python = RunPython(
            dependencies=[f"timbal @ file://{repo_root}"],
            env_passthrough=["ANTHROPIC_API_KEY", "SETUPTOOLS_SCM_PRETEND_VERSION"],
            timeout=900,
            executor="uv",
        )

        outer = Agent(
            name="orchestrator",
            model="anthropic/claude-haiku-4-5",
            max_tokens=1024,
            system_prompt=(
                "You orchestrate work by writing Python for the run_python tool. "
                "When asked to delegate, write code that imports `from timbal import Agent`, "
                "creates an inner Agent with model 'anthropic/claude-haiku-4-5' and max_tokens=64, "
                "calls `result = await inner(prompt=...).collect()`, and ends with the expression "
                "`result.output.collect_text()` so the value is returned. "
                "Then report the inner agent's answer to the user."
            ),
            tools=[run_python],
        )

        response = await outer(
            prompt=(
                "Delegate to an inner Timbal agent: ask it to translate the word "
                "'hello' to Spanish. Use run_python to create and run that inner agent."
            )
        ).collect()
        assert response.error is None, response.error
        assert response.output is not None
        assert "hola" in response.output.collect_text().lower()
