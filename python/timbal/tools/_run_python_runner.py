"""Stdlib-only child runner for RunPython.

Executed as a standalone script in an isolated subprocess. The parent passes
user code and configuration via environment variables and files — no string
codegen.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import json
import os
import socket
import textwrap
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

_RPC_ENV_VAR = "TIMBAL_RPC_SOCKET"
_RESULT_PATH_ENV = "TIMBAL_RESULT_PATH"
_USER_CODE_PATH_ENV = "TIMBAL_USER_CODE_PATH"
_EXPOSED_TOOLS_ENV = "TIMBAL_EXPOSED_TOOLS"


def call_tool(name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Call an exposed Timbal tool via the parent RPC socket."""
    sock_path = os.environ.get(_RPC_ENV_VAR)
    if not sock_path:
        raise RuntimeError("Code mode is not available: RPC socket not configured")
    payload = json.dumps({"name": name, "args": args, "kwargs": kwargs}).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(sock_path)
        sock.sendall(payload + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    line = b"".join(chunks).split(b"\n", 1)[0]
    response = json.loads(line.decode())
    if response.get("error"):
        err = response["error"]
        exc_name = err.get("type", "RuntimeError")
        message = err.get("message", str(err))
        exc_cls = getattr(builtins, exc_name, None)
        if exc_cls is None or not isinstance(exc_cls, type) or not issubclass(exc_cls, BaseException):
            if exc_name == "KeyError" and "Unknown tool" in message:
                raise NameError(f"name '{name}' is not defined")
            exc_cls = RuntimeError
            message = f"{exc_name}: {message}"
        raise exc_cls(message)
    return response.get("output")


def write_result(payload: dict[str, Any]) -> None:
    """Persist execution result to the path configured by the parent."""
    result_path = os.environ.get(_RESULT_PATH_ENV, "")
    if result_path:
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)


def has_top_level_await(tree: ast.AST) -> bool:
    """Return True if the module tree contains await at module scope."""

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, _node: ast.Lambda) -> None:
            return None

        def visit_Await(self, _node: ast.Await) -> None:
            self.found = True

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.found = True
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            self.found = True
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.found


def execute_user_code(code: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Execute user code and capture return_value or structured error."""
    result: dict[str, Any] = {"return_value": None, "error": None}
    try:
        tree = ast.parse(code)
        exec_namespace = dict(namespace)
        exec_namespace["__name__"] = "__main__"
        if has_top_level_await(tree):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                ret = ast.Return(value=tree.body[-1].value)
                ast.copy_location(ret, tree.body[-1])
                tree.body[-1] = ret
                ast.fix_missing_locations(tree)
            wrapped = "async def __timbal_async_main__():\n" + textwrap.indent(ast.unparse(tree), "    ")
            exec(compile(wrapped, "<user>", "exec"), exec_namespace)
            result["return_value"] = asyncio.run(exec_namespace["__timbal_async_main__"]())
        elif tree.body and isinstance(tree.body[-1], ast.Expr):
            if len(tree.body) > 1:
                exec(
                    compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<user>", "exec"),
                    exec_namespace,
                )
            last_expr = compile(ast.Expression(body=tree.body[-1].value), "<user>", "eval")
            result["return_value"] = eval(last_expr, exec_namespace)
        else:
            exec(compile(tree, "<user>", "exec"), exec_namespace)
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    if result["error"] is None:
        try:
            json.dumps(result["return_value"])
        except (TypeError, ValueError):
            result["return_value"] = repr(result["return_value"])
    return result


def build_proxies(tool_names: list[str]) -> dict[str, Callable[..., Any]]:
    """Build callable proxies for exposed tools (code mode)."""
    proxies: dict[str, Callable[..., Any]] = {}
    for name in tool_names:
        def _make_proxy(tool_name: str) -> Callable[..., Any]:
            def proxy(*args: Any, **kwargs: Any) -> Any:
                return call_tool(tool_name, list(args), kwargs)

            proxy.__name__ = tool_name
            return proxy

        proxies[name] = _make_proxy(name)
    return proxies


def main() -> None:
    """Entry point: read config from env, execute user code, write result."""
    user_code_path = os.environ.get(_USER_CODE_PATH_ENV, "")
    if not user_code_path:
        write_result(
            {
                "return_value": None,
                "error": {
                    "type": "RuntimeError",
                    "message": f"{_USER_CODE_PATH_ENV} is not set",
                },
            }
        )
        return

    code = Path(user_code_path).read_text(encoding="utf-8")
    exposed_raw = os.environ.get(_EXPOSED_TOOLS_ENV, "[]")
    try:
        tool_names = json.loads(exposed_raw)
    except json.JSONDecodeError:
        tool_names = []

    namespace: dict[str, Any] = {"__name__": "__main__", **build_proxies(tool_names)}
    write_result(execute_user_code(code, namespace))


if __name__ == "__main__":
    main()
