"""Stdlib-only runner sent to Modal. Never execute agent code on the host."""

import ast
import asyncio
import inspect
import json
import sys
import traceback
from pathlib import Path


def execute(code: str) -> dict:
    """Evaluate the final expression, preserving module scope and top-level await."""
    namespace = {"__name__": "__main__"}
    try:
        tree = ast.parse(code, filename="<run_python>")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            expression = tree.body[-1]
            tree.body[-1] = ast.copy_location(
                ast.Assign(targets=[ast.Name(id="__timbal_result__", ctx=ast.Store())], value=expression.value),
                expression,
            )
        compiled = compile(ast.fix_missing_locations(tree), "<run_python>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        pending = eval(compiled, namespace)
        if inspect.isawaitable(pending):
            asyncio.run(pending)
        value = namespace.get("__timbal_result__")
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            value = repr(value)
        return {"return_value": value, "error": None}
    except BaseException as exc:
        # SystemExit must not discard the execution result.
        return {
            "return_value": None,
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }


def main() -> None:
    request = json.load(sys.stdin)
    result = execute(request["code"])
    payload = json.dumps(result, allow_nan=False).encode()
    if len(payload) > request["max_result_bytes"]:
        payload = json.dumps(
            {
                "return_value": None,
                "error": {"type": "OutputLimitError", "message": "Result exceeds max_result_bytes."},
            }
        ).encode()
    Path("/tmp/timbal-result.json").write_bytes(payload)


if __name__ == "__main__":
    main()
