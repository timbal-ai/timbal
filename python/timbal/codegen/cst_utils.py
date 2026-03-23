from __future__ import annotations

from typing import TYPE_CHECKING

import libcst as cst

from .tool_discovery import get_framework_tool_names

if TYPE_CHECKING:
    pass

ENTRY_POINT_TYPES = {"Agent", "Workflow"}


def resolve_entry_point_type(tree: cst.Module, entry_point: str) -> str | None:
    """Return the constructor class name ('Agent' or 'Workflow') for the entry point variable.

    Inspects top-level assignments to find `entry_point = ClassName(...)` and returns
    the class name if it's a known entry point type. Returns None if not found.
    """
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name) and target.target.value == entry_point:
                            # Walk down chained method calls (e.g. Workflow(...).step(...).step(...))
                            # to find the root constructor.
                            call = item.value
                            while isinstance(call, cst.Call) and isinstance(call.func, cst.Attribute):
                                call = call.func.value
                            if isinstance(call, cst.Call) and isinstance(call.func, cst.Name):
                                cls_name = call.func.value
                                if cls_name in ENTRY_POINT_TYPES:
                                    return cls_name
    return None


def _get_string_value(node: cst.BaseExpression) -> str | None:
    """Extract the string value from a CST string node."""
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value
    if isinstance(node, cst.ConcatenatedString):
        return node.evaluated_value
    return None


def _get_kwarg(call: cst.Call, name: str) -> cst.BaseExpression | None:
    """Get the value of a keyword argument from a Call node."""
    for arg in call.args:
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == name:
            return arg.value
    return None


def _name_from_call(call: cst.Call) -> str | None:
    """Extract the runnable name from a Call node (constructor invocation).

    Mirrors the runtime resolution order:
    1. Explicit name= kwarg  →  that string
    2. handler= kwarg that is a Name  →  the function name (= __name__ at runtime)
    3. Callable name itself (e.g. WebSearch() → "WebSearch")
    """
    name_val = _get_kwarg(call, "name")
    if name_val is not None:
        return _get_string_value(name_val)

    handler_val = _get_kwarg(call, "handler")
    if isinstance(handler_val, cst.Name):
        return handler_val.value

    # Fall back to the callable name, mapping to runtime name for framework tools.
    if isinstance(call.func, cst.Name):
        return get_framework_tool_names().get(call.func.value, call.func.value)

    return None


def resolve_runnable_name(
    element: cst.BaseExpression,
    assignments: dict[str, cst.Call] | None = None,
) -> str | None:
    """Resolve the runtime name of a runnable from a CST element.

    Given a CST node from inside a list (e.g. tools=[...], steps=[...]),
    determine what name the runnable will have at runtime.

    Cases:
    - Bare Name (e.g. `my_func`):
        Look up the variable in assignments. If it's assigned to a Call,
        try to extract name from that Call. Otherwise fall back to the
        variable name itself (bare function → __name__ = variable name).
    - Inline Call (e.g. `CalaSearch(name="x")`):
        Extract name from the Call's kwargs.
    """
    if isinstance(element, cst.Name):
        var_name = element.value
        if assignments and var_name in assignments:
            call = assignments[var_name]
            resolved = _name_from_call(call)
            if resolved is not None:
                return resolved
        # Bare function reference — name = variable name.
        return var_name

    if isinstance(element, cst.Call):
        return _name_from_call(element)

    return None


def build_cst_value(value: object) -> cst.BaseExpression:
    """Recursively convert a Python value into a CST expression."""
    if isinstance(value, bool):
        return cst.Name("True" if value else "False")
    if isinstance(value, int):
        if value < 0:
            return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer(str(-value)))
        return cst.Integer(str(value))
    if isinstance(value, float):
        # Ensure the string representation has a decimal point (cst.Float rejects "0").
        s = str(value)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        if value < 0:
            return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Float(s[1:]))
        return cst.Float(s)
    if isinstance(value, str):
        # Use repr() to properly escape special characters (newlines, quotes, etc.)
        # then wrap as a SimpleString CST node.
        return cst.SimpleString(repr(value))
    if value is None:
        return cst.Name("None")
    if isinstance(value, list):
        elements = [cst.Element(value=build_cst_value(v)) for v in value]
        return cst.List(elements=elements)
    if isinstance(value, dict):
        elements = [cst.DictElement(key=build_cst_value(k), value=build_cst_value(v)) for k, v in value.items()]
        return cst.Dict(elements=elements)
    raise TypeError(f"Unsupported type for CST conversion: {type(value)}")


def collect_assignments(tree: cst.Module) -> dict[str, cst.Call]:
    """Build a map of variable_name -> Call node for all top-level assignments."""
    result = {}
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.Assign) and isinstance(item.value, cst.Call):
                    for target in item.targets:
                        if isinstance(target.target, cst.Name):
                            result[target.target.value] = item.value
    return result


def has_import(tree: cst.Module, module: str, name: str) -> bool:
    """Check if `from <module> import <name>` already exists."""
    for stmt in tree.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for item in stmt.body:
                if isinstance(item, cst.ImportFrom) and not isinstance(item.names, cst.ImportStar):
                    parts = []
                    node = item.module
                    while isinstance(node, cst.Attribute):
                        parts.append(node.attr.value)
                        node = node.value
                    if isinstance(node, cst.Name):
                        parts.append(node.value)
                    mod = ".".join(reversed(parts))

                    if mod == module:
                        for alias in item.names:
                            if isinstance(alias, cst.ImportAlias):
                                imported = alias.name.value if isinstance(alias.name, cst.Name) else ""
                                if imported == name:
                                    return True
    return False
