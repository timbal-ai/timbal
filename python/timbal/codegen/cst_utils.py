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


def collect_step_names(
    tree: cst.Module,
    entry_point: str,
    assignments: dict[str, cst.Call],
) -> dict[str, str]:
    """Build a mapping of step variable name → runtime name.

    Scans the tree for ``<entry_point>.step(<var>, ...)`` calls and resolves
    each step's runtime name from its assignment.  Returns a dict like
    ``{"agent_b": "Agent 2", "translator": "translator"}``.
    """
    var_to_name: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not isinstance(item, cst.Expr) or not isinstance(item.value, cst.Call):
                continue
            call = item.value
            if not (
                isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name)
                and call.func.value.value == entry_point
                and call.func.attr.value == "step"
                and call.args
            ):
                continue
            first_arg = call.args[0].value
            if isinstance(first_arg, cst.Name):
                var_name = first_arg.value
                resolved = resolve_runnable_name(
                    assignments[var_name],
                ) if var_name in assignments else None
                var_to_name[var_name] = resolved if resolved is not None else var_name
    return var_to_name


def is_bare_function_step(
    tree: cst.Module,
    entry_point: str,
    step_name: str,
    assignments: dict[str, cst.Call],
) -> bool:
    """Check if *step_name* is a bare ``def`` used directly in a ``.step()`` call.

    Returns ``True`` when:
    1. No top-level assignment resolves to *step_name* (i.e. it's not already
       wrapped in ``Tool`` / ``Agent`` / etc.).
    2. A ``FunctionDef`` with that name exists at the module level.
    3. That name appears as the first positional arg in an
       ``<entry_point>.step(step_name, ...)`` call.
    """
    # Already wrapped?
    for _var, call in assignments.items():
        if resolve_runnable_name(call) == step_name:
            return False

    # Has a matching FunctionDef?
    has_func_def = any(
        isinstance(stmt, cst.FunctionDef) and stmt.name.value == step_name
        for stmt in tree.body
    )
    if not has_func_def:
        return False

    # Used as a workflow step?
    for stmt in tree.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for item in stmt.body:
            if not (isinstance(item, cst.Expr) and isinstance(item.value, cst.Call)):
                continue
            call = item.value
            if (
                isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name)
                and call.func.value.value == entry_point
                and call.func.attr.value == "step"
                and call.args
                and isinstance(call.args[0].value, cst.Name)
                and call.args[0].value.value == step_name
            ):
                return True

    return False


class _BareFunctionWrapper(cst.CSTTransformer):
    """Rename ``def step_name(...)`` → ``def step_name_fn(...)`` and insert a
    ``Tool(name="step_name", handler=step_name_fn)`` assignment."""

    def __init__(self, entry_point: str, step_name: str) -> None:
        self.entry_point = entry_point
        self.step_name = step_name
        self._func_renamed = False

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef  # noqa: ARG002
    ) -> cst.FunctionDef:
        if updated_node.name.value == self.step_name:
            self._func_renamed = True
            return updated_node.with_changes(name=cst.Name(f"{self.step_name}_fn"))
        return updated_node

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if not self._func_renamed:
            return updated_node

        body = list(updated_node.body)

        # Add ``from timbal.core import Tool`` if missing.
        if not has_import(original_node, "timbal.core", "Tool"):
            import_insert_idx = 0
            for i, stmt in enumerate(body):
                if isinstance(stmt, cst.SimpleStatementLine):
                    for item_node in stmt.body:
                        if isinstance(item_node, (cst.Import, cst.ImportFrom)):
                            import_insert_idx = i + 1
            body.insert(import_insert_idx, cst.parse_statement("from timbal.core import Tool\n"))

        # Build ``step_name = Tool(name="step_name", handler=step_name_fn)``
        assignment_code = (
            f'{self.step_name} = Tool(name="{self.step_name}", handler={self.step_name}_fn)\n'
        )

        # Insert before the entry-point assignment.
        insert_idx = len(body)
        for i, stmt in enumerate(body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for item_node in stmt.body:
                    if isinstance(item_node, cst.Assign):
                        for t in item_node.targets:
                            if isinstance(t.target, cst.Name) and t.target.value == self.entry_point:
                                insert_idx = min(insert_idx, i)
        body.insert(insert_idx, cst.parse_statement(assignment_code))

        return updated_node.with_changes(body=body)


def wrap_bare_function_step(tree: cst.Module, entry_point: str, step_name: str) -> cst.Module:
    """Wrap a bare function step in a ``Tool`` and return the modified tree."""
    return tree.visit(_BareFunctionWrapper(entry_point, step_name))


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
