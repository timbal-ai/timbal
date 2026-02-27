import importlib
import io
import sys

import libcst as cst

# Class name -> runtime name for framework tools.
FRAMEWORK_TOOL_NAMES: dict[str, str] = {
    "Bash": "bash",
    "CalaSearch": "cala_search",
    "Edit": "edit",
    "Read": "read",
    "WebSearch": "web_search",
    "Write": "write",
}

# Framework tools that accept a config model: class name -> (module, config class name)
FRAMEWORK_TOOL_CONFIGS: dict[str, tuple[str, str]] = {
    "WebSearch": ("timbal.tools.web_search", "WebSearchConfig"),
    "CalaSearch": ("timbal.tools.cala", "CalaConfig"),
}


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
        return FRAMEWORK_TOOL_NAMES.get(call.func.value, call.func.value)

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
        return cst.Integer(str(value))
    if isinstance(value, float):
        return cst.Float(str(value))
    if isinstance(value, str):
        return cst.SimpleString(f'"{value}"')
    if value is None:
        return cst.Name("None")
    if isinstance(value, list):
        elements = [cst.Element(value=build_cst_value(v)) for v in value]
        return cst.List(elements=elements)
    if isinstance(value, dict):
        elements = [
            cst.DictElement(key=build_cst_value(k), value=build_cst_value(v))
            for k, v in value.items()
        ]
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


def validate_tool_config(tool_type: str, config: dict) -> None:
    """Validate config keys against the tool's config model fields."""
    config_ref = FRAMEWORK_TOOL_CONFIGS.get(tool_type)
    if config_ref is None:
        raise ValueError(f"--config is not supported for {tool_type}.")
    module_path, class_name = config_ref
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        mod = importlib.import_module(module_path)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    config_cls = getattr(mod, class_name)
    valid_fields = set(config_cls.model_fields.keys())
    unknown = set(config.keys()) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config field(s) for {tool_type}: {', '.join(sorted(unknown))}. "
            f"Valid fields: {', '.join(sorted(valid_fields))}."
        )
