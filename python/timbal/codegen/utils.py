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
