import contextlib
import importlib
import inspect
import io
import re
import sys

import libcst as cst


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class FrameworkTool:
    """Metadata for a framework tool discovered from timbal.tools."""

    __slots__ = ("module", "name", "description")

    def __init__(self, module: str, name: str, description: str | None):
        self.module = module
        self.name = name
        self.description = description


def get_framework_tools() -> dict[str, FrameworkTool]:
    """Discover framework tools from timbal.tools.

    Returns dict: class_name -> FrameworkTool
    e.g. {"WebSearch": FrameworkTool(module="timbal.tools", name="web_search", ...), ...}
    """
    redirect = io.StringIO()
    with contextlib.redirect_stdout(redirect):
        from pydantic_core import PydanticUndefined

        import timbal.tools as tools_module
        from timbal.core.runnable import Runnable

        registry: dict[str, FrameworkTool] = {}
        for cls_name, cls in inspect.getmembers(tools_module, inspect.isclass):
            if not issubclass(cls, Runnable) or cls is Runnable:
                continue
            name_field = cls.model_fields.get("name")
            if name_field and name_field.default is not PydanticUndefined:
                runtime_name = name_field.default
            else:
                runtime_name = _camel_to_snake(cls_name)
            desc_field = cls.model_fields.get("description")
            description = desc_field.default if desc_field and desc_field.default is not PydanticUndefined else None
            registry[cls_name] = FrameworkTool(module="timbal.tools", name=runtime_name, description=description)

    return registry


def get_framework_tool_names() -> dict[str, str]:
    """Return class_name -> runtime_name mapping for framework tools."""
    return {cls: ft.name for cls, ft in get_framework_tools().items()}


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
                            if isinstance(item.value, cst.Call) and isinstance(item.value.func, cst.Name):
                                cls_name = item.value.func.value
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


def validate_tool_config(tool_type: str, config: dict) -> None:
    """Validate config keys against the tool's config model fields."""
    # Map tool type to (module_path, class_name).
    registry = get_framework_tools()
    if tool_type in registry:
        module_path = registry[tool_type].module
    elif tool_type == "Tool":
        module_path = "timbal.core"
    else:
        raise ValueError(f"--config is not supported for {tool_type}.")
    redirect = io.StringIO()
    with contextlib.redirect_stdout(redirect):
        mod = importlib.import_module(module_path)
    config_cls = getattr(mod, tool_type)
    valid_fields = set(config_cls.model_fields.keys())
    unknown = set(config.keys()) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config field(s) for {tool_type}: {', '.join(sorted(unknown))}. "
            f"Valid fields: {', '.join(sorted(valid_fields))}."
        )
