import contextlib
import importlib
import inspect
import io
import re


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
        for cls_name in getattr(tools_module, "__all__", []):
            cls = getattr(tools_module, cls_name, None)
            if cls is None or not inspect.isclass(cls) or not issubclass(cls, Runnable) or cls is Runnable:
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
