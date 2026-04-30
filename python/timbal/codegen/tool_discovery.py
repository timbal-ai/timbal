import contextlib
import hashlib
import importlib
import inspect
import io
import json
import re
import typing
from pathlib import Path
from typing import Annotated


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class FrameworkTool:
    """Metadata for a framework tool discovered from timbal.tools."""

    __slots__ = ("module", "name", "description", "provider", "provider_logo")

    def __init__(
        self, module: str, name: str, description: str | None, provider: str | None, provider_logo: str | None
    ):
        self.module = module
        self.name = name
        self.description = description
        self.provider = provider
        self.provider_logo = provider_logo


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / ".tool_cache"
_CACHE_FILE = _CACHE_DIR / "framework_tools.json"


def _tools_fingerprint() -> str:
    """Hash the tools package __init__.py to detect when the tool list changes."""
    init_path = Path(__file__).resolve().parent.parent / "tools" / "__init__.py"
    if init_path.exists():
        return hashlib.md5(init_path.read_bytes()).hexdigest()
    return ""


def _load_cache(fingerprint: str) -> dict[str, FrameworkTool] | None:
    """Load cached tool registry if the fingerprint matches."""
    try:
        if not _CACHE_FILE.exists():
            return None
        data = json.loads(_CACHE_FILE.read_text())
        if data.get("fingerprint") != fingerprint:
            return None
        registry: dict[str, FrameworkTool] = {}
        for cls_name, entry in data["tools"].items():
            registry[cls_name] = FrameworkTool(
                module=entry["module"],
                name=entry["name"],
                description=entry["description"],
                provider=entry["provider"],
                provider_logo=entry["provider_logo"],
            )
        return registry
    except Exception:
        return None


def _save_cache(fingerprint: str, registry: dict[str, FrameworkTool]) -> None:
    """Persist the tool registry to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "fingerprint": fingerprint,
            "tools": {
                cls_name: {
                    "module": ft.module,
                    "name": ft.name,
                    "description": ft.description,
                    "provider": ft.provider,
                    "provider_logo": ft.provider_logo,
                }
                for cls_name, ft in registry.items()
            },
        }
        _CACHE_FILE.write_text(json.dumps(data))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_framework_tools() -> dict[str, FrameworkTool]:
    """Introspect timbal.tools classes (the slow path)."""
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
            provider = None
            provider_logo = None
            integration_field = cls.model_fields.get("integration")
            if integration_field and integration_field.annotation is not None:
                from timbal.platform.integrations import Integration as _Integration

                # The annotation may be Union[Annotated[str, Integration(...)], None],
                # so unwrap union args and check each for Annotated metadata.
                union_args = typing.get_args(integration_field.annotation)
                candidates = union_args if union_args else [integration_field.annotation]
                for candidate in candidates:
                    if typing.get_origin(candidate) is Annotated:
                        for meta in typing.get_args(candidate)[1:]:
                            if isinstance(meta, _Integration):
                                provider = meta.provider
                                provider_logo = f"https://content.timbal.ai/assets/{provider}_favicon.svg"
                                break
                    if provider:
                        break
            registry[cls_name] = FrameworkTool(
                module="timbal.tools",
                name=runtime_name,
                description=description,
                provider=provider,
                provider_logo=provider_logo,
            )

    return registry


def invalidate_cache() -> None:
    """Delete the disk cache, forcing a full rediscovery on the next call."""
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
    except Exception:
        pass


def get_framework_tools(*, no_cache: bool = False) -> dict[str, FrameworkTool]:
    """Discover framework tools from timbal.tools.

    Uses a disk cache keyed by a hash of ``timbal/tools/__init__.py`` to avoid
    the expensive Pydantic class introspection on repeated calls.

    Args:
        no_cache: If True, skip the cache and force a full rediscovery.

    Returns dict: class_name -> FrameworkTool
    e.g. {"WebSearch": FrameworkTool(module="timbal.tools", name="web_search", ...), ...}
    """
    fingerprint = _tools_fingerprint()

    if not no_cache:
        cached = _load_cache(fingerprint)
        if cached is not None:
            return cached

    registry = _discover_framework_tools()
    _save_cache(fingerprint, registry)
    return registry


def get_provider_summaries(*, no_cache: bool = False) -> list[dict]:
    """Return provider summaries with tool counts, sorted by count descending.

    Tools with no provider are grouped under ``"system"``.
    """
    tools = get_framework_tools(no_cache=no_cache)
    groups: dict[str, dict] = {}
    for ft in tools.values():
        provider = ft.provider or "system"
        if provider not in groups:
            groups[provider] = {
                "name": provider,
                "logo": ft.provider_logo,
                "tool_count": 0,
            }
        groups[provider]["tool_count"] += 1
    return sorted(groups.values(), key=lambda g: g["tool_count"], reverse=True)


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
