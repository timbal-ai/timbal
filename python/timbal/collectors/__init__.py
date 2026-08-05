# ruff: noqa: F401
from typing import Any

from .registry import CollectorRegistry

# Private registry instance
_collector_registry = CollectorRegistry()
_base_collectors_loaded = False
_sdk_collectors_loaded: dict[str, bool] = {"anthropic": False, "openai": False}


def _ensure_sdk_collector_for(event: Any) -> None:
    """Import a provider-SDK collector only when a chunk from that SDK appears.

    A stream chunk's class lives in its SDK's module tree, so the module name
    prefix identifies the provider without importing anything. This keeps an
    anthropic-only process from ever importing openai (and vice versa) — the
    collector modules import their SDK's types at module level.
    """
    root = type(event).__module__.split(".", 1)[0]
    if _sdk_collectors_loaded.get(root) is False:
        _sdk_collectors_loaded[root] = True
        if root == "anthropic":
            from .impl import anthropic  # noqa: F811
        else:
            from .impl import openai  # noqa: F811


def _ensure_base_collectors_loaded():
    """Lazily import the SDK-free collector implementations on first use.

    NOTE: Do not register the default collector, this might lead to
    unexpected behavior when detecting the correct collector type.
    """
    global _base_collectors_loaded
    if _base_collectors_loaded:
        return
    _base_collectors_loaded = True
    from .impl import string, timbal  # noqa: F811, E402

    _collector_registry.lazy_loader = _ensure_sdk_collector_for


def get_collector_registry():
    """Get the global collector registry instance."""
    _ensure_base_collectors_loaded()
    return _collector_registry


def register_collector(collector_class):
    """Decorator to automatically register collectors with the module registry."""
    _collector_registry.register(collector_class)
    return collector_class
