# ruff: noqa: F401
from .registry import CollectorRegistry

# Private registry instance
_collector_registry = CollectorRegistry()
_collectors_loaded = False


def _ensure_collectors_loaded():
    """Lazily import collector implementations on first use.

    This avoids eagerly importing anthropic/openai SDKs at package import time.
    """
    global _collectors_loaded
    if _collectors_loaded:
        return
    _collectors_loaded = True
    # Import all collector implementations to register them.
    # NOTE: Do not register the default collector, this might lead to
    # unexpected behavior when detecting the correct collector type.
    from .impl import anthropic, openai, string, timbal  # noqa: F811, E402


def get_collector_registry():
    """Get the global collector registry instance."""
    _ensure_collectors_loaded()
    return _collector_registry


def register_collector(collector_class):
    """Decorator to automatically register collectors with the module registry."""
    _collector_registry.register(collector_class)
    return collector_class
