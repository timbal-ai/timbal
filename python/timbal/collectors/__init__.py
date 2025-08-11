# ruff: noqa: F401
from .registry import CollectorRegistry

# Private registry instance
_collector_registry = CollectorRegistry()

def get_collector_registry():
    """Get the global collector registry instance."""
    return _collector_registry

def register_collector(collector_class):
    """Decorator to automatically register collectors with the module registry."""
    _collector_registry.register(collector_class)
    return collector_class

# Import all collector implementations to register them (must be after creating the decorator)
# NOTE: Do not register the default collector, this might lead to unexpected behavior when detecting the correct collector type
from .impl import anthropic, openai, string, timbal  # noqa: E402
