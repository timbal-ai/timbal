# ruff: noqa: F401
from .base import Exporter, TracingProvider
from .in_memory import InMemoryTracingProvider
from .jsonl import JsonlTracingProvider
from .platform import PlatformTracingProvider
from .sqlite import SqliteTracingProvider


class _TracingProviderUnset:
    """Sentinel: tracing_provider was not explicitly set — use default resolution logic."""
    _instance: "_TracingProviderUnset | None" = None

    def __new__(cls) -> "_TracingProviderUnset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "TRACING_UNSET"


TRACING_UNSET = _TracingProviderUnset()
