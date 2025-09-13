from typing import TYPE_CHECKING

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .. import Tracing
from .base import TracingProvider

if TYPE_CHECKING:
    from ...context import RunContext


class InMemoryTracingProvider(TracingProvider):
    """In-memory tracing provider using system-wide storage."""

    # Class-level storage: run_id -> Tracing
    _storage: dict[str, Tracing] = {}
    
    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Tracing | None:
        """See base class."""
        return cls._storage.get(run_context.parent_id)
    
    @classmethod
    @override
    async def put(cls, run_context: "RunContext") -> None:
        """See base class."""
        cls._storage[run_context.id] = run_context._tracing
    