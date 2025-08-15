from typing import override

from ..data import Tracing
from .base import TracingProvider


class InMemoryTracingProvider(TracingProvider):
    """In-memory tracing provider using system-wide storage."""

    # Class-level storage: run_id -> Tracing
    _storage: dict[str, Tracing] = {}
    
    @classmethod
    @override
    async def get_tracing(cls, run_id: str) -> Tracing | None:
        """See base class."""
        return cls._storage.get(run_id)
    
    @classmethod
    @override
    async def put_tracing(cls, run_id: str, tracing: Tracing) -> None:
        """See base class."""
        cls._storage[run_id] = tracing
    