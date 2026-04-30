from typing import TYPE_CHECKING

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..trace import Trace
from .base import TracingProvider

if TYPE_CHECKING:
    from ...context import RunContext


class InMemoryTracingProvider(TracingProvider):
    """In-memory tracing provider using a process-local dictionary.

    The default provider when no platform credentials are configured.
    Traces are stored in ``cls._storage`` (a plain dict keyed by run_id),
    which makes it fast and zero-dependency — ideal for tests, local
    development, and short-lived scripts.

    .. warning::

        **Not suitable for production use.** Storage lives in process memory
        and is lost on restart. It also grows unboundedly — there is no
        eviction or size limit. For production, use ``PlatformTracingProvider``
        or implement a provider backed by a proper database.
    """

    # Class-level storage: run_id -> Trace
    _storage: dict[str, Trace] = {}
    
    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """See base class."""
        return cls._storage.get(run_context.parent_id)
    
    @classmethod
    @override
    async def _store(cls, run_context: "RunContext") -> None:
        """See base class."""
        cls._storage[run_context.id] = run_context._trace
    