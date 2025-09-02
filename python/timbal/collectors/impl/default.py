from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ...state.context import RunContext
from ..base import EventCollector


class DefaultCollector(EventCollector):
    """Default fallback collector that handles any event type."""
    
    def __init__(self, run_context: RunContext):
        super().__init__(run_context)
        self._events: list[Any] = []
    
    @classmethod
    @override
    def can_handle(cls, _event: Any) -> bool:
        return True
    
    @override
    def process(self, event: Any) -> Any:
        """Collects events in array."""
        self._events.append(event)
        return event
    
    @override
    def collect(self) -> list[Any]:
        return self._events