from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..base import BaseCollector


class DefaultCollector(BaseCollector):
    """Default fallback collector that handles any event type."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
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
    def result(self) -> Any:
        """Returns all the collected events."""
        return self._events
