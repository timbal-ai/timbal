from typing import Any, override

from ...state.context import RunContext
from .. import register_collector
from ..base import EventCollector


@register_collector
class StringCollector(EventCollector):
    """Collector for simple string events."""
    
    def __init__(self, run_context: RunContext):
        super().__init__(run_context)
        self._result: str = ""
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, str)
    
    @override
    def process(self, event: str) -> str:
        """Concatenates string events."""
        self._result += event
        return event
    
    @override
    def collect(self) -> str:
        """Returns concatenated string."""
        return self._result