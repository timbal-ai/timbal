from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ...state.context import RunContext
from .. import register_collector
from ..base import EventCollector


@register_collector
class StringCollector(EventCollector):
    """Collector for simple string events."""
    
    def __init__(self, run_context: RunContext, start: float):
        super().__init__(run_context, start)
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