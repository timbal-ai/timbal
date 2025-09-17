from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .. import register_collector
from ..base import BaseCollector


@register_collector
class StringCollector(BaseCollector):
    """Collector for simple string events."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._result = ""
    
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
    def result(self) -> Any:
        """Returns the concatenated string."""
        return self._result
