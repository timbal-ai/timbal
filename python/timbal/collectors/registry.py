from typing import Any

from .base import BaseCollector
from .impl.default import DefaultCollector


class CollectorRegistry:
    """Registry for managing event collector types."""
    
    def __init__(self):
        self._collector_types: list[type[BaseCollector]] = []
    
    def register(self, collector_type: type[BaseCollector]) -> None:
        """Register a new event collector type."""
        self._collector_types.append(collector_type)
    
    def get_collector_type(self, event: Any) -> type[BaseCollector] | None:
        """Get the appropriate collector type for the given event."""
        for collector_type in self._collector_types:
            if collector_type.can_handle(event):
                return collector_type
        
        return DefaultCollector
