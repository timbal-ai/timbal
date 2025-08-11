from typing import Any

import structlog

from .base import EventCollector
from .impl.default import DefaultCollector

logger = structlog.get_logger("timbal.collectors.registry")


class CollectorRegistry:
    """Registry for managing event collector types."""
    
    def __init__(self):
        self._collector_types: list[type[EventCollector]] = []
    
    def register(self, collector_type: type[EventCollector]) -> None:
        """Register a new event collector type."""
        self._collector_types.append(collector_type)
        logger.info(f"Registered collector: {collector_type.__name__}")
    
    def get_collector_type(self, event: Any) -> type[EventCollector] | None:
        """Get the appropriate collector type for the given event."""
        for collector_type in self._collector_types:
            if collector_type.can_handle(event):
                return collector_type
        
        return DefaultCollector
