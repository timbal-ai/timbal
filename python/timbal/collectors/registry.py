from collections.abc import Callable
from typing import Any

from .base import BaseCollector
from .impl.default import DefaultCollector


class CollectorRegistry:
    """Registry for managing event collector types."""

    def __init__(self):
        self._collector_types: list[type[BaseCollector]] = []
        self.lazy_loader: Callable[[Any], None] | None = None
        """Optional hook called with the event before dispatch — used to
        lazily import provider-SDK collector implementations only when a
        chunk from that SDK actually appears (see collectors.__init__)."""

    def register(self, collector_type: type[BaseCollector]) -> None:
        """Register a new event collector type."""
        self._collector_types.append(collector_type)

    def get_collector_type(self, event: Any) -> type[BaseCollector] | None:
        """Get the appropriate collector type for the given event."""
        if self.lazy_loader is not None:
            self.lazy_loader(event)
        for collector_type in self._collector_types:
            if collector_type.can_handle(event):
                return collector_type

        return DefaultCollector
