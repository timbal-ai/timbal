from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog

from ...state.context import RunContext
from ...types.events.base import BaseEvent as TimbalBaseEvent
from ...types.events.chunk import ChunkEvent as TimbalChunkEvent
from ...types.events.output import OutputEvent as TimbalOutputEvent
from ...types.events.start import StartEvent as TimbalStartEvent
from .. import register_collector
from ..base import EventCollector

logger = structlog.get_logger("timbal.collectors.impl.timbal")


@register_collector
class TimbalCollector(EventCollector):
    """Collector for Timbal events."""
    
    def __init__(self, run_context: RunContext):
        super().__init__(run_context)
        self._output: Any = None
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, TimbalBaseEvent)

    @override
    def process(self, event: TimbalBaseEvent) -> TimbalBaseEvent | None:
        """Processes Timbal events."""
        if isinstance(event, TimbalStartEvent):
            return self._handle_start_event(event)
        
        if isinstance(event, TimbalChunkEvent):
            return self._handle_chunk_event(event)
        
        if isinstance(event, TimbalOutputEvent):
            return self._handle_output_event(event)
        
        logger.warning("Unknown Timbal event type", event_type=type(event), event=event.model_dump())
        return None
    
    def _handle_start_event(self, event: TimbalStartEvent) -> TimbalStartEvent:
        return event
    
    def _handle_chunk_event(self, event: TimbalChunkEvent) -> TimbalChunkEvent:
        return event
    
    def _handle_output_event(self, event: TimbalOutputEvent) -> TimbalOutputEvent:
        """Handle output events with usage statistics and final results."""
        self._output = event.output
        return event
    
    @override
    def collect(self) -> Any:
        return self._output