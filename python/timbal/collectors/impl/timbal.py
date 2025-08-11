from typing import Any, override

from ...state.context import RunContext
from ...types.events.base import BaseEvent as TimbalBaseEvent
from ...types.events.chunk import ChunkEvent as TimbalChunkEvent
from ...types.events.output import OutputEvent as TimbalOutputEvent
from ...types.events.start import StartEvent as TimbalStartEvent
from .. import register_collector
from ..base import EventCollector


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
    def process(self, event: TimbalBaseEvent) -> dict[str, Any] | None:
        """Processes Timbal events."""
        if isinstance(event, TimbalStartEvent):
            return self._handle_start_event(event)
        
        if isinstance(event, TimbalChunkEvent):
            return self._handle_chunk_event(event)
        
        if isinstance(event, TimbalOutputEvent):
            return self._handle_output_event(event)
        
        return None
    
    def _handle_start_event(self, _event: TimbalStartEvent) -> None:
        return None
    
    def _handle_chunk_event(self, _event: TimbalChunkEvent) -> None:
        return None
    
    def _handle_output_event(self, event: TimbalOutputEvent) -> None:
        """Handle output events with usage statistics and final results."""
        self._output = event.output
        return None
    
    @override
    def collect(self) -> Any:
        return self._output