from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog

from ...errors import InterruptError
from ...types.events.base import BaseEvent as TimbalBaseEvent
from ...types.events.chunk import ChunkEvent as TimbalChunkEvent
from ...types.events.output import OutputEvent as TimbalOutputEvent
from ...types.events.start import StartEvent as TimbalStartEvent
from .. import register_collector
from ..base import BaseCollector

logger = structlog.get_logger("timbal.collectors.impl.timbal")


@register_collector
class TimbalCollector(BaseCollector):
    """Collector for Timbal events."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._output_event = None
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, TimbalBaseEvent)

    @override
    def process(self, event: TimbalBaseEvent) -> TimbalBaseEvent | None:
        """Processes Timbal events."""
        if isinstance(event, TimbalStartEvent):
            return event
        elif isinstance(event, TimbalChunkEvent):
            return event
        elif isinstance(event, TimbalOutputEvent):
            self._output_event = event
            return event
        elif isinstance(event, InterruptError):
            pass
        else:
            logger.warning("Unknown Timbal event type", event_type=type(event), event=event)

    @override
    def result(self) -> Any:
        """Returns the output of the last output event received."""
        return self._output_event
