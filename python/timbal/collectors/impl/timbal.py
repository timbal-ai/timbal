from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog

from ...types.events.approval import ApprovalEvent as TimbalApprovalEvent
from ...types.events.base import BaseEvent as TimbalBaseEvent
from ...types.events.delta import DeltaEvent as TimbalDeltaEvent
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
        self._output_event: TimbalOutputEvent | None = None
        # Capture every approval gate that fires during the stream so callers
        # of .collect() can react to all pending approvals — not just the
        # first one — when concurrent runnables (parallel workflow steps,
        # multiplexed tools) gate on the same iteration.
        self._pending_approvals: list[dict[str, Any]] = []

    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, TimbalBaseEvent)

    @override
    def process(self, event: TimbalBaseEvent) -> TimbalBaseEvent | None:
        """Processes Timbal events."""
        if isinstance(event, TimbalStartEvent):
            return event
        elif isinstance(event, TimbalDeltaEvent):
            return event
        elif isinstance(event, TimbalApprovalEvent):
            self._pending_approvals.append({
                "approval_id": event.approval_id,
                "runnable_path": event.runnable_path,
                "runnable_name": event.runnable_name,
                "runnable_type": event.runnable_type,
                "input": event.input,
                "prompt": event.prompt,
                "description": event.description,
                "metadata": event.metadata,
                "t0": event.t0,
                "call_id": event.call_id,
                "parent_call_id": event.parent_call_id,
            })
            return event
        elif isinstance(event, TimbalOutputEvent):
            self._output_event = event
            return event
        elif isinstance(event, TimbalBaseEvent):
            return event
        else:
            logger.warning("Unknown Timbal event type", event_type=type(event), event=event)

    @override
    def result(self) -> Any:
        """Returns the final OutputEvent enriched with pending_approvals.

        When concurrent runnables gate, the OutputEvent only references the
        *first* pending approval through ``status``/``output``. We attach the
        full list under ``metadata['pending_approvals']`` so consumers driving
        the resume loop can see every gate from one ``.collect()`` call.
        """
        if self._output_event is not None and self._pending_approvals:
            self._output_event.metadata = {
                **(self._output_event.metadata or {}),
                "pending_approvals": list(self._pending_approvals),
            }
        return self._output_event
