# ruff: noqa: F401
from typing import Any

from .approval import ApprovalEvent
from .base import BaseEvent
from .delta import DeltaEvent
from .interaction import InteractionEvent
from .output import OutputEvent
from .start import StartEvent

# Union of all possible event types. Deserialization dispatches on the 'type'
# field via validate_event() (events are plain classes, not pydantic models).
Event = StartEvent | OutputEvent | DeltaEvent | ApprovalEvent | InteractionEvent

_EVENT_TYPES: dict[str, type[BaseEvent]] = {
    StartEvent.type: StartEvent,
    OutputEvent.type: OutputEvent,
    DeltaEvent.type: DeltaEvent,
    ApprovalEvent.type: ApprovalEvent,
    InteractionEvent.type: InteractionEvent,
}


def validate_event(data: BaseEvent | dict[str, Any]) -> Event:
    """Rehydrate an event from its ``model_dump()`` wire form.

    Dispatches on the ``type`` discriminator. Nested structures (RunStatus,
    DeltaItem) are rebuilt by the target class. Unknown keys are ignored,
    matching the old pydantic ``extra="ignore"`` behavior.
    """
    if isinstance(data, BaseEvent):
        return data
    event_type = data.get("type")
    cls = _EVENT_TYPES.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type {event_type!r}. Must be one of {sorted(_EVENT_TYPES)}.")
    return cls(**{k: v for k, v in data.items() if k != "type"})
