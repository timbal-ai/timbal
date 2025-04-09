from .base import BaseEvent


class StartEvent(BaseEvent):
    """Event emitted when a step starts execution."""
    type: str = "START"
