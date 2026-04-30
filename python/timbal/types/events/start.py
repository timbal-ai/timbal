from typing import Literal

from .base import BaseEvent


class StartEvent(BaseEvent):
    """Event emitted when a step starts execution."""

    type: Literal["START"] = "START"
