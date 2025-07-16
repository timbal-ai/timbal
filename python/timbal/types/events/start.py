from typing import Literal

from .base import BaseEvent


class StartEvent(BaseEvent):
    """Event emitted when a step starts execution."""
    type: Literal["START"] = "START"

    status_text: str | None = None
    """Optional user-facing text describing the action the step is performing.
    
    Intended for display in UIs to show agent activity, e.g., 
    "Thinking...", "Searching the web...", "Running tool: get_weather".
    """
