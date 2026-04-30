from typing import Any, Literal

from pydantic import Field

from .base import BaseEvent


class ApprovalEvent(BaseEvent):
    """Event emitted when runnable execution is waiting on human approval."""

    type: Literal["APPROVAL"] = "APPROVAL"

    t0: int
    """Unix-ms timestamp at which approval was requested. Useful for SLA timers."""
    approval_id: str
    """Stable identifier used to approve or deny this runnable invocation."""
    runnable_path: str
    """Full runnable path that requires approval."""
    runnable_name: str
    """Runnable name that requires approval."""
    runnable_type: str
    """Runnable class/type that requires approval."""
    input: Any
    """Validated input that would be passed to the runnable."""
    prompt: str | None = None
    """Optional human-readable prompt explaining what needs approval."""
    description: str | None = None
    """Optional runnable or policy description."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional policy metadata for future approval engines."""
