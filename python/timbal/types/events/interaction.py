from typing import Any, Literal

from pydantic import Field

from .base import BaseEvent


class InteractionEvent(BaseEvent):
    """Event emitted when a runnable suspends and waits for an external value.

    This is the general counterpart to :class:`ApprovalEvent`: instead of a
    yes/no permission decision, the run pauses for arbitrary input (a user's
    answer, a picked option, a confirmation, ...). The frontend renders
    ``payload`` based on ``kind`` and resumes the run with
    ``resume={interaction_id: value}``.
    """

    type: Literal["INTERACTION"] = "INTERACTION"

    t0: int
    """Unix-ms timestamp at which the run suspended. Useful for SLA timers."""
    interaction_id: str
    """Stable identifier used to resume this suspension (the suspension_id)."""
    kind: str
    """Discriminator the frontend uses to pick a renderer (e.g. ``ask_user``)."""
    runnable_path: str
    """Full runnable path that suspended."""
    runnable_name: str
    """Runnable name that suspended."""
    runnable_type: str
    """Runnable class/type that suspended."""
    tool_call_id: str | None = None
    """The LLM tool_call id that triggered this suspension, when it happened inside
    an agent tool. Lets the frontend correlate the interaction with the exact
    tool_use block in the chat transcript. ``None`` for direct (non-agent) calls."""
    payload: dict[str, Any] = Field(default_factory=dict)
    """JSON-serializable data describing what the caller must supply."""
    response_schema: dict[str, Any] | None = None
    """Optional JSON Schema describing the shape the resume value must match.
    The frontend can validate the user's input client-side before resuming with
    ``resume={interaction_id: value}``. ``None`` means any value is accepted."""
