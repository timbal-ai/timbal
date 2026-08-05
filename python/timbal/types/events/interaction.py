from typing import Any

from .base import BaseEvent


class InteractionEvent(BaseEvent):
    """Event emitted when a runnable suspends and waits for an external value.

    This is the general counterpart to :class:`ApprovalEvent`: instead of a
    yes/no permission decision, the run pauses for arbitrary input (a user's
    answer, a picked option, a confirmation, ...). The frontend renders
    ``payload`` based on ``kind`` and resumes the run with
    ``resume={interaction_id: value}``.
    """

    __slots__ = (
        "t0",
        "interaction_id",
        "kind",
        "runnable_path",
        "runnable_name",
        "runnable_type",
        "tool_call_id",
        "payload",
        "response_schema",
    )

    type = "INTERACTION"

    _FIELDS = BaseEvent._FIELDS + (
        "t0",
        "interaction_id",
        "kind",
        "runnable_path",
        "runnable_name",
        "runnable_type",
        "tool_call_id",
        "payload",
        "response_schema",
    )

    def __init__(
        self,
        *,
        run_id: str,
        path: str,
        call_id: str,
        t0: int,
        interaction_id: str,
        kind: str,
        runnable_path: str,
        runnable_name: str,
        runnable_type: str,
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        tool_call_id: str | None = None,
        payload: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            run_id=run_id,
            path=path,
            call_id=call_id,
            parent_run_id=parent_run_id,
            parent_call_id=parent_call_id,
        )
        self.t0 = t0
        """Unix-ms timestamp at which the run suspended. Useful for SLA timers."""
        self.interaction_id = interaction_id
        """Stable identifier used to resume this suspension (the suspension_id)."""
        self.kind = kind
        """Discriminator the frontend uses to pick a renderer (e.g. ``ask_user``)."""
        self.runnable_path = runnable_path
        """Full runnable path that suspended."""
        self.runnable_name = runnable_name
        """Runnable name that suspended."""
        self.runnable_type = runnable_type
        """Runnable class/type that suspended."""
        self.tool_call_id = tool_call_id
        """The LLM tool_call id that triggered this suspension, when it happened inside
        an agent tool. Lets the frontend correlate the interaction with the exact
        tool_use block in the chat transcript. ``None`` for direct (non-agent) calls."""
        self.payload = payload if payload is not None else {}
        """JSON-serializable data describing what the caller must supply."""
        self.response_schema = response_schema
        """Optional JSON Schema describing the shape the resume value must match.
        The frontend can validate the user's input client-side before resuming with
        ``resume={interaction_id: value}``. ``None`` means any value is accepted."""
