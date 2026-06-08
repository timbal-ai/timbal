from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class ApprovalPolicyDecision(BaseModel):
    """Normalized approval policy decision for a runnable invocation.

    The ``approval_id`` is derived from ``(runnable_path, validated_input)``: a
    single decision approves every invocation with the same path + input. For
    irreversible operations (money movement, destructive deletes) include a
    unique value in the input (e.g. ``idempotency_key=uuid4()``) so each call
    derives a distinct ``approval_id`` and requires a fresh decision.
    """

    required: bool
    prompt: str | None = None
    description: str | None = None
    kind: str | None = None
    """Renderer discriminator the frontend uses to pick an approval card component
    (e.g. ``"create_project"``, ``"wire_transfer"``). Mirrors ``InteractionEvent.kind``."""
    ui: dict[str, Any] | None = None
    """Structured, presentation-only JSON for rendering a rich approval card
    (title, fields, severity, ...). Built from the *redacted* input, so secrets
    excluded via ``approval_redactor`` / ``approval_redact_keys`` never reach it."""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalResolution(BaseModel):
    """Decision supplied by a human or caller to resume an approval gate.

    Resolutions can carry an optional ``expires_at`` (Unix ms). Stale resolutions
    are ignored at gate time and the gate emits a fresh ``ApprovalEvent``.

    Audit fields (``approver_id``, ``comment``, ``decided_at``) are first-class
    so traces serialize them verbatim and downstream queries
    (``span.metadata.approval.resolution.approver_id``) work without grepping
    a free-form ``metadata`` dict. ``metadata`` remains for org-specific extras.
    """

    approved: bool
    reason: str | None = None
    override_input: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Edit-on-approve: when approved, these key/value pairs are merged over the "
            "originally-proposed input (override wins) and re-validated before the handler "
            "runs. Lets a human tweak the proposal (e.g. fix a recipient) without rejecting it. "
            "Ignored when approved is False. The resulting input is recorded in the trace audit."
        ),
    )
    expires_at: int | None = Field(
        default=None,
        description=(
            "Optional Unix-ms timestamp after which this resolution is no longer honoured. None means never expires."
        ),
    )
    approver_id: str | None = Field(
        default=None,
        description="Identifier of the human/caller who decided this resolution. Surfaced in trace audit.",
    )
    comment: str | None = Field(
        default=None,
        description="Free-form human reasoning for the decision. Surfaced in trace audit.",
    )
    decided_at: int | None = Field(
        default=None,
        description=(
            "Unix-ms timestamp when the decision was made. Defaults to construction time when not given. "
            "Pass an explicit value to preserve idempotency across replays."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.decided_at is None:
            self.decided_at = int(time.time() * 1000)

    def is_expired(self, now_ms: int | None = None) -> bool:
        if self.expires_at is None:
            return False
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return now_ms >= self.expires_at


# Discriminator used to recognize a cancel sent as plain JSON over the wire
# (HTTP clients can't ship a Cancel instance). A resume value shaped like
# ``{"type": "timbal.cancel", "reason": "..."}`` is coerced to ``Cancel``.
CANCEL_TYPE = "timbal.cancel"


class Cancel(BaseModel):
    """Universal resume value that *aborts the whole run*.

    Pass it on the ``resume=`` channel keyed by any pending approval or
    suspension id — ``resume={pending_id: Cancel("user closed the dialog")}``.
    When that gate/suspension is reached the run terminates with status
    ``cancelled`` / reason ``cancelled`` and the handler never executes.

    This is distinct from a *decline*:

    - **decline** (approval ``approved=False`` / a handler-interpreted suspend
      value) feeds a result back to the model so the agent keeps going on a
      different path.
    - **cancel** stops the entire run. Nothing is fed back to the model.

    Over HTTP/JSON (where you can't pass a Python object) send the equivalent
    tagged dict: ``{"type": "timbal.cancel", "reason": "..."}``.
    """

    type: Literal["timbal.cancel"] = CANCEL_TYPE
    reason: str | None = Field(
        default=None,
        description="Optional human-readable reason, surfaced on the cancelled span/status.",
    )

    @classmethod
    def matches(cls, value: Any) -> bool:
        """Return True if a raw resume value is a cancel (instance or tagged dict)."""
        if isinstance(value, cls):
            return True
        return isinstance(value, dict) and value.get("type") == CANCEL_TYPE
