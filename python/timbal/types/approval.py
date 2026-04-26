from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class ApprovalDecision(BaseModel):
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalResolution(BaseModel):
    """Decision supplied by a human or caller to resume an approval gate.

    Resolutions can carry an optional ``expires_at`` (Unix ms). Stale resolutions
    are ignored at gate time and the gate emits a fresh ``ApprovalEvent``.
    """

    approved: bool
    reason: str | None = None
    expires_at: int | None = Field(
        default=None,
        description=(
            "Optional Unix-ms timestamp after which this resolution is no longer honoured. None means never expires."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, now_ms: int | None = None) -> bool:
        if self.expires_at is None:
            return False
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return now_ms >= self.expires_at
