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
    tool_call_id: str | None = None
    """The LLM tool_call id that triggered this gate, when the approval happened
    inside an agent tool. Lets the frontend correlate the approval card with the
    exact tool_use block in the chat transcript. ``None`` for direct calls."""
    input: Any
    """Validated (redacted, if configured) input that would be passed to the runnable.
    The *values* for a structured approval card. Pair with ``input_schema`` to render
    a typed form with zero per-tool frontend code."""
    input_schema: dict[str, Any] | None = None
    """JSON Schema of the runnable's parameters (titles/descriptions/types). Render
    ``input`` against this for a generic, typed approval form — Tier 0, no custom UI."""
    prompt: str | None = None
    """Optional human-readable summary. Text fallback for CLIs/logs/non-rich clients."""
    description: str | None = None
    """Optional runnable or policy description."""
    kind: str | None = None
    """Renderer discriminator for a rich approval card (mirrors ``InteractionEvent.kind``).
    The frontend dispatches ``(kind, ui)`` exactly like it does ``(kind, payload)`` for
    interactions. ``None`` means render generically from ``input`` + ``input_schema``."""
    ui: dict[str, Any] | None = None
    """Structured, presentation-only JSON for the card (title, fields, severity, ...).
    Authored via the tool's ``approval_ui``. Already redacted; safe to render verbatim."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Additional policy metadata for future approval engines."""
