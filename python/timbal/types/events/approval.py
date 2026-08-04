from typing import Any

from .base import BaseEvent


class ApprovalEvent(BaseEvent):
    """Event emitted when runnable execution is waiting on human approval."""

    __slots__ = (
        "t0",
        "approval_id",
        "runnable_path",
        "runnable_name",
        "runnable_type",
        "tool_call_id",
        "input",
        "input_schema",
        "prompt",
        "description",
        "kind",
        "ui",
        "metadata",
    )

    type = "APPROVAL"

    _FIELDS = BaseEvent._FIELDS + (
        "t0",
        "approval_id",
        "runnable_path",
        "runnable_name",
        "runnable_type",
        "tool_call_id",
        "input",
        "input_schema",
        "prompt",
        "description",
        "kind",
        "ui",
        "metadata",
    )

    def __init__(
        self,
        *,
        run_id: str,
        path: str,
        call_id: str,
        t0: int,
        approval_id: str,
        runnable_path: str,
        runnable_name: str,
        runnable_type: str,
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        tool_call_id: str | None = None,
        input: Any = None,
        input_schema: dict[str, Any] | None = None,
        prompt: str | None = None,
        description: str | None = None,
        kind: str | None = None,
        ui: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
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
        """Unix-ms timestamp at which approval was requested. Useful for SLA timers."""
        self.approval_id = approval_id
        """Stable identifier used to approve or deny this runnable invocation."""
        self.runnable_path = runnable_path
        """Full runnable path that requires approval."""
        self.runnable_name = runnable_name
        """Runnable name that requires approval."""
        self.runnable_type = runnable_type
        """Runnable class/type that requires approval."""
        self.tool_call_id = tool_call_id
        """The LLM tool_call id that triggered this gate, when the approval happened
        inside an agent tool. Lets the frontend correlate the approval card with the
        exact tool_use block in the chat transcript. ``None`` for direct calls."""
        self.input = input
        """Validated (redacted, if configured) input that would be passed to the runnable.
        The *values* for a structured approval card. Pair with ``input_schema`` to render
        a typed form with zero per-tool frontend code."""
        self.input_schema = input_schema
        """JSON Schema of the runnable's parameters (titles/descriptions/types). Render
        ``input`` against this for a generic, typed approval form — Tier 0, no custom UI."""
        self.prompt = prompt
        """Optional human-readable summary. Text fallback for CLIs/logs/non-rich clients."""
        self.description = description
        """Optional runnable or policy description."""
        self.kind = kind
        """Renderer discriminator for a rich approval card (mirrors ``InteractionEvent.kind``).
        The frontend dispatches ``(kind, ui)`` exactly like it does ``(kind, payload)`` for
        interactions. ``None`` means render generically from ``input`` + ``input_schema``."""
        self.ui = ui
        """Structured, presentation-only JSON for the card (title, fields, severity, ...).
        Authored via the tool's ``approval_ui``. Already redacted; safe to render verbatim."""
        self.metadata = metadata if metadata is not None else {}
        """Additional policy metadata for future approval engines."""
