from typing import Any

from .base import BaseEvent


class GuardrailEvent(BaseEvent):
    """Emitted when a guardrail triggers (including shadowed and errored rails).

    First-class in the stream so UIs can react the moment a rail fires — render a
    "response withheld by moderation" notice, badge a redaction, or log a shadow-mode
    verdict — without waiting for the final OutputEvent.
    """

    __slots__ = ("rail", "stage", "action", "reason", "latency_ms", "shadow", "metadata")

    type = "GUARDRAIL"

    _FIELDS = BaseEvent._FIELDS + ("rail", "stage", "action", "reason", "latency_ms", "shadow", "metadata")

    def __init__(
        self,
        *,
        run_id: str,
        path: str,
        call_id: str,
        rail: str,
        stage: str,
        action: str,
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        reason: str | None = None,
        latency_ms: int = 0,
        shadow: bool = False,
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
        self.rail = rail
        """Name of the guardrail that triggered."""
        self.stage = stage
        """Stage the rail fired on: input, model_output, tool_args, tool_result."""
        self.action = action
        """Verdict action: block, replace, retry, escalate, warn — or error if the rail crashed."""
        self.reason = reason
        """Dev-facing explanation (never end-user copy)."""
        self.latency_ms = latency_ms
        """Time the check took."""
        self.shadow = shadow
        """True when the verdict was recorded but not enforced."""
        self.metadata = metadata if metadata is not None else {}
        """Rail-specific extras (e.g. matched categories, scores)."""
