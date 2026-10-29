"""Outbound voice calls as tools: start a call, then wait for its result.

Both work as workflow steps and as agent tools::

    from timbal import Workflow
    from timbal.tools import StartVoiceCall, WaitForVoiceCall

    reminder = (
        Workflow(name="appointment_reminder")
        .step(load_appointment)
        .step(
            StartVoiceCall(workforce="receptionist", outcome_schema=OUTCOME),
            to=lambda: get_run_context().step_span("load_appointment").output["phone"],
            context=lambda: get_run_context().step_span("load_appointment").output["context"],
            idempotency_key=lambda: f"appt:{get_run_context().step_span('load_appointment').output['id']}:reminder",
        )
        .step(WaitForVoiceCall(), call=lambda: get_run_context().step_span("start_voice_call").output)
        .step(apply_outcome, result=lambda: get_run_context().step_span("wait_for_voice_call").output)
    )

See :mod:`timbal.platform.voice_calls` for the contract and the plain
``start_voice_call`` / ``wait_for_voice_call`` coroutines.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..core.tool import Tool
from ..platform.voice_calls import start_voice_call, wait_for_voice_call


class StartVoiceCall(Tool):
    """Place an outbound call through a voice workforce and return its handle at once."""

    name: str = "start_voice_call"
    description: str | None = (
        "Phone a person through the voice agent. Pass the destination number in E.164 format and the "
        "business context the voice agent needs (why we call, customer and appointment details). "
        "Returns the call handle immediately; it does not wait for the conversation."
    )
    workforce: str | None = None
    outcome_schema: dict[str, Any] | None = None
    from_number: str | None = None
    trunk_id: int | None = None
    rev: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        async def _start_voice_call(
            to: str = Field(..., description="Number to dial, E.164 (e.g. +34600000000)"),
            context: dict[str, Any] | None = Field(
                None, description="Business context for the voice agent: identifier-like keys, JSON values"
            ),
            greeting: str | None = Field(None, description="Opener spoken when the callee picks up"),
            workforce: str | None = Field(None, description="Voice workforce to call through"),
            outcome_schema: dict[str, Any] | None = Field(
                None, description="JSON Schema of the outcome the voice agent must report"
            ),
            idempotency_key: str | None = Field(
                None, description="Business key: the same key never dials twice (e.g. appt:123:reminder)"
            ),
            attempt: int = Field(0, description="Bump to redial on purpose with the default key"),
        ) -> dict[str, Any]:
            target = workforce or self.workforce
            if not target:
                raise ValueError("workforce is required: configure it on the tool or pass it.")
            call = await start_voice_call(
                target,
                to,
                context=context,
                outcome=outcome_schema or self.outcome_schema,
                greeting=greeting,
                from_number=self.from_number,
                trunk_id=self.trunk_id,
                idempotency_key=idempotency_key,
                attempt=attempt,
                rev=self.rev,
            )
            return call.model_dump(mode="json")

        super().__init__(handler=_start_voice_call, **kwargs)


class WaitForVoiceCall(Tool):
    """Suspend until the platform delivers the call's final result."""

    name: str = "wait_for_voice_call"
    description: str | None = (
        "Wait for a phone call started with start_voice_call to finish and return its result: "
        "telephony status, and outcome_status / outcome as reported by the voice agent."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _wait_for_voice_call(
            call: dict[str, Any] = Field(..., description="The handle returned by start_voice_call"),
        ) -> dict[str, Any]:
            result = await wait_for_voice_call(call)
            return result.model_dump(mode="json")

        super().__init__(handler=_wait_for_voice_call, **kwargs)
