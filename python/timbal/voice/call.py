"""What a phone call tells the agent: who is on the line, why we called, and how to report back.

``call_context()`` is the one accessor for every transport:

* platform media: each ``/stream`` turn carries ``context.call_context``;
* runtimes that join LiveKit themselves: the SIP participant's attributes,
  merged with the platform's ``timbal.call_context`` attribute;
* Twilio / Telnyx: the start frame's identity parameters.

The map holds the caller's business ``context`` (values kept as JSON) plus
``call_id``, ``direction``, ``from`` / ``to`` (named by call direction, as in
SIP) and ``user_number`` / ``agent_number`` (the person on the line and our
line, the same in both directions). Prompts should use ``user_number`` and
``agent_number`` so they never branch on direction.

When the call was started with an outcome schema, the agent gets a
``report_call_outcome`` tool whose parameters are that schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..state import get_run_context

logger = structlog.get_logger("timbal.voice.call")

REPORT_CALL_OUTCOME = "report_call_outcome"
CALL_TOKEN_HEADER = "X-Timbal-Sip-Call-Token"

REPORT_CALL_OUTCOME_DESCRIPTION = (
    "Report the outcome of this phone call in the required structure. "
    "Call it as soon as the outcome is clear and BEFORE saying goodbye, hanging up or transferring: "
    "reports only count until shortly after the line closes, and without one the result is 'unavailable'. "
    "If the report is rejected, correct the listed fields and call it again. "
    "A later report replaces an earlier one."
)


def call_context() -> dict[str, Any]:
    """The current call's context, or ``{}`` outside a phone call.

    Safe to call from a callable ``system_prompt``, a tool, or a hook.
    """
    ctx = get_run_context()
    if ctx is not None:
        extra = getattr(ctx, "call_context", None)
        if isinstance(extra, dict):
            return dict(extra)
    session = _current_voice_session()
    if session is not None:
        stored = getattr(session, "call_context", None)
        if isinstance(stored, dict) and stored:
            return dict(stored)
    return {}


def _current_voice_session() -> Any:
    from .session import current_voice_session

    return current_voice_session()


@dataclass
class CallControl:
    """Call-scoped credential for reporting a call's outcome. Never persist or log it."""

    url: str
    token: str = field(repr=False)
    outcome_schema: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> CallControl | None:
        """Build from the ``context.call_control`` object the platform sends on ``/stream``."""
        if not isinstance(payload, dict):
            return None
        url = str(payload.get("url") or "").strip()
        token = str(payload.get("token") or "").strip()
        if not url or not token:
            return None
        schema = payload.get("outcome_schema")
        if isinstance(schema, str):
            schema = _json_object(schema)
        return cls(url=url, token=token, outcome_schema=schema if isinstance(schema, dict) else None)

    def report_tool(self) -> Any:
        """A fresh ``report_call_outcome`` tool for this call, or ``None`` when no outcome was requested."""
        if not self.outcome_schema:
            return None
        return report_call_outcome_tool(self)


def current_call_control(agent: Any = None) -> CallControl | None:
    """The running call's control handle: from the ``/stream`` turn, or from the voice session.

    When ``agent`` is given, sub-agents (``"." in agent._path``) and internal
    session helpers (``voice_greeting``, ``voice_filler``, ``voice_user_idle``)
    are excluded so only the root call agent gets ``report_call_outcome``.
    """
    if agent is not None and "." in getattr(agent, "_path", ""):
        return None
    session = _current_voice_session()
    if agent is not None and session is not None and getattr(session, "agent", agent) is not agent:
        return None
    ctx = get_run_context()
    control = getattr(ctx, "_call_control", None) if ctx is not None else None
    if isinstance(control, CallControl):
        return control
    control = getattr(session, "call_control", None) if session is not None else None
    return control if isinstance(control, CallControl) else None


def _json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        logger.warning("voice_call_attribute_not_json")
        return None
    return value if isinstance(value, dict) else None


def _endpoint(url: str) -> str:
    return f"{url.rstrip('/')}/result"


def _outcome_client() -> Any:
    import httpx

    return httpx.AsyncClient(timeout=httpx.Timeout(10.0))


async def _post_outcome(control: CallControl, outcome: dict[str, Any]) -> str:
    import httpx

    try:
        async with _outcome_client() as client:
            response = await client.post(
                _endpoint(control.url),
                headers={CALL_TOKEN_HEADER: control.token},
                json={"outcome": outcome},
            )
    except httpx.HTTPError as exc:
        logger.warning("voice_call_outcome_unreachable", error=type(exc).__name__)
        return "The outcome could not be sent right now. Try report_call_outcome once more."
    status = response.status_code
    if 200 <= status < 300:
        return "Outcome recorded."
    detail = _error_detail(response)
    if status == 422:
        return f"The outcome was rejected. Fix these fields and call report_call_outcome again: {detail}"
    if status == 409:
        return "The call result is already final. Do not report again."
    if status in (401, 403):
        logger.warning("voice_call_outcome_forbidden", status=status)
        return "Reporting is not available for this call. Do not report again; continue the conversation."
    logger.warning("voice_call_outcome_failed", status=status)
    return f"The outcome could not be recorded ({status}). Try report_call_outcome once more."


def _error_detail(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list):
                return "; ".join(str(item) for item in value)
    return json.dumps(body)[:500]


def report_call_outcome_tool(control: CallControl) -> Any:
    """A tool whose parameters are the call's outcome schema; it posts the report to the platform."""
    from pydantic import ConfigDict, create_model

    from ..core.tool import Tool

    async def report_call_outcome(**outcome: Any) -> str:
        return await _post_outcome(control, outcome)

    tool = Tool(
        name=REPORT_CALL_OUTCOME,
        description=REPORT_CALL_OUTCOME_DESCRIPTION,
        handler=report_call_outcome,
    )
    # The schema comes from the call, not from the handler's signature.
    tool.__dict__["params_model"] = create_model("ReportCallOutcomeParams", __config__=ConfigDict(extra="allow"))
    schema = dict(control.outcome_schema or {})
    schema.setdefault("properties", {})
    tool.__dict__["params_model_schema"] = schema
    return tool


def run_context_from_wire(raw: Any) -> Any:
    """Build a RunContext from a request's ``context``, keeping ``call_control`` off its fields.

    ``call_control`` is a short-lived credential: it lives on a private attribute,
    so it never reaches the trace, the session bag or logs.
    """
    from ..state import RunContext

    if isinstance(raw, RunContext):
        return raw
    if not isinstance(raw, dict):
        return RunContext.model_validate(raw)
    data = dict(raw)
    control = CallControl.from_payload(data.pop("call_control", None))
    run_context = RunContext.model_validate(data)
    run_context._call_control = control
    return run_context
