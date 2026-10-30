"""Outbound voice calls placed through the Timbal platform.

A workflow (or an agent) asks the platform to dial someone through a voice
workforce, hands it the business context the agent needs before its first
turn, and later receives the call's result::

    from timbal.platform.voice_calls import start_voice_call, wait_for_voice_call

    call = await start_voice_call(
        workforce="receptionist",
        to=customer.phone,
        context={"customer_name": customer.name, "appointment": {"at": appt.at}},
        outcome=AppointmentOutcome,
        idempotency_key=f"appt:{appt.id}:reminder",
    )
    result = await wait_for_voice_call(call)
    if result.outcome_status == "reported" and result.outcome.outcome == "confirmed":
        ...

``wait_for_voice_call`` suspends the run with ``suspend(..., kind="voice_call")``
until the platform resumes it with the final call view, so no worker or HTTP
request is held open while the phone rings.

Carrier credentials, dial policy, authorization and billing stay on the
platform. This module only speaks the platform contract:

* ``POST /orgs/{org}/projects/{project}/workforce/{workforce}/voice/calls``
* ``GET  /orgs/{org}/projects/{project}/workforce/{workforce}/voice/calls/{id}``
* ``POST /orgs/{org}/projects/{project}/workforce/{workforce}/voice/calls/{id}/cancel``
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from ..errors import PlatformError
from ..state import _suspension_id_for, get_or_create_run_context, get_run_context, suspend
from .utils import _request

logger = structlog.get_logger("timbal.platform.voice_calls")

VOICE_CALL_KIND = "voice_call"
"""``suspend()`` kind for a wait on a call. The platform resumes these, not a human."""

RESERVED_CONTEXT_KEYS = frozenset({"call_id", "direction", "from", "to", "user_number", "agent_number"})
"""Keys the platform writes into the agent's call context. A caller may not send them."""

MAX_CONTEXT_KEYS = 64
MAX_JSON_BYTES = 16 * 1024

_CONTEXT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SPAN_METADATA_KEY = "voice_calls"
_SPAN_RESULTS_METADATA_KEY = "voice_call_results"
_ROOT_METADATA_KEY = "voice_call_root_id"


class VoiceCallError(PlatformError):
    """A call could not be started, read or cancelled."""


class VoiceCallsUnavailable(VoiceCallError):
    """The platform cannot place outbound calls for this project (disabled, or too old)."""


class IdempotencyKeyReused(VoiceCallError):
    """The idempotency key was already used for a different dial."""


class VoiceCall(BaseModel):
    """Handle to a call the platform accepted. Pass it to :func:`wait_for_voice_call`."""

    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    workforce: str
    org_id: str
    project_id: str
    rev: str | None = None
    idempotency_key: str
    replayed: bool = False
    """True when the platform (or the parent run) already had this call: nothing was dialed."""

    _outcome_model: type[BaseModel] | None = PrivateAttr(default=None)


class VoiceCallResult(BaseModel):
    """Final state of a call: telephony status and business outcome, kept apart.

    A completed call is not a confirmed outcome. Branch on ``outcome_status``
    first: ``"reported"`` means the agent reported an outcome, ``"unavailable"``
    means none arrived (no answer, hangup before reporting, or the 15 s
    reporting window after hangup closed).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    direction: str | None = None
    end_reason: str | None = None
    failure_reason: str | None = None
    sip_status_code: int | None = None
    answered_at: datetime | None = None
    ended_at: datetime | None = None
    outcome_status: Literal["pending", "reported", "unavailable"] = "pending"
    outcome: Any = None
    outcome_reported_at: datetime | None = None
    finalized_at: datetime | None = None
    session_id: str | None = None
    origin: dict[str, Any] | None = None

    @property
    def is_final(self) -> bool:
        return self.finalized_at is not None


def _subject() -> Any:
    ctx = get_or_create_run_context()
    return ctx.platform_config.subject if ctx.platform_config else None


def _resolve_scope(org_id: str | None, project_id: str | None, rev: str | None) -> tuple[str, str, str | None]:
    """Explicit argument, then the run's platform subject, then ``TIMBAL_*`` env vars."""
    subject = _subject()
    org = org_id or (subject.org_id if subject else None) or os.environ.get("TIMBAL_ORG_ID")
    project = project_id or (subject.project_id if subject else None) or os.environ.get("TIMBAL_PROJECT_ID")
    revision = rev or (subject.rev if subject else None) or os.environ.get("TIMBAL_PROJECT_REV") or None
    if not org:
        raise ValueError("org_id is required. Pass org_id=, run on the platform, or set TIMBAL_ORG_ID.")
    if not project:
        raise ValueError("project_id is required. Pass project_id=, run on the platform, or set TIMBAL_PROJECT_ID.")
    return str(org), str(project), revision


def _calls_path(org: str, project: str, workforce: str) -> str:
    return f"orgs/{org}/projects/{project}/workforce/{workforce}/voice/calls"


def _json_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), default=str).encode())


def normalize_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate the per-call business context the way the platform will, and drop nulls."""
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object.")
    cleaned = {key: value for key, value in context.items() if value is not None}
    for key in cleaned:
        if not isinstance(key, str) or not _CONTEXT_KEY.match(key):
            raise ValueError(f"context key {key!r} must be identifier-like (letters, digits, underscore).")
        if key in RESERVED_CONTEXT_KEYS:
            raise ValueError(f"context key {key!r} is reserved: the platform sets it on every call.")
    if len(cleaned) > MAX_CONTEXT_KEYS:
        raise ValueError(f"context has {len(cleaned)} keys; at most {MAX_CONTEXT_KEYS} are allowed.")
    if _json_size(cleaned) > MAX_JSON_BYTES:
        raise ValueError(f"context is larger than {MAX_JSON_BYTES} bytes once serialized.")
    return json.loads(json.dumps(cleaned, default=str))


def _external_refs(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            found.append(ref)
        for value in node.values():
            found.extend(_external_refs(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_external_refs(value))
    return found


def outcome_schema_for(outcome: type[BaseModel] | dict[str, Any] | None) -> dict[str, Any] | None:
    """JSON Schema for ``outcome=``: a Pydantic model or a schema dict with ``"type": "object"``."""
    if outcome is None:
        return None
    if isinstance(outcome, type) and issubclass(outcome, BaseModel):
        schema = outcome.model_json_schema()
    elif isinstance(outcome, dict):
        schema = outcome
    else:
        raise TypeError("outcome must be a Pydantic model class or a JSON Schema dict.")
    if schema.get("type") != "object":
        raise ValueError('outcome schema must have "type": "object".')
    if _json_size(schema) > MAX_JSON_BYTES:
        raise ValueError(f"outcome schema is larger than {MAX_JSON_BYTES} bytes once serialized.")
    external = _external_refs(schema)
    if external:
        raise ValueError(f"outcome schema may only reference itself; found {external[0]!r}.")
    try:
        import jsonschema
    except ImportError:
        return schema
    validator = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
    try:
        validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"outcome schema is not a valid JSON Schema: {exc.message}") from exc
    return schema


async def _root_run_id() -> str:
    """The id of the run this chain of resumes started from.

    A resumed run re-executes its handlers with a new run id, so an idempotency
    key derived from ``run_context.id`` would dial again on every resume. The
    root id is the first run's id, inherited through every resume and stamped
    on the root span so the next resume in a chain finds it too.
    """
    ctx = get_or_create_run_context()
    cached = getattr(ctx, "_voice_call_root_id", None)
    if cached:
        return cached
    root_id = ctx.id
    if ctx.parent_id and ctx._resume_values:
        parent = await ctx._get_parent_trace()
        inherited = None
        if parent is not None and parent._root_call_id is not None:
            parent_root = parent.get(parent._root_call_id)
            inherited = (getattr(parent_root, "metadata", None) or {}).get(_ROOT_METADATA_KEY)
        root_id = inherited or ctx.parent_id
    ctx._voice_call_root_id = root_id
    root = ctx.root_span()
    if root is not None:
        root.metadata[_ROOT_METADATA_KEY] = root_id
    return root_id


def _current_span() -> Any:
    ctx = get_run_context()
    if ctx is None:
        return None
    try:
        return ctx.current_span()
    except RuntimeError:
        return None


async def default_idempotency_key(attempt: int = 0) -> str:
    """``sha256(root run id, span path, tool call id or occurrence, attempt)``.

    Stable across resume and retries of the same handler, distinct for each
    call site and for each call the same site makes in one run.
    """
    ctx = get_or_create_run_context()
    span = _current_span()
    path = span.path if span is not None else ""
    tool_call_id = (span.metadata or {}).get("tool_call_id") if span is not None else None
    counters: dict[str, int] = getattr(ctx, "_voice_call_occurrences", None) or {}
    ctx._voice_call_occurrences = counters
    site = f"{path}\0{tool_call_id or ''}"
    occurrence = counters.get(site, 0)
    counters[site] = occurrence + 1
    root_id = await _root_run_id()
    raw = "\0".join([root_id, path, tool_call_id or "", str(occurrence), str(attempt)])
    return hashlib.sha256(raw.encode()).hexdigest()


def _dial_fingerprint(
    workforce: str,
    rev: str | None,
    to: str,
    from_number: str | None,
    trunk_id: int | str | None,
    greeting: str | None,
) -> list[Any]:
    return [
        workforce,
        rev,
        to,
        from_number or None,
        int(trunk_id) if trunk_id is not None else None,
        greeting or None,
    ]


async def _replayed_from_parent(key: str, fingerprint: list[Any]) -> dict[str, Any] | None:
    """The call a previous run in this resume chain already started with ``key``."""
    ctx = get_or_create_run_context()
    if not (ctx.parent_id and ctx._resume_values):
        return None
    parent = await ctx._get_parent_trace()
    if parent is None:
        return None
    for span in parent.values():
        records = (getattr(span, "metadata", None) or {}).get(_SPAN_METADATA_KEY)
        record = records.get(key) if isinstance(records, dict) else None
        if isinstance(record, dict) and record.get("id"):
            stored_fp = record.get("fingerprint")
            if isinstance(stored_fp, list) and stored_fp != fingerprint:
                raise IdempotencyKeyReused(
                    "Starting the voice call failed: the idempotency key was already used for a different dial "
                    "(workforce, rev, to, from, trunk or greeting). Use a new key or bump attempt=.",
                    status_code=422,
                )
            return {k: v for k, v in record.items() if k != "fingerprint"}
    return None


def _record_on_span(call: VoiceCall, fingerprint: list[Any]) -> None:
    span = _current_span()
    if span is None:
        return
    records = span.metadata.setdefault(_SPAN_METADATA_KEY, {})
    records[call.idempotency_key] = {
        "id": call.id,
        "status": call.status,
        "workforce": call.workforce,
        "org_id": call.org_id,
        "project_id": call.project_id,
        "rev": call.rev,
        "idempotency_key": call.idempotency_key,
        "fingerprint": fingerprint,
    }


def _platform_error(exc: PlatformError, action: str) -> VoiceCallError:
    status = getattr(exc, "status_code", None)
    text = str(exc)
    if status == 422 and "IDEMPOTENCY_KEY_REUSED" in text:
        return IdempotencyKeyReused(
            f"{action}: the idempotency key was already used for a different dial "
            "(workforce, rev, to, from, trunk or greeting). Use a new key or bump attempt=.\n" + text,
            status_code=status,
        )
    if status in (404, 501) or (status == 409 and "not enabled" in text.lower()):
        return VoiceCallsUnavailable(
            f"{action}: outbound voice calls are not available here.\n{text}", status_code=status
        )
    return VoiceCallError(f"{action}:\n{text}", status_code=status)


async def start_voice_call(
    workforce: str,
    to: str,
    *,
    context: dict[str, Any] | None = None,
    outcome: type[BaseModel] | dict[str, Any] | None = None,
    greeting: str | None = None,
    from_number: str | None = None,
    trunk_id: int | str | None = None,
    idempotency_key: str | None = None,
    attempt: int = 0,
    rev: str | None = None,
    org_id: str | None = None,
    project_id: str | None = None,
) -> VoiceCall:
    """Ask the platform to dial ``to`` through the voice workforce ``workforce``.

    Returns as soon as the platform accepts the call. ``context`` reaches the
    agent before its first turn (read it with :func:`timbal.voice.call_context`);
    ``greeting`` only replaces the spoken opener. ``outcome`` (a Pydantic model or
    a JSON Schema) gives the agent a ``report_call_outcome`` tool and types the
    result.

    Idempotent: the same ``idempotency_key`` never dials twice. The default key
    is stable across retries and resume of the same handler. Pass a business key
    (``f"appt:{appt.id}:reminder"``) when duplicate triggers start separate runs,
    and a new key or ``attempt=`` to redial on purpose.
    """
    if not workforce:
        raise ValueError("workforce is required.")
    if not to:
        raise ValueError("to is required.")
    org, project, revision = _resolve_scope(org_id, project_id, rev)
    body_context = normalize_context(context)
    schema = outcome_schema_for(outcome)
    outcome_model = outcome if isinstance(outcome, type) and issubclass(outcome, BaseModel) else None
    key = idempotency_key or await default_idempotency_key(attempt)
    fingerprint = _dial_fingerprint(workforce, revision, to, from_number, trunk_id, greeting)

    prior = await _replayed_from_parent(key, fingerprint)
    if prior is not None:
        call = VoiceCall(**{**prior, "replayed": True})
        call._outcome_model = outcome_model
        _record_on_span(call, fingerprint)
        return call

    body: dict[str, Any] = {"to": to}
    if from_number:
        body["from_number"] = from_number
    if trunk_id is not None:
        body["trunk_id"] = int(trunk_id)
    if greeting:
        body["greeting"] = greeting
    if body_context:
        body["context"] = body_context
    if schema is not None:
        body["outcome_schema"] = schema
    ctx = get_or_create_run_context()
    subject = _subject()
    app_id = (subject.app_id if subject else None) or os.environ.get("TIMBAL_APP_ID")
    origin = {"run_id": ctx.id, **({"app_id": app_id} if app_id else {})}
    body["origin"] = origin

    params = {"rev": revision} if revision else None
    try:
        res = await _request(
            "POST",
            _calls_path(org, project, workforce),
            headers={"Idempotency-Key": key},
            params=params,
            json=body,
        )
    except PlatformError as exc:
        raise _platform_error(exc, "Starting the voice call failed") from exc
    view = res.json()
    if not isinstance(view, dict) or not view.get("id"):
        raise VoiceCallError("Starting the voice call failed: the platform returned no call id.")
    if "outcome_status" not in view:
        raise VoiceCallsUnavailable(
            "The platform accepted the call but predates call results (no outcome_status), "
            "so a wait would never resume. Upgrade the platform before using wait_for_voice_call."
        )
    replayed = str(res.headers.get("idempotent-replayed", "")).lower() == "true"
    call = VoiceCall(
        id=str(view["id"]),
        status=str(view.get("status") or "queued"),
        workforce=workforce,
        org_id=org,
        project_id=project,
        rev=revision,
        idempotency_key=key,
        replayed=replayed,
    )
    call._outcome_model = outcome_model
    _record_on_span(call, fingerprint)
    logger.info("voice_call_started", call_id=call.id, workforce=workforce, replayed=replayed)
    return call


def _handle(call: VoiceCall | dict[str, Any]) -> VoiceCall:
    if isinstance(call, VoiceCall):
        return call
    if isinstance(call, dict):
        return VoiceCall.model_validate(call)
    raise TypeError("call must be the VoiceCall returned by start_voice_call (or its dict form).")


def _result(view: Any, call: VoiceCall, outcome_model: type[BaseModel] | None) -> VoiceCallResult:
    if not isinstance(view, dict):
        raise VoiceCallError(f"The call view for {call.id} is not an object.")
    if str(view.get("id")) != call.id:
        raise VoiceCallError(f"Received the result of call {view.get('id')!r} while waiting for {call.id!r}.")
    result = VoiceCallResult.model_validate(view)
    if outcome_model is not None and result.outcome_status == "reported":
        try:
            result.outcome = outcome_model.model_validate(result.outcome)
        except ValidationError as exc:
            raise VoiceCallError(
                f"Call {call.id} reported an outcome that does not match {outcome_model.__name__}: {exc}"
            ) from exc
    return result


async def get_voice_call(
    call: VoiceCall | dict[str, Any],
    *,
    outcome: type[BaseModel] | None = None,
) -> VoiceCallResult:
    """Read where a call stands now. Does not wait."""
    handle = _handle(call)
    params = {"rev": handle.rev} if handle.rev else None
    try:
        res = await _request(
            "GET",
            f"{_calls_path(handle.org_id, handle.project_id, handle.workforce)}/{handle.id}",
            params=params,
        )
    except PlatformError as exc:
        raise _platform_error(exc, f"Reading voice call {handle.id} failed") from exc
    return _result(res.json(), handle, outcome or handle._outcome_model)


async def cancel_voice_call(call: VoiceCall | dict[str, Any]) -> VoiceCallResult:
    """Cancel a call that has not ended. Does not schedule another one."""
    handle = _handle(call)
    params = {"rev": handle.rev} if handle.rev else None
    try:
        res = await _request(
            "POST",
            f"{_calls_path(handle.org_id, handle.project_id, handle.workforce)}/{handle.id}/cancel",
            params=params,
            json={},
        )
    except PlatformError as exc:
        raise _platform_error(exc, f"Cancelling voice call {handle.id} failed") from exc
    body = res.json() if res.content else {"id": handle.id, "status": "cancelled"}
    return _result(body, handle, handle._outcome_model)


async def _replayed_result_from_parent(call_id: str) -> dict[str, Any] | None:
    """The final result a previous run in this resume chain already received for ``call_id``."""
    ctx = get_or_create_run_context()
    if not (ctx.parent_id and ctx._resume_values):
        return None
    parent = await ctx._get_parent_trace()
    if parent is None:
        return None
    for span in parent.values():
        results = (getattr(span, "metadata", None) or {}).get(_SPAN_RESULTS_METADATA_KEY)
        view = results.get(call_id) if isinstance(results, dict) else None
        if isinstance(view, dict) and view.get("finalized_at") is not None:
            return view
    return None


def _record_result_on_span(call_id: str, result: VoiceCallResult) -> None:
    span = _current_span()
    if span is None:
        return
    results = span.metadata.setdefault(_SPAN_RESULTS_METADATA_KEY, {})
    results[call_id] = result.model_dump(mode="json")


async def wait_for_voice_call(
    call: VoiceCall | dict[str, Any],
    *,
    outcome: type[BaseModel] | None = None,
) -> VoiceCallResult:
    """Wait for the call's final result without holding a worker.

    Returns at once when the call is already final. Otherwise the run suspends
    (status ``cancelled`` / ``input_required``, kind ``"voice_call"``) and the
    platform resumes it with the final call view, which is validated here: a
    result for another call, or one that is not final, is rejected.

    The parked run is found through its trace, so tracing must reach the
    platform. A run with tracing off is never resumed.
    """
    handle = _handle(call)
    model = outcome or handle._outcome_model
    payload = {"call_id": handle.id}
    ctx = get_or_create_run_context()
    span = _current_span()
    if span is not None:
        suspension_id = _suspension_id_for(span.path, payload, (span.metadata or {}).get("tool_call_id"))
        if suspension_id in ctx._resume_values:
            result = _final(suspend(payload, kind=VOICE_CALL_KIND), handle, model)
            await _claim_delivery(ctx, suspension_id, handle.id)
            _record_result_on_span(handle.id, result)
            return result
    prior_view = await _replayed_result_from_parent(handle.id)
    if prior_view is not None:
        result = _final(prior_view, handle, model)
        _record_result_on_span(handle.id, result)
        return result
    current = await get_voice_call(handle, outcome=model)
    if current.is_final:
        _record_result_on_span(handle.id, current)
        return current
    if ctx._tracing_provider is None:
        logger.warning(
            "voice_call_wait_untraced",
            call_id=handle.id,
            hint="Tracing is off, so the platform cannot find this wait and will not resume it.",
        )
    return _final(suspend(payload, kind=VOICE_CALL_KIND), handle, model)


async def _claim_delivery(ctx: Any, suspension_id: str, call_id: str) -> None:
    """Let one resume consume a call's result. A duplicate delivery cancels its run.

    Uses the tracing provider's atomic ``(parent_id, id)`` claim, the same one
    that stops an approval from being executed twice.
    """
    provider = ctx._tracing_provider
    if provider is None or not ctx.parent_id:
        return
    claimed = await provider.claim_approval(str(ctx.parent_id), suspension_id, str(ctx.id))
    if not claimed:
        from ..errors import RunCancelled

        logger.info("voice_call_duplicate_delivery", call_id=call_id, parent_id=ctx.parent_id)
        raise RunCancelled(f"The result of call {call_id} was already delivered to another resume of this run.")


def _final(view: Any, call: VoiceCall, outcome_model: type[BaseModel] | None) -> VoiceCallResult:
    result = _result(view, call, outcome_model)
    if not result.is_final:
        raise VoiceCallError(f"Resumed with call {call.id} before its result was final.")
    return result
