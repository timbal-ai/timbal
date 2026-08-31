"""Voice over LiveKit — the agent dials out to a room the monolith already created.

Sibling of :mod:`timbal.server.rtc`. ``rtc.py`` is untouched — it is the
rollback rung. Two ways in, one driver:

* **Boot env** (serverless): the platform spawns one server per call, so the
  lifespan starts the driver when ``TIMBAL_VOICE_TRANSPORT=livekit`` and the
  dial rides the process env. See :func:`maybe_start_livekit_session`.
* **Per-request** (ECS / on-premise / any long-lived server): the dial rides
  ``POST /voice/rtc`` as ``{"transport": "livekit", "url", "token", …}``,
  the same route the SDP offer uses, and the process serves room after room
  without exiting. See :func:`start_livekit_session`.

Env contract for the boot-env path (all platform-owned):

* ``TIMBAL_VOICE_TRANSPORT=livekit`` selects this driver; unset/``webrtc``
  keeps ``rtc.py``. **Never set it on a long-lived deployment** — joining at
  boot would pin one stale token for the process's whole life.
* ``TIMBAL_LIVEKIT_URL`` — ``ws://<sfu-private-ip>:7880``
* ``TIMBAL_LIVEKIT_TOKEN`` — agent join token (the room is already pinned)
* ``TIMBAL_LIVEKIT_ROOM`` — informational; the token already pins it
* ``TIMBAL_LIVEKIT_CALLER_IDENTITY`` — logging hint only; the human is the
  first eligible remote participant (STANDARD/SIP), not a prefix match
* ``TIMBAL_LIVEKIT_AGENT_IDENTITY`` — this box's identity (exclude from caller resolution)
* ``TIMBAL_VOICE_CALL_ID`` — platform call id for SIP transfer API paths
* ``TIMBAL_VOICE_CLIENT_CONFIG`` — JSON, same keys as the WS hello / rtc config.
  Overlay: after the caller publishes a mic, the driver waits up to 2s for an
  untyped data-message hello (playground dropdowns) and merges it on top.
* ``TIMBAL_VOICE_PARENT_RUN_ID`` — run id this call continues (text → voice).
  Session identity, not a voice knob: it is read here (and off the dial body
  on the per-request path), never off the data-channel hello.
* ``TIMBAL_VOICE_ABANDON_SECS`` — default 45; see ``SingleSessionGuard``

Env contract for the per-request path (both optional, both recommended on
anything long-lived — the dial tells the process where to connect and what to
spend, so an open ``POST /voice/rtc`` is a way to make someone else's box
place calls for you):

* ``TIMBAL_LIVEKIT_URL`` — when set, a dial to any *other* url is 403'd. On
  the boot-env path this is already the dial's url, so pinning it costs
  nothing there.
* ``TIMBAL_VOICE_DIAL_SECRET`` — when set, a dial must present it in
  ``X-Timbal-Dial-Secret``. See :mod:`timbal.server.rtc`.

On the per-request path the process may hold several rooms at once. Session
count is bounded by :mod:`timbal.server.capacity` — and on *this* path the
``auto`` ceiling applies even when nothing is configured, because no
deployment can regress to a cap on a path that did not exist. A full process
answers 503 rather than degrading the calls it already has.

``TIMBAL_VOICE_SINGLE_SESSION=1`` still applies where it is set (serverless).
``claim()`` happens at room-join rather than offer-receipt; ``mark_connected()``
fires on the caller's mic track being subscribed. Session + TTS track are built
only after that subscribe (and the hello window), so playground config actually
applies. Without the guard (the per-request path) nothing exits the process:
one room ends, the next request starts another.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import uuid
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

import structlog

from ..voice.config import VoiceConfig
from .capacity import acquire_session_slot, max_concurrent_sessions, release_session_slot
from .voice import build_voice_session, event_to_payloads, merge_client_voice_overrides

logger = structlog.get_logger("timbal.server.livekit_session")

_MAX_RELIABLE_BYTES = 12 * 1024
_EVENTS_TOPIC = "timbal.events"
_HELLO_WAIT_SECS = 2.0

# How long a per-request join may take before the caller gets a 504. The SFU is
# in the same VPC, so this covers the FFI import on a cold process, not a WAN
# round trip.
_JOIN_TIMEOUT_SECS = 15.0


@dataclass(frozen=True)
class LivekitDial:
    """One room's join parameters, however they arrived.

    The token already pins the room and the grants; ``room`` is carried for
    logging and for keying concurrent sessions on a long-lived server.
    """

    url: str
    token: str
    room: str = ""
    caller_identity: str = ""
    agent_identity: str = ""
    call_id: str = ""
    call_control_url: str = ""
    call_control_token: str = ""
    # JSON string (not a dict) so the env and body paths share one type.
    client_config: str = "{}"
    # Run id this call continues (text → voice). Session identity, not a voice
    # knob, so it rides the dial — minted by whoever authorized the call — and
    # never the data-channel hello, which the browser controls.
    parent_id: str = ""


def dial_from_env() -> LivekitDial:
    """The boot-env dial (serverless: one process per call)."""
    return LivekitDial(
        url=os.environ.get("TIMBAL_LIVEKIT_URL", "").strip(),
        token=os.environ.get("TIMBAL_LIVEKIT_TOKEN", "").strip(),
        room=os.environ.get("TIMBAL_LIVEKIT_ROOM", "").strip(),
        caller_identity=os.environ.get("TIMBAL_LIVEKIT_CALLER_IDENTITY", "").strip(),
        agent_identity=os.environ.get("TIMBAL_LIVEKIT_AGENT_IDENTITY", "").strip(),
        call_id=os.environ.get("TIMBAL_VOICE_CALL_ID", "").strip(),
        call_control_url=os.environ.get("TIMBAL_VOICE_CALL_CONTROL_URL", "").strip(),
        call_control_token=os.environ.get("TIMBAL_VOICE_CALL_CONTROL_TOKEN", "").strip(),
        client_config=os.environ.get("TIMBAL_VOICE_CLIENT_CONFIG") or "{}",
        parent_id=os.environ.get("TIMBAL_VOICE_PARENT_RUN_ID", "").strip(),
    )


def is_livekit_dial(body: Any) -> bool:
    """Body discriminator for ``POST /voice/rtc``. Anything else is an SDP offer."""
    return isinstance(body, dict) and body.get("transport") == "livekit"


def room_from_token(token: str) -> str:
    """The room the token actually pins, from its ``video.room`` grant.

    Parsed, not verified: the SFU is what validates the signature, and a
    forged claim only ever keys a session that the SFU will then refuse. The
    point is that the *token* pins the room while the body's ``room`` field is
    just a label — keying on the label lets one token join the same real room
    twice under two names, putting two agents in it talking over each other.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return ""
    # Real JWT segments are unpadded base64url.
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        # binascii.Error subclasses ValueError, so one except covers both the
        # decode and the parse.
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except ValueError:
        return ""
    if not isinstance(claims, dict):
        return ""
    video = claims.get("video")
    return str(video.get("room") or "") if isinstance(video, dict) else ""


def dial_from_body(body: dict[str, Any]) -> LivekitDial:
    """The per-request dial. ``config`` keys match the WS hello / rtc config."""
    config = body.get("config")
    return LivekitDial(
        url=str(body.get("url") or "").strip(),
        token=str(body.get("token") or "").strip(),
        room=str(body.get("room") or "").strip(),
        caller_identity=str(body.get("caller_identity") or "").strip(),
        agent_identity=str(body.get("agent_identity") or "").strip(),
        call_id=str(body.get("call_id") or "").strip(),
        call_control_url=str(body.get("call_control_url") or "").strip(),
        call_control_token=str(body.get("call_control_token") or "").strip(),
        client_config=json.dumps(config) if isinstance(config, dict) else "{}",
        parent_id=str(body.get("parent_id") or "").strip(),
    )


class _Join:
    """Handshake between a per-request caller and the driver task.

    The request must not answer 200 until the agent is actually in the room —
    that is the whole readiness contract the platform relies on (it replaces
    the SDP answer). ``status`` is the HTTP status the failure should surface.
    """

    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.error: str | None = None
        self.status = 502

    def ok(self) -> None:
        self.done.set()

    def fail(self, reason: str, status: int = 502) -> None:
        if self.done.is_set():
            return
        self.error = reason
        self.status = status
        self.done.set()


def _sessions(app: Any) -> dict[str, asyncio.Task]:
    """Live per-request sessions keyed by room. Absent on the boot-env path."""
    existing = getattr(app.state, "livekit_sessions", None)
    if existing is None:
        existing = {}
        app.state.livekit_sessions = existing
    return existing


def maybe_start_livekit_session(app: Any) -> asyncio.Task | None:
    """Lifespan hook: start the dial-out driver when the transport is LiveKit."""
    if os.environ.get("TIMBAL_VOICE_TRANSPORT", "").strip().lower() != "livekit":
        return None
    return asyncio.create_task(_run_livekit_session(app), name="voice-livekit-session")


async def start_livekit_session(
    app: Any,
    dial: LivekitDial,
    *,
    timeout: float = _JOIN_TIMEOUT_SECS,
) -> tuple[int, dict[str, Any]]:
    """Join one room on behalf of a request. Returns ``(status, body)``.

    Returns only once the agent is in the room (or the join failed), so a 200
    means the caller can connect and expect a participant to already be there.
    The session then outlives this request — it ends when the caller leaves
    (or the platform deletes the room), not when this coroutine returns.
    """
    if not dial.url or not dial.token:
        return 400, {"error": "livekit dial requires 'url' and 'token'"}

    pinned = os.environ.get("TIMBAL_LIVEKIT_URL", "").strip()
    if pinned and dial.url != pinned:
        logger.warning("voice_livekit_url_not_pinned", url=dial.url)
        return 403, {"error": "livekit url is not the one this deployment is pinned to"}

    sessions = _sessions(app)
    # The token's grant, not the body's label — see `room_from_token`.
    key = room_from_token(dial.token) or dial.room or dial.token
    live = sessions.get(key)
    if live is not None and not live.done():
        return 409, {"error": f"a voice session is already live for room {dial.room or key}"}

    # Rejecting here beats degrading the calls already on this process. This is
    # the one path that defaults to a ceiling: nothing predates it, so nothing
    # regresses, and a request-driven join is otherwise unbounded.
    if not acquire_session_slot(default_auto=True):
        return 503, {
            "error": f"server is at its voice session capacity "
            f"({max_concurrent_sessions(default_auto=True)} concurrent)"
        }

    join = _Join()
    task = asyncio.create_task(
        _run_livekit_session(app, dial, join),
        name=f"voice-livekit-session:{dial.room or 'unnamed'}",
    )
    sessions[key] = task

    def _forget(finished: asyncio.Task) -> None:
        # The slot belongs to the *session*, not to this request: it is held
        # until the driver task ends, however it ends (call over, join
        # failed, cancelled by the timeout below).
        release_session_slot()
        # Only clear our own entry: a later join for the same room may have
        # already replaced it.
        if sessions.get(key) is finished:
            sessions.pop(key, None)

    task.add_done_callback(_forget)

    try:
        await asyncio.wait_for(join.done.wait(), timeout=timeout)
    except TimeoutError:
        task.cancel()
        # This session is abandoned, but its teardown still has awaits to get
        # through (sender, room.disconnect, guard.finish), so the task is not
        # done yet. Free the key now or the caller's retry — the obvious
        # response to a 504 — is answered 409 for a room nobody is serving.
        # `_forget` only clears its own entry, so the replacement survives.
        if sessions.get(key) is task:
            sessions.pop(key, None)
        logger.error("voice_livekit_join_timeout", room=dial.room, timeout=timeout)
        return 504, {"error": f"agent did not join room {dial.room} within {timeout:g}s"}

    if join.error is not None:
        return join.status, {"error": join.error}
    return 200, {"transport": "livekit", "room": dial.room, "status": "joined"}


def chunk_data_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split payloads that would exceed LiveKit's ~15 KiB reliable-data cap.

    Two chunkers, because there are two client contracts:

    * ``session_transcript`` keeps its entry-wise split — chunks are
      ``{"type":"session_transcript","seq":i,"total":n,"entries":[...]}`` and
      every client that already reassembles by ``seq`` is unaffected.
    * Anything else is wrapped in a generic
      ``{"type":"chunk","chunk_id","msg_type","seq","total","data"}`` envelope,
      where ``data`` is a base64 slice of the whole encoded payload. A client
      concatenates by ``seq``, base64-decodes and parses one message.

    The generic envelope exists because an ``agent_approval`` carrying a build
    brief in ``ui``/``input`` blows past 12 KiB routinely, and sending it whole
    means LiveKit rejects it — indistinguishable, from the client, from never
    emitting approvals at all. Splitting bytes rather than fields keeps it
    payload-agnostic: it works for any event we add later.
    """
    out: list[dict[str, Any]] = []
    for payload in payloads:
        raw = _dumps(payload)
        if len(raw) <= _MAX_RELIABLE_BYTES:
            out.append(payload)
            continue
        if payload.get("type") == "session_transcript":
            out.extend(_chunk_transcript(payload))
            continue
        out.extend(_chunk_opaque(payload, raw))
    return out


def _chunk_opaque(payload: dict[str, Any], raw: bytes) -> list[dict[str, Any]]:
    """Wrap one oversized payload in ``chunk`` envelopes carrying base64 slices.

    Base64 rather than raw JSON text so a slice boundary can never land inside a
    multi-byte codepoint or need escaping: the encoded form is ASCII, so byte
    length is character length and any cut is valid.
    """
    chunk_id = uuid.uuid4().hex
    msg_type = payload.get("type")
    data = base64.b64encode(raw).decode("ascii")

    def _envelope(seq: int, total: int, body: str) -> dict[str, Any]:
        return {
            "type": "chunk",
            "chunk_id": chunk_id,
            "msg_type": msg_type,
            "seq": seq,
            "total": total,
            "data": body,
        }

    # Size the body against a worst-case header (counters far wider than any
    # real chunk count) so no envelope can overshoot once seq/total are filled in.
    capacity = _MAX_RELIABLE_BYTES - len(_dumps(_envelope(10**9, 10**9, "")))
    if capacity <= 0:
        logger.warning("livekit_payload_unchunkable", msg_type=msg_type, bytes=len(raw))
        return [payload]

    slices = [data[i : i + capacity] for i in range(0, len(data), capacity)]
    total = len(slices)
    logger.debug("livekit_payload_chunked", msg_type=msg_type, bytes=len(raw), chunks=total)
    return [_envelope(i, total, body) for i, body in enumerate(slices)]


def _chunk_transcript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = list(payload.get("entries") or [])
    started_at = payload.get("started_at")

    def _size(ents: list[Any], *, seq: int = 0, total: int = 1) -> int:
        body: dict[str, Any] = {
            "type": "session_transcript",
            "seq": seq,
            "total": total,
            "entries": ents,
        }
        if started_at is not None:
            body["started_at"] = started_at
        return len(_dumps(body))

    packed: list[list[Any]] = []
    current: list[Any] = []
    for entry in entries:
        trial = current + [entry]
        if current and _size(trial) > _MAX_RELIABLE_BYTES:
            packed.append(current)
            current = [entry]
        else:
            current = trial
    if current or not packed:
        packed.append(current)
    total = len(packed)
    chunks: list[dict[str, Any]] = []
    for i, ents in enumerate(packed):
        chunk: dict[str, Any] = {
            "type": "session_transcript",
            "seq": i,
            "total": total,
            "entries": ents,
        }
        if started_at is not None:
            chunk["started_at"] = started_at
        chunks.append(chunk)
    return chunks


def _dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def is_config_hello(data: dict[str, Any]) -> bool:
    """Untyped JSON object — same rule as the WS hello (`type` absent or null)."""
    return data.get("type") is None


def merge_client_config(env_raw: str, hello: dict[str, Any] | None) -> dict[str, Any]:
    """Env JSON is the base; the data-message hello overlays it."""
    config: dict[str, Any] = {}
    try:
        parsed = json.loads(env_raw or "{}")
        if isinstance(parsed, dict):
            config = parsed
    except ValueError:
        logger.warning("voice_livekit_bad_client_config")
    if hello:
        config = {**config, **hello}
    return config


def _local_identity(room: Any, dial: LivekitDial) -> str:
    if dial.agent_identity:
        return dial.agent_identity
    local = getattr(room, "local_participant", None)
    if local is None:
        return ""
    return getattr(local, "identity", "") or ""


def _resolve_call_id(dial: LivekitDial, sip_attributes: dict[str, str] | None) -> str:
    if dial.call_id:
        return dial.call_id
    from .livekit_sip import SIP_ATTR_CALL_ID, call_id_from_env

    env = call_id_from_env()
    if env:
        return env
    if sip_attributes:
        val = sip_attributes.get(SIP_ATTR_CALL_ID)
        if isinstance(val, str) and val:
            return val
    return ""


async def _run_livekit_session(
    app: Any,
    dial: LivekitDial | None = None,
    join: _Join | None = None,
) -> None:
    """Drive one room to completion. ``dial=None`` reads the boot env."""
    if dial is None:
        dial = dial_from_env()

    def _reject(reason: str, status: int = 502) -> None:
        if join is not None:
            join.fail(reason, status)

    try:
        from livekit import rtc
    except ImportError:
        logger.error(
            "voice_livekit_missing_extra",
            hint="the livekit voice transport requires timbal[voice-livekit]",
        )
        _reject("the livekit voice transport requires timbal[voice-livekit]", 501)
        return

    from ..core.agent import Agent
    from ..voice import AudioOutput
    from ..voice.livekit import LkPacedSource, audio_stream_to_pcm
    from .livekit_call_control import LivekitCallControl, livekit_call_control_tools, with_call_tools
    from .livekit_sip import (
        CallerDisconnectAction,
        caller_disconnect_action,
        dtmf_code,
        dtmf_event_payload,
        find_eligible_caller,
        is_eligible_caller,
        is_sip_participant,
        phone_tuned_voice_config,
        sip_abandon_secs,
        sip_call_context,
        sip_recording_meta,
    )

    url = dial.url
    token = dial.token
    if not url or not token:
        logger.error("voice_livekit_missing_env", has_url=bool(url), has_token=bool(token))
        _reject("livekit dial requires 'url' and 'token'", 400)
        return

    runnable = app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error(
            "voice_livekit_rejected",
            reason="runnable is not an Agent",
            type=type(runnable).__name__,
        )
        _reject("voice requires an Agent runnable", 400)
        return

    guard = getattr(app.state, "single_session_guard", None)
    if guard is not None and not guard.claim():
        logger.info("voice_livekit_rejected", reason="single-session server already served its session")
        _reject("single-session server: a voice session was already served", 409)
        return

    caller_hint = dial.caller_identity
    room_name = dial.room
    local_identity = dial.agent_identity

    # Room constructed on this loop — see voice/livekit.py docstring.
    # Session + paced source wait until the caller is in (and the config hello
    # window closes) so playground STT/TTS/turn-detector dropdowns apply.
    room = rtc.Room()
    session_holder: dict[str, Any] = {}
    caller_participant: Any | None = None
    caller_identity: str | None = None
    caller_is_sip = False
    mic_tracks: asyncio.Queue[Any] = asyncio.Queue()
    caller_ready = asyncio.Event()
    session_aborted = asyncio.Event()
    hello_holder: dict[str, Any] = {"hello": None}
    hello_event = asyncio.Event()
    send_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    call_control_holder: dict[str, LivekitCallControl | None] = {"c": None}
    pending_disconnect: asyncio.Task | None = None

    async def _send(payload: dict[str, Any]) -> None:
        dest = [caller_identity] if caller_identity else None
        kwargs: dict[str, Any] = {"reliable": True, "topic": _EVENTS_TOPIC}
        if dest:
            kwargs["destination_identities"] = dest
        try:
            await room.local_participant.publish_data(_dumps(payload), **kwargs)
        except Exception as e:
            logger.debug("voice_livekit_send_failed", error=str(e), msg_type=payload.get("type"))

    async def _sender() -> None:
        while True:
            payload = await send_q.get()
            if payload is None:
                return
            await _send(payload)

    def _cancel_pending_disconnect() -> None:
        nonlocal pending_disconnect
        if pending_disconnect is not None:
            pending_disconnect.cancel()
            pending_disconnect = None

    def _abort_before_media() -> None:
        # Early caller resolution (_note_caller from remote_participants /
        # participant_connected) can run before a mic track arrives.
        # Disconnect then never sets caller_ready — unblock the wait so the
        # driver can release its room key and capacity slot.
        if not caller_ready.is_set():
            session_aborted.set()

    async def _close_session_later(delay: float) -> None:
        await asyncio.sleep(delay)
        sess = session_holder.get("s")
        if sess is not None:
            await sess.close()
            return
        if not caller_ready.is_set():
            _abort_before_media()
            return
        if guard is not None:
            await guard.finish()
        else:
            await room.disconnect()

    async def _wait_ready_or_abort() -> None:
        if caller_ready.is_set() or session_aborted.is_set():
            return
        ready_task = asyncio.create_task(caller_ready.wait())
        abort_task = asyncio.create_task(session_aborted.wait())
        _done, pending = await asyncio.wait(
            {ready_task, abort_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _arm_caller_disconnect(participant: Any) -> None:
        nonlocal pending_disconnect
        action = caller_disconnect_action(participant)
        logger.info(
            "voice_livekit_caller_disconnected",
            identity=getattr(participant, "identity", ""),
            kind=getattr(participant, "kind", None),
            reason=getattr(participant, "disconnect_reason", None),
            action=action.value,
        )
        if action is CallerDisconnectAction.CLOSE:
            _cancel_pending_disconnect()
            sess = session_holder.get("s")
            if sess is not None:
                asyncio.create_task(sess.close())
            elif caller_ready.is_set():
                if guard is not None:
                    asyncio.create_task(guard.finish())
                else:
                    asyncio.create_task(room.disconnect())
            else:
                _abort_before_media()
            return
        if action is CallerDisconnectAction.SHORT_ABANDON:
            _cancel_pending_disconnect()
            pending_disconnect = asyncio.create_task(
                _close_session_later(sip_abandon_secs()),
                name="voice-livekit-sip-abandon",
            )
            return
        if guard is not None:
            guard.mark_disconnected(on_abandon=_close_session_if_any)

    def _note_caller(participant: Any) -> None:
        nonlocal caller_participant, caller_identity, caller_is_sip
        if caller_participant is not None:
            return
        local_id = _local_identity(room, dial) or local_identity
        if not is_eligible_caller(participant, local_identity=local_id, caller_hint=caller_hint):
            return
        caller_participant = participant
        caller_identity = getattr(participant, "identity", "") or ""
        caller_is_sip = is_sip_participant(participant)
        attrs = dict(getattr(participant, "attributes", None) or {})
        call_control_holder["c"] = LivekitCallControl(
            room=room,
            room_name=room_name or getattr(room, "name", "") or "",
            caller_identity=caller_identity,
            call_id=_resolve_call_id(dial, attrs),
            is_sip=caller_is_sip,
            call_control_url=dial.call_control_url,
            call_control_token=dial.call_control_token,
        )

    def _on_sub(track: Any, pub: Any, participant: Any) -> None:
        _note_caller(participant)
        if caller_participant is not participant:
            return
        kind = getattr(track, "kind", None)
        audio_kind = getattr(rtc.TrackKind, "KIND_AUDIO", 1)
        if kind != audio_kind:
            return
        mic_tracks.put_nowait(track)
        if guard is not None:
            guard.mark_connected()
            guard.mark_reconnected()
        caller_ready.set()

    def _close_session_if_any() -> Any:
        sess = session_holder.get("s")
        return sess.close() if sess is not None else None

    def _on_participant_disconnected(participant: Any) -> None:
        identity = getattr(participant, "identity", "") or ""
        if caller_identity and identity == caller_identity:
            _arm_caller_disconnect(participant)

    def _on_participant_connected(participant: Any) -> None:
        identity = getattr(participant, "identity", "") or ""
        _note_caller(participant)
        if caller_identity and identity == caller_identity:
            logger.info("voice_livekit_caller_reconnected", identity=identity)
            _cancel_pending_disconnect()
            if guard is not None:
                guard.mark_reconnected()

    def _on_disconnected(*_args: object) -> None:
        logger.warning("voice_livekit_sfu_disconnected")
        sess = session_holder.get("s")
        if sess is not None:
            asyncio.create_task(sess.close())

    def _on_data(pkt: Any) -> None:
        try:
            data = json.loads(bytes(pkt.data))
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        if (
            getattr(pkt, "topic", "") == "timbal.sip.control"
            and getattr(pkt, "participant", None) is None
            and data.get("type") == "timbal.sip.send_dtmf"
        ):
            digits = str(data.get("digits") or "").strip().upper()
            target = str(data.get("participant_identity") or "")
            if (
                caller_is_sip
                and caller_identity
                and target == caller_identity
                and 0 < len(digits) <= 32
                and all(ch in "0123456789*#ABCD" for ch in digits)
            ):

                async def _publish_control_dtmf() -> None:
                    publish = getattr(room.local_participant, "publish_dtmf", None)
                    if publish is None:
                        logger.error("voice_livekit_publish_dtmf_unavailable")
                        return
                    for digit in digits:
                        await publish(code=dtmf_code(digit), digit=digit)

                asyncio.create_task(_publish_control_dtmf(), name="voice-livekit-control-dtmf")
            else:
                logger.warning("voice_livekit_rejected_dtmf_control")
            return
        if is_config_hello(data):
            hello_holder["hello"] = data
            hello_event.set()
            return
        typ = data.get("type")
        sess = session_holder.get("s")
        if typ == "playback" and sess is not None:
            try:
                sess.playback.on_playback_ack(float(data["played_ms"]))
            except (KeyError, TypeError, ValueError):
                logger.debug("voice_livekit_bad_playback_ack", data=str(data)[:120])

    def _on_sip_dtmf(dtmf: Any) -> None:
        digit = getattr(dtmf, "digit", "") or ""
        code = getattr(dtmf, "code", None)
        identity = getattr(getattr(dtmf, "participant", None), "identity", "") or caller_identity or ""
        if not digit:
            return
        logger.info("voice_livekit_dtmf_received", digit=digit, identity=identity)
        payload = dtmf_event_payload(
            digit=str(digit),
            code=int(code) if code is not None else 0,
            identity=identity,
        )
        asyncio.create_task(send_q.put(payload))

    room.on("track_subscribed", _on_sub)
    room.on("participant_disconnected", _on_participant_disconnected)
    room.on("participant_connected", _on_participant_connected)
    room.on("disconnected", _on_disconnected)
    room.on("data_received", _on_data)
    if hasattr(room, "on"):
        with contextlib.suppress(Exception):
            room.on("sip_dtmf_received", _on_sip_dtmf)

    sender_task = asyncio.create_task(_sender(), name="voice-livekit-sender")
    downlink = None
    try:
        await room.connect(url, token)
        local_identity = _local_identity(room, dial) or local_identity
        remotes = getattr(room, "remote_participants", None)
        if remotes is not None:
            values = remotes.values() if hasattr(remotes, "values") else remotes
            found = find_eligible_caller(values, local_identity=local_identity, caller_hint=caller_hint)
            if found is not None:
                _note_caller(found)
        logger.info(
            "voice_livekit_joined",
            url=url,
            room=room_name or getattr(room, "name", None),
            caller_identity=caller_identity,
            caller_is_sip=caller_is_sip,
        )
        # In the room — this is the readiness proof a per-request caller waits
        # on. Everything below (caller mic, hello, session build) happens after
        # the platform has already answered 200.
        if join is not None:
            join.ok()
        await _wait_ready_or_abort()
        if session_aborted.is_set() and not caller_ready.is_set():
            if guard is not None:
                guard.release()
            return
        if not hello_event.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(hello_event.wait(), timeout=_HELLO_WAIT_SECS)
        config = merge_client_config(dial.client_config, hello_holder["hello"])
        if caller_is_sip:
            config = phone_tuned_voice_config(config)
        sip_ctx: dict[str, str] = {}
        sip_meta: dict[str, str] = {}
        if caller_is_sip:
            sip_meta["transport_detail"] = "livekit_sip"
            if caller_participant is not None:
                attrs = dict(getattr(caller_participant, "attributes", None) or {})
                sip_ctx = sip_call_context(attrs)
                sip_meta.update(sip_recording_meta(attrs))
        defaults = getattr(app.state, "voice_config", None) or VoiceConfig()
        sample_rate = int(merge_client_voice_overrides(defaults, config).sample_rate)

        downlink = LkPacedSource(sample_rate=sample_rate)
        session_runnable = runnable
        if caller_is_sip and isinstance(runnable, Agent):
            session_runnable = with_call_tools(
                runnable,
                livekit_call_control_tools(call_control_holder["c"]),
            )
        session, meta = build_voice_session(
            session_runnable,
            defaults,
            config,
            playback_tracker=downlink.tracker,
            call_context=sip_ctx or None,
            # From the dial only — server-minted. The data-channel hello is the
            # browser's, and a browser must not pick the thread it joins.
            parent_run_id=dial.parent_id or None,
        )
        meta = {
            "playback_acks": "native",
            "transport": "livekit",
            "caller_is_sip": caller_is_sip,
            **sip_meta,
            **meta,
        }
        session.recording_meta = {**(session.recording_meta or {}), **meta}
        session_holder["s"] = session

        track = rtc.LocalAudioTrack.create_audio_track("agent", downlink.source)
        await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await downlink.start()

        async def _mic_pcm() -> Any:
            # Stay open across caller blips: a rejoin publishes a new track and
            # we pick it up here instead of ending the session's audio_in.
            while True:
                remote = await mic_tracks.get()
                stream = rtc.AudioStream(remote, sample_rate=sample_rate, num_channels=1)
                async for chunk in audio_stream_to_pcm(stream):
                    yield chunk

        try:
            async with aclosing(session.run(_mic_pcm())) as events:
                async for event in events:
                    if isinstance(event, AudioOutput):
                        downlink.write(event.data)
                        continue
                    for payload in chunk_data_payloads(event_to_payloads(event, session, meta)):
                        await send_q.put(payload)
        except Exception as e:
            logger.error("voice_livekit_session_error", error=str(e), exc_info=True)
    except BaseException as e:
        # A failure before `join.ok()` is the request's answer; after it, the
        # request is long gone and this is just teardown (`fail` is a no-op
        # once the handshake completed). The reason the caller sees is
        # deliberately fixed — `e` carries internal hostnames and stack detail
        # and this route is reachable by whoever can reach the process — so the
        # detail goes to the log, once, and only for a real join failure
        # (a cancel here is a shutdown or the 504 path, not news).
        if join is not None and not join.done.is_set() and not isinstance(e, asyncio.CancelledError):
            logger.error("voice_livekit_join_failed", room=room_name, error=str(e), exc_info=True)
        _reject("the agent could not join the room")
        # Before the caller's mic subscribed, the idle timer is still armed —
        # release the claim and let it own the exit. After subscribe,
        # mark_connected disarmed it, so any failure (build_voice_session,
        # publish_track, …) must exit through finish() in the finally below;
        # release() here would leave the box unclaimed, idle-disarmed and
        # alive forever.
        if guard is not None and not caller_ready.is_set():
            guard.release()
        raise
    finally:
        # Never leave a per-request caller waiting on a handshake that can no
        # longer complete (cancelled task, driver returned early).
        _reject("livekit session ended before the agent joined")
        _cancel_pending_disconnect()
        # Every await below is a cancellation point: a (re-)delivered
        # CancelledError is not an Exception, and letting it out of any step
        # would skip the rest — room left connected, guard.finish never runs.
        # Suppress BaseException per step so the whole tail always executes;
        # the original unwind (if any) re-raises when the finally completes.
        if downlink is not None:
            with contextlib.suppress(BaseException):
                await downlink.aclose()
        send_q.put_nowait(None)  # unbounded queue — no cancellation point
        with contextlib.suppress(BaseException):
            await sender_task
        with contextlib.suppress(BaseException):
            await room.disconnect()
        logger.info("voice_livekit_disconnected")
        if guard is not None and caller_ready.is_set():
            with contextlib.suppress(BaseException):
                await guard.finish()
