"""Voice over LiveKit — the box dials out to a room the monolith already created.

Sibling of :mod:`timbal.server.rtc`. The aiortc path is a route
(``POST /voice/rtc``) driven by an incoming offer; this driver starts from
the server lifespan when ``TIMBAL_VOICE_TRANSPORT=livekit`` and joins the
room as a participant. ``rtc.py`` is untouched — it is the rollback rung.

Env contract (all platform-owned):

* ``TIMBAL_VOICE_TRANSPORT=livekit`` selects this driver; unset/``webrtc``
  keeps ``rtc.py``.
* ``TIMBAL_LIVEKIT_URL`` — ``ws://<sfu-private-ip>:7880``
* ``TIMBAL_LIVEKIT_TOKEN`` — agent join token (the room is already pinned)
* ``TIMBAL_LIVEKIT_ROOM`` — informational; the token already pins it
* ``TIMBAL_LIVEKIT_CALLER_IDENTITY`` — identity prefix treated as the human
* ``TIMBAL_VOICE_CLIENT_CONFIG`` — JSON, same keys as the WS hello / rtc config
* ``TIMBAL_VOICE_ABANDON_SECS`` — default 45; see ``SingleSessionGuard``

``TIMBAL_VOICE_SINGLE_SESSION=1`` still applies. ``claim()`` happens at
room-join rather than offer-receipt; ``mark_connected()`` fires on the
caller's mic track being subscribed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import aclosing
from typing import Any

import structlog

from ..voice.config import VoiceConfig
from .voice import build_voice_session, event_to_payloads, merge_client_voice_overrides

logger = structlog.get_logger("timbal.server.livekit_session")

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_MAX_RELIABLE_BYTES = 12 * 1024
_EVENTS_TOPIC = "timbal.events"


def maybe_start_livekit_session(app: Any) -> asyncio.Task | None:
    """Lifespan hook: start the dial-out driver when the transport is LiveKit."""
    if os.environ.get("TIMBAL_VOICE_TRANSPORT", "").strip().lower() != "livekit":
        return None
    return asyncio.create_task(_run_livekit_session(app), name="voice-livekit-session")


def chunk_data_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split payloads that would exceed LiveKit's ~15 KiB reliable-data cap.

    ``session_transcript`` is the only event that grows without bound. Chunks
    are ``{"type":"session_transcript","seq":i,"total":n,"entries":[...]}``;
    a client reassembles by ``seq``. Other oversized payloads are logged and
    sent as-is — we cannot split them without a protocol change.
    """
    out: list[dict[str, Any]] = []
    for payload in payloads:
        raw_len = len(json.dumps(payload, separators=(",", ":")).encode())
        if raw_len <= _MAX_RELIABLE_BYTES:
            out.append(payload)
            continue
        if payload.get("type") != "session_transcript":
            logger.warning(
                "livekit_payload_oversized",
                msg_type=payload.get("type"),
                bytes=raw_len,
            )
            out.append(payload)
            continue
        out.extend(_chunk_transcript(payload))
    return out


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
        return len(json.dumps(body, separators=(",", ":")).encode())

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


def _is_caller(identity: str, prefix: str) -> bool:
    if not prefix:
        return True
    return identity.startswith(prefix)


async def _run_livekit_session(app: Any) -> None:
    try:
        from livekit import rtc
    except ImportError:
        logger.error(
            "voice_livekit_missing_extra",
            hint="TIMBAL_VOICE_TRANSPORT=livekit requires timbal[voice-livekit]",
        )
        return

    from ..core.agent import Agent
    from ..voice import AudioOutput
    from ..voice.livekit import LkPacedSource, audio_stream_to_pcm

    url = os.environ.get("TIMBAL_LIVEKIT_URL", "").strip()
    token = os.environ.get("TIMBAL_LIVEKIT_TOKEN", "").strip()
    if not url or not token:
        logger.error("voice_livekit_missing_env", has_url=bool(url), has_token=bool(token))
        return

    runnable = app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error(
            "voice_livekit_rejected",
            reason="runnable is not an Agent",
            type=type(runnable).__name__,
        )
        return

    guard = getattr(app.state, "single_session_guard", None)
    if guard is not None and not guard.claim():
        logger.info("voice_livekit_rejected", reason="single-session server already served its session")
        return

    try:
        raw_config = os.environ.get("TIMBAL_VOICE_CLIENT_CONFIG") or "{}"
        config = json.loads(raw_config)
        if not isinstance(config, dict):
            config = {}
    except ValueError:
        logger.warning("voice_livekit_bad_client_config")
        config = {}

    caller_prefix = os.environ.get("TIMBAL_LIVEKIT_CALLER_IDENTITY", "").strip()
    room_name = os.environ.get("TIMBAL_LIVEKIT_ROOM", "").strip()

    defaults = getattr(app.state, "voice_config", None) or VoiceConfig()
    sample_rate = int(merge_client_voice_overrides(defaults, config).sample_rate)

    # Room + source constructed on this loop — see voice/livekit.py docstring.
    room = rtc.Room()
    downlink = LkPacedSource(sample_rate=sample_rate)
    session, meta = build_voice_session(
        runnable, defaults, config, playback_tracker=downlink.tracker
    )
    meta = {"playback_acks": "native", "transport": "livekit", **meta}
    session.recording_meta = meta

    caller_identity: str | None = None
    mic_stream: asyncio.Queue[Any] = asyncio.Queue()
    caller_ready = asyncio.Event()
    pending_payloads: list[dict[str, Any]] = []
    send_lock = asyncio.Lock()

    async def _send(payload: dict[str, Any]) -> None:
        dest = [caller_identity] if caller_identity else None
        kwargs: dict[str, Any] = {"reliable": True, "topic": _EVENTS_TOPIC}
        if dest:
            kwargs["destination_identities"] = dest
        try:
            await room.local_participant.publish_data(
                json.dumps(payload).encode(), **kwargs
            )
        except Exception as e:
            logger.debug("voice_livekit_send_failed", error=str(e), msg_type=payload.get("type"))

    async def _flush_pending() -> None:
        async with send_lock:
            queued = pending_payloads[:]
            pending_payloads.clear()
        for payload in queued:
            await _send(payload)

    def _queue_or_send(payload: dict[str, Any]) -> None:
        if caller_ready.is_set():
            asyncio.create_task(_send(payload))
        else:
            pending_payloads.append(payload)

    def _on_sub(track: Any, pub: Any, participant: Any) -> None:
        nonlocal caller_identity
        identity = getattr(participant, "identity", "") or ""
        if not _is_caller(identity, caller_prefix):
            return
        kind = getattr(track, "kind", None)
        audio_kind = getattr(rtc.TrackKind, "KIND_AUDIO", 1)
        if kind != audio_kind:
            return
        caller_identity = identity
        mic_stream.put_nowait(rtc.AudioStream(track, sample_rate=sample_rate, num_channels=1))
        if guard is not None:
            guard.mark_connected()
            guard.mark_reconnected()
        if not caller_ready.is_set():
            caller_ready.set()
            asyncio.create_task(_flush_pending())

    def _on_participant_disconnected(participant: Any) -> None:
        identity = getattr(participant, "identity", "") or ""
        if caller_identity and identity == caller_identity:
            logger.info("voice_livekit_caller_disconnected", identity=identity)
            if guard is not None:
                guard.mark_disconnected(on_abandon=session.close)

    def _on_participant_connected(participant: Any) -> None:
        identity = getattr(participant, "identity", "") or ""
        if caller_identity and identity == caller_identity:
            logger.info("voice_livekit_caller_reconnected", identity=identity)
            if guard is not None:
                guard.mark_reconnected()

    def _on_disconnected(*_args: object) -> None:
        logger.warning("voice_livekit_sfu_disconnected")
        asyncio.create_task(session.close())

    def _on_data(pkt: Any) -> None:
        try:
            data = json.loads(bytes(pkt.data))
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        typ = data.get("type")
        if typ == "playback":
            try:
                session.playback.on_playback_ack(float(data["played_ms"]))
            except (KeyError, TypeError, ValueError):
                logger.debug("voice_livekit_bad_playback_ack", data=str(data)[:120])

    room.on("track_subscribed", _on_sub)
    room.on("participant_disconnected", _on_participant_disconnected)
    room.on("participant_connected", _on_participant_connected)
    room.on("disconnected", _on_disconnected)
    room.on("data_received", _on_data)

    try:
        await room.connect(url, token)
        logger.info(
            "voice_livekit_joined",
            url=url,
            room=room_name or getattr(room, "name", None),
        )
        track = rtc.LocalAudioTrack.create_audio_track("agent", downlink.source)
        await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await downlink.start()
    except Exception:
        await downlink.aclose()
        with contextlib.suppress(Exception):
            await room.disconnect()
        if guard is not None:
            guard.release()
        raise

    async def _mic_pcm() -> Any:
        # Stay open across caller blips: a rejoin publishes a new track and
        # we pick it up here instead of ending the session's audio_in.
        while True:
            stream = await mic_stream.get()
            async for chunk in audio_stream_to_pcm(stream):
                yield chunk

    try:
        await caller_ready.wait()
        async with aclosing(session.run(_mic_pcm())) as events:
            async for event in events:
                if isinstance(event, AudioOutput):
                    downlink.write(event.data)
                    continue
                for payload in chunk_data_payloads(event_to_payloads(event, session, meta)):
                    _queue_or_send(payload)
    except Exception as e:
        logger.error("voice_livekit_session_error", error=str(e), exc_info=True)
    finally:
        await downlink.aclose()
        with contextlib.suppress(Exception):
            await room.disconnect()
        logger.info("voice_livekit_disconnected")
        if guard is not None:
            await guard.finish()
