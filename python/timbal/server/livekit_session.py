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
* ``TIMBAL_VOICE_CLIENT_CONFIG`` — JSON, same keys as the WS hello / rtc config.
  Overlay: after the caller publishes a mic, the driver waits up to 2s for an
  untyped data-message hello (playground dropdowns) and merges it on top.
* ``TIMBAL_VOICE_ABANDON_SECS`` — default 45; see ``SingleSessionGuard``

``TIMBAL_VOICE_SINGLE_SESSION=1`` still applies. ``claim()`` happens at
room-join rather than offer-receipt; ``mark_connected()`` fires on the
caller's mic track being subscribed. Session + TTS track are built only
after that subscribe (and the hello window), so playground config actually
applies.
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

_MAX_RELIABLE_BYTES = 12 * 1024
_EVENTS_TOPIC = "timbal.events"
_HELLO_WAIT_SECS = 2.0


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
        raw_len = len(_dumps(payload))
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

    caller_prefix = os.environ.get("TIMBAL_LIVEKIT_CALLER_IDENTITY", "").strip()
    room_name = os.environ.get("TIMBAL_LIVEKIT_ROOM", "").strip()

    # Room constructed on this loop — see voice/livekit.py docstring.
    # Session + paced source wait until the caller is in (and the config hello
    # window closes) so playground STT/TTS/turn-detector dropdowns apply.
    room = rtc.Room()
    session_holder: dict[str, Any] = {}
    caller_identity: str | None = None
    mic_tracks: asyncio.Queue[Any] = asyncio.Queue()
    caller_ready = asyncio.Event()
    hello_holder: dict[str, Any] = {"hello": None}
    hello_event = asyncio.Event()
    send_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

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
        mic_tracks.put_nowait(track)
        if guard is not None:
            guard.mark_connected()
            guard.mark_reconnected()
        caller_ready.set()

    def _on_participant_disconnected(participant: Any) -> None:
        identity = getattr(participant, "identity", "") or ""
        if caller_identity and identity == caller_identity:
            logger.info("voice_livekit_caller_disconnected", identity=identity)
            sess = session_holder.get("s")
            if guard is not None and sess is not None:
                guard.mark_disconnected(on_abandon=sess.close)

    def _on_participant_connected(participant: Any) -> None:
        identity = getattr(participant, "identity", "") or ""
        if caller_identity and identity == caller_identity:
            logger.info("voice_livekit_caller_reconnected", identity=identity)
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

    room.on("track_subscribed", _on_sub)
    room.on("participant_disconnected", _on_participant_disconnected)
    room.on("participant_connected", _on_participant_connected)
    room.on("disconnected", _on_disconnected)
    room.on("data_received", _on_data)

    sender_task = asyncio.create_task(_sender(), name="voice-livekit-sender")
    downlink = None
    session_started = False
    try:
        await room.connect(url, token)
        logger.info(
            "voice_livekit_joined",
            url=url,
            room=room_name or getattr(room, "name", None),
        )
        await caller_ready.wait()
        if not hello_event.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(hello_event.wait(), timeout=_HELLO_WAIT_SECS)
        config = merge_client_config(
            os.environ.get("TIMBAL_VOICE_CLIENT_CONFIG") or "{}",
            hello_holder["hello"],
        )
        defaults = getattr(app.state, "voice_config", None) or VoiceConfig()
        sample_rate = int(merge_client_voice_overrides(defaults, config).sample_rate)

        downlink = LkPacedSource(sample_rate=sample_rate)
        session, meta = build_voice_session(
            runnable, defaults, config, playback_tracker=downlink.tracker
        )
        meta = {"playback_acks": "native", "transport": "livekit", **meta}
        session.recording_meta = meta
        session_holder["s"] = session

        track = rtc.LocalAudioTrack.create_audio_track("agent", downlink.source)
        await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await downlink.start()
        session_started = True

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
    except BaseException:
        if not session_started and guard is not None:
            guard.release()
        raise
    finally:
        if downlink is not None:
            await downlink.aclose()
        await send_q.put(None)
        with contextlib.suppress(Exception):
            await sender_task
        with contextlib.suppress(Exception):
            await room.disconnect()
        logger.info("voice_livekit_disconnected")
        if session_started and guard is not None:
            await guard.finish()
