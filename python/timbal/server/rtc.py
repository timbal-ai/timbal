"""Voice over WebRTC — offer/answer signaling for the same runnable as ``/voice/ws``.

``POST /voice/rtc`` takes ``{"sdp": ..., "type": "offer", "config": {...}}``
(``config`` uses the same keys as the WebSocket hello) and returns the SDP
answer — WHIP-style, one round trip, no trickle ICE (aiortc finishes
gathering before answering).

The same route also accepts a LiveKit dial (``{"transport": "livekit", …}``),
which joins a platform-created room instead of answering an offer and is
handled by :mod:`timbal.server.livekit_session`. That fork is what lets a
long-lived server (ECS / on-premise) serve LiveKit calls without the
one-process-per-call boot env the serverless path uses. Unlike an offer, a
dial names an outbound destination, so it is gated on
``TIMBAL_VOICE_DIAL_SECRET`` / ``TIMBAL_LIVEKIT_URL`` where those are set —
see :func:`_dial_authorized`.

Protocol expectations for clients:

* The offer must contain one audio track (the mic) **and** a data channel —
  the channel rides the offer's SCTP m-line, so the client creates it; the
  server binds to whatever channel arrives, whatever its label.
* Session events arrive as JSON on the data channel — the same payloads as
  the WebSocket transport, except ``audio``: TTS is a real audio track here,
  paced by the server, so there are no base64 audio messages and **no
  playback acks** — the server's own pacing clock is the played position,
  which makes barge-in ``heard_text`` exact instead of estimated.

Requires the ``timbal[voice]`` extra (aiortc); the route answers 501 without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
from contextlib import aclosing
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..voice.config import VoiceConfig
from .capacity import acquire_session_slot, max_concurrent_sessions, release_session_slot
from .voice import (
    build_voice_session,
    client_call_context,
    event_to_payloads,
    merge_client_voice_overrides,
)

logger = structlog.get_logger("timbal.server.rtc")

router = APIRouter(prefix="/voice", tags=["voice"])

# Peer connections (and their driver tasks) stay alive via these references;
# entries are discarded when the session driver tears the connection down.
_pcs: set[Any] = set()
_drivers: set[asyncio.Task] = set()

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# One warning per process for a dial route with neither a secret nor a pinned
# url — see `_dial_authorized`.
_warned_open_dial = False


def _force_relay() -> bool:
    """``TIMBAL_VOICE_RTC_FORCE_RELAY=1`` *and* a TURN server is configured.

    Serverless boxes sit in private subnets: host candidates are private IPs
    and srflx can't receive unsolicited inbound, so every non-relay candidate
    is dead weight that slows the browser's ICE convergence. Without a TURN
    server relay-only would leave *no* reachable candidate, so a misconfigured
    flag degrades (loudly) to normal gathering instead of a broken answer.
    """
    if os.environ.get("TIMBAL_VOICE_RTC_FORCE_RELAY", "").strip().lower() not in _TRUTHY:
        return False
    if not os.environ.get("TIMBAL_TURN_URL"):
        logger.error(
            "voice_rtc_force_relay_without_turn",
            hint="TIMBAL_VOICE_RTC_FORCE_RELAY=1 requires TIMBAL_TURN_URL/"
            "TIMBAL_TURN_USERNAME/TIMBAL_TURN_PASSWORD; serving all candidates",
        )
        return False
    return True


def _ice_servers(*, relay_only: bool = False) -> list[Any]:
    from aiortc import RTCIceServer

    servers = []
    # Empty TIMBAL_STUN_URL disables STUN (loopback/tests); unset keeps the
    # default. Relay-only skips STUN entirely — srflx would be filtered from
    # the answer anyway, so gathering it just burns signaling budget.
    stun_url = os.environ.get("TIMBAL_STUN_URL", "stun:stun.l.google.com:19302")
    if stun_url and not relay_only:
        servers.append(RTCIceServer(urls=stun_url))
    turn_url = os.environ.get("TIMBAL_TURN_URL")
    if turn_url:
        servers.append(
            RTCIceServer(
                urls=turn_url,
                username=os.environ.get("TIMBAL_TURN_USERNAME"),
                credential=os.environ.get("TIMBAL_TURN_PASSWORD"),
            )
        )
    return servers


def _dial_authorized(request: Request) -> bool:
    """Gate the LiveKit dial on ``TIMBAL_VOICE_DIAL_SECRET`` when it is set.

    An SDP offer only ever costs this process a session; a dial tells it which
    SFU to connect to and then spends STT/TTS/LLM budget streaming to whoever
    is in that room, so it is the one body on this route worth authenticating.
    Same shape as the Twilio/Telnyx signature checks in ``telephony.py``.

    Unset means no check — the serverless platform fronts this route and the
    tests do not carry a secret. A deployment reachable by anything else wants
    this set, alongside ``TIMBAL_LIVEKIT_URL`` to pin the destination.
    """
    global _warned_open_dial
    secret = os.environ.get("TIMBAL_VOICE_DIAL_SECRET", "").strip()
    if not secret:
        if not _warned_open_dial and not os.environ.get("TIMBAL_LIVEKIT_URL", "").strip():
            # Once per process: it is a deployment fact, not a per-call event.
            _warned_open_dial = True
            logger.warning(
                "voice_rtc_dial_unauthenticated",
                hint="set TIMBAL_VOICE_DIAL_SECRET and/or TIMBAL_LIVEKIT_URL: any client that "
                "can reach this route can make this process join an arbitrary room",
            )
        return True
    return hmac.compare_digest(request.headers.get("x-timbal-dial-secret", ""), secret)


def _strip_non_relay_candidates(sdp: str) -> str:
    """Drop host/srflx ``a=candidate`` lines from an answer, keeping relay only.

    aiortc has no ``iceTransportPolicy``: it always gathers host candidates,
    so relay-only has to be enforced on the answer SDP. The browser only
    forms pairs with candidates it was told about, so filtering here is
    enough — aiortc's internal host candidates are never checked.

    Degrades to the unfiltered SDP when filtering would leave no candidates
    (e.g. the TURN allocation failed): a slow answer beats an unconnectable
    one. The check is global rather than per m-section because all sections
    share one gather (BUNDLE) and therefore identical candidate sets.
    """
    lines = sdp.split("\r\n")
    kept: list[str] = []
    removed = relayed = 0
    for line in lines:
        if line.startswith("a=candidate:"):
            if " typ relay" in line:
                relayed += 1
            else:
                removed += 1
                continue
        kept.append(line)
    if removed and not relayed:
        logger.warning(
            "voice_rtc_force_relay_no_relay_candidates",
            hint="TURN allocation yielded no relay candidates; answering with all candidates",
        )
        return sdp
    if removed:
        logger.debug("voice_rtc_relay_filtered", removed=removed, relay=relayed)
    return "\r\n".join(kept)


@router.post("/rtc")
async def voice_rtc(request: Request) -> JSONResponse:
    # Body-discriminated fork. `{"transport": "livekit", "url", "token", …}`
    # joins a room the platform already created; anything else is an SDP offer
    # and keeps the aiortc path byte-for-byte. Sniffed *before* the aiortc
    # import so a deployment pinned to timbal[voice-livekit] without
    # timbal[voice] isn't 501'd on a transport it does support.
    from .livekit_session import dial_from_body, is_livekit_dial, start_livekit_session

    raw = await request.body()
    try:
        sniffed = json.loads(raw)
    except ValueError:
        sniffed = None
    if is_livekit_dial(sniffed):
        if not _dial_authorized(request):
            logger.warning("voice_rtc_dial_unauthorized")
            return JSONResponse(status_code=401, content={"error": "Invalid dial credentials."})
        status, payload = await start_livekit_session(request.app, dial_from_body(sniffed))
        return JSONResponse(status_code=status, content=payload)

    try:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        from ..voice.webrtc import PacedPlaybackTracker, PcmQueueTrack, track_to_pcm
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={"error": "The WebRTC voice transport requires the timbal[voice] extra."},
        )

    from ..core.agent import Agent
    from ..voice import AudioOutput

    body = await request.json()
    sdp = body.get("sdp")
    if not isinstance(sdp, str) or body.get("type") != "offer":
        return JSONResponse(status_code=400, content={"error": "Body must carry an SDP offer."})
    config = body.get("config")
    if not isinstance(config, dict):
        config = {}

    runnable = request.app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error("voice_rtc_rejected", reason="runnable is not an Agent", type=type(runnable).__name__)
        return JSONResponse(status_code=400, content={"error": "Voice requires an Agent runnable."})

    guard = getattr(request.app.state, "single_session_guard", None)
    if guard is not None and not guard.claim():
        logger.info("voice_rtc_rejected", reason="single-session server already served its session")
        return JSONResponse(
            status_code=409,
            content={"error": "Single-session server: a voice session was already served."},
        )

    # Held for exactly as long as the guard's claim, so every release below
    # sits next to an existing one. On a single-session box the guard is the
    # real limit and this never rejects.
    if not acquire_session_slot():
        if guard is not None:
            guard.release()
        return JSONResponse(
            status_code=503,
            content={
                "error": f"Server is at its voice session capacity "
                f"({max_concurrent_sessions()} concurrent)."
            },
        )

    pc: Any = None
    try:
        defaults = getattr(request.app.state, "voice_config", None) or VoiceConfig()
        sample_rate = int(merge_client_voice_overrides(defaults, config).sample_rate)

        downlink = PcmQueueTrack(sample_rate=sample_rate)
        session, meta = build_voice_session(
            runnable,
            defaults,
            config,
            playback_tracker=PacedPlaybackTracker(downlink),
            call_context=client_call_context(config),
        )
        meta = {"playback_acks": "native", "transport": "webrtc", **meta}
        session.recording_meta = meta

        force_relay = _force_relay()
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=_ice_servers(relay_only=force_relay)))
        _pcs.add(pc)

        mic_track: Any = None
        channel: Any = None
        channel_ready = asyncio.Event()
        # Session events can fire before SCTP is up; buffer until the client's
        # data channel arrives.
        pending_payloads: list[str] = []

        def _dc_send(payload: dict) -> None:
            msg = json.dumps(payload)
            if channel is not None and channel.readyState == "open":
                try:
                    channel.send(msg)
                except Exception as e:
                    logger.debug("voice_rtc_dc_send_failed", error=str(e), msg_type=payload.get("type"))
            else:
                pending_payloads.append(msg)

        @pc.on("track")
        def on_track(track: Any) -> None:
            nonlocal mic_track
            if track.kind == "audio" and mic_track is None:
                mic_track = track

        @pc.on("datachannel")
        def on_datachannel(ch: Any) -> None:
            nonlocal channel
            channel = ch
            for msg in pending_payloads:
                try:
                    ch.send(msg)
                except Exception as e:
                    logger.debug("voice_rtc_dc_flush_failed", error=str(e))
                    break
            pending_payloads.clear()
            channel_ready.set()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.debug("voice_rtc_connection_state", state=pc.connectionState)
            if pc.connectionState == "connected" and guard is not None:
                # Media established: the single-session idle-exit window (boot →
                # media connected) no longer applies.
                guard.mark_connected()
            if pc.connectionState in ("failed", "closed"):
                # Client is gone: end the session now rather than waiting for the
                # STT provider to time out on silence. Idempotent when the driver
                # already closed it. Deliberately *not* "disconnected": per W3C
                # semantics that's a possibly-transient ICE blip that can return
                # to "connected" (aiortc doesn't emit it today, but if it ever
                # does, tearing down a recoverable call would be wrong).
                await session.close()

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        except Exception as e:
            _pcs.discard(pc)
            release_session_slot()
            if guard is not None:
                guard.release()
            logger.warning("voice_rtc_bad_offer", error=str(e))
            return JSONResponse(status_code=400, content={"error": f"Invalid SDP offer: {e}"})

        if mic_track is None:
            _pcs.discard(pc)
            with contextlib.suppress(Exception):
                await pc.close()
            release_session_slot()
            if guard is not None:
                guard.release()
            return JSONResponse(status_code=400, content={"error": "Offer must contain an audio track."})

        # After setRemoteDescription, so the TTS track reuses the offer's audio
        # transceiver instead of adding a second m-line the client never offered.
        pc.addTrack(downlink)
    except Exception:
        # A setup crash after claim() (build_voice_session, pc construction,
        # addTrack) must not park the box in a claimed state with no
        # finish(): reopen the slot and let the 500 out — the idle timer is
        # still armed (media never connected), so the box lifetime stays
        # bounded. Once the driver task below exists, lifetime is its
        # finish()'s responsibility instead, never a release. The 400 paths
        # above return (not raise) and do their own release.
        if pc is not None:
            _pcs.discard(pc)
            with contextlib.suppress(Exception):
                await pc.close()
        release_session_slot()
        if guard is not None:
            guard.release()
        raise

    async def _drive() -> None:
        # Do not start the session before the transport is usable: STT starts
        # its clock on connect, and a session that dies immediately (bad API
        # key) must still get its error payload to a client whose SCTP
        # handshake hasn't finished. ICE failure lands in
        # connectionstatechange, so this only times out on a client that
        # connected but never offered a data channel.
        try:
            try:
                await asyncio.wait_for(channel_ready.wait(), timeout=15.0)
            except TimeoutError:
                logger.warning("voice_rtc_no_datachannel")
                with contextlib.suppress(Exception):
                    await pc.close()
                _pcs.discard(pc)
                return
            mic_pcm = track_to_pcm(mic_track, sample_rate=sample_rate)
            try:
                async with aclosing(session.run(mic_pcm)) as events:
                    async for event in events:
                        if isinstance(event, AudioOutput):
                            downlink.write(event.data)
                            continue
                        for payload in event_to_payloads(event, session, meta):
                            _dc_send(payload)
            except Exception as e:
                logger.error("voice_rtc_session_error", error=str(e), exc_info=True)
            finally:
                downlink.stop()
                with contextlib.suppress(Exception):
                    await pc.close()
                _pcs.discard(pc)
                logger.info("voice_rtc_disconnected")
        finally:
            # From here the driver owns the slot: the answer went out, so the
            # session's end is the only thing that frees it.
            release_session_slot()
            # Single-session box: this offer was the one session, however it
            # ended (call finished, ICE never completed, no data channel) —
            # finalize (recording already pushed by session cleanup above)
            # and exit 0 so the platform reaps the box.
            if guard is not None:
                await guard.finish()

    driver = asyncio.create_task(_drive())
    _drivers.add(driver)
    driver.add_done_callback(_drivers.discard)

    answer = await pc.createAnswer()
    # Completes ICE gathering before returning — the answer is complete.
    await pc.setLocalDescription(answer)
    answer_sdp = pc.localDescription.sdp
    if force_relay:
        answer_sdp = _strip_non_relay_candidates(answer_sdp)
    logger.info("voice_rtc_connected")
    return JSONResponse(content={"sdp": answer_sdp, "type": pc.localDescription.type})
