"""Voice over WebRTC — offer/answer signaling for the same runnable as ``/voice/ws``.

``POST /voice/rtc`` takes ``{"sdp": ..., "type": "offer", "config": {...}}``
(``config`` uses the same keys as the WebSocket hello) and returns the SDP
answer — WHIP-style, one round trip, no trickle ICE (aiortc finishes
gathering before answering).

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
import json
import os
from contextlib import aclosing
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .voice import build_voice_session, event_to_payloads, merge_client_voice_overrides

logger = structlog.get_logger("timbal.server.rtc")

router = APIRouter(prefix="/voice", tags=["voice"])

# Peer connections (and their driver tasks) stay alive via these references;
# entries are discarded when the session driver tears the connection down.
_pcs: set[Any] = set()
_drivers: set[asyncio.Task] = set()


def _ice_servers() -> list[Any]:
    from aiortc import RTCIceServer

    servers = []
    # Empty TIMBAL_STUN_URL disables STUN (loopback/tests); unset keeps the default.
    stun_url = os.environ.get("TIMBAL_STUN_URL", "stun:stun.l.google.com:19302")
    if stun_url:
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


@router.post("/rtc")
async def voice_rtc(request: Request) -> JSONResponse:
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

    defaults: dict = getattr(request.app.state, "voice_config", None) or {}
    sample_rate = int(merge_client_voice_overrides(defaults, config).get("sample_rate", 16_000))

    downlink = PcmQueueTrack(sample_rate=sample_rate)
    session, meta = build_voice_session(
        runnable, defaults, config, playback_tracker=PacedPlaybackTracker(downlink)
    )
    meta = {"playback_acks": "native", "transport": "webrtc", **meta}
    session.recording_meta = meta

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=_ice_servers()))
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
        if pc.connectionState in ("failed", "closed"):
            # Client is gone: end the session now rather than waiting for the
            # STT provider to time out on silence. Idempotent when the driver
            # already closed it.
            await session.close()

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
    except Exception as e:
        _pcs.discard(pc)
        logger.warning("voice_rtc_bad_offer", error=str(e))
        return JSONResponse(status_code=400, content={"error": f"Invalid SDP offer: {e}"})

    if mic_track is None:
        _pcs.discard(pc)
        with contextlib.suppress(Exception):
            await pc.close()
        return JSONResponse(status_code=400, content={"error": "Offer must contain an audio track."})

    # After setRemoteDescription, so the TTS track reuses the offer's audio
    # transceiver instead of adding a second m-line the client never offered.
    pc.addTrack(downlink)

    async def _drive() -> None:
        # Do not start the session before the transport is usable: STT starts
        # its clock on connect, and a session that dies immediately (bad API
        # key) must still get its error payload to a client whose SCTP
        # handshake hasn't finished. ICE failure lands in
        # connectionstatechange, so this only times out on a client that
        # connected but never offered a data channel.
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

    driver = asyncio.create_task(_drive())
    _drivers.add(driver)
    driver.add_done_callback(_drivers.discard)

    answer = await pc.createAnswer()
    # Completes ICE gathering before returning — the answer is complete.
    await pc.setLocalDescription(answer)
    logger.info("voice_rtc_connected")
    return JSONResponse(
        content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )
