"""Voice routes — browser-based voice session over WebSocket.

Serves ``GET /voice`` (HTML) and ``/voice/ws`` for the same runnable as ``/run``.
Defaults come from :func:`default_voice_config_from_env` and optional
``runnable.voice_config`` (dict or callable); the client can override with a JSON
first message on the socket.

Heavy imports (``VoiceSession``, ElevenLabs) load on first WebSocket connection only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import aclosing
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .. import __version__ as timbal_version

logger = structlog.get_logger("timbal.server.voice")

router = APIRouter(prefix="/voice", tags=["voice"])

_HTML_PATH = Path(__file__).parent / "voice.html"

_DEFAULT_VOICE_ID = "851ejYcv2BoNPjrkw93G"


def default_voice_config_from_env() -> dict[str, Any]:
    """STT/TTS defaults for ``/voice/ws`` (ElevenLabs). Override with env or ``runnable.voice_config``."""
    return {
        "stt_model": os.environ.get("TIMBAL_STT_MODEL", "scribe_v2_realtime"),
        "tts_model": os.environ.get("TIMBAL_TTS_MODEL", "eleven_flash_v2_5"),
        "voice": (os.environ.get("ELEVENLABS_VOICE_ID") or os.environ.get("TIMBAL_VOICE_ID") or _DEFAULT_VOICE_ID),
        "language": os.environ.get("TIMBAL_VOICE_LANGUAGE", "es"),
        "sample_rate": 16_000,
        "stt_extra": {
            "commit_strategy": "vad",
            "min_speech_duration_ms": 300,
            "vad_silence_threshold_secs": 1.2,
            "vad_threshold": 0.4,
        },
        "tts_extra": {"auto_mode": True},
    }


def merge_voice_config(runnable: Any) -> dict[str, Any]:
    """Env defaults, then optional ``runnable.voice_config`` dict or ``lambda -> dict``."""
    base = default_voice_config_from_env()
    vc = getattr(runnable, "voice_config", None)
    if callable(vc):
        vc = vc()
    if not isinstance(vc, dict):
        return base
    skip = frozenset({"stt_extra", "tts_extra"})
    merged = {
        **base,
        **{k: v for k, v in vc.items() if v is not None and k not in skip},
    }
    if isinstance(vc.get("stt_extra"), dict):
        merged["stt_extra"] = {**base.get("stt_extra", {}), **vc["stt_extra"]}
    if isinstance(vc.get("tts_extra"), dict):
        merged["tts_extra"] = {**base.get("tts_extra", {}), **vc["tts_extra"]}
    return merged


def merge_client_voice_overrides(server_defaults: dict[str, Any], client: dict[str, Any]) -> dict[str, Any]:
    """Apply optional first WebSocket JSON message over ``app.state.voice_config``."""
    return {**server_defaults, **{k: v for k, v in client.items() if v is not None}}


def runnable_meta_for_voice_page(runnable: Any, import_spec: str) -> dict[str, str]:
    """Serializable identity for the voice UI (same object as ``/run``)."""
    name = str(getattr(runnable, "name", "") or "").strip()
    kind = ""
    md = getattr(runnable, "metadata", None)
    if isinstance(md, dict) and md.get("type"):
        kind = str(md["type"])
    if not kind:
        kind = type(runnable).__name__
    return {"name": name, "kind": kind, "import_spec": (import_spec or "").strip()}


_VOICE_HTML_META_TOKEN = "__TIMBAL_VOICE_RUNNABLE_META_JSON__"


@router.get("/")
async def voice_page(request: Request) -> HTMLResponse:
    runnable = getattr(request.app.state, "runnable", None)
    import_spec = os.environ.get("TIMBAL_RUNNABLE", "")
    meta = (
        runnable_meta_for_voice_page(runnable, import_spec)
        if runnable is not None
        else {"name": "", "kind": "", "import_spec": import_spec}
    )
    meta["version"] = timbal_version
    html = _HTML_PATH.read_text(encoding="utf-8")
    if _VOICE_HTML_META_TOKEN not in html:
        msg = f"voice.html is missing the {_VOICE_HTML_META_TOKEN!r} placeholder"
        raise RuntimeError(msg)
    body = json.dumps(meta)
    html = html.replace(_VOICE_HTML_META_TOKEN, body)
    return HTMLResponse(html)


@router.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    from ..core.agent import Agent
    from ..voice import (
        AgentTextDelta,
        AgentTextDone,
        AudioInputConfig,
        AudioOutput,
        AudioOutputConfig,
        SessionEnded,
        SessionError,
        SessionInterrupted,
        SessionStarted,
        TranscriptCommitted,
        TranscriptPartial,
        VoiceSession,
        VoiceSessionEvent,
    )
    from ..voice.elevenlabs import ElevenLabsRealtimeSTT, ElevenLabsStreamTTS

    await ws.accept()
    logger.info("voice_ws_connected")

    runnable = ws.app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error("voice_ws_rejected", reason="runnable is not an Agent", type=type(runnable).__name__)
        await ws.close(code=1008, reason="Voice requires an Agent runnable")
        return

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    config: dict = {}
    try:
        first = await asyncio.wait_for(ws.receive(), timeout=2.0)
        if "text" in first and first["text"]:
            config = json.loads(first["text"])
        elif "bytes" in first and first["bytes"]:
            await audio_queue.put(first["bytes"])
    except TimeoutError:
        pass
    except Exception as e:
        logger.warning("voice_ws_first_frame_error", error=str(e))

    defaults: dict = getattr(ws.app.state, "voice_config", None) or {}
    merged = merge_client_voice_overrides(defaults, config)

    stt = ElevenLabsRealtimeSTT()
    tts = ElevenLabsStreamTTS()

    audio_in = AudioInputConfig(
        model=merged.get("stt_model"),
        language=merged.get("language"),
        sample_rate=merged.get("sample_rate", 16_000),
        encoding=merged.get("encoding", "pcm_s16le"),
        extra=merged.get("stt_extra", {}),
    )
    audio_out = AudioOutputConfig(
        model=merged.get("tts_model"),
        voice=merged.get("voice"),
        sample_rate=merged.get("sample_rate", 16_000),
        encoding=merged.get("encoding", "pcm_s16le"),
        extra=merged.get("tts_extra", {}),
    )

    session = VoiceSession(
        agent=runnable,
        stt=stt,
        tts=tts,
        audio_input=audio_in,
        audio_output=audio_out,
    )

    async def _recv_loop() -> None:
        """Read frames from the browser and feed PCM into the audio queue."""
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    await audio_queue.put(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "audio":
                        await audio_queue.put(base64.b64decode(data["data"]))
        except WebSocketDisconnect:
            pass
        finally:
            await audio_queue.put(b"")

    async def _mic_stream():
        """Yield PCM chunks from the browser mic (echo-cancelled by getUserMedia)."""
        while True:
            chunk = await audio_queue.get()
            if not chunk:
                break
            yield chunk

    def _send_failed_is_benign(exc: BaseException) -> bool:
        msg = str(exc).lower()
        if "unexpected asgi message" in msg and "websocket.send" in msg:
            return True
        if "websocket.close" in msg and "after" in msg:
            return True
        return False

    async def _send_json(data: dict) -> None:
        # Note: ``ws.state`` is Starlette's request-scoped :class:`starlette.datastructures.State`,
        # not a WebSocketState enum — never use it to gate sends.
        try:
            await ws.send_json(data)
        except Exception as e:
            if _send_failed_is_benign(e):
                logger.debug("voice_ws_send_skipped_closed", msg_type=data.get("type"))
                return
            logger.warning("voice_ws_send_failed", error=str(e), msg_type=data.get("type"))

    async def _handle(event: VoiceSessionEvent) -> None:
        """Forward session events to the browser over WebSocket."""
        if isinstance(event, SessionStarted):
            await _send_json({"type": "session_started"})
        elif isinstance(event, TranscriptPartial):
            await _send_json({"type": "transcript_partial", "text": event.text})
        elif isinstance(event, TranscriptCommitted):
            await _send_json({"type": "transcript_committed", "text": event.text})
        elif isinstance(event, AgentTextDelta):
            await _send_json({"type": "agent_text_delta", "text": event.text})
        elif isinstance(event, AgentTextDone):
            await _send_json({"type": "agent_text_done", "text": event.text})
        elif isinstance(event, AudioOutput):
            await _send_json(
                {
                    "type": "audio",
                    "data": base64.b64encode(event.data).decode("ascii"),
                }
            )
        elif isinstance(event, SessionInterrupted):
            await _send_json({"type": "interrupted"})
        elif isinstance(event, SessionError):
            await _send_json({"type": "error", "message": event.message})
        elif isinstance(event, SessionEnded):
            await _send_json(
                {
                    "type": "session_transcript",
                    "entries": [e.model_dump() for e in session.transcript],
                }
            )
            await _send_json({"type": "session_ended"})

    recv_task = asyncio.create_task(_recv_loop())
    try:
        async with aclosing(session.run(_mic_stream())) as event_iter:
            async for event in event_iter:
                await _handle(event)
    finally:
        if not recv_task.done():
            recv_task.cancel()
        await asyncio.gather(recv_task, return_exceptions=True)
        try:
            await session.close()
        except Exception as e:
            logger.debug("voice_session_close_suppressed", error=str(e))
        logger.info("voice_ws_disconnected")
