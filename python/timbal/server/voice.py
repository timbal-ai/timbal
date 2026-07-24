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
import time
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

# Override with ELEVENLABS_VOICE_ID / TIMBAL_VOICE_ID (cloned/custom voices are account-specific).
_DEFAULT_VOICE_ID = "1SM7GgM6IMuvQlz2BwM3"


def default_voice_config_from_env() -> dict[str, Any]:
    """STT/TTS defaults for ``/voice/ws`` (ElevenLabs). Override with env or ``runnable.voice_config``."""
    return {
        "stt_provider": os.environ.get("TIMBAL_STT_PROVIDER", "elevenlabs"),
        "stt_model": os.environ.get("TIMBAL_STT_MODEL", "scribe_v2_realtime"),
        "tts_model": os.environ.get("TIMBAL_TTS_MODEL", "eleven_flash_v2_5"),
        "voice": (os.environ.get("ELEVENLABS_VOICE_ID") or os.environ.get("TIMBAL_VOICE_ID") or _DEFAULT_VOICE_ID),
        "language": os.environ.get("TIMBAL_VOICE_LANGUAGE", "es"),
        "sample_rate": 16_000,
        "stt_extra": {
            "commit_strategy": "vad",
            # 100ms is what ElevenLabs' own realtime examples use. 300ms made
            # short replies ("work.", "yes.") transcribe as partials but never
            # commit — the session then stalls until the user speaks again.
            "min_speech_duration_ms": 100,
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


def runnable_meta_for_voice_page(runnable: Any, import_spec: str) -> dict[str, Any]:
    """Serializable identity for the voice UI (same object as ``/run``)."""
    name = str(getattr(runnable, "name", "") or "").strip()
    kind = ""
    md = getattr(runnable, "metadata", None)
    if isinstance(md, dict) and md.get("type"):
        kind = str(md["type"])
    if not kind:
        kind = type(runnable).__name__
    model = getattr(runnable, "model", None)
    model_s = str(model).strip() if isinstance(model, str) else ""
    # Slim catalog for the playground model picker (from models.yaml via codegen).
    from ..codegen.model_discovery import get_models

    models = [
        {
            "id": m["id"],
            "provider": m["provider"],
            "display_name": m.get("display_name") or m["id"].split("/", 1)[-1],
        }
        for m in get_models()
    ]
    return {
        "name": name,
        "kind": kind,
        "import_spec": (import_spec or "").strip(),
        "model": model_s,
        "models": models,
    }


_VOICE_HTML_META_TOKEN = "__TIMBAL_VOICE_RUNNABLE_META_JSON__"


async def warmup_voice_stack(voice_config: dict[str, Any]) -> None:
    """Background warmup at server boot so the first voice session starts fast.

    Two tiers, both best-effort:

    * **Imports** (always): voice adapters (ElevenLabs + Deepgram) and the
      ``timbal[voice]`` extra (numpy/onnxruntime) — ~1s that otherwise lands
      on the first WebSocket connection.
    * **Models**: load Smart Turn + Namo + Silero when the server default
      turn detector is local (mode string **or** a ``LocalAudioTurnDetector``
      instance — demos often set the resolved instance on ``voice_config``).
      Eager-loads those ONNX models whenever the voice extra is installed so
      playground users who pick "Smart Turn" on first Start don't eat the
      HuggingFace cold path mid-handshake.
    """
    loop = asyncio.get_running_loop()

    def _import_stack() -> None:
        import importlib

        importlib.import_module("timbal.voice.elevenlabs")
        importlib.import_module("timbal.voice.deepgram")
        try:
            importlib.import_module("timbal.voice.smart_turn")
            importlib.import_module("timbal.voice.namo")
            importlib.import_module("timbal.voice.vad")
        except ImportError:
            pass  # timbal[voice] extra not installed — heuristics only

    try:
        await loop.run_in_executor(None, _import_stack)
    except Exception as e:
        logger.debug("voice_warmup_import_failed", error=str(e))
        return

    try:
        from ..voice.turn_detection import LocalAudioTurnDetector, resolve_turn_detector
        from ..voice.vad import SileroVad

        td = voice_config.get("turn_detector")
        detector = None
        if isinstance(td, LocalAudioTurnDetector):
            detector = td
        elif isinstance(td, str) and td.strip().lower() in ("local", "audio", "smart_turn"):
            detector = resolve_turn_detector(td)
        elif td is None:
            # Playground often switches to Smart Turn on first Start — warm it.
            detector = resolve_turn_detector("local")

        if isinstance(detector, LocalAudioTurnDetector):
            from ..voice.session import AudioInputConfig

            await detector.start(AudioInputConfig(sample_rate=16_000))
            await SileroVad().start(sample_rate=16_000)
            logger.info("voice_models_warmed")
    except Exception as e:
        logger.debug("voice_warmup_models_failed", error=str(e))


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
        AgentStatus,
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
        TurnMetricsEvent,
        VoiceSession,
        VoiceSessionEvent,
    )
    from ..voice.deepgram import DeepgramFluxSTT, effective_stt_model, resolve_stt
    from ..voice.elevenlabs import ElevenLabsStreamTTS

    await ws.accept()
    logger.info("voice_ws_connected")

    runnable = ws.app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error("voice_ws_rejected", reason="runnable is not an Agent", type=type(runnable).__name__)
        await ws.close(code=1008, reason="Voice requires an Agent runnable")
        return

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # Read the config hello. Protocol frames ("playback" acks, "audio",
    # "mic_change") can race ahead of it; the hello is the only JSON message
    # *without* a "type" field, so skip typed frames — however many arrive —
    # until the hello shows up or the 2s deadline passes (a swallowed hello
    # silently drops sample_rate / turn_detector overrides). A *binary* first
    # frame means a client that streams raw PCM without a hello: start with
    # defaults immediately instead of burning the deadline.
    config: dict = {}
    deadline = time.monotonic() + 2.0
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            first = await asyncio.wait_for(ws.receive(), timeout=remaining)
            if "text" in first and first["text"]:
                # Per-frame errors (invalid JSON, malformed audio payload) must
                # not end the scan: a valid hello may still be in flight.
                try:
                    data = json.loads(first["text"])
                except ValueError as e:
                    logger.warning("voice_ws_bad_handshake_frame", error=str(e))
                    continue
                if isinstance(data, dict) and data.get("type") is not None:
                    if data.get("type") == "audio":
                        try:
                            await audio_queue.put(base64.b64decode(data["data"]))
                        except (KeyError, TypeError, ValueError) as e:
                            logger.warning("voice_ws_bad_handshake_frame", error=str(e))
                    # Playback acks before any TTS audio carry no information.
                    continue
                config = data if isinstance(data, dict) else {}
            elif "bytes" in first and first["bytes"]:
                await audio_queue.put(first["bytes"])
            break
    except TimeoutError:
        pass
    except Exception as e:
        logger.warning("voice_ws_first_frame_error", error=str(e))

    defaults: dict = getattr(ws.app.state, "voice_config", None) or {}
    merged = merge_client_voice_overrides(defaults, config)

    stt_provider = merged.get("stt_provider")
    stt_model_requested = merged.get("stt_model")
    try:
        stt = resolve_stt(stt_provider, model=stt_model_requested)
    except ValueError as e:
        logger.warning(
            "voice_ws_bad_stt_provider",
            error=str(e),
            requested_provider=stt_provider,
            requested_model=stt_model_requested,
        )
        # Fallback must not keep a Flux/Nova model id on the ElevenLabs wire,
        # or the client/config log will claim Deepgram while Scribe runs.
        stt = resolve_stt("elevenlabs")
        stt_provider = "elevenlabs"
        stt_model_requested = None
    stt_is_flux = isinstance(stt, DeepgramFluxSTT)
    stt_label = type(stt).__name__
    stt_model = effective_stt_model(stt, stt_model_requested)
    tts = ElevenLabsStreamTTS()

    stt_extra = dict(merged.get("stt_extra", {}))
    if stt_is_flux:
        # Scribe-tuned VAD knobs don't apply to Flux's turn machine.
        for k in ("commit_strategy", "min_speech_duration_ms", "vad_silence_threshold_secs", "vad_threshold"):
            stt_extra.pop(k, None)
    audio_in = AudioInputConfig(
        model=stt_model,
        language=merged.get("language"),
        sample_rate=merged.get("sample_rate", 16_000),
        encoding=merged.get("encoding", "pcm_s16le"),
        extra=stt_extra,
    )
    audio_out = AudioOutputConfig(
        model=merged.get("tts_model"),
        voice=merged.get("voice"),
        sample_rate=merged.get("sample_rate", 16_000),
        encoding=merged.get("encoding", "pcm_s16le"),
        extra=merged.get("tts_extra", {}),
    )

    # ``runnable.voice_config`` may supply a TurnDetector instance, factory, or
    # a mode name ("heuristic"|"provider"|"local"|"lexical"). The client JSON
    # may additionally select a *mode name* (string only — useful for A/B
    # testing detectors from the playground); instances and factories can never
    # come over the wire.
    from ..voice.turn_detection import resolve_turn_detector

    turn_detector = None
    raw_td = defaults.get("turn_detector")
    client_td = config.get("turn_detector")
    if isinstance(client_td, str) and client_td.strip():
        raw_td = client_td
    elif client_td is not None and not isinstance(client_td, str):
        logger.warning("voice_ws_bad_turn_detector", error="client turn_detector must be a mode name string")
    if stt_is_flux:
        # Flux owns EOU (~260ms). Local/lexical run *after* EndOfTurn and add
        # a second HOLD tax; they also disable the useful Provider path. Force
        # provider unless the client explicitly picked heuristic/raw/provider.
        td_mode = raw_td.strip().lower() if isinstance(raw_td, str) else None
        if td_mode is None or td_mode in ("local", "audio", "smart_turn", "lexical"):
            if td_mode is not None:
                logger.info(
                    "voice_ws_flux_overrides_turn_detector",
                    requested=raw_td,
                    using="provider",
                )
            raw_td = "provider"
    if raw_td is not None:
        try:
            # voice_config is process-wide; VoiceSession clones the resolved
            # detector so concurrent connections never share buffers/lifecycle.
            turn_detector = resolve_turn_detector(raw_td)
        except (TypeError, ValueError) as e:
            logger.warning("voice_ws_bad_turn_detector", error=str(e))
    turn_detector_label = type(turn_detector).__name__ if turn_detector is not None else "HeuristicTurnDetector"

    # Optional bool: force the local VAD endpointing fast path on/off. Default
    # (absent / non-bool) is auto — on when the detector has an audio EOU model
    # and timbal[voice] is installed. Client hello may override the server value.
    vad_endpointing = merged.get("vad_endpointing")
    if not isinstance(vad_endpointing, bool):
        vad_endpointing = None
    if stt_is_flux:
        # Flux has no force-commit (commit() is a no-op); the Silero fast path
        # would just burn CPU scoring audio it can never act on.
        vad_endpointing = False

    # Playground / client may override the Agent's LLM for this session only.
    raw_model = merged.get("model")
    model_override = (
        raw_model.strip()
        if isinstance(raw_model, str) and "/" in raw_model.strip()
        else None
    )
    llm_model = model_override or (
        str(runnable.model) if isinstance(getattr(runnable, "model", None), str) else None
    )
    logger.info(
        "voice_ws_session_config",
        stt=stt_label,
        stt_provider=stt_provider,
        stt_model=stt_model,
        stt_model_requested=merged.get("stt_model"),
        model=llm_model,
        turn_detector=turn_detector_label,
        vad_endpointing="auto" if vad_endpointing is None else vad_endpointing,
    )

    # TODO(tool-filler): read a server-side `filler_phrases` voice_config key
    # (list of phrases or `(tool_name) -> str | None` callable) and pass it to
    # `VoiceSession(filler=...)` once tool-call filler speech returns.
    session = VoiceSession(
        agent=runnable,
        stt=stt,
        tts=tts,
        audio_input=audio_in,
        audio_output=audio_out,
        turn_detector=turn_detector,
        vad_endpointing=vad_endpointing,
        model=model_override,
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
                    elif data.get("type") == "playback":
                        # Cumulative ms of TTS audio the client actually played.
                        try:
                            session.playback.on_playback_ack(float(data["played_ms"]))
                        except (KeyError, TypeError, ValueError):
                            logger.debug("voice_ws_bad_playback_ack", data=str(data)[:120])
        except WebSocketDisconnect:
            pass
        finally:
            await audio_queue.put(b"")
            # Client is gone (disconnect / receive error): end the session now.
            # Without this the session lingers until the STT provider times out
            # on silence (~15s with ElevenLabs), delaying teardown and leaking
            # agent turns into the void. Idempotent when already closed.
            await session.close()

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

    # One-time per session: an interruption without any playback acks means the
    # heard-text truncation ran on the wall-clock estimate only.
    warned_ack_degraded = False

    async def _handle(event: VoiceSessionEvent) -> None:
        """Forward session events to the browser over WebSocket."""
        nonlocal warned_ack_degraded
        if isinstance(event, SessionStarted):
            # ``playback_acks`` advertises that the server understands
            # ``{"type": "playback", "played_ms": ...}`` and expects clients that
            # play audio to send them (see server/README.md).
            await _send_json(
                {
                    "type": "session_started",
                    "playback_acks": "recommended",
                    "stt_provider": stt_label,
                    "stt_model": stt_model,
                    "model": llm_model,
                    "turn_detector": turn_detector_label,
                    # The endpointer arms during session startup (before this
                    # event is emitted), so this reflects the real state — not
                    # the requested config.
                    "vad_endpointing": session._endpointer is not None,
                }
            )
        elif isinstance(event, TranscriptPartial):
            await _send_json({"type": "transcript_partial", "text": event.text})
        elif isinstance(event, TranscriptCommitted):
            payload: dict = {"type": "transcript_committed", "text": event.text}
            if event.replace:
                payload["replace"] = True
            await _send_json(payload)
        elif isinstance(event, AgentStatus):
            await _send_json({"type": "agent_status", "text": event.text})
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
        elif isinstance(event, TurnMetricsEvent):
            await _send_json({"type": "metrics", "metrics": event.metrics.model_dump()})
        elif isinstance(event, SessionInterrupted):
            if not warned_ack_degraded and not session.playback.ack_received:
                warned_ack_degraded = True
                logger.warning(
                    "voice_ws_truncation_degraded",
                    hint="client sent no playback acks before a barge-in; heard-text "
                    "truncation used the wall-clock estimate. Send "
                    '{"type": "playback", "played_ms": ...} every ~250ms while audio plays.',
                )
            await _send_json({"type": "interrupted", "heard_text": event.heard_text})
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
