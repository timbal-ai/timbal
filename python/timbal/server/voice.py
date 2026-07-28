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
    cfg: dict[str, Any] = {
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
    return cfg


def _recording_config_from_env() -> dict[str, Any]:
    """Recording knobs the platform injects as env vars.

    Deliberately *not* part of :func:`default_voice_config_from_env`: that
    runs once at server boot, while serverless session boxes can be
    CRIU-restored from a warm snapshot with env arriving at restore time.
    Recording env must therefore be read per session (call sites are in
    :func:`build_voice_session`).
    """
    cfg: dict[str, Any] = {}
    if recording_dir := os.environ.get("TIMBAL_VOICE_RECORDING_DIR"):
        cfg["dir"] = recording_dir
    if layout := os.environ.get("TIMBAL_VOICE_RECORDING_LAYOUT"):
        cfg["layout"] = layout
    if bitrate := os.environ.get("TIMBAL_VOICE_RECORDING_BITRATE_KBPS"):
        cfg["bitrate_kbps"] = bitrate
    return cfg


# Platform identity env → manifest ``meta`` keys. Makes recording files
# self-describing for sweeper ingest and crash recovery (platform ask).
_RECORDING_IDENTITY_ENV = (
    ("TIMBAL_ORG_ID", "org_id"),
    ("TIMBAL_PROJECT_ID", "project_id"),
    ("TIMBAL_PROJECT_ENV_ID", "project_env_id"),
    ("TIMBAL_APP_ID", "app_id"),
    ("TIMBAL_PROJECT_REV", "project_rev"),
)


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
            # "local" is the default for non-Flux STT (see the session setup), and
            # the playground also switches to Smart Turn on first Start.
            detector = resolve_turn_detector("local")

        if isinstance(detector, LocalAudioTurnDetector):
            from ..voice.providers import AudioInputConfig

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


def select_turn_detector_spec(server_spec: Any, client_spec: Any, *, stt_is_flux: bool) -> Any:
    """Choose the turn detector for a session: a mode name, instance, or factory.

    ``server_spec`` comes from ``runnable.voice_config`` and may be any of those
    three. ``client_spec`` comes over the wire and is honoured only as a mode
    name — a browser must not be able to inject a callable — but is otherwise
    preferred, so the playground can A/B detectors.

    The choice is made here rather than in
    :func:`~timbal.voice.turn_detection.resolve_turn_detector` because it turns
    on the STT: whichever mode is best depends on how well the provider
    endpoints, and only the session setup knows which provider is in play.
    """
    spec = server_spec
    if isinstance(client_spec, str) and client_spec.strip():
        spec = client_spec
    elif client_spec is not None and not isinstance(client_spec, str):
        logger.warning("voice_ws_bad_turn_detector", error="client turn_detector must be a mode name string")

    mode = spec.strip().lower() if isinstance(spec, str) else None
    if stt_is_flux:
        # Flux owns EOU (~260ms). Local/lexical run *after* EndOfTurn and add a
        # second HOLD tax; they also disable the useful Provider path. Force
        # provider unless the caller explicitly picked heuristic/raw/provider.
        if mode is None or mode in ("local", "audio", "smart_turn", "lexical"):
            if spec is not None:
                # Includes an instance or factory from voice_config, which this
                # overrides too — worth saying so rather than silently ignoring.
                logger.info("voice_ws_flux_overrides_turn_detector", requested=str(spec), using="provider")
            return "provider"
        return spec
    if spec is not None:
        return spec

    # Nothing chosen. The holdless heuristic is the worst of the four on every
    # backend that endpoints on silence — 65-69% against 96-100% for detectors
    # that can hold — because Nova and ElevenLabs commit each fragment of a
    # paused utterance separately, and only a hold puts them back together.
    #
    # The warmup path already resolved "local" for this case, so a server with no
    # `turn_detector` configured was loading Smart Turn, Namo and Silero at
    # startup and then handing the session a detector that used none of them.
    #
    # Without `timbal[voice]`, `local` returns the heuristic decision verbatim
    # (its punctuation fallback sits behind the audio-EOU branch), so it would buy
    # nothing; `lexical` holds an unfinished transcript with no extra deps. Asking
    # the resolver beats probing imports: the degradation rule stays in one place.
    from ..voice.turn_detection import resolve_turn_detector

    mode = "local" if getattr(resolve_turn_detector("local"), "audio_eou", None) is not None else "lexical"
    logger.info("voice_ws_default_turn_detector", using=mode)
    return mode


def build_voice_session(
    runnable: Any,
    defaults: dict[str, Any],
    client_config: dict[str, Any],
    *,
    playback_tracker: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve voice config into a ``VoiceSession`` plus ``session_started`` metadata.

    Transport-agnostic: the WebSocket handler and the WebRTC route both call
    this with their own client config dict. ``playback_tracker`` lets a paced
    transport substitute its own position source for the default
    client-acked estimate.

    Returns ``(session, meta)`` where ``meta`` holds the resolved identity
    keys (``stt_provider``, ``stt_model``, ``model``, ``turn_detector``) that
    transports include in their ``session_started`` payload.
    """
    from ..voice import VoiceSession
    from ..voice.deepgram import (
        DeepgramFluxSTT,
        effective_stt_model,
        resolve_stt,
        stt_provider_id,
    )
    from ..voice.elevenlabs import ElevenLabsStreamTTS
    from ..voice.providers import AudioInputConfig, AudioOutputConfig
    from ..voice.turn_detection import resolve_turn_detector

    merged = merge_client_voice_overrides(defaults, client_config)

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
        stt_model_requested = None
    stt_is_flux = isinstance(stt, DeepgramFluxSTT)
    # Config id for clients/logs (``deepgram-flux``), not the class name.
    stt_provider = stt_provider_id(stt)
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
    # come over the wire. When neither picks one, the server chooses by STT
    # rather than taking ``resolve_turn_detector``'s zero-dep default, because
    # only here is the STT's endpointing behaviour known.
    turn_detector = None
    raw_td = select_turn_detector_spec(
        defaults.get("turn_detector"),
        client_config.get("turn_detector"),
        stt_is_flux=stt_is_flux,
    )
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
        "voice_session_config",
        stt=type(stt).__name__,
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
    session_kwargs: dict[str, Any] = {}
    if "turn_timeout_secs" in merged:
        try:
            session_kwargs["turn_timeout_secs"] = float(merged["turn_timeout_secs"])
        except (TypeError, ValueError):
            logger.warning("voice_ws_bad_turn_timeout_secs", value=repr(merged.get("turn_timeout_secs")))
    if "turn_timeout_fallback" in merged:
        fb = merged["turn_timeout_fallback"]
        session_kwargs["turn_timeout_fallback"] = None if fb is None else str(fb)
    if playback_tracker is not None:
        session_kwargs["playback_tracker"] = playback_tracker

    # Call recording is read from *server* config only — env (per session,
    # CRIU-safe) under ``runnable.voice_config["recording"]`` (user keys win)
    # — never from the merged dict: merge_client_voice_overrides applies
    # client keys freely, and a browser must not be able to switch recording
    # on or off.
    user_recording = defaults.get("recording")
    recording_cfg = {
        **_recording_config_from_env(),
        **(user_recording if isinstance(user_recording, dict) else {}),
    }
    if recording_cfg.get("dir"):
        try:
            from uuid_extensions import uuid7

            from ..voice.recording import CallRecorder

            on_saved = recording_cfg.get("on_saved")
            if on_saved is None and os.environ.get("TIMBAL_VOICE_RECORDING_UPLOAD") == "platform":
                from .recording_upload import platform_recording_upload_hook

                on_saved = platform_recording_upload_hook()
            session_id = uuid7(as_type="str").replace("-", "")
            session_kwargs["session_id"] = session_id
            session_kwargs["recorder"] = CallRecorder(
                Path(recording_cfg["dir"]) / f"{session_id}.mp3",
                sample_rate=int(merged.get("sample_rate", 16_000)),
                layout=recording_cfg.get("layout", "mixed"),
                bitrate_kbps=int(recording_cfg.get("bitrate_kbps", 32)),
                on_saved=on_saved,
                meta={k: v for env_key, k in _RECORDING_IDENTITY_ENV if (v := os.environ.get(env_key))} or None,
            )
        except ImportError:
            logger.warning("voice_recording_unavailable", hint="call recording requires timbal[voice] (av + numpy)")
        except Exception as e:
            # A misconfigured recorder must not take voice down with it.
            logger.error("voice_recording_setup_failed", error=str(e), exc_info=True)

    session = VoiceSession(
        agent=runnable,
        stt=stt,
        tts=tts,
        audio_input=audio_in,
        audio_output=audio_out,
        turn_detector=turn_detector,
        vad_endpointing=vad_endpointing,
        model=model_override,
        **session_kwargs,
    )
    meta = {
        "session_id": session.session_id,
        "stt_provider": stt_provider,
        "stt_model": stt_model,
        "model": llm_model,
        "turn_detector": turn_detector_label,
    }
    return session, meta


def event_to_payloads(event: Any, session: Any, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Map one ``VoiceSessionEvent`` to the JSON payloads a transport sends.

    Shared by the WebSocket handler and the WebRTC data channel — the wire
    format is identical. ``AudioOutput`` is serialized as base64 here; a paced
    transport (WebRTC) intercepts it *before* calling this and feeds the PCM
    to its audio track instead.

    ``meta`` is the dict from :func:`build_voice_session`, extended by the
    transport with its own keys (``transport``, ``playback_acks``).
    """
    from ..voice import (
        AgentStatus,
        AgentTextDelta,
        AgentTextDone,
        AudioOutput,
        SessionEnded,
        SessionError,
        SessionInterrupted,
        SessionStarted,
        TranscriptCommitted,
        TranscriptPartial,
        TurnMetricsEvent,
    )

    if isinstance(event, SessionStarted):
        return [
            {
                "type": "session_started",
                **meta,
                # The endpointer arms during session startup (before this event
                # is emitted), so this reflects the real state — not the
                # requested config.
                "vad_endpointing": session._endpointer is not None,
            }
        ]
    if isinstance(event, TranscriptPartial):
        return [{"type": "transcript_partial", "text": event.text}]
    if isinstance(event, TranscriptCommitted):
        payload: dict[str, Any] = {"type": "transcript_committed", "text": event.text}
        if event.replace:
            payload["replace"] = True
        return [payload]
    if isinstance(event, AgentStatus):
        return [{"type": "agent_status", "text": event.text}]
    if isinstance(event, AgentTextDelta):
        return [{"type": "agent_text_delta", "text": event.text}]
    if isinstance(event, AgentTextDone):
        return [{"type": "agent_text_done", "text": event.text}]
    if isinstance(event, AudioOutput):
        return [{"type": "audio", "data": base64.b64encode(event.data).decode("ascii")}]
    if isinstance(event, TurnMetricsEvent):
        return [{"type": "metrics", "metrics": event.metrics.model_dump()}]
    if isinstance(event, SessionInterrupted):
        return [{"type": "interrupted", "heard_text": event.heard_text}]
    if isinstance(event, SessionError):
        return [{"type": "error", "message": event.message}]
    if isinstance(event, SessionEnded):
        # Entries carry absolute wall-clock timestamps; offset_ms (relative to
        # started_at) saves every client the subtraction — it's what a
        # conversation-review UI actually renders.
        started_at = getattr(session, "started_at", None)
        entries = []
        for e in session.transcript:
            d = e.model_dump()
            if started_at is not None:
                d["offset_ms"] = max(0, round((e.timestamp - started_at) * 1000))
            entries.append(d)
        transcript_payload: dict[str, Any] = {"type": "session_transcript", "entries": entries}
        if started_at is not None:
            transcript_payload["started_at"] = started_at
        return [transcript_payload, {"type": "session_ended"}]
    return []


@router.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    from ..core.agent import Agent
    from ..voice import SessionInterrupted, VoiceSessionEvent

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
    session, meta = build_voice_session(runnable, defaults, config)
    meta = {"playback_acks": "recommended", "transport": "websocket", **meta}
    session.recording_meta = meta

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
        if isinstance(event, SessionInterrupted) and not warned_ack_degraded and not session.playback.ack_received:
            warned_ack_degraded = True
            logger.warning(
                "voice_ws_truncation_degraded",
                hint="client sent no playback acks before a barge-in; heard-text "
                "truncation used the wall-clock estimate. Send "
                '{"type": "playback", "played_ms": ...} every ~250ms while audio plays.',
            )
        for payload in event_to_payloads(event, session, meta):
            await _send_json(payload)

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
