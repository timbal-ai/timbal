"""Voice routes — browser-based voice session over WebSocket.

Serves ``GET /voice`` (HTML) and ``/voice/ws`` for the same runnable as ``/run``.
Defaults come from :func:`default_voice_config_from_env` and optional
``runnable.voice_config`` (dict, callable, or :class:`VoiceConfig`); the client
can override allowlisted keys with a JSON first message on the socket.

Heavy imports (``VoiceSession``, ElevenLabs) load on first WebSocket connection
only — ``timbal.voice.config`` itself is import-light.
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
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from .. import __version__ as timbal_version
from .._slots import dump_value
from ..voice.ambience import PRESETS as AMBIENT_PRESETS
from ..voice.ambience import ensure_ambient_source
from ..voice.config import (
    DEFAULT_VOICE_ID,
    FillerConfig,
    GreetingConfig,
    RecordingConfig,
    VoiceConfig,
)
from .capacity import acquire_session_slot, release_session_slot

logger = structlog.get_logger("timbal.server.voice")

router = APIRouter(prefix="/voice", tags=["voice"])

_HTML_PATH = Path(__file__).parent / "voice.html"


def default_voice_config_from_env() -> VoiceConfig:
    """:class:`VoiceConfig` defaults, with env overrides where set."""
    kwargs: dict[str, Any] = {}
    if v := os.environ.get("TIMBAL_STT_PROVIDER"):
        kwargs["stt_provider"] = v
    if v := os.environ.get("TIMBAL_STT_MODEL"):
        kwargs["stt_model"] = v
    if v := os.environ.get("TIMBAL_TTS_MODEL"):
        kwargs["tts_model"] = v
    if v := os.environ.get("ELEVENLABS_VOICE_ID") or os.environ.get("TIMBAL_VOICE_ID"):
        kwargs["voice"] = v
    if v := os.environ.get("TIMBAL_VOICE_LANGUAGE"):
        kwargs["language"] = v
    # Plain text only: the env channel cannot express the rest of
    # GreetingConfig, and a fixed opener is the path worth having here (the
    # platform injects the line it wants a deployed number answered with).
    if v := os.environ.get("TIMBAL_VOICE_GREETING"):
        kwargs["greeting"] = v
    if v := os.environ.get("TIMBAL_VOICE_AMBIENT_SOURCE"):
        kwargs["ambient"] = {"source": v}
        if vol := os.environ.get("TIMBAL_VOICE_AMBIENT_VOLUME"):
            kwargs["ambient"]["volume"] = vol
    filler: dict[str, Any] = {}
    if v := os.environ.get("TIMBAL_VOICE_FILLER_SYSTEM_PROMPT"):
        filler["system_prompt"] = v
    if v := os.environ.get("TIMBAL_VOICE_FILLER_MODEL"):
        filler["model"] = v
    if v := os.environ.get("TIMBAL_VOICE_FILLER_DELAY_SECS"):
        filler["delay_secs"] = v
    if v := os.environ.get("TIMBAL_VOICE_FILLER_REPEAT_SECS"):
        filler["repeat_secs"] = v
    # TIMBAL_VOICE_FILLER=1 enables with defaults; any detail var implies it.
    enabled = os.environ.get("TIMBAL_VOICE_FILLER", "").strip().lower()
    if enabled in ("1", "true", "yes", "on") or (filler and enabled not in ("0", "false", "no", "off")):
        kwargs["filler"] = filler
    return VoiceConfig(**kwargs)


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


def merge_voice_config(runnable: Any) -> VoiceConfig:
    """Env defaults, then optional ``runnable.voice_config`` (dict, callable, or ``VoiceConfig``).

    Validated strictly: an unknown key or bad value raises at server boot
    instead of silently falling back to defaults on the first call.
    """
    base = default_voice_config_from_env()
    vc = getattr(runnable, "voice_config", None)
    if callable(vc):
        vc = vc()
    if isinstance(vc, VoiceConfig):
        dumped = vc.model_dump(include=vc.model_fields_set)
        # Top-level ``include`` dumps nested models in full; redo filler and
        # greeting sparsely so unset fields don't clobber env values below.
        if isinstance(vc.filler, FillerConfig):
            dumped["filler"] = vc.filler.model_dump(include=vc.filler.model_fields_set)
        if isinstance(vc.greeting, GreetingConfig):
            dumped["greeting"] = vc.greeting.model_dump(include=vc.greeting.model_fields_set)
        vc = dumped
    if not isinstance(vc, dict):
        return base
    skip = frozenset({"stt_extra", "tts_extra", "filler", "greeting"})
    data = {
        **base.model_dump(),
        **{k: v for k, v in vc.items() if v is not None and k not in skip},
    }
    if isinstance(vc.get("stt_extra"), dict):
        data["stt_extra"] = {**base.stt_extra, **vc["stt_extra"]}
    if isinstance(vc.get("tts_extra"), dict):
        data["tts_extra"] = {**base.tts_extra, **vc["tts_extra"]}
    filler = vc.get("filler")
    if isinstance(filler, FillerConfig):
        filler = filler.model_dump(include=filler.model_fields_set)
    if isinstance(filler, dict):
        base_filler = base.filler.model_dump(include=base.filler.model_fields_set) if base.filler else {}
        data["filler"] = {**base_filler, **filler}
    elif filler is not None:
        data["filler"] = filler
    greeting = vc.get("greeting")
    if isinstance(greeting, GreetingConfig):
        greeting = greeting.model_dump(include=greeting.model_fields_set)
    if isinstance(greeting, dict):
        # Same deep merge as filler: an agent tweaking ``delay_ms`` must not drop
        # the text TIMBAL_VOICE_GREETING supplied (and lose the whole opener to a
        # "needs text or instructions" boot failure).
        base_greeting = base.greeting.model_dump(include=base.greeting.model_fields_set) if base.greeting else {}
        data["greeting"] = {**base_greeting, **greeting}
    elif greeting is not None:
        # Includes ``""``, which the field validator reads as "no opener" — how
        # an agent switches off one the environment configured.
        data["greeting"] = greeting
    return VoiceConfig(**data)


# Keys a browser may override via the WS/RTC hello. Everything else —
# ``recording`` above all — is server policy. ``turn_detector`` is negotiated
# separately in :func:`select_turn_detector_spec` (mode names only, never
# callables). ``model`` is deliberately client-settable for the playground
# model picker; trim this set in deployments where the client is untrusted.
CLIENT_SETTABLE_VOICE_FIELDS = frozenset({
    "stt_provider",
    "stt_model",
    "tts_provider",
    "tts_model",
    "voice",
    "language",
    "sample_rate",
    "encoding",
    "stt_extra",
    "tts_extra",
    "vad_endpointing",
    "model",
    "turn_timeout_secs",
    "turn_timeout_fallback",
    "filler",
    # Per-call opener: the telephony webhook's <Parameter name="greeting"> and
    # the LiveKit hello both arrive through this allowlist, and "what this call
    # opens with" is per-call by nature (an outbound campaign sets it per dial).
    "greeting",
})


def merge_client_voice_overrides(server_defaults: VoiceConfig, client: dict[str, Any]) -> VoiceConfig:
    """Apply the optional first WebSocket JSON message over the server config.

    Allowlist-filtered. Values are deliberately NOT re-validated here — client
    input gets per-key guards with fallbacks in :func:`build_voice_session`,
    so one bad value degrades that knob instead of closing the socket.
    ``filler`` and ``greeting`` are the exceptions: they're nested models, so
    they are deep-merged over the server default (a client tweaking
    ``delay_secs`` keeps the server's custom ``system_prompt``) and validated
    here — invalid → keep server's.
    """
    updates = {k: v for k, v in client.items() if k in CLIENT_SETTABLE_VOICE_FIELDS and v is not None}
    # ``turn_detector``, ``call_context`` and ``parent_id`` are read straight
    # off the hello by the transport, not through VoiceConfig — reporting them
    # as ignored would be a lie in both directions.
    ignored = sorted(
        k for k, v in client.items()
        if v is not None and k not in CLIENT_SETTABLE_VOICE_FIELDS and k not in ("turn_detector", "call_context", "parent_id")
    )
    if ignored:
        logger.info("voice_client_config_ignored", keys=ignored)
    if "filler" in updates:
        base = server_defaults.filler
        merged_filler = {
            **(base.model_dump(include=base.model_fields_set) if base is not None else {}),
            **(updates["filler"] if isinstance(updates["filler"], dict) else {}),
        }
        try:
            updates["filler"] = FillerConfig.model_validate(merged_filler)
        except ValidationError:
            logger.info("voice_client_filler_invalid", value=repr(updates["filler"]))
            del updates["filler"]
    if "greeting" in updates:
        _merge_client_greeting(server_defaults, updates)
    return server_defaults.model_copy(update=updates)


def _merge_client_greeting(server_defaults: VoiceConfig, updates: dict[str, Any]) -> None:
    """Resolve the client's ``greeting`` override in place.

    ``model_copy`` below runs no validators, so the coercion the ``VoiceConfig``
    field does for a bare string has to happen here too — and a bare string is
    the only spelling the telephony ``<Parameter>`` channel can send. It is
    treated as a *text* override so the server keeps owning policy
    (``interruptible``, ``delay_ms``); ``""`` switches the opener off for this
    call, which is the one way a client can subtract a nested default.
    """
    base = server_defaults.greeting
    base_data = base.model_dump(include=base.model_fields_set) if base is not None else {}
    raw = updates["greeting"]
    if isinstance(raw, str):
        if not raw.strip():
            updates["greeting"] = None
            return
        patch: Any = {"text": raw}
    elif isinstance(raw, GreetingConfig):
        patch = raw.model_dump(include=raw.model_fields_set)
    else:
        patch = raw
    if not isinstance(patch, dict):
        logger.info("voice_client_greeting_invalid", value=repr(raw))
        del updates["greeting"]
        return
    try:
        updates["greeting"] = GreetingConfig.model_validate({**base_data, **patch})
    except ValidationError:
        logger.info("voice_client_greeting_invalid", value=repr(raw))
        del updates["greeting"]


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
        # Lets the page enable its call-context field instead of offering a
        # control the server would silently drop.
        "allow_client_call_context": client_call_context_allowed(),
    }


_VOICE_HTML_META_TOKEN = "__TIMBAL_VOICE_RUNNABLE_META_JSON__"


_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY = frozenset({"0", "false", "f", "no", "n", "off"})


def client_call_context_allowed() -> bool:
    """Whether a browser may seed ``call_context`` over the hello.

    Off by default, and it must stay that way anywhere real: call context is
    caller *identity* (``rep_id``, ``from``) that a callable ``system_prompt``
    trusts, and on telephony it is established by the webhook — not by whoever
    is on the far end. ``TIMBAL_VOICE_ALLOW_CLIENT_CALL_CONTEXT=1`` opens it so
    the playground can exercise an identity-driven prompt without placing a
    real call.
    """
    return os.environ.get("TIMBAL_VOICE_ALLOW_CLIENT_CALL_CONTEXT", "").strip().lower() in _TRUTHY


def client_call_context(config: dict[str, Any]) -> dict[str, str]:
    """Read the hello's ``call_context``, or ``{}`` when the door is shut."""
    raw = config.get("call_context")
    if not isinstance(raw, dict) or not raw:
        return {}
    if not client_call_context_allowed():
        logger.info("voice_client_call_context_ignored", keys=sorted(map(str, raw)))
        return {}
    out = {str(k): str(v) for k, v in raw.items() if isinstance(v, str | int | float) and str(v)}
    if out:
        logger.warning(
            "voice_client_call_context_accepted",
            keys=sorted(out),
            hint="TIMBAL_VOICE_ALLOW_CLIENT_CALL_CONTEXT is a development switch — the caller is setting their own identity",
        )
    return out


def client_parent_run_id(config: dict[str, Any]) -> str | None:
    """Read the hello's ``parent_id``, or ``None`` when the door is shut.

    A parent run id is not a voice knob — it says which conversation this
    caller is joining, and a browser that can set it can attach itself to
    somebody else's thread and read its memory. In production it must ride the
    server-minted dial; the hello path exists for the playground and sits
    behind the same gate as ``call_context``, deliberately — both are the
    caller asserting its own identity, and one policy switch is enough.
    """
    raw = config.get("parent_id")
    if not isinstance(raw, str) or not raw.strip():
        return None
    if not client_call_context_allowed():
        logger.info("voice_client_parent_id_ignored")
        return None
    logger.warning(
        "voice_client_parent_id_accepted",
        parent_id=raw.strip(),
        hint="TIMBAL_VOICE_ALLOW_CLIENT_CALL_CONTEXT is a development switch — the caller is choosing the thread it joins",
    )
    return raw.strip()


def voice_warmup_intended(runnable: Any) -> bool:
    """Whether server boot should pre-import the voice stack and pre-load ONNX models.

    Historically this ran for every Agent app whenever the ``timbal[voice]``
    extra was installed, which meant non-voice deployments (e.g. platform
    images built from ``timbal[all]``) downloaded and loaded Smart Turn +
    Namo + Silero at boot for nothing.

    Policy, in order:

    1. ``TIMBAL_VOICE_WARMUP`` env: truthy forces warmup (the playground
       launcher sets this for its child servers), falsy disables it.
    2. The runnable declares ``voice_config`` — clearly a voice app.
    3. Any ``TIMBAL_VOICE_*`` / ``ELEVENLABS_VOICE_ID`` env is set — the
       deployment is voice-configured even if the runnable isn't.
    4. Otherwise: no warmup.
    """
    override = os.environ.get("TIMBAL_VOICE_WARMUP", "").strip().lower()
    if override in _TRUTHY:
        return True
    if override in _FALSY:
        return False
    if getattr(runnable, "voice_config", None) is not None:
        return True
    if os.environ.get("ELEVENLABS_VOICE_ID"):
        return True
    return any(k.startswith("TIMBAL_VOICE_") for k in os.environ)


def voice_onnx_warmup_intended(voice_config: VoiceConfig) -> bool:
    """Whether boot should pre-load Smart Turn / Namo / Silero.

    Same detector choice the session will make
    (:func:`select_turn_detector_spec`). Native-EOU STT (Flux, Munsit) forces
    the provider turn detector, so those ONNX models never run — loading them
    anyway races the first turn with HuggingFace downloads on a cold box.

    Can be *heavy*: with ``turn_detector=None`` the spec resolution builds the
    default detector, importing onnxruntime/transformers. Call it off the
    event loop (``warmup_voice_stack`` runs it in the import executor).
    """
    from ..voice.deepgram import resolve_stt
    from ..voice.turn_detection import LocalAudioTurnDetector

    try:
        stt = resolve_stt(voice_config.stt_provider, model=voice_config.stt_model)
    except ValueError:
        stt = None
    spec = select_turn_detector_spec(
        voice_config.turn_detector,
        None,
        stt_native_eou=bool(getattr(stt, "native_eou", False)),
    )
    if isinstance(spec, LocalAudioTurnDetector):
        return True
    if isinstance(spec, str) and spec.strip().lower() in ("local", "audio", "smart_turn"):
        return True
    return False


async def warmup_voice_stack(voice_config: VoiceConfig, *, livekit: bool | None = None) -> None:
    """Background warmup at server boot so the first voice session starts fast.

    Three tiers, all best-effort:

    * **Imports** (always): voice adapters (ElevenLabs + Deepgram). The
      ``timbal[voice]`` extra (numpy/onnxruntime + Smart Turn/Namo/Silero)
      is imported only when those ONNX models will actually run.
    * **LiveKit FFI**: ``livekit.rtc`` loads a native library on first import,
      which otherwise lands inside the first dial's join budget (see
      ``_JOIN_TIMEOUT_SECS`` in :mod:`timbal.server.livekit_session`).
      ``livekit=None`` auto-detects from the env: ``TIMBAL_VOICE_TRANSPORT``
      (serverless boot-env dials) or ``TIMBAL_LIVEKIT_URL`` (long-lived
      servers pin the dialable URL — the composer sidecar gets it from the
      monolith supervisor). Pass a bool to override either way.
    * **Models**: load Smart Turn + Namo + Silero when the session's turn
      detector is local. Skipped for Flux / ``provider`` EOU — see
      :func:`voice_onnx_warmup_intended`.
    """
    if livekit is None:
        livekit = (
            os.environ.get("TIMBAL_VOICE_TRANSPORT", "").strip().lower() == "livekit"
            or bool(os.environ.get("TIMBAL_LIVEKIT_URL", "").strip())
        )
    loop = asyncio.get_running_loop()

    def _import_stack() -> bool:
        import importlib

        importlib.import_module("timbal.voice.elevenlabs")
        importlib.import_module("timbal.voice.deepgram")
        if livekit:
            try:
                importlib.import_module("livekit.rtc")
            except ImportError:
                pass  # timbal[voice-livekit] not installed — the dial will 501
        # The detector-choice probe can itself import onnxruntime/transformers
        # (default turn_detector resolution) — must stay in this executor, off
        # the event loop and under the caller's except.
        if not voice_onnx_warmup_intended(voice_config):
            return False
        try:
            importlib.import_module("timbal.voice.smart_turn")
            importlib.import_module("timbal.voice.namo")
            importlib.import_module("timbal.voice.vad")
        except ImportError:
            pass  # timbal[voice] extra not installed — heuristics only
        return True

    try:
        load_onnx = await loop.run_in_executor(None, _import_stack)
    except Exception as e:
        logger.debug("voice_warmup_import_failed", error=str(e))
        return

    if not load_onnx:
        logger.info("voice_warmup_skip_onnx", reason="provider_eou")
        return

    try:
        from ..voice.turn_detection import LocalAudioTurnDetector, resolve_turn_detector
        from ..voice.vad import SileroVad

        td = voice_config.turn_detector
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


@router.get("/meta")
async def voice_meta(request: Request) -> dict[str, Any]:
    """Runnable identity + model catalog as JSON.

    Same payload as the blob injected into the served HTML — this endpoint
    exists for the *standalone* playground (opened from the launcher or the
    filesystem), which can't get the injection and fetches it cross-origin
    from whatever local server it's pointed at.
    """
    runnable = getattr(request.app.state, "runnable", None)
    import_spec = os.environ.get("TIMBAL_RUNNABLE", "")
    meta = (
        runnable_meta_for_voice_page(runnable, import_spec)
        if runnable is not None
        else {"name": "", "kind": "", "import_spec": import_spec}
    )
    meta["version"] = timbal_version
    return meta


@router.get("/ambience")
async def ambience_index() -> dict[str, Any]:
    """Preset catalog for the playground picker."""
    return {"presets": sorted(AMBIENT_PRESETS)}


@router.get("/ambience/current")
async def ambience_current(request: Request) -> FileResponse:
    """The track configured on this server (preset or custom file).

    This is the only way a custom server-side file reaches the browser —
    clients never send paths.
    """
    cfg = getattr(request.app.state, "voice_config", None)
    ambient = cfg.ambient if cfg is not None else None
    if ambient is None:
        raise HTTPException(status_code=404, detail="No ambient audio configured")
    return FileResponse(await _ambient_path(ambient.source))


@router.get("/ambience/{name}")
async def ambience_preset(name: str) -> FileResponse:
    """Known presets only — never an arbitrary path."""
    name = name.strip().lower()
    if name not in AMBIENT_PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown ambience preset {name!r}")
    return FileResponse(await _ambient_path(name))


async def _ambient_path(source: str) -> Path:
    """Presets download lazily from the CDN into ``~/.cache/timbal/ambience``."""
    try:
        return await asyncio.to_thread(ensure_ambient_source, source)
    except Exception as e:
        logger.warning("ambience_fetch_failed", source=source, error=str(e))
        raise HTTPException(status_code=502, detail=f"Ambience source unavailable: {e}") from e


def select_turn_detector_spec(server_spec: Any, client_spec: Any, *, stt_native_eou: bool) -> Any:
    """Choose the turn detector for a session: a mode name, instance, or factory.

    ``server_spec`` comes from ``runnable.voice_config`` and may be any of those
    three. ``client_spec`` comes over the wire and is honoured only as a mode
    name — a browser must not be able to inject a callable — but is otherwise
    preferred, so the playground can A/B detectors.

    The choice is made here rather than in
    :func:`~timbal.voice.turn_detection.resolve_turn_detector` because it turns
    on the STT: whichever mode is best depends on how well the provider
    endpoints, and only the session setup knows which provider is in play.

    ``stt_native_eou`` is the provider's
    :attr:`~timbal.voice.providers.SpeechToText.native_eou` capability —
    Deepgram Flux and Munsit today.
    """
    spec = server_spec
    if isinstance(client_spec, str) and client_spec.strip():
        spec = client_spec
    elif client_spec is not None and not isinstance(client_spec, str):
        logger.warning("voice_ws_bad_turn_detector", error="client turn_detector must be a mode name string")

    mode = spec.strip().lower() if isinstance(spec, str) else None
    if stt_native_eou:
        # The provider owns EOU (Flux TurnInfo ~260ms; Munsit endpointing +
        # smart_turn). Local/lexical run *after* the provider's turn end and
        # add a second HOLD tax; they also disable the useful Provider path.
        # Force provider unless the caller explicitly picked heuristic/raw/
        # provider.
        if mode is None or mode in ("local", "audio", "smart_turn", "lexical"):
            if spec is not None:
                # Includes an instance or factory from voice_config, which this
                # overrides too — worth saying so rather than silently ignoring.
                logger.info("voice_ws_native_eou_overrides_turn_detector", requested=str(spec), using="provider")
            return "provider"
        return spec
    if spec is not None:
        return spec

    # Nothing chosen — same rule as ``resolve_turn_detector(None)``: local when
    # the voice extra is present, lexical otherwise. Never the holdless heuristic.
    from ..voice.turn_detection import resolve_turn_detector

    resolved = resolve_turn_detector(None)
    mode = "local" if getattr(resolved, "audio_eou", None) is not None else "lexical"
    logger.info("voice_ws_default_turn_detector", using=mode)
    return mode


def build_voice_session(
    runnable: Any,
    defaults: VoiceConfig,
    client_config: dict[str, Any],
    *,
    playback_tracker: Any = None,
    call_context: dict[str, str] | None = None,
    parent_run_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve voice config into a ``VoiceSession`` plus ``session_started`` metadata.

    Transport-agnostic: the WebSocket handler and the WebRTC route both call
    this with their own client config dict. ``playback_tracker`` lets a paced
    transport substitute its own position source for the default
    client-acked estimate. ``call_context`` is per-call identity (``rep_id``,
    ``from``, …) that a callable ``system_prompt`` reads off the session bag —
    deliberately separate from ``client_config``, which is voice knobs only.
    ``parent_run_id`` is the run this call continues (text → voice): session
    identity like ``call_context``, minted by whoever authorized the call.

    Returns ``(session, meta)`` where ``meta`` holds the resolved identity
    keys (``stt_provider``, ``stt_model``, ``model``, ``turn_detector``) that
    transports include in their ``session_started`` payload.
    """
    from ..voice import VoiceSession
    from ..voice.deepgram import (
        effective_stt_model,
        resolve_stt,
        stt_provider_id,
    )
    from ..voice.providers import AudioInputConfig, AudioOutputConfig, resolve_tts
    from ..voice.turn_detection import resolve_turn_detector

    merged = merge_client_voice_overrides(defaults, client_config)

    stt_provider = merged.stt_provider
    stt_model_requested = merged.stt_model
    try:
        stt = resolve_stt(stt_provider, model=stt_model_requested)
    except ValueError as e:
        logger.warning(
            "voice_ws_bad_stt_provider",
            error=str(e),
            requested_provider=stt_provider,
            requested_model=stt_model_requested,
        )
        # Fallback must not keep a Flux/Nova/Munsit model id on the ElevenLabs
        # wire, or the client/config log will claim another provider while
        # Scribe runs.
        stt = resolve_stt("elevenlabs")
        stt_model_requested = None
    stt_native_eou = bool(getattr(stt, "native_eou", False))
    # Config id for clients/logs (``deepgram-flux``), not the class name.
    stt_provider = stt_provider_id(stt)
    stt_model = effective_stt_model(stt, stt_model_requested)

    tts_model_requested = merged.tts_model
    tts_voice_requested = merged.voice
    try:
        tts = resolve_tts(merged.tts_provider)
    except ValueError as e:
        logger.warning("voice_ws_bad_tts_provider", error=str(e), requested_provider=merged.tts_provider)
        # Same rule as the STT fallback: a Munsit/Fish model id or voice must
        # not survive onto the ElevenLabs wire, or the socket fails on a config
        # the session reports as ElevenLabs. The voice is a required *path*
        # segment there, so substitute the default rather than clearing it.
        tts = resolve_tts("elevenlabs")
        tts_model_requested = None
        tts_voice_requested = DEFAULT_VOICE_ID
    # Config id for clients/logs (``fishaudio``), not the class name.
    tts_provider = tts.provider_id

    # Client extras are unvalidated (model_copy in the merge): tolerate a
    # non-dict rather than 500-ing the socket.
    stt_extra = dict(merged.stt_extra) if isinstance(merged.stt_extra, dict) else {}
    tts_extra = dict(merged.tts_extra) if isinstance(merged.tts_extra, dict) else {}
    if stt_native_eou:
        # Scribe-tuned VAD knobs don't apply to a provider-side turn machine
        # (Flux, Munsit).
        for k in ("commit_strategy", "min_speech_duration_ms", "vad_silence_threshold_secs", "vad_threshold"):
            stt_extra.pop(k, None)
    audio_in = AudioInputConfig(
        model=stt_model,
        language=merged.language,
        sample_rate=merged.sample_rate,
        encoding=merged.encoding,
        extra=stt_extra,
    )
    audio_out = AudioOutputConfig(
        model=tts_model_requested,
        voice=tts_voice_requested,
        sample_rate=merged.sample_rate,
        encoding=merged.encoding,
        extra=tts_extra,
    )

    # ``runnable.voice_config`` may supply a TurnDetector instance, factory, or
    # a mode name ("heuristic"|"provider"|"local"|"lexical"). The client JSON
    # may additionally select a *mode name* (string only — useful for A/B
    # testing detectors from the playground); instances and factories can never
    # come over the wire. When neither picks one, the server chooses by STT
    # (native EOU → provider; else the resolver default — local / lexical).
    turn_detector = None
    raw_td = select_turn_detector_spec(
        defaults.turn_detector,
        client_config.get("turn_detector"),
        stt_native_eou=stt_native_eou,
    )
    try:
        # voice_config is process-wide; VoiceSession clones the resolved
        # detector so concurrent connections never share buffers/lifecycle.
        # ``None`` / bad specs fall through to the resolver default (local).
        turn_detector = resolve_turn_detector(raw_td)
    except (TypeError, ValueError) as e:
        logger.warning("voice_ws_bad_turn_detector", error=str(e))
        turn_detector = resolve_turn_detector(None)
    turn_detector_label = type(turn_detector).__name__

    # Optional bool: force the local VAD endpointing fast path on/off. Default
    # (absent / non-bool) is auto — on when the detector has an audio EOU model
    # and timbal[voice] is installed. Client hello may override the server value.
    vad_endpointing = merged.vad_endpointing
    if not isinstance(vad_endpointing, bool):
        vad_endpointing = None
    if stt_native_eou:
        # Flux/Munsit have no force-commit (commit() is a no-op); the Silero
        # fast path would just burn CPU scoring audio it can never act on.
        vad_endpointing = False

    # Playground / client may override the Agent's LLM for this session only.
    raw_model = merged.model
    model_override = raw_model.strip() if isinstance(raw_model, str) and "/" in raw_model.strip() else None
    llm_model = model_override or (str(runnable.model) if isinstance(getattr(runnable, "model", None), str) else None)
    logger.info(
        "voice_session_config",
        stt=type(stt).__name__,
        stt_provider=stt_provider,
        stt_model=stt_model,
        stt_model_requested=merged.stt_model,
        tts=type(tts).__name__,
        tts_provider=tts_provider,
        model=llm_model,
        turn_detector=turn_detector_label,
        vad_endpointing="auto" if vad_endpointing is None else vad_endpointing,
        greeting=(None if merged.greeting is None else ((merged.greeting.text or "")[:80] or "<generated>")),
    )

    session_kwargs: dict[str, Any] = {}
    if merged.filler is not None and merged.filler.enabled:
        session_kwargs["filler"] = merged.filler
    if merged.greeting is not None:
        session_kwargs["greeting"] = merged.greeting
    if merged.turn_timeout_secs is not None:
        try:
            session_kwargs["turn_timeout_secs"] = float(merged.turn_timeout_secs)
        except (TypeError, ValueError):
            logger.warning("voice_ws_bad_turn_timeout_secs", value=repr(merged.turn_timeout_secs))
    if merged.turn_timeout_fallback is not None:
        # "" disables the spoken apology; None means "VoiceSession default".
        session_kwargs["turn_timeout_fallback"] = str(merged.turn_timeout_fallback)
    if playback_tracker is not None:
        session_kwargs["playback_tracker"] = playback_tracker
    if call_context:
        session_kwargs["call_context"] = call_context
    if parent_run_id:
        session_kwargs["parent_run_id"] = parent_run_id

    # Call recording is read from *server* config only — env (per session,
    # CRIU-safe) under ``runnable.voice_config["recording"]`` (user keys win).
    # ``recording`` is not in CLIENT_SETTABLE_VOICE_FIELDS: a browser must not
    # be able to switch recording on or off.
    user_recording = defaults.recording
    recording_data = {
        **_recording_config_from_env(),
        **(user_recording.model_dump(include=user_recording.model_fields_set) if user_recording else {}),
    }
    if recording_data.get("dir"):
        try:
            from uuid_extensions import uuid7

            from ..voice.recording import CallRecorder

            recording_cfg = RecordingConfig(**recording_data)
            on_saved = recording_cfg.on_saved
            if on_saved is None and os.environ.get("TIMBAL_VOICE_RECORDING_UPLOAD") == "platform":
                from .recording_upload import platform_recording_upload_hook

                on_saved = platform_recording_upload_hook()
            session_id = uuid7(as_type="hex")
            session_kwargs["session_id"] = session_id
            session_kwargs["recorder"] = CallRecorder(
                Path(recording_cfg.dir) / f"{session_id}.mp3",
                sample_rate=int(merged.sample_rate),
                layout=recording_cfg.layout,
                bitrate_kbps=recording_cfg.bitrate_kbps,
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
        "tts_provider": tts_provider,
        "model": llm_model,
        "turn_detector": turn_detector_label,
        # Server config, not client-settable. Phase 1: the browser mixes this
        # locally (fetch /voice/ambience/current); nothing is mixed server-side.
        "ambient": (
            {"source": defaults.ambient.source, "volume": defaults.ambient.volume} if defaults.ambient else None
        ),
    }
    if session.parent_run_id:
        # Lets the client confirm which conversation this call joined.
        meta["parent_run_id"] = session.parent_run_id
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
        AgentApproval,
        AgentInteraction,
        AgentStatus,
        AgentTextDelta,
        AgentTextDone,
        AudioOutput,
        FillerSpoken,
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
    if isinstance(event, AgentInteraction | AgentApproval):
        # ``input``/``payload`` are ``Any`` — the runnable's validated input can hold
        # a File or any other model. ``dump_value`` in json mode stringifies whatever
        # json.dumps would choke on, so a transport can never fail to send a
        # suspension (which would look exactly like not emitting one). Same coercion
        # the SSE path applies to the underlying run event.
        return [dump_value(event.model_dump(), mode="json")]
    if isinstance(event, FillerSpoken):
        return [{"type": "filler", "text": event.text}]
    if isinstance(event, AgentTextDelta):
        return [{"type": "agent_text_delta", "text": event.text}]
    if isinstance(event, AgentTextDone):
        # run_id → parent_id on POST /stream continues this conversation on
        # another transport. None for text with no run behind it (greeting,
        # realtime) — kept in the payload so the shape is stable.
        return [{"type": "agent_text_done", "text": event.text, "run_id": event.run_id}]
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

    await ws.accept()
    logger.info("voice_ws_connected")

    runnable = ws.app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error("voice_ws_rejected", reason="runnable is not an Agent", type=type(runnable).__name__)
        await ws.close(code=1008, reason="Voice requires an Agent runnable")
        return

    # 1013 (Try Again Later) rather than 1008: the client is not wrong, this
    # process is full. On a single-session box the guard below is the real
    # limit and this never fires.
    if not acquire_session_slot():
        logger.warning("voice_ws_rejected", reason="server at voice session capacity")
        await ws.close(code=1013, reason="Server is at its voice session capacity")
        return

    guard = getattr(ws.app.state, "single_session_guard", None)
    if guard is None:
        try:
            await _serve_voice_ws(ws, runnable)
        finally:
            release_session_slot()
        return

    if not guard.claim():
        logger.info("voice_ws_rejected", reason="single-session server already served its session")
        release_session_slot()
        await ws.close(code=1008, reason="Single-session server: a voice session was already served")
        return
    # On this transport the socket *is* the media connection.
    guard.mark_connected()
    try:
        await _serve_voice_ws(ws, runnable)
    finally:
        release_session_slot()
        # However the session ended — including an exception in the handshake
        # or session build — this socket was the one session. Without this,
        # a failure between claim() (idle timer disarmed) and the session's
        # own teardown would leave the process alive forever.
        await guard.finish()


async def _serve_voice_ws(ws: WebSocket, runnable: Any) -> None:
    """One WebSocket voice session: hello → session → events, until close."""
    from ..voice import SessionInterrupted, VoiceSessionEvent

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

    defaults: VoiceConfig = getattr(ws.app.state, "voice_config", None) or VoiceConfig()
    session, meta = build_voice_session(
        runnable,
        defaults,
        config,
        call_context=client_call_context(config),
        parent_run_id=client_parent_run_id(config),
    )
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
