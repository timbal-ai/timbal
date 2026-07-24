"""Deepgram realtime STT for :class:`~timbal.voice.VoiceSession`.

Two providers over raw WebSockets (no Deepgram SDK):

* :class:`DeepgramFluxSTT` — ``wss://api.deepgram.com/v2/listen`` (Flux).
  Model-native end-of-turn detection: ``TurnInfo`` events carry the whole
  turn transcript plus an EOU confidence. ``EndOfTurn`` maps to a committed
  transcript, so pair it with ``ProviderTurnDetector`` — Flux already did the
  Smart Turn / Namo / HOLD work server-side (~260ms EOU, ``eot_threshold``).
* :class:`DeepgramNovaSTT` — ``wss://api.deepgram.com/v1/listen`` (Nova-3 &
  friends). ASR only: interim results map to partials; ``is_final`` segments
  are buffered and flushed as one committed utterance on ``speech_final``
  (Deepgram's documented recipe). Timbal turn detection stays fully engaged,
  including the VAD endpointing fast path via ``Finalize``.

Requires ``websockets`` (already a core dep) and ``DEEPGRAM_API_KEY``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import structlog
from pydantic import SecretStr
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .session import (
    AudioInputConfig,
    SpeechToText,
    TranscriptEvent,
)

logger = structlog.get_logger("timbal.voice.deepgram")

_DG_HOST = "api.deepgram.com"

DEFAULT_FLUX_MODEL = "flux-general-multi"
DEFAULT_NOVA_MODEL = "nova-3"

# Flux docs strongly recommend ~80ms chunks for model latency; Nova is happy
# with the same cadence (their examples use 100ms). Flush as soon as we have
# that much PCM16 mono @ 16 kHz — don't wait for a timer tick (timer-only
# flush made Flux Update partials feel "wait until you shut up").
_AUDIO_FLUSH_INTERVAL = 0.08
_AUDIO_FLUSH_BYTES = int(16_000 * _AUDIO_FLUSH_INTERVAL * 2)  # 2560
# Nova drops the socket with NET-0001 after ~10s without audio; send KeepAlive
# well inside that window (Deepgram recommends every 3-5s).
_NOVA_KEEPALIVE_INTERVAL = 5.0

# Query params Flux accepts (used to filter stt_extra passthrough).
_FLUX_QUERY_KEYS = frozenset(
    {
        "eot_threshold",
        "eager_eot_threshold",
        "eot_timeout_ms",
        "keyterm",
        "tag",
        "mip_opt_out",
        "profanity_filter",
        "numerals",
        "redact",
    }
)


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    if isinstance(explicit, SecretStr):
        return explicit.get_secret_value()
    if explicit:
        return explicit
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise ValueError("Set DEEPGRAM_API_KEY or pass api_key to the provider.")
    return key


def _encoding_param(config: AudioInputConfig) -> str:
    """Timbal speaks PCM16LE mono end-to-end; Deepgram calls that ``linear16``."""
    if config.encoding in ("pcm_s16le", "linear16", ""):
        return "linear16"
    return config.encoding


class _DeepgramSTTBase(SpeechToText):
    """Shared WS plumbing: buffered audio flusher, receiver task, event queue."""

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._ws: Any = None
        self._buf = bytearray()
        # Covers buffer mutation *and* ``_ws.send``: ``push_audio`` (threshold
        # flush) and ``_flush_loop`` both drain PCM onto the socket; without a
        # single lock those awaits interleave and Deepgram can see out-of-order
        # or concurrent frames. Control frames (KeepAlive / Finalize / Close)
        # share it too.
        self._wire_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._flusher: asyncio.Task[None] | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._input_config: AudioInputConfig | None = None

    def _build_uri(self, config: AudioInputConfig) -> str:
        raise NotImplementedError

    async def connect(self, config: AudioInputConfig) -> None:
        self._api_key = _resolve_api_key(self._api_key_explicit)
        self._input_config = config
        uri = self._build_uri(config)
        logger.debug("dg_stt_connecting", uri=uri[:160])
        self._ws = await ws_connect(
            uri,
            additional_headers={"Authorization": f"Token {self._api_key}"},
        )
        self._stop.clear()
        self._flusher = asyncio.create_task(self._flush_loop())
        self._receiver = asyncio.create_task(self._receive_loop())

    async def push_audio(self, chunk: bytes) -> None:
        """Forward mic PCM to Deepgram with minimal buffering.

        Flux wants ~80ms frames for low-latency ``Update`` partials; a
        timer-only flusher made captions feel commit-gated ("wait until you
        shut up"). Send as soon as we have ≥80ms, and let the flush loop
        drain any trailing remainder.
        """
        if not chunk:
            return
        async with self._wire_lock:
            self._buf.extend(chunk)
            if len(self._buf) < _AUDIO_FLUSH_BYTES or self._ws is None:
                return
            raw = bytes(self._buf)
            self._buf.clear()
            try:
                await self._ws.send(raw)
            except ConnectionClosed:
                pass

    async def _flush_audio(self) -> None:
        async with self._wire_lock:
            if self._ws is None:
                return
            raw = bytes(self._buf)
            self._buf.clear()
            if not raw:
                return
            try:
                await self._ws.send(raw)
            except ConnectionClosed:
                pass

    async def _flush_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(_AUDIO_FLUSH_INTERVAL)
                await self._flush_audio()
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            pass

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        raise NotImplementedError

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw_msg in self._ws:
                if isinstance(raw_msg, bytes):
                    continue
                try:
                    msg = json.loads(raw_msg)
                except ValueError:
                    continue
                await self._handle_message(msg)
        except ConnectionClosed as e:
            logger.debug("dg_stt_ws_closed", error=str(e))
            # Deepgram closes normally after CloseStream; only surface abnormal
            # closures as errors so the session tears down loudly.
            if not self._stop.is_set() and e.rcvd is not None and e.rcvd.code not in (1000, 1001):
                await self._queue.put(
                    TranscriptEvent(type="error", text=f"STT connection closed: {e}")
                )
        except Exception as e:
            logger.error("dg_stt_receive_error", error=str(e), exc_info=True)
            await self._queue.put(TranscriptEvent(type="error", text=f"STT receive error: {e}"))
        finally:
            await self._queue.put(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.type == "error":
                raise RuntimeError(item.text)
            if item.text:
                yield item

    async def _send_json(self, payload: dict[str, Any]) -> None:
        async with self._wire_lock:
            if self._ws is None:
                return
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps(payload))

    async def close(self) -> None:
        self._stop.set()
        if self._flusher and not self._flusher.done():
            self._flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher
        self._flusher = None
        with contextlib.suppress(Exception):
            await self._flush_audio()
        await self._send_json({"type": "CloseStream"})
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._receiver and not self._receiver.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receiver
        self._receiver = None


class DeepgramFluxSTT(_DeepgramSTTBase):
    """Deepgram Flux (``/v2/listen``) — conversational STT with native EOU.

    ``TurnInfo`` mapping:

    * ``Update`` / ``StartOfTurn`` → ``partial`` (transcript is the whole turn
      so far, matching how the session treats Scribe partials)
    * ``EndOfTurn`` → ``committed``
    * ``EagerEndOfTurn`` / ``TurnResumed`` → logged only (speculative-LLM
      lifecycle is a follow-up)

    ``commit()`` is a no-op: the Flux turn machine owns endpointing, there is
    no client-side force-commit. Run with ``turn_detector="provider"`` and
    VAD endpointing off (the server wires both automatically).
    """

    def _build_uri(self, config: AudioInputConfig) -> str:
        extra = dict(config.extra)
        host = str(extra.pop("stt_host", _DG_HOST))
        # Ignore foreign model ids (e.g. env default scribe_v2_realtime left in
        # place when only TIMBAL_STT_PROVIDER was switched).
        model = config.model if is_flux_model(config.model) else DEFAULT_FLUX_MODEL

        params: list[tuple[str, str]] = [
            ("model", model),
            ("encoding", _encoding_param(config)),
            ("sample_rate", str(config.sample_rate)),
        ]
        # Flux rejects `language`; multi accepts language_hint (repeatable).
        if config.language and model.endswith("-multi"):
            params.append(("language_hint", config.language))
        # Default 5s leaves stranded Update partials hanging in the UI until
        # the session sweeper synthesizes a commit. Prefer Flux's own
        # EndOfTurn; keep this under the session stale window (2.5s).
        if "eot_timeout_ms" not in extra:
            params.append(("eot_timeout_ms", "2000"))
        for k, v in extra.items():
            if v is None or k.startswith("_") or k not in _FLUX_QUERY_KEYS:
                continue
            if isinstance(v, (list, tuple)):
                params.extend((k, str(item)) for item in v)
            else:
                params.append((k, str(v).lower() if isinstance(v, bool) else str(v)))
        return f"wss://{host}/v2/listen?{urlencode(params)}"

    async def commit(self) -> None:
        """No-op: Flux's turn machine owns endpointing (no force-commit API)."""

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        mt = msg.get("type", "")
        if mt == "Connected":
            logger.info("dg_flux_session_started", request_id=msg.get("request_id"))
            return
        if mt == "TurnInfo":
            event = msg.get("event", "")
            text = (msg.get("transcript") or "").strip()
            if event == "EndOfTurn":
                logger.debug(
                    "dg_flux_end_of_turn",
                    eou_confidence=msg.get("end_of_turn_confidence"),
                    turn_index=msg.get("turn_index"),
                )
                if text:
                    await self._queue.put(TranscriptEvent(type="committed", text=text))
            elif event in ("Update", "StartOfTurn"):
                if text:
                    await self._queue.put(TranscriptEvent(type="partial", text=text))
            elif event in ("EagerEndOfTurn", "TurnResumed"):
                # Speculative reply lifecycle not wired yet — visibility only.
                logger.info(
                    "dg_flux_eager_event",
                    turn_event=event,
                    eou_confidence=msg.get("end_of_turn_confidence"),
                    text_preview=text[:80],
                )
            return
        if mt == "FatalError":
            err = msg.get("description") or msg.get("message") or "Unknown Flux error"
            logger.error("dg_flux_fatal", error=err, code=msg.get("code"))
            await self._queue.put(TranscriptEvent(type="error", text=f"STT fatal: {err}"))


class DeepgramNovaSTT(_DeepgramSTTBase):
    """Deepgram Nova (``/v1/listen``) — plain streaming ASR.

    Per Deepgram's endpointing guide, ``is_final`` segments are buffered and
    concatenated; ``speech_final`` flushes the buffer as one committed
    utterance. Interims map to partials (buffer + interim, so the session
    sees the whole utterance so far, like Scribe).

    ``commit()`` sends ``Finalize`` — Deepgram flushes its audio buffer and
    replies with a final (``from_finalize: true``) result, so the Timbal VAD
    endpointing fast path works unchanged.
    """

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        super().__init__(api_key)
        self._segments: list[str] = []
        self._keepalive: asyncio.Task[None] | None = None

    def _build_uri(self, config: AudioInputConfig) -> str:
        extra = dict(config.extra)
        host = str(extra.pop("stt_host", _DG_HOST))

        model = config.model or ""
        if not model or is_flux_model(model) or model.startswith(("scribe", "eleven")):
            model = DEFAULT_NOVA_MODEL
        params: dict[str, Any] = {
            "model": model,
            "encoding": _encoding_param(config),
            "sample_rate": str(config.sample_rate),
            "channels": "1",
            "interim_results": "true",
            "smart_format": "true",
            "punctuate": "true",
            "endpointing": "300",
        }
        if config.language:
            params["language"] = config.language
        for k, v in extra.items():
            if v is not None and not k.startswith("_"):
                params[k] = str(v).lower() if isinstance(v, bool) else str(v)
        return f"wss://{host}/v1/listen?{urlencode(params)}"

    async def connect(self, config: AudioInputConfig) -> None:
        await super().connect(config)
        self._keepalive = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(_NOVA_KEEPALIVE_INTERVAL)
                await self._send_json({"type": "KeepAlive"})
        except asyncio.CancelledError:
            raise

    async def commit(self) -> None:
        """Force-finalize whatever Deepgram is holding (VAD endpointing path)."""
        await self._flush_audio()
        await self._send_json({"type": "Finalize"})

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        mt = msg.get("type", "")
        if mt == "Results":
            alternatives = (msg.get("channel") or {}).get("alternatives") or [{}]
            text = (alternatives[0].get("transcript") or "").strip()
            is_final = bool(msg.get("is_final"))
            speech_final = bool(msg.get("speech_final"))
            if not is_final:
                if text:
                    partial = " ".join((*self._segments, text)) if self._segments else text
                    await self._queue.put(TranscriptEvent(type="partial", text=partial))
                return
            if text:
                self._segments.append(text)
            # speech_final only rides on is_final frames; a Finalize response
            # (from_finalize) also arrives as is_final and must flush too.
            if (speech_final or msg.get("from_finalize")) and self._segments:
                utterance = " ".join(self._segments)
                self._segments = []
                await self._queue.put(TranscriptEvent(type="committed", text=utterance))
            return
        if mt == "UtteranceEnd":
            # Only reachable when the caller opted into utterance_end_ms.
            if self._segments:
                utterance = " ".join(self._segments)
                self._segments = []
                await self._queue.put(TranscriptEvent(type="committed", text=utterance))
            return
        if mt == "Metadata":
            logger.info("dg_nova_session_started", request_id=msg.get("request_id"))
            return
        if mt == "Error":
            err = msg.get("description") or msg.get("message") or "Unknown Nova error"
            logger.error("dg_nova_error", error=err)
            await self._queue.put(TranscriptEvent(type="error", text=f"STT error: {err}"))

    async def close(self) -> None:
        if self._keepalive and not self._keepalive.done():
            self._keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive
        self._keepalive = None
        await super().close()


def is_flux_model(model: str | None) -> bool:
    return bool(model) and model.strip().lower().startswith("flux")


def effective_stt_model(provider_instance: SpeechToText, requested: str | None) -> str | None:
    """Model id actually sent to the provider (foreign leftovers swapped out)."""
    if isinstance(provider_instance, DeepgramFluxSTT):
        return requested if is_flux_model(requested) else DEFAULT_FLUX_MODEL
    if isinstance(provider_instance, DeepgramNovaSTT):
        m = requested or ""
        if not m or is_flux_model(m) or m.startswith(("scribe", "eleven")):
            return DEFAULT_NOVA_MODEL
        return requested
    # ElevenLabs (and any non-Deepgram STT): never pass flux/nova ids through —
    # e.g. unknown-provider fallback keeps the merged model string otherwise.
    m = (requested or "").strip().lower()
    if not m or is_flux_model(m) or m.startswith("nova"):
        return None
    return requested


def stt_provider_id(provider_instance: SpeechToText) -> str:
    """Config-style provider id for the running STT instance.

    Matches playground / ``voice_config`` values (``elevenlabs``,
    ``deepgram-flux``, ``deepgram-nova``) — not the Python class name.
    """
    if isinstance(provider_instance, DeepgramFluxSTT):
        return "deepgram-flux"
    if isinstance(provider_instance, DeepgramNovaSTT):
        return "deepgram-nova"
    return "elevenlabs"


def resolve_stt(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | SecretStr | None = None,
) -> SpeechToText:
    """STT factory for the voice server.

    ``provider`` is ``"elevenlabs"`` / ``"deepgram"`` (case-insensitive; also
    accepts UI labels like ``"deepgram-flux"`` / ``"deepgram-nova"``). When
    ``None``, inferred from the model id: ``flux-*`` / ``nova-*`` → Deepgram,
    anything else (including ``scribe_*``) → ElevenLabs.

    Bare ``"deepgram"`` defaults to Flux (voice-agent native EOU). Only an
    explicit ``nova`` in the provider label or a ``nova-*`` model selects Nova
    — a leftover ``scribe_*`` env model must not silently route to Nova.
    """
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()
    if not p:
        p = "deepgram" if (m.startswith("flux") or m.startswith("nova")) else "elevenlabs"
    if p in ("elevenlabs", "el", "11labs"):
        from .elevenlabs import ElevenLabsRealtimeSTT

        return ElevenLabsRealtimeSTT(api_key=api_key)
    if p.startswith("deepgram") or p == "dg":
        # UI labels win. Bare "deepgram"/"dg" → Flux unless model is clearly nova-*.
        if "nova" in p:
            return DeepgramNovaSTT(api_key=api_key)
        if "flux" in p:
            return DeepgramFluxSTT(api_key=api_key)
        if m.startswith("nova"):
            return DeepgramNovaSTT(api_key=api_key)
        return DeepgramFluxSTT(api_key=api_key)
    raise ValueError(f"Unknown STT provider: {provider!r}")
