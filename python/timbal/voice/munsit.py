"""Munsit (Faseeh) Arabic voice providers for :class:`~timbal.voice.VoiceSession`.

Two providers, both authenticated with ``MUNSIT_API_KEY``:

* :class:`MunsitStreamSTT` — live streaming STT over
  ``WSS /api/v1/listen`` (interim results, word timestamps, and
  provider-native end-of-turn detection via ``endpointing`` + ``smart_turn``).
* :class:`MunsitStreamTTS` — TTS over the HTTP chunked-streaming endpoint
  ``POST /api/v1/text-to-speech/{model_id}`` with ``streaming: true`` —
  raw PCM16 chunks are yielded as they are generated.

**Why TTS is HTTP and not the WebSocket** (re-verified 2026-08-31 against
https://docs.munsit.com/reference/websocket): the documented TTS socket
(``/api/v1/websocket/text-to-speech``) still accepts ``initConnection`` (and
answers ``connectionInitialized``) and ``text`` frames, but produces **no
audio** — probed live on 2026-08-31 across query-param / initConnection auth
and every ``flush`` / ``try_trigger_generation`` combination; the documented
header and Bearer auth are additionally rejected with error 40101. Even per
its docs the socket tops out at ``pcm_24000`` while HTTP serves the
engine-native 48 kHz, and Munsit itself recommends HTTP when the segment text
is known upfront (our case: the session flushes complete segments).

Decisively: Munsit's own first-party plugins do not use their TTS socket
either. ``livekit-plugins-munsit`` 0.4.0 implements its incremental
``stream()`` as a client-side sentence chunker issuing one HTTP
``streaming: true`` POST per chunk, and ``pipecat-plugins-munsit`` 0.2.0 does
the same — zero WebSocket code in either TTS module (verified from the PyPI
sdists, 2026-08-31). There is no prosody-continuous incremental-feed path in
this API today, so no ``open_stream`` capability: the session falls back to
per-segment ``synthesize``, which is exactly the transport the vendor's own
plugins use.
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

from .config import DEFAULT_VOICE_ID as _ELEVENLABS_DEFAULT_VOICE_ID
from .providers import (
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
)

logger = structlog.get_logger("timbal.voice.munsit")

DEFAULT_TTS_MODEL = "faseeh-v1-preview"
# Munsit's LiveKit examples use this Emirati voice; the docs' own example voice
# (ar-najdi-male-2) is Najdi Saudi, which surprised the Emirati deployments this
# provider was added for. Override with MUNSIT_VOICE_ID (cloned/custom voices
# are account-specific) or set an explicit voice on the session.
DEFAULT_MUNSIT_VOICE_ID = "ar-uae-male-1"

DEFAULT_STT_MODEL = "munsit-en-ar"

_DEFAULT_HOST = "api.munsit.com"
_SAMPLE_RATE_MIN = 8_000
_SAMPLE_RATE_MAX = 48_000


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    if isinstance(explicit, SecretStr):
        return explicit.get_secret_value()
    if explicit:
        return explicit
    key = os.environ.get("MUNSIT_API_KEY")
    if not key:
        raise ValueError("Set MUNSIT_API_KEY or pass api_key to the provider.")
    return key


# ---------------------------------------------------------------------------
# Streaming STT — WSS /api/v1/listen
# ---------------------------------------------------------------------------

# The /listen socket accepts exactly these rates (anything else is a fatal 4002).
_STT_SAMPLE_RATES = frozenset({8_000, 16_000})
# Munsit is happy with 20-200ms frames; 80ms matches the Deepgram adapters so
# the session's mic cadence behaves identically across providers.
_STT_AUDIO_FLUSH_INTERVAL = 0.08
# 12s with neither audio nor KeepAlive closes the socket with 1011; Munsit's
# docs say to send KeepAlive during pauses — well inside that window.
_STT_KEEPALIVE_INTERVAL = 5.0
# After CloseStream the server finalizes the in-flight turn and sends the
# billing Metadata before closing with 1000. Closing our side first aborts
# finalization (documented), so close() waits for the server — but bounded.
_STT_CLOSE_DRAIN_TIMEOUT = 5.0

# Query params /api/v1/listen accepts (used to filter stt_extra passthrough).
# `model`, `language`, `encoding`, `sample_rate` are owned by the config.
_STT_QUERY_KEYS = frozenset(
    {
        "channels",
        "endpointing",
        "smart_turn",
        "hotwords",
        "interim_results",
        "correlation_id",
        "metadata",
    }
)


def is_munsit_stt_model(model: str | None) -> bool:
    return bool(model) and model.strip().lower().startswith("munsit")


def effective_stt_model(requested: str | None) -> str:
    """Model id actually sent to Munsit (foreign leftovers swapped out).

    Same pattern as :func:`effective_tts_model`: a session that only switched
    ``stt_provider`` must not put ``scribe_v2_realtime`` / ``flux-*`` on the
    Munsit wire. Defaults to ``munsit-en-ar`` (code-switching) rather than the
    raw API's ``munsit`` because the deployments pinning Munsit mix Arabic and
    English mid-utterance — same default as Munsit's own LiveKit plugin.
    """
    return requested.strip() if is_munsit_stt_model(requested) else DEFAULT_STT_MODEL


class MunsitStreamSTT(SpeechToText):
    """Munsit live streaming STT (``WSS /api/v1/listen``) with native EOU.

    The wire protocol is Deepgram-flavored. Event mapping:

    * interim ``Results`` → ``partial`` (the transcript is the whole turn so
      far; buffered forced-split segments are prepended)
    * ``Results`` with ``speech_final: true`` → ``committed`` — Munsit's own
      endpointing + smart-turn model decided the speaker finished
    * ``Results`` with ``is_final: true`` but ``speech_final: false`` is the
      ~60s forced split during unbroken speech: the text is stable but the
      speaker is still talking, so it is buffered and **never** committed on
      its own (no ``UtteranceEnd`` fires for it either)
    * ``UtteranceEnd`` → backup flush of any buffered segments; normally a
      no-op because the ``speech_final`` frame already committed the turn
    * ``Gender`` / ``Sentiment`` enrichment and ``SpeechStarted`` → logged only
    * ``Error`` with ``recoverable: true`` → logged; ``recoverable: false``
      always precedes a close and surfaces as a session error

    ``commit()`` is a no-op: the Munsit turn machine owns endpointing —
    ``Configure`` can retune the silence window mid-session but there is no
    client force-commit (``CloseStream`` ends the whole session). Pair with
    ``turn_detector="provider"``; the server wires that automatically via
    :attr:`native_eou`, and disables the VAD endpointing fast path the same
    way it does for Deepgram Flux.

    ``stt_extra`` passthrough: ``endpointing`` (ms, 100-5000), ``smart_turn``,
    ``hotwords`` (ignored by ``munsit-en-ar``), ``channels``,
    ``correlation_id``, ``metadata``, ``interim_results``; plus ``stt_host``
    for a self-hosted deployment.
    """

    native_eou = True

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._ws: Any = None
        self._buf = bytearray()
        # Single lock over buffer mutation *and* ws.send — same rationale as
        # the Deepgram adapters: threshold pushes, the flush loop and control
        # frames must never interleave frames on the socket.
        self._wire_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._flusher: asyncio.Task[None] | None = None
        self._keepalive: asyncio.Task[None] | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._segments: list[str] = []
        self._flush_bytes = int(16_000 * _STT_AUDIO_FLUSH_INTERVAL * 2)

    def _build_uri(self, config: AudioInputConfig) -> str:
        extra = dict(config.extra)
        host = str(extra.pop("stt_host", _DEFAULT_HOST))
        if config.sample_rate not in _STT_SAMPLE_RATES:
            # Fail at session start rather than let the server kill the socket
            # with a fatal 4002 on the first frame.
            raise ValueError(f"Munsit streaming STT supports 8000 or 16000 Hz; got {config.sample_rate}.")
        encoding = "linear16" if config.encoding in ("pcm_s16le", "linear16", "") else config.encoding
        params: list[tuple[str, str]] = [
            ("model", effective_stt_model(config.model)),
            ("encoding", encoding),
            ("sample_rate", str(config.sample_rate)),
        ]
        # `ar` is the only supported value in v1 and invalid connection params
        # are fatal — drop foreign language leftovers rather than 4002 the
        # socket ("ar-AE" style region tags collapse to "ar").
        lang = (config.language or "").strip().lower()
        if lang.startswith("ar"):
            params.append(("language", "ar"))
        elif lang:
            logger.debug("munsit_stt_language_dropped", language=config.language)
        for k, v in extra.items():
            if v is None or k.startswith("_") or k not in _STT_QUERY_KEYS:
                continue
            params.append((k, str(v).lower() if isinstance(v, bool) else str(v)))
        return f"wss://{host}/api/v1/listen?{urlencode(params)}"

    async def connect(self, config: AudioInputConfig) -> None:
        self._api_key = _resolve_api_key(self._api_key_explicit)
        uri = self._build_uri(config)
        self._flush_bytes = int(config.sample_rate * _STT_AUDIO_FLUSH_INTERVAL * 2)
        logger.debug("munsit_stt_connecting", uri=uri[:160])
        from websockets.asyncio.client import connect as ws_connect

        self._ws = await ws_connect(
            uri,
            # Server-side x-api-key header — never the api_key query param,
            # which puts the key in any log that records the URI.
            additional_headers={"x-api-key": self._api_key},
        )
        self._stop.clear()
        self._flusher = asyncio.create_task(self._flush_loop())
        self._keepalive = asyncio.create_task(self._keepalive_loop())
        self._receiver = asyncio.create_task(self._receive_loop())

    async def push_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        from websockets.exceptions import ConnectionClosed

        async with self._wire_lock:
            self._buf.extend(chunk)
            if len(self._buf) < self._flush_bytes or self._ws is None:
                return
            raw = bytes(self._buf)
            self._buf.clear()
            try:
                await self._ws.send(raw)
            except ConnectionClosed:
                pass

    async def _flush_audio(self) -> None:
        from websockets.exceptions import ConnectionClosed

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
        from websockets.exceptions import ConnectionClosed

        try:
            while not self._stop.is_set():
                await asyncio.sleep(_STT_AUDIO_FLUSH_INTERVAL)
                await self._flush_audio()
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            pass

    async def _keepalive_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(_STT_KEEPALIVE_INTERVAL)
                await self._send_json({"type": "KeepAlive"})
        except asyncio.CancelledError:
            raise

    async def commit(self) -> None:
        """No-op: Munsit's turn machine owns endpointing (no force-commit API)."""

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        mt = msg.get("type", "")
        if mt == "Results":
            text = (msg.get("transcript") or "").strip()
            is_final = bool(msg.get("is_final"))
            speech_final = bool(msg.get("speech_final"))
            if not is_final:
                if text:
                    partial = " ".join((*self._segments, text)) if self._segments else text
                    await self._queue.put(TranscriptEvent(type="partial", text=partial))
                return
            if not speech_final:
                # Forced ~60s split: stable text, speaker still talking under
                # the next turn_id. Buffer; committing here would answer a
                # user who is mid-sentence.
                if text:
                    self._segments.append(text)
                logger.debug("munsit_stt_forced_split", turn_id=msg.get("turn_id"), text_preview=text[:80])
                return
            if text:
                self._segments.append(text)
            if self._segments:
                utterance = " ".join(self._segments)
                self._segments = []
                await self._queue.put(TranscriptEvent(type="committed", text=utterance))
            return
        if mt == "UtteranceEnd":
            # Fires right after the final Results of a completed turn, so the
            # speech_final frame has normally already flushed — this is the
            # backup for a final that carried speech_final: false variants.
            if self._segments:
                utterance = " ".join(self._segments)
                self._segments = []
                await self._queue.put(TranscriptEvent(type="committed", text=utterance))
            return
        if mt == "SpeechStarted":
            logger.debug("munsit_stt_speech_started", ts=msg.get("ts"))
            return
        if mt in ("Gender", "Sentiment"):
            # Enrichment is out of scope for the voice session; visibility only.
            logger.debug("munsit_stt_enrichment", enrichment=mt, label=msg.get("label"), score=msg.get("score"))
            return
        if mt == "Metadata":
            if "audio_seconds_billed" in msg:
                logger.info(
                    "munsit_stt_session_closed",
                    session_id=msg.get("session_id"),
                    audio_seconds_billed=msg.get("audio_seconds_billed"),
                    turn_count=msg.get("turn_count"),
                )
            else:
                logger.info("munsit_stt_session_started", session_id=msg.get("session_id"), model=msg.get("model"))
                if msg.get("dropped_hotwords"):
                    logger.warning("munsit_stt_dropped_hotwords", dropped=msg["dropped_hotwords"])
            return
        if mt == "Error":
            err = msg.get("message") or "Unknown Munsit error"
            if msg.get("recoverable"):
                logger.warning("munsit_stt_recoverable_error", error=err, code=msg.get("code"))
                return
            logger.error("munsit_stt_fatal", error=err, code=msg.get("code"))
            await self._queue.put(TranscriptEvent(type="error", text=f"STT fatal: {err}"))

    async def _receive_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

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
            if self._stop.is_set():
                logger.debug("munsit_stt_ws_closed", error=str(e))
            else:
                # Provider-initiated hangup is not the user hanging up — same
                # rule as the Deepgram adapters.
                logger.warning("munsit_stt_ws_closed_unrequested", error=str(e))
                await self._queue.put(TranscriptEvent(type="error", text=f"STT connection closed: {e}"))
        except Exception as e:
            logger.error("munsit_stt_receive_error", error=str(e), exc_info=True)
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
        for task in (self._flusher, self._keepalive):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._flusher = None
        self._keepalive = None
        with contextlib.suppress(Exception):
            await self._flush_audio()
        await self._send_json({"type": "CloseStream"})
        # Wait for the server to finalize and close (1000) — closing our side
        # first aborts finalization and loses the final Results + billing
        # Metadata (documented). Bounded so a wedged server can't hang teardown.
        if self._receiver and not self._receiver.done():
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(self._receiver), timeout=_STT_CLOSE_DRAIN_TIMEOUT)
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._receiver and not self._receiver.done():
            self._receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receiver
        self._receiver = None


# ---------------------------------------------------------------------------
# Streaming TTS — POST /api/v1/text-to-speech/{model_id} (streaming: true)
# ---------------------------------------------------------------------------


def effective_sample_rate(config: AudioOutputConfig) -> int:
    """Sample rate actually requested (endpoint accepts 8000–48000 Hz).

    Out-of-range rates raise instead of remapping: the session clocks playback
    at ``config.sample_rate``, so silently requesting a different rate would
    desync audio (chipmunk/slow-motion speech).

    Munsit's engine-native rate is 48000 — lower rates are downsampled
    server-side (a quality cost, not a latency one). The session-wide default
    of 16 kHz is honoured as requested; deployments that want engine-native
    audio should raise the session ``sample_rate`` (e.g. LiveKit/WebRTC
    transports, which resample to 48 kHz anyway), not this provider.
    """
    sr = config.sample_rate
    if not (_SAMPLE_RATE_MIN <= sr <= _SAMPLE_RATE_MAX):
        raise ValueError(f"Munsit TTS supports {_SAMPLE_RATE_MIN}-{_SAMPLE_RATE_MAX} Hz; got {sr}.")
    return sr


def effective_tts_model(config: AudioOutputConfig) -> str:
    """Model id actually sent to Munsit (foreign leftovers swapped out).

    The server-wide ``tts_model`` default is ElevenLabs' — a session that only
    switched ``tts_provider`` must not put ``eleven_flash_v2_5`` on the Munsit
    wire.
    """
    m = (config.model or "").strip()
    if not m or m.lower().startswith(("eleven", "scribe")):
        return DEFAULT_TTS_MODEL
    return m


def effective_voice_id(config: AudioOutputConfig) -> str:
    """Voice id actually sent to Munsit.

    The server-wide ``voice`` default is an ElevenLabs voice id; swap it for a
    Munsit voice unless the caller picked one explicitly.
    """
    v = (config.voice or "").strip()
    if not v or v == _ELEVENLABS_DEFAULT_VOICE_ID:
        return os.environ.get("MUNSIT_VOICE_ID") or DEFAULT_MUNSIT_VOICE_ID
    return v


class MunsitStreamTTS(TextToSpeech):
    """Munsit TTS via HTTP chunked streaming, one request per segment.

    A persistent ``httpx.AsyncClient`` keeps connections pooled across
    segments, so per-segment requests skip the TCP+TLS handshake.

    ``tts_extra`` passthrough: ``stability`` (default 0.5), ``speed`` (default
    1.0), ``dialect`` (``auto`` | ``emirati`` | ``fusha`` — pronunciation hint,
    unset means the API's ``auto``), and ``tts_host``. An Emirati deployment
    sets ``tts_extra={"dialect": "emirati"}`` with no code change here.
    """

    provider_id = "munsit"

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._out: AudioOutputConfig | None = None
        self._client: Any = None

    async def connect(self, config: AudioOutputConfig) -> None:
        self._api_key = _resolve_api_key(self._api_key_explicit)
        # Fail at session start, not first reply, on an unsupported rate.
        effective_sample_rate(config)
        self._out = config
        import httpx

        # read=None: chunk gaps track generation pace and a hard read timeout
        # would sever long segments mid-audio.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0, read=None))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not self._api_key or self._client is None or self._out is None:
            raise RuntimeError("Call connect() before synthesize().")
        stripped = text.strip()
        if not stripped:
            return

        cfg = self._out
        extra = dict(cfg.extra)
        host = str(extra.get("tts_host", _DEFAULT_HOST))
        model_id = effective_tts_model(cfg)
        payload: dict[str, Any] = {
            "voice_id": effective_voice_id(cfg),
            "text": stripped,
            "stability": float(extra.get("stability", 0.5)),
            "speed": float(extra.get("speed", 1.0)),
            "streaming": True,
            "sample_rate": effective_sample_rate(cfg),
        }
        if extra.get("dialect"):
            payload["dialect"] = str(extra["dialect"])

        url = f"https://{host}/api/v1/text-to-speech/{model_id}"
        logger.debug("munsit_tts_request", model_id=model_id, text_chars=len(stripped), text_preview=stripped[:120])
        chunk_count = 0
        async with self._client.stream(
            "POST",
            url,
            headers={"x-api-key": self._api_key},
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"Munsit TTS error {response.status_code}: {body[:500]}")
            async for chunk in response.aiter_bytes():
                if chunk:
                    chunk_count += 1
                    yield chunk
        logger.debug("munsit_tts_request_done", audio_chunks=chunk_count)

    async def close(self) -> None:
        if self._client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
        self._out = None
