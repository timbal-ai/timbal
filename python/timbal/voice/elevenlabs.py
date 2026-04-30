"""ElevenLabs realtime STT + streaming TTS for :class:`~timbal.voice.VoiceSession`.

Uses the official WebSocket APIs:

* ``wss://api.elevenlabs.io/v1/speech-to-text/realtime`` — Scribe realtime
* ``wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input`` — TTS stream

Requires ``websockets`` (``pip install timbal[server]``) and ``ELEVENLABS_API_KEY``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote, urlencode

import structlog
from pydantic import SecretStr
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .session import (
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
)

logger = structlog.get_logger("timbal.voice.elevenlabs")

STT_FLUSH_INTERVAL = 0.1

# ElevenLabs stream-input closes with 1008 if no new *text* is sent for `inactivity_timeout`
# seconds while the socket is open (default 20). Long single-chunk playback can exceed that.
_DEFAULT_TTS_INACTIVITY_TIMEOUT = 180
# Space pings reset stream-input inactivity; too frequent can inject extra “text” mid-receive.
_TTS_KEEPALIVE_INTERVAL_SEC = 55.0

STT_FATAL_MESSAGE_TYPES = frozenset(
    {
        "auth_error",
        "quota_exceeded",
        "rate_limited",
        "resource_exhausted",
        "session_time_limit_exceeded",
    }
)

_DEFAULT_STT_MODEL = "scribe_v2_realtime"
_DEFAULT_TTS_MODEL = "eleven_flash_v2_5"


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    if isinstance(explicit, SecretStr):
        return explicit.get_secret_value()
    if explicit:
        return explicit
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise ValueError("Set ELEVENLABS_API_KEY or pass api_key to the provider.")
    return key


def _stt_audio_format(config: AudioInputConfig) -> str:
    """Map sample rate to ElevenLabs ``audio_format`` query value."""
    sr = config.sample_rate
    mapping = {
        8000: "pcm_8000",
        16000: "pcm_16000",
        22050: "pcm_22050",
        24000: "pcm_24000",
        44100: "pcm_44100",
        48000: "pcm_48000",
    }
    if sr in mapping:
        return mapping[sr]
    return config.extra.get("audio_format") or "pcm_16000"


def _tts_output_format(config: AudioOutputConfig) -> str:
    sr = config.sample_rate
    mapping = {
        8000: "pcm_8000",
        16000: "pcm_16000",
        22050: "pcm_22050",
        24000: "pcm_24000",
        44100: "pcm_44100",
    }
    if config.extra.get("output_format"):
        return str(config.extra["output_format"])
    if sr in mapping:
        return mapping[sr]
    return "pcm_16000"


class ElevenLabsRealtimeSTT(SpeechToText):
    """ElevenLabs Scribe v2 realtime WebSocket (VAD commits by default)."""

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._ws: Any = None
        self._buf = bytearray()
        self._buf_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._flusher: asyncio.Task[None] | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._input_config: AudioInputConfig | None = None

    async def connect(self, config: AudioInputConfig) -> None:
        self._api_key = _resolve_api_key(self._api_key_explicit)
        self._input_config = config
        extra = dict(config.extra)
        host = str(extra.pop("stt_host", "api.elevenlabs.io"))
        commit_strategy = str(extra.pop("commit_strategy", "vad"))

        params: dict[str, Any] = {
            "model_id": config.model or _DEFAULT_STT_MODEL,
            "audio_format": extra.pop("audio_format", None) or _stt_audio_format(config),
            "commit_strategy": commit_strategy,
        }
        if config.language:
            params["language_code"] = config.language
        for k, v in extra.items():
            if v is not None and not k.startswith("_"):
                params[k] = v

        query = urlencode(params)
        uri = f"wss://{host}/v1/speech-to-text/realtime?{query}"
        self._ws = await ws_connect(
            uri,
            additional_headers={"xi-api-key": self._api_key},
        )
        self._stop.clear()
        self._flusher = asyncio.create_task(self._periodic_flush())
        self._receiver = asyncio.create_task(self._receive_loop())

    async def push_audio(self, chunk: bytes) -> None:
        if chunk:
            async with self._buf_lock:
                self._buf.extend(chunk)

    async def commit(self) -> None:
        """Force-commit the current STT buffer (manual commit strategy)."""
        await self._flush_audio(commit=True)

    async def _flush_audio(self, commit: bool = False) -> None:
        if self._ws is None:
            return
        async with self._buf_lock:
            raw = bytes(self._buf)
            self._buf.clear()
        if not raw and not commit:
            return
        msg: dict[str, Any] = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(raw).decode("ascii") if raw else "",
            "commit": commit,
            "sample_rate": self._input_config.sample_rate if self._input_config else 16_000,
        }
        await self._ws.send(json.dumps(msg))

    async def _periodic_flush(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(STT_FLUSH_INTERVAL)
                await self._flush_audio(commit=False)
        except asyncio.CancelledError:
            raise

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw_msg in self._ws:
                msg = json.loads(raw_msg)
                mt = msg.get("message_type", "")

                if mt == "session_started":
                    logger.info("el_stt_session_started", session_id=msg.get("session_id"))

                elif mt == "partial_transcript":
                    text = msg.get("text", "")
                    if text:
                        await self._queue.put(TranscriptEvent(type="partial", text=text))

                elif mt in ("committed_transcript", "committed_transcript_with_timestamps"):
                    text = (msg.get("text") or "").strip()
                    if text:
                        await self._queue.put(TranscriptEvent(type="committed", text=text))

                elif mt == "error":
                    err = msg.get("error", "Unknown STT error")
                    logger.error("el_stt_error", error=err)
                    await self._queue.put(TranscriptEvent(type="error", text=f"STT error: {err}"))
                    break

                elif mt in STT_FATAL_MESSAGE_TYPES:
                    err = msg.get("error", mt)
                    logger.error("el_stt_fatal", error=err, type=mt)
                    await self._queue.put(TranscriptEvent(type="error", text=f"STT fatal ({mt}): {err}"))
                    break
        except ConnectionClosed as e:
            logger.debug("el_stt_ws_closed", error=str(e))
            await self._queue.put(TranscriptEvent(type="error", text=f"STT connection closed: {e}"))
        except Exception as e:
            logger.error("el_stt_receive_error", error=str(e), exc_info=True)
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

    async def close(self) -> None:
        self._stop.set()
        if self._flusher and not self._flusher.done():
            self._flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher
        self._flusher = None
        with contextlib.suppress(Exception):
            await self._flush_audio(commit=False)
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._receiver and not self._receiver.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receiver
        self._receiver = None


class ElevenLabsStreamTTS(TextToSpeech):
    """ElevenLabs TTS via the **multi-context** WebSocket.

    Keeps ONE persistent WS connection across ``synthesize`` calls.  Each call
    creates an independent *context* — ElevenLabs returns per-context audio and
    ``is_final`` markers, which eliminates the "second segment silence" bug that
    occurred with one-connection-per-segment.

    Endpoint: ``/v1/text-to-speech/{voice_id}/multi-stream-input``

    Multi-context WS is **not** available for the ``eleven_v3`` model but works
    fine with ``eleven_flash_v2_5`` (the default for real-time voice).
    """

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._out: AudioOutputConfig | None = None
        # Persistent multi-context WS state
        self._ws: Any = None
        self._ws_open: bool = False
        self._reader_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._audio_queues: dict[str, asyncio.Queue[dict | None]] = {}
        self._active_contexts: set[str] = set()
        self._ctx_counter: int = 0

    async def connect(self, config: AudioOutputConfig) -> None:
        self._api_key = _resolve_api_key(self._api_key_explicit)
        if not config.voice:
            raise ValueError("AudioOutputConfig.voice (ElevenLabs voice_id) is required.")
        self._out = config

    # -- persistent WS lifecycle --------------------------------------------

    async def _ensure_ws(self) -> None:
        """Lazily open the multi-context WS if not already connected."""
        if self._ws is not None and self._ws_open:
            return
        if self._ws is not None:
            await self._teardown_ws()

        cfg = self._out
        assert cfg is not None
        extra = dict(cfg.extra)
        host = str(extra.pop("tts_host", "api.elevenlabs.io"))
        model_id = cfg.model or _DEFAULT_TTS_MODEL
        output_format = _tts_output_format(cfg)
        inactivity_timeout = int(extra.pop("inactivity_timeout", _DEFAULT_TTS_INACTIVITY_TIMEOUT))
        inactivity_timeout = max(20, min(inactivity_timeout, 180))
        auto_mode = str(extra.pop("auto_mode", "true")).lower() in ("1", "true", "yes")
        keepalive_interval = float(extra.pop("tts_keepalive_interval", _TTS_KEEPALIVE_INTERVAL_SEC))
        apply_text_normalization = str(extra.pop("apply_text_normalization", "on"))

        params: dict[str, Any] = {
            "model_id": model_id,
            "output_format": output_format,
            "inactivity_timeout": inactivity_timeout,
            "apply_text_normalization": apply_text_normalization,
        }
        if auto_mode:
            params["auto_mode"] = "true"
        for k, v in extra.items():
            if v is not None and not str(k).startswith("_"):
                params[k] = v

        path = f"/v1/text-to-speech/{quote(cfg.voice, safe='')}/multi-stream-input"
        uri = f"wss://{host}{path}?{urlencode(params)}"

        logger.debug("el_tts_ws_connecting", uri=uri[:160])
        self._ws = await ws_connect(
            uri,
            additional_headers={"xi-api-key": self._api_key},
        )
        self._ws_open = True
        self._stop.clear()
        self._reader_task = asyncio.create_task(self._read_loop())
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(max(5.0, keepalive_interval)),
        )
        logger.debug("el_tts_ws_connected")

    async def _teardown_ws(self) -> None:
        self._stop.set()
        self._ws_open = False
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
        self._keepalive_task = None
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        self._reader_task = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        for q in list(self._audio_queues.values()):
            with contextlib.suppress(Exception):
                q.put_nowait(None)
        self._audio_queues.clear()
        self._active_contexts.clear()

    # -- background tasks ---------------------------------------------------

    async def _read_loop(self) -> None:
        """Dispatch incoming WS messages to per-context queues."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                ctx = msg.get("contextId")
                if ctx and ctx in self._audio_queues:
                    await self._audio_queues[ctx].put(msg)
        except ConnectionClosed as e:
            logger.warning("el_tts_ws_closed", error=str(e))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("el_tts_reader_error", error=str(e), exc_info=True)
        finally:
            self._ws_open = False
            for q in list(self._audio_queues.values()):
                with contextlib.suppress(Exception):
                    q.put_nowait(None)

    async def _keepalive_loop(self, interval: float) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return
            except TimeoutError:
                pass
            try:
                if self._ws is not None and self._ws_open:
                    await self._ws.send(json.dumps({"context_id": "_ka", "text": ""}))
            except Exception:
                return

    # -- public interface ---------------------------------------------------

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not self._api_key or not self._out:
            raise RuntimeError("Call connect() before synthesize().")
        stripped = text.strip()
        if not stripped:
            return

        await self._ensure_ws()

        self._ctx_counter += 1
        ctx_id = f"ctx_{self._ctx_counter}"
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._audio_queues[ctx_id] = queue
        self._active_contexts.add(ctx_id)

        chunk_count = 0
        try:
            logger.debug(
                "el_tts_context_send",
                context_id=ctx_id,
                text_chars=len(stripped),
                text_preview=stripped[:120],
            )
            await self._ws.send(
                json.dumps(
                    {
                        "text": stripped + " ",
                        "context_id": ctx_id,
                        "flush": True,
                    }
                )
            )
            # Close the context right after flush so ElevenLabs finishes
            # generating audio for the buffered text and then sends is_final.
            # Without this, is_final never arrives and synthesize deadlocks.
            await self._ws.send(
                json.dumps(
                    {
                        "context_id": ctx_id,
                        "close_context": True,
                    }
                )
            )

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                if msg.get("audio"):
                    chunk_count += 1
                    yield base64.b64decode(msg["audio"])
                if msg.get("is_final") or msg.get("isFinal"):
                    break
        finally:
            self._audio_queues.pop(ctx_id, None)
            self._active_contexts.discard(ctx_id)
            logger.debug(
                "el_tts_context_done",
                context_id=ctx_id,
                audio_chunks=chunk_count,
            )

    async def close(self) -> None:
        await self._teardown_ws()
        self._out = None
