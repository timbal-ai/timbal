"""Deepgram Aura streaming TTS over /v1/speak (no provider SDK required)."""

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

from .deepgram import _resolve_api_key
from .providers import AudioOutputConfig, TextToSpeech, TTSStream

logger = structlog.get_logger("timbal.voice.deepgram_tts")
DEFAULT_TTS_MODEL = "aura-2-thalia-en"


def effective_tts_model(config: AudioOutputConfig) -> str:
    """Aura combines model, voice and language in one id; ignore foreign defaults.

    A full Aura ``model`` wins over ``voice``, then DEEPGRAM_TTS_MODEL, then
    Thalia. Accept future Aura voice ids without maintaining a frozen catalog.
    """
    for value in (config.model, config.voice, os.getenv("DEEPGRAM_TTS_MODEL")):
        if value and value.strip().startswith("aura-"):
            return value.strip()
    return DEFAULT_TTS_MODEL


def build_tts_uri(config: AudioOutputConfig) -> str:
    # VoiceSession transports, playback accounting and recording expect PCM16.
    if config.encoding not in ("pcm_s16le", "linear16", ""):
        raise ValueError("Deepgram voice sessions require pcm_s16le (linear16) output.")
    if config.sample_rate not in (8000, 16000, 24000, 32000, 48000):
        raise ValueError("Deepgram TTS sample_rate must be 8000, 16000, 24000, 32000 or 48000.")
    params: dict[str, Any] = {
        "model": effective_tts_model(config),
        "encoding": "linear16",
        "sample_rate": config.sample_rate,
    }
    # Deliberately drop ElevenLabs defaults and output-format overrides: raw
    # PCM must stay consistent with the session's advertised format.
    for key in ("speed", "mip_opt_out"):
        value = config.extra.get(key)
        if value is not None:
            params[key] = str(value).lower() if isinstance(value, bool) else value
    host = config.extra.get("tts_host", "api.deepgram.com")
    return f"wss://{host}/v1/speak?{urlencode(params)}"


class DeepgramStreamTTS(TextToSpeech):
    """Aura/Aura-2: one incremental text stream per reply.

    Aura has no multiplexed context ids. Separate sockets isolate replies
    (including filler/greeting synthesis) and discard in-flight audio on
    barge-in without waiting for a Clear acknowledgement before the next turn.
    """

    provider_id = "deepgram"

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._uri: str | None = None
        self._streams: set[_DeepgramTTSStream] = set()

    async def connect(self, config: AudioOutputConfig) -> None:
        await self.close()
        self._api_key = _resolve_api_key(self._api_key_explicit)
        self._uri = build_tts_uri(config)

    def open_stream(self) -> TTSStream:
        if self._uri is None or self._api_key is None:
            raise RuntimeError("Call connect() before open_stream().")
        stream = _DeepgramTTSStream(self, self._uri, self._api_key)
        self._streams.add(stream)
        return stream

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        stream = self.open_stream()
        try:
            await stream.feed(text)
            await stream.end()
            async for chunk in stream.audio():
                yield chunk
        finally:
            await stream.abort()

    async def close(self) -> None:
        self._uri = None
        self._api_key = None
        await asyncio.gather(*(stream.abort() for stream in list(self._streams)))


class _DeepgramTTSStream(TTSStream):
    def __init__(self, tts: DeepgramStreamTTS, uri: str, api_key: str) -> None:
        self._tts = tts
        self._uri = uri
        self._api_key = api_key
        self._ws: Any = None
        self._opening: asyncio.Future[Any] | None = None
        self._reader: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        self._wire_lock = asyncio.Lock()
        self._ended = False
        self._aborted = False
        self._finished = False

    async def _send(self, message: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(message))

    async def feed(self, text: str) -> None:
        async with self._wire_lock:
            if self._ended or self._aborted or self._finished or not text.strip():
                return
            try:
                if self._ws is None:
                    self._opening = asyncio.ensure_future(
                        ws_connect(
                            self._uri,
                            additional_headers={"Authorization": f"Token {self._api_key}"},
                            open_timeout=10,
                            close_timeout=1,
                        )
                    )
                    try:
                        self._ws = await self._opening
                    finally:
                        self._opening = None
                    self._reader = asyncio.create_task(self._read_loop())
                # abort() can run while the handshake is pending.
                if self._aborted:
                    return
                await self._send({"type": "Speak", "text": text})
            except asyncio.CancelledError:
                self._queue.put_nowait(None)
                if not self._aborted:
                    raise
            except Exception as exc:
                self._queue.put_nowait(exc)
                raise

    async def end(self) -> None:
        async with self._wire_lock:
            if self._ended or self._aborted or self._finished:
                return
            self._ended = True
            if self._ws is None:
                self._queue.put_nowait(None)
                self._tts._streams.discard(self)
                return
            try:
                # One Flush per reply; frequent flushes degrade prosody and
                # hit Aura's per-connection flush limit.
                await self._send({"type": "Flush"})
            except Exception as exc:
                self._queue.put_nowait(exc)
                raise

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    if raw:
                        self._queue.put_nowait(raw)
                    continue
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "Flushed" and self._ended:
                    self._finished = True
                    return
                if kind == "Error":
                    raise RuntimeError(message.get("description") or message.get("message") or str(message))
                if kind == "Warning":
                    logger.warning("deepgram_tts_warning", code=message.get("code"), message=message.get("description"))
            if not self._aborted:
                raise RuntimeError("Deepgram TTS connection closed before Flushed acknowledgement.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._queue.put_nowait(exc)
        finally:
            self._queue.put_nowait(None)

    async def _cleanup(self) -> None:
        if self._reader is not None and self._reader is not asyncio.current_task():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        self._tts._streams.discard(self)

    async def abort(self) -> None:
        # Mark/drop queued audio before awaiting the wire lock so audio()
        # unblocks immediately, even if a handshake/send is in progress.
        self._aborted = True
        self._queue.put_nowait(None)
        if self._opening is not None:
            self._opening.cancel()
        async with self._wire_lock:
            await self._cleanup()

    async def audio(self) -> AsyncIterator[bytes]:
        try:
            while not self._aborted:
                message = await self._queue.get()
                if self._aborted or message is None:
                    return
                if isinstance(message, Exception):
                    raise RuntimeError(f"Deepgram TTS failed: {message}") from message
                yield message
        finally:
            await self.abort()
