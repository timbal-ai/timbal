"""Fish Audio streaming TTS for :class:`~timbal.voice.VoiceSession`.

Uses the official live TTS WebSocket:

* ``wss://api.fish.audio/v1/tts/live`` — MessagePack frames

One connection per agent reply: a ``start`` event carries the TTS request
(``format: pcm`` at the session sample rate), ``text`` events stream the reply
incrementally, ``flush`` + ``stop`` end it, and the server replies with
``audio`` events followed by a terminal ``finish`` event.

Requires ``websockets`` + ``ormsgpack`` (``pip install timbal[server]``) and
``FISH_API_KEY``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import Any

import structlog
from pydantic import SecretStr
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from .config import DEFAULT_VOICE_ID as _ELEVENLABS_DEFAULT_VOICE_ID
from .providers import (
    AudioOutputConfig,
    TextToSpeech,
    TTSStream,
)

logger = structlog.get_logger("timbal.voice.fish_audio")

DEFAULT_TTS_MODEL = "s2.1-pro"
_KNOWN_MODELS = ("s1", "s2-pro", "s2.1-pro", "s2.1-pro-free")

_DEFAULT_HOST = "api.fish.audio"


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    if isinstance(explicit, SecretStr):
        return explicit.get_secret_value()
    if explicit:
        return explicit
    key = os.environ.get("FISH_API_KEY")
    if not key:
        raise ValueError("Set FISH_API_KEY or pass api_key to the provider.")
    return key


def effective_tts_model(config: AudioOutputConfig) -> str:
    """Model id actually sent to Fish Audio (foreign leftovers swapped out).

    The server-wide ``tts_model`` default is ElevenLabs' — a session that only
    switched ``tts_provider`` must not put ``eleven_flash_v2_5`` on the Fish
    Audio wire. Unknown ids also fall back: the server would silently degrade
    to its own default anyway, so make the substitution explicit.
    """
    m = (config.model or "").strip()
    if m in _KNOWN_MODELS:
        return m
    return DEFAULT_TTS_MODEL


def effective_reference_id(config: AudioOutputConfig) -> str | None:
    """Fish Audio voice model id, or None for the platform default voice.

    The server-wide ``voice`` default is an ElevenLabs voice id; drop it so
    Fish Audio picks its own default unless the caller chose a real reference.
    """
    v = (config.voice or "").strip()
    if not v or v == _ELEVENLABS_DEFAULT_VOICE_ID:
        return os.environ.get("FISH_VOICE_ID") or None
    return v


def build_start_request(config: AudioOutputConfig) -> dict[str, Any]:
    """The ``request`` payload of the ``start`` event (HTTP TTS API fields)."""
    extra = dict(config.extra)
    request: dict[str, Any] = {
        "text": "",
        "format": "pcm",
        "sample_rate": config.sample_rate,
        # "balanced" trades a little quality for latency — the right default
        # for live voice; override via tts_extra for narration-grade output.
        "latency": str(extra.get("latency", "balanced")),
        "prosody": {
            "speed": float(extra.get("speed", 1.0)),
            "volume": float(extra.get("volume", 0.0)),
        },
        # Explicit (matches the server default): previous audio conditions
        # later chunks, keeping one consistent voice across a reply's flushes.
        "condition_on_previous_chunks": bool(extra.get("condition_on_previous_chunks", True)),
    }
    reference_id = effective_reference_id(config)
    if reference_id:
        request["reference_id"] = reference_id
    for key in ("temperature", "top_p", "chunk_length", "normalize"):
        if key in extra:
            request[key] = extra[key]
    return request


class FishAudioStreamTTS(TextToSpeech):
    """Fish Audio TTS via the live WebSocket, one connection per reply.

    ``open_stream`` returns a fresh :class:`_FishAudioTTSStream` per agent
    reply, so all flushes of a reply share one session and voice consistency
    is kept via ``condition_on_previous_chunks`` (sent explicitly in the
    ``start`` request). ``synthesize`` is the per-segment fallback
    (stream = feed once + end).
    """

    provider_id = "fishaudio"

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._api_key: str | None = None
        self._out: AudioOutputConfig | None = None

    async def connect(self, config: AudioOutputConfig) -> None:
        self._api_key = _resolve_api_key(self._api_key_explicit)
        # Fail at session start, not first reply, if the extra is missing.
        import ormsgpack  # noqa: F401

        self._out = config

    def open_stream(self) -> _FishAudioTTSStream:
        if not self._api_key or not self._out:
            raise RuntimeError("Call connect() before open_stream().")
        return _FishAudioTTSStream(self)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not self._api_key or not self._out:
            raise RuntimeError("Call connect() before synthesize().")
        if not text.strip():
            return
        stream = self.open_stream()
        await stream.feed(text)
        await stream.end()
        async for chunk in stream.audio():
            yield chunk

    async def close(self) -> None:
        self._out = None


class _FishAudioTTSStream(TTSStream):
    """One Fish Audio WS session fed incrementally over an agent reply.

    Lifecycle: lazy-connect on first ``feed`` (``start`` event carries the TTS
    request), ``end`` sends ``flush`` + ``stop``; ``audio()`` terminates on the
    server's ``finish`` event.
    """

    def __init__(self, tts: FishAudioStreamTTS) -> None:
        self._tts = tts
        self._ws: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        # Created eagerly so ``audio()`` can be iterated before the first feed
        # opens the connection (the session starts the pump task immediately).
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._ended = False
        self._aborted = False
        self._chunks = 0
        self._last_error: str | None = None

    async def _send(self, payload: dict[str, Any]) -> None:
        import ormsgpack

        await self._ws.send(ormsgpack.packb(payload))

    async def _open(self) -> None:
        tts = self._tts
        cfg = tts._out
        assert cfg is not None and tts._api_key is not None
        host = str(cfg.extra.get("tts_host", _DEFAULT_HOST))
        model = effective_tts_model(cfg)

        uri = f"wss://{host}/v1/tts/live"
        logger.debug("fish_tts_ws_connecting", uri=uri, model=model)
        try:
            self._ws = await ws_connect(
                uri,
                additional_headers={
                    "Authorization": f"Bearer {tts._api_key}",
                    "model": model,
                },
            )
        except InvalidStatus as e:
            status = e.response.status_code
            if status == 402:
                hint = (
                    f" — insufficient wallet balance for model {model!r}; use 's2.1-pro-free' or top up at fish.audio"
                )
            elif status in (401, 403):
                hint = " — check FISH_API_KEY"
            else:
                hint = ""
            raise RuntimeError(f"Fish Audio rejected the connection (HTTP {status}){hint}") from e
        await self._send({"event": "start", "request": build_start_request(cfg)})
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.debug("fish_tts_ws_connected")

    async def _read_loop(self) -> None:
        import ormsgpack

        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = ormsgpack.unpackb(raw)
                event = msg.get("event")
                if event == "finish":
                    if msg.get("reason") == "error":
                        self._last_error = "Fish Audio finished with reason=error"
                    break
                # Unknown events are ignored per protocol docs.
                await self._queue.put(msg)
        except ConnectionClosed as e:
            if not self._ended and not self._aborted:
                self._last_error = str(e)
                logger.warning("fish_tts_ws_closed_unrequested", error=str(e))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._last_error = str(e)
            logger.error("fish_tts_reader_error", error=str(e), exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                self._queue.put_nowait(None)

    async def feed(self, text: str) -> None:
        if self._ended or self._aborted or not text.strip():
            return
        if self._ws is None:
            await self._open()
        logger.debug("fish_tts_stream_feed", text_chars=len(text), text_preview=text[:120])
        await self._send({"event": "text", "text": text})

    async def end(self) -> None:
        if self._ended or self._aborted:
            return
        self._ended = True
        if self._ws is None:
            # Nothing was ever fed — unblock audio() directly.
            self._queue.put_nowait(None)
            return
        try:
            # flush forces synthesis of buffered text; stop makes the server
            # drain remaining audio and emit finish, which terminates audio().
            await self._send({"event": "flush"})
            await self._send({"event": "stop"})
        except Exception as e:
            logger.warning("fish_tts_stream_end_failed", error=str(e))
            self._queue.put_nowait(None)

    async def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._send({"event": "stop"})
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._queue.put_nowait(None)

    async def audio(self) -> AsyncIterator[bytes]:
        try:
            while True:
                msg = await self._queue.get()
                if msg is None:
                    if self._last_error and not self._aborted:
                        raise RuntimeError(f"Fish Audio TTS failed: {self._last_error}")
                    return
                audio = msg.get("audio")
                if audio:
                    self._chunks += 1
                    yield bytes(audio)
        finally:
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader_task
            self._reader_task = None
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.close()
                self._ws = None
            logger.debug("fish_tts_stream_done", audio_chunks=self._chunks)
