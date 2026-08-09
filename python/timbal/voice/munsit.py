"""Munsit (Faseeh) streaming Arabic TTS for :class:`~timbal.voice.VoiceSession`.

Uses the HTTP chunked-streaming endpoint:

* ``POST /api/v1/text-to-speech/{model_id}`` with ``streaming: true`` —
  raw PCM16 chunks are yielded as they are generated.

Munsit also documents a TTS WebSocket (``/websocket/text-to-speech``), but as
of 2026-08 it accepts ``initConnection``/``text`` frames and never produces
audio (verified empirically across auth methods, flush flags and long texts),
so this provider deliberately uses HTTP streaming — which Munsit itself
recommends when the text of a segment is available upfront. No ``open_stream``
capability: the session falls back to per-segment ``synthesize``.

Requires ``MUNSIT_API_KEY``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import structlog
from pydantic import SecretStr

from .config import DEFAULT_VOICE_ID as _ELEVENLABS_DEFAULT_VOICE_ID
from .providers import (
    AudioOutputConfig,
    TextToSpeech,
)

logger = structlog.get_logger("timbal.voice.munsit")

DEFAULT_TTS_MODEL = "faseeh-v1-preview"
# Override with MUNSIT_VOICE_ID (cloned/custom voices are account-specific).
DEFAULT_MUNSIT_VOICE_ID = "ar-najdi-male-2"

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


def effective_sample_rate(config: AudioOutputConfig) -> int:
    """Sample rate actually requested (endpoint accepts 8000–48000 Hz).

    Out-of-range rates raise instead of remapping: the session clocks playback
    at ``config.sample_rate``, so silently requesting a different rate would
    desync audio (chipmunk/slow-motion speech).
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
