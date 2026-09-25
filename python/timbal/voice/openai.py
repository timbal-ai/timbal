"""OpenAI transcription-only Realtime STT and streaming Speech API TTS.

The agent/LLM stays in Timbal. Both audio APIs use 24 kHz PCM16; adapters
resample to/from the session rate using the optional timbal[voice] extra.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

import httpx
from pydantic import SecretStr
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .._openai_audio import STT_MODELS, TTS_MODELS, VOICES, is_openai_stt_model, validate_speech
from .events import VoiceUsageEvent
from .providers import AudioInputConfig, AudioOutputConfig, SpeechToText, TextToSpeech, TranscriptEvent
from .telephony import PcmResampler

DEFAULT_STT_MODEL = "gpt-transcribe"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "coral"
WIRE_RATE = 24_000


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    key = explicit.get_secret_value() if isinstance(explicit, SecretStr) else explicit
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key to the provider.")
    return key


def is_stt_model(model: str | None) -> bool:
    return (model or "").strip() in STT_MODELS


def effective_stt_model(model: str | None) -> str:
    m = (model or "").strip()
    if is_stt_model(m):
        return m
    if is_openai_stt_model(m):
        raise ValueError(f"Unsupported OpenAI realtime transcription model: {m!r}")
    return DEFAULT_STT_MODEL


def effective_tts_model(config: AudioOutputConfig) -> str:
    m = (config.model or "").strip()
    if m in TTS_MODELS:
        return m
    if m.startswith(("tts-", "gpt-")):
        raise ValueError(f"Unsupported OpenAI speech model: {m!r}")
    return DEFAULT_TTS_MODEL


def effective_voice(config: AudioOutputConfig) -> str:
    for voice in (config.voice, os.getenv("OPENAI_TTS_VOICE")):
        if voice in VOICES:
            return voice
    return DEFAULT_VOICE


def _validate_pcm(config: AudioInputConfig | AudioOutputConfig) -> None:
    if config.encoding not in ("pcm_s16le", "linear16", "pcm16", ""):
        raise ValueError("OpenAI voice providers require PCM16 mono audio.")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")


def build_transcription_session(config: AudioInputConfig) -> dict[str, Any]:
    """GA Realtime transcription schema; never forward foreign provider extras."""
    model = effective_stt_model(config.model)
    transcription: dict[str, Any] = {"model": model}
    if config.language:
        if model == "gpt-transcribe":
            transcription["languages"] = [config.language]
        else:
            transcription["language"] = config.language
    if config.extra.get("prompt"):
        transcription["prompt"] = config.extra["prompt"]
    audio_input: dict[str, Any] = {
        "format": {"type": "audio/pcm", "rate": WIRE_RATE},
        "transcription": transcription,
        "turn_detection": {
            "type": "server_vad",
            "threshold": config.extra.get("threshold", 0.5),
            "prefix_padding_ms": config.extra.get("prefix_padding_ms", 300),
            "silence_duration_ms": config.extra.get("silence_duration_ms", 500),
        },
    }
    if "noise_reduction" in config.extra:
        kind = config.extra["noise_reduction"]
        if kind not in (None, "near_field", "far_field"):
            raise ValueError("noise_reduction must be near_field, far_field, or null")
        audio_input["noise_reduction"] = {"type": kind} if kind else None
    return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": audio_input}}}


class OpenAIRealtimeSTT(SpeechToText):
    """Realtime transcription with server VAD and optional local force-commit.

    Transcript deltas are accumulated per item. Completed turns are emitted
    in audio-commit order, even when transcription finishes out of order.
    Server VAD is a silence detector, so Timbal's turn detector stays active.
    """

    provider_id = "openai"

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._ws: Any = None
        self._receiver: asyncio.Task[None] | None = None
        self._wire_lock = asyncio.Lock()
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._closed = True
        self._resampler: PcmResampler | None = None
        self._remainder = b""
        self._uncommitted_bytes = 0
        self._commit_counter = 0
        self._commit_ids: deque[str] = deque(maxlen=128)
        self._order: deque[str] = deque()
        self._partials: dict[str, str] = {}
        self._completed: dict[str, str] = {}
        self._usage_scope = uuid4().hex
        self._model = DEFAULT_STT_MODEL
        self._pending_usage: set[str] = set()

    async def connect(self, config: AudioInputConfig) -> None:
        await self.close()
        _validate_pcm(config)
        update = build_transcription_session(config)
        self._model = effective_stt_model(config.model)
        self._usage_scope = uuid4().hex
        key = _resolve_api_key(self._api_key_explicit)
        self._resampler = PcmResampler(config.sample_rate, WIRE_RATE) if config.sample_rate != WIRE_RATE else None
        self._queue = asyncio.Queue()
        self._order.clear()
        self._partials.clear()
        self._completed.clear()
        self._commit_ids.clear()
        self._remainder = b""
        self._uncommitted_bytes = 0
        host = config.extra.get("stt_host", "api.openai.com")
        self._ws = await ws_connect(
            f"wss://{host}/v1/realtime?intent=transcription",
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
            close_timeout=1,
        )
        try:
            await self._ws.send(json.dumps(update))
            # Surface invalid model/options during startup, not on the first
            # user turn. Only start feeding audio after the update is accepted.
            async with asyncio.timeout(10):
                async for raw in self._ws:
                    event = json.loads(raw)
                    if event.get("type") == "error":
                        raise RuntimeError(f"OpenAI STT configuration failed: {event.get('error')}")
                    if event.get("type") == "session.updated":
                        break
                else:
                    raise RuntimeError("OpenAI STT closed before session.updated")
        except BaseException:
            await self.close()
            raise
        self._closed = False
        self._receiver = asyncio.create_task(self._receive_loop())

    async def _append(self, pcm: bytes) -> None:
        if not pcm or self._ws is None or self._closed:
            return
        await self._ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        )
        self._uncommitted_bytes += len(pcm)

    async def push_audio(self, chunk: bytes) -> None:
        if self._closed or not chunk:
            return
        async with self._wire_lock:
            raw = self._remainder + chunk
            end = len(raw) // 2 * 2
            pcm, self._remainder = raw[:end], raw[end:]
            if self._resampler is not None:
                pcm = self._resampler.process(pcm)
            with contextlib.suppress(ConnectionClosed):
                await self._append(pcm)

    async def commit(self) -> None:
        async with self._wire_lock:
            if self._closed or self._ws is None or not self._uncommitted_bytes:
                return
            # Include the resampler tail before checking the API's 100 ms
            # minimum; exactly 100 ms at 16 kHz otherwise looks too short.
            if self._resampler is not None:
                with contextlib.suppress(ConnectionClosed):
                    await self._append(self._resampler.flush())
            if self._uncommitted_bytes < WIRE_RATE // 10 * 2:
                return
            self._commit_counter += 1
            event_id = f"timbal_commit_{self._commit_counter}"
            self._commit_ids.append(event_id)
            with contextlib.suppress(ConnectionClosed):
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit", "event_id": event_id}))
            self._uncommitted_bytes = 0

    def _handle_message(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        item = event.get("item_id", "")
        previous_head = self._order[0] if self._order else None
        if kind == "input_audio_buffer.committed":
            # An acknowledgement can arrive after we sent the next turn.
            # Only the local commit resets our byte estimate; server-VAD
            # races can safely return the owned commit_empty error below.
            self._order.append(item)
            self._pending_usage.add(item)
        elif kind == "conversation.item.input_audio_transcription.delta":
            self._partials[item] = self._partials.get(item, "") + event.get("delta", "")
            if self._order and item == self._order[0]:
                self._queue.put_nowait(TranscriptEvent(type="partial", text=self._partials[item]))
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._completed[item] = event.get("transcript", "")
            self._pending_usage.discard(item)
            self._report_usage(item, event.get("usage"), event.get("content_index", 0))
        elif kind == "error":
            error = event.get("error", {})
            # A server VAD commit may win the race against local force-commit.
            if error.get("code") == "input_audio_buffer_commit_empty" and error.get("event_id") in self._commit_ids:
                self._commit_ids.remove(error["event_id"])
                return
            raise RuntimeError(f"OpenAI STT error: {error.get('message', error)}")
        elif kind == "conversation.item.input_audio_transcription.failed":
            raise RuntimeError(f"OpenAI STT transcription failed: {event.get('error')}")
        while self._order and self._order[0] in self._completed:
            item = self._order.popleft()
            text = self._completed.pop(item)
            self._partials.pop(item, None)
            if text.strip():
                self._queue.put_nowait(TranscriptEvent(type="committed", text=text))
        if self._order and self._order[0] != previous_head:
            text = self._partials.get(self._order[0])
            if text:
                self._queue.put_nowait(TranscriptEvent(type="partial", text=text))

    def _report_usage(self, item: str, usage: Any = None, content_index: int = 0) -> None:
        self._emit_usage(
            VoiceUsageEvent(
                usage_id=f"{self._usage_scope}:{item}:{content_index}",
                provider=self.provider_id,
                operation="stt",
                model=self._model,
                item_id=item,
                status="complete" if _valid_usage(usage) else "incomplete",
                usage=usage if isinstance(usage, dict) else None,
            )
        )

    def _report_pending_usage(self) -> None:
        for item in self._pending_usage:
            self._report_usage(item)
        self._pending_usage.clear()

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                self._handle_message(json.loads(raw))
            if not self._closed:
                raise RuntimeError("OpenAI STT connection closed unexpectedly")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                self._queue.put_nowait(TranscriptEvent(type="error", text=f"OpenAI STT connection closed: {exc}"))
        finally:
            self._report_pending_usage()
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            if event.type == "error":
                raise RuntimeError(event.text)
            if event.text:
                yield event

    async def close(self) -> None:
        self._closed = True
        if self._receiver is not None:
            self._receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver
            self._receiver = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        self._report_pending_usage()
        self._queue.put_nowait(None)


def _valid_usage(usage: Any) -> bool:
    """Accept reported counters, including legitimate zeros; never invent them."""
    if not isinstance(usage, dict):
        return False
    if usage.get("type") == "duration":
        seconds = usage.get("seconds")
        return type(seconds) in (int, float) and 0 <= seconds < float("inf")
    if usage.get("type", "tokens") != "tokens":
        return False
    return all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens", "total_tokens"))


async def _speech_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse SSE framing across arbitrary HTTP chunks, comments and CRLF."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield json.loads("\n".join(data))
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield json.loads("\n".join(data))


class OpenAIStreamTTS(TextToSpeech):
    """Speech API streaming audio, using VoiceSession's per-segment fallback.

    /audio/speech takes a complete text input, so open_stream() intentionally
    stays unsupported. Each synthesis owns its response and resampler; closing
    or cancelling it cannot leak audio into another reply.
    """

    provider_id = "openai"

    def __init__(self, api_key: str | SecretStr | None = None) -> None:
        self._api_key_explicit = api_key
        self._client: httpx.AsyncClient | None = None
        self._config: AudioOutputConfig | None = None
        self._responses: set[httpx.Response] = set()
        self._pending_usage: dict[str, Callable[[], None]] = {}

    async def connect(self, config: AudioOutputConfig) -> None:
        await self.close()
        _validate_pcm(config)
        validate_speech(effective_tts_model(config), effective_voice(config), config.extra.get("instructions"))
        if config.sample_rate != WIRE_RATE:
            # Fail at session start if the resampling extra is missing.
            PcmResampler(WIRE_RATE, config.sample_rate)
        key = _resolve_api_key(self._api_key_explicit)
        self._config = config.model_copy(deep=True)
        host = config.extra.get("tts_host", "api.openai.com")
        self._client = httpx.AsyncClient(
            base_url=f"https://{host}/v1/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(60, connect=10),
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        client, config = self._client, self._config
        if client is None or config is None:
            raise RuntimeError("Call connect() before synthesize().")
        if not text.strip():
            return
        model = effective_tts_model(config)
        payload: dict[str, Any] = {
            "model": model,
            "voice": effective_voice(config),
            "input": text,
            "response_format": "pcm",
        }
        structured = model.startswith("gpt-4o-mini-tts")
        if structured:
            payload["stream_format"] = "sse"
        if "speed" in config.extra:
            payload["speed"] = config.extra["speed"]
        if config.extra.get("instructions") and model.startswith("gpt-4o-mini-tts"):
            payload["instructions"] = config.extra["instructions"]
        resampler = PcmResampler(WIRE_RATE, config.sample_rate) if config.sample_rate != WIRE_RATE else None
        remainder = b""
        usage_id = uuid4().hex
        reported = False
        request_id = None

        def report(usage: Any = None) -> None:
            nonlocal reported
            if reported:
                return
            reported = True
            self._pending_usage.pop(usage_id, None)
            self._emit_usage(
                VoiceUsageEvent(
                    usage_id=usage_id,
                    provider=self.provider_id,
                    operation="tts",
                    model=model,
                    request_id=request_id,
                    status="complete" if _valid_usage(usage) else "incomplete",
                    usage=usage if isinstance(usage, dict) else None,
                )
            )

        async def chunks(response: httpx.Response) -> AsyncIterator[bytes]:
            if not structured:
                async for chunk in response.aiter_bytes():
                    yield chunk
                return
            async for event in _speech_events(response):
                kind = event.get("type")
                if kind == "speech.audio.delta":
                    yield base64.b64decode(event["audio"], validate=True)
                elif kind == "speech.audio.done":
                    report(event.get("usage"))
                    return
                elif kind in ("error", "speech.audio.error"):
                    raise RuntimeError(f"OpenAI TTS error: {event.get('error')}")
            raise RuntimeError("OpenAI TTS stream ended before speech.audio.done")

        if structured:
            self._pending_usage[usage_id] = report
        try:
            async with client.stream("POST", "audio/speech", json=payload) as response:
                request_id = response.headers.get("x-request-id")
                response.raise_for_status()
                self._responses.add(response)
                try:
                    async with contextlib.aclosing(chunks(response)) as stream:
                        async for chunk in stream:
                            if self._client is not client:
                                return
                            raw = remainder + chunk
                            end = len(raw) // 2 * 2
                            pcm, remainder = raw[:end], raw[end:]
                            if resampler is not None:
                                pcm = resampler.process(pcm)
                            if pcm:
                                yield pcm
                            if self._client is not client:
                                return
                    if self._client is not client:
                        return
                    if remainder:
                        raise RuntimeError("OpenAI TTS returned an incomplete PCM16 sample")
                    if resampler is not None:
                        tail = resampler.flush()
                        if tail:
                            yield tail
                finally:
                    self._responses.discard(response)
        finally:
            # Includes cancellation, generator.aclose(), HTTP errors and EOF.
            # Legacy character-priced models keep their existing metering path.
            if structured and not reported:
                report()

    async def close(self) -> None:
        client, self._client = self._client, None
        self._config = None
        for response in list(self._responses):
            await response.aclose()
        self._responses.clear()
        for report in list(self._pending_usage.values()):
            report()
        if client is not None:
            await client.aclose()
