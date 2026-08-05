from collections import deque
from unittest.mock import AsyncMock, patch

import ormsgpack
import pytest
from pydantic import SecretStr
from timbal.voice.config import DEFAULT_VOICE_ID
from timbal.voice.fish_audio import (
    DEFAULT_TTS_MODEL,
    FishAudioStreamTTS,
    _resolve_api_key,
    build_start_request,
    effective_reference_id,
    effective_tts_model,
)
from timbal.voice.providers import AudioOutputConfig


def _cfg(**kwargs) -> AudioOutputConfig:
    return AudioOutputConfig(**kwargs)


def test_resolve_api_key_explicit_and_secret():
    assert _resolve_api_key("plain") == "plain"
    assert _resolve_api_key(SecretStr("secret")) == "secret"


def test_resolve_api_key_env(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "env-key")
    assert _resolve_api_key(None) == "env-key"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FISH_API_KEY"):
        _resolve_api_key(None)


def test_effective_tts_model_swaps_foreign_ids():
    assert effective_tts_model(_cfg(model=None)) == DEFAULT_TTS_MODEL
    assert effective_tts_model(_cfg(model="eleven_flash_v2_5")) == DEFAULT_TTS_MODEL
    assert effective_tts_model(_cfg(model="faseeh-v1-preview")) == DEFAULT_TTS_MODEL
    assert effective_tts_model(_cfg(model="s1")) == "s1"
    assert effective_tts_model(_cfg(model="s2.1-pro-free")) == "s2.1-pro-free"


def test_effective_reference_id(monkeypatch):
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    # Foreign/empty voice → None → Fish Audio's platform default voice.
    assert effective_reference_id(_cfg(voice=None)) is None
    assert effective_reference_id(_cfg(voice=DEFAULT_VOICE_ID)) is None
    assert effective_reference_id(_cfg(voice="9a9cf47702da476aa4629e2506d4a857")) == "9a9cf47702da476aa4629e2506d4a857"
    monkeypatch.setenv("FISH_VOICE_ID", "env-voice")
    assert effective_reference_id(_cfg(voice=None)) == "env-voice"


def test_build_start_request(monkeypatch):
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    req = build_start_request(
        _cfg(voice="voice-1", sample_rate=16000, extra={"temperature": 0.6, "speed": 1.1}),
    )
    assert req == {
        "text": "",
        "format": "pcm",
        "sample_rate": 16000,
        "latency": "balanced",
        "prosody": {"speed": 1.1, "volume": 0.0},
        "reference_id": "voice-1",
        "temperature": 0.6,
    }
    # No reference_id key at all when the platform default voice is used.
    assert "reference_id" not in build_start_request(_cfg(voice=None, sample_rate=16000))


class FakeWS:
    """Scripted Fish Audio live socket: iteration serves msgpack frames."""

    def __init__(self, frames: list[dict]) -> None:
        self._frames = deque(ormsgpack.packb(f) for f in frames)
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw: bytes) -> None:
        self.sent.append(ormsgpack.unpackb(raw))

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.popleft()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_stream_protocol_roundtrip(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    pcm = b"\x01\x02" * 8
    fake = FakeWS(
        [
            {"event": "audio", "audio": pcm},
            {"event": "audio", "audio": pcm},
            {"event": "finish", "reason": "stop"},
        ]
    )

    tts = FishAudioStreamTTS()
    await tts.connect(_cfg(model="eleven_flash_v2_5", voice=DEFAULT_VOICE_ID, sample_rate=16000))

    with patch("timbal.voice.fish_audio.ws_connect", AsyncMock(return_value=fake)) as connect_mock:
        stream = tts.open_stream()
        await stream.feed("Hello, ")
        await stream.feed("world.")
        await stream.end()
        chunks = [chunk async for chunk in stream.audio()]

    assert chunks == [pcm, pcm]
    assert fake.closed

    uri = connect_mock.await_args.args[0]
    assert uri == "wss://api.fish.audio/v1/tts/live"
    headers = connect_mock.await_args.kwargs["additional_headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["model"] == DEFAULT_TTS_MODEL

    start, text1, text2, flush, stop = fake.sent
    assert start["event"] == "start"
    assert start["request"]["format"] == "pcm"
    assert start["request"]["sample_rate"] == 16000
    assert "reference_id" not in start["request"]
    assert text1 == {"event": "text", "text": "Hello, "}
    assert text2 == {"event": "text", "text": "world."}
    assert flush == {"event": "flush"}
    assert stop == {"event": "stop"}


@pytest.mark.asyncio
async def test_finish_error_raises(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    fake = FakeWS([{"event": "finish", "reason": "error"}])

    tts = FishAudioStreamTTS()
    await tts.connect(_cfg(voice="voice-1"))

    with patch("timbal.voice.fish_audio.ws_connect", AsyncMock(return_value=fake)):
        stream = tts.open_stream()
        await stream.feed("Hello")
        await stream.end()
        with pytest.raises(RuntimeError, match="reason=error"):
            async for _ in stream.audio():
                pass


@pytest.mark.asyncio
async def test_synthesize_wraps_stream(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    pcm = b"\x00\x01" * 4
    fake = FakeWS(
        [
            {"event": "audio", "audio": pcm},
            {"event": "finish", "reason": "stop"},
        ]
    )

    tts = FishAudioStreamTTS()
    await tts.connect(_cfg(voice="voice-1"))

    with patch("timbal.voice.fish_audio.ws_connect", AsyncMock(return_value=fake)):
        chunks = [chunk async for chunk in tts.synthesize("Hello")]

    assert chunks == [pcm]


@pytest.mark.asyncio
async def test_http_402_rejection_gives_actionable_error(monkeypatch):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    monkeypatch.setenv("FISH_API_KEY", "test-key")
    rejection = InvalidStatus(Response(402, "Payment Required", Headers()))

    tts = FishAudioStreamTTS()
    await tts.connect(_cfg(voice="voice-1"))

    with patch("timbal.voice.fish_audio.ws_connect", AsyncMock(side_effect=rejection)):
        stream = tts.open_stream()
        with pytest.raises(RuntimeError, match="HTTP 402.*s2.1-pro-free"):
            await stream.feed("Hello")


@pytest.mark.asyncio
async def test_abort_unblocks_audio(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    tts = FishAudioStreamTTS()
    await tts.connect(_cfg(voice="voice-1"))

    stream = tts.open_stream()
    await stream.abort()
    chunks = [chunk async for chunk in stream.audio()]
    assert chunks == []


@pytest.mark.asyncio
async def test_unknown_events_ignored(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    pcm = b"\x07\x08"
    fake = FakeWS(
        [
            {"event": "log", "message": "future extension"},
            {"event": "audio", "audio": pcm},
            {"event": "finish", "reason": "stop"},
        ]
    )

    tts = FishAudioStreamTTS()
    await tts.connect(_cfg(voice="voice-1"))

    with patch("timbal.voice.fish_audio.ws_connect", AsyncMock(return_value=fake)):
        chunks = [chunk async for chunk in tts.synthesize("Hi")]

    assert chunks == [pcm]
