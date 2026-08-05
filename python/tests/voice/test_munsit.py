from contextlib import asynccontextmanager

import pytest
from pydantic import SecretStr
from timbal.voice.config import DEFAULT_VOICE_ID
from timbal.voice.munsit import (
    DEFAULT_MUNSIT_VOICE_ID,
    DEFAULT_TTS_MODEL,
    MunsitStreamTTS,
    _resolve_api_key,
    effective_sample_rate,
    effective_tts_model,
    effective_voice_id,
)
from timbal.voice.providers import AudioOutputConfig


def _cfg(**kwargs) -> AudioOutputConfig:
    return AudioOutputConfig(**kwargs)


def test_resolve_api_key_explicit_and_secret():
    assert _resolve_api_key("plain") == "plain"
    assert _resolve_api_key(SecretStr("secret")) == "secret"


def test_resolve_api_key_env(monkeypatch):
    monkeypatch.setenv("MUNSIT_API_KEY", "env-key")
    assert _resolve_api_key(None) == "env-key"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("MUNSIT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MUNSIT_API_KEY"):
        _resolve_api_key(None)


def test_effective_sample_rate():
    assert effective_sample_rate(_cfg(sample_rate=16000)) == 16000
    assert effective_sample_rate(_cfg(sample_rate=48000)) == 48000
    assert effective_sample_rate(_cfg(sample_rate=96000)) == 24000


def test_effective_tts_model_swaps_foreign_ids():
    assert effective_tts_model(_cfg(model=None)) == DEFAULT_TTS_MODEL
    assert effective_tts_model(_cfg(model="eleven_flash_v2_5")) == DEFAULT_TTS_MODEL
    assert effective_tts_model(_cfg(model="faseeh-v1-preview")) == "faseeh-v1-preview"


def test_effective_voice_id_swaps_elevenlabs_default(monkeypatch):
    monkeypatch.delenv("MUNSIT_VOICE_ID", raising=False)
    assert effective_voice_id(_cfg(voice=None)) == DEFAULT_MUNSIT_VOICE_ID
    assert effective_voice_id(_cfg(voice=DEFAULT_VOICE_ID)) == DEFAULT_MUNSIT_VOICE_ID
    assert effective_voice_id(_cfg(voice="ar-hijazi-female-1")) == "ar-hijazi-female-1"
    monkeypatch.setenv("MUNSIT_VOICE_ID", "cloned-voice")
    assert effective_voice_id(_cfg(voice=None)) == "cloned-voice"


class FakeResponse:
    def __init__(self, status_code: int, chunks: list[bytes] | None = None, body: bytes = b"") -> None:
        self.status_code = status_code
        self._chunks = chunks or []
        self._body = body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        return self._body


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.closed = False

    @asynccontextmanager
    async def stream(self, method, url, *, headers=None, json=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json})
        yield self._response

    async def aclose(self) -> None:
        self.closed = True


async def _connected(response: FakeResponse, config: AudioOutputConfig, monkeypatch) -> tuple[MunsitStreamTTS, FakeClient]:
    monkeypatch.setenv("MUNSIT_API_KEY", "test-key")
    tts = MunsitStreamTTS()
    await tts.connect(config)
    fake = FakeClient(response)
    await tts._client.aclose()
    tts._client = fake
    return tts, fake


@pytest.mark.asyncio
async def test_synthesize_builds_request_and_streams(monkeypatch):
    monkeypatch.delenv("MUNSIT_VOICE_ID", raising=False)
    pcm = [b"\x01\x02" * 8, b"\x03\x04" * 8]
    tts, fake = await _connected(
        FakeResponse(200, chunks=pcm),
        _cfg(model="eleven_flash_v2_5", voice=DEFAULT_VOICE_ID, sample_rate=16000, extra={"dialect": "fusha"}),
        monkeypatch,
    )

    chunks = [chunk async for chunk in tts.synthesize("مرحبا بك ")]
    assert chunks == pcm

    (call,) = fake.calls
    assert call["method"] == "POST"
    assert call["url"] == "https://api.munsit.com/api/v1/text-to-speech/faseeh-v1-preview"
    assert call["headers"] == {"x-api-key": "test-key"}
    assert call["json"] == {
        "voice_id": DEFAULT_MUNSIT_VOICE_ID,
        "text": "مرحبا بك",
        "stability": 0.5,
        "speed": 1.0,
        "streaming": True,
        "sample_rate": 16000,
        "dialect": "fusha",
    }


@pytest.mark.asyncio
async def test_synthesize_error_status_raises(monkeypatch):
    tts, _ = await _connected(
        FakeResponse(402, body=b'{"errorCode":40201,"errorMessage":"Insufficient wallet balance"}'),
        _cfg(voice="ar-najdi-male-2"),
        monkeypatch,
    )
    with pytest.raises(RuntimeError, match="402"):
        async for _ in tts.synthesize("مرحبا"):
            pass


@pytest.mark.asyncio
async def test_synthesize_empty_text_is_noop(monkeypatch):
    tts, fake = await _connected(FakeResponse(200), _cfg(voice="ar-najdi-male-2"), monkeypatch)
    chunks = [chunk async for chunk in tts.synthesize("   ")]
    assert chunks == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_synthesize_requires_connect():
    tts = MunsitStreamTTS(api_key="k")
    with pytest.raises(RuntimeError, match="connect"):
        async for _ in tts.synthesize("مرحبا"):
            pass


@pytest.mark.asyncio
async def test_no_open_stream_capability(monkeypatch):
    tts, _ = await _connected(FakeResponse(200), _cfg(voice="ar-najdi-male-2"), monkeypatch)
    assert tts.open_stream() is None


@pytest.mark.asyncio
async def test_close_shuts_client(monkeypatch):
    tts, fake = await _connected(FakeResponse(200), _cfg(voice="ar-najdi-male-2"), monkeypatch)
    await tts.close()
    assert fake.closed
    assert tts._client is None


def test_default_voice_settings_are_overridable():
    # tts_extra flows into AudioOutputConfig.extra; stability/speed ride there.
    cfg = _cfg(voice="ar-najdi-male-2", extra={"stability": 0.8, "speed": 1.1})
    assert cfg.extra["stability"] == 0.8
