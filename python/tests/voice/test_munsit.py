import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr
from timbal.voice.config import DEFAULT_VOICE_ID
from timbal.voice.munsit import (
    DEFAULT_MUNSIT_VOICE_ID,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    MunsitStreamSTT,
    MunsitStreamTTS,
    _resolve_api_key,
    effective_sample_rate,
    effective_stt_model,
    effective_tts_model,
    effective_voice_id,
)
from timbal.voice.providers import AudioInputConfig, AudioOutputConfig


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
    # Out of range raises — a silent remap would desync session playback,
    # which is clocked at config.sample_rate.
    with pytest.raises(ValueError, match="8000-48000"):
        effective_sample_rate(_cfg(sample_rate=96000))


@pytest.mark.asyncio
async def test_connect_rejects_unsupported_sample_rate(monkeypatch):
    monkeypatch.setenv("MUNSIT_API_KEY", "test-key")
    tts = MunsitStreamTTS()
    with pytest.raises(ValueError, match="8000-48000"):
        await tts.connect(_cfg(sample_rate=96000))


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


async def _connected(
    response: FakeResponse, config: AudioOutputConfig, monkeypatch
) -> tuple[MunsitStreamTTS, FakeClient]:
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


# ---------------------------------------------------------------------------
# Streaming STT (WSS /api/v1/listen)
# ---------------------------------------------------------------------------


def _stt_cfg(**kwargs) -> AudioInputConfig:
    return AudioInputConfig(**kwargs)


def _drain(stt: MunsitStreamSTT) -> list:
    events = []
    while not stt._queue.empty():
        events.append(stt._queue.get_nowait())
    return events


def _results(text: str, *, is_final: bool, speech_final: bool = False, **kw) -> dict:
    return {
        "type": "Results",
        "channel": 0,
        "turn_id": 1,
        "transcript": text,
        "is_final": is_final,
        "speech_final": speech_final,
        **kw,
    }


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False

    async def send(self, payload) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True


class TestMunsitSttUri:
    def _parse(self, config: AudioInputConfig) -> dict:
        uri = MunsitStreamSTT()._build_uri(config)
        parsed = urlparse(uri)
        assert parsed.scheme == "wss"
        assert parsed.path == "/api/v1/listen"
        return parse_qs(parsed.query)

    def test_defaults(self) -> None:
        q = self._parse(_stt_cfg())
        assert q["model"] == [DEFAULT_STT_MODEL]
        assert q["encoding"] == ["linear16"]
        assert q["sample_rate"] == ["16000"]
        assert "language" not in q

    def test_foreign_model_replaced(self) -> None:
        # Only stt_provider switched; env stt_model still Scribe's / Flux's.
        assert self._parse(_stt_cfg(model="scribe_v2_realtime"))["model"] == [DEFAULT_STT_MODEL]
        assert self._parse(_stt_cfg(model="flux-general-multi"))["model"] == [DEFAULT_STT_MODEL]

    def test_munsit_models_kept(self) -> None:
        assert self._parse(_stt_cfg(model="munsit"))["model"] == ["munsit"]
        assert self._parse(_stt_cfg(model="munsit-en-ar"))["model"] == ["munsit-en-ar"]

    def test_language_ar_passes_others_dropped(self) -> None:
        # `ar` is the only supported value in v1; invalid params are fatal 4002.
        assert self._parse(_stt_cfg(language="ar"))["language"] == ["ar"]
        assert self._parse(_stt_cfg(language="ar-AE"))["language"] == ["ar"]
        assert "language" not in self._parse(_stt_cfg(language="en"))

    def test_extra_passthrough_filtered(self) -> None:
        q = self._parse(
            _stt_cfg(
                extra={
                    "endpointing": 300,
                    "smart_turn": True,
                    "hotwords": "فصيح,منصت",
                    # Scribe-only knob must not leak into the query (fatal 4002).
                    "commit_strategy": "vad",
                    "vad_silence_threshold_secs": 1.2,
                }
            )
        )
        assert q["endpointing"] == ["300"]
        assert q["smart_turn"] == ["true"]
        assert q["hotwords"] == ["فصيح,منصت"]
        assert "commit_strategy" not in q
        assert "vad_silence_threshold_secs" not in q

    def test_host_override(self) -> None:
        uri = MunsitStreamSTT()._build_uri(_stt_cfg(extra={"stt_host": "munsit.internal"}))
        assert uri.startswith("wss://munsit.internal/api/v1/listen?")

    def test_unsupported_sample_rate_negotiates_16k_wire(self) -> None:
        # The session applies one sample_rate to both STT and TTS, and 48 kHz
        # is the documented rate for engine-native Munsit TTS — the STT leg
        # must negotiate a legal wire rate and resample, not refuse to start.
        assert self._parse(_stt_cfg(sample_rate=48_000))["sample_rate"] == ["16000"]
        assert self._parse(_stt_cfg(sample_rate=8_000))["sample_rate"] == ["8000"]

    def test_unsupported_rate_with_compressed_encoding_raises(self) -> None:
        # Can't resample a non-PCM16 stream client-side — fail at session
        # start, not with a fatal 4002 mid-connect.
        with pytest.raises(ValueError, match="cannot resample"):
            MunsitStreamSTT()._build_uri(_stt_cfg(sample_rate=48_000, encoding="mulaw"))


@pytest.mark.asyncio
async def test_stt_connect_requires_api_key(monkeypatch):
    monkeypatch.delenv("MUNSIT_API_KEY", raising=False)
    stt = MunsitStreamSTT()
    with pytest.raises(ValueError, match="MUNSIT_API_KEY"):
        await stt.connect(_stt_cfg())


class TestMunsitSttEventMapping:
    async def test_interim_is_partial(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("مرحبا", is_final=False))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("partial", "مرحبا")]

    async def test_speech_final_commits(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("لا تتكلم هكذا", is_final=True, speech_final=True))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "لا تتكلم هكذا")]

    async def test_forced_split_does_not_commit(self) -> None:
        """is_final without speech_final is the ~60s forced split — the text is
        stable but the speaker is still talking. Committing it would answer a
        user mid-sentence."""
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("first sixty seconds", is_final=True, speech_final=False))
        assert _drain(stt) == []
        assert stt._segments == ["first sixty seconds"]

    async def test_forced_split_then_speech_final_commits_joined(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("first sixty seconds", is_final=True, speech_final=False))
        await stt._handle_message(_results("and the ending", is_final=True, speech_final=True, turn_id=2))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "first sixty seconds and the ending")]
        assert stt._segments == []

    async def test_partial_includes_buffered_segments(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("first sixty seconds", is_final=True, speech_final=False))
        await stt._handle_message(_results("and the", is_final=False, turn_id=2))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("partial", "first sixty seconds and the")]

    async def test_utterance_end_after_speech_final_does_not_double_commit(self) -> None:
        """UtteranceEnd fires right after the final Results of a completed turn;
        the speech_final frame already committed, so it must be a no-op."""
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("نعم", is_final=True, speech_final=True))
        await stt._handle_message({"type": "UtteranceEnd", "channel": 0, "turn_id": 1, "last_word_end": 1.2})
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "نعم")]

    async def test_utterance_end_flushes_leftover_segments(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("hello there", is_final=True, speech_final=False))
        await stt._handle_message({"type": "UtteranceEnd", "channel": 0, "turn_id": 1})
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "hello there")]

    async def test_empty_transcripts_skipped(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message(_results("", is_final=False))
        await stt._handle_message(_results("  ", is_final=True, speech_final=True))
        assert _drain(stt) == []

    async def test_enrichment_and_speech_started_emit_nothing(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message({"type": "SpeechStarted", "channel": 0, "ts": 4.31})
        await stt._handle_message({"type": "Gender", "channel": 0, "turn_id": 1, "label": "female", "score": 0.99})
        await stt._handle_message({"type": "Sentiment", "channel": 0, "turn_id": 1, "label": "negative", "score": 0.87})
        assert _drain(stt) == []

    async def test_metadata_emits_nothing(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message({"type": "Metadata", "session_id": "s1", "model": "munsit-v2"})
        await stt._handle_message(
            {"type": "Metadata", "session_id": "s1", "audio_seconds_billed": 12.4, "turn_count": 3}
        )
        assert _drain(stt) == []

    async def test_recoverable_error_logged_fatal_surfaces(self) -> None:
        stt = MunsitStreamSTT()
        await stt._handle_message({"type": "Error", "code": 4002, "message": "bad Configure", "recoverable": True})
        assert _drain(stt) == []
        await stt._handle_message(
            {"type": "Error", "code": 1008, "message": "insufficient balance", "recoverable": False}
        )
        events = _drain(stt)
        assert len(events) == 1
        assert events[0].type == "error"
        assert "insufficient balance" in events[0].text

    async def test_commit_is_noop(self) -> None:
        """Munsit's turn machine owns endpointing — no client force-commit."""
        stt = MunsitStreamSTT()
        stt._ws = _FakeWs()
        await stt.commit()
        assert stt._ws.sent == []


@pytest.mark.asyncio
async def test_stt_close_sends_close_stream_and_closes_ws():
    stt = MunsitStreamSTT()
    ws = _FakeWs()
    stt._ws = ws
    await stt.close()
    assert [json.loads(m) for m in ws.sent if isinstance(m, str)] == [{"type": "CloseStream"}]
    assert ws.closed
    assert stt._ws is None


@pytest.mark.asyncio
async def test_stt_connect_48k_session_resamples_to_16k_wire(monkeypatch):
    """A 48 kHz session (the rate engine-native Munsit TTS wants, shared with
    STT by the session config) must connect and downsample the mic leg
    client-side instead of failing before the socket opens."""
    pytest.importorskip("av")
    monkeypatch.setenv("MUNSIT_API_KEY", "k")

    class _IterFakeWs(_FakeWs):
        def __aiter__(self):
            return self

        async def __anext__(self):
            # Behave like the real server: stay open until CloseStream.
            while not any(isinstance(m, str) and "CloseStream" in m for m in self.sent):
                await asyncio.sleep(0.005)
            raise StopAsyncIteration

    ws = _IterFakeWs()
    captured: dict = {}

    async def fake_connect(uri, **_kwargs):
        captured["uri"] = uri
        return ws

    import websockets.asyncio.client as ws_client

    monkeypatch.setattr(ws_client, "connect", fake_connect)

    stt = MunsitStreamSTT()
    await stt.connect(_stt_cfg(sample_rate=48_000))
    try:
        assert "sample_rate=16000" in captured["uri"]
        assert stt._resampler is not None
        # Flush cadence sized for the 16 kHz wire, not the 48 kHz session.
        assert stt._flush_bytes == int(16_000 * 0.08 * 2)
        for _ in range(10):
            await stt.push_audio(b"\x00\x00" * 4_800)  # 100ms @ 48 kHz
    finally:
        await stt.close()
    wire = sum(len(m) for m in ws.sent if isinstance(m, (bytes, bytearray)))
    expected = 10 * 9_600 // 3  # 1s of audio at the 16 kHz wire rate
    # Tolerance: sub-threshold tail not flushed at close + FIR filter delay.
    assert expected - 4_096 <= wire <= expected


@pytest.mark.asyncio
async def test_stt_connect_16k_session_has_no_resampler(monkeypatch):
    monkeypatch.setenv("MUNSIT_API_KEY", "k")

    class _IterFakeWs(_FakeWs):
        def __aiter__(self):
            return self

        async def __anext__(self):
            while not any(isinstance(m, str) and "CloseStream" in m for m in self.sent):
                await asyncio.sleep(0.005)
            raise StopAsyncIteration

    async def fake_connect(_uri, **_kwargs):
        return _IterFakeWs()

    import websockets.asyncio.client as ws_client

    monkeypatch.setattr(ws_client, "connect", fake_connect)

    stt = MunsitStreamSTT()
    await stt.connect(_stt_cfg(sample_rate=16_000))
    try:
        assert stt._resampler is None
    finally:
        await stt.close()


def test_stt_effective_model_helper():
    assert effective_stt_model(None) == DEFAULT_STT_MODEL
    assert effective_stt_model("scribe_v2_realtime") == DEFAULT_STT_MODEL
    assert effective_stt_model("flux-general-multi") == DEFAULT_STT_MODEL
    assert effective_stt_model("munsit") == "munsit"
    assert effective_stt_model("munsit-en-ar") == "munsit-en-ar"


def test_stt_native_eou_capability():
    # The server keys turn-detector forcing and VAD-endpointing disable off this.
    assert MunsitStreamSTT.native_eou is True
