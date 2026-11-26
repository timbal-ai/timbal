"""Gemini TTS adapter: prompt layout, voice/model guards, clause streaming, retry, abort."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from timbal.voice.gemini_tts import (
    DEFAULT_ACCENT,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    GeminiStreamTTS,
    build_prompt,
    effective_tts_model,
    effective_voice,
    split_clauses,
)
from timbal.voice.providers import AudioOutputConfig, resolve_tts

RATE = 24_000


def _pcm(ms: int, value: int = 1000) -> bytes:
    return value.to_bytes(2, "little", signed=True) * (RATE * ms // 1000)


def _sse(*chunks: bytes) -> bytes:
    lines = []
    for c in chunks:
        part = {"inlineData": {"mimeType": "audio/L16;rate=24000", "data": base64.b64encode(c).decode()}}
        lines.append("data: " + json.dumps({"candidates": [{"content": {"parts": [part]}}]}) + "\n\n")
    return "".join(lines).encode()


class Recorder:
    """Mock transport: records every request and answers from ``plan`` (one entry per request, last repeats)."""

    def __init__(self, *plan: httpx.Response | bytes | Exception) -> None:
        self.requests: list[httpx.Request] = []
        self.plan = list(plan) or [_sse(_pcm(200))]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.plan[min(len(self.requests) - 1, len(self.plan) - 1)]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, content=item)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def texts(self) -> list[str]:
        return [json.loads(r.content)["contents"][0]["parts"][0]["text"] for r in self.requests]


async def _make(rec: Recorder, **cfg) -> GeminiStreamTTS:
    tts = GeminiStreamTTS(api_key="test-key", transport=rec.transport)
    await tts.connect(AudioOutputConfig(sample_rate=RATE, **cfg))
    return tts


async def _collect(stream) -> bytes:
    async with asyncio.timeout(5):
        return b"".join([chunk async for chunk in stream.audio()])


def test_foreign_model_and_voice_fall_back():
    cfg = AudioOutputConfig(model="eleven_v4_turbo", voice="UOIqAnmS11Reiei1Ytkc")
    assert effective_tts_model(cfg) == DEFAULT_TTS_MODEL
    assert effective_voice(cfg) == DEFAULT_VOICE
    assert effective_tts_model(AudioOutputConfig(model="gemini-3.8-flash-lite-tts")) == "gemini-3.8-flash-lite-tts"
    assert effective_voice(AudioOutputConfig(voice="puck")) == "Puck"


def test_prompt_puts_the_brief_before_the_transcript():
    prompt = build_prompt("Hola, buenas.")
    assert prompt.endswith("#### TRANSCRIPCIÓN\nHola, buenas.")
    assert DEFAULT_ACCENT in prompt
    assert "'eh...'" in prompt  # written hesitations are spoken as written
    assert "lee en voz alta" not in prompt.lower()  # that phrasing makes Gemini read the notes aloud


def test_accent_and_director_notes_override():
    assert "andaluz" in build_prompt("x", accent="andaluz de Sevilla")
    custom = build_prompt("Hola.", director_notes="Habla muy bajito.")
    assert custom == "Habla muy bajito.\n#### TRANSCRIPCIÓN\nHola."


def test_split_clauses_cuts_at_sentences_and_commas_but_not_tiny_ones():
    done, rest = split_clauses("Vale, sin problema. Tómate todo tu tiempo, que yo te espero. Cuando")
    assert done == ["Vale, sin problema.", "Tómate todo tu tiempo,"]
    assert rest == "que yo te espero. Cuando"
    # Short pieces ("Sí,") are glued to the next one instead of becoming their own request.
    assert split_clauses("Sí, claro, perfecto. Ahora") == (["Sí, claro, perfecto."], "Ahora")
    assert split_clauses("Una sola frase sin cortes") == ([], "Una sola frase sin cortes")


def test_resolve_tts_knows_gemini():
    assert isinstance(resolve_tts("gemini"), GeminiStreamTTS)
    assert resolve_tts("Gemini-TTS").provider_id == "gemini"


async def test_synthesize_sends_key_in_header_and_voice_in_body():
    rec = Recorder()
    tts = await _make(rec, voice="Fenrir", model="gemini-3.8-flash-tts")
    audio = b"".join([c async for c in tts.synthesize("Hola, ¿qué tal?")])
    assert len(audio) == len(_pcm(200))
    req = rec.requests[0]
    assert req.headers["x-goog-api-key"] == "test-key"
    assert "test-key" not in str(req.url)
    assert req.url.path.endswith("gemini-3.8-flash-tts:streamGenerateContent")
    assert req.url.params["alt"] == "sse"
    body = json.loads(req.content)
    assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Fenrir"
    assert rec.texts()[0].endswith("Hola, ¿qué tal?")
    await tts.close()


async def test_audio_leaves_in_100ms_frames():
    rec = Recorder(_sse(_pcm(250)))
    tts = await _make(rec)
    frames = [c async for c in tts.synthesize("Hola, buenas tardes a todos.")]
    assert [len(f) for f in frames] == [4800, 4800, 2400]
    await tts.close()


async def test_resamples_to_the_session_rate():
    pytest.importorskip("av")
    rec = Recorder(_sse(_pcm(1000)))
    tts = GeminiStreamTTS(api_key="k", transport=rec.transport)
    await tts.connect(AudioOutputConfig(sample_rate=16_000))
    audio = b"".join([c async for c in tts.synthesize("Hola, buenas tardes a todos.")])
    assert abs(len(audio) - 16_000 * 2) < 16_000 * 2 * 0.05  # one second at 16 kHz, within filter delay
    await tts.close()


async def test_stream_plays_clauses_in_order():
    first, second = _pcm(100, 1000), _pcm(100, 2000)
    rec = Recorder(_sse(first), _sse(second))
    tts = await _make(rec, extra={"prefetch": 1})
    stream = tts.open_stream()
    await stream.feed("Perfecto, Mónica Álvarez. Te lo repito para confirmar el código.")
    await stream.end()
    audio = await _collect(stream)
    assert audio == first + second
    assert [t.rsplit("\n", 1)[-1] for t in rec.texts()] == [
        "Perfecto, Mónica Álvarez.",
        "Te lo repito para confirmar el código.",
    ]
    await tts.close()


async def test_drain_waits_for_what_was_requested():
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec)
    stream = tts.open_stream()
    await stream.feed("Voy a mirarlo ahora mismo.")
    consumer = asyncio.create_task(_collect(stream))
    assert await asyncio.wait_for(stream.drain(), 5) is True
    await stream.feed("Ya lo tengo aquí delante.")
    await stream.end()
    assert len(await consumer) == 2 * len(_pcm(100))
    await tts.close()


async def test_retries_once_when_gemini_answers_without_audio():
    text_only = b"data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "no"}]}}]}).encode() + b"\n\n"
    rec = Recorder(text_only, _sse(_pcm(100)))
    tts = await _make(rec)
    audio = b"".join([c async for c in tts.synthesize("Hola, buenas tardes.")])
    assert len(audio) == len(_pcm(100))
    assert len(rec.requests) == 2
    await tts.close()


async def test_gives_up_after_one_retry():
    rec = Recorder(httpx.Response(503, content=b"overloaded"))
    tts = await _make(rec)
    with pytest.raises(RuntimeError, match="503"):
        _ = [c async for c in tts.synthesize("Hola, buenas tardes.")]
    assert len(rec.requests) == 2
    await tts.close()


async def test_stream_error_surfaces_from_audio():
    rec = Recorder(httpx.Response(400, content=b"bad voice"))
    tts = await _make(rec)
    stream = tts.open_stream()
    await stream.feed("Hola, buenas tardes a todos.")
    await stream.end()
    with pytest.raises(RuntimeError, match="Gemini TTS failed"):
        await _collect(stream)
    await tts.close()


async def test_abort_unblocks_audio_and_stops_requests():
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec)
    stream = tts.open_stream()
    await stream.feed("Esto es una frase larga que se corta enseguida.")
    await stream.abort()
    assert await _collect(stream) == b""
    await stream.feed("Ya no debería pedirse nada más.")
    assert len(rec.requests) <= 1
    await tts.close()


async def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        await GeminiStreamTTS().connect(AudioOutputConfig(sample_rate=RATE))


async def test_non_pcm_output_is_rejected():
    with pytest.raises(ValueError, match="pcm_s16le"):
        await GeminiStreamTTS(api_key="k").connect(AudioOutputConfig(sample_rate=RATE, encoding="mulaw"))


@pytest.mark.parametrize(("provider", "kept"), [("gemini", True), ("elevenlabs", False), ("deepgram", False)])
def test_client_accent_reaches_only_gemini(provider, kept):
    """The playground sets the accent line; ElevenLabs and Deepgram would forward the key into their query string."""
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice import VoiceConfig

    session, _ = build_voice_session(
        Agent(name="voice_test", model=TestModel(responses=["Hola"]), tools=[]),
        VoiceConfig(turn_detector="heuristic"),
        {"tts_provider": provider, "tts_extra": {"accent": "andaluz marcado", "speed": 1.1}},
    )
    assert ("accent" in session.audio_output.extra) is kept
    assert session.audio_output.extra["speed"] == 1.1
