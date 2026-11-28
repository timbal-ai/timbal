"""Gemini TTS adapter: what a request sends, voice/model guards, sentence streaming, retry, abort."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from timbal.voice.gemini_tts import (
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    GeminiStreamTTS,
    build_prompt,
    effective_tts_model,
    effective_voice,
    split_sentences,
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

    def speech_configs(self) -> list[dict]:
        return [json.loads(r.content)["generationConfig"]["speechConfig"] for r in self.requests]


async def _make(rec: Recorder, *, transport: httpx.AsyncBaseTransport | None = None, **cfg) -> GeminiStreamTTS:
    """A connected adapter. Sentences are requested as they come (``hold_ms`` 0) unless a test sets the hold."""
    tts = GeminiStreamTTS(api_key="test-key", transport=transport or rec.transport)
    cfg["extra"] = {"hold_ms": 0, **cfg.get("extra", {})}
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


def test_the_prompt_is_the_text_and_nothing_else():
    # Anything around the text is performed: with a brief Gemini added words of its own or read the notes aloud.
    assert build_prompt("Hola, buenas.") == "Hola, buenas."
    assert build_prompt("¿Qué tiempo hace hoy?", accent="  ", director_notes="") == "¿Qué tiempo hace hoy?"


def test_a_brief_is_sent_only_when_the_agent_asks_for_one():
    with_accent = build_prompt("Hola.", accent="andaluz de Sevilla")
    assert with_accent.endswith("Acento: andaluz de Sevilla\n#### TRANSCRIPCIÓN\nHola.")
    assert "lee en voz alta" not in with_accent.lower()  # that phrasing makes Gemini read the notes aloud
    custom = build_prompt("Hola.", director_notes="Habla muy bajito.")
    assert custom == "Habla muy bajito.\n#### TRANSCRIPCIÓN\nHola."


@pytest.mark.parametrize(
    ("text", "ready", "rest"),
    [
        # The sentences that are complete come back together, for one request. Never a cut at a comma.
        (
            "Vale, sin problema. Tómate todo tu tiempo, que yo te espero. Cuando",
            "Vale, sin problema. Tómate todo tu tiempo, que yo te espero.",
            "Cuando",
        ),
        ("¿Me dices la matrícula del coche? Así lo busco", "¿Me dices la matrícula del coche?", "Así lo busco"),
        ("Te paso con el Sr. García ahora mismo. ", "Te paso con el Sr. García ahora mismo.", ""),
        ("Eh... pues déjame mirarlo un momento.", "Eh... pues déjame mirarlo un momento.", ""),
        # A colon does not end the take: what it announces is said with it.
        ("Sí, aquí lo tengo: el envío salió ayer.", "Sí, aquí lo tengo: el envío salió ayer.", ""),
        ("Sí, aquí lo tengo:", "", "Sí, aquí lo tengo:"),
        # Not complete yet: no sentence end, a comma, an abbreviation's period.
        ("Una sola frase sin cortes", "", "Una sola frase sin cortes"),
        ("Vale, para poder mirar tu factura,", "", "Vale, para poder mirar tu factura,"),
        ("Te paso con el Sr. García ahora mismo", "", "Te paso con el Sr. García ahora mismo"),
        # A short opener waits for the sentence after it.
        ("Perfecto, Marta.", "", "Perfecto, Marta."),
        ("Perfecto, Marta. Tu factura es de ochenta euros.", "Perfecto, Marta. Tu factura es de ochenta euros.", ""),
        ("", "", ""),
    ],
)
def test_split_sentences_cuts_after_a_sentence_and_nowhere_else(text, ready, rest):
    assert split_sentences(text) == (ready, rest)


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
    assert rec.texts() == ["Hola, ¿qué tal?"]
    assert "languageCode" not in rec.speech_configs()[0]
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


async def test_stream_plays_its_requests_in_order():
    first, second = _pcm(100, 1000), _pcm(100, 2000)
    rec = Recorder(_sse(first), _sse(second))
    tts = await _make(rec, extra={"prefetch": 1})
    stream = tts.open_stream()
    await stream.feed("Perfecto, Mónica Álvarez, ya lo tengo.")
    await stream.feed(" Te lo repito para confirmar el código.")
    await stream.end()
    audio = await _collect(stream)
    assert audio == first + second
    assert rec.texts() == ["Perfecto, Mónica Álvarez, ya lo tengo.", "Te lo repito para confirmar el código."]
    await tts.close()


async def test_what_the_session_feeds_together_is_one_request():
    """Each request is a take of its own. A segment cut again at its commas and sentences changed voice inside
    one reply."""
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec)
    stream = tts.open_stream()
    await stream.feed("Vale, para poder mirar tu factura, necesito tu DNI, por favor. ¿Me lo dices despacio?")
    await stream.end()
    await _collect(stream)
    assert rec.texts() == ["Vale, para poder mirar tu factura, necesito tu DNI, por favor. ¿Me lo dices despacio?"]
    await tts.close()


async def test_sentences_right_behind_the_first_go_in_its_request():
    """A model writes a short reply's sentences milliseconds apart. Asked for one by one they are as many takes,
    each with its own pitch; held for a moment they are one."""
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec, extra={"hold_ms": 60_000})
    stream = tts.open_stream()
    await stream.feed("Ya está hecho. ")
    await stream.feed("Te acabo de enviar la confirmación por mensaje. ")
    for _ in range(20):
        await asyncio.sleep(0)
    assert rec.requests == []
    await stream.feed("Que tengas muy buen día.")
    await stream.end()  # the reply's end is not held: nothing more is coming
    assert len(await _collect(stream)) == len(_pcm(100))
    assert rec.texts() == ["Ya está hecho. Te acabo de enviar la confirmación por mensaje. Que tengas muy buen día."]
    await tts.close()


async def test_a_sentence_nothing_follows_is_requested_when_the_hold_is_over():
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec, extra={"hold_ms": 20})
    stream = tts.open_stream()
    await stream.feed("Ya lo tengo delante, un momento. ")
    await asyncio.sleep(0)
    assert rec.requests == []
    async with asyncio.timeout(5):
        while not rec.requests:
            await asyncio.sleep(0.005)
    assert rec.texts() == ["Ya lo tengo delante, un momento."]
    await stream.abort()
    await tts.close()


async def test_an_interrupted_reply_requests_nothing_after_its_hold():
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec, extra={"hold_ms": 20})
    stream = tts.open_stream()
    await stream.feed("Ya lo tengo delante, un momento. ")
    await stream.abort()
    await asyncio.sleep(0.06)
    assert rec.requests == []
    await tts.close()


async def test_sentences_that_arrive_while_a_request_generates_go_together():
    """Every request is a take of its own, so a reply is its first sentence and then the rest, not a request per
    sentence. Audio is generated faster than it plays: waiting for the request in flight delays nothing."""
    release = asyncio.Event()
    texts: list[str] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["contents"][0]["parts"][0]["text"])
        if len(texts) == 1:
            await release.wait()
        return httpx.Response(200, content=_sse(_pcm(100)))

    async def requests(n: int) -> None:
        async with asyncio.timeout(5):
            while len(texts) < n:
                await asyncio.sleep(0)

    tts = await _make(Recorder(), transport=httpx.MockTransport(answer))
    stream = tts.open_stream()
    await stream.feed("Ya lo tengo delante. ")
    await requests(1)
    await stream.feed("El técnico pasó el martes. ")
    await stream.feed("La pieza llega el jueves. ")
    for _ in range(20):
        await asyncio.sleep(0)
    assert texts == ["Ya lo tengo delante."]
    release.set()
    await requests(2)
    await stream.end()
    assert len(await _collect(stream)) == 2 * len(_pcm(100))
    assert texts == ["Ya lo tengo delante.", "El técnico pasó el martes. La pieza llega el jueves."]
    await tts.close()


async def test_the_end_of_the_reply_does_not_wait_for_the_request_in_flight():
    release = asyncio.Event()
    texts: list[str] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["contents"][0]["parts"][0]["text"])
        if len(texts) == 1:
            await release.wait()
        return httpx.Response(200, content=_sse(_pcm(100)))

    tts = await _make(Recorder(), transport=httpx.MockTransport(answer))
    stream = tts.open_stream()
    await stream.feed("Ya lo tengo delante. ")
    await stream.feed("El técnico pasó el martes. ")
    await stream.end()
    async with asyncio.timeout(5):
        while len(texts) < 2:
            await asyncio.sleep(0)
    assert texts == ["Ya lo tengo delante.", "El técnico pasó el martes."]
    release.set()
    assert len(await _collect(stream)) == 2 * len(_pcm(100))
    await tts.close()


async def test_an_unfinished_sentence_waits_for_its_end():
    rec = Recorder(_sse(_pcm(100)))
    tts = await _make(rec)
    stream = tts.open_stream()
    await stream.feed("Perfecto, Marta.")  # a short opener
    await stream.feed(" Sí, aquí lo tengo:")  # a segment the session closed at a colon
    await asyncio.sleep(0)
    assert rec.requests == []
    await stream.feed(" el envío salió ayer por la tarde.")
    await stream.feed(" Y llega")
    await stream.end()
    await _collect(stream)
    assert rec.texts() == ["Perfecto, Marta. Sí, aquí lo tengo: el envío salió ayer por la tarde.", "Y llega"]
    await tts.close()


async def test_language_code_is_a_request_field_not_prompt_text():
    rec = Recorder()
    tts = await _make(rec, extra={"language_code": "es_es"})
    _ = [c async for c in tts.synthesize("Hola, buenas tardes.")]
    assert rec.speech_configs()[0]["languageCode"] == "es-es"
    assert rec.texts() == ["Hola, buenas tardes."]
    await tts.close()


async def test_a_value_that_is_not_a_language_tag_is_not_sent():
    rec = Recorder()
    tts = await _make(rec, extra={"language_code": "castellano de España"})
    _ = [c async for c in tts.synthesize("Hola, buenas tardes.")]
    assert "languageCode" not in rec.speech_configs()[0]
    await tts.close()


async def test_a_language_gemini_refuses_is_dropped_for_the_rest_of_the_call():
    refused = httpx.Response(
        400, json={"error": {"code": 400, "message": "Invalid language_code 'xx-YY' in speech_config."}}
    )
    rec = Recorder(refused, _sse(_pcm(100)))
    tts = await _make(rec, extra={"language_code": "xx-YY"})
    assert len(b"".join([c async for c in tts.synthesize("Hola, buenas tardes.")])) == len(_pcm(100))
    _ = [c async for c in tts.synthesize("¿Sigues ahí?")]
    assert ["languageCode" in cfg for cfg in rec.speech_configs()] == [True, False, False]
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
def test_client_accent_and_language_code_reach_only_gemini(provider, kept):
    """A playground sets them; ElevenLabs and Deepgram would forward the keys into their query string."""
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice import VoiceConfig

    session, _ = build_voice_session(
        Agent(name="voice_test", model=TestModel(responses=["Hola"]), tools=[]),
        VoiceConfig(turn_detector="heuristic"),
        {"tts_provider": provider, "tts_extra": {"accent": "andaluz marcado", "language_code": "es-ES", "speed": 1.1}},
    )
    assert ("accent" in session.audio_output.extra) is kept
    assert ("language_code" in session.audio_output.extra) is kept
    assert session.audio_output.extra["speed"] == 1.1


@pytest.mark.parametrize(("provider", "kept"), [("gemini", True), ("elevenlabs", False), ("deepgram", False)])
def test_server_gemini_extras_do_not_follow_a_call_to_another_provider(provider, kept):
    """A Gemini agent's ``director_notes`` / ``prefetch`` / ``hold_ms`` are server config: a client that picks ElevenLabs or
    Deepgram must not carry them into that provider's query string."""
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice import VoiceConfig

    session, _ = build_voice_session(
        Agent(name="voice_test", model=TestModel(responses=["Hola"]), tools=[]),
        VoiceConfig(
            turn_detector="heuristic",
            tts_provider="gemini",
            tts_extra={"accent": "rioplatense", "director_notes": "Habla despacio.", "prefetch": 3, "hold_ms": 0},
        ),
        {"tts_provider": provider},
    )
    for key in ("accent", "director_notes", "prefetch", "hold_ms"):
        assert (key in session.audio_output.extra) is kept, key
