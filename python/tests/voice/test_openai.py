import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr
from timbal.voice import OpenAIRealtimeSTT, OpenAIStreamTTS
from timbal.voice.config import DEFAULT_VOICE_ID
from timbal.voice.deepgram import effective_stt_model, resolve_stt, stt_provider_id
from timbal.voice.openai import build_transcription_session, effective_tts_model, effective_voice
from timbal.voice.providers import AudioInputConfig, AudioOutputConfig, resolve_tts


class FakeWS:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def receive(self, *events):
        for event in events:
            self.incoming.put_nowait(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.incoming.get()
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    async def close(self):
        self.closed = True


@pytest.fixture
def ws(monkeypatch):
    socket = FakeWS()
    socket.receive({"type": "session.created"}, {"type": "session.updated"})
    connect = AsyncMock(return_value=socket)
    monkeypatch.setattr("timbal.voice.openai.ws_connect", connect)
    return socket, connect


def test_provider_selection_and_cross_provider_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_TTS_VOICE", raising=False)
    stt = resolve_stt(" OPENAI ")
    assert isinstance(stt, OpenAIRealtimeSTT)
    assert stt_provider_id(stt) == "openai"
    assert isinstance(resolve_stt(model="gpt-4o-mini-transcribe"), OpenAIRealtimeSTT)
    assert effective_stt_model(stt, "scribe_v2_realtime") == "gpt-transcribe"
    assert effective_stt_model(stt, "whisper-1") == "whisper-1"
    assert effective_stt_model(resolve_stt("elevenlabs"), "gpt-4o-transcribe") is None
    assert effective_stt_model(resolve_stt("deepgram-nova"), "gpt-4o-transcribe") == "nova-3"
    assert isinstance(resolve_tts("openai"), OpenAIStreamTTS)
    out = AudioOutputConfig(model="eleven_flash_v2_5", voice=DEFAULT_VOICE_ID)
    assert effective_tts_model(out) == "gpt-4o-mini-tts"
    assert effective_voice(out) == "coral"
    monkeypatch.setenv("OPENAI_TTS_VOICE", "alloy")
    assert effective_voice(out) == "alloy"
    assert effective_voice(AudioOutputConfig(voice="marin")) == "marin"


def test_ga_session_schema_ignores_foreign_and_reserved_options():
    request = build_transcription_session(
        AudioInputConfig(
            language="es",
            extra={
                "commit_strategy": "vad",
                "audio_format": "mp3",
                "model": "bad",
                "prompt": "Timbal",
                "noise_reduction": "near_field",
                "threshold": 0.6,
                "silence_duration_ms": 700,
            },
        )
    )
    assert request["type"] == "session.update"
    assert request["session"]["type"] == "transcription"
    cfg = request["session"]["audio"]["input"]
    assert cfg["format"] == {"type": "audio/pcm", "rate": 24000}
    assert cfg["transcription"] == {"model": "gpt-transcribe", "languages": ["es"], "prompt": "Timbal"}
    assert cfg["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.6,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 700,
    }
    assert cfg["noise_reduction"] == {"type": "near_field"}


async def test_stt_wire_audio_alignment_and_commit(ws):
    socket, connect = ws
    stt = OpenAIRealtimeSTT(api_key=SecretStr("key"))
    await stt.connect(AudioInputConfig(sample_rate=24000))
    try:
        assert connect.call_args.kwargs["additional_headers"] == {"Authorization": "Bearer key"}
        await stt.commit()  # no audio: never send an empty commit
        await stt.push_audio(b"\x01")
        await stt.push_audio(b"\x02" + b"\x00\x01" * 2400)
        await stt.commit()
        await stt.commit()  # repeat commit is harmless
        frames = socket.sent[1:]
        assert len(frames) == 2
        assert frames[0]["type"] == "input_audio_buffer.append"
        assert base64.b64decode(frames[0]["audio"]) == b"\x01\x02" + b"\x00\x01" * 2400
        assert frames[1]["type"] == "input_audio_buffer.commit"
        # Only a race caused by this adapter's own commit is suppressed.
        stt._handle_message(
            {"type": "error", "error": {"code": "input_audio_buffer_commit_empty", "event_id": frames[1]["event_id"]}}
        )
        with pytest.raises(RuntimeError):
            stt._handle_message(
                {"type": "error", "error": {"code": "input_audio_buffer_commit_empty", "event_id": "unknown"}}
            )
    finally:
        await stt.close()
    assert socket.closed


def test_transcript_partials_and_out_of_order_finals():
    stt = OpenAIRealtimeSTT(api_key="key")
    for item in ("a", "b"):
        stt._handle_message({"type": "input_audio_buffer.committed", "item_id": item})
    for text in ("Hello", " world"):
        stt._handle_message(
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": "a", "delta": text}
        )
    assert stt._queue.get_nowait().text == "Hello"
    assert stt._queue.get_nowait().text == "Hello world"
    stt._handle_message(
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "b", "transcript": "Second"}
    )
    assert stt._queue.empty()
    stt._handle_message(
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "First"}
    )
    assert [stt._queue.get_nowait().text for _ in range(2)] == ["First", "Second"]
    assert not stt._partials and not stt._completed and not stt._order


@pytest.mark.parametrize(
    "event",
    [
        None,
        RuntimeError("disconnected"),
        {"type": "error", "error": {"message": "rate limited"}},
        {"type": "conversation.item.input_audio_transcription.failed", "error": {"message": "failed"}},
    ],
)
async def test_stt_provider_failure_terminates_events(ws, event):
    socket, _ = ws
    stt = OpenAIRealtimeSTT(api_key="key")
    await stt.connect(AudioInputConfig(sample_rate=24000))
    socket.receive(event)
    with pytest.raises(RuntimeError, match="OpenAI STT"):
        await asyncio.wait_for(anext(stt.events()), 1)
    await stt.close()


async def test_stt_bad_config_closes_socket(ws):
    socket, _ = ws
    socket.incoming = asyncio.Queue()
    socket.receive({"type": "error", "error": {"message": "invalid model"}})
    stt = OpenAIRealtimeSTT(api_key="key")
    with pytest.raises(RuntimeError, match="invalid model"):
        await stt.connect(AudioInputConfig(sample_rate=24000))
    assert socket.closed and stt._receiver is None


async def test_stt_shutdown_unblocks_events_and_can_reconnect(ws):
    socket, _ = ws
    stt = OpenAIRealtimeSTT(api_key="key")
    await stt.connect(AudioInputConfig(sample_rate=24000))
    pending = asyncio.create_task(anext(stt.events(), None))
    await asyncio.sleep(0)
    await stt.close()
    assert await asyncio.wait_for(pending, 1) is None
    socket.receive({"type": "session.updated"})
    await stt.connect(AudioInputConfig(sample_rate=24000))
    await stt.close()


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
async def test_stt_resampling_preserves_duration(ws, rate):
    pytest.importorskip("av")
    socket, _ = ws
    stt = OpenAIRealtimeSTT(api_key="key")
    await stt.connect(AudioInputConfig(sample_rate=rate))
    try:
        await stt.push_audio(b"\x00\x01" * rate)
        await stt.commit()
        audio = b"".join(base64.b64decode(e["audio"]) for e in socket.sent if e["type"] == "input_audio_buffer.append")
        assert abs(len(audio) - 48000) <= 2
    finally:
        await stt.close()


class RawAudioStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self):
        self.closed = True


def sse(event):
    return ("data: " + json.dumps(event) + "\r\n\r\n").encode()


class AudioStream(RawAudioStream):
    """PCM chunks carried in the mini-TTS SSE protocol."""

    async def __aiter__(self):
        async for chunk in super().__aiter__():
            yield sse({"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode()})
        yield sse(
            {
                "type": "speech.audio.done",
                "usage": {
                    "input_tokens": 14,
                    "output_tokens": 101,
                    "total_tokens": 115,
                },
            }
        )


@pytest.fixture
def http(monkeypatch):
    requests, responses = [], []

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    client = httpx.AsyncClient
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr("timbal.voice.openai.httpx.AsyncClient", lambda **kwargs: client(transport=transport, **kwargs))
    return requests, responses


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
async def test_tts_streaming_pcm_resampling_and_wire_format(http, rate):
    if rate != 24000:
        pytest.importorskip("av")
    requests, responses = http
    audio = AudioStream([b"\x00", b"\x01" + b"\x00\x01" * 23999])
    responses.append(httpx.Response(200, stream=audio))
    tts = OpenAIStreamTTS(api_key="key")
    await tts.connect(
        AudioOutputConfig(
            sample_rate=rate,
            extra={"instructions": "Speak calmly", "speed": 1.1, "auto_mode": True, "response_format": "mp3"},
        )
    )
    try:
        pcm = b"".join([chunk async for chunk in tts.synthesize("Hello")])
        assert abs(len(pcm) - rate * 2) <= 2
        payload = json.loads(requests[0].content)
        assert payload == {
            "input": "Hello",
            "model": "gpt-4o-mini-tts",
            "voice": "coral",
            "response_format": "pcm",
            "stream_format": "sse",
            "instructions": "Speak calmly",
            "speed": 1.1,
        }
        assert requests[0].url.path == "/v1/audio/speech"
        assert requests[0].headers["Authorization"] == "Bearer key"
        assert tts.open_stream() is None
        assert audio.closed
    finally:
        await tts.close()


async def test_tts_generator_close_and_provider_close_release_responses(http):
    _, responses = http
    first, second = AudioStream([b"\x00\x01"] * 10), AudioStream([b"\x02\x03"] * 10)
    responses.extend([httpx.Response(200, stream=first), httpx.Response(200, stream=second)])
    tts = OpenAIStreamTTS(api_key="key")
    await tts.connect(AudioOutputConfig(sample_rate=24000))
    old, new = tts.synthesize("First"), tts.synthesize("Second")
    assert await anext(old) == b"\x00\x01"
    assert await anext(new) == b"\x02\x03"
    await old.aclose()
    assert first.closed and not second.closed
    await tts.close()
    assert second.closed
    assert await anext(new, None) is None
    with pytest.raises(RuntimeError, match="connect"):
        await anext(tts.synthesize("closed"))


@pytest.mark.parametrize("response", [httpx.Response(401), httpx.Response(200, stream=AudioStream([b"\x01"]))])
async def test_tts_errors_are_not_silently_successful(http, response):
    _, responses = http
    responses.append(response)
    tts = OpenAIStreamTTS(api_key="key")
    await tts.connect(AudioOutputConfig(sample_rate=24000))
    try:
        with pytest.raises((httpx.HTTPStatusError, RuntimeError)):
            async for _ in tts.synthesize("Hello"):
                pass
        assert response.is_closed
    finally:
        await tts.close()


def test_server_selects_providers_and_rejects_client_host_overrides():
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice import VoiceConfig

    agent = Agent(name="voice_test", model=TestModel(responses=["Hello"]), tools=[])
    session, meta = build_voice_session(
        agent,
        VoiceConfig(turn_detector="heuristic"),
        {
            "stt_provider": "openai",
            "tts_provider": "openai",
            "stt_extra": {"stt_host": "evil.test", "threshold": 0.6},
            "tts_extra": {"tts_host": "evil.test", "instructions": "Speak softly"},
        },
    )
    assert isinstance(session.stt, OpenAIRealtimeSTT)
    assert isinstance(session.tts, OpenAIStreamTTS)
    assert meta["stt_provider"] == meta["tts_provider"] == "openai"
    assert meta["stt_model"] == "gpt-transcribe"
    assert "stt_host" not in session.audio_input.extra
    assert "tts_host" not in session.audio_output.extra
    assert session.audio_input.extra["threshold"] == 0.6
    assert session.audio_output.extra["instructions"] == "Speak softly"


@pytest.mark.integration
async def test_live_openai_speech_roundtrip():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set OPENAI_API_KEY for live speech tests")
    pytest.importorskip("av")
    tts, stt = OpenAIStreamTTS(), OpenAIRealtimeSTT()
    try:
        await stt.connect(AudioInputConfig(language="en"))
        await tts.connect(AudioOutputConfig())
        async with asyncio.timeout(45):
            pcm = b"".join([chunk async for chunk in tts.synthesize("The blue bicycle is outside.")])
            assert len(pcm) > 1000
            # Stream in 100 ms microphone-sized chunks, then force a commit.
            for offset in range(0, len(pcm), 3200):
                await stt.push_audio(pcm[offset : offset + 3200])
            await stt.commit()
            async for event in stt.events():
                if event.type == "committed":
                    assert "bicycle" in event.text.lower()
                    break
            # The default flow must also finish turns without local/manual
            # commit (heuristic/provider turn detectors have no local VAD).
            await stt.push_audio(b"\x00\x00" * 8000)
            for offset in range(0, len(pcm), 3200):
                await stt.push_audio(pcm[offset : offset + 3200])
            await stt.push_audio(b"\x00\x00" * 16000)
            async for event in stt.events():
                if event.type == "committed":
                    assert "bicycle" in event.text.lower()
                    break
    finally:
        await stt.close()
        await tts.close()


def test_existing_providers_do_not_import_openai_adapter():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from timbal.voice.deepgram import resolve_stt, effective_stt_model, stt_provider_id
from timbal.voice.providers import resolve_tts
for provider in ('elevenlabs', 'deepgram-flux', 'deepgram-nova', 'munsit'):
    stt = resolve_stt(provider)
    effective_stt_model(stt, None)
    stt_provider_id(stt)
for provider in ('elevenlabs', 'deepgram', 'munsit', 'fishaudio'):
    resolve_tts(provider)
assert 'timbal.voice.openai' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "model", ["gpt-live-transcribe", "gpt-realtime-whisper", "gpt-4o-transcribe-diarize", "gpt-4o-mini-transcribe-typo"]
)
def test_unsupported_stt_models_fail_explicitly(model):
    stt = resolve_stt(model=model)
    assert isinstance(stt, OpenAIRealtimeSTT)
    with pytest.raises(ValueError, match="Unsupported OpenAI"):
        effective_stt_model(stt, model)


@pytest.mark.parametrize(
    "config",
    [
        AudioOutputConfig(model="tts-1", voice="marin"),
        AudioOutputConfig(model="tts-1-hd", voice="verse"),
        AudioOutputConfig(model="tts-1", extra={"instructions": "Whisper"}),
        AudioOutputConfig(model="gpt-4o-mini-tts-typo"),
    ],
)
async def test_invalid_tts_model_options_fail_at_connect(http, config):
    requests, _ = http
    with pytest.raises(ValueError):
        await OpenAIStreamTTS(api_key="key").connect(config)
    assert not requests


def test_later_turn_partials_cannot_overwrite_earlier_turn():
    stt = OpenAIRealtimeSTT(api_key="key")
    for item in ("a", "b"):
        stt._handle_message({"type": "input_audio_buffer.committed", "item_id": item})
    stt._handle_message(
        {"type": "conversation.item.input_audio_transcription.delta", "item_id": "b", "delta": "Second"}
    )
    assert stt._queue.empty()
    stt._handle_message(
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "First"}
    )
    assert stt._queue.get_nowait().text == "First"
    assert stt._queue.get_nowait().text == "Second"


async def test_delayed_commit_ack_does_not_discard_next_turn_audio(ws):
    socket, _ = ws
    stt = OpenAIRealtimeSTT(api_key="key")
    await stt.connect(AudioInputConfig(sample_rate=24000))
    try:
        await stt.push_audio(b"\0\0" * 4800)
        await stt.commit()
        await stt.push_audio(b"\0\0" * 4800)
        stt._handle_message({"type": "input_audio_buffer.committed", "item_id": "first"})
        await stt.commit()
        assert sum(e["type"] == "input_audio_buffer.commit" for e in socket.sent) == 2
    finally:
        await stt.close()


async def test_cancelling_stt_session_setup_closes_socket(ws):
    socket, _ = ws
    socket.incoming = asyncio.Queue()
    stt = OpenAIRealtimeSTT(api_key="key")
    task = asyncio.create_task(stt.connect(AudioInputConfig(sample_rate=24000)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.closed
    assert stt._ws is None and stt._receiver is None


async def test_cancelling_tts_read_closes_response_and_allows_next_turn(http):
    _, responses = http
    waiting = asyncio.Event()

    class BlockedAudio(AudioStream):
        async def __aiter__(self):
            waiting.set()
            await asyncio.Event().wait()
            yield b""

    stream = BlockedAudio([])
    responses.extend([httpx.Response(200, stream=stream), httpx.Response(200, stream=AudioStream([b"\0\0"]))])
    tts = OpenAIStreamTTS(api_key="key")
    await tts.connect(AudioOutputConfig(sample_rate=24000))
    try:
        pending = asyncio.create_task(anext(tts.synthesize("First")))
        await asyncio.wait_for(waiting.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert stream.closed and not tts._responses
        assert b"".join([c async for c in tts.synthesize("Second")]) == b"\0\0"
    finally:
        await tts.close()


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
async def test_exactly_100ms_audio_can_commit(ws, rate):
    pytest.importorskip("av")
    socket, _ = ws
    stt = OpenAIRealtimeSTT(api_key="key")
    await stt.connect(AudioInputConfig(sample_rate=rate))
    try:
        await stt.push_audio(b"\0\0" * (rate // 10))
        await stt.commit()
        assert socket.sent[-1]["type"] == "input_audio_buffer.commit"
    finally:
        await stt.close()


@pytest.mark.parametrize("provider", ["elevenlabs", "deepgram-nova"])
def test_switching_providers_drops_openai_only_extras(provider):
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice import VoiceConfig

    session, _ = build_voice_session(
        Agent(name="voice_test", model=TestModel(responses=["Hello"]), tools=[]),
        VoiceConfig(turn_detector="heuristic"),
        {
            "stt_provider": provider,
            "tts_provider": "elevenlabs",
            "stt_extra": {"threshold": 0.6, "noise_reduction": "far_field"},
            "tts_extra": {"instructions": "Whisper", "speed": 1.1},
        },
    )
    assert "threshold" not in session.audio_input.extra
    assert "noise_reduction" not in session.audio_input.extra
    assert "instructions" not in session.audio_output.extra
    assert session.audio_output.extra["speed"] == 1.1


def test_legacy_transcription_uses_singular_language():
    request = build_transcription_session(AudioInputConfig(model="whisper-1", language="es"))
    assert request["session"]["audio"]["input"]["transcription"] == {"model": "whisper-1", "language": "es"}


def test_existing_server_extra_passthrough_is_preserved():
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice import VoiceConfig

    session, _ = build_voice_session(
        Agent(name="voice_test", model=TestModel(responses=["Hello"]), tools=[]),
        VoiceConfig(turn_detector="heuristic", stt_extra={"prompt": "Server setting"}),
        {"stt_extra": {"prompt": "New client setting"}},
    )
    assert session.audio_input.extra["prompt"] == "Server setting"


@pytest.mark.parametrize(
    "model,usage",
    [
        (
            "gpt-4o-transcribe",
            {
                "type": "tokens",
                "input_tokens": 17,
                "output_tokens": 9,
                "total_tokens": 26,
                "input_token_details": {"text_tokens": 2, "audio_tokens": 15},
            },
        ),
        ("whisper-1", {"type": "duration", "seconds": 1.25}),
    ],
)
@pytest.mark.usefixtures("ws")
async def test_stt_reports_usage_independently_of_transcript_order(model, usage):
    stt = OpenAIRealtimeSTT(api_key="key")
    events = []
    stt.add_usage_listener(events.append)
    await stt.connect(AudioInputConfig(model=model, sample_rate=24000))
    for item in ("a", "b"):
        stt._handle_message({"type": "input_audio_buffer.committed", "item_id": item})
    completion = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "b",
        "content_index": 0,
        "transcript": "",
        "usage": usage,
    }
    stt._handle_message(completion)
    stt._handle_message(completion)
    assert stt._queue.empty()  # Blank transcript / blocked ordering must not hide spend.
    assert events[0].usage == usage and events[0].model == model
    assert events[0].status == "complete"
    assert events[0].usage_id == events[1].usage_id  # Durable consumers can deduplicate.
    await stt.close()
    assert events[-1].item_id == "a" and events[-1].status == "incomplete"
    assert events[-1].usage is None
    count = len(events)
    await stt.close()
    assert len(events) == count


async def test_stt_usage_scope_changes_on_reconnect(ws):
    socket, _ = ws
    stt = OpenAIRealtimeSTT(api_key="key")
    events = []
    stt.add_usage_listener(events.append)
    for _ in range(2):
        await stt.connect(AudioInputConfig(sample_rate=24000))
        stt._handle_message(
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "same", "transcript": ""}
        )
        await stt.close()
        socket.receive({"type": "session.updated"})
    assert events[0].usage_id != events[1].usage_id
    assert all(e.status == "incomplete" for e in events)


async def test_sse_usage_and_audio_survive_arbitrary_http_boundaries(http):
    _, responses = http
    usage = {"input_tokens": 14, "output_tokens": 101, "total_tokens": 115}
    wire = (
        b": keepalive\r\nevent: speech.audio.delta\r\n"
        + sse({"type": "speech.audio.delta", "audio": "AAE="})
        + b'data: {"type": "speech.audio.done",\r\ndata: "usage": '
        + json.dumps(usage).encode()
        + b"}\r\n\r\n"
    )
    responses.append(
        httpx.Response(
            200, headers={"x-request-id": "req_123"}, stream=RawAudioStream([wire[i : i + 1] for i in range(len(wire))])
        )
    )
    tts = OpenAIStreamTTS(api_key="key")
    events = []
    tts.add_usage_listener(events.append)
    await tts.connect(AudioOutputConfig(model="eleven_flash_v2_5", sample_rate=24000))
    try:
        assert b"".join([c async for c in tts.synthesize("Hello")]) == b"\x00\x01"
        assert len(events) == 1
        assert events[0].usage == usage and events[0].status == "complete"
        assert events[0].request_id == "req_123" and events[0].model == "gpt-4o-mini-tts"
    finally:
        await tts.close()
    assert len(events) == 1


@pytest.mark.parametrize(
    "ending",
    [
        b"",
        b"data: invalid json\n\n",
        sse({"type": "speech.audio.error", "error": {"message": "failed"}}),
        sse({"type": "speech.audio.delta", "audio": "not-base64!"}),
    ],
)
async def test_failed_sse_does_not_manufacture_zero_usage(http, ending):
    _, responses = http
    stream = RawAudioStream([sse({"type": "speech.audio.delta", "audio": "AAE="}), ending])
    responses.append(httpx.Response(200, stream=stream))
    tts = OpenAIStreamTTS(api_key="key")
    events = []
    tts.add_usage_listener(events.append)
    await tts.connect(AudioOutputConfig(sample_rate=24000))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            async for _ in tts.synthesize("Hello"):
                pass
        assert stream.closed
        assert len(events) == 1 and events[0].status == "incomplete" and events[0].usage is None
    finally:
        await tts.close()


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0}])
async def test_missing_or_invalid_terminal_usage_is_incomplete(http, usage):
    _, responses = http
    responses.append(httpx.Response(200, stream=RawAudioStream([sse({"type": "speech.audio.done", "usage": usage})])))
    tts = OpenAIStreamTTS(api_key="key")
    events = []
    tts.add_usage_listener(events.append)
    await tts.connect(AudioOutputConfig(sample_rate=24000))
    try:
        assert [c async for c in tts.synthesize("Hello")] == []
        assert len(events) == 1 and events[0].status == "incomplete"
    finally:
        await tts.close()


async def test_concurrent_synthesis_close_reports_each_request_once(http):
    _, responses = http
    responses.extend(
        httpx.Response(200, headers={"x-request-id": f"req_{i}"}, stream=AudioStream([b"\0\0"] * 2)) for i in range(2)
    )
    tts = OpenAIStreamTTS(api_key="key")
    events = []
    tts.add_usage_listener(events.append)
    await tts.connect(AudioOutputConfig(sample_rate=24000))
    first, second = tts.synthesize("First"), tts.synthesize("Second")
    await anext(first)
    await anext(second)
    await first.aclose()
    assert len(events) == 1 and events[0].request_id == "req_0"
    await tts.close()
    assert len(events) == 2 and events[1].request_id == "req_1"
    await second.aclose()
    assert len(events) == 2 and len({e.usage_id for e in events}) == 2
    assert all(e.status == "incomplete" for e in events)


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd"])
async def test_legacy_tts_keeps_raw_pcm_and_character_metering(http, model):
    requests, responses = http
    responses.append(httpx.Response(200, stream=RawAudioStream([b"\0", b"\1"])))
    tts = OpenAIStreamTTS(api_key="key")
    events = []
    tts.add_usage_listener(events.append)
    await tts.connect(AudioOutputConfig(model=model, voice="alloy", sample_rate=24000))
    try:
        assert b"".join([c async for c in tts.synthesize("Hello")]) == b"\0\1"
        assert "stream_format" not in json.loads(requests[0].content)
        assert events == []
    finally:
        await tts.close()
