"""Aura's wire protocol and lifecycle, without a network connection."""

import asyncio
import contextlib
import json
import os
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr
from timbal.voice import DeepgramStreamTTS
from timbal.voice.config import DEFAULT_VOICE_ID
from timbal.voice.deepgram_tts import DEFAULT_TTS_MODEL, build_tts_uri, effective_tts_model
from timbal.voice.providers import AudioOutputConfig, resolve_tts


class FakeWS:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def receive(self, *messages):
        for message in messages:
            self.incoming.put_nowait(json.dumps(message) if isinstance(message, dict) else message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True


@pytest.fixture
def wire(monkeypatch):
    socket = FakeWS()
    connect = AsyncMock(return_value=socket)
    monkeypatch.setattr("timbal.voice.deepgram_tts.ws_connect", connect)
    return socket, connect


async def collect(stream):
    return [chunk async for chunk in stream.audio()]


async def provider():
    tts = DeepgramStreamTTS(api_key=SecretStr("test-key"))
    await tts.connect(AudioOutputConfig())
    return tts


def test_resolution_and_foreign_defaults(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_TTS_MODEL", raising=False)
    assert isinstance(resolve_tts(" DEEPGRAM "), DeepgramStreamTTS)
    assert isinstance(resolve_tts("deepgram-aura"), DeepgramStreamTTS)
    config = AudioOutputConfig(model="eleven_flash_v2_5", voice=DEFAULT_VOICE_ID, extra={"auto_mode": True})
    assert effective_tts_model(config) == DEFAULT_TTS_MODEL
    assert parse_qs(urlparse(build_tts_uri(config)).query) == {
        "model": [DEFAULT_TTS_MODEL],
        "encoding": ["linear16"],
        "sample_rate": ["16000"],
    }
    monkeypatch.setenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")
    assert effective_tts_model(config) == "aura-asteria-en"
    config.voice = "aura-2-apollo-en"
    assert effective_tts_model(config) == "aura-2-apollo-en"
    config.model = "aura-2-thalia-en"
    assert effective_tts_model(config) == "aura-2-thalia-en"


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 32000, 48000])
def test_output_format_and_options(rate):
    config = AudioOutputConfig(
        sample_rate=rate,
        extra={
            "tts_host": "api.eu.deepgram.com",
            "speed": 1.1,
            "mip_opt_out": True,
            "encoding": "mp3",
            "sample_rate": 44100,
            "output_format": "mp3_44100_128",
        },
    )
    uri = urlparse(build_tts_uri(config))
    assert uri.hostname == "api.eu.deepgram.com"
    params = parse_qs(uri.query)
    assert params["sample_rate"] == [str(rate)]
    assert params["encoding"] == ["linear16"]
    assert params["mip_opt_out"] == ["true"]
    assert params["speed"] == ["1.1"]
    assert "output_format" not in params


@pytest.mark.parametrize("config", [AudioOutputConfig(encoding="mp3"), AudioOutputConfig(sample_rate=44100)])
async def test_reject_invalid_format(config):
    with pytest.raises(ValueError):
        await DeepgramStreamTTS(api_key="test").connect(config)


async def test_incremental_audio_before_end_and_flush(wire):
    socket, connect = wire
    tts = await provider()
    stream = tts.open_stream()
    audio = stream.audio()
    pending = asyncio.create_task(anext(audio))
    await stream.feed("Hello, ")
    socket.receive({"type": "Metadata"}, b"\x01\x02")
    assert await asyncio.wait_for(pending, 1) == b"\x01\x02"
    await stream.feed("world.")
    await stream.end()
    await stream.end()
    socket.receive(b"\x03\x04", {"type": "Flushed", "sequence_id": 0})
    assert [chunk async for chunk in audio] == [b"\x03\x04"]
    assert socket.sent == [{"type": "Speak", "text": "Hello, "}, {"type": "Speak", "text": "world."}, {"type": "Flush"}]
    assert connect.call_args.kwargs["additional_headers"] == {"Authorization": "Token test-key"}
    assert socket.closed and not tts._streams


async def test_empty_stream_and_idempotence(wire):
    _, connect = wire
    tts = await provider()
    stream = tts.open_stream()
    pending = asyncio.create_task(collect(stream))
    await stream.feed("  ")
    await stream.end()
    await stream.feed("ignored")
    assert await asyncio.wait_for(pending, 1) == []
    await stream.abort()
    await stream.abort()
    connect.assert_not_called()


async def test_abort_discards_buffered_audio_and_unblocks_consumer(wire):
    socket, _ = wire
    tts = await provider()
    stream = tts.open_stream()
    await stream.feed("This will be interrupted.")
    socket.receive(b"old audio")
    await asyncio.sleep(0)
    await stream.abort()
    assert await asyncio.wait_for(collect(stream), 1) == []
    assert socket.closed and not tts._streams
    empty = tts.open_stream()
    pending = asyncio.create_task(collect(empty))
    await tts.close()
    assert await asyncio.wait_for(pending, 1) == []
    with pytest.raises(RuntimeError, match="connect"):
        tts.open_stream()


@pytest.mark.parametrize("response", [None, RuntimeError("wire lost"), {"type": "Error", "description": "bad request"}])
async def test_errors_after_partial_audio_are_not_silent_success(wire, response):
    socket, _ = wire
    tts = await provider()
    stream = tts.open_stream()
    await stream.feed("Hello.")
    await stream.end()
    socket.receive(b"partial", response)
    audio = stream.audio()
    assert await anext(audio) == b"partial"
    with pytest.raises(RuntimeError, match="Deepgram TTS"):
        await anext(audio)
    assert socket.closed and not tts._streams


async def test_handshake_failure_unblocks_audio(wire):
    _, connect = wire
    connect.side_effect = RuntimeError("unauthorized")
    tts = await provider()
    stream = tts.open_stream()
    pending = asyncio.create_task(collect(stream))
    with pytest.raises(RuntimeError, match="unauthorized"):
        await stream.feed("Hello")
    with pytest.raises(RuntimeError, match="unauthorized"):
        await asyncio.wait_for(pending, 1)
    assert not tts._streams


async def test_abort_cancels_inflight_handshake(wire):
    socket, connect = wire
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handshake(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    connect.side_effect = handshake
    tts = await provider()
    stream = tts.open_stream()
    pending = asyncio.create_task(collect(stream))
    feed = asyncio.create_task(stream.feed("Hello"))
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(tts.close(), 1)
    await asyncio.wait_for(feed, 1)
    assert await asyncio.wait_for(pending, 1) == []
    assert cancelled.is_set() and not socket.sent and not tts._streams


@pytest.mark.parametrize("during_flush", [False, True])
async def test_send_failure_unblocks_audio_and_cleans_up(wire, during_flush):
    socket, _ = wire
    tts = await provider()
    stream = tts.open_stream()
    pending = asyncio.create_task(collect(stream))
    if during_flush:
        await stream.feed("Hello")
    socket.send = AsyncMock(side_effect=RuntimeError("send failed"))
    with pytest.raises(RuntimeError, match="send failed"):
        if during_flush:
            await stream.end()
        else:
            await stream.feed("Hello")
    with pytest.raises(RuntimeError, match="send failed"):
        await asyncio.wait_for(pending, 1)
    assert socket.closed and not tts._streams


async def test_concurrent_replies_are_isolated(wire):
    first, connect = wire
    second = FakeWS()
    connect.side_effect = [first, second]
    tts = await provider()
    old, new = tts.open_stream(), tts.open_stream()
    await old.feed("Old reply")
    await new.feed("New reply")
    await old.abort()
    first.receive(b"stale")
    await new.end()
    second.receive(b"new", {"type": "Flushed"})
    assert await collect(old) == []
    assert await collect(new) == [b"new"]
    assert first.closed and second.closed


async def test_synthesize_and_shutdown_close_live_sockets(wire):
    socket, _ = wire
    tts = await provider()
    task = asyncio.create_task(_synthesize(tts))
    await asyncio.sleep(0)
    socket.receive(b"pcm", {"type": "Flushed"})
    assert await asyncio.wait_for(task, 1) == [b"pcm"]
    stream = tts.open_stream()
    await stream.feed("unfinished")
    await tts.close()
    assert socket.closed and not tts._streams


async def _synthesize(tts):
    return [chunk async for chunk in tts.synthesize("Hello")]


def test_server_build_selects_deepgram_and_filters_client_extras():
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.server.voice import build_voice_session
    from timbal.voice.config import VoiceConfig

    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]), tools=[])
    session, meta = build_voice_session(
        agent,
        VoiceConfig(turn_detector="heuristic"),
        {
            "tts_provider": "deepgram",
            "tts_model": "aura-2-thalia-en",
            "tts_extra": {"tts_host": "attacker.example", "callback": "https://attacker.example", "speed": 1.1},
        },
    )
    assert isinstance(session.tts, DeepgramStreamTTS)
    assert meta["tts_provider"] == "deepgram"
    assert effective_tts_model(session.audio_output) == "aura-2-thalia-en"
    assert "tts_host" not in session.audio_output.extra
    assert "callback" not in session.audio_output.extra
    assert session.audio_output.extra["speed"] == 1.1


@pytest.mark.integration
async def test_live_deepgram_streaming():
    if not os.getenv("DEEPGRAM_API_KEY"):
        pytest.skip("Set DEEPGRAM_API_KEY to run live Aura tests")
    tts = DeepgramStreamTTS()
    await tts.connect(AudioOutputConfig(model="aura-2-thalia-en"))
    audio = None
    try:
        async with asyncio.timeout(30):
            stream = tts.open_stream()
            audio = asyncio.create_task(collect(stream))
            await stream.feed("Hello from Timbal. ")
            await stream.feed("This is a voice integration test.")
            await stream.end()
            chunks = await audio
            assert chunks and sum(map(len, chunks)) > 1000
            assert sum(map(len, chunks)) % 2 == 0
            assert not chunks[0].startswith(b"RIFF")
            interrupted = tts.open_stream()
            await interrupted.feed("This reply will be interrupted before it finishes.")
            await interrupted.end()
            interrupted_audio = interrupted.audio()
            assert await anext(interrupted_audio)
            await interrupted.abort()
            assert [chunk async for chunk in interrupted_audio] == []
            # A subsequent reply must also produce audio after the first drains.
            assert await _synthesize(tts)
    finally:
        await tts.close()
        if audio is not None:
            audio.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await audio
