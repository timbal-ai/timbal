"""ElevenLabs adapters: what actually goes on the wire."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from timbal.voice import elevenlabs as el
from timbal.voice.providers import AudioInputConfig, AudioOutputConfig


class _IdleWs:
    """A socket that never yields and accepts sends — enough to let ``connect`` return."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.closed = False

    def __aiter__(self) -> _IdleWs:
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


async def _connect_uri(monkeypatch: pytest.MonkeyPatch, config: AudioInputConfig) -> dict[str, list[str]]:
    seen: dict[str, str] = {}

    async def fake_connect(uri: str, **_kw: Any) -> _IdleWs:
        seen["uri"] = uri
        return _IdleWs()

    monkeypatch.setattr(el, "ws_connect", fake_connect)
    stt = el.ElevenLabsRealtimeSTT(api_key="test-key")
    await stt.connect(config)
    try:
        return parse_qs(urlsplit(seen["uri"]).query, keep_blank_values=True)
    finally:
        await stt.close()


class TestScribeRealtimeQuery:
    async def test_list_extras_become_repeated_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``keyterms`` is the one vocabulary-biasing knob Scribe has, and the
        API only reads it as ``keyterms=a&keyterms=b``. A stringified list
        (``keyterms=['a', 'b']``) silently biases towards one bogus term."""
        q = await _connect_uri(
            monkeypatch,
            AudioInputConfig(extra={"keyterms": ["Acme", "Timbal AI"], "secondary_languages": ["es", "fr"]}),
        )
        assert q["keyterms"] == ["Acme", "Timbal AI"]
        assert q["secondary_languages"] == ["es", "fr"]
        assert q["model_id"] == [el._DEFAULT_STT_MODEL]
        assert q["commit_strategy"] == ["vad"]

    async def test_scalars_and_booleans_are_wire_spelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = await _connect_uri(
            monkeypatch,
            AudioInputConfig(
                language="en",
                extra={"vad_threshold": 0.4, "include_timestamps": True, "no_verbatim": False, "_private": "x"},
            ),
        )
        assert q["language_code"] == ["en"]
        assert q["vad_threshold"] == ["0.4"]
        assert q["include_timestamps"] == ["true"]
        assert q["no_verbatim"] == ["false"]
        assert "_private" not in q
        assert "stt_host" not in q


class _TTSWs:
    """Queue server frames explicitly so audio ordering/errors are exercised."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.incoming: asyncio.Queue[dict | None] = asyncio.Queue()
        self.closed = False
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.allow_close.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self.incoming.get()
        if msg is None:
            raise StopAsyncIteration
        return json.dumps(msg)

    async def send(self, raw):
        msg = json.loads(raw)
        if msg.get("close_context"):
            self.close_started.set()
            await self.allow_close.wait()
        self.sent.append(msg)

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(None)


@pytest.fixture
async def tts_factory(monkeypatch):
    providers = []

    async def make(model="eleven_v4_turbo", **extra):
        ws = _TTSWs()
        seen = {}

        async def connect(uri, **kwargs):
            seen.update(uri=uri, **kwargs)
            return ws

        monkeypatch.setattr(el, "ws_connect", connect)
        p = el.ElevenLabsStreamTTS(api_key="test-key")
        providers.append(p)
        await p.connect(AudioOutputConfig(model=model, voice="test-voice", extra=extra))
        await p._preconnect_task
        return p, ws, seen

    yield make
    for p in providers:
        await p.close()


async def _collect(stream):
    async with asyncio.timeout(2):
        return b"".join([chunk async for chunk in stream.audio()])


@pytest.mark.parametrize("model", ["eleven_v4", "eleven_v4_turbo"])
async def test_dialogue_stream_registers_once_and_routes_audio(tts_factory, model):
    p, ws, seen = await tts_factory(model, auto_mode=True, inactivity_timeout=180)
    uri = urlsplit(seen["uri"])
    assert uri.path == "/v1/text-to-dialogue/multi-stream-input"
    assert parse_qs(uri.query) == {
        "model_id": [model],
        "output_format": ["pcm_16000"],
        "apply_text_normalization": ["on"],
    }
    assert seen["additional_headers"] == {"xi-api-key": "test-key"}
    stream = p.open_stream()
    await stream.feed("Hello ")
    await stream.feed("world.")
    await stream.end()
    assert ws.sent == [
        {"context_id": "ctx_1", "voices": ["test-voice"]},
        {"context_id": "ctx_1", "inputs": [{"text": "Hello ", "voice_id": "test-voice"}]},
        {"context_id": "ctx_1", "inputs": [{"text": "world.", "voice_id": "test-voice"}]},
        {"context_id": "ctx_1", "flush": True},
        {"context_id": "ctx_1", "close_context": True},
    ]
    ws.incoming.put_nowait({"context_id": "ctx_1", "audio": base64.b64encode(b"\x01\x02").decode()})
    ws.incoming.put_nowait({"context_id": "ctx_1", "is_final_audio_for_turn": True})
    ws.incoming.put_nowait({"context_id": "ctx_1", "audio": base64.b64encode(b"\x03\x04").decode()})
    ws.incoming.put_nowait({"context_id": "ctx_1", "is_final": True})
    assert await _collect(stream) == b"\x01\x02\x03\x04"
    assert not p._audio_queues


@pytest.mark.parametrize("model", ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2"])
async def test_legacy_tts_keeps_original_protocol(tts_factory, model):
    p, ws, seen = await tts_factory(model)
    uri = urlsplit(seen["uri"])
    assert uri.path == "/v1/text-to-speech/test-voice/multi-stream-input"
    assert parse_qs(uri.query)["auto_mode"] == ["true"]
    assert parse_qs(uri.query)["inactivity_timeout"] == ["180"]
    stream = p.open_stream()
    await stream.feed("Hello.")
    await stream.end()
    await p._send_keepalive()
    assert ws.sent == [
        {"context_id": "ctx_1", "text": "Hello."},
        {"context_id": "ctx_1", "text": " ", "flush": True},
        {"context_id": "ctx_1", "close_context": True},
        {"context_id": "_ka", "text": ""},
    ]
    ws.incoming.put_nowait({"contextId": "ctx_1", "audio": base64.b64encode(b"\x01\x02").decode()})
    ws.incoming.put_nowait({"contextId": "ctx_1", "isFinal": True})
    assert await _collect(stream) == b"\x01\x02"


async def test_dialogue_synthesize_and_concurrent_stream_do_not_mix_audio(tts_factory):
    p, ws, _ = await tts_factory()
    greeting = p.synthesize("Welcome.")
    first = asyncio.create_task(anext(greeting))
    await asyncio.wait_for(ws.close_started.wait(), timeout=2)
    stream = p.open_stream()
    await stream.feed("Answer.")
    await stream.end()
    for ctx, audio in [("ctx_2", b"\x03\x04"), ("ctx_1", b"\x01\x02")]:
        ws.incoming.put_nowait({"context_id": ctx, "audio": base64.b64encode(audio).decode()})
        ws.incoming.put_nowait({"context_id": ctx, "is_final": True})
    assert await asyncio.wait_for(first, timeout=2) == b"\x01\x02"
    assert await _collect(stream) == b"\x03\x04"
    assert [chunk async for chunk in greeting] == []
    assert ws.sent[1] == {
        "context_id": "ctx_1",
        "inputs": [{"text": "Welcome. ", "voice_id": "test-voice"}],
        "flush": True,
    }


async def test_abort_drops_queued_audio_and_does_not_close_context_twice(tts_factory):
    p, ws, _ = await tts_factory()
    stream = p.open_stream()
    await stream.feed("Interrupt me.")
    await stream.end()
    stream._queue.put_nowait({"audio": base64.b64encode(b"\x01\x02").decode()})
    await stream.abort()
    await stream.abort()
    await p._send_keepalive()
    assert await _collect(stream) == b""
    assert sum(bool(msg.get("close_context")) for msg in ws.sent) == 1
    assert not any(msg.get("keep_alive") for msg in ws.sent)


async def test_keepalive_only_targets_open_registered_contexts(tts_factory):
    p, ws, _ = await tts_factory()
    await p._send_keepalive()
    assert not ws.sent
    first, second = p.open_stream(), p.open_stream()
    await first.feed("One.")
    await second.feed("Two.")
    await first.abort()
    await p._send_keepalive()
    assert [msg for msg in ws.sent if msg.get("keep_alive")] == [{"context_id": "ctx_2", "keep_alive": True}]
    await second.abort()


async def test_keepalive_cannot_race_context_close(tts_factory):
    p, ws, _ = await tts_factory()
    stream = p.open_stream()
    await stream.feed("One.")
    ws.allow_close.clear()
    closing = asyncio.create_task(stream.end())
    await asyncio.wait_for(ws.close_started.wait(), timeout=2)
    keepalive = asyncio.create_task(p._send_keepalive())
    ws.allow_close.set()
    await asyncio.wait_for(asyncio.gather(closing, keepalive), timeout=2)
    assert ws.sent[-1] == {"context_id": "ctx_1", "close_context": True}
    assert not any(msg.get("keep_alive") for msg in ws.sent)


async def test_dialogue_keepalive_interval_is_below_server_timeout(tts_factory, monkeypatch):
    intervals = []

    async def keepalive(self, interval):
        intervals.append(interval)
        await self._stop.wait()

    monkeypatch.setattr(el.ElevenLabsStreamTTS, "_keepalive_loop", keepalive)
    await tts_factory(tts_keepalive_interval=55)
    await asyncio.sleep(0)
    assert intervals == [10.0]


@pytest.mark.parametrize("partial_audio", [False, True])
async def test_unscoped_server_error_fails_all_streams_even_after_audio(tts_factory, partial_audio):
    p, ws, _ = await tts_factory()
    streams = [p.open_stream(), p.open_stream()]
    for stream in streams:
        await stream.feed("Hello.")
    if partial_audio:
        ws.incoming.put_nowait({"context_id": "ctx_1", "audio": base64.b64encode(b"\x01\x02").decode()})
    ws.incoming.put_nowait({"error": "context_limit_exceeded"})
    for stream in streams:
        with pytest.raises(RuntimeError, match="context_limit_exceeded"):
            await _collect(stream)
