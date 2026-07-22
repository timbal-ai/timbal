"""End-to-end WebSocket tests for ``/voice/ws``.

Mocks ElevenLabs STT/TTS at the module boundary so the full ``voice_ws``
handler, JSON serialization, audio base64 encoding, session_transcript, and
event ordering are exercised through a real Starlette TestClient WebSocket.
"""

# ruff: noqa: ARG002
from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from timbal.server.http import create_app
from timbal.voice.session import (
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
)

from .voice_env import VOICE_ENV_KEYS

# ---------------------------------------------------------------------------
# Mock factories — produce classes whose zero-arg constructor matches
# ElevenLabsRealtimeSTT() / ElevenLabsStreamTTS() so voice_ws can
# instantiate them without changes.
# ---------------------------------------------------------------------------


def _make_stt_class(script: list[TranscriptEvent] | None = None):
    """Return an STT class that replays *script* on connect()."""
    _script = list(script or [])

    class _STT(SpeechToText):
        def __init__(self, api_key=None):
            self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

        async def connect(self, config: AudioInputConfig) -> None:
            for ev in _script:
                await self._queue.put(ev)
            await self._queue.put(None)

        async def push_audio(self, chunk: bytes) -> None:
            pass

        async def commit(self) -> None:
            pass

        async def events(self) -> AsyncIterator[TranscriptEvent]:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                if item.type == "error":
                    raise RuntimeError(item.text)
                if item.text:
                    yield item

        async def close(self) -> None:
            pass

    return _STT


def _make_tts_class(chunk: bytes = b"\x00\x01" * 16, num_chunks: int = 2):
    """Return a TTS class that yields fixed PCM chunks per synthesize call."""
    _chunk, _n = chunk, num_chunks

    class _TTS(TextToSpeech):
        def __init__(self, api_key=None):
            pass

        async def connect(self, config: AudioOutputConfig) -> None:
            pass

        async def synthesize(self, text: str) -> AsyncIterator[bytes]:
            for _ in range(_n):
                yield _chunk

        async def close(self) -> None:
            pass

    return _TTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_agent_module(tmp_path: Path, *, responses: list[str] | None = None) -> str:
    """Write a temp module with a TestModel Agent and return its import spec."""
    resp_repr = repr(responses or ["Hello from agent!"])
    mod = tmp_path / "voice_agent.py"
    mod.write_text(
        "from timbal import Agent\n"
        "from timbal.core.test_model import TestModel\n"
        f"agent = Agent(name='voice_test', model=TestModel(responses={resp_repr}), tools=[])\n"
    )
    return f"{mod.resolve()}::agent"


def _collect_ws_messages(ws, *, until: str = "session_ended") -> list[dict]:
    """Read JSON messages from the WS until we see *until* or disconnect."""
    messages: list[dict] = []
    while True:
        try:
            msg = ws.receive_json()
            messages.append(msg)
            if msg.get("type") == until:
                break
        except Exception:
            break
    return messages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoiceWsRoundTrip:
    """Full round-trip: config frame → session → JSON events over WS."""

    def test_single_turn_produces_correct_event_sequence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path, responses=["Hi there!"])
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        stt_cls = _make_stt_class([TranscriptEvent(type="committed", text="Hello")])
        tts_cls = _make_tts_class(chunk=b"\xAB" * 32, num_chunks=2)
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", stt_cls)
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", tts_cls)

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"language": "en"})
                messages = _collect_ws_messages(ws)

        types = [m["type"] for m in messages]

        assert types[0] == "session_started"
        assert "transcript_committed" in types
        assert "agent_text_done" in types
        assert "audio" in types
        assert types[-2] == "session_transcript"
        assert types[-1] == "session_ended"

        committed = next(m for m in messages if m["type"] == "transcript_committed")
        assert committed["text"] == "Hello"

        done = next(m for m in messages if m["type"] == "agent_text_done")
        assert "Hi there" in done["text"]

        audio_msgs = [m for m in messages if m["type"] == "audio"]
        assert len(audio_msgs) == 2
        decoded = base64.b64decode(audio_msgs[0]["data"])
        assert decoded == b"\xAB" * 32

    def test_empty_session_still_sends_transcript_and_ended(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """No STT events → session should still cleanly send transcript + ended."""
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        types = [m["type"] for m in messages]
        assert types[0] == "session_started"
        assert types[-2] == "session_transcript"
        assert types[-1] == "session_ended"

        transcript_msg = next(m for m in messages if m["type"] == "session_transcript")
        assert transcript_msg["entries"] == []


class TestVoiceWsMetrics:
    """Per-turn metrics should arrive as a ``metrics`` JSON message."""

    def test_metrics_message_forwarded_after_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path, responses=["Hi there!"])
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="Hello")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        types = [m["type"] for m in messages]
        assert "metrics" in types
        assert types.index("metrics") > types.index("agent_text_done")

        metrics_msg = next(m for m in messages if m["type"] == "metrics")
        m = metrics_msg["metrics"]
        assert m["turn_index"] == 1
        assert m["user_text_chars"] == len("Hello")
        assert m["interrupted"] is False
        assert m["eou_to_first_audio_ms"] is not None and m["eou_to_first_audio_ms"] >= 0
        assert m["turn_total_ms"] >= 0
        assert m["tts_segments"] >= 1
        assert m["audio_bytes"] > 0
        # No acks were sent and the turn was not interrupted.
        assert m["playback_acks_received"] is False
        assert m["heard_bytes"] is None


class TestVoiceWsPlaybackAck:
    """The ``playback`` uplink message must feed the session's playback tracker."""

    def test_playback_ack_accepted_and_session_completes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path, responses=["Hi there!"])
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="Hello")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                ws.send_json({"type": "playback", "played_ms": 125.0})
                ws.send_json({"type": "playback"})  # malformed — must be ignored
                messages = _collect_ws_messages(ws)

        types = [m["type"] for m in messages]
        assert "error" not in types
        assert types[-1] == "session_ended"
        assert "agent_text_done" in types

    def test_session_started_advertises_playback_acks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        started = next(m for m in messages if m["type"] == "session_started")
        assert started["playback_acks"] == "recommended"

    def test_interrupted_message_carries_heard_text_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Barge-in mid-turn → ``interrupted`` message includes ``heard_text``."""
        spec = _write_agent_module(
            tmp_path,
            responses=["First reply that is reasonably long for playback", "Second reply"],
        )
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class(
                [
                    TranscriptEvent(type="committed", text="Hello there my friend"),
                    TranscriptEvent(type="committed", text="Actually let me ask about something else entirely"),
                ]
            ),
        )
        # Enough audio that playback is still in flight when the barge-in lands.
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsStreamTTS",
            _make_tts_class(chunk=b"\x00\x01" * 8000, num_chunks=4),
        )

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        interrupted = [m for m in messages if m["type"] == "interrupted"]
        assert interrupted, f"no interrupted message in {[m['type'] for m in messages]}"
        assert "heard_text" in interrupted[0]


class TestVoiceWsSessionTranscript:
    """Verify session_transcript payload structure and ordering."""

    def test_transcript_contains_user_and_assistant_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path, responses=["Sure thing!"])
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="What time is it?")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        transcript_msg = next(m for m in messages if m["type"] == "session_transcript")
        entries = transcript_msg["entries"]

        assert len(entries) == 2
        assert entries[0]["role"] == "user"
        assert entries[0]["text"] == "What time is it?"
        assert "timestamp" in entries[0]
        assert entries[1]["role"] == "assistant"
        assert "Sure thing" in entries[1]["text"]
        assert "timestamp" in entries[1]

    def test_session_transcript_arrives_before_session_ended(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="Hi")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        types = [m["type"] for m in messages]
        idx_transcript = types.index("session_transcript")
        idx_ended = types.index("session_ended")
        assert idx_transcript == idx_ended - 1


class TestVoiceWsErrorPropagation:
    """STT errors should arrive as ``error`` JSON messages on the WS."""

    def test_stt_error_forwarded_as_error_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="error", text="STT auth failed")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        types = [m["type"] for m in messages]
        assert "error" in types

        error_msg = next(m for m in messages if m["type"] == "error")
        assert "STT" in error_msg["message"]

        assert types[-1] == "session_ended"


class TestVoiceWsAgentValidation:
    """Non-Agent runnables should be rejected at the WS level."""

    def test_ws_rejects_non_agent_runnable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        mod = tmp_path / "plain.py"
        mod.write_text("class NotAgent: pass\nrunnable = NotAgent()\n")
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::runnable")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                try:
                    ws.receive_json()
                    pytest.fail("Expected WebSocket to be closed by server")
                except Exception:
                    pass


class TestVoiceWsTurnDetectorIsolation:
    """A TurnDetector instance in voice_config must be cloned per session."""

    def test_shared_instance_is_cloned_per_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from timbal.voice.turn_detection import HeuristicTurnDetector

        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        started: list = []

        class _TrackingDetector(HeuristicTurnDetector):
            async def start(self, config) -> None:
                started.append(self)

        shared = _TrackingDetector()
        app = create_app()
        with TestClient(app) as client:
            app.state.voice_config = {**(app.state.voice_config or {}), "turn_detector": shared}
            for _ in range(2):
                with client.websocket_connect("/voice/ws") as ws:
                    ws.send_json({})
                    _collect_ws_messages(ws)

        assert len(started) == 2
        assert started[0] is not shared
        assert started[1] is not shared
        assert started[0] is not started[1]


class TestVoiceWsClientTurnDetector:
    """The client hello may pick a turn-detector *mode name* per session."""

    def _run_session(self, monkeypatch, tmp_path, hello: dict, server_td=None) -> dict:
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            if server_td is not None:
                app.state.voice_config = {**(app.state.voice_config or {}), "turn_detector": server_td}
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json(hello)
                messages = _collect_ws_messages(ws)
        return next(m for m in messages if m["type"] == "session_started")

    def test_client_mode_name_selects_detector(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(monkeypatch, tmp_path, {"turn_detector": "provider"})
        assert started["turn_detector"] == "ProviderTurnDetector"

    def test_client_mode_overrides_server_default(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(
            monkeypatch, tmp_path, {"turn_detector": "lexical"}, server_td="provider"
        )
        assert started["turn_detector"] == "LexicalTurnDetector"

    def test_default_is_heuristic_and_advertised(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(monkeypatch, tmp_path, {})
        assert started["turn_detector"] == "HeuristicTurnDetector"

    def test_non_string_client_value_is_ignored(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(monkeypatch, tmp_path, {"turn_detector": {"evil": True}})
        assert started["turn_detector"] == "HeuristicTurnDetector"

    def test_unknown_mode_name_falls_back_to_default(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(monkeypatch, tmp_path, {"turn_detector": "quantum"})
        assert started["turn_detector"] == "HeuristicTurnDetector"

    def test_racing_playback_ack_does_not_eat_config(self, monkeypatch, tmp_path: Path) -> None:
        """A playback ack sent before the hello must not be mistaken for config."""
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"type": "playback", "played_ms": 0})
                ws.send_json({"turn_detector": "provider"})
                messages = _collect_ws_messages(ws)

        started = next(m for m in messages if m["type"] == "session_started")
        assert started["turn_detector"] == "ProviderTurnDetector"


class TestVoiceWsAudioTransport:
    """Verify audio bytes survive the base64 round-trip over WS."""

    def test_audio_chunks_are_valid_base64_pcm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pcm_chunk = bytes(range(256)) * 4
        spec = _write_agent_module(tmp_path, responses=["ok"])
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="test")]),
        )
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsStreamTTS",
            _make_tts_class(chunk=pcm_chunk, num_chunks=1),
        )

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)

        audio_msgs = [m for m in messages if m["type"] == "audio"]
        assert len(audio_msgs) == 1
        decoded = base64.b64decode(audio_msgs[0]["data"])
        assert decoded == pcm_chunk
