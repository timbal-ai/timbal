"""End-to-end WebSocket tests for ``/voice/ws``.

Mocks ElevenLabs STT/TTS at the module boundary so the full ``voice_ws``
handler, JSON serialization, audio base64 encoding, session_transcript, and
event ordering are exercised through a real Starlette TestClient WebSocket.
"""

# ruff: noqa: ARG002
from __future__ import annotations

import asyncio
import base64
import json
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


def _write_agent_module(tmp_path: Path, *, responses: list[str] | None = None, extra: str = "") -> str:
    """Write a temp module with a TestModel Agent and return its import spec."""
    resp_repr = repr(responses or ["Hello from agent!"])
    mod = tmp_path / "voice_agent.py"
    mod.write_text(
        "from timbal import Agent\n"
        "from timbal.core.test_model import TestModel\n"
        f"agent = Agent(name='voice_test', model=TestModel(responses={resp_repr}), tools=[])\n"
        + extra
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
                # Pinned, not defaulted: this asserts what a detector *without* an
                # audio EOU advertises, so it must not follow the server default.
                ws.send_json({"turn_detector": "heuristic"})
                messages = _collect_ws_messages(ws)

        started = next(m for m in messages if m["type"] == "session_started")
        assert started["playback_acks"] == "recommended"
        # The heuristic detector has no audio EOU model → the local VAD
        # endpointing fast path never arms, and session_started must say so.
        assert started["vad_endpointing"] is False

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

    # A holding detector: which one depends on whether timbal[voice] is installed
    # (see test_voice_detector_choice.py, which pins that branch directly). The
    # contract asserted here is that an unconfigured session gets a detector that
    # *can* hold — the holdless heuristic splits paused utterances into several
    # turns on any STT that endpoints on silence.
    _HOLDS = ("LocalAudioTurnDetector", "LexicalTurnDetector")

    def test_default_holds_and_is_advertised(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(monkeypatch, tmp_path, {})
        assert started["turn_detector"] in self._HOLDS

    def test_non_string_client_value_is_ignored(self, monkeypatch, tmp_path: Path) -> None:
        started = self._run_session(monkeypatch, tmp_path, {"turn_detector": {"evil": True}})
        assert started["turn_detector"] in self._HOLDS

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

    def test_many_racing_protocol_frames_do_not_exhaust_handshake(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The handshake must skip typed frames until the hello, not give up
        after a fixed count — a burst of acks/mic_change silently dropped the
        client's sample_rate / turn_detector overrides."""
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                for i in range(8):
                    ws.send_json({"type": "playback", "played_ms": float(i)})
                # A typed non-audio/playback frame must be skipped too, not
                # mistaken for the config hello (the hello has no "type").
                ws.send_json({"type": "mic_change"})
                ws.send_json({"turn_detector": "provider"})
                messages = _collect_ws_messages(ws)

        started = next(m for m in messages if m["type"] == "session_started")
        assert started["turn_detector"] == "ProviderTurnDetector"

    def test_malformed_early_frames_do_not_end_handshake(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Invalid JSON or a broken audio payload before the hello must be
        skipped, not abort the scan — the hello behind them still applies."""
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_text("{not valid json")
                ws.send_json({"type": "audio"})  # missing "data"
                ws.send_json({"type": "audio", "data": "!!!not-base64!!!"})
                ws.send_json({"turn_detector": "provider"})
                messages = _collect_ws_messages(ws)

        started = next(m for m in messages if m["type"] == "session_started")
        assert started["turn_detector"] == "ProviderTurnDetector"


class TestVoiceWsTurnTimeoutConfig:
    """``turn_timeout_secs`` / ``turn_timeout_fallback`` must reach the session."""

    def _capture_session_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        import timbal.voice as voice_pkg

        real = voice_pkg.VoiceSession
        captured: dict = {}

        class _CapturingSession(real):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(voice_pkg, "VoiceSession", _CapturingSession, raising=False)
        return captured

    def test_turn_timeout_keys_are_plumbed_into_the_session(
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
        captured = self._capture_session_kwargs(monkeypatch)

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"turn_timeout_secs": 12, "turn_timeout_fallback": "hold on"})
                _collect_ws_messages(ws)

        assert captured["turn_timeout_secs"] == 12.0
        assert captured["turn_timeout_fallback"] == "hold on"

    def test_bad_turn_timeout_value_keeps_the_session_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A non-numeric value must be dropped, not zero the watchdog out."""
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _make_stt_class([]))
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())
        captured = self._capture_session_kwargs(monkeypatch)

        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"turn_timeout_secs": "soon"})
                messages = _collect_ws_messages(ws)

        assert "turn_timeout_secs" not in captured
        assert any(m["type"] == "session_started" for m in messages)


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
                # This is a transport test: it needs the injected commit to start a
                # turn immediately. A holding detector (the server default) would
                # correctly park unpunctuated "test" for seconds and produce no
                # audio at all.
                ws.send_json({"turn_detector": "heuristic"})
                messages = _collect_ws_messages(ws)

        audio_msgs = [m for m in messages if m["type"] == "audio"]
        assert len(audio_msgs) == 1
        decoded = base64.b64decode(audio_msgs[0]["data"])
        assert decoded == pcm_chunk


class TestVoiceWsRecording:
    """Call recording: server-defaults-only config → MP3 + manifest per session."""

    def _run_session(self, monkeypatch, spec: str, hello: dict, extra_env: dict | None = None) -> list[dict]:
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in (extra_env or {}).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="Hello")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())
        # Holding detectors (server default) park unpunctuated "Hello" and
        # can end the session with zero TTS — empty MP3 close then flakes on
        # Windows. Pin heuristic like the audio transport tests.
        hello = {"turn_detector": "heuristic", **hello}
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json(hello)
                return _collect_ws_messages(ws)

    def test_server_configured_recording_writes_audio_and_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(
            tmp_path,
            responses=["Hi there!"],
            extra=f"agent.voice_config = {{'recording': {{'dir': {str(rec_dir)!r}}}}}\n",
        )
        messages = self._run_session(monkeypatch, spec, {"language": "en"})

        started = next(m for m in messages if m["type"] == "session_started")
        session_id = started["session_id"]
        assert session_id

        # Files are finalized before session_ended reaches the client.
        assert (rec_dir / f"{session_id}.mp3").exists()
        manifest = json.loads((rec_dir / f"{session_id}.json").read_text())
        assert manifest["session_id"] == session_id
        assert manifest["meta"]["transport"] == "websocket"
        assert [e["role"] for e in manifest["transcript"]] == ["user", "assistant"]
        assert all(e["offset_ms"] >= 0 for e in manifest["transcript"])

        # The wire transcript carries the same timing info.
        transcript = next(m for m in messages if m["type"] == "session_transcript")
        assert transcript["started_at"] == pytest.approx(manifest["started_at"])
        assert all("offset_ms" in e for e in transcript["entries"])

    def test_client_hello_cannot_switch_recording_on(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(tmp_path)  # no recording in server defaults
        messages = self._run_session(
            monkeypatch, spec, {"recording": {"dir": str(rec_dir)}}
        )
        assert any(m["type"] == "session_ended" for m in messages)
        assert not rec_dir.exists()

    def test_env_var_enables_recording(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(tmp_path)
        monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TIMBAL_VOICE_RECORDING_DIR", str(rec_dir))
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_stt_class([TranscriptEvent(type="committed", text="Hello")]),
        )
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"language": "en", "turn_detector": "heuristic"})
                messages = _collect_ws_messages(ws)

        started = next(m for m in messages if m["type"] == "session_started")
        assert (rec_dir / f"{started['session_id']}.mp3").exists()

    def test_env_knobs_set_layout_and_bitrate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(tmp_path)
        messages = self._run_session(
            monkeypatch, spec, {"language": "en"},
            extra_env={
                "TIMBAL_VOICE_RECORDING_DIR": str(rec_dir),
                "TIMBAL_VOICE_RECORDING_LAYOUT": "split",
                "TIMBAL_VOICE_RECORDING_BITRATE_KBPS": "64",
            },
        )
        started = next(m for m in messages if m["type"] == "session_started")
        manifest = json.loads((rec_dir / f"{started['session_id']}.json").read_text())
        assert manifest["audio"]["layout"] == "split"
        assert manifest["audio"]["bitrate_kbps"] == 64

    def test_user_voice_config_wins_over_env_knobs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(
            tmp_path,
            extra=f"agent.voice_config = {{'recording': {{'dir': {str(rec_dir)!r}, 'layout': 'mixed'}}}}\n",
        )
        messages = self._run_session(
            monkeypatch, spec, {"language": "en"},
            extra_env={"TIMBAL_VOICE_RECORDING_LAYOUT": "split"},
        )
        started = next(m for m in messages if m["type"] == "session_started")
        manifest = json.loads((rec_dir / f"{started['session_id']}.json").read_text())
        assert manifest["audio"]["layout"] == "mixed"

    def test_platform_identity_env_is_stamped_into_manifest_meta(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(tmp_path)
        messages = self._run_session(
            monkeypatch, spec, {"language": "en"},
            extra_env={
                "TIMBAL_VOICE_RECORDING_DIR": str(rec_dir),
                "TIMBAL_ORG_ID": "org-9",
                "TIMBAL_PROJECT_ID": "proj-7",
                "TIMBAL_PROJECT_ENV_ID": "env-3",
                "TIMBAL_APP_ID": "app-1",
                "TIMBAL_PROJECT_REV": "rev-42",
            },
        )
        started = next(m for m in messages if m["type"] == "session_started")
        # Identity lands in the manifest (self-describing files for sweeper
        # ingest), not in the wire payload.
        assert "org_id" not in started
        manifest = json.loads((rec_dir / f"{started['session_id']}.json").read_text())
        assert manifest["meta"]["org_id"] == "org-9"
        assert manifest["meta"]["project_id"] == "proj-7"
        assert manifest["meta"]["project_env_id"] == "env-3"
        assert manifest["meta"]["app_id"] == "app-1"
        assert manifest["meta"]["project_rev"] == "rev-42"
        assert manifest["meta"]["transport"] == "websocket"  # session meta still wins/merges

    def test_no_tmp_manifest_left_behind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The manifest write is atomic (tmp + rename) — sweepers key on json presence."""
        pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
        rec_dir = tmp_path / "recordings"
        spec = _write_agent_module(tmp_path)
        self._run_session(
            monkeypatch, spec, {"language": "en"},
            extra_env={"TIMBAL_VOICE_RECORDING_DIR": str(rec_dir)},
        )
        assert not list(rec_dir.glob("*.tmp"))
        assert len(list(rec_dir.glob("*.json"))) == 1
