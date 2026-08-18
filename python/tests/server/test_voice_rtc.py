"""End-to-end tests for ``POST /voice/rtc`` — a real aiortc loopback in one process.

The test plays the browser: its own ``RTCPeerConnection``, a silent mic
track, and a client-created data channel, with SDP exchanged through the
actual FastAPI endpoint. Media and SCTP flow over localhost UDP between the
test's event loop and the TestClient portal loop. STT/TTS are mocked at the
module boundary exactly like ``test_voice_ws.py`` — everything from the
signaling down through ICE, DTLS, Opus, and the session is real.
"""

# ruff: noqa: ARG002
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aiortc", reason="timbal[voice] extra (aiortc) not installed")

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from timbal.server.http import create_app  # noqa: E402
from timbal.voice import (  # noqa: E402
    AudioInputConfig,
    SpeechToText,
    TranscriptEvent,
)

from .test_voice_ws import _make_tts_class, _write_agent_module  # noqa: E402
from .voice_env import VOICE_ENV_KEYS  # noqa: E402


def _make_delayed_stt_class(script: list[tuple[float, TranscriptEvent]], *, end_after: float = 1.0):
    """An STT class replaying ``(delay_secs, event)`` pairs, then ending.

    The WS mocks replay instantly on connect; over RTC the session must
    outlive the ICE/DTLS/SCTP handshake for anything to reach the client, so
    events are spaced on a real clock.
    """
    _script = list(script)

    class _STT(SpeechToText):
        def __init__(self, api_key: Any = None) -> None:
            self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
            self._feeder: asyncio.Task | None = None

        async def connect(self, config: AudioInputConfig) -> None:
            async def _feed() -> None:
                for delay, ev in _script:
                    await asyncio.sleep(delay)
                    await self._queue.put(ev)
                await asyncio.sleep(end_after)
                await self._queue.put(None)

            self._feeder = asyncio.create_task(_feed())

        async def push_audio(self, chunk: bytes) -> None:
            pass

        async def commit(self) -> None:
            pass

        async def events(self):
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                if item.text:
                    yield item

        async def close(self) -> None:
            if self._feeder is not None and not self._feeder.done():
                self._feeder.cancel()

    return _STT


async def _rtc_call(
    client: TestClient,
    *,
    config: dict | None = None,
    timeout: float = 20.0,
) -> tuple[list[dict], list[Any]]:
    """Run one full call as the browser would; returns (messages, downlink frames)."""
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    pc.addTrack(AudioStreamTrack())  # silent mic
    channel = pc.createDataChannel("events")

    messages: list[dict] = []
    frames: list[Any] = []
    ended = asyncio.Event()

    @channel.on("message")
    def on_message(msg: Any) -> None:
        data = json.loads(msg)
        messages.append(data)
        if data.get("type") == "session_ended":
            ended.set()

    @pc.on("track")
    def on_track(track: Any) -> None:
        async def _pull() -> None:
            with contextlib.suppress(Exception):
                while True:
                    frames.append(await track.recv())

        asyncio.ensure_future(_pull())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    resp = await asyncio.to_thread(
        client.post,
        "/voice/rtc",
        json={"sdp": pc.localDescription.sdp, "type": "offer", "config": config or {}},
    )
    assert resp.status_code == 200, resp.text
    await pc.setRemoteDescription(RTCSessionDescription(**resp.json()))
    try:
        await asyncio.wait_for(ended.wait(), timeout=timeout)
    finally:
        await pc.close()
    return messages, frames


def _setup_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, responses: list[str] | None = None) -> None:
    spec = _write_agent_module(tmp_path, responses=responses)
    monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
    for k in VOICE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TIMBAL_STUN_URL", "")  # loopback: host candidates only


class TestVoiceRtcRoundTrip:
    async def test_full_call_over_webrtc(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path, responses=["Hi there!"])
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_delayed_stt_class(
                [(0.2, TranscriptEvent(type="committed", text="Hello"))],
                end_after=1.5,
            ),
        )
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsStreamTTS",
            _make_tts_class(chunk=b"\x01\x02" * 800, num_chunks=2),
        )

        app = create_app()
        with TestClient(app) as client:
            messages, frames = await _rtc_call(client, config={"turn_detector": "heuristic"})

        types = [m["type"] for m in messages]
        started = next(m for m in messages if m["type"] == "session_started")
        assert started["transport"] == "webrtc"
        assert started["playback_acks"] == "native"
        assert "transcript_committed" in types
        assert "agent_text_done" in types
        assert types[-1] == "session_ended"
        # TTS rides the audio track, never the data channel.
        assert "audio" not in types
        # Media actually flowed: the downlink track delivered paced frames.
        assert len(frames) > 10

    async def test_error_before_handshake_still_reaches_the_client(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A session that dies instantly must not vanish into a closed pc.

        The driver waits for the data channel before starting the session, so
        even an immediate failure (here: STT connect raising, live: a bad API
        key) is delivered as an error payload followed by session_ended.
        """

        class _BrokenSTT(SpeechToText):
            def __init__(self, api_key: Any = None) -> None:
                pass

            async def connect(self, config: AudioInputConfig) -> None:
                raise RuntimeError("bad credentials")

            async def push_audio(self, chunk: bytes) -> None:
                pass

            async def commit(self) -> None:
                pass

            async def events(self):
                return
                yield

            async def close(self) -> None:
                pass

        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", _BrokenSTT)
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())

        app = create_app()
        with TestClient(app) as client:
            messages, _ = await _rtc_call(client)

        types = [m["type"] for m in messages]
        assert "error" in types
        assert types[-1] == "session_ended"


class TestVoiceRtcSingleSession:
    async def test_one_call_then_exit_zero_then_409(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """TIMBAL_VOICE_SINGLE_SESSION: serve the call, exit 0, refuse a second offer."""
        _setup_env(monkeypatch, tmp_path, responses=["Hi there!"])
        monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
        exits: list[int] = []
        monkeypatch.setattr("timbal.server.single_session._exit_process", exits.append)
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
            _make_delayed_stt_class(
                [(0.2, TranscriptEvent(type="committed", text="Hello"))],
                end_after=1.5,
            ),
        )
        monkeypatch.setattr(
            "timbal.voice.elevenlabs.ElevenLabsStreamTTS",
            _make_tts_class(chunk=b"\x01\x02" * 800, num_chunks=2),
        )

        app = create_app()
        with TestClient(app) as client:
            messages, _ = await _rtc_call(client)
            assert messages[-1]["type"] == "session_ended"

            # The driver's finally (→ guard.finish) runs after the pc closes.
            for _ in range(100):
                if exits:
                    break
                await asyncio.sleep(0.05)
            assert exits == [0]

            # One process = one session: the 409 fires before any SDP work.
            resp = await asyncio.to_thread(
                client.post, "/voice/rtc", json={"sdp": "v=0", "type": "offer"}
            )
            assert resp.status_code == 409

    async def test_409_while_a_session_is_live(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
        exits: list[int] = []
        monkeypatch.setattr("timbal.server.single_session._exit_process", exits.append)

        app = create_app()
        with TestClient(app) as client:
            assert app.state.single_session_guard.claim()  # a session is live
            resp = await asyncio.to_thread(
                client.post, "/voice/rtc", json={"sdp": "v=0", "type": "offer"}
            )
        assert resp.status_code == 409
        assert exits == []

    async def test_setup_crash_releases_the_slot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A 500 between claim() and the driver task must not park the box in a
        claimed state: the slot reopens and the idle timer bounds the lifetime."""
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
        exits: list[int] = []
        monkeypatch.setattr("timbal.server.single_session._exit_process", exits.append)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("recorder misconfigured")

        monkeypatch.setattr("timbal.server.rtc.build_voice_session", _boom)

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = await asyncio.to_thread(
                client.post, "/voice/rtc", json={"sdp": "v=0", "type": "offer"}
            )
            assert resp.status_code == 500
            assert app.state.single_session_guard.claim()
        assert exits == []

    async def test_rejected_offer_releases_the_slot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A 400 offer never became a session — the slot must reopen."""
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
        exits: list[int] = []
        monkeypatch.setattr("timbal.server.single_session._exit_process", exits.append)

        app = create_app()
        with TestClient(app) as client:
            resp = await asyncio.to_thread(
                client.post, "/voice/rtc", json={"sdp": "not sdp at all", "type": "offer"}
            )
            assert resp.status_code == 400
            assert app.state.single_session_guard.claim()
        assert exits == []


class TestVoiceRtcSignalingErrors:
    def test_rejects_body_without_an_offer(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _setup_env(monkeypatch, tmp_path)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/voice/rtc", json={"type": "offer"})
        assert resp.status_code == 400

    async def test_rejects_offer_without_an_audio_track(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _setup_env(monkeypatch, tmp_path)
        app = create_app()

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        pc.createDataChannel("events")  # SCTP only, no mic
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        try:
            with TestClient(app) as client:
                resp = await asyncio.to_thread(
                    client.post,
                    "/voice/rtc",
                    json={"sdp": pc.localDescription.sdp, "type": "offer"},
                )
        finally:
            await pc.close()
        assert resp.status_code == 400
        assert "audio track" in resp.json()["error"]

    def test_501_without_aiortc(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import sys

        _setup_env(monkeypatch, tmp_path)
        # Simulate a missing extra: None in sys.modules makes `from aiortc
        # import ...` raise ImportError inside the route.
        monkeypatch.setitem(sys.modules, "aiortc", None)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/voice/rtc", json={"sdp": "v=0", "type": "offer"})
        assert resp.status_code == 501
        assert "timbal[voice]" in resp.json()["error"]


class TestVoiceRtcLivekitFork:
    """The same route also takes a LiveKit dial — how a long-lived deployment
    (ECS / on-premise) serves LiveKit calls without a per-call process."""

    def test_livekit_dial_joins_without_aiortc(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The dial is sniffed before the aiortc import: a deployment pinned to
        timbal[voice-livekit] only must not be told the transport is missing."""
        import sys
        from types import SimpleNamespace

        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setitem(sys.modules, "aiortc", None)

        class _FakeRoom:
            def __init__(self) -> None:
                self.local_participant = SimpleNamespace()

            def on(self, name: str, fn: object) -> None:
                pass

            async def connect(self, url: str, token: str) -> None:
                pass

            async def disconnect(self) -> None:
                pass

        monkeypatch.setitem(
            sys.modules,
            "livekit",
            SimpleNamespace(rtc=SimpleNamespace(Room=_FakeRoom, TrackKind=SimpleNamespace(KIND_AUDIO=1))),
        )

        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/voice/rtc",
                json={
                    "transport": "livekit",
                    "url": "ws://sfu:7880",
                    "token": "agent-jwt",
                    "room": "v1_1_2_3_abc",
                    "caller_identity": "caller-abc",
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json() == {
                "transport": "livekit",
                "room": "v1_1_2_3_abc",
                "status": "joined",
            }
            # The call outlives the request; don't leak it into loop teardown.
            live = app.state.livekit_sessions["v1_1_2_3_abc"]
            live.get_loop().call_soon_threadsafe(live.cancel)

    def test_livekit_dial_without_credentials_is_a_400(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _setup_env(monkeypatch, tmp_path)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/voice/rtc", json={"transport": "livekit", "room": "r"})
        assert resp.status_code == 400
        assert "token" in resp.json()["error"]

    def test_livekit_extra_missing_is_a_501(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setitem(sys.modules, "livekit", None)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/voice/rtc",
                json={"transport": "livekit", "url": "ws://sfu:7880", "token": "t", "room": "r"},
            )
        assert resp.status_code == 501
        assert "voice-livekit" in resp.json()["error"]


class TestVoiceRtcDialSecret:
    """An SDP offer costs this process a session; a dial tells it which SFU to
    call and then spends STT/TTS/LLM budget streaming to whoever is in that
    room. So the dial — and only the dial — can be gated on a shared secret."""

    def _dial(self) -> dict:
        return {
            "transport": "livekit",
            "url": "ws://sfu:7880",
            "token": "agent-jwt",
            "room": "v1_1_2_3_abc",
        }

    def test_a_dial_without_the_secret_is_a_401(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_DIAL_SECRET", "s3cret")
        # Fails before the transport is even loaded — a rejected dial must not
        # be distinguishable from a missing extra by how far it gets.
        monkeypatch.setitem(sys.modules, "livekit", None)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/voice/rtc", json=self._dial())
        assert resp.status_code == 401

    def test_a_wrong_secret_is_a_401(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_DIAL_SECRET", "s3cret")
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/voice/rtc", json=self._dial(), headers={"X-Timbal-Dial-Secret": "guess"}
            )
        assert resp.status_code == 401

    def test_the_right_secret_gets_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_DIAL_SECRET", "s3cret")
        monkeypatch.setitem(sys.modules, "livekit", None)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post(
                "/voice/rtc", json=self._dial(), headers={"X-Timbal-Dial-Secret": "s3cret"}
            )
        # Past the gate: 501 is the missing extra, not the credentials.
        assert resp.status_code == 501

    def test_an_sdp_offer_is_not_gated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The secret guards the dial only — an offer carries no destination."""
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_DIAL_SECRET", "s3cret")
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/voice/rtc", json={"sdp": "not-an-sdp", "type": "offer"})
        assert resp.status_code == 400
