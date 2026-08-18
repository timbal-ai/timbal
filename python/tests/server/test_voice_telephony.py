"""End-to-end tests for the telephony routes (Twilio + Telnyx).

Mocks ElevenLabs STT/TTS at the module boundary (like ``test_voice_ws``) so
the full media-WS bridge — start-frame handshake, μ-law decode/encode,
resampling, mark bookkeeping — is exercised through a real Starlette
TestClient WebSocket speaking each provider's dialect.
"""

# ruff: noqa: ARG002
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from timbal.server.http import create_app
from timbal.voice import (
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
)
from timbal.voice.telephony import ulaw_decode

from .voice_env import VOICE_ENV_KEYS

# ---------------------------------------------------------------------------
# Mocks (constructor-compatible with the ElevenLabs classes)
# ---------------------------------------------------------------------------


def _make_stt_class(script: list[TranscriptEvent] | None = None):
    """STT that replays *script* on connect(), then ends the event stream."""
    _script = list(script or [])

    class _STT(SpeechToText):
        pushed: list[bytes] = []

        def __init__(self, api_key=None):
            self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

        async def connect(self, config: AudioInputConfig) -> None:
            for ev in _script:
                await self._queue.put(ev)
            await self._queue.put(None)

        async def push_audio(self, chunk: bytes) -> None:
            type(self).pushed.append(chunk)

        async def commit(self) -> None:
            pass

        async def events(self) -> AsyncIterator[TranscriptEvent]:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                if item.text:
                    yield item

        async def close(self) -> None:
            pass

    return _STT


def _make_manual_stt_class():
    """STT whose event stream ends only when the session closes it."""

    class _STT(SpeechToText):
        pushed: list[bytes] = []

        def __init__(self, api_key=None):
            self._queue: asyncio.Queue[None] = asyncio.Queue()

        async def connect(self, config: AudioInputConfig) -> None:
            pass

        async def push_audio(self, chunk: bytes) -> None:
            type(self).pushed.append(chunk)

        async def commit(self) -> None:
            pass

        async def events(self) -> AsyncIterator[TranscriptEvent]:
            while True:
                if await self._queue.get() is None:
                    break
                yield  # pragma: no cover

        async def close(self) -> None:
            await self._queue.put(None)

    return _STT


def _make_tts_class(chunk: bytes, num_chunks: int = 2):
    class _TTS(TextToSpeech):
        def __init__(self, api_key=None):
            pass

        async def connect(self, config: AudioOutputConfig) -> None:
            pass

        async def synthesize(self, text: str) -> AsyncIterator[bytes]:
            for _ in range(num_chunks):
                yield chunk

        async def close(self) -> None:
            pass

    return _TTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_agent_module(tmp_path: Path) -> str:
    mod = tmp_path / "phone_agent.py"
    mod.write_text(
        "from timbal import Agent\n"
        "from timbal.core.test_model import TestModel\n"
        "agent = Agent(name='phone_test', model=TestModel(responses=['Hi there!']), tools=[])\n"
    )
    return f"{mod.resolve()}::agent"


def _setup_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stt_cls, tts_cls):
    monkeypatch.setenv("TIMBAL_RUNNABLE", _write_agent_module(tmp_path))
    for k in VOICE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TELNYX_PUBLIC_KEY", raising=False)
    monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", stt_cls)
    monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", tts_cls)
    return create_app()


def _collect_frames(ws) -> list[dict]:
    frames: list[dict] = []
    while True:
        try:
            frames.append(ws.receive_json())
        except Exception:
            break
    return frames


def _twilio_start_frame(custom: dict | None = None) -> dict:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "streamSid": "MZ_test_stream",
        "start": {
            "accountSid": "AC_test",
            "callSid": "CA_test_call",
            "tracks": ["inbound"],
            "customParameters": {"turn_detector": "heuristic", **(custom or {})},
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
    }


def _telnyx_start_frame() -> dict:
    return {
        "event": "start",
        "sequence_number": "1",
        "stream_id": "tx-stream-1",
        "start": {
            "user_id": "u-1",
            "call_control_id": "v2:CC_test",
            "from": "+13120000001",
            "to": "+13120000002",
            "custom_parameters": {"turn_detector": "heuristic"},
            "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
        },
    }


# 100ms of PCM16 @ 16kHz per TTS chunk → ~800 μ-law bytes downlink per chunk.
_TTS_CHUNK = b"\x00\x01" * 1600


# ---------------------------------------------------------------------------
# Media WS bridge
# ---------------------------------------------------------------------------


class TestTwilioBridge:
    def test_turn_produces_media_and_marks(self, monkeypatch, tmp_path) -> None:
        pytest.importorskip("av")
        stt_cls = _make_stt_class([TranscriptEvent(type="committed", text="Hello")])
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
            ws.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            ws.send_json(_twilio_start_frame())
            frames = _collect_frames(ws)

        media = [f for f in frames if f["event"] == "media"]
        marks = [f for f in frames if f["event"] == "mark"]
        assert media, f"no media frames in {[f['event'] for f in frames]}"
        assert all(f["streamSid"] == "MZ_test_stream" for f in media + marks)

        # Two 100ms TTS chunks → ~1600 μ-law bytes total on the wire.
        total_ulaw = sum(len(base64.b64decode(f["media"]["payload"])) for f in media)
        assert 1300 <= total_ulaw <= 1900

        # Marks ride cumulative byte counts and pair 1:1 with media frames.
        assert len(marks) == len(media)
        names = [int(f["mark"]["name"]) for f in marks]
        assert names == sorted(names)
        assert names[-1] == total_ulaw

        # Session teardown may fire a trailing clear (audio still "playing"
        # when the session ends counts as an interruption) — but no clear may
        # interrupt the media mid-stream.
        events = [f["event"] for f in frames]
        last_media = max(i for i, e in enumerate(events) if e == "media")
        assert "clear" not in events[:last_media]

    def test_uplink_audio_reaches_stt_resampled(self, monkeypatch, tmp_path) -> None:
        pytest.importorskip("av")
        stt_cls = _make_manual_stt_class()
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        # 40ms of μ-law silence @8k → 640 PCM bytes → ~1280 bytes at 16k.
        payload = base64.b64encode(b"\xff" * 320).decode()
        with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
            ws.send_json({"event": "connected"})
            ws.send_json(_twilio_start_frame())
            ws.send_json({
                "event": "media",
                "streamSid": "MZ_test_stream",
                "media": {"track": "inbound", "chunk": "1", "timestamp": "5", "payload": payload},
            })
            ws.send_json({"event": "stop", "streamSid": "MZ_test_stream", "stop": {}})
            _collect_frames(ws)

        pushed = b"".join(stt_cls.pushed)
        assert 1100 <= len(pushed) <= 1300
        # μ-law 0xFF decodes to digital zero — silence in, silence out.
        assert set(pushed) == {0}

    def test_outbound_track_media_is_ignored(self, monkeypatch, tmp_path) -> None:
        pytest.importorskip("av")
        stt_cls = _make_manual_stt_class()
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        payload = base64.b64encode(b"\xff" * 320).decode()
        with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
            ws.send_json(_twilio_start_frame())
            ws.send_json({
                "event": "media",
                "streamSid": "MZ_test_stream",
                "media": {"track": "outbound", "payload": payload},
            })
            ws.send_json({"event": "stop", "streamSid": "MZ_test_stream", "stop": {}})
            _collect_frames(ws)

        assert stt_cls.pushed == []

    def test_no_start_frame_closes_socket(self, monkeypatch, tmp_path) -> None:
        stt_cls = _make_stt_class([])
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
            # A frame that is not a start → handshake keeps scanning; closing
            # the socket from our side must not hang the handler.
            ws.send_json({"event": "connected"})
            ws.close()

    def test_unsupported_encoding_rejected(self, monkeypatch, tmp_path) -> None:
        stt_cls = _make_stt_class([])
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        frame = _twilio_start_frame()
        frame["start"]["mediaFormat"]["encoding"] = "audio/l16"
        with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
            ws.send_json(frame)
            frames = _collect_frames(ws)
        assert frames == []  # closed without any media


class TestTelephonyCapacity:
    """A phone call is a voice session like any other. On a long-lived box it is
    the transport most likely to produce real concurrency, so the cap has to
    reach it — otherwise the ceiling is a ceiling with a hole in it."""

    def test_a_full_process_refuses_the_call(self, monkeypatch, tmp_path) -> None:
        from starlette.websockets import WebSocketDisconnect
        from timbal.server import capacity

        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        capacity.reset_for_tests()
        assert capacity.acquire_session_slot()  # the box is now full

        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect) as excinfo:  # noqa: PT012
                with client.websocket_connect("/voice/twilio/stream") as ws:
                    ws.send_json(_twilio_start_frame())
                    ws.receive_json()
        # 1013 Try Again Later: the provider is not wrong, this process is full.
        assert excinfo.value.code == 1013

    def test_the_slot_comes_back_when_the_call_ends(self, monkeypatch, tmp_path) -> None:
        from timbal.server import capacity

        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        capacity.reset_for_tests()

        for _ in range(2):
            with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
                ws.send_json(_twilio_start_frame())
                ws.send_json({"event": "stop", "streamSid": "MZ_test_stream", "stop": {}})
                _collect_frames(ws)
            # A leak here would shrink the box one call at a time until every
            # caller gets a busy signal.
            assert capacity.active_sessions() == 0


class TestTelnyxBridge:
    def test_turn_produces_media_and_marks_in_telnyx_dialect(self, monkeypatch, tmp_path) -> None:
        pytest.importorskip("av")
        stt_cls = _make_stt_class([TranscriptEvent(type="committed", text="Hello")])
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        with TestClient(app) as client, client.websocket_connect("/voice/telnyx/stream") as ws:
            ws.send_json({"event": "connected", "version": "1.0.0"})
            ws.send_json(_telnyx_start_frame())
            frames = _collect_frames(ws)

        media = [f for f in frames if f["event"] == "media"]
        marks = [f for f in frames if f["event"] == "mark"]
        assert media
        assert len(marks) == len(media)
        # Telnyx client frames carry no stream id.
        assert all("stream_id" not in f and "streamSid" not in f for f in media + marks)
        # Telnyx requires >=20ms per RTP chunk (160 bytes of μ-law @8k).
        assert all(len(base64.b64decode(f["media"]["payload"])) >= 160 for f in media)

    def test_uplink_snake_case_media(self, monkeypatch, tmp_path) -> None:
        pytest.importorskip("av")
        stt_cls = _make_manual_stt_class()
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(_TTS_CHUNK))

        payload = base64.b64encode(b"\xff" * 320).decode()
        with TestClient(app) as client, client.websocket_connect("/voice/telnyx/stream") as ws:
            ws.send_json(_telnyx_start_frame())
            ws.send_json({
                "event": "media",
                "sequence_number": "2",
                "stream_id": "tx-stream-1",
                "media": {"track": "inbound", "chunk": 1, "timestamp": "5", "payload": payload},
            })
            ws.send_json({"event": "stop", "sequence_number": "3", "stream_id": "tx-stream-1", "stop": {}})
            _collect_frames(ws)

        assert len(b"".join(stt_cls.pushed)) > 0


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


def _twilio_signature(url: str, params: dict[str, str], token: str) -> str:
    payload = url + "".join(k + v for k, v in sorted(params.items()))
    return base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()).decode()


class TestTwilioWebhook:
    def test_returns_twiml_pointing_at_stream_ws(self, monkeypatch, tmp_path) -> None:
        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        with TestClient(app) as client:
            r = client.post(
                "/voice/twilio/incoming",
                data={"CallSid": "CA123", "From": "+15550001111", "To": "+15550002222"},
            )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/xml")
        assert "<Connect><Stream url=" in r.text
        assert "ws://testserver/voice/twilio/stream" in r.text
        # Caller metadata tunnels through custom parameters.
        assert '<Parameter name="from" value="+15550001111" />' in r.text
        assert '<Parameter name="call_sid" value="CA123" />' in r.text

    def test_identity_query_joins_from_to_parameters(self, monkeypatch, tmp_path) -> None:
        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        with TestClient(app) as client:
            r = client.post(
                "/voice/twilio/incoming?rep_id=R001&task=eod_checkin",
                data={"CallSid": "CA123", "From": "+15550001111", "To": "+15550002222"},
            )
        assert '<Parameter name="rep_id" value="R001" />' in r.text
        assert '<Parameter name="task" value="eod_checkin" />' in r.text
        assert '<Parameter name="from" value="+15550001111" />' in r.text

    def test_https_proxy_yields_wss_url(self, monkeypatch, tmp_path) -> None:
        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        with TestClient(app) as client:
            r = client.post(
                "/voice/twilio/incoming",
                data={"CallSid": "CA123"},
                headers={"x-forwarded-proto": "https", "x-forwarded-host": "phone.example.com"},
            )
        assert "wss://phone.example.com/voice/twilio/stream" in r.text

    def test_signature_enforced_when_token_set(self, monkeypatch, tmp_path) -> None:
        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret-token")
        params = {"CallSid": "CA123", "From": "+15550001111"}
        good = _twilio_signature("http://testserver/voice/twilio/incoming", params, "secret-token")

        with TestClient(app) as client:
            ok = client.post("/voice/twilio/incoming", data=params, headers={"X-Twilio-Signature": good})
            bad = client.post("/voice/twilio/incoming", data=params, headers={"X-Twilio-Signature": "nope"})
            missing = client.post("/voice/twilio/incoming", data=params)
        assert ok.status_code == 200
        assert bad.status_code == 403
        assert missing.status_code == 403


class TestTelnyxWebhook:
    def test_returns_texml_with_rtp_mode(self, monkeypatch, tmp_path) -> None:
        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        with TestClient(app) as client:
            r = client.post("/voice/telnyx/incoming", data={"CallSid": "CC123"})
        assert r.status_code == 200
        assert "ws://testserver/voice/telnyx/stream" in r.text
        assert 'bidirectionalMode="rtp"' in r.text
        assert 'bidirectionalCodec="PCMU"' in r.text

    def test_identity_query_becomes_texml_parameters(self, monkeypatch, tmp_path) -> None:
        """``rep_id`` / ``task`` on our webhook URL ride TeXML Parameters.

        Random query keys (and ``rev``) must not — they are not identity and
        must not become start-frame custom_parameters.
        """
        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        with TestClient(app) as client:
            r = client.post(
                "/voice/telnyx/incoming?rev=main&rep_id=R001&task=eod_checkin&evil=dropme",
                data={"CallSid": "CC123"},
            )
        assert r.status_code == 200
        assert '<Parameter name="rep_id" value="R001" />' in r.text
        assert '<Parameter name="task" value="eod_checkin" />' in r.text
        assert "evil" not in r.text
        assert 'name="rev"' not in r.text

    def test_ed25519_signature_enforced_when_key_set(self, monkeypatch, tmp_path) -> None:
        crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        priv = crypto.Ed25519PrivateKey.generate()
        pub_b64 = base64.b64encode(priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()

        app = _setup_app(monkeypatch, tmp_path, _make_stt_class([]), _make_tts_class(_TTS_CHUNK))
        monkeypatch.setenv("TELNYX_PUBLIC_KEY", pub_b64)

        body = b"CallSid=CC123&From=%2B13120000001"
        timestamp = "1700000000"
        signature = base64.b64encode(priv.sign(f"{timestamp}|".encode() + body)).decode()

        with TestClient(app) as client:
            ok = client.post(
                "/voice/telnyx/incoming",
                content=body,
                headers={
                    "content-type": "application/x-www-form-urlencoded",
                    "telnyx-signature-ed25519": signature,
                    "telnyx-timestamp": timestamp,
                },
            )
            bad = client.post(
                "/voice/telnyx/incoming",
                content=body,
                headers={
                    "content-type": "application/x-www-form-urlencoded",
                    "telnyx-signature-ed25519": base64.b64encode(b"\x00" * 64).decode(),
                    "telnyx-timestamp": timestamp,
                },
            )
        assert ok.status_code == 200
        assert bad.status_code == 403

    def test_missing_cryptography_fails_closed(self, monkeypatch) -> None:
        """Key set + cryptography unimportable must reject, not bypass auth."""
        import sys

        from timbal.server.telephony import _telnyx_signature_ok

        monkeypatch.setitem(sys.modules, "cryptography.exceptions", None)
        assert _telnyx_signature_ok(b"body", "123", base64.b64encode(b"\x00" * 64).decode(), "notakey") is False


class TestUlawWire:
    def test_media_payload_is_valid_ulaw(self, monkeypatch, tmp_path) -> None:
        """The downlink payload must decode to real PCM (sanity vs base64 PCM)."""
        pytest.importorskip("av")
        stt_cls = _make_stt_class([TranscriptEvent(type="committed", text="Hello")])
        # Loud constant tone so μ-law bytes are far from the 0xFF silence code.
        loud = (16_000).to_bytes(2, "little", signed=True) * 1600
        app = _setup_app(monkeypatch, tmp_path, stt_cls, _make_tts_class(loud))

        with TestClient(app) as client, client.websocket_connect("/voice/twilio/stream") as ws:
            ws.send_json(_twilio_start_frame())
            frames = _collect_frames(ws)

        media = [f for f in frames if f["event"] == "media"]
        assert media
        ulaw = base64.b64decode(media[0]["media"]["payload"])
        pcm = ulaw_decode(ulaw)
        values = [int.from_bytes(pcm[i : i + 2], "little", signed=True) for i in range(0, len(pcm), 2)]
        # Resampler ramps in, but the steady-state must sit near 16000.
        steady = values[len(values) // 2 :]
        assert sum(steady) / len(steady) > 10_000


class TestCallContext:
    def test_leftover_custom_not_in_config_keys(self) -> None:
        from timbal.server.telephony import _CONFIG_PARAM_KEYS, _call_context_from_start

        info = {"from": "+1555", "to": "+1666", "call_id": "CA1"}
        custom = {
            "rep_id": "R001",
            "task": "eod_checkin",
            "greeting": "Hey",
            "stt_provider": "deepgram",
            "turn_detector": "heuristic",
        }
        ctx = _call_context_from_start(info, custom)
        assert ctx["rep_id"] == "R001"
        assert ctx["task"] == "eod_checkin"
        assert ctx["from"] == "+1555"
        assert ctx["to"] == "+1666"
        assert ctx["call_id"] == "CA1"
        for key in _CONFIG_PARAM_KEYS:
            assert key not in ctx

    def test_info_fallbacks_do_not_clobber_custom(self) -> None:
        from timbal.server.telephony import _call_context_from_start

        ctx = _call_context_from_start(
            {"from": "+info", "to": "+info-to", "call_id": "info-id"},
            {"from": "+custom", "rep_id": "R2"},
        )
        assert ctx["from"] == "+custom"
        assert ctx["to"] == "+info-to"
        assert ctx["call_id"] == "info-id"
        assert ctx["rep_id"] == "R2"

    def test_identity_params_allowlist(self) -> None:
        from timbal.server.telephony import _identity_params

        assert _identity_params({"rep_id": "R001", "task": "eod_checkin", "rev": "main", "evil": "x"}) == {
            "rep_id": "R001",
            "task": "eod_checkin",
        }
        assert _identity_params({"rev": "main"}) == {}
        assert _identity_params(None) == {}

    def test_identity_params_are_configurable(self, monkeypatch) -> None:
        """The allowlist is the only thing between a query string and the
        prompt, so it is env-owned — never widened from the wire."""
        from timbal.server.telephony import _identity_params

        monkeypatch.setenv("TIMBAL_VOICE_IDENTITY_PARAMS", "tenant, agent_id")
        assert _identity_params({"tenant": "acme", "agent_id": "A1", "rep_id": "R001"}) == {
            "tenant": "acme",
            "agent_id": "A1",
        }

    def test_blank_identity_env_keeps_the_defaults(self, monkeypatch) -> None:
        from timbal.server.telephony import _identity_params

        monkeypatch.setenv("TIMBAL_VOICE_IDENTITY_PARAMS", " , ")
        assert _identity_params({"rep_id": "R001"}) == {"rep_id": "R001"}
