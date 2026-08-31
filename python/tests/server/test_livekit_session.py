"""LiveKit session driver: env gating, reliable-data chunking, cancellation.

Does not talk to a real SFU — the FFI extra is not required; the cancellation
tests inject a fake ``livekit`` module.
"""

# ruff: noqa: ARG001  — fakes have to match the real signatures they stand in for
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import sys
from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server import capacity
from timbal.server.livekit_session import (
    _run_livekit_session,
    chunk_data_payloads,
    dial_from_body,
    dial_from_env,
    is_config_hello,
    is_livekit_dial,
    maybe_start_livekit_session,
    merge_client_config,
    room_from_token,
    start_livekit_session,
)
from timbal.server.livekit_sip import is_eligible_caller


def _jwt(payload: str) -> str:
    """A JWT-shaped string around a raw payload — unpadded, as real ones are."""
    encoded = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _token_for(room: str, *, identity: str = "agent") -> str:
    """A LiveKit-shaped JWT: only the payload's ``video.room`` grant matters."""
    return _jwt(json.dumps({"sub": identity, "video": {"room": room, "roomJoin": True}}))


def _reassemble(chunks: list[dict]) -> dict:
    """What a client does with ``chunk`` envelopes: concat by seq, decode, parse."""
    ordered = sorted(chunks, key=lambda c: c["seq"])
    return json.loads(base64.b64decode("".join(c["data"] for c in ordered)))


class TestMaybeStart:
    def test_off_when_transport_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("TIMBAL_VOICE_TRANSPORT", raising=False)
        assert maybe_start_livekit_session(SimpleNamespace()) is None

    def test_off_when_transport_is_webrtc(self, monkeypatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_TRANSPORT", "webrtc")
        assert maybe_start_livekit_session(SimpleNamespace()) is None

    async def test_starts_a_task_when_livekit(self, monkeypatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_TRANSPORT", "livekit")
        task = maybe_start_livekit_session(SimpleNamespace(state=SimpleNamespace(runnable=None)))
        assert task is not None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestCallerIdentity:
    def test_standard_remote_is_eligible_without_prefix(self) -> None:
        from types import SimpleNamespace

        p = SimpleNamespace(identity="any-id", kind="PARTICIPANT_KIND_STANDARD")
        assert is_eligible_caller(p, local_identity="agent-1", caller_hint="caller-")

    def test_egress_is_not_eligible(self) -> None:
        from types import SimpleNamespace

        p = SimpleNamespace(identity="rec", kind="PARTICIPANT_KIND_EGRESS")
        assert not is_eligible_caller(p, local_identity="agent-1")


class TestChunkDataPayloads:
    def test_small_payloads_pass_through(self) -> None:
        payloads = [{"type": "interrupted", "heard_text": "hi"}]
        assert chunk_data_payloads(payloads) == payloads

    def test_oversized_transcript_is_split_with_seq_total(self) -> None:
        # Each entry is ~200 bytes; 100 of them blow past 12 KiB.
        entries = [{"role": "assistant", "text": "x" * 180, "timestamp": 1.0 + i} for i in range(100)]
        payloads = [{"type": "session_transcript", "entries": entries, "started_at": 1.0}]
        chunks = chunk_data_payloads(payloads)
        assert len(chunks) > 1
        assert all(c["type"] == "session_transcript" for c in chunks)
        assert [c["seq"] for c in chunks] == list(range(len(chunks)))
        assert all(c["total"] == len(chunks) for c in chunks)
        assert all(c["started_at"] == 1.0 for c in chunks)
        rebuilt = [e for c in chunks for e in c["entries"]]
        assert rebuilt == entries

    def test_transcript_chunks_keep_their_own_type(self) -> None:
        """The entry-wise split predates the generic envelope; clients that
        already reassemble transcripts by seq must not have to change."""
        entries = [{"role": "user", "text": "z" * 180, "timestamp": float(i)} for i in range(100)]
        chunks = chunk_data_payloads([{"type": "session_transcript", "entries": entries}])
        assert {c["type"] for c in chunks} == {"session_transcript"}

    def test_oversized_payload_is_chunked_not_dropped(self) -> None:
        """Sending an oversized payload whole means the SFU rejects it, which
        from the client is indistinguishable from never emitting it."""
        payload = {"type": "agent_text_done", "text": "y" * 20_000}
        chunks = chunk_data_payloads([payload])

        assert len(chunks) > 1
        assert {c["type"] for c in chunks} == {"chunk"}
        assert {c["msg_type"] for c in chunks} == {"agent_text_done"}
        assert len({c["chunk_id"] for c in chunks}) == 1
        assert [c["seq"] for c in chunks] == list(range(len(chunks)))
        assert all(c["total"] == len(chunks) for c in chunks)
        assert _reassemble(chunks) == payload

    def test_every_chunk_fits_under_the_cap(self) -> None:
        approval = {
            "type": "agent_approval",
            "run_id": "r1",
            "approval_id": "a1",
            "ui": {"prompt": "schema: " + "s" * 60_000},
        }
        chunks = chunk_data_payloads([approval])
        assert all(len(json.dumps(c, separators=(",", ":")).encode()) <= 12 * 1024 for c in chunks)
        assert _reassemble(chunks) == approval

    def test_non_ascii_survives_a_chunk_boundary(self) -> None:
        """Slices are taken over base64, so a cut can never land inside a
        multi-byte codepoint."""
        payload = {"type": "agent_approval", "run_id": "r", "prompt": "€ø漢" * 8_000}
        assert _reassemble(chunk_data_payloads([payload])) == payload

    def test_interleaved_payloads_keep_distinct_chunk_ids(self) -> None:
        a = {"type": "agent_approval", "approval_id": "a", "ui": {"p": "a" * 20_000}}
        b = {"type": "agent_approval", "approval_id": "b", "ui": {"p": "b" * 20_000}}
        chunks = chunk_data_payloads([a, b])
        by_id: dict[str, list[dict]] = {}
        for c in chunks:
            by_id.setdefault(c["chunk_id"], []).append(c)
        assert len(by_id) == 2
        assert sorted(_reassemble(g)["approval_id"] for g in by_id.values()) == ["a", "b"]


class TestMergeClientConfig:
    def test_hello_overlays_env(self) -> None:
        merged = merge_client_config(
            '{"stt_provider": "elevenlabs", "model": "env/model"}',
            {"stt_provider": "deepgram-flux", "turn_detector": "provider"},
        )
        assert merged["stt_provider"] == "deepgram-flux"
        assert merged["turn_detector"] == "provider"
        assert merged["model"] == "env/model"

    def test_bad_env_json_is_empty_base(self) -> None:
        assert merge_client_config("not-json", {"tts_provider": "elevenlabs"}) == {"tts_provider": "elevenlabs"}

    def test_no_hello_keeps_env(self) -> None:
        assert merge_client_config('{"voice": "abc"}', None) == {"voice": "abc"}


class TestConfigHello:
    def test_untyped_object_is_hello(self) -> None:
        assert is_config_hello({"sample_rate": 16000, "stt_provider": "elevenlabs"})

    def test_typed_frame_is_not_hello(self) -> None:
        assert not is_config_hello({"type": "playback", "played_ms": 12})

    def test_null_type_is_hello(self) -> None:
        assert is_config_hello({"type": None, "sample_rate": 16000})


class _FakeParticipant:
    def __init__(self) -> None:
        self.dtmf_published: list[tuple[int, str]] = []

    async def publish_data(self, *args: object, **kwargs: object) -> None:
        pass

    async def publish_track(self, *args: object, **kwargs: object) -> None:
        pass

    async def publish_dtmf(self, *, code: int, digit: str) -> None:
        self.dtmf_published.append((code, digit))


class _FakeRoom:
    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.disconnect_entered = asyncio.Event()
        self.disconnect_gate: asyncio.Event | None = None
        self.disconnected = False
        self.local_participant = _FakeParticipant()
        self.local_participant.identity = "agent-test"
        self.remote_participants: dict[str, SimpleNamespace] = {}
        self.handlers: dict[str, object] = {}

    def on(self, name: str, fn: object) -> None:
        self.handlers[name] = fn

    async def connect(self, url: str, token: str) -> None:
        self.connected.set()

    async def disconnect(self) -> None:
        self.disconnect_entered.set()
        if self.disconnect_gate is not None:
            await self.disconnect_gate.wait()
        self.disconnected = True


class _FakeGuard:
    def __init__(self) -> None:
        self.released = False
        self.finished = False
        self.connected = False
        self.on_abandon: object = None
        self.disconnect_calls = 0

    def claim(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True

    def mark_connected(self) -> None:
        self.connected = True

    def mark_reconnected(self) -> None:
        pass

    def mark_disconnected(self, *, on_abandon: object = None) -> None:
        self.disconnect_calls += 1
        self.on_abandon = on_abandon

    async def finish(self) -> None:
        self.finished = True


class _LogRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def _rec(self, event: str, **kwargs: object) -> None:
        self.events.append(event)

    info = warning = error = debug = _rec


@pytest.fixture
def driver_env(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeRoom, _FakeGuard, _LogRecorder, object]:
    """Fake livekit module + minimal app so the driver runs without the FFI."""
    room = _FakeRoom()
    fake_rtc = SimpleNamespace(
        Room=lambda: room,
        TrackKind=SimpleNamespace(KIND_AUDIO=1),
        ParticipantKind=SimpleNamespace(
            PARTICIPANT_KIND_STANDARD=0,
            PARTICIPANT_KIND_SIP=3,
            PARTICIPANT_KIND_EGRESS=1,
            PARTICIPANT_KIND_INGRESS=2,
        ),
        AudioSource=lambda *a, **k: SimpleNamespace(clear_queue=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=fake_rtc))
    monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "ws://fake:7880")
    monkeypatch.setenv("TIMBAL_LIVEKIT_TOKEN", "tok")
    monkeypatch.setenv("TIMBAL_LIVEKIT_AGENT_IDENTITY", "agent-test")
    monkeypatch.delenv("TIMBAL_VOICE_CLIENT_CONFIG", raising=False)
    log = _LogRecorder()
    monkeypatch.setattr("timbal.server.livekit_session.logger", log)
    guard = _FakeGuard()
    agent = Agent(name="lk_test", model=TestModel(responses=["hi"]), tools=[])
    app = SimpleNamespace(state=SimpleNamespace(runnable=agent, single_session_guard=guard, voice_config=None))
    return room, guard, log, app


class TestCancellationCleanup:
    """Lifespan teardown cancels the driver — cleanup must still complete."""

    async def test_cancel_while_waiting_for_caller_disconnects_and_releases(
        self, driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object]
    ) -> None:
        room, guard, log, app = driver_env
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)  # let the driver reach caller_ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert room.disconnected
        assert guard.released
        assert not guard.finished  # session never started
        assert "voice_livekit_disconnected" in log.events

    async def test_second_cancel_mid_finally_does_not_skip_the_tail(
        self, driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object]
    ) -> None:
        """Emulates the loop-shutdown mass-cancel: a CancelledError delivered
        at an await inside the finally must not skip the steps after it."""
        room, guard, log, app = driver_env
        room.disconnect_gate = asyncio.Event()
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait_for(room.disconnect_entered.wait(), timeout=1.0)
        task.cancel()  # second cancel lands at the await inside disconnect()
        room.disconnect_gate.set()
        done, _ = await asyncio.wait({task}, timeout=1.0)
        assert task in done and task.cancelled()
        assert "voice_livekit_disconnected" in log.events  # tail still ran


def _subscribe_caller(
    room: _FakeRoom,
    *,
    identity: str = "playground",
    kind: str = "PARTICIPANT_KIND_STANDARD",
    attributes: dict | None = None,
) -> SimpleNamespace:
    participant = SimpleNamespace(
        identity=identity,
        kind=kind,
        attributes=attributes or {},
        disconnect_reason=None,
    )
    track = SimpleNamespace(kind=1)
    room.handlers["track_subscribed"](track, None, participant)
    return participant


def _deliver_hello(room: _FakeRoom, hello: dict) -> None:
    import json as _json

    room.handlers["data_received"](SimpleNamespace(data=_json.dumps(hello).encode()))


class TestGuardLifetimeAroundSessionBuild:
    async def test_build_failure_after_media_exits_via_finish(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mark_connected (mic subscribe) disarms the idle timer before the
        session is built — a build failure must finish(), not release(),
        or the box is unclaimed, idle-disarmed and immortal."""
        room, guard, _log, app = driver_env

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("recorder misconfigured")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _boom)
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        _deliver_hello(room, {"sample_rate": 16000})  # skip the 2s hello wait
        _subscribe_caller(room)
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done
        assert isinstance(task.exception(), RuntimeError)
        assert guard.connected
        assert guard.finished
        assert not guard.released
        assert room.disconnected

    async def test_failure_before_media_releases_the_claim(
        self, driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object]
    ) -> None:
        """Before the mic subscribe the idle timer still owns the exit."""
        room, guard, _log, app = driver_env

        async def _connect_boom(url: str, token: str) -> None:
            raise RuntimeError("sfu unreachable")

        room.connect = _connect_boom
        task = asyncio.create_task(_run_livekit_session(app))
        done, _ = await asyncio.wait({task}, timeout=1.0)
        assert task in done
        assert guard.released
        assert not guard.finished

    async def test_caller_drop_before_session_build_arms_abandon(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A drop during the hello wait / session build (session_holder still
        empty) must arm the abandon window; the closure closes whichever
        session exists at abandon time."""
        room, guard, _log, app = driver_env

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("stop before session exists")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _boom)
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        _deliver_hello(room, {"sample_rate": 16000})
        participant = _subscribe_caller(room)
        # Drop before the driver has built the session (it is still parked
        # behind caller_ready in the same loop turn).
        room.handlers["participant_disconnected"](participant)
        assert guard.disconnect_calls == 1
        assert guard.on_abandon is not None
        assert guard.on_abandon() is None  # no session yet — closure is a no-op
        await asyncio.wait({task}, timeout=2.0)

    async def test_hello_before_subscribe_is_applied_to_the_session(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The page may re-send the hello as soon as the agent joins — before
        the mic track is subscribed. It must be buffered and merged."""
        room, guard, _log, app = driver_env
        seen: dict = {}

        def _capture(runnable: object, defaults: object, config: dict, **kwargs: object):
            seen["config"] = config
            raise RuntimeError("stop after capture")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        _deliver_hello(room, {"sample_rate": 16000, "stt_provider": "deepgram-flux"})
        _subscribe_caller(room)
        await asyncio.wait({task}, timeout=2.0)
        assert seen["config"]["stt_provider"] == "deepgram-flux"
        assert guard.finished


class TestSipRuntime:
    async def test_sip_bye_before_media_releases_guard(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
    ) -> None:
        """_note_caller from remotes already in the room, then a SIP BYE
        before the mic track — must not stay parked on caller_ready."""
        room, guard, _log, app = driver_env
        sip = SimpleNamespace(
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
            attributes={"sip.callID": "c1"},
            disconnect_reason=SimpleNamespace(name="CLIENT_INITIATED"),
        )
        room.remote_participants[sip.identity] = sip
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert not task.done()
        room.handlers["participant_disconnected"](sip)
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done
        assert guard.released
        assert not guard.finished
        assert room.disconnected

    async def test_sip_bye_finishes_guard_when_session_not_built(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
    ) -> None:
        room, guard, _log, app = driver_env
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        participant = _subscribe_caller(
            room,
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
        )
        participant.disconnect_reason = SimpleNamespace(name="CLIENT_INITIATED")
        room.handlers["participant_disconnected"](participant)
        await asyncio.sleep(0.05)
        assert guard.finished
        assert guard.disconnect_calls == 0
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_sip_dtmf_handler_is_registered(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
    ) -> None:
        room, _guard, _log, app = driver_env
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        assert "sip_dtmf_received" in room.handlers
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_sip_path_applies_phone_tuned_config(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        room, guard, _log, app = driver_env
        seen: dict = {}

        def _capture(runnable: object, defaults: object, config: dict, **kwargs: object):
            seen["config"] = config
            raise RuntimeError("stop")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        _deliver_hello(room, {"sample_rate": 16000})
        _subscribe_caller(
            room,
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
            attributes={"sip.phoneNumber": "+34111"},
        )
        await asyncio.wait({task}, timeout=2.0)
        assert seen["config"]["stt_extra"]["vad_threshold"] == 0.55
        assert guard.finished

    async def test_browser_caller_does_not_invoke_sip_recording_meta(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> dict:
            raise AssertionError("sip_recording_meta must not run for browser callers")

        monkeypatch.setattr("timbal.server.livekit_sip.sip_recording_meta", _boom)

        def _stop(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("stop")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _stop)
        room, guard, _log, app = driver_env
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        _deliver_hello(room, {"sample_rate": 16000})
        _subscribe_caller(room, attributes={"lk.theme": "dark"})
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done
        assert isinstance(task.exception(), RuntimeError)
        assert guard.finished

    async def test_sip_blip_during_hello_does_not_build_session(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Short-abandon after media but before session_holder must abort
        the hello wait — not tear the room and then still build_voice_session."""
        room, guard, _log, app = driver_env
        built: list[object] = []

        def _capture(*_args: object, **_kwargs: object) -> None:
            built.append(True)
            raise RuntimeError("should not build after short-abandon")

        monkeypatch.setenv("TIMBAL_VOICE_SIP_ABANDON_SECS", "0.05")
        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        participant = _subscribe_caller(
            room,
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
        )
        participant.disconnect_reason = SimpleNamespace(name="STATE_MISMATCH")
        room.handlers["participant_disconnected"](participant)
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done
        assert built == []
        assert guard.finished

    async def test_late_media_after_sip_bye_does_not_build(
        self,
        driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """BYE sets session_aborted; a late track_subscribed must not flip
        caller_ready and sneak past the abort gate."""
        room, guard, _log, app = driver_env
        built: list[object] = []

        def _capture(*_args: object, **_kwargs: object) -> None:
            built.append(True)
            raise RuntimeError("should not build after BYE")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        sip = SimpleNamespace(
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
            attributes={"sip.callID": "c1"},
            disconnect_reason=SimpleNamespace(name="CLIENT_INITIATED"),
        )
        room.remote_participants[sip.identity] = sip
        task = asyncio.create_task(_run_livekit_session(app))
        await asyncio.wait_for(room.connected.wait(), timeout=1.0)
        await asyncio.sleep(0)
        room.handlers["participant_disconnected"](sip)
        _subscribe_caller(room, identity="+34111", kind="PARTICIPANT_KIND_SIP")
        done, _ = await asyncio.wait({task}, timeout=2.0)
        assert task in done
        assert built == []
        assert guard.released
        assert not guard.finished


class TestDialParsing:
    def test_transport_discriminates_the_body(self) -> None:
        assert is_livekit_dial({"transport": "livekit"})
        assert not is_livekit_dial({"sdp": "v=0...", "type": "offer"})
        assert not is_livekit_dial({"transport": "webrtc"})
        assert not is_livekit_dial(None)

    def test_body_dial_carries_config_as_json(self) -> None:
        dial = dial_from_body(
            {
                "transport": "livekit",
                "url": "ws://sfu:7880",
                "token": "tok",
                "room": "v1_1_2_3_abc",
                "caller_identity": "caller-abc",
                "config": {"stt_provider": "deepgram-flux"},
            }
        )
        assert (dial.url, dial.token, dial.room) == ("ws://sfu:7880", "tok", "v1_1_2_3_abc")
        assert dial.caller_identity == "caller-abc"
        assert merge_client_config(dial.client_config, None) == {"stt_provider": "deepgram-flux"}

    def test_non_object_config_is_dropped(self) -> None:
        dial = dial_from_body({"transport": "livekit", "url": "u", "token": "t", "config": "x"})
        assert dial.client_config == "{}"


class TestDialParentId:
    """The run a call continues (text → voice) rides the dial — minted by
    whoever authorized the call — never the browser's data-channel hello."""

    def test_body_dial_carries_it(self) -> None:
        dial = dial_from_body({"transport": "livekit", "url": "u", "token": "t", "parent_id": " run-1 "})
        assert dial.parent_id == "run-1"

    def test_absent_means_fresh_thread(self) -> None:
        assert dial_from_body({"transport": "livekit", "url": "u", "token": "t"}).parent_id == ""

    def test_boot_env_dial_reads_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_PARENT_RUN_ID", "run-2")
        assert dial_from_env().parent_id == "run-2"

    async def test_driver_passes_it_to_the_session_build(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        room, app = ecs_app
        seen: dict = {}

        def _capture(runnable: object, defaults: object, config: dict, **kwargs: object):
            seen["parent_run_id"] = kwargs.get("parent_run_id")
            raise RuntimeError("stop after capture")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        status, _body = await start_livekit_session(app, dial_from_body(_dial(parent_id="run-3")))
        assert status == 200
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        _deliver_hello(room, {"sample_rate": 16000})
        _subscribe_caller(room)
        await asyncio.wait({live}, timeout=2.0)
        assert seen["parent_run_id"] == "run-3"

    async def test_a_hello_supplied_parent_id_is_not_the_dials(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The data channel is the browser's: a hello parent_id must not become
        the seed on this transport."""
        room, app = ecs_app
        seen: dict = {}

        def _capture(runnable: object, defaults: object, config: dict, **kwargs: object):
            seen["parent_run_id"] = kwargs.get("parent_run_id")
            raise RuntimeError("stop after capture")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        status, _body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 200
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        _deliver_hello(room, {"sample_rate": 16000, "parent_id": "attacker-thread"})
        _subscribe_caller(room)
        await asyncio.wait({live}, timeout=2.0)
        assert seen["parent_run_id"] is None


class TestRoomFromToken:
    """The token pins the room; the body's `room` is only a label."""

    def test_reads_the_video_room_grant(self) -> None:
        assert room_from_token(_token_for("v1_1_2_3_abc")) == "v1_1_2_3_abc"

    def test_padding_is_restored_before_decoding(self) -> None:
        # base64 payloads whose length isn't a multiple of 4 are the common case.
        for room in ("a", "ab", "abc", "abcd", "room-with-a-longer-name"):
            assert room_from_token(_token_for(room)) == room

    @pytest.mark.parametrize(
        "token",
        [
            "",
            "not-a-jwt",
            "header.!!!not-base64!!!.sig",
            _jwt("[1, 2, 3]"),  # payload is not an object
            _jwt("{}"),  # no grants
            _jwt('{"video": "nope"}'),  # grants are not an object
            _jwt('{"video": {"roomJoin": true}}'),  # no room claim
            _jwt("not json at all"),
        ],
    )
    def test_anything_unparseable_is_empty_not_an_error(self, token: str) -> None:
        """A bad token is the SFU's problem to reject; this only needs a key."""
        assert room_from_token(token) == ""


@pytest.fixture
def ecs_app(
    driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeRoom, object]:
    """A long-lived server: no single-session guard, no LiveKit boot env."""
    room, _guard, _log, app = driver_env
    app.state.single_session_guard = None
    # The dial path defaults to the `auto` ceiling, which is sized from this
    # machine's CPU — pin it so a test that runs two rooms doesn't depend on how
    # many cores the runner has. Capacity tests below set their own.
    monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "16")
    capacity.reset_for_tests()
    return room, app


def _dial(**over: object) -> dict:
    body = {
        "transport": "livekit",
        "url": "ws://fake:7880",
        "token": "tok",
        "room": "v1_1_2_3_abc",
        "agent_identity": "agent-test",
        "caller_identity": "caller-hint",
    }
    body.update(over)
    return body


class TestStartLivekitSession:
    """Per-request join (ECS / on-premise): the process serves room after room."""

    async def test_join_answers_200_and_leaves_the_session_running(self, ecs_app: tuple[_FakeRoom, object]) -> None:
        room, app = ecs_app
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 200
        assert body == {"transport": "livekit", "room": "v1_1_2_3_abc", "status": "joined"}
        # 200 means "the agent is in the room", not "the call is over".
        assert room.connected.is_set()
        assert not room.disconnected
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        assert not live.done()
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live

    async def test_second_join_for_the_same_room_conflicts(self, ecs_app: tuple[_FakeRoom, object]) -> None:
        _room, app = ecs_app
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 409
        assert "already live" in body["error"]
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live

    async def test_a_finished_room_frees_its_key_for_the_next_call(self, ecs_app: tuple[_FakeRoom, object]) -> None:
        """The point of the per-request path: one call ending must not retire
        the process (nor its room key)."""
        _room, app = ecs_app
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live
        assert "v1_1_2_3_abc" not in app.state.livekit_sessions
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        again = app.state.livekit_sessions["v1_1_2_3_abc"]
        again.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await again

    async def test_missing_credentials_is_a_400(self, ecs_app: tuple[_FakeRoom, object]) -> None:
        _room, app = ecs_app
        status, body = await start_livekit_session(app, dial_from_body(_dial(token="")))
        assert status == 400
        assert "url" in body["error"]

    async def test_unreachable_sfu_surfaces_the_join_failure(
        self, ecs_app: tuple[_FakeRoom, object], driver_env: tuple[object, object, _LogRecorder, object]
    ) -> None:
        room, app = ecs_app
        _r, _g, log, _a = driver_env

        async def _connect_boom(url: str, token: str) -> None:
            raise RuntimeError("sfu unreachable")

        room.connect = _connect_boom
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 502
        # The caller learns the join failed; *why* is in the log (see
        # `test_a_join_failure_does_not_echo_internals_to_the_caller`).
        assert body == {"error": "the agent could not join the room"}
        assert "voice_livekit_join_failed" in log.events

    async def test_a_hung_join_times_out_and_cancels_the_task(self, ecs_app: tuple[_FakeRoom, object]) -> None:
        room, app = ecs_app

        async def _connect_hang(url: str, token: str) -> None:
            await asyncio.Event().wait()

        room.connect = _connect_hang
        status, body = await start_livekit_session(app, dial_from_body(_dial()), timeout=0.05)
        assert status == 504
        assert "did not join" in body["error"]
        # The key is freed with the 504 (see the retry test below), so reach the
        # task by name rather than looking it up by room.
        live = next(
            t for t in asyncio.all_tasks() if t.get_name().startswith("voice-livekit-session:")
        )
        assert live.cancelled() or live.cancelling()
        with contextlib.suppress(asyncio.CancelledError):
            await live
        assert "v1_1_2_3_abc" not in app.state.livekit_sessions

    async def test_a_504_frees_the_room_for_the_retry(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        """A caller that got a 504 retries. The abandoned task still has awaits
        left in its teardown, so keying on "not done" would answer that retry
        with a 409 for a room nothing is serving."""
        room, app = ecs_app
        hang = asyncio.Event()

        async def _connect_hang(url: str, token: str) -> None:
            await hang.wait()

        room.connect = _connect_hang
        assert (await start_livekit_session(app, dial_from_body(_dial()), timeout=0.05))[0] == 504
        assert "v1_1_2_3_abc" not in app.state.livekit_sessions

        # The retry succeeds instead of colliding with the corpse.
        hang.set()
        del room.connect  # back to the fixture's instant connect
        status, _body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 200
        for task in list(app.state.livekit_sessions.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_one_token_cannot_join_its_room_twice(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        """Sessions key on the token's `video.room` grant, not the body's label.
        Keying on the label lets one token put two agents in one real room,
        both publishing, talking over each other."""
        _room, app = ecs_app
        token = _token_for("v1_1_2_3_abc")

        assert (await start_livekit_session(app, dial_from_body(_dial(token=token))))[0] == 200
        # Same token — so the same actual room — under a different label.
        status, body = await start_livekit_session(
            app, dial_from_body(_dial(token=token, room="pretending-to-be-elsewhere"))
        )
        assert status == 409
        assert "already live" in body["error"]
        assert len(app.state.livekit_sessions) == 1

        for task in list(app.state.livekit_sessions.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_two_rooms_still_run_side_by_side(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        """The dedupe is per room, not a second single-session guard."""
        _room, app = ecs_app
        assert (
            await start_livekit_session(app, dial_from_body(_dial(token=_token_for("room-a"))))
        )[0] == 200
        assert (
            await start_livekit_session(app, dial_from_body(_dial(token=_token_for("room-b"))))
        )[0] == 200
        assert sorted(app.state.livekit_sessions) == ["room-a", "room-b"]

        for task in list(app.state.livekit_sessions.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_a_dial_to_an_unpinned_url_is_a_403(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With TIMBAL_LIVEKIT_URL set, the body cannot redirect this process at
        an SFU of the caller's choosing — the dial spends this deployment's
        STT/TTS/LLM budget, streaming to whoever is in that room."""
        _room, app = ecs_app
        monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "ws://ours:7880")
        status, body = await start_livekit_session(
            app, dial_from_body(_dial(url="ws://attacker.example:7880"))
        )
        assert status == 403
        assert "pinned" in body["error"]
        # Refused before any state is touched: no session dict, no slot held.
        assert not hasattr(app.state, "livekit_sessions")
        assert capacity.active_sessions() == 0

    async def test_the_pinned_url_still_joins(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _room, app = ecs_app
        monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "ws://fake:7880")
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        for task in list(app.state.livekit_sessions.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_a_join_failure_does_not_echo_internals_to_the_caller(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        """The 502 body goes to whoever posted the dial, so the SFU's hostname
        and error text belong in the log, not the response."""
        room, app = ecs_app

        async def _connect_boom(url: str, token: str) -> None:
            raise RuntimeError("connect failed: sfu-internal.vpc.local:7880 refused")

        room.connect = _connect_boom
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 502
        assert body == {"error": "the agent could not join the room"}
        assert "vpc.local" not in json.dumps(body)

    async def test_missing_extra_is_a_501(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _room, app = ecs_app
        monkeypatch.setitem(sys.modules, "livekit", None)
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 501
        assert "voice-livekit" in body["error"]

    async def test_at_capacity_the_next_dial_is_refused(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full process rejects instead of degrading the calls it is already
        carrying."""
        _room, app = ecs_app
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        capacity.reset_for_tests()

        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        status, body = await start_livekit_session(app, dial_from_body(_dial(room="v1_1_2_3_other")))
        assert status == 503
        assert "capacity" in body["error"]

        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live

    async def test_the_slot_comes_back_when_the_session_ends(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The slot tracks the session, not the request — and a leak here
        would silently shrink the box one call at a time."""
        _room, app = ecs_app
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        capacity.reset_for_tests()

        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        assert capacity.active_sessions() == 1
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live
        assert capacity.active_sessions() == 0
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        again = app.state.livekit_sessions["v1_1_2_3_abc"]
        again.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await again

    async def test_a_rejected_dial_does_not_hold_a_slot(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """400/409 happen before the acquire; a failed join releases on the
        task's done callback."""
        _room, app = ecs_app
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "2")
        capacity.reset_for_tests()

        assert (await start_livekit_session(app, dial_from_body(_dial(token=""))))[0] == 400
        assert capacity.active_sessions() == 0

        monkeypatch.setitem(sys.modules, "livekit", None)
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 501
        await asyncio.sleep(0)
        assert capacity.active_sessions() == 0

    async def test_body_config_reaches_the_session(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No TIMBAL_VOICE_CLIENT_CONFIG on this path — the config rides the POST."""
        room, app = ecs_app
        seen: dict = {}

        def _capture(runnable: object, defaults: object, config: dict, **kwargs: object):
            seen["config"] = config
            raise RuntimeError("stop after capture")

        monkeypatch.setattr("timbal.server.livekit_session.build_voice_session", _capture)
        status, _body = await start_livekit_session(
            app, dial_from_body(_dial(config={"stt_provider": "deepgram-flux"}))
        )
        assert status == 200
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        _deliver_hello(room, {"sample_rate": 16000})
        _subscribe_caller(room)
        await asyncio.wait({live}, timeout=2.0)
        assert seen["config"]["stt_provider"] == "deepgram-flux"

    async def test_sip_bye_before_media_releases_the_slot(
        self,
        ecs_app: tuple[_FakeRoom, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Per-request path: early SIP note + BYE before media must free the
        room key and capacity slot, not sit on caller_ready.wait()."""
        room, app = ecs_app
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        capacity.reset_for_tests()

        sip = SimpleNamespace(
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
            attributes={"sip.callID": "c1"},
            disconnect_reason=SimpleNamespace(name="CLIENT_INITIATED"),
        )
        room.remote_participants[sip.identity] = sip

        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        assert capacity.active_sessions() == 1
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        assert not live.done()

        room.handlers["participant_disconnected"](sip)
        done, _ = await asyncio.wait({live}, timeout=2.0)
        assert live in done
        assert capacity.active_sessions() == 0
        assert "v1_1_2_3_abc" not in app.state.livekit_sessions

    async def test_sip_blip_before_media_aborts_after_short_window(
        self,
        ecs_app: tuple[_FakeRoom, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        room, app = ecs_app
        monkeypatch.setenv("TIMBAL_VOICE_SIP_ABANDON_SECS", "0.05")
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        capacity.reset_for_tests()

        sip = SimpleNamespace(
            identity="+34111",
            kind="PARTICIPANT_KIND_SIP",
            attributes={},
            disconnect_reason=SimpleNamespace(name="STATE_MISMATCH"),
        )
        room.remote_participants[sip.identity] = sip

        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        room.handlers["participant_disconnected"](sip)
        done, _ = await asyncio.wait({live}, timeout=2.0)
        assert live in done
        assert capacity.active_sessions() == 0

    async def test_server_control_packet_publishes_dtmf_to_sip_leg(self, ecs_app: tuple[_FakeRoom, object]) -> None:
        room, app = ecs_app
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        sip = SimpleNamespace(
            identity="sip-caller",
            kind=SimpleNamespace(name="PARTICIPANT_KIND_SIP"),
            attributes={},
        )
        room.handlers["participant_connected"](sip)
        packet = SimpleNamespace(
            data=json.dumps(
                {
                    "type": "timbal.sip.send_dtmf",
                    "digits": "3#",
                    "participant_identity": "sip-caller",
                }
            ).encode(),
            topic="timbal.sip.control",
            participant=None,
        )
        room.handlers["data_received"](packet)
        for _ in range(10):
            if room.local_participant.dtmf_published:
                break
            await asyncio.sleep(0.01)
        assert room.local_participant.dtmf_published == [(3, "3"), (11, "#")]

        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live
