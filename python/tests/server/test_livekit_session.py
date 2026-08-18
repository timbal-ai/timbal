"""LiveKit session driver: env gating, reliable-data chunking, cancellation.

Does not talk to a real SFU — the FFI extra is not required; the cancellation
tests inject a fake ``livekit`` module.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.livekit_session import (
    _is_caller,
    _run_livekit_session,
    chunk_data_payloads,
    dial_from_body,
    is_config_hello,
    is_livekit_dial,
    maybe_start_livekit_session,
    merge_client_config,
    start_livekit_session,
)


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
    def test_empty_prefix_matches_anyone(self) -> None:
        assert _is_caller("laptop", "") is True

    def test_prefix_match(self) -> None:
        assert _is_caller("user-abc", "user") is True
        assert _is_caller("agent", "user") is False


class TestChunkDataPayloads:
    def test_small_payloads_pass_through(self) -> None:
        payloads = [{"type": "interrupted", "heard_text": "hi"}]
        assert chunk_data_payloads(payloads) == payloads

    def test_oversized_transcript_is_split_with_seq_total(self) -> None:
        # Each entry is ~200 bytes; 100 of them blow past 12 KiB.
        entries = [
            {"role": "assistant", "text": "x" * 180, "timestamp": 1.0 + i}
            for i in range(100)
        ]
        payloads = [{"type": "session_transcript", "entries": entries, "started_at": 1.0}]
        chunks = chunk_data_payloads(payloads)
        assert len(chunks) > 1
        assert all(c["type"] == "session_transcript" for c in chunks)
        assert [c["seq"] for c in chunks] == list(range(len(chunks)))
        assert all(c["total"] == len(chunks) for c in chunks)
        assert all(c["started_at"] == 1.0 for c in chunks)
        rebuilt = [e for c in chunks for e in c["entries"]]
        assert rebuilt == entries

    def test_non_transcript_oversized_is_left_intact(self) -> None:
        payload = {"type": "agent_text_done", "text": "y" * 20_000}
        assert chunk_data_payloads([payload]) == [payload]


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
        assert merge_client_config("not-json", {"tts_provider": "elevenlabs"}) == {
            "tts_provider": "elevenlabs"
        }

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
    async def publish_data(self, *args: object, **kwargs: object) -> None:
        pass

    async def publish_track(self, *args: object, **kwargs: object) -> None:
        pass


class _FakeRoom:
    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.disconnect_entered = asyncio.Event()
        self.disconnect_gate: asyncio.Event | None = None
        self.disconnected = False
        self.local_participant = _FakeParticipant()
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
        AudioSource=lambda *a, **k: SimpleNamespace(clear_queue=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=fake_rtc))
    monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "ws://fake:7880")
    monkeypatch.setenv("TIMBAL_LIVEKIT_TOKEN", "tok")
    monkeypatch.delenv("TIMBAL_VOICE_CLIENT_CONFIG", raising=False)
    log = _LogRecorder()
    monkeypatch.setattr("timbal.server.livekit_session.logger", log)
    guard = _FakeGuard()
    agent = Agent(name="lk_test", model=TestModel(responses=["hi"]), tools=[])
    app = SimpleNamespace(
        state=SimpleNamespace(runnable=agent, single_session_guard=guard, voice_config=None)
    )
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


def _subscribe_caller(room: _FakeRoom) -> SimpleNamespace:
    participant = SimpleNamespace(identity="playground")
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


@pytest.fixture
def ecs_app(driver_env: tuple[_FakeRoom, _FakeGuard, _LogRecorder, object]) -> tuple[_FakeRoom, object]:
    """A long-lived server: no single-session guard, no LiveKit boot env."""
    room, _guard, _log, app = driver_env
    app.state.single_session_guard = None
    return room, app


def _dial(**over: object) -> dict:
    body = {
        "transport": "livekit",
        "url": "ws://fake:7880",
        "token": "tok",
        "room": "v1_1_2_3_abc",
        # Matches `_subscribe_caller`'s participant so the mic is treated as
        # the human's.
        "caller_identity": "playground",
    }
    body.update(over)
    return body


class TestStartLivekitSession:
    """Per-request join (ECS / on-premise): the process serves room after room."""

    async def test_join_answers_200_and_leaves_the_session_running(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
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

    async def test_second_join_for_the_same_room_conflicts(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        _room, app = ecs_app
        assert (await start_livekit_session(app, dial_from_body(_dial())))[0] == 200
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 409
        assert "already live" in body["error"]
        live = app.state.livekit_sessions["v1_1_2_3_abc"]
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live

    async def test_a_finished_room_frees_its_key_for_the_next_call(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
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
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        room, app = ecs_app

        async def _connect_boom(url: str, token: str) -> None:
            raise RuntimeError("sfu unreachable")

        room.connect = _connect_boom
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 502
        assert "sfu unreachable" in body["error"]

    async def test_a_hung_join_times_out_and_cancels_the_task(
        self, ecs_app: tuple[_FakeRoom, object]
    ) -> None:
        room, app = ecs_app

        async def _connect_hang(url: str, token: str) -> None:
            await asyncio.Event().wait()

        room.connect = _connect_hang
        status, body = await start_livekit_session(app, dial_from_body(_dial()), timeout=0.05)
        assert status == 504
        assert "did not join" in body["error"]
        live = app.state.livekit_sessions.get("v1_1_2_3_abc")
        assert live is not None and live.cancelled() or live.cancelling()
        with contextlib.suppress(asyncio.CancelledError):
            await live
        assert "v1_1_2_3_abc" not in app.state.livekit_sessions

    async def test_missing_extra_is_a_501(
        self, ecs_app: tuple[_FakeRoom, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _room, app = ecs_app
        monkeypatch.setitem(sys.modules, "livekit", None)
        status, body = await start_livekit_session(app, dial_from_body(_dial()))
        assert status == 501
        assert "voice-livekit" in body["error"]

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
