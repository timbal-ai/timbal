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
    is_config_hello,
    maybe_start_livekit_session,
    merge_client_config,
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

    def on(self, name: str, fn: object) -> None:
        pass

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

    def claim(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True

    def mark_connected(self) -> None:
        pass

    def mark_reconnected(self) -> None:
        pass

    def mark_disconnected(self, **kwargs: object) -> None:
        pass

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
