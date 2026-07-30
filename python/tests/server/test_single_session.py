"""Tests for ``TIMBAL_VOICE_SINGLE_SESSION`` — one voice session per process, then exit.

``_exit_process`` is monkeypatched everywhere (the real thing is
``os._exit(0)``); a finished guard therefore stays around, which is exactly
what the refuse-after-served assertions need.

The WebSocket transport exercises the full claim → connect → finish flow
without the aiortc dependency; the WebRTC side of the guard is covered in
``test_voice_rtc.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from timbal.server.http import create_app
from timbal.server.single_session import SingleSessionGuard, init_single_session_guard
from timbal.voice import TranscriptEvent

from .test_voice_ws import _collect_ws_messages, _make_stt_class, _make_tts_class, _write_agent_module
from .voice_env import VOICE_ENV_KEYS


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record exit codes instead of killing the test process."""
    calls: list[int] = []
    monkeypatch.setattr("timbal.server.single_session._exit_process", calls.append)
    return calls


class TestSingleSessionGuard:
    async def test_claim_is_exclusive_until_released(self, exits: list[int]) -> None:
        guard = SingleSessionGuard(idle_exit_secs=60.0)
        assert guard.claim()
        assert not guard.claim()
        guard.release()
        assert guard.claim()
        assert exits == []

    async def test_finish_exits_zero_and_refuses_forever(self, exits: list[int]) -> None:
        guard = SingleSessionGuard(idle_exit_secs=60.0)
        assert guard.claim()
        await guard.finish()
        assert exits == [0]
        assert not guard.claim()
        # A release after finish must not reopen the slot.
        guard.release()
        assert not guard.claim()
        # Idempotent: a second finish doesn't exit twice.
        await guard.finish()
        assert exits == [0]

    async def test_idle_timer_exits_when_nothing_connects(self, exits: list[int]) -> None:
        guard = SingleSessionGuard(idle_exit_secs=0.05)
        guard.start()
        await asyncio.sleep(0.2)
        assert exits == [0]

    async def test_idle_timer_fires_on_claimed_but_never_connected(self, exits: list[int]) -> None:
        """An offer whose ICE never completes must not keep the box alive."""
        guard = SingleSessionGuard(idle_exit_secs=0.05)
        guard.start()
        assert guard.claim()
        await asyncio.sleep(0.2)
        assert exits == [0]

    async def test_idle_exit_waits_for_recording_uploads(self, exits: list[int]) -> None:
        """The idle path must exit through finish(): never race an in-flight PUT."""
        from timbal.server import recording_upload

        uploaded = asyncio.Event()

        async def _slow_upload() -> None:
            await asyncio.sleep(0.1)
            uploaded.set()

        task = asyncio.create_task(_slow_upload())
        recording_upload._upload_tasks.add(task)
        task.add_done_callback(recording_upload._upload_tasks.discard)

        guard = SingleSessionGuard(idle_exit_secs=0.05)
        guard.start()
        await asyncio.sleep(0.4)
        assert uploaded.is_set()
        assert exits == [0]

    async def test_mark_connected_disarms_the_idle_timer(self, exits: list[int]) -> None:
        guard = SingleSessionGuard(idle_exit_secs=0.05)
        guard.start()
        guard.mark_connected()
        await asyncio.sleep(0.2)
        assert exits == []

    async def test_idle_exit_aborts_if_connected_at_deadline(self, exits: list[int]) -> None:
        """Media establishing in the same turn as the timer must not kill the call.

        After sleep wakes, _idle_exit yields once so a queued mark_connected()
        can run before exit; clearing _idle_task must not strand a late
        connect with an inevitable exit.
        """
        guard = SingleSessionGuard(idle_exit_secs=0.05)
        guard.start()
        assert guard.claim()

        async def _connect_at_deadline() -> None:
            await asyncio.sleep(0.05)
            guard.mark_connected()

        asyncio.create_task(_connect_at_deadline())
        await asyncio.sleep(0.3)
        assert exits == []
        assert guard._connected
        assert not guard._finished

    async def test_idle_exit_aborts_if_connected_during_drain(
        self, exits: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mark_connected during the idle upload drain must not exit the box."""
        guard = SingleSessionGuard(idle_exit_secs=0.05)

        async def _drain_and_connect() -> None:
            guard.mark_connected()

        monkeypatch.setattr(
            "timbal.server.recording_upload.drain_upload_tasks",
            _drain_and_connect,
        )
        guard.start()
        assert guard.claim()
        await asyncio.sleep(0.3)
        assert exits == []
        assert guard._connected
        assert not guard._finished

    async def test_finish_waits_for_recording_uploads(self, exits: list[int]) -> None:
        """The exit must not race the platform PUT — the process holds the data."""
        from timbal.server import recording_upload

        uploaded = asyncio.Event()

        async def _slow_upload() -> None:
            await asyncio.sleep(0.05)
            uploaded.set()

        task = asyncio.create_task(_slow_upload())
        recording_upload._upload_tasks.add(task)
        task.add_done_callback(recording_upload._upload_tasks.discard)

        guard = SingleSessionGuard(idle_exit_secs=60.0)
        assert guard.claim()
        await guard.finish()
        assert uploaded.is_set()
        assert exits == [0]


class TestInitSingleSessionGuard:
    async def test_disabled_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TIMBAL_VOICE_SINGLE_SESSION", raising=False)
        assert init_single_session_guard() is None

    async def test_enabled_with_default_idle_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
        monkeypatch.delenv("TIMBAL_VOICE_IDLE_EXIT_SECS", raising=False)
        guard = init_single_session_guard()
        assert guard is not None
        assert guard.idle_exit_secs == 60.0
        guard.shutdown()

    async def test_idle_window_from_env_with_bad_value_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
        monkeypatch.setenv("TIMBAL_VOICE_IDLE_EXIT_SECS", "5")
        guard = init_single_session_guard()
        assert guard is not None and guard.idle_exit_secs == 5.0
        guard.shutdown()

        monkeypatch.setenv("TIMBAL_VOICE_IDLE_EXIT_SECS", "not-a-number")
        guard = init_single_session_guard()
        assert guard is not None and guard.idle_exit_secs == 60.0
        guard.shutdown()


def _setup_ws_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = _write_agent_module(tmp_path, responses=["Hi there!"])
    monkeypatch.setenv("TIMBAL_RUNNABLE", spec)
    for k in VOICE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TIMBAL_VOICE_SINGLE_SESSION", "1")
    monkeypatch.setattr(
        "timbal.voice.elevenlabs.ElevenLabsRealtimeSTT",
        _make_stt_class([TranscriptEvent(type="committed", text="Hello")]),
    )
    monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())


class TestSingleSessionOverWebSocket:
    def test_one_session_then_exit_then_refuse(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exits: list[int]
    ) -> None:
        _setup_ws_env(monkeypatch, tmp_path)
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                messages = _collect_ws_messages(ws)
            assert messages[-1]["type"] == "session_ended"

            # The handler's finally runs after the socket closes; poll briefly.
            deadline = time.monotonic() + 5.0
            while not exits and time.monotonic() < deadline:
                time.sleep(0.02)
            assert exits == [0]

            # One process = one session: a second connect is refused.
            with client.websocket_connect("/voice/ws") as ws2:
                with pytest.raises(WebSocketDisconnect) as excinfo:
                    ws2.receive_json()
            assert excinfo.value.code == 1008

    def test_session_build_failure_still_exits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exits: list[int]
    ) -> None:
        """An exception between claim() and the session's own teardown must not
        leave the process alive: the idle timer is already disarmed by then."""
        _setup_ws_env(monkeypatch, tmp_path)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("recorder misconfigured")

        monkeypatch.setattr("timbal.server.voice.build_voice_session", _boom)

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            with contextlib.suppress(Exception), client.websocket_connect("/voice/ws") as ws:
                ws.send_json({})
                ws.receive_json()
            deadline = time.monotonic() + 5.0
            while not exits and time.monotonic() < deadline:
                time.sleep(0.02)
        assert exits == [0]

    def test_idle_exit_when_nobody_connects(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exits: list[int]
    ) -> None:
        _setup_ws_env(monkeypatch, tmp_path)
        monkeypatch.setenv("TIMBAL_VOICE_IDLE_EXIT_SECS", "0.1")
        app = create_app()
        with TestClient(app):
            deadline = time.monotonic() + 5.0
            while not exits and time.monotonic() < deadline:
                time.sleep(0.02)
        assert exits == [0]

    def test_no_single_session_env_means_no_lifetime_management(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exits: list[int]
    ) -> None:
        _setup_ws_env(monkeypatch, tmp_path)
        monkeypatch.delenv("TIMBAL_VOICE_SINGLE_SESSION")
        app = create_app()
        with TestClient(app) as client:
            for _ in range(2):  # sequential sessions stay allowed
                with client.websocket_connect("/voice/ws") as ws:
                    ws.send_json({})
                    messages = _collect_ws_messages(ws)
                assert messages[-1]["type"] == "session_ended"
        assert exits == []
