"""``notify`` frames: parsing, acks, the WebRTC data channel, and who may enable them."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server import voice as voice_routes
from timbal.server.livekit_session import deliver_notify, notify_ack, parse_notify
from timbal.server.rtc import handle_client_message
from timbal.state.background import MAX_SESSION_EVENT_CHARS, BackgroundTaskStore
from timbal.voice import VoiceConfig, VoiceEventsConfig


class _Session:
    """Stands in for VoiceSession: records what the transport asked of it."""

    def __init__(self, *, client_events: bool = True, raises: BaseException | None = None) -> None:
        self.client_events = client_events
        self.raises = raises
        self.store = BackgroundTaskStore()
        self.notified: list[tuple[str, str | None, str]] = []
        self.announced: list[str | None] = []
        self.taps: list[tuple[str, str, str | None]] = []

    async def notify(self, text: str, *, key: str | None = None, source: str = "app", announce: Any = None) -> Any:
        self.notified.append((text, key, source))
        self.announced.append(announce)
        if self.raises is not None:
            raise self.raises
        return self.store.post_event(text, key=key, source=source, announce=announce)

    async def submit_user_text(self, text: str, *, interaction_id: str, run_id: str | None) -> bool:
        self.taps.append((text, interaction_id, run_id))
        return True


class TestParse:
    def test_text_optional_key_and_announce(self) -> None:
        assert parse_notify({"type": "notify", "text": "  Task finished.  "}) == ("Task finished.", None, None)
        assert parse_notify({"type": "notify", "text": "x", "key": " task:1 ", "policy": "when_idle"}) == (
            "x",
            "task:1",
            "when_idle",
        )
        assert parse_notify({"type": "notify", "text": "x", "announce": "silent"}) == ("x", None, "silent")

    @pytest.mark.parametrize(
        "frame",
        [
            {"type": "interaction_answer", "text": "x"},
            {"type": "notify"},
            {"type": "notify", "text": "   "},
            {"type": "notify", "text": 42},
            {"type": "notify", "text": "x", "key": 7},
            {"type": "notify", "text": "x", "policy": "now"},
            {"type": "notify", "text": "x", "announce": "later"},
            {"type": "notify", "text": "x" * (MAX_SESSION_EVENT_CHARS + 1)},
        ],
    )
    def test_rejects(self, frame: dict) -> None:
        assert parse_notify(frame) is None


class TestDeliver:
    async def test_receipt_on_success(self) -> None:
        session = _Session()
        ack = await deliver_notify(session, "Task finished.", "task:1", source="client")
        [event] = session.store.pending_events()
        assert ack == {"type": "notify_ack", "event_id": event.id, "key": "task:1", "state": "pending", "run_id": None}
        assert session.notified == [("Task finished.", "task:1", "client")]

    async def test_client_frames_need_the_session_to_accept_them(self) -> None:
        session = _Session(client_events=False)
        assert await deliver_notify(session, "x", "k", source="client") == notify_ack("k", reason="not_enabled")
        assert session.notified == []
        assert (await deliver_notify(session, "x", None, source="app"))["state"] == "pending"

    async def test_only_the_backend_may_interrupt(self) -> None:
        session = _Session()
        ack = await deliver_notify(session, "x", "k", source="client", announce="interrupt")
        assert ack == notify_ack("k", reason="not_allowed")
        assert (await deliver_notify(session, "y", None, source="app", announce="interrupt"))["state"] == "pending"
        assert session.announced == ["interrupt"]

    @pytest.mark.parametrize(
        ("error", "reason"),
        [(PermissionError("guardrail"), "blocked"), (RuntimeError("ended"), "closed"), (ValueError("x"), "invalid")],
    )
    async def test_rejections(self, error: BaseException, reason: str) -> None:
        ack = await deliver_notify(_Session(raises=error), "x", "k", source="client")
        assert ack == {"type": "notify_ack", "key": "k", "state": "rejected", "reason": reason}


class TestWebRTCDataChannel:
    async def test_notify_frame_is_acked_with_its_receipt(self) -> None:
        session = _Session()
        sent: list[dict] = []
        task = handle_client_message(session, json.dumps({"type": "notify", "text": "Done.", "key": "a"}), sent.append)
        await asyncio.wait_for(task, 1)
        assert session.notified == [("Done.", "a", "client")]
        assert sent[0]["type"] == "notify_ack" and sent[0]["state"] == "pending"

    async def test_interaction_answer_is_submitted_and_acked(self) -> None:
        session = _Session()
        sent: list[dict] = []
        frame = {"type": "interaction_answer", "interaction_id": "i1", "value": "Yes", "run_id": "r1"}
        await asyncio.wait_for(handle_client_message(session, json.dumps(frame), sent.append), 1)
        assert session.taps == [("Yes", "i1", "r1")]
        assert sent == [{"type": "interaction_answer_ack", "interaction_id": "i1"}]

    @pytest.mark.parametrize("message", ["not json", "[]", json.dumps({"type": "playback", "played_ms": 3})])
    def test_other_frames_are_ignored(self, message: str) -> None:
        assert handle_client_message(_Session(), message, lambda _payload: None) is None

    async def test_invalid_notify_frame_is_rejected_with_invalid_reason(self) -> None:
        session = _Session()
        sent: list[dict] = []
        task = handle_client_message(session, json.dumps({"type": "notify", "text": "   ", "key": "k"}), sent.append)
        assert task is not None
        await asyncio.wait_for(task, 1)
        assert sent == [{"type": "notify_ack", "key": "k", "state": "rejected", "reason": "invalid"}]


class TestWhoEnablesEvents:
    @staticmethod
    def _agent() -> Agent:
        return Agent(name="voice_test", model=TestModel(responses=["hi"]), tools=[])

    def test_off_by_default(self) -> None:
        session, _ = voice_routes.build_voice_session(self._agent(), VoiceConfig(turn_detector="heuristic"), {})
        assert session.client_events is False
        assert session.announce_background_results is False

    def test_server_config_turns_them_on(self) -> None:
        defaults = VoiceConfig(
            turn_detector="heuristic",
            events=VoiceEventsConfig(announce_background_results=True, client_notify=True),
        )
        session, _ = voice_routes.build_voice_session(self._agent(), defaults, {})
        assert session.client_events is True
        assert session.announce_background_results is True

    def test_the_hello_cannot(self) -> None:
        hello = {"events": {"client_notify": True, "announce_background_results": True}}
        session, _ = voice_routes.build_voice_session(self._agent(), VoiceConfig(turn_detector="heuristic"), hello)
        assert session.client_events is False
        assert session.announce_background_results is False

    def test_an_empty_block_is_unset(self) -> None:
        assert VoiceConfig(events={}).events is None

    def test_server_config_sets_how_other_turns_are_admitted(self) -> None:
        defaults = VoiceConfig(turn_detector="heuristic", events=VoiceEventsConfig(on_busy="enqueue"))
        session, _ = voice_routes.build_voice_session(self._agent(), defaults, {})
        assert session._scheduler.on_busy == "enqueue"


def test_event_turn_started_reaches_the_client() -> None:
    from timbal.voice import EventTurnStarted

    payloads = voice_routes.event_to_payloads(
        EventTurnStarted(task_ids=["t1"], event_ids=["e1"], keys=["k"], urgent=True), None, {}
    )
    assert payloads == [
        {"type": "event_turn_started", "task_ids": ["t1"], "event_ids": ["e1"], "keys": ["k"], "urgent": True}
    ]
