"""Client frames on the WebRTC data channel: ``interaction_answer`` and ``notify``.

``handle_client_message`` is the whole server side of it; the loopback test in
``test_voice_rtc.py`` covers the channel itself.
"""

from __future__ import annotations

import json

import pytest
from timbal.server.rtc import handle_client_message


class _Session:
    def __init__(self, *, own_turn: bool = True, answer_ok: bool = True, fail: bool = False) -> None:
        self.notified: list[tuple[str, str | None, str]] = []
        self.answers: list[tuple[str, str | None, str | None]] = []
        self._own_turn = own_turn
        self._answer_ok = answer_ok
        self._fail = fail

    async def notify(self, text: str, *, key: str | None = None, policy: str = "when_idle") -> bool:
        if self._fail:
            raise RuntimeError("boom")
        self.notified.append((text, key, policy))
        return self._own_turn

    async def submit_user_text(
        self, text: str, *, interaction_id: str | None = None, run_id: str | None = None
    ) -> bool:
        self.answers.append((text, interaction_id, run_id))
        return self._answer_ok


class TestHandleClientMessage:
    @pytest.mark.parametrize("encode", [str, str.encode])
    async def test_notify_is_placed_and_acked(self, encode) -> None:
        session, sent = _Session(own_turn=False), []
        frame = {"type": "notify", "text": "Task 'report' finished.", "key": "task:report", "policy": "now"}
        task = handle_client_message(session, encode(json.dumps(frame)), sent.append)
        assert task is not None
        await task
        assert session.notified == [("Task 'report' finished.", "task:report", "now")]
        assert sent == [{"type": "notify_ack", "key": "task:report", "own_turn": False}]

    async def test_interaction_answer_is_submitted_and_acked(self) -> None:
        session, sent = _Session(), []
        frame = {"type": "interaction_answer", "interaction_id": "i1", "value": "A spreadsheet", "run_id": "r1"}
        await handle_client_message(session, json.dumps(frame), sent.append)
        assert session.answers == [("A spreadsheet", "i1", "r1")]
        assert sent == [{"type": "interaction_answer_ack", "interaction_id": "i1"}]

    async def test_answer_that_started_no_turn_is_not_acked(self) -> None:
        session, sent = _Session(answer_ok=False), []
        frame = {"type": "interaction_answer", "interaction_id": "i1", "value": "x"}
        await handle_client_message(session, json.dumps(frame), sent.append)
        assert sent == []

    async def test_a_failing_session_sends_nothing(self) -> None:
        session, sent = _Session(fail=True), []
        await handle_client_message(session, json.dumps({"type": "notify", "text": "x"}), sent.append)
        assert sent == []

    @pytest.mark.parametrize(
        "message",
        [
            "not json",
            b"\xff\xfe",
            json.dumps(["notify"]),
            json.dumps({"type": "playback", "played_ms": 10}),
            json.dumps({"type": "notify", "text": "   "}),
            json.dumps({"type": "notify", "text": "x", "policy": "eventually"}),
        ],
    )
    def test_anything_else_is_ignored(self, message: str | bytes) -> None:
        session, sent = _Session(), []
        assert handle_client_message(session, message, sent.append) is None
        assert sent == [] and session.notified == [] and session.answers == []
