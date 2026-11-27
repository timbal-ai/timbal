"""TurnResumed takes back what the eager event did to a buffer, and nothing a confirmed request said.

From the review of timbal-ai/timbal#323 on 1931da8e.
"""

from __future__ import annotations

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import TranscriptEvent
from timbal.voice.config import VoiceConfig

from .test_predictive_eou import _stub_tts
from .test_review_regressions import _flux_session, _protect, _until


def _resumed() -> TranscriptEvent:
    return TranscriptEvent(type="resumed", text="", confidence=0.1)


def _eager(text: str) -> TranscriptEvent:
    return TranscriptEvent(type="eager", text=text, confidence=0.9)


def _recording_session(monkeypatch: pytest.MonkeyPatch, got: list[str]):
    """A Flux session whose ``record`` tool gets what the caller said after "Record"."""
    from timbal.server.voice import build_voice_session

    def record(value: str) -> str:
        got.append(value)
        return "recorded"

    def reply(messages: list) -> Message | str:
        last = messages[-1]
        if last.role == "user":
            value = last.collect_text().split("Record", 1)[-1].strip(" .")
            return Message(
                role="assistant",
                content=[ToolUseContent(id="call-1", name="record", input={"value": value})],
                stop_reason="tool_use",
            )
        return "Done."

    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    agent = Agent(name="voice_test", model=TestModel(handler=reply), tools=[record])
    session, _ = build_voice_session(agent, VoiceConfig(predict_during_speech=True), {})
    _stub_tts(session)
    return session


class TestACorrectionOfAConfirmedValue:
    """Said with the same words, different values: the retraction must give the confirmed one back."""

    @pytest.mark.parametrize(
        ("confirmed", "correction", "value"),
        [("Record 5 + 3", "Record 5 - 3", "5 + 3"), ("Record 1250", "Record 1.250", "1250")],
    )
    async def test_the_tool_gets_the_confirmed_value(
        self, monkeypatch: pytest.MonkeyPatch, confirmed: str, correction: str, value: str
    ) -> None:
        got: list[str] = []
        session = _recording_session(monkeypatch, got)
        _protect(session)
        await session._handle_provider_commit(confirmed)
        await session._handle_provider_prediction(_eager(correction))
        assert session._deferred_user_text == correction
        await session._handle_provider_prediction(_resumed())
        assert session._deferred_user_text == confirmed
        session._turn_tool_batch_inflight = 0  # the protected tool ends: the confirmed request runs
        await _until(lambda: got)
        assert got == [value]


class TestARefinementThatRewritesAConfirmedWord:
    @pytest.mark.parametrize(
        ("confirmed", "refinement"),
        [
            ("Check the booking.", "Check the bookings today."),
            ("Book a table for two.", "Book a table for 2 people."),
            ("Book a table for two.", "Book a table for two, at noon."),
        ],
    )
    async def test_the_deferred_request_comes_back_as_written(
        self, monkeypatch: pytest.MonkeyPatch, confirmed: str, refinement: str
    ) -> None:
        session = _flux_session(monkeypatch, [])
        _protect(session)
        await session._handle_provider_commit(confirmed)
        await session._handle_provider_prediction(_eager(refinement))
        assert session._deferred_user_text == refinement  # the refinement replaced it
        await session._handle_provider_prediction(_resumed())
        assert session._deferred_user_text == confirmed

    async def test_the_queued_turn_gives_the_request_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        session.hold_timeout_secs = 0.1
        _protect(session)
        await session._handle_provider_commit("Check the booking.")
        await session._handle_provider_prediction(_eager("Check the bookings today."))
        session._turn_tool_batch_inflight = 0  # the queued turn starts on the refinement
        await _until(lambda: session._eager_turn_task is not None)
        assert session._eager_prefix == "Check the booking."
        await session._handle_provider_prediction(_resumed())
        await _until(lambda: ran)
        assert [e.text for e in session.transcript if e.role == "user"][-1] == "Check the booking."

    async def test_a_held_request_comes_back_as_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.voice.turn_detection import LexicalTurnDetector

        session = _flux_session(monkeypatch, [])
        session.turn_detector = LexicalTurnDetector()
        session.hold_timeout_secs = 5.0
        await session._handle_provider_commit("I want to check the booking for")
        await session._handle_provider_prediction(_eager("I want to check the bookings for the"))
        await session._handle_provider_prediction(_resumed())
        assert session._held_user_text == "I want to check the booking for"

    async def test_a_held_number_comes_back_as_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _flux_session(monkeypatch, [], entity_patience=True, entity_patience_secs=5.0)
        await session._handle_provider_commit("My booking code is 415 555")
        await session._handle_provider_prediction(_eager("My booking code is 415-555-12"))
        await session._handle_provider_prediction(_resumed())
        assert session._held_user_text == "My booking code is 415 555"


class TestTheBufferRewrittenSinceTheRefinement:
    @pytest.mark.parametrize(
        ("current", "left"),
        [
            ("check the bookings today, send a receipt", "Check the booking. send a receipt"),
            ("Check the bookings today! Send a receipt.", "Check the booking. Send a receipt."),
        ],
    )
    def test_the_confirmed_request_is_put_back_in_its_place(self, current: str, left: str) -> None:
        from timbal.voice.session import _eager_addition, _without_addition

        addition = _eager_addition("Check the booking.", "Check the bookings today.", "Check the bookings today.")
        assert addition is not None
        assert _without_addition(current, addition) == left


class TestOnlyTheEagerOccurrenceGoes:
    async def test_the_eager_word_not_the_typed_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _flux_session(monkeypatch, [])
        _protect(session)
        await session._handle_provider_prediction(_eager("today"))
        session.started_at = 1.0
        await session.submit_user_text("Send a receipt today.")
        assert session._deferred_user_text == "today Send a receipt today."
        await session._handle_provider_prediction(_resumed())
        assert session._deferred_user_text == "Send a receipt today."

    async def test_a_sentence_the_refinement_closed_stays_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        session.hold_timeout_secs = 0.1
        _protect(session)
        await session._handle_provider_commit("Check the booking")
        await session._handle_provider_prediction(_eager("Check the booking today."))
        session.started_at = 1.0
        await session.submit_user_text("Send a receipt.")
        session._turn_tool_batch_inflight = 0
        await _until(lambda: session._eager_turn_task is not None)
        await session._handle_provider_prediction(_resumed())
        await _until(lambda: ran)
        assert [e.text for e in session.transcript if e.role == "user"][-1] == "Check the booking. Send a receipt."


class TestTwoEagerEvents:
    async def test_the_prefix_keeps_neither_unconfirmed_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _flux_session(monkeypatch, [], entity_patience=True, entity_patience_secs=5.0)
        await session._handle_provider_prediction(_eager("My booking code is 415 555"))
        assert session._held_user_text == "My booking code is 415 555"
        await session._handle_provider_prediction(_eager("My booking code is 415 555 1212, that's all."))
        await _until(lambda: session._eager_turn_task is not None)
        prefix = session._eager_prefix
        await session._handle_provider_prediction(_resumed())
        assert (prefix, session._held_user_text) == ("", None)
