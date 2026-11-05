"""Entity / spelling patience: hold dictation, ignore Flux eager, commit when finished.

Off by default. These tests never touch a provider socket.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice import (
    CommitAction,
    CommitDecision,
    ProviderTurnDetector,
    TranscriptEvent,
    VoiceConfig,
    VoiceSession,
    VoiceSessionEvent,
)
from timbal.voice.entity_patience import apply_entity_patience, entity_capture

from .test_session import DelayedMockSTT, MockTTS
from .test_turn_detection import _state

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("my number is 415 555", "digits"),
        ("415-555-0199", "digits"),
        ("415.555.1212", "digits"),
        ("the code is 4 1 5", "digits"),
        ("612 34 um", "digits"),
        ("order AB12", "code"),
        ("#X7K", "code"),
        ("it is A 1 B 2", "code"),
        ("spelled J O H", "spelling"),
        ("J. O. H.", "spelling"),
        ("Ñ O L", "spelling"),
        ("ada@examp", "email"),
        ("mail ada@", "email"),
        ("ada@example.com", None),
        ("ada@example.com.", None),
        ("my number is 415 555 1212 thanks", None),
        ("the code is AB12 confirmed", None),
        ("hello there", None),
        ("I have 2", None),
        ("I have 2 cats", None),
        ("o sea", None),
        ("um", None),
        ("415?", None),
        ("", None),
    ],
)
def test_entity_capture(text: str, kind: str | None) -> None:
    assert entity_capture(text) == kind


# ---------------------------------------------------------------------------
# Decision wrapper
# ---------------------------------------------------------------------------


def _new(text: str, reason: str = "provider") -> CommitDecision:
    return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason=reason)


class TestApplyEntityPatience:
    def test_off_is_unchanged(self) -> None:
        decision = _new("my number is 415 555")
        assert apply_entity_patience(decision, decision.text, _state(), enabled=False, hold_secs=2.5) is decision

    def test_open_digits_hold(self) -> None:
        decision = apply_entity_patience(
            _new("my number is 415 555"), "my number is 415 555", _state(), enabled=True, hold_secs=2.5
        )
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "entity_patience"
        assert decision.hold_timeout_secs == 2.5
        assert decision.text == "my number is 415 555"

    def test_followup_digits_stay_held_and_merged(self) -> None:
        decision = apply_entity_patience(
            _new("1212"),
            "1212",
            _state(holding=True, active_user_text="my number is 415 555"),
            enabled=True,
            hold_secs=2.5,
        )
        assert decision.action is CommitAction.HOLD
        assert decision.text == "my number is 415 555 1212"

    def test_finished_entity_commits_the_whole_utterance(self) -> None:
        decision = apply_entity_patience(
            _new("thanks"),
            "thanks",
            _state(holding=True, active_user_text="my number is 415 555 1212"),
            enabled=True,
            hold_secs=2.5,
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "entity_patience_done"
        assert decision.text == "my number is 415 555 1212 thanks"

    def test_complete_email_commits(self) -> None:
        decision = apply_entity_patience(
            _new("ada@example.com"), "ada@example.com", _state(), enabled=True, hold_secs=2.5
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "provider"

    def test_ordinary_speech_unchanged_when_enabled(self) -> None:
        decision = _new("what time is it")
        assert apply_entity_patience(decision, decision.text, _state(), enabled=True, hold_secs=2.5) is decision

    def test_ignore_stays_ignore(self) -> None:
        decision = CommitDecision(action=CommitAction.IGNORE, text="um", reason="hesitation")
        assert apply_entity_patience(decision, "um", _state(), enabled=True, hold_secs=2.5) is decision

    def test_existing_hold_is_kept_and_only_lengthened(self) -> None:
        short = CommitDecision(
            action=CommitAction.HOLD, text="the code is AB12", reason="hold_unfinished", hold_timeout_secs=1.2
        )
        out = apply_entity_patience(short, short.text, _state(), enabled=True, hold_secs=2.5)
        assert out.action is CommitAction.HOLD
        assert out.reason == "hold_unfinished"
        assert out.hold_timeout_secs == 2.5

        longer = CommitDecision(
            action=CommitAction.HOLD, text="the code is AB12", reason="audio_hold", hold_timeout_secs=4.0
        )
        out = apply_entity_patience(longer, longer.text, _state(), enabled=True, hold_secs=2.5)
        assert out.reason == "audio_hold"
        assert out.hold_timeout_secs == 4.0

        prose = CommitDecision(action=CommitAction.HOLD, text="es que", reason="hold_unfinished", hold_timeout_secs=1.2)
        assert apply_entity_patience(prose, prose.text, _state(), enabled=True, hold_secs=2.5) is prose

    def test_continue_of_a_live_turn_is_not_parked(self) -> None:
        decision = CommitDecision(action=CommitAction.CONTINUE_TURN, text="AB12", reason="lexical_continuation")
        out = apply_entity_patience(
            decision, "AB12", _state(assistant_active=True, active_user_text="the order"), enabled=True, hold_secs=2.5
        )
        assert out is decision

    def test_continue_while_holding_stays_a_hold(self) -> None:
        decision = CommitDecision(action=CommitAction.CONTINUE_TURN, text="1212", reason="lexical_continuation")
        out = apply_entity_patience(
            decision,
            "1212",
            _state(holding=True, active_user_text="order 88", assistant_active=True),
            enabled=True,
            hold_secs=2.5,
        )
        assert out.action is CommitAction.HOLD
        assert out.text == "order 88 1212"

    def test_flux_patience_already_live_does_not_hold_again(self) -> None:
        decision = apply_entity_patience(
            _new("my number is 415 555 1212"),
            "my number is 415 555 1212",
            _state(),
            enabled=True,
            hold_secs=2.5,
            provider_patience_live=True,
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "entity_patience_provider_timeout"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class _RecordingSTT(DelayedMockSTT):
    def __init__(self) -> None:
        super().__init__()
        self.eager_log: list[bool] = []
        self.configure_calls: list[dict] = []
        self.patient_thresholds_live = False
        self.eager_suppressed = False

    def __setattr__(self, name: str, value: object) -> None:
        if name == "eager_suppressed" and "eager_log" in self.__dict__:
            self.__dict__["eager_log"].append(bool(value))
        super().__setattr__(name, value)

    async def configure_turn(self, **kwargs: object) -> None:
        # ConfigureSuccess is a later socket message. Flipping the flag here
        # would make the very next commit look like Flux had already waited.
        self.configure_calls.append(dict(kwargs))


def _session(stt: _RecordingSTT, *, patience: bool, secs: float = 2.5) -> VoiceSession:
    agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
    return VoiceSession(
        agent=agent,
        stt=stt,
        tts=MockTTS(),
        turn_detector=ProviderTurnDetector(),
        entity_patience=patience,
        entity_patience_secs=secs,
    )


async def _until(pred, *, timeout: float = 2.0, msg: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {msg}")


class TestEntityPatienceSession:
    async def _run(self, session: VoiceSession) -> list[VoiceSessionEvent]:
        events: list[VoiceSessionEvent] = []

        async def _empty() -> AsyncIterator[bytes]:
            return
            yield  # pragma: no cover

        async def _consume() -> None:
            async for ev in session.run(_empty()):
                events.append(ev)

        task = asyncio.create_task(_consume())
        await _until(lambda: any(getattr(e, "type", None) == "session_started" for e in events), msg="session_started")
        self._task = task
        return events

    async def _finish(self, stt: _RecordingSTT) -> None:
        await stt.finish()
        await self._task

    async def test_dictation_is_held_then_finished_entity_commits(self) -> None:
        stt = _RecordingSTT()
        session = _session(stt, patience=True, secs=2.5)
        events = await self._run(session)
        await stt.inject(TranscriptEvent(type="partial", text="order AB12"))
        await _until(lambda: bool(stt.configure_calls) and stt.eager_suppressed, msg="eager suppressed")
        await stt.inject(TranscriptEvent(type="committed", text="order AB12"))
        await _until(lambda: session._held_user_text == "order AB12", msg="hold")
        assert session.transcript == []
        assert not any(getattr(e, "type", None) == "transcript_committed" for e in events)
        assert stt.configure_calls[0]["patient"] is True
        assert stt.configure_calls[0]["eot_threshold"] == 1.0
        assert stt.eager_log[-1] is True

        await stt.inject(TranscriptEvent(type="committed", text="confirmed"))
        await _until(
            lambda: any(getattr(e, "type", None) == "transcript_committed" for e in events),
            msg="commit",
        )
        committed = next(e for e in events if e.type == "transcript_committed")
        assert committed.text == "order AB12 confirmed"
        assert session.transcript[0].text == "order AB12 confirmed"
        await _until(
            lambda: stt.configure_calls[-1].get("patient") is False and stt.eager_suppressed is False,
            msg="eager restored",
        )
        assert stt.configure_calls[-1]["patient"] is False
        assert stt.configure_calls[-1]["eot_threshold"] == 0.8
        await self._finish(stt)

    async def test_flux_patience_already_live_commits_without_a_second_hold(self) -> None:
        stt = _RecordingSTT()
        stt.patient_thresholds_live = True
        session = _session(stt, patience=True)
        events = await self._run(session)
        await stt.inject(TranscriptEvent(type="committed", text="my number is 415 555 1212"))
        await _until(
            lambda: any(getattr(e, "type", None) == "transcript_committed" for e in events),
            msg="commit",
        )
        assert session._held_user_text is None
        assert session.transcript[0].text == "my number is 415 555 1212"
        await self._finish(stt)

    async def test_off_does_not_hold_or_touch_eager(self) -> None:
        stt = _RecordingSTT()
        session = _session(stt, patience=False)
        events = await self._run(session)
        await stt.inject(TranscriptEvent(type="partial", text="order AB12"))
        await stt.inject(TranscriptEvent(type="committed", text="order AB12"))
        await _until(
            lambda: any(getattr(e, "type", None) == "transcript_committed" for e in events),
            msg="commit",
        )
        assert session._held_user_text is None
        assert session.transcript[0].text == "order AB12"
        assert stt.configure_calls == []
        assert True not in stt.eager_log
        await self._finish(stt)

    async def test_configure_endpointing_while_idle(self) -> None:
        stt = _RecordingSTT()
        session = _session(stt, patience=True)
        await self._run(session)
        await session.configure_endpointing(eot_threshold=0.9, eot_timeout_ms=7000)
        assert stt.configure_calls[-1]["eot_threshold"] == 0.9
        assert stt.configure_calls[-1]["eot_timeout_ms"] == 7000
        assert stt.configure_calls[-1]["patient"] is False
        await self._finish(stt)


def test_voice_config_defaults_off() -> None:
    cfg = VoiceConfig()
    assert cfg.entity_patience is False
    assert cfg.entity_patience_secs == 2.5
