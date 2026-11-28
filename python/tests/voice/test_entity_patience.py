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
from timbal.voice.entity_patience import (
    apply_entity_patience,
    entity_capture,
    entity_capture_partial,
    entity_capture_with_held,
    repeats_held,
)

from .test_session import DelayedMockSTT, MockTTS
from .test_turn_detection import _state

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("my number is 415 555", "digits"),
        ("four eight", "digits"),
        ("nine zero", "digits"),
        ("The order number is four eight", "digits"),
        ("seis cinco cinco", "digits"),
        ("Mi teléfono es seis cinco cinco", "digits"),
        ("cero uno dos", "digits"),
        ("tres, cuatro", "digits"),
        ("Me apellido jota", "spelling"),
        ("me apellido J", "spelling"),
        ("apellido J.", "spelling"),
        ("ene e ese", "spelling"),
        ("jota o", "spelling"),
        ("aitch jay", "spelling"),
        ("ese ele", "spelling"),
        ("ese, ese", "spelling"),
        ("double u", "spelling"),
        ("my name is double u", "spelling"),
        ("four double five", "digits"),
        ("Dame ese.", None),
        ("Quiero ese", None),
        ("My name is Jay.", None),
        ("So do I.", None),
        ("Make it a double.", None),
        ("No entiendo ni jota.", None),
        ("Ni jota.", None),
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
        ("I need one", None),
        ("de", None),
        ("en", None),
        ("o", None),
        ("I have 2 cats", None),
        ("o sea", None),
        ("um", None),
        ("415?", None),
        ("", None),
    ],
)
def test_entity_capture(text: str, kind: str | None) -> None:
    assert entity_capture(text) == kind


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        # A word still being captioned, from live Flux partials.
        ("Hi. What's the status of order t", None),
        ("Change the delivery note for ten o", None),
        ("Vámonos, v", None),
        # Dictation still opens patience before the commit.
        ("R U I", "spelling"),
        ("Mi apellido es Ruiz, R, U", "spelling"),
        ("apellido J.", "spelling"),
        ("415 555", "digits"),
        ("A B C one two", "code"),
    ],
)
def test_entity_capture_partial(text: str, kind: str | None) -> None:
    assert entity_capture_partial(text) == kind


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

    def test_flux_patience_live_starts_a_followup_of_a_held_fragment(self) -> None:
        """Flux waited out its patient silence for the follow-up too: not another local hold on top."""
        decision = apply_entity_patience(
            _new("seis siete ocho"),
            "seis siete ocho",
            _state(holding=True, active_user_text="mi telefono es 612 345"),
            enabled=True,
            hold_secs=2.5,
            provider_patience_live=True,
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "entity_patience_provider_timeout"
        assert decision.text == "mi telefono es 612 345 seis siete ocho"

    def test_a_detector_hold_stays_a_hold_with_flux_patience_live(self) -> None:
        held = CommitDecision(action=CommitAction.HOLD, text="415 555", reason="detector", hold_timeout_secs=1.0)
        decision = apply_entity_patience(
            held,
            "415 555",
            _state(holding=True, active_user_text="my number is"),
            enabled=True,
            hold_secs=2.5,
            provider_patience_live=True,
        )
        assert decision.action is CommitAction.HOLD

    @pytest.mark.parametrize(("waited", "expected"), [(0.0, 2.5), (1.0, 1.5), (3.0, 0.0)])
    def test_silence_already_heard_counts_against_the_budget(self, waited: float, expected: float) -> None:
        decision = apply_entity_patience(
            _new("my number is 415 555 1212"),
            "my number is 415 555 1212",
            _state(),
            enabled=True,
            hold_secs=2.5,
            waited_secs=waited,
        )
        assert decision.action is CommitAction.HOLD
        assert decision.hold_timeout_secs == pytest.approx(expected)

    def test_detector_hold_keeps_its_own_timeout_after_the_budget_is_spent(self) -> None:
        detector = CommitDecision(
            action=CommitAction.HOLD, text="es 52 87 13 46 g", reason="audio_hold_text_complete", hold_timeout_secs=1.2
        )
        decision = apply_entity_patience(
            detector, "es 52 87 13 46 g", _state(), enabled=True, hold_secs=2.5, waited_secs=2.6
        )
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "audio_hold_text_complete"
        assert decision.hold_timeout_secs == pytest.approx(1.2)


class TestHeldFragmentHelpers:
    @pytest.mark.parametrize("partial", ["seis", "Z", "1", "678"])
    def test_a_short_group_after_a_held_number_is_still_the_entity(self, partial: str) -> None:
        assert entity_capture_with_held("mi telefono es 612 345", partial) is not None

    @pytest.mark.parametrize(("held", "partial"), [("my number is", "y"), ("quiero hablar con", "J")])
    def test_a_held_phrase_that_is_not_an_entity_is_not_continued(self, held: str, partial: str) -> None:
        assert entity_capture_with_held(held, partial) == entity_capture(partial)

    def test_without_a_hold_the_partial_is_read_alone(self) -> None:
        assert entity_capture_with_held(None, "seis") == entity_capture("seis")
        assert entity_capture_with_held("", "my number is 415 555") == entity_capture("my number is 415 555")

    @pytest.mark.parametrize(
        ("held", "partial", "repeat"),
        [
            ("mi telefono es 612 345", "mi telefono es 612 345", True),
            ("mi telefono es 612 345", "Mi telefono es  612", True),  # a refinement of the same caption
            ("mi telefono es 612 345", "678", False),  # Flux's next turn: shorter, and new
            ("mi telefono es 612 345", "mi telefono es 612 345 678", False),
            ("mi telefono es 612 345", "", True),
            ("mi telefono es 612 345", "Mi teléfono es 612 345", True),
            ("mi telefono es 612 345", "mi telefono es 612, 345.", True),
            ("mi telefono es 612 345", "mi telefono es 612345", True),
        ],
    )
    def test_repeats_held(self, held: str, partial: str, repeat: bool) -> None:
        assert repeats_held(held, partial) is repeat


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
        await asyncio.wait_for(self._task, timeout=30)

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

    @pytest.mark.parametrize("silent_secs", [2.5, 0.0])
    async def test_patience_counts_silence_already_waited(self, silent_secs: float) -> None:
        """One budget from the caller's last speech, however late the commit lands.

        Nova + local VAD committed a dictated DNI after the endpointer's 2.5 s
        floor (or the stale-partial rescue) and then held another 2.5 s: ~5 s
        of silence live. A commit right after speech still waits the window.
        """
        last_speech = time.monotonic() - silent_secs

        class _Endpointer:
            patience_floor_secs: float | None = None
            last_predictive = False

            def notify_committed(self) -> None:
                pass

            def last_speech_at(self) -> float:
                return last_speech

            def speech_secs_in_window(self, _window_secs: float) -> None:
                return None

        stt = _RecordingSTT()
        session = _session(stt, patience=True)
        events = await self._run(session)
        session._endpointer = _Endpointer()
        await stt.inject(TranscriptEvent(type="committed", text="my number is 415 555 1212"))
        if silent_secs:
            await _until(
                lambda: any(getattr(e, "type", None) == "transcript_committed" for e in events),
                msg="commit",
            )
            assert session._held_user_text is None
        else:
            await _until(lambda: session._held_user_text is not None, msg="hold")
            assert not any(getattr(e, "type", None) == "transcript_committed" for e in events)
        session._endpointer = None
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


class TestHoldKeepsFluxPatient(TestEntityPatienceSession):
    async def test_a_one_word_group_during_a_hold_does_not_flap_flux_patience(self) -> None:
        stt = _RecordingSTT()
        session = _session(stt, patience=True, secs=2.5)
        await self._run(session)
        await stt.inject(TranscriptEvent(type="partial", text="mi telefono es 612 345"))
        await _until(lambda: stt.eager_suppressed, msg="patient")
        await stt.inject(TranscriptEvent(type="committed", text="mi telefono es 612 345"))
        await _until(lambda: session._held_user_text is not None, msg="hold")
        sent = len(stt.configure_calls)
        for text in ("seis", "seis siete", "Z"):
            await stt.inject(TranscriptEvent(type="partial", text=text))
            await _until(lambda t=text: session._latest_partial_text == t, msg=text)
            await session._sync_entity_patience(text)  # settle this partial's sync before reading
            assert stt.eager_suppressed, text
        assert all(call["patient"] for call in stt.configure_calls[sent:])
        await self._finish(stt)


def test_voice_config_defaults_off() -> None:
    cfg = VoiceConfig()
    assert cfg.entity_patience is False
    assert cfg.entity_patience_secs == 2.5
