"""Opt-in backchannel suppression and unfinished-commit holds.

Accept criteria from the voice roadmap:

* pure EN/ES acknowledgements do not interrupt TTS, on the partial or the commit
* "no", "wait", "hold on", "stop", "actually", "espera", "para", "oh, wait",
  and "sí, pero…" still interrupt immediately
* with ``hold_unfinished_secs=1.2``, "…es que", "Necesito, eh,", and "because"
  are held and merged with the follow-up

Both knobs default off. Unfinished holds stay on the provider detector and do
not fire while the assistant is speaking, so a later resumed-merge check still
sees those commits as new turns.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice import VoiceSession
from timbal.voice.config import VoiceConfig
from timbal.voice.turn_detection import (
    _BACKCHANNEL_GRACE_SECS,
    CommitAction,
    HeuristicTurnDetector,
    LexicalTurnDetector,
    PartialDecision,
    ProviderTurnDetector,
    TurnState,
)

from .test_session import MockSTT, MockTTS


def _state(**overrides) -> TurnState:
    defaults = dict(
        assistant_active=False,
        audio_playing=False,
        assistant_text="Let me check that for you.",
        active_user_text="",
        seconds_since_turn_start=4.0,
        seconds_since_last_commit=4.0,
        seconds_since_assistant_active=None,
        partials_since_last_commit=2,
        holding=False,
        mic_speech_since_last_commit=True,
    )
    defaults.update(overrides)
    return TurnState(**defaults)


def _playing(**overrides) -> TurnState:
    return _state(assistant_active=True, audio_playing=True, seconds_since_assistant_active=None, **overrides)


_BACKCHANNELS = (
    "mhmm",
    "Mhmm.",
    "uh-huh",
    "UH-HUH",
    "uh huh",
    "yeah",
    "Yeah!",
    "yep",
    "yup",
    "okay",
    "OKAY",
    "ok",
    "right",
    "got it",
    "Got it!",
    "gotcha",
    "vale",
    "ajá",
    "Ajá.",
    "aja",
    "claro",
    "sí",
    "Sí.",
    "si",
    "yeah, okay",
    "vale, claro",
    "yeah yeah yeah",
)

_INTERRUPTS = (
    "no",
    "No.",
    "wait",
    "hold on",
    "Hold on.",
    "stop",
    "Stop!",
    "actually",
    "espera",
    "¡Espera!",
    "para",
    "¡para!",
    "oh, wait",
    "oh wait",
    "OH, WAIT",
    "sí, pero…",
    "Sí, pero quiero otra",
    "si, pero",
)


class TestBackchannelsOffByDefault:
    async def test_acknowledgement_still_takes_a_turn(self) -> None:
        det = ProviderTurnDetector()
        decision = await det.on_committed("Yeah.", _playing())
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "provider"

    async def test_three_yeahs_still_barge_in(self) -> None:
        det = ProviderTurnDetector()
        decision = await det.on_partial("yeah yeah yeah", _playing())
        assert decision is PartialDecision.BARGE_IN

    async def test_short_stop_partial_stays_behind_the_word_gate(self) -> None:
        """The min-words gate is unchanged until the knob is on."""
        for det in (ProviderTurnDetector(), HeuristicTurnDetector()):
            decision = await det.on_partial("Stop.", _playing())
            assert decision is PartialDecision.IGNORE

    async def test_because_is_a_turn(self) -> None:
        det = ProviderTurnDetector()
        decision = await det.on_committed("because", _state())
        assert decision.action is CommitAction.NEW_TURN
        assert decision.hold_timeout_secs is None


class TestIgnoreBackchannels:
    def _detector(self, cls=ProviderTurnDetector):
        det = cls()
        det.ignore_backchannels = True
        return det

    @pytest.mark.parametrize("text", _BACKCHANNELS)
    async def test_commit_during_playback_is_ignored(self, text: str) -> None:
        for det in (self._detector(), self._detector(HeuristicTurnDetector)):
            decision = await det.on_committed(text, _playing())
            assert decision.action is CommitAction.IGNORE, text
            assert decision.reason == "backchannel", text

    @pytest.mark.parametrize("text", _BACKCHANNELS)
    async def test_partial_during_playback_does_not_barge_in(self, text: str) -> None:
        for det in (self._detector(), self._detector(HeuristicTurnDetector)):
            decision = await det.on_partial(text, _playing())
            assert decision is PartialDecision.IGNORE, text

    async def test_repeated_acknowledgement_would_have_cleared_the_word_gate(self) -> None:
        det = self._detector(HeuristicTurnDetector)
        assert await det.on_partial("yeah yeah yeah", _playing()) is PartialDecision.IGNORE
        bare = HeuristicTurnDetector()
        assert await bare.on_partial("yeah yeah yeah", _playing()) is PartialDecision.BARGE_IN

    @pytest.mark.parametrize("text", _INTERRUPTS)
    async def test_commands_still_commit_as_turns(self, text: str) -> None:
        det = self._detector()
        decision = await det.on_committed(text, _playing())
        assert decision.action is CommitAction.NEW_TURN, text
        assert decision.reason == "provider", text

    @pytest.mark.parametrize("text", _INTERRUPTS)
    async def test_commands_barge_in_on_the_partial(self, text: str) -> None:
        """One-word "no" / "para" are under the min-words gate. With the knob on
        they cut TTS immediately instead of waiting next to the dropped "yeah"."""
        for det in (self._detector(), self._detector(HeuristicTurnDetector)):
            decision = await det.on_partial(text, _playing())
            assert decision is PartialDecision.BARGE_IN, text

    async def test_a_content_word_keeps_the_interruption(self) -> None:
        det = self._detector()
        decision = await det.on_committed("okay stop", _playing())
        assert decision.action is CommitAction.NEW_TURN
        decision = await det.on_partial("okay stop that", _playing())
        assert decision is PartialDecision.BARGE_IN

    async def test_lexical_detector_inherits_the_filter(self) -> None:
        det = LexicalTurnDetector()
        det.ignore_backchannels = True
        decision = await det.on_committed("vale", _playing())
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "backchannel"

    @pytest.mark.parametrize("since", [0.0, 0.6, _BACKCHANNEL_GRACE_SECS])
    async def test_late_commit_inside_the_grace_window_is_ignored(self, since: float) -> None:
        det = self._detector()
        state = _state(assistant_active=False, audio_playing=False, seconds_since_assistant_active=since)
        decision = await det.on_committed("yeah", state)
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "backchannel"

    async def test_answer_after_the_grace_window_is_a_turn(self) -> None:
        det = self._detector()
        state = _state(
            assistant_active=False,
            audio_playing=False,
            seconds_since_assistant_active=_BACKCHANNEL_GRACE_SECS + 0.01,
        )
        decision = await det.on_committed("okay", state)
        assert decision.action is CommitAction.NEW_TURN

    async def test_acknowledgement_before_the_assistant_has_spoken_is_a_turn(self) -> None:
        det = self._detector()
        decision = await det.on_committed("okay", _state(seconds_since_assistant_active=None))
        assert decision.action is CommitAction.NEW_TURN

    async def test_playback_tail_without_assistant_active_still_filters(self) -> None:
        """audio_playing can outlive the generation flag by the client's buffer."""
        det = self._detector()
        state = _state(assistant_active=False, audio_playing=True, seconds_since_assistant_active=None)
        decision = await det.on_committed("claro", state)
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "backchannel"

    async def test_echo_still_wins_over_a_command(self) -> None:
        det = self._detector()
        state = _playing(assistant_text="If you need me to stop, say so.")
        decision = await det.on_partial("stop", state)
        assert decision is PartialDecision.IGNORE
        decision = await det.on_committed("stop", state)
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "echo"

    async def test_backchannel_does_not_merge_into_a_hold(self) -> None:
        det = ProviderTurnDetector(hold_unfinished_secs=1.2)
        det.ignore_backchannels = True
        decision = await det.on_committed(
            "vale",
            _playing(holding=True, active_user_text="es que"),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "backchannel"
        assert decision.text == "vale"


class TestHoldUnfinished:
    def _detector(self) -> ProviderTurnDetector:
        return ProviderTurnDetector(hold_unfinished_secs=1.2)

    @pytest.mark.parametrize(
        "text",
        ["…es que", "es que", "Necesito, eh,", "because", "because.", "I need it because"],
    )
    async def test_open_phrases_are_held_for_the_configured_timeout(self, text: str) -> None:
        decision = await self._detector().on_committed(text, _state())
        assert decision.action is CommitAction.HOLD, text
        assert decision.reason == "unfinished", text
        assert decision.hold_timeout_secs == 1.2
        assert decision.text == text

    @pytest.mark.parametrize(
        ("held", "follow", "merged"),
        [
            ("…es que", "está al lado de la unidad", "…es que está al lado de la unidad"),
            ("Necesito, eh,", "el repuesto del fondo", "Necesito, eh, el repuesto del fondo"),
            ("because", "the pump is leaking", "because the pump is leaking"),
        ],
    )
    async def test_follow_up_is_merged_into_one_turn(self, held: str, follow: str, merged: str) -> None:
        det = self._detector()
        first = await det.on_committed(held, _state())
        assert first.action is CommitAction.HOLD
        second = await det.on_committed(follow, _state(holding=True, active_user_text=first.text))
        assert second.action is CommitAction.NEW_TURN
        assert second.reason == "unfinished_merge"
        assert second.text == merged

    async def test_a_still_open_follow_up_is_held_again(self) -> None:
        det = self._detector()
        first = await det.on_committed("I need it for", _state())
        assert first.action is CommitAction.HOLD
        second = await det.on_committed("the", _state(holding=True, active_user_text=first.text))
        assert second.action is CommitAction.HOLD
        assert second.reason == "unfinished_merge"
        assert second.text == "I need it for the"
        assert second.hold_timeout_secs == 1.2
        third = await det.on_committed("blue one", _state(holding=True, active_user_text=second.text))
        assert third.action is CommitAction.NEW_TURN
        assert third.text == "I need it for the blue one"

    async def test_a_filler_follow_up_stays_on_the_hold(self) -> None:
        """ "eh" alone is a hesitation. After an open phrase it is the rest of it."""
        det = self._detector()
        first = await det.on_committed("I was going to", _state())
        assert first.action is CommitAction.HOLD
        second = await det.on_committed("eh", _state(holding=True, active_user_text=first.text))
        assert second.action is CommitAction.HOLD
        assert second.text == "I was going to eh"
        ghost = await det.on_committed("I", _state(holding=True, active_user_text=first.text))
        assert ghost.action is CommitAction.IGNORE
        assert ghost.reason == "contentless"

    async def test_cumulative_transcript_is_not_doubled(self) -> None:
        det = self._detector()
        decision = await det.on_committed(
            "Necesito, eh, el repuesto",
            _state(holding=True, active_user_text="Necesito, eh,"),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.text == "Necesito, eh, el repuesto"

    async def test_commands_and_finished_phrases_are_not_held(self) -> None:
        det = self._detector()
        for text in ("para", "¡para!", "stop", "And?", "So?", "Thank you.", "Tell me a story", "Él.", "sí"):
            decision = await det.on_committed(text, _state())
            assert decision.action is CommitAction.NEW_TURN, text
            assert decision.reason == "provider", text

    async def test_trailing_preposition_inside_a_phrase_is_held(self) -> None:
        decision = await self._detector().on_committed("lo necesito para", _state())
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "unfinished"
        decision = await self._detector().on_committed("dame el", _state())
        assert decision.action is CommitAction.HOLD

    async def test_question_that_ends_on_a_conjunction_is_a_turn(self) -> None:
        decision = await self._detector().on_committed("And?", _state())
        assert decision.action is CommitAction.NEW_TURN

    async def test_open_phrase_during_playback_interrupts_instead_of_holding(self) -> None:
        """HOLD during TTS is deferred, which would swallow "sí, pero…"."""
        det = self._detector()
        det.ignore_backchannels = True
        for text in ("because", "sí, pero…", "lo necesito para", "stop"):
            decision = await det.on_committed(text, _playing())
            assert decision.action is CommitAction.NEW_TURN, text

    async def test_follow_up_over_playback_starts_the_merged_turn(self) -> None:
        det = self._detector()
        decision = await det.on_committed(
            "the pump is leaking",
            _playing(holding=True, active_user_text="because"),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.text == "because the pump is leaking"
        assert decision.reason == "unfinished_merge"

    async def test_interrupt_while_holding_does_not_keep_waiting(self) -> None:
        det = self._detector()
        decision = await det.on_committed("para", _state(holding=True, active_user_text="es que"))
        assert decision.action is CommitAction.NEW_TURN
        assert decision.text == "es que para"

    async def test_lone_function_word_is_still_dropped(self) -> None:
        decision = await self._detector().on_committed("the", _state())
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "contentless"

    async def test_heuristic_detector_does_not_grow_this_hold(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed("because", _state())
        assert decision.action is CommitAction.NEW_TURN

    @pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, "1.2"])
    def test_rejects_a_bad_timeout(self, value: object) -> None:
        with pytest.raises(ValueError):
            ProviderTurnDetector(hold_unfinished_secs=value)  # type: ignore[arg-type]

    def test_clone_keeps_the_timeout_and_the_flag(self) -> None:
        det = ProviderTurnDetector(hold_unfinished_secs=1.2)
        det.ignore_backchannels = True
        cloned = det.clone()
        assert cloned is not det
        assert cloned.hold_unfinished_secs == 1.2
        assert cloned.ignore_backchannels is True
        cloned.hold_unfinished_secs = 2.0
        assert det.hold_unfinished_secs == 1.2


class TestVoiceConfigAndSession:
    def test_defaults_are_off(self) -> None:
        cfg = VoiceConfig()
        assert cfg.ignore_backchannels is False
        assert cfg.hold_unfinished_secs is None

    @pytest.mark.parametrize("value", [0, -0.5, math.inf])
    def test_config_rejects_a_bad_timeout(self, value: float) -> None:
        with pytest.raises(ValidationError):
            VoiceConfig(hold_unfinished_secs=value)

    def test_session_wires_both_knobs_onto_the_clone(self) -> None:
        original = ProviderTurnDetector()
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
            stt=MockSTT(script=[]),
            tts=MockTTS(),
            turn_detector=original,
            ignore_backchannels=True,
            hold_unfinished_secs=1.2,
        )
        assert session.turn_detector is not original
        assert session.turn_detector.ignore_backchannels is True
        assert session.turn_detector.hold_unfinished_secs == 1.2
        assert original.ignore_backchannels is False
        assert original.hold_unfinished_secs is None

    def test_detector_timeout_survives_the_session_clone(self) -> None:
        original = ProviderTurnDetector(hold_unfinished_secs=1.2)
        original.ignore_backchannels = True
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
            stt=MockSTT(script=[]),
            tts=MockTTS(),
            turn_detector=original,
        )
        assert session.turn_detector.hold_unfinished_secs == 1.2
        assert session.turn_detector.ignore_backchannels is True

    def test_hold_requires_the_provider_detector(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            VoiceSession(
                agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
                stt=MockSTT(script=[]),
                tts=MockTTS(),
                turn_detector="heuristic",
                hold_unfinished_secs=1.2,
            )
