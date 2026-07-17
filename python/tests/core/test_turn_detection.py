"""Tests for the TurnDetector abstraction and the default HeuristicTurnDetector.

Covers:
* the moved heuristics (garbage commits, refinements) at function level,
* ``HeuristicTurnDetector.on_partial`` / ``on_committed`` decisions,
* that ``VoiceSession`` honors an injected detector's decisions.
"""

from __future__ import annotations

from contextlib import aclosing

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice.session import TranscriptEvent, VoiceSession, VoiceSessionEvent
from timbal.voice.turn_detection import (
    CommitAction,
    CommitDecision,
    HeuristicTurnDetector,
    PartialDecision,
    TurnDetector,
    TurnState,
    _is_garbage_commit,
    _is_same_user_utterance_refinement,
)

from .test_voice_session import MockSTT, MockTTS

# ---------------------------------------------------------------------------
# Moved heuristic functions (ported from test_voice_session_stt_refinement.py)
# ---------------------------------------------------------------------------


class TestGarbageCommit:
    def test_garbage_lone_open_paren(self) -> None:
        assert _is_garbage_commit("(")
        assert _is_garbage_commit(" ( ")

    def test_garbage_music_close_caption_hallucination(self) -> None:
        assert _is_garbage_commit("Music)")
        assert _is_garbage_commit("Applause)")

    def test_garbage_incomplete_open_caption(self) -> None:
        assert _is_garbage_commit("(Music")
        assert _is_garbage_commit("(water splashing")

    def test_garbage_not_real_utterances(self) -> None:
        assert not _is_garbage_commit("Probably, yeah, maybe you can tell me a story.")
        assert not _is_garbage_commit("no")
        assert not _is_garbage_commit("Hello")


class TestRefinement:
    def test_refinement_prefix_extension(self) -> None:
        a = "Hello, hello"
        b = "Hello, hello, how are you?"
        assert _is_same_user_utterance_refinement(a, b)

    def test_refinement_duplicate(self) -> None:
        t = "Hello, hello, how are you?"
        assert _is_same_user_utterance_refinement(t, t)

    def test_refinement_substring_when_long_enough(self) -> None:
        a = "Hello, hello, how are you"
        b = "Well hello, hello, how are you today?"
        assert len(a) >= 10
        assert _is_same_user_utterance_refinement(a, b)

    def test_barge_in_shorter_not_refinement(self) -> None:
        active = "Hello, hello, how are you?"
        new = "stop"
        assert not _is_same_user_utterance_refinement(active, new)

    def test_barge_in_unrelated_longer_not_refinement(self) -> None:
        active = "What is the weather"
        new = "Tell me a short story about space"
        assert not _is_same_user_utterance_refinement(active, new)


# ---------------------------------------------------------------------------
# HeuristicTurnDetector decisions
# ---------------------------------------------------------------------------


def _state(**overrides) -> TurnState:
    defaults = dict(
        assistant_active=False,
        audio_playing=False,
        assistant_text="",
        active_user_text="",
        seconds_since_turn_start=10.0,
        seconds_since_last_commit=10.0,
        partials_since_last_commit=2,
    )
    defaults.update(overrides)
    return TurnState(**defaults)


class TestOnPartial:
    async def test_ignores_when_no_audio_playing(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_partial("stop right there", _state(audio_playing=False))
        assert decision is PartialDecision.IGNORE

    async def test_barge_in_on_real_speech_while_playing(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_partial(
            "wait, stop please",
            _state(audio_playing=True, assistant_active=True, assistant_text="Once upon a time"),
        )
        assert decision is PartialDecision.BARGE_IN

    async def test_ignores_noise_caption(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_partial("(music playing)", _state(audio_playing=True))
        assert decision is PartialDecision.IGNORE

    async def test_ignores_too_short(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_partial("uh", _state(audio_playing=True))
        assert decision is PartialDecision.IGNORE

    async def test_ignores_own_echo(self) -> None:
        det = HeuristicTurnDetector()
        assistant = "I'm doing great, thank you for asking! How are you today?"
        decision = await det.on_partial(
            "thank you for asking",
            _state(audio_playing=True, assistant_active=True, assistant_text=assistant),
        )
        assert decision is PartialDecision.IGNORE


class TestOnCommitted:
    async def test_plain_commit_starts_new_turn(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed("Tell me a story", _state())
        assert decision.action is CommitAction.NEW_TURN
        assert decision.text == "Tell me a story"

    async def test_noise_ignored(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed("(wind blowing)", _state())
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "noise"

    async def test_garbage_ignored(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed("Music)", _state())
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "garbage"

    async def test_hallucination_ignored(self) -> None:
        det = HeuristicTurnDetector()
        text = "This is a long hallucinated sentence with no partials at all."
        assert len(text) > 40
        decision = await det.on_committed(text, _state(partials_since_last_commit=0, assistant_active=False))
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "hallucination"

    async def test_long_commit_with_partials_not_hallucination(self) -> None:
        det = HeuristicTurnDetector()
        text = "This is a long real sentence the user actually spoke out loud."
        decision = await det.on_committed(text, _state(partials_since_last_commit=3))
        assert decision.action is CommitAction.NEW_TURN

    async def test_echo_ignored_while_assistant_active(self) -> None:
        det = HeuristicTurnDetector()
        assistant = "I'm doing great, thank you for asking! How are you today?"
        decision = await det.on_committed(
            "thank you for asking! How are you",
            _state(assistant_active=True, assistant_text=assistant),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "echo"

    async def test_refinement_ignored(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "Hello, hello, how are you?",
            _state(assistant_active=True, active_user_text="Hello, hello"),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "refinement"

    async def test_early_duplicate_ignored(self) -> None:
        det = HeuristicTurnDetector()
        # Similar enough for the early-duplicate window (ratio ~0.61 >= 0.58) but a
        # shorter re-commit, so the refinement check does not already swallow it.
        decision = await det.on_committed(
            "tell me the story",
            _state(
                assistant_active=True,
                active_user_text="tell me a story about dragons",
                seconds_since_turn_start=0.5,
                seconds_since_last_commit=0.5,
            ),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "early_duplicate"

    async def test_fast_short_fragment_becomes_continuation(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "estás?",
            _state(
                assistant_active=True,
                active_user_text="Hola, ¿qué tal",
                seconds_since_turn_start=2.0,
                seconds_since_last_commit=1.0,
            ),
        )
        assert decision.action is CommitAction.CONTINUE_TURN
        assert decision.text == "Hola, ¿qué tal estás?"
        assert decision.reason == "continuation"

    async def test_long_new_query_during_turn_is_new_turn(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "Actually I want to ask about something else entirely",
            _state(
                assistant_active=True,
                active_user_text="Hello there",
                seconds_since_turn_start=2.0,
                seconds_since_last_commit=1.0,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN


# ---------------------------------------------------------------------------
# VoiceSession honors an injected detector
# ---------------------------------------------------------------------------


class _ScriptedDetector(TurnDetector):
    """Detector that returns pre-scripted commit decisions in order."""

    def __init__(self, decisions: list[CommitDecision]) -> None:
        self._decisions = list(decisions)
        self.started = False
        self.closed = False
        self.audio_chunks: list[bytes] = []
        self.seen_commits: list[str] = []

    async def start(self, config) -> None:  # noqa: ARG002
        self.started = True

    async def close(self) -> None:
        self.closed = True

    def push_audio(self, chunk: bytes) -> None:
        self.audio_chunks.append(chunk)

    async def on_partial(self, text: str, state: TurnState) -> PartialDecision:  # noqa: ARG002
        return PartialDecision.IGNORE

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:  # noqa: ARG002
        self.seen_commits.append(text)
        return self._decisions.pop(0)


async def _run_session_with_detector(
    detector: TurnDetector,
    stt_script: list[TranscriptEvent],
    responses: list[str],
) -> tuple[VoiceSession, list[VoiceSessionEvent]]:
    agent = Agent(name="t", model=TestModel(responses=responses), tools=[])
    session = VoiceSession(
        agent=agent,
        stt=MockSTT(script=stt_script),
        tts=MockTTS(),
        turn_detector=detector,
    )

    async def _empty_audio():
        return
        yield  # noqa: RET504

    events: list[VoiceSessionEvent] = []
    async with aclosing(session.run(_empty_audio())) as stream:
        async for ev in stream:
            events.append(ev)
    return session, events


class TestSessionHonorsDetector:
    async def test_ignore_decision_suppresses_turn(self) -> None:
        detector = _ScriptedDetector([CommitDecision(action=CommitAction.IGNORE, text="Hello", reason="test")])
        session, events = await _run_session_with_detector(
            detector,
            [TranscriptEvent(type="committed", text="Hello")],
            ["Should never be spoken"],
        )
        assert detector.seen_commits == ["Hello"]
        assert session.transcript == []
        assert not any(e.type == "agent_text_done" for e in events)

    async def test_detector_text_override_used_for_turn(self) -> None:
        detector = _ScriptedDetector(
            [CommitDecision(action=CommitAction.NEW_TURN, text="rewritten text", reason="test")]
        )
        session, events = await _run_session_with_detector(
            detector,
            [TranscriptEvent(type="committed", text="original text")],
            ["ok"],
        )
        committed = next(e for e in events if e.type == "transcript_committed")
        assert committed.text == "rewritten text"
        assert session.transcript[0].text == "rewritten text"

    async def test_lifecycle_and_audio_forwarding(self) -> None:
        detector = _ScriptedDetector([])
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector=detector)

        async def _audio():
            yield b"\x00\x01" * 8

        async with aclosing(session.run(_audio())) as stream:
            async for _ in stream:
                pass
        await session.close()

        assert detector.started
        assert detector.closed
        assert detector.audio_chunks == [b"\x00\x01" * 8]

    async def test_default_detector_is_heuristic(self) -> None:
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS())
        assert isinstance(session.turn_detector, HeuristicTurnDetector)
