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
from timbal.voice.eou import AudioEouModel, PunctuationEouPredictor, TextEouPredictor
from timbal.voice.session import TranscriptEvent, VoiceSession, VoiceSessionEvent
from timbal.voice.turn_detection import (
    CommitAction,
    CommitDecision,
    HeuristicTurnDetector,
    LexicalTurnDetector,
    LocalAudioTurnDetector,
    PartialDecision,
    ProviderTurnDetector,
    SemanticTurnDetector,
    TurnDetector,
    TurnState,
    _is_garbage_commit,
    _is_same_user_utterance_refinement,
    resolve_turn_detector,
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
        holding=False,
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


class _FixedTextEou(TextEouPredictor):
    """Text predictor returning a constant score."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def predict_eou(self, text: str) -> float:  # noqa: ARG002
        return self.score


class _FixedAudioEou(AudioEouModel):
    """Audio EOU model returning a constant score."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:  # noqa: ARG002
        self.calls += 1
        return self.score


class TestLexicalTurnDetector:
    async def test_alias_semantic_is_lexical(self) -> None:
        assert SemanticTurnDetector is LexicalTurnDetector

    async def test_defaults_to_punctuation_predictor(self) -> None:
        det = LexicalTurnDetector()
        assert isinstance(det.text_eou, PunctuationEouPredictor)

    async def test_lifecycle_forwards_to_predictor(self) -> None:
        eou = _FixedTextEou(0.9)
        det = LexicalTurnDetector(text_eou=eou)
        await det.start(None)
        await det.close()
        assert eou.started and eou.closed

    async def test_idle_incomplete_holds(self) -> None:
        det = LexicalTurnDetector(text_eou=_FixedTextEou(0.1))
        decision = await det.on_committed("I was wondering about the", _state())
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "lexical_hold"

    async def test_idle_complete_is_new_turn(self) -> None:
        det = LexicalTurnDetector(text_eou=_FixedTextEou(0.99))
        decision = await det.on_committed("Tell me a story.", _state())
        assert decision.action is CommitAction.NEW_TURN

    async def test_incomplete_fragment_merges(self) -> None:
        det = LexicalTurnDetector(text_eou=_FixedTextEou(0.1))
        active = "I was wondering if you could tell me about"
        decision = await det.on_committed(
            "the weather forecast for tomorrow please now",
            _state(
                assistant_active=True,
                active_user_text=active,
                seconds_since_turn_start=3.0,
                seconds_since_last_commit=1.0,
            ),
        )
        assert decision.action is CommitAction.CONTINUE_TURN
        assert decision.reason == "lexical_continuation"

    async def test_punctuation_end_to_end_merge(self) -> None:
        det = LexicalTurnDetector()
        commit = "about your refund policy for damaged items please"
        decision = await det.on_committed(
            commit,
            _state(
                assistant_active=True,
                active_user_text="I have a question about the",
                seconds_since_turn_start=3.0,
                seconds_since_last_commit=1.0,
            ),
        )
        assert decision.action is CommitAction.CONTINUE_TURN
        assert decision.reason == "lexical_continuation"


class TestProviderTurnDetector:
    async def test_trusts_real_commits(self) -> None:
        det = ProviderTurnDetector()
        decision = await det.on_committed("Tell me a story", _state())
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "provider"

    async def test_still_filters_noise_and_garbage(self) -> None:
        det = ProviderTurnDetector()
        assert (await det.on_committed("(music)", _state())).action is CommitAction.IGNORE
        assert (await det.on_committed("Music)", _state())).action is CommitAction.IGNORE

    async def test_no_heuristic_continuation(self) -> None:
        # Short fast fragment that HeuristicTurnDetector would CONTINUE — provider does not.
        det = ProviderTurnDetector()
        decision = await det.on_committed(
            "estás?",
            _state(
                assistant_active=True,
                active_user_text="Hola, ¿qué tal",
                seconds_since_turn_start=2.0,
                seconds_since_last_commit=1.0,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN


class TestLocalAudioTurnDetector:
    async def test_without_model_matches_heuristic(self) -> None:
        det = LocalAudioTurnDetector(audio_eou=None)
        decision = await det.on_committed("Tell me a story", _state())
        assert decision.action is CommitAction.NEW_TURN

    async def test_incomplete_audio_holds(self) -> None:
        model = _FixedAudioEou(0.1)
        det = LocalAudioTurnDetector(audio_eou=model)
        await det.start(type("C", (), {"sample_rate": 16000})())
        # >= 0.5s of PCM16 @ 16kHz so the detector has enough signal.
        det.push_audio(b"\x00\x01" * 8000)
        decision = await det.on_committed("I was wondering", _state())
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "audio_hold"
        assert model.calls == 1
        await det.close()

    async def test_complete_audio_new_turn(self) -> None:
        det = LocalAudioTurnDetector(audio_eou=_FixedAudioEou(0.9))
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 8000)
        decision = await det.on_committed("Tell me a story", _state())
        assert decision.action is CommitAction.NEW_TURN
        await det.close()

    async def test_short_audio_incomplete_text_holds(self) -> None:
        """< 0.5s of buffered PCM must not pass an incomplete utterance through
        as NEW_TURN — the lexical fallback holds it (audio model never called)."""
        model = _FixedAudioEou(0.9)
        det = LocalAudioTurnDetector(audio_eou=model)
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 100)  # ~12ms of audio
        decision = await det.on_committed("I was wondering about", _state())
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "audio_short_lexical_hold"
        assert model.calls == 0
        await det.close()

    async def test_short_audio_complete_text_passes_through(self) -> None:
        model = _FixedAudioEou(0.1)  # would HOLD if it were consulted
        det = LocalAudioTurnDetector(audio_eou=model)
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 100)
        decision = await det.on_committed("Tell me a story.", _state())
        assert decision.action is CommitAction.NEW_TURN
        assert model.calls == 0
        await det.close()


class TestResolveTurnDetector:
    def test_none_and_heuristic(self) -> None:
        assert isinstance(resolve_turn_detector(None), HeuristicTurnDetector)
        assert isinstance(resolve_turn_detector("heuristic"), HeuristicTurnDetector)

    def test_named_modes(self) -> None:
        assert isinstance(resolve_turn_detector("provider"), ProviderTurnDetector)
        assert isinstance(resolve_turn_detector("local"), LocalAudioTurnDetector)
        assert isinstance(resolve_turn_detector("lexical"), LexicalTurnDetector)
        assert isinstance(resolve_turn_detector("semantic"), LexicalTurnDetector)

    def test_passthrough_instance(self) -> None:
        det = ProviderTurnDetector()
        assert resolve_turn_detector(det) is det

    def test_factory_callable_builds_per_call(self) -> None:
        a = resolve_turn_detector(ProviderTurnDetector)
        b = resolve_turn_detector(ProviderTurnDetector)
        assert isinstance(a, ProviderTurnDetector)
        assert a is not b

    def test_factory_returning_non_detector_raises(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            resolve_turn_detector(lambda: object())

    def test_unknown_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            resolve_turn_detector("nope")


class TestTurnDetectorClone:
    """Per-session copies so a shared voice_config instance is never reused."""

    def test_base_clone_is_distinct_object(self) -> None:
        det = HeuristicTurnDetector()
        assert det.clone() is not det

    async def test_local_audio_clone_isolates_pcm_buffer(self) -> None:
        model = _FixedAudioEou(0.9)
        det = LocalAudioTurnDetector(audio_eou=model, completion_threshold=0.7, hold_timeout_secs=3.0)
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 8000)

        other = det.clone()
        assert other is not det
        assert other.audio_eou is model  # weights shared, buffers not
        assert other.completion_threshold == 0.7
        assert other.hold_timeout_secs == 3.0
        assert other._pcm is not det._pcm
        assert len(other._pcm) == 0 and len(det._pcm) == 1

        # One session's close() must not clear another's buffer.
        await other.start(type("C", (), {"sample_rate": 16000})())
        other.push_audio(b"\x00\x01" * 8000)
        await det.close()
        assert len(other._pcm) == 1


class TestHoldDecisions:
    """HOLD must not be conflated with mid-turn assistant_active."""

    async def test_hold_refinement_updates_not_ignored(self) -> None:
        det = HeuristicTurnDetector()
        held = "I was wondering about"
        longer = "I was wondering about the weather tomorrow"
        decision = await det.on_committed(
            longer,
            _state(
                holding=True,
                active_user_text=held,
                assistant_active=False,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "hold_refinement"
        assert decision.text == longer

    async def test_hold_merge_short_continuation(self) -> None:
        """VAD-split continuation while holding still glues onto the fragment."""
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "the weather tomorrow",
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                seconds_since_last_commit=0.8,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "hold_merge"
        assert decision.text == "I was wondering about the weather tomorrow"

    async def test_hold_does_not_merge_long_new_utterance(self) -> None:
        """A separate long commit while holding must not be concatenated."""
        det = HeuristicTurnDetector()
        new = "Actually what is the capital of France please tell me"
        assert len(new) >= HeuristicTurnDetector.CONTINUATION_MAX_CHARS
        decision = await det.on_committed(
            new,
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                seconds_since_last_commit=1.0,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "hold_supersede"
        assert decision.text == new

    async def test_heuristic_hold_supersedes_stop_command(self) -> None:
        """Heuristic must not glue a self-contained short commit onto a HOLD."""
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "stop",
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                seconds_since_last_commit=0.5,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "hold_supersede"
        assert decision.text == "stop"

    async def test_lexical_hold_supersedes_stop_command(self) -> None:
        """Complete short command while holding must not produce garbled merge text."""
        det = LexicalTurnDetector()  # real punctuation EOU
        decision = await det.on_committed(
            "stop",
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                seconds_since_last_commit=0.5,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.text == "stop"
        assert decision.reason == "hold_supersede"

    async def test_lexical_does_not_rehold_parent_supersede(self) -> None:
        """Low EOU must not convert hold_supersede back into HOLD."""
        det = LexicalTurnDetector(text_eou=_FixedTextEou(0.1))
        decision = await det.on_committed(
            "stop",
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                seconds_since_last_commit=0.5,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "hold_supersede"
        assert decision.text == "stop"

    async def test_local_does_not_rehold_parent_supersede(self) -> None:
        """Incomplete audio must not convert hold_supersede back into HOLD."""
        det = LocalAudioTurnDetector(audio_eou=_FixedAudioEou(0.1))
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 8000)
        decision = await det.on_committed(
            "stop",
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                seconds_since_last_commit=0.5,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "hold_supersede"
        assert decision.text == "stop"
        await det.close()

    async def test_hallucination_still_ignored_while_holding(self) -> None:
        """Pending HOLD must not disable the silence-hallucination guard."""
        det = HeuristicTurnDetector()
        text = "This is a long hallucinated sentence with no partials at all."
        assert len(text) > 40
        decision = await det.on_committed(
            text,
            _state(
                holding=True,
                active_user_text="I was wondering",
                assistant_active=False,
                partials_since_last_commit=0,
            ),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "hallucination"

    async def test_lexical_hold_update_then_complete(self) -> None:
        """Longer re-commit while holding can graduate to NEW_TURN when complete."""
        det = LexicalTurnDetector(text_eou=_FixedTextEou(0.99))
        decision = await det.on_committed(
            "I was wondering about the weather tomorrow.",
            _state(
                holding=True,
                active_user_text="I was wondering about",
                assistant_active=False,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "lexical_hold_complete"


class TestHoldInSession:
    async def test_hold_timeout_starts_turn(self) -> None:
        """HOLD must not start the agent until the hold timer expires."""
        import asyncio

        from .test_voice_session import DelayedMockSTT

        class _HoldOnce(TurnDetector):
            def __init__(self) -> None:
                self.n = 0

            async def on_partial(self, text, state):  # noqa: ARG002
                return PartialDecision.IGNORE

            async def on_committed(self, text, state):  # noqa: ARG002
                self.n += 1
                if self.n == 1:
                    return CommitDecision(
                        action=CommitAction.HOLD,
                        text=text,
                        reason="test_hold",
                        hold_timeout_secs=0.05,
                    )
                return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="test")

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=agent,
            stt=stt,
            tts=MockTTS(),
            turn_detector=_HoldOnce(),
            hold_timeout_secs=0.05,
        )

        events: list[VoiceSessionEvent] = []

        async def _empty():
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="I was wondering about"))
            # Wait past the hold timeout so the deferred turn starts.
            while not any(getattr(e, "type", None) == "agent_text_done" for e in events):
                await asyncio.sleep(0.01)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=5)
        assert any(e.type == "agent_text_done" for e in events)
        assert session.transcript[0].text == "I was wondering about"

    async def test_hold_interrupts_playing_audio(self) -> None:
        """HOLD while TTS is still buffered must interrupt like NEW_TURN."""
        import asyncio

        from .test_voice_session import DelayedMockSTT, FakePlaybackTracker

        class _AlwaysHold(TurnDetector):
            async def on_partial(self, text, state):  # noqa: ARG002
                return PartialDecision.IGNORE

            async def on_committed(self, text, state):  # noqa: ARG002
                return CommitDecision(
                    action=CommitAction.HOLD,
                    text=text,
                    reason="test_hold",
                    hold_timeout_secs=5.0,
                )

        tracker = FakePlaybackTracker()
        tracker.playing = True
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=agent,
            stt=stt,
            tts=MockTTS(),
            turn_detector=_AlwaysHold(),
            playback_tracker=tracker,
            hold_timeout_secs=5.0,
        )
        events: list[VoiceSessionEvent] = []
        held_during: list[str | None] = []

        async def _empty():
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="I was wondering about"))
            for _ in range(50):
                if tracker.interrupt_calls > 0 and session._held_user_text:
                    held_during.append(session._held_user_text)
                    break
                await asyncio.sleep(0.01)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=5)
        assert tracker.interrupt_calls >= 1
        assert held_during == ["I was wondering about"]
        assert any(e.type == "interrupted" for e in events)

    async def test_hold_refinement_updates_session_fragment(self) -> None:
        """Longer STT re-commit while HOLDing must replace the held fragment."""
        import asyncio

        from .test_voice_session import DelayedMockSTT

        class _HoldRefine(TurnDetector):
            async def on_partial(self, text, state):  # noqa: ARG002
                return PartialDecision.IGNORE

            async def on_committed(self, text, state):  # noqa: ARG002
                # Exercise the real heuristic hold-refinement path via session state.
                return await HeuristicTurnDetector().on_committed(text, state)

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=agent,
            stt=stt,
            tts=MockTTS(),
            turn_detector=_HoldRefine(),
            hold_timeout_secs=5.0,
        )
        # Seed a pending HOLD the same way _arm_hold would.
        session._held_user_text = "I was wondering about"
        session._last_commit_at = 0.0
        events: list[VoiceSessionEvent] = []
        held_during: list[str | None] = []
        # Keep under HALLUCINATION_MIN_CHARS so 0-partial commits aren't dropped.
        longer = "I was wondering about weather"

        async def _empty():
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text=longer))
            for _ in range(50):
                if session._held_user_text == longer:
                    held_during.append(session._held_user_text)
                    break
                await asyncio.sleep(0.01)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=5)
        assert held_during == [longer]

    async def test_stale_hold_expiry_identity_guard(self) -> None:
        """A woken stale expiry timer must not wipe a re-armed hold.

        Deterministic version of the race: detach the first hold task so the
        re-arm's ``_cancel_hold`` cannot cancel it (mimicking a timer that
        already woke before the cancel landed), then let it fire — the
        ``self._hold_task is not me`` guard must bail without touching the
        new hold or starting a turn.
        """
        import asyncio

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS())

        await session._arm_hold("I was wondering about", 0.01)
        stale = session._hold_task
        # Simulate the stale timer surviving the re-arm's cancel.
        session._hold_task = None
        refined = "I was wondering about the weather"
        await session._arm_hold(refined, 60.0)

        await asyncio.sleep(0.05)  # stale timer fires and must no-op
        assert stale.done()
        assert session._held_user_text == refined
        assert session._current_turn_task is None
        session._cancel_hold()
        await asyncio.sleep(0)  # let the cancellation land

    async def test_hold_expiry_does_not_wipe_rearmed_hold(self) -> None:
        """Re-arm while holding, then the re-armed hold expires into a turn.

        State-driven (no racing wall-clock sleeps — Windows CI timers are too
        coarse for that): the refinement is injected only after the first hold
        is observably armed, and its timeout is generous enough to still be
        pending then.
        """
        import asyncio

        from .test_voice_session import DelayedMockSTT

        class _HoldThenRefine(TurnDetector):
            def __init__(self) -> None:
                self.n = 0

            async def on_partial(self, text, state):  # noqa: ARG002
                return PartialDecision.IGNORE

            async def on_committed(self, text, state):  # noqa: ARG002
                self.n += 1
                if self.n == 1:
                    return CommitDecision(
                        action=CommitAction.HOLD,
                        text=text,
                        reason="hold1",
                        hold_timeout_secs=2.0,
                    )
                if state.holding:
                    return CommitDecision(
                        action=CommitAction.HOLD,
                        text=text,
                        reason="hold2",
                        hold_timeout_secs=0.1,
                    )
                return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="new")

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=agent,
            stt=stt,
            tts=MockTTS(),
            turn_detector=_HoldThenRefine(),
        )
        events: list[VoiceSessionEvent] = []
        refined = "I was wondering about the weather"

        async def _empty():
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="I was wondering about"))
            while session._held_user_text != "I was wondering about":
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text=refined))
            # The re-armed hold replaces the fragment, then expires into a real
            # turn (it may expire before this loop observes the held text).
            while session._held_user_text != refined and not any(
                getattr(e, "type", None) == "agent_text_done" for e in events
            ):
                await asyncio.sleep(0.01)
            while not any(getattr(e, "type", None) == "agent_text_done" for e in events):
                await asyncio.sleep(0.01)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)
        assert any(e.type == "agent_text_done" for e in events)
        assert session.transcript[0].text == refined

    async def test_heuristic_never_holds(self) -> None:
        """Default path: no HOLD action, no deferred turns."""
        det = HeuristicTurnDetector()
        d = await det.on_committed("Tell me a story", _state())
        assert d.action is not CommitAction.HOLD


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

        # The session runs a per-session clone, not the passed-in object.
        assert session.turn_detector.started
        assert session.turn_detector.closed
        assert session.turn_detector.audio_chunks == [b"\x00\x01" * 8]

    async def test_default_detector_is_heuristic(self) -> None:
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS())
        assert isinstance(session.turn_detector, HeuristicTurnDetector)

    async def test_session_clones_shared_detector_instance(self) -> None:
        """Two sessions built from one instance (or a singleton factory) must
        not share a detector — its start/push_audio/close lifecycle is owned
        per session."""
        shared = LocalAudioTurnDetector(audio_eou=None)
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        s1 = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector=shared)
        s2 = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector=shared)
        assert s1.turn_detector is not shared
        assert s2.turn_detector is not shared
        assert s1.turn_detector is not s2.turn_detector
        assert s1.turn_detector._pcm is not s2.turn_detector._pcm

        # Factory that (incorrectly) returns a singleton — still isolated.
        f1 = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector=lambda: shared)
        assert f1.turn_detector is not shared
