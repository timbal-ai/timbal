"""Adversarial tests for turn detection, HOLD, PCM buffers, and session races.

These encode the failure modes we keep discovering in live voice sessions —
and a few more that live testing hasn't hit yet but the code paths allow.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice.eou import AudioEouModel
from timbal.voice.session import TranscriptEvent, VoiceSession, VoiceSessionEvent
from timbal.voice.eou import PunctuationEouPredictor
from timbal.voice.turn_detection import (
    CommitAction,
    CommitDecision,
    HeuristicTurnDetector,
    LocalAudioTurnDetector,
    PartialDecision,
    TurnDetector,
    _is_hesitation_only,
    _looks_like_fresh_hold_utterance,
)

from .test_turn_detection import _FixedAudioEou, _state
from .test_voice_session import DelayedMockSTT, MockTTS

# ---------------------------------------------------------------------------
# Pure helpers — edge cases that have bitten us / look sharp
# ---------------------------------------------------------------------------


class TestHesitationAdversarial:
    def test_variants_and_punctuation(self) -> None:
        for text in (
            "Uh...",
            "Um, uh...",
            "HMM.",
            "eh eh eh",
            "Mmm, mhm",
            "  uhh  ",
            "Uh-huh",  # hyphenated — words split may leave "uh" + "huh" or "uh-huh"
        ):
            # "Uh-huh" is a grey area: if the regex splits on hyphen we may not
            # treat it as hesitation-only. Either way it must not crash.
            _is_hesitation_only(text)

        assert _is_hesitation_only("Uh...")
        assert _is_hesitation_only("Um, uh...")
        assert not _is_hesitation_only("Uh, tell me a story")
        assert not _is_hesitation_only("no")
        assert not _is_hesitation_only("Sí")
        assert not _is_hesitation_only("Okay")
        assert not _is_hesitation_only("")

    def test_empty_and_punctuation_only_not_hesitation(self) -> None:
        # Punctuation-only is garbage, not hesitation — different IGNORE reason.
        assert not _is_hesitation_only("...")
        assert not _is_hesitation_only("???")


class TestFreshHoldUtteranceAdversarial:
    def test_dangling_continuation_not_fresh(self) -> None:
        # The live failure: "About, uh, a story…" — "about" is a dangling token.
        assert not _looks_like_fresh_hold_utterance("About, uh, a story that my parent told me")
        assert not _looks_like_fresh_hold_utterance("the weather in Paris tomorrow")
        assert not _looks_like_fresh_hold_utterance("and then what happened")

    def test_stop_and_new_question_are_fresh(self) -> None:
        assert _looks_like_fresh_hold_utterance("Stop")
        assert _looks_like_fresh_hold_utterance("Forget it, new question")
        assert _looks_like_fresh_hold_utterance("Actually can you help with something else")


class TestHallucinationWhileHolding:
    """Long zero-partial commits while HOLDing must NOT be eaten as hallucinations.

    Live path: user pauses mid-thought → HOLD → STT eventually commits a long
    continuation that may arrive with few/no partials after the thinking pause.
    """

    async def test_long_zero_partial_commit_while_holding_is_not_hallucination(self) -> None:
        det = HeuristicTurnDetector()
        held = "Hello. Uh, I was thinking about..."
        continuation = "About, uh, a story that my parent told me, uh, last week."
        assert len(continuation) >= HeuristicTurnDetector.HALLUCINATION_MIN_CHARS
        decision = await det.on_committed(
            continuation,
            _state(
                holding=True,
                active_user_text=held,
                assistant_active=False,
                partials_since_last_commit=0,
                seconds_since_last_commit=7.0,
            ),
        )
        assert decision.action is not CommitAction.IGNORE
        assert decision.reason != "hallucination"

    async def test_idle_long_zero_partial_still_hallucination(self) -> None:
        det = HeuristicTurnDetector()
        text = "This is a long hallucinated sentence with no partials at all."
        decision = await det.on_committed(
            text,
            _state(partials_since_last_commit=0, assistant_active=False, holding=False),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "hallucination"


# ---------------------------------------------------------------------------
# LocalAudioTurnDetector — buffers + merge window
# ---------------------------------------------------------------------------


class TestPcmBufferAdversarial:
    async def test_buffer_caps_at_window(self) -> None:
        det = LocalAudioTurnDetector(audio_eou=_FixedAudioEou(0.9))
        await det.start(type("C", (), {"sample_rate": 16000})())
        # 12s of PCM16 @ 16kHz → must keep only last 8s.
        chunk = b"\x00\x01" * 1600  # 100ms
        for _ in range(120):
            det.push_audio(chunk)
        assert det._pcm_bytes <= det._max_pcm_bytes
        # At most one leftover chunk over the cap (whole-chunk drops).
        assert det._pcm_bytes <= det._max_pcm_bytes + len(chunk)
        await det.close()

    async def test_empty_chunks_ignored(self) -> None:
        det = LocalAudioTurnDetector(audio_eou=_FixedAudioEou(0.9))
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"")
        det.push_audio(b"\x00\x01" * 100)
        assert det._pcm_bytes == 200
        await det.close()

    async def test_clone_isolates_pcm_buffer(self) -> None:
        shared = _FixedAudioEou(0.9)
        det = LocalAudioTurnDetector(audio_eou=shared)
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 8000)
        other = det.clone()
        await other.start(type("C", (), {"sample_rate": 16000})())
        assert other._pcm_bytes == 0
        assert other.audio_eou is shared
        other.push_audio(b"\x00\x01" * 100)
        assert det._pcm_bytes == 16000
        assert other._pcm_bytes == 200
        await det.close()
        await other.close()

    async def test_audio_eou_exception_falls_back_to_parent(self) -> None:
        class _Boom(AudioEouModel):
            async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:  # noqa: ARG002
                raise RuntimeError("onnx died")

        det = LocalAudioTurnDetector(audio_eou=_Boom())
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 8000)
        decision = await det.on_committed("Tell me a story.", _state())
        # Parent heuristic would NEW_TURN; exception must not raise.
        assert decision.action is CommitAction.NEW_TURN
        await det.close()


class TestLocalHoldMergeLiveFailure:
    """Exact shape of the 2026-07-21 live session that split one thought into two turns."""

    async def test_about_continuation_merges_after_long_pause(self) -> None:
        det = LocalAudioTurnDetector(audio_eou=_FixedAudioEou(0.95))
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 16000)
        held = "Hello. Uh, I was thinking about..."
        continuation = "About, uh, a story that my parent told me, uh, last week."
        decision = await det.on_committed(
            continuation,
            _state(
                holding=True,
                active_user_text=held,
                assistant_active=False,
                seconds_since_last_commit=7.0,
                partials_since_last_commit=3,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "audio_complete"
        assert held in decision.text
        assert "story that my parent told me" in decision.text
        await det.close()

    async def test_stop_still_supersedes_after_long_pause(self) -> None:
        det = LocalAudioTurnDetector(audio_eou=_FixedAudioEou(0.95))
        await det.start(type("C", (), {"sample_rate": 16000})())
        det.push_audio(b"\x00\x01" * 16000)
        decision = await det.on_committed(
            "Stop. Forget the previous question entirely please",
            _state(
                holding=True,
                active_user_text="Hello. Uh, I was thinking about...",
                assistant_active=False,
                seconds_since_last_commit=7.0,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN
        assert decision.reason == "hold_supersede"
        assert decision.text.startswith("Stop.")
        await det.close()


# ---------------------------------------------------------------------------
# Session-level races — the nasty ones
# ---------------------------------------------------------------------------


class _SlowAudioEou(AudioEouModel):
    """Scores from a script after a controllable delay — forces event-loop yield."""

    def __init__(self, scores: list[float], delay: float) -> None:
        self.scores = list(scores)
        self.delay = delay
        self.calls = 0

    async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:  # noqa: ARG002
        self.calls += 1
        await asyncio.sleep(self.delay)
        idx = min(self.calls - 1, len(self.scores) - 1)
        return self.scores[idx]


class _HoldThenMerge(TurnDetector):
    """First commit HOLDs; second commit NEW_TURNs the merged text."""

    def __init__(self, hold_timeout: float = 0.15) -> None:
        self.n = 0
        self.hold_timeout = hold_timeout
        self.decisions: list[str] = []

    async def on_partial(self, text, state):  # noqa: ARG002
        return PartialDecision.IGNORE

    async def on_committed(self, text, state):  # noqa: ARG002
        self.n += 1
        if self.n == 1:
            self.decisions.append("hold")
            return CommitDecision(
                action=CommitAction.HOLD,
                text=text,
                reason="test_hold",
                hold_timeout_secs=self.hold_timeout,
            )
        if state.holding and state.active_user_text:
            merged = state.active_user_text.rstrip(", ") + " " + text
            self.decisions.append("merge")
            return CommitDecision(action=CommitAction.NEW_TURN, text=merged, reason="test_merge")
        self.decisions.append("new")
        return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="test")


async def _run_session(session: VoiceSession, drive) -> list[VoiceSessionEvent]:
    events: list[VoiceSessionEvent] = []

    async def _empty():
        return
        yield  # noqa: RET504

    async def _collect() -> None:
        async with aclosing(session.run(_empty())) as stream:
            driver = asyncio.create_task(drive(events))
            async for ev in stream:
                events.append(ev)
            await driver

    await asyncio.wait_for(_collect(), timeout=8)
    return events


class TestHoldSessionRaces:
    async def test_hold_expiry_does_not_race_slow_commit_decision(self) -> None:
        """Hold timer must not fire a turn while a follow-up commit is mid-decision.

        Reproduces: HOLD armed with short timeout → continuation commit arrives
        and awaits a slow audio EOU → without the timer cancel, expiry starts a
        turn on the fragment AND the commit starts a second turn.
        """
        # First commit incomplete → HOLD; second (during slow await) complete → merge.
        model = _SlowAudioEou(scores=[0.05, 0.95], delay=0.25)
        det = LocalAudioTurnDetector(audio_eou=model, hold_timeout_secs=0.05)
        agent = Agent(name="t", model=TestModel(responses=["ok", "ok2"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(agent=agent, stt=stt, tts=MockTTS(), turn_detector=det)

        async def _drive(events: list) -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            # Seed enough PCM so local mode scores audio (not lexical fallback).
            det.push_audio(b"\x00\x01" * 16000)
            await stt.inject(TranscriptEvent(type="committed", text="I was thinking about"))
            # Wait until hold is armed, then immediately fire the continuation.
            # Its slow EOU await (0.25s) outlasts the 0.05s hold timer — without
            # cancelling the timer at commit-start, expiry starts a fragment turn
            # mid-decision and we get two user entries.
            for _ in range(80):
                if session._held_user_text:
                    break
                await asyncio.sleep(0.01)
            assert session._held_user_text, "hold never armed"
            await stt.inject(TranscriptEvent(type="committed", text="about the weather in Paris tomorrow please"))
            for _ in range(100):
                if sum(1 for e in events if getattr(e, "type", None) == "agent_text_done") >= 1:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
            await stt.finish()

        events = await _run_session(session, _drive)
        user_entries = [e.text for e in session.transcript if e.role == "user"]
        # Must be exactly one merged user turn — not fragment + continuation.
        assert len(user_entries) == 1, user_entries
        assert "thinking about" in user_entries[0]
        assert "weather in Paris" in user_entries[0]
        assert sum(1 for e in events if getattr(e, "type", None) == "agent_text_done") == 1

    async def test_ignore_mid_hold_does_not_freeze_hold(self) -> None:
        """A hesitation/noise IGNORE while HOLDing must re-arm the expiry timer."""

        class _HoldIgnore(TurnDetector):
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
                        hold_timeout_secs=0.15,
                    )
                return CommitDecision(action=CommitAction.IGNORE, text=text, reason="hesitation")

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(agent=agent, stt=stt, tts=MockTTS(), turn_detector=_HoldIgnore(), hold_timeout_secs=0.15)

        async def _drive(events: list) -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="I was thinking about"))
            for _ in range(50):
                if session._held_user_text:
                    break
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Uh..."))
            # Hold must still expire into a turn after the IGNORE.
            for _ in range(80):
                if any(getattr(e, "type", None) == "agent_text_done" for e in events):
                    break
                await asyncio.sleep(0.02)
            await stt.finish()

        events = await _run_session(session, _drive)
        assert any(getattr(e, "type", None) == "agent_text_done" for e in events)
        assert session.transcript[0].text == "I was thinking about"

    async def test_hold_refine_keeps_longer_text(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "Hello. Uh, I was thinking about the weather",
            _state(
                holding=True,
                active_user_text="Hello. Uh, I was thinking about",
                assistant_active=False,
                partials_since_last_commit=2,
            ),
        )
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "hold_refinement"
        assert decision.text == "Hello. Uh, I was thinking about the weather"

    async def test_hold_refine_does_not_shrink(self) -> None:
        det = HeuristicTurnDetector()
        longer = "Hello. Uh, I was thinking about the weather tomorrow"
        decision = await det.on_committed(
            "Hello. Uh, I was thinking about",
            _state(
                holding=True,
                active_user_text=longer,
                assistant_active=False,
                partials_since_last_commit=1,
            ),
        )
        assert decision.action is CommitAction.HOLD
        assert decision.text == longer

    async def test_partial_grace_survives_vad_silence_gap(self) -> None:
        """Partials every < grace window must keep deferring hold expiry."""
        det = _HoldThenMerge(hold_timeout=0.08)
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(agent=agent, stt=stt, tts=MockTTS(), turn_detector=det)
        session._hold_partial_grace_secs = 0.25
        turn_at: list[float] = []
        partials_end: list[float] = []

        async def _drive(events: list) -> None:
            while not any(getattr(e, "type", None) == "session_started" for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="I was thinking about"))
            for _ in range(50):
                if session._held_user_text:
                    break
                await asyncio.sleep(0.01)
            # Speak for ~0.5s with partials every 0.1s (past the 0.08s hold).
            for _ in range(5):
                await stt.inject(TranscriptEvent(type="partial", text="the weather"))
                await asyncio.sleep(0.1)
            partials_end.append(time.monotonic())
            for _ in range(80):
                if any(getattr(e, "type", None) == "transcript_committed" for e in events):
                    # first committed is the HOLD arm (TranscriptCommitted is only
                    # emitted when a turn begins — HOLD does not emit it).
                    pass
                if any(getattr(e, "type", None) == "agent_text_done" for e in events):
                    turn_at.append(time.monotonic())
                    break
                await asyncio.sleep(0.02)
            await stt.finish()

        events = await _run_session(session, _drive)
        assert any(getattr(e, "type", None) == "agent_text_done" for e in events)
        assert partials_end and turn_at
        assert turn_at[0] >= partials_end[0] - 0.2


class TestEchoAndBargeInAdversarial:
    async def test_short_echo_tail_ignored(self) -> None:
        det = HeuristicTurnDetector()
        assistant = "I'm doing well, thank you! How about you?"
        decision = await det.on_committed(
            "How about you?",
            _state(assistant_active=True, assistant_text=assistant),
        )
        assert decision.action is CommitAction.IGNORE
        assert decision.reason == "echo"

    async def test_real_barge_in_not_echo(self) -> None:
        det = HeuristicTurnDetector()
        decision = await det.on_committed(
            "Actually stop, I have a different question about parking",
            _state(
                assistant_active=True,
                assistant_text="I'm doing well, thank you! How about you?",
                active_user_text="Hello",
                seconds_since_turn_start=3.0,
                seconds_since_last_commit=2.0,
            ),
        )
        assert decision.action is CommitAction.NEW_TURN

    async def test_hesitation_partial_never_barges(self) -> None:
        det = HeuristicTurnDetector()
        for text in ("Uh...", "Um", "Hmm."):
            d = await det.on_partial(
                text,
                _state(audio_playing=True, assistant_active=True, assistant_text="Hello there"),
            )
            assert d is PartialDecision.IGNORE, text
