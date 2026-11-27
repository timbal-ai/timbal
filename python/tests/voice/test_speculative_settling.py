"""Speculative attempts: which dropped commit settles one, and what the turn detector hears around it.

From the review of timbal-ai/timbal#323 on 1931da8e. A held reply is silence to the caller: it is neither
the assistant speaking (backchannels, unfinished holds) nor something that can echo.
"""

from __future__ import annotations

import time
import types

import pytest
from timbal.voice import TranscriptEvent

from .test_speculative import _Drive, _session, _until


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    import timbal.voice.session as session_module

    clock = [time.monotonic()]
    monkeypatch.setattr(
        session_module,
        "time",
        types.SimpleNamespace(monotonic=lambda: clock[0], time=time.time, perf_counter=time.perf_counter),
    )
    return clock


def _record_decisions(session) -> list[tuple[str, str | None]]:
    decisions: list[tuple[str, str | None]] = []
    decide = session.turn_detector.on_committed

    async def recorded(text, state):
        decision = await decide(text, state)
        decisions.append((text, decision.reason))
        return decision

    session.turn_detector.on_committed = recorded
    return decisions


async def _heard_reply(session, stt, drive, prompt: str) -> int:
    await stt.inject(TranscriptEvent(type="committed", text=prompt))
    await _until(lambda: drive.audio())
    await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
    await _until(lambda: not session._assistant_audio_playing)
    return len(drive.audio())


class TestASmartTurnFinalSettlesItsAttempt:
    async def test_a_hesitation_final_that_drops_a_word_releases_the_attempt(self) -> None:
        session, stt, _tts, model = _session(["Reply."])
        session._stale_partial_commit_secs = 60.0
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="partial", text="um okay"))
            await _until(lambda: session._latest_partial_text == "um okay")
            await session._maybe_start_speculative("um okay", source="smart_turn")
            await _until(lambda: model.call_count >= 1)
            await stt.inject(TranscriptEvent(type="committed", text="Um."))  # dropped: a hesitation
            await _until(lambda: session._speculative_attempt is None)
            await _until(lambda: not session._is_speaking)

    async def test_a_final_with_more_words_dropped_as_echo_releases_the_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _fake_clock(monkeypatch)
        session, stt, _tts, model = _session(["Should I book a table for two?", "Booked.", "Fallback."])
        session._stale_partial_commit_secs = 60.0
        async with _Drive(session, stt) as drive:
            await _heard_reply(session, stt, drive, "I want to eat out")
            clock[0] += 0.3
            await stt.inject(TranscriptEvent(type="partial", text="book a table"))
            await _until(lambda: session._latest_partial_text == "book a table")
            await session._maybe_start_speculative("book a table", source="smart_turn")
            await _until(lambda: model.call_count >= 2)
            clock[0] += 0.5
            await stt.inject(TranscriptEvent(type="committed", text="book a table for two"))  # dropped: echo
            await _until(lambda: session._speculative_attempt is None)

    @pytest.mark.xfail(
        strict=True,
        reason="The stale-partial sweep (on main) still rescues a partial whose final was dropped: an IGNOREd "
        "commit does not move _last_commit_at, by design for echo the AEC let through.",
    )
    async def test_the_sweep_does_not_answer_a_partial_the_final_revised_away(self) -> None:
        session, stt, _tts, model = _session(["Reply to um okay.", "Other."])
        session._stale_partial_commit_secs = 0.5
        session._stale_partial_poll_secs = 0.05
        decisions = _record_decisions(session)
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="partial", text="um okay"))
            await _until(lambda: session._latest_partial_text == "um okay")
            await session._maybe_start_speculative("um okay", source="smart_turn")
            await _until(lambda: model.call_count >= 1)
            await stt.inject(TranscriptEvent(type="committed", text="Um."))
            # The final, then the sweep's synthesized commit of the stranded partial.
            await _until(lambda: len(decisions) >= 2, timeout=5)
            await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
            users = [e.text for e in session.transcript if e.role == "user"]
        assert users == [] and not drive.audio(), (users, len(drive.audio()))


class TestAHeldReplyIsSilence:
    @pytest.mark.parametrize("speculate", [False, True])
    async def test_an_okay_answering_a_question_is_a_turn(
        self, monkeypatch: pytest.MonkeyPatch, speculate: bool
    ) -> None:
        clock = _fake_clock(monkeypatch)
        session, stt, _tts, model = _session(["Shall I book it?", "Booked.", "Fallback."], speculative=speculate)
        session.turn_detector.ignore_backchannels = True
        session._stale_partial_commit_secs = 60.0
        async with _Drive(session, stt) as drive:
            heard = await _heard_reply(session, stt, drive, "I need a table")
            clock[0] += 3.0  # past every grace
            if speculate:
                await stt.inject(TranscriptEvent(type="eager", text="okay"))
                await _until(lambda: model.call_count >= 2)
            await stt.inject(TranscriptEvent(type="committed", text="Okay."))
            await _until(lambda: len(drive.audio()) > heard)
        assert [e.text for e in session.transcript if e.role == "user"] == ["I need a table", "Okay."]

    @pytest.mark.parametrize("speculate", [False, True])
    async def test_an_unfinished_final_is_held(self, speculate: bool) -> None:
        from timbal.voice.turn_detection import ProviderTurnDetector

        session, stt, _tts, model = _session(["A reply to half a sentence.", "Fallback."], speculative=speculate)
        session.turn_detector = ProviderTurnDetector(hold_unfinished_secs=5.0)
        session._stale_partial_commit_secs = 60.0
        async with _Drive(session, stt):
            if speculate:
                await stt.inject(TranscriptEvent(type="eager", text="I need it for the"))
                await _until(lambda: model.call_count >= 1)
            await stt.inject(TranscriptEvent(type="committed", text="I need it for the"))
            await _until(lambda: session._held_user_text == "I need it for the")


class TestTheEchoReferenceAroundADiscardedAttempt:
    async def test_the_heard_reply_still_echoes_after_an_attempt_is_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _fake_clock(monkeypatch)
        session, stt, _tts, model = _session(["Say cancel to stop the booking.", "Unheard attempt reply.", "Fallback."])
        async with _Drive(session, stt) as drive:
            await _heard_reply(session, stt, drive, "hi")
            clock[0] += 0.3
            await stt.inject(TranscriptEvent(type="eager", text="cancel"))
            await _until(lambda: "Unheard" in session._turn_assistant_text)
            await stt.inject(TranscriptEvent(type="resumed", text=""))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
            clock[0] += 0.3
            assert "Say cancel to stop the booking." in session._turn_state().assistant_text

    async def test_an_unheard_discarded_reply_is_not_an_echo_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _fake_clock(monkeypatch)
        session, stt, _tts, model = _session(["Hello there.", "Sure, booked for Friday.", "Fallback."])
        async with _Drive(session, stt) as drive:
            await _heard_reply(session, stt, drive, "hi")
            clock[0] += 5.0
            await stt.inject(TranscriptEvent(type="eager", text="book it"))
            await _until(lambda: "Friday" in session._turn_assistant_text)
            await stt.inject(TranscriptEvent(type="resumed", text=""))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
            clock[0] += 0.3
            assert "Friday" not in session._turn_state().assistant_text

    @pytest.mark.parametrize("speculate", [False, True])
    async def test_a_late_echo_after_a_discarded_attempt_is_not_a_turn(
        self, monkeypatch: pytest.MonkeyPatch, speculate: bool
    ) -> None:
        clock = _fake_clock(monkeypatch)
        session, stt, _tts, model = _session(
            ["Say cancel to stop the booking.", "Unheard attempt reply.", "Ghost reply."], speculative=speculate
        )
        decisions = _record_decisions(session)
        async with _Drive(session, stt) as drive:
            await _heard_reply(session, stt, drive, "hi")
            session._last_assistant_active_at = clock[0]  # a partial saw the reply's audio playing to its end
            decisions.clear()
            clock[0] += 0.3
            if speculate:
                await stt.inject(TranscriptEvent(type="eager", text="I want"))
                await _until(lambda: "Unheard" in session._turn_assistant_text)
                await stt.inject(TranscriptEvent(type="partial", text="I want to"))  # the caller kept going
                await _until(lambda: session.speculative_generations_discarded == 1)
                await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
            clock[0] += 0.5  # the reply's echo lands 0.8 s after its audio ended: inside the grace
            await stt.inject(TranscriptEvent(type="committed", text="Say cancel to stop the booking"))
            await _until(lambda: decisions)
        assert decisions == [("Say cancel to stop the booking", "echo")]
        assert "Say cancel to stop the booking" not in [e.text for e in session.transcript if e.role == "user"]
