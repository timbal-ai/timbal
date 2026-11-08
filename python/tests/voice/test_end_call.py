"""``VoiceSession.end_call`` — the agent hangs up after its goodbye has been heard."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from timbal.core.tool import Tool
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    CommitAction,
    CommitDecision,
    HeuristicTurnDetector,
    SessionStarted,
    TranscriptEntry,
    TranscriptEvent,
    TurnState,
    VoiceSessionEvent,
    current_voice_session,
)
from timbal.voice.session import _hangup_farewell_phrase, _hangup_utterance_kind

from .test_session import FakePlaybackTracker, MockTTS, _FakeEndpointer
from .test_voice_events import _Model, _session, _texts, _until

GOODBYE = "Perfecto, hasta luego."


async def end_call() -> str:
    session = current_voice_session()
    assert session is not None
    session.end_call()
    return "ok"


def _hangs_up_with(reply: str, *later: str) -> _Model:
    return _Model(
        Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name="end_call", input={})],
            stop_reason="tool_use",
        ),
        reply,
        *later,
    )


async def _call(session, events: list[VoiceSessionEvent]) -> asyncio.Task[None]:
    """Run ``session`` in the background; the STT stream stays open, as on a live line."""

    async def _no_audio() -> AsyncIterator[bytes]:
        return
        yield  # noqa: RET504 — make it an async generator

    async def _go() -> None:
        async with aclosing(session.run(_no_audio())) as stream:
            async for event in stream:
                events.append(event)

    task = asyncio.create_task(_go())
    await _until(lambda: any(isinstance(e, SessionStarted) for e in events))
    return task


class TestEndCall:
    async def test_hangs_up_once_the_goodbye_has_played(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_hangs_up_with(GOODBYE), tools=[end_call], tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed, "hung up over its own goodbye"

        tracker.playing = False
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_the_caller_speaking_again_keeps_the_call_up(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_hangs_up_with(GOODBYE, "Dime."), tools=[end_call], tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        await stt.inject(TranscriptEvent(type="committed", text="Espera, una cosa más."))
        await _until(lambda: "Dime." in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.5)

        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_partial_barge_in_during_goodbye_holds_hangup_until_commit_cancels_it(self) -> None:
        class FlushingTracker(FakePlaybackTracker):
            def on_interrupted(self) -> None:
                super().on_interrupted()
                self.playing = False

        tracker = FlushingTracker()
        session, stt, _tts = _session(_hangs_up_with(GOODBYE, "Dime."), tools=[end_call], tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = True
        # Partial barge-in interrupts immediately and flushes playback before the commit lands.
        await stt.inject(TranscriptEvent(type="partial", text="Espera, una cosa"))
        await _until(lambda: not tracker.playing)
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed, "hung up while caller was mid-utterance after barge-in"

        await stt.inject(TranscriptEvent(type="committed", text="Espera, una cosa más."))
        await _until(lambda: "Dime." in _texts(events))
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_local_vad_speech_holds_hangup(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_hangs_up_with(GOODBYE), tools=[end_call], tracker=tracker)
        endpointer = _FakeEndpointer(speech_secs=0.4)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        session._endpointer = endpointer  # type: ignore[assignment]

        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed, "hung up while local VAD heard caller speech"

        endpointer.speech_secs = 0.0
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_committed_utterance_awaiting_turn_detector_holds_hangup(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedDetector(HeuristicTurnDetector):
            async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
                if "Wait" in text or "noise" in text:
                    entered.set()
                    await release.wait()
                if "noise" in text:
                    return CommitDecision(action=CommitAction.IGNORE, reason="noise")
                return await super().on_committed(text, state)

        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE, "Dime.", GOODBYE),
            tools=[end_call],
            tracker=tracker,
        )
        session.turn_detector = GatedDetector()
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        await stt.inject(TranscriptEvent(type="partial", text="Wait, one more question"))
        await stt.inject(TranscriptEvent(type="committed", text="Wait, one more question"))
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        tracker.playing = False

        # Even past the quiet window + hang-up tail, an in-flight detector decision blocks hang-up.
        await asyncio.sleep(session.EVENT_TURN_QUIET_SECS + session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed, "hung up while committed utterance was awaiting turn_detector.on_committed"

        # Accepted speech cancels the hang-up.
        release.set()
        await _until(lambda: "Dime." in _texts(events))
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed

        # Now request end_call again, and send an utterance that the detector ignores as noise.
        entered.clear()
        release.clear()
        session.end_call()
        await stt.inject(TranscriptEvent(type="partial", text="noise"))
        await stt.inject(TranscriptEvent(type="committed", text="noise"))
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not session.closed, "hung up before detector finished evaluating noise"

        release.set()
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_end_call_in_a_later_turn_after_barge_in_hangs_up(self) -> None:
        tracker = FakePlaybackTracker()
        model = _Model(
            Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="end_call", input={})],
                stop_reason="tool_use",
            ),
            GOODBYE,
            Message(
                role="assistant",
                content=[ToolUseContent(id="c2", name="end_call", input={})],
                stop_reason="tool_use",
            ),
            "Ahora sí, adiós.",
        )
        session, stt, _tts = _session(model, tools=[end_call], tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        # Caller speaks again and immediately asks to end the call before the first task's sleep wakes.
        await stt.inject(TranscriptEvent(type="committed", text="Nada, cuelga ya."))
        await _until(lambda: "Ahora sí, adiós." in _texts(events))
        tracker.playing = False
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_hangs_up_anyway_if_the_goodbye_never_finishes(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_hangs_up_with(GOODBYE), tools=[end_call], tracker=tracker)
        session.END_CALL_MAX_WAIT_SECS = 0.3
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_after_the_call_ended_it_does_nothing(self) -> None:
        session, _stt, _tts = _session(_Model())
        await session.close()
        session.end_call()
        assert session._end_call_task is None

    async def test_grace_off_hangs_up_as_soon_as_the_goodbye_is_heard(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.wait_for(call, timeout=session.END_CALL_TAIL_SECS + 1.2)
        assert session.closed

    async def test_grace_silence_then_hangup(self) -> None:
        tracker = FakePlaybackTracker()
        grace = 0.45
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=grace,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        # Past the playback tail, still inside the confirmation window.
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.15)
        assert not session.closed, "hung up before hangup_grace_secs elapsed"
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_commit_during_grace_cancels_hangup(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE, "Dime."),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.0,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.2)
        assert not session.closed, "closed before the retraction could land"
        await stt.inject(TranscriptEvent(type="committed", text="espera"))
        await _until(lambda: "Dime." in _texts(events))
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.4)
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_partial_during_grace_cancels_without_a_phrase_list(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.2,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Goodbye."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.35)
        assert not session.closed
        # Not a configured retraction word — any non-echo speech cancels.
        await stt.inject(TranscriptEvent(type="partial", text="hold on"))
        await _until(lambda: session._end_call_task is None)
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_echo_of_the_goodbye_during_grace_still_hangs_up(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.4,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.1)
        await stt.inject(TranscriptEvent(type="partial", text="hasta luego"))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_vad_speech_during_grace_cancels_even_after_it_stops(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.2,
        )
        endpointer = _FakeEndpointer(speech_secs=0.0)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        session._endpointer = endpointer  # type: ignore[assignment]

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.35)
        assert not session.closed
        endpointer.speech_secs = 0.4
        await _until(lambda: session._end_call_task is None)
        assert not session.closed
        endpointer.speech_secs = 0.0
        await asyncio.sleep(0.3)
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_tool_grace_overrides_a_session_that_hangs_up_immediately(self) -> None:
        async def end_call_later() -> str:
            session = current_voice_session()
            assert session is not None
            session.end_call(grace_secs=0.8)
            return "ok"

        tracker = FakePlaybackTracker()
        model = _Model(
            Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="end_call_later", input={})],
                stop_reason="tool_use",
            ),
            GOODBYE,
        )
        session, stt, _tts = _session(model, tools=[end_call_later], tracker=tracker)
        assert session.hangup_grace_secs == 0
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.25)
        assert not session.closed
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_tool_can_hang_up_immediately_when_the_session_waits(self) -> None:
        async def end_call_now() -> str:
            session = current_voice_session()
            assert session is not None
            session.end_call(grace_secs=0)
            return "ok"

        tracker = FakePlaybackTracker()
        model = _Model(
            Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="end_call_now", input={})],
                stop_reason="tool_use",
            ),
            GOODBYE,
        )
        session, stt, _tts = _session(
            model,
            tools=[end_call_now],
            tracker=tracker,
            hangup_grace_secs=2.0,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.wait_for(call, timeout=session.END_CALL_TAIL_SECS + 1.2)
        assert session.closed

    async def test_vad_in_the_unanchored_lookback_does_not_abort_grace(self) -> None:
        """Energy the 0.5s lookback already holds must not skip the window.

        A short follow-up only fills the anchored window once it has been open
        for a moment. Aborting on the first unanchored sample used to drop it.
        """
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        session._end_call_grace_secs = 1.0

        class LengthGated:
            def speech_secs_in_window(self, window_secs: float) -> float:
                return 0.2 if window_secs >= 0.12 else 0.0

        session._endpointer = LengthGated()  # type: ignore[assignment]
        assert await session._end_call_grace(0.0)
        assert not session.closed

    async def test_pre_grace_vad_still_waits_out_the_window(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.3)
        session._end_call_grace_secs = 0.3

        class OnlyUnanchored:
            def speech_secs_in_window(self, window_secs: float) -> float:
                return 0.4 if window_secs >= 0.45 else 0.0

        session._endpointer = OnlyUnanchored()  # type: ignore[assignment]
        started = time.monotonic()
        assert not await session._end_call_grace(0.0)
        assert time.monotonic() - started >= 0.25
        assert not session.closed

    def _arm_grace(self, session) -> None:
        session._end_call_grace_secs = session.hangup_grace_secs

    def test_vad_during_the_playback_tail_is_not_a_retraction(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        self._arm_grace(session)
        session._endpointer = _FakeEndpointer(speech_secs=0.4)  # type: ignore[assignment]
        # 0.2s into the 0.4s tail: this energy is leftover goodbye, not a new turn.
        assert not session._cancel_hangup_for_speech(0.0, time.monotonic() - 0.2)

    def test_vad_that_continues_past_the_tail_cancels(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        self._arm_grace(session)
        session._endpointer = _FakeEndpointer(speech_secs=0.4)  # type: ignore[assignment]
        since = time.monotonic() - (session.END_CALL_TAIL_SECS + 0.2)
        assert session._cancel_hangup_for_speech(0.0, since)

    def test_echo_that_aged_out_of_the_tail_does_not_cancel(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        self._arm_grace(session)

        class AgedOut:
            def speech_secs_in_window(self, window_secs: float) -> float:
                # Still inside a 0.5s lookback, gone from a window that starts at the tail.
                return 0.4 if window_secs >= 0.35 else 0.0

        session._endpointer = AgedOut()  # type: ignore[assignment]
        since = time.monotonic() - (session.END_CALL_TAIL_SECS + 0.05)
        assert not session._cancel_hangup_for_speech(0.0, since)

    def test_a_partial_during_the_tail_still_cancels(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        self._arm_grace(session)
        session._transcript.append(TranscriptEntry(role="assistant", text=GOODBYE))
        session._endpointer = _FakeEndpointer(speech_secs=0.4)  # type: ignore[assignment]
        session._latest_partial_text = "hold on"
        session._last_partial_at = time.monotonic()
        assert session._cancel_hangup_for_speech(0.0, time.monotonic() - 0.2)
        session._latest_partial_text = "hasta luego"
        session._last_partial_at = time.monotonic()
        assert not session._cancel_hangup_for_speech(0.0, time.monotonic() - 0.2)

    async def test_vad_after_the_goodbye_cancels_without_a_commit(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.2,
        )

        class LengthGated:
            def speech_secs_in_window(self, window_secs: float) -> float:
                return 0.25 if window_secs >= 0.15 else 0.0

            def push(self, _chunk: bytes) -> None:
                pass

            def notify_committed(self) -> None:
                pass

            async def close(self) -> None:
                pass

        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        session._endpointer = LengthGated()  # type: ignore[assignment]
        await _until(lambda: session._end_call_task is None)
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    def test_bad_grace_does_not_start_a_hangup(self) -> None:
        session, _stt, _tts = _session(_Model())
        with pytest.raises(ValueError):
            session.end_call(grace_secs=-1)
        assert session._end_call_task is None
        with pytest.raises((TypeError, ValueError)):
            session.end_call(grace_secs=True)  # type: ignore[arg-type]
        assert session._end_call_task is None


class TestCommitHangup:
    async def test_caller_speech_skips_the_tool_side_effect(self) -> None:
        committed: list[bool] = []

        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            ok = await session.commit_hangup()
            committed.append(ok)
            return "ended" if ok else "cancelled"

        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE, "Dime."),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.2,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Cuelga la llamada."))
        await asyncio.sleep(0.15)
        # A new commit interrupts the turn that is still inside commit_hangup.
        # The handler must not reach the hangup, either because the wait
        # returns false or because the interruption cancels it first.
        await stt.inject(TranscriptEvent(type="committed", text="No, espera, todavía necesito ayuda."))
        tracker.playing = False
        await asyncio.sleep(0.4)
        assert committed == [] or committed == [False]
        assert session._end_call_task is None
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_quiet_window_then_requests_hangup(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.2)
        session._last_commit_at = time.monotonic()
        assert await session.commit_hangup() is True
        assert session._end_call_task is not None

    async def test_repeated_alright_does_not_cancel_the_grace_window(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.35)
        session._active_turn_user_text = "Adiós."

        async def again() -> None:
            await asyncio.sleep(0.05)
            session._latest_partial_text = "Alright. Alright."
            session._last_partial_at = time.monotonic()

        session._last_commit_at = time.monotonic()
        asyncio.create_task(again())
        assert await session.commit_hangup() is True
        assert session._end_call_task is not None

    @pytest.mark.parametrize("text", ["Thank you so much.", "Vamos, vámonos.", "Gracias."])
    async def test_thanks_and_vamos_do_not_start_a_hangup(self, text: str) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.2)
        session._active_turn_user_text = text
        session._last_commit_at = time.monotonic()
        assert await session.commit_hangup() is False
        assert session._end_call_task is None

    async def test_a_later_commit_does_not_request_hangup(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        session._last_commit_at = time.monotonic()

        async def retract() -> None:
            await asyncio.sleep(0.05)
            session._last_commit_at = time.monotonic()

        asyncio.create_task(retract())
        assert await session.commit_hangup() is False
        assert session._end_call_task is None


def _tool_only(name: str = "end_call", call_id: str = "c1") -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id, name=name, input={})],
        stop_reason="tool_use",
    )


_HANGUP_FILLER = {
    "enabled": True,
    "soft_timeout_secs": 0.05,
    "message": "One moment.",
    "delay_secs": 0.0,
}


class TestHangupUtterance:
    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            ("Alright.", "ack"),
            ("Okay.", "ack"),
            ("Yeah.", "ack"),
            ("Vale.", "ack"),
            ("Bye.", "goodbye"),
            ("Goodbye.", "goodbye"),
            ("Adiós.", "goodbye"),
            ("Never mind. Bye.", "goodbye"),
            ("Please hang up.", "goodbye"),
            ("Cuelga la llamada.", "goodbye"),
            ("Wait. One more thing.", "retract"),
            ("Espera, una cosa más.", "retract"),
            ("No, espera, todavía necesito ayuda.", "retract"),
            ("What's my order?", "retract"),
            ("Alright. Alright.", "ack"),
            ("Alright. Alright. Alright.", "ack"),
            ("Yeah. Okay.", "ack"),
            ("Thank you.", "ack"),
            ("Thank you so much.", "retract"),
            ("Vamos, vámonos.", "retract"),
            ("No, I meant how antibiotics work.", "retract"),
            ("Vale, gracias. Adiós.", "goodbye"),
            ("Ok, thanks, bye.", "goodbye"),
            ("Thank you. Goodbye.", "goodbye"),
            ("Vale, gracias.", "ack"),
            ("Gracias, espera. Adiós.", "retract"),
            ("Thank you so much, bye.", "retract"),
        ],
    )
    def test_kind(self, text: str, kind: str) -> None:
        assert _hangup_utterance_kind(text) == kind

    def test_farewell_follows_the_configured_language(self) -> None:
        assert _hangup_farewell_phrase("en") == "Goodbye."
        assert _hangup_farewell_phrase("es-ES") == "Adiós."
        assert _hangup_farewell_phrase(None) == "Goodbye."

    @pytest.mark.parametrize("text", ["Al", "All", "Alri", "Oka", "By", "B", "va", "ye"])
    def test_unfinished_ack_partial_is_pending(self, text: str) -> None:
        assert _hangup_utterance_kind(text) == "pending"

    @pytest.mark.parametrize(
        "text",
        ["Wa", "Wait", "Act", "Actually", "one", "whats the status", "Alright, wait", "no"],
    )
    def test_retract_partial_is_not_pending(self, text: str) -> None:
        assert _hangup_utterance_kind(text) == "retract"


class TestHangupFillerAndRearm:
    async def test_ack_partial_does_not_cancel_commit(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.35)

        async def ack() -> None:
            await asyncio.sleep(0.05)
            session._latest_partial_text = "Alright."
            session._last_partial_at = time.monotonic()

        session._last_commit_at = time.monotonic()
        asyncio.create_task(ack())
        assert await session.commit_hangup() is True
        assert session._end_call_task is not None

    async def test_wait_partial_cancels_commit(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)

        async def retract() -> None:
            await asyncio.sleep(0.05)
            session._latest_partial_text = "Wait. One more thing."
            session._last_partial_at = time.monotonic()

        session._last_commit_at = time.monotonic()
        asyncio.create_task(retract())
        assert await session.commit_hangup() is False
        assert session._end_call_task is None
        assert session._hangup_retracted

    async def test_no_filler_and_framework_farewell_when_the_model_skips_text(self) -> None:
        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "The caller spoke again. Do not hang up. Keep helping."
            return "Ending the call."

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only()),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.2,
            filler=_HANGUP_FILLER,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert "Goodbye." in tts.synthesized_texts
        assert "One moment." not in tts.synthesized_texts
        assert not any(entry.filler for entry in session.transcript)

    async def test_custom_hangup_tool_name_also_skips_filler(self) -> None:
        async def finish_call() -> str:
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "cancelled"
            return "ended"

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only("finish_call")),
            tools=[finish_call],
            tracker=tracker,
            hangup_grace_secs=0.25,
            filler={**_HANGUP_FILLER, "soft_timeout_secs": 0.08},
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert "One moment." not in tts.synthesized_texts
        assert "Goodbye." in tts.synthesized_texts

    async def test_model_goodbye_before_the_tool_is_left_alone(self) -> None:
        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            assert await session.commit_hangup()
            return "Ending the call."

        spoken = Message(
            role="assistant",
            content=[
                TextContent(text="Talk soon."),
                ToolUseContent(id="c1", name="end_call", input={}),
            ],
            stop_reason="tool_use",
        )
        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(spoken),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.15,
            filler=_HANGUP_FILLER,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert "Talk soon." in tts.synthesized_texts
        assert "Goodbye." not in tts.synthesized_texts
        assert "One moment." not in tts.synthesized_texts

    async def test_spanish_farewell(self) -> None:
        session, _stt, tts = _session(_Model(), hangup_grace_secs=0.05)
        session.audio_input.language = "es"
        session._last_commit_at = time.monotonic()
        assert await session.commit_hangup() is True
        assert "Adiós." in tts.synthesized_texts

    async def test_ack_during_hangup_still_drops_without_another_model_turn(self) -> None:
        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "The caller spoke again. Do not hang up. Keep helping."
            return "Ending the call."

        model = _Model(_tool_only(), "Goodbye.")
        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            model,
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.45,
            filler=_HANGUP_FILLER,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await _until(lambda: session._hangup_open)
        await stt.inject(TranscriptEvent(type="committed", text="Alright."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert "One moment." not in tts.synthesized_texts
        assert "Goodbye." in tts.synthesized_texts
        last_users = [m.collect_text() for call_messages in model.calls for m in call_messages if m.role == "user"]
        assert all("Alright" not in text for text in last_users)

    async def test_cancel_then_retract_stays_up_and_a_later_goodbye_rearms(self) -> None:
        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "The caller spoke again. Do not hang up. Keep helping."
            return "Ending the call."

        model = _Model(_tool_only(), "Sure, what else?", "It is packed.", "Goodbye.")
        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            model,
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.8,
            filler=_HANGUP_FILLER,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await _until(lambda: session._hangup_open)
        await stt.inject(TranscriptEvent(type="committed", text="Wait. One more thing."))
        await _until(lambda: "Sure, what else?" in _texts(events))
        await asyncio.sleep(0.3)
        assert not session.closed
        assert session._end_call_task is None or session._end_call_task.done()
        assert "One moment." not in tts.synthesized_texts

        await stt.inject(TranscriptEvent(type="committed", text="What's my order?"))
        await _until(lambda: "It is packed." in _texts(events))
        await asyncio.sleep(0.2)
        assert not session.closed

        await stt.inject(TranscriptEvent(type="committed", text="Bye."))
        await _until(lambda: "Goodbye." in _texts(events))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed

    async def test_evolving_ack_partials_do_not_cancel(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.45)
        session._last_commit_at = time.monotonic()

        async def partials() -> None:
            await asyncio.sleep(0.02)
            for piece in ("Al", "All", "Alri", "Alright.", "Oka", "Okay.", "By", "Bye bye"):
                session._latest_partial_text = piece
                session._last_partial_at = time.monotonic()
                await asyncio.sleep(0.03)

        asyncio.create_task(partials())
        assert await session.commit_hangup() is True
        assert session._end_call_task is not None

    async def test_ack_partial_that_turns_into_a_retract_cancels(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.8)
        session._last_commit_at = time.monotonic()
        started = time.monotonic()

        async def partials() -> None:
            await asyncio.sleep(0.02)
            for piece in ("Al", "Alright", "Alright, wait, one more thing"):
                session._latest_partial_text = piece
                session._last_partial_at = time.monotonic()
                await asyncio.sleep(0.03)

        asyncio.create_task(partials())
        assert await session.commit_hangup() is False
        assert time.monotonic() - started < 0.5
        assert session._end_call_task is None

    async def test_retract_partial_cancels_before_the_commit(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=1.0)
        session._last_commit_at = time.monotonic()
        started = time.monotonic()

        async def partials() -> None:
            await asyncio.sleep(0.04)
            session._latest_partial_text = "Act"
            session._last_partial_at = time.monotonic()

        asyncio.create_task(partials())
        assert await session.commit_hangup() is False
        assert time.monotonic() - started < 0.4

    async def test_post_tool_goodbye_is_not_spoken_again(self) -> None:
        class _SlowAfterFirst(MockTTS):
            async def synthesize(self, text: str):
                self.synthesized_texts.append(text)
                if len(self.synthesized_texts) > 1:
                    await asyncio.sleep(3.0)
                yield self._chunk

        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "The caller spoke again. Do not hang up. Keep helping."
            return "Ending the call."

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only(), "Goodbye."),
            tools=[end_call],
            tts=_SlowAfterFirst(),
            tracker=tracker,
            hangup_grace_secs=0.12,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await asyncio.wait_for(call, timeout=1.5)
        assert session.closed
        assert tts.synthesized_texts.count("Goodbye.") == 1
        assert sum(1 for entry in session.transcript if entry.role == "assistant" and "Goodbye" in entry.text) == 1

    async def test_later_goodbye_rearms_without_calling_the_tool_again(self) -> None:
        used = 0

        async def end_call() -> str:
            nonlocal used
            used += 1
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "The caller spoke again. Do not hang up. Keep helping."
            return "Ending the call."

        # The bye turn is text only. The tool must not run again.
        model = _Model(_tool_only(), "Sure, what else?", "Goodbye.")
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            model,
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.35,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await _until(lambda: session._hangup_open)
        await stt.inject(TranscriptEvent(type="committed", text="Wait. One more thing."))
        await _until(lambda: "Sure, what else?" in _texts(events))
        assert used == 1
        assert not session.closed

        await stt.inject(TranscriptEvent(type="committed", text="Bye."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert used == 1
        bye_users = [m.collect_text() for m in model.calls[-1] if m.role == "user"]
        assert any("Bye" in text for text in bye_users)


class TestHangupEventSequences:
    """Provider-shaped sequences. Partials arrive on the STT queue, not by poking fields.

    The 870c51aa tests that Albert's Flux run falsified only ever applied the
    finished word "Alright." (or a single already-complete partial). TestModel
    also returns a whole message, so a second "Goodbye." was either absent or
    only checked with ``in``, which stays true when the line is spoken twice.
    """

    async def test_flux_partials_during_grace_then_alright_commit_still_hangs_up(self) -> None:
        results: list[bool] = []

        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            ok = await session.commit_hangup()
            results.append(ok)
            return "Ending the call." if ok else "The caller spoke again. Do not hang up. Keep helping."

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only(), "Still here."),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.2,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        started = time.monotonic()
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await _until(lambda: "Goodbye." in tts.synthesized_texts and session._hangup_open)
        # Past the 0.4s trailing-crumb window, still inside the 1.2s grace.
        await asyncio.sleep(0.55)
        for piece in ("Al", "All", "Alri", "Alright"):
            await stt.inject(TranscriptEvent(type="partial", text=piece))
            await asyncio.sleep(0.05)
        await stt.inject(TranscriptEvent(type="committed", text="Alright."))
        await asyncio.wait_for(call, timeout=3.0)
        elapsed = time.monotonic() - started
        assert results == [True], results
        assert session.closed
        assert not session._hangup_retracted
        assert tts.synthesized_texts.count("Goodbye.") == 1
        assert "Still here." not in tts.synthesized_texts
        assert elapsed < 3.0

    async def test_ack_partials_during_farewell_playback_do_not_cancel(self) -> None:
        results: list[bool] = []

        class _HoldDrain(MockTTS):
            async def synthesize(self, text: str):
                self.synthesized_texts.append(text)
                tracker.playing = True
                yield self._chunk

        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            ok = await session.commit_hangup()
            results.append(ok)
            return "Ending the call." if ok else "cancelled"

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only()),
            tools=[end_call],
            tts=_HoldDrain(),
            tracker=tracker,
            hangup_grace_secs=0.3,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await _until(lambda: tracker.playing and session._hangup_open)
        for piece in ("Al", "All", "Alri"):
            await stt.inject(TranscriptEvent(type="partial", text=piece))
            await asyncio.sleep(0.04)
        await asyncio.sleep(0.15)
        assert results == []
        assert not session._hangup_retracted
        tracker.playing = False
        await asyncio.wait_for(call, timeout=3.0)
        assert results == [True]
        assert session.closed
        assert tts.synthesized_texts == ["Goodbye."]

    async def test_retract_partial_during_farewell_drain_cancels_immediately(self) -> None:
        results: list[bool] = []

        class _HoldDrain(MockTTS):
            async def synthesize(self, text: str):
                self.synthesized_texts.append(text)
                tracker.playing = True
                yield self._chunk

        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            ok = await session.commit_hangup()
            results.append(ok)
            return "Ending the call." if ok else "The caller spoke again. Do not hang up. Keep helping."

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only(), "Sure, go ahead."),
            tools=[end_call],
            tts=_HoldDrain(),
            tracker=tracker,
            hangup_grace_secs=1.0,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await _until(lambda: tracker.playing and session._hangup_open)
        marked = time.monotonic()
        await stt.inject(TranscriptEvent(type="partial", text="Wait"))
        await _until(lambda: results == [False] or "Sure, go ahead." in _texts(events))
        assert time.monotonic() - marked < 0.5
        assert results == [False]
        tracker.playing = False
        await _until(lambda: "Sure, go ahead." in _texts(events))
        assert not session.closed
        assert "Goodbye." in tts.synthesized_texts
        assert tts.synthesized_texts.count("Goodbye.") == 1
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_grace_expires_on_an_unfinished_ack_partial(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.28)
        session._last_commit_at = time.monotonic()
        started = time.monotonic()

        async def hold_prefix() -> None:
            while time.monotonic() - started < 0.45:
                session._latest_partial_text = "Al"
                session._last_partial_at = time.monotonic()
                await asyncio.sleep(0.02)

        asyncio.create_task(hold_prefix())
        assert await session.commit_hangup() is True
        elapsed = time.monotonic() - started
        assert 0.2 < elapsed < 0.7

    async def test_retract_in_the_last_moment_of_grace_still_cancels(self) -> None:
        session, _stt, _tts = _session(_Model(), hangup_grace_secs=0.3)
        session._last_commit_at = time.monotonic()
        started = time.monotonic()

        async def late_retract() -> None:
            await asyncio.sleep(0.22)
            session._latest_partial_text = "Wait"
            session._last_partial_at = time.monotonic()

        asyncio.create_task(late_retract())
        assert await session.commit_hangup() is False
        elapsed = time.monotonic() - started
        assert elapsed < 0.4

    async def test_streamed_second_goodbye_does_not_play_or_delay_disconnect(self) -> None:
        from timbal.types.events.delta import DeltaEvent, TextDelta
        from timbal.types.events.output import OutputEvent

        async def end_call() -> str:
            session = current_voice_session()
            assert session is not None
            if not await session.commit_hangup():
                return "The caller spoke again. Do not hang up. Keep helping."
            return "Ending the call."

        tracker = FakePlaybackTracker()
        session, stt, tts = _session(
            _Model(_tool_only(), "Goodbye."),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=0.12,
        )
        inner = session.agent

        class _StreamLateGoodbye:
            def __getattr__(self, name: str):
                return getattr(inner, name)

            def __call__(self, **kwargs):
                source = inner(**kwargs)

                async def events():
                    saw_tool = False
                    async with aclosing(source):
                        async for event in source:
                            content = getattr(getattr(event, "output", None), "content", None)
                            is_tool = bool(content) and any(isinstance(block, ToolUseContent) for block in content)
                            if is_tool:
                                saw_tool = True
                            text = ""
                            if (
                                saw_tool
                                and not is_tool
                                and isinstance(event, OutputEvent)
                                and getattr(event, "output", None) is not None
                                and hasattr(event.output, "collect_text")
                            ):
                                text = event.output.collect_text() or ""
                            if text.strip():
                                await asyncio.sleep(2.0)
                                for piece in (text[:4], text[4:]):
                                    if piece:
                                        yield DeltaEvent(
                                            run_id=event.run_id,
                                            path=event.path,
                                            call_id=event.call_id,
                                            item=TextDelta(id="g", text_delta=piece),
                                        )
                            yield event

                return events()

        session.agent = _StreamLateGoodbye()
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        started = time.monotonic()
        await stt.inject(TranscriptEvent(type="committed", text="Please hang up."))
        await asyncio.wait_for(call, timeout=1.8)
        elapsed = time.monotonic() - started
        assert session.closed
        assert elapsed < 1.5, elapsed
        assert tts.synthesized_texts.count("Goodbye.") == 1

    async def test_bare_end_call_still_cancels_when_the_caller_says_alright(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(
            _hangs_up_with(GOODBYE, "Dime."),
            tools=[end_call],
            tracker=tracker,
            hangup_grace_secs=1.0,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: GOODBYE in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.25)
        assert not session.closed
        await stt.inject(TranscriptEvent(type="committed", text="Alright."))
        await _until(lambda: "Dime." in _texts(events))
        await asyncio.sleep(0.4)
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)


async def _committing_end_call() -> str:
    session = current_voice_session()
    assert session is not None
    if not await session.commit_hangup():
        return "Do not hang up. Keep helping."
    return "Ending the call."


def _tool_results(messages: list[Message]) -> dict[str, str]:
    return {
        block.id: " ".join(c.text for c in block.content if isinstance(c, TextContent))
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultContent)
    }


def _stream_tool_calls_under_item_ids(session) -> None:
    """Announce each tool call as a ToolUse delta under another id, before its final message.

    OpenAI Responses streams ``fc_…`` and finishes with ``call_…``. Anthropic streams
    a block index. TestModel streams nothing, so it never showed the mismatch.
    """
    from timbal.types.events.delta import DeltaEvent, ToolUse
    from timbal.types.events.output import OutputEvent

    inner = session.agent

    class _Streaming:
        def __getattr__(self, name: str):
            return getattr(inner, name)

        def __call__(self, **kwargs):
            source = inner(**kwargs)

            async def events():
                async with aclosing(source):
                    async for event in source:
                        if isinstance(event, OutputEvent) and event.path.endswith(".llm"):
                            for block in getattr(event.output, "content", None) or []:
                                if isinstance(block, ToolUseContent):
                                    yield DeltaEvent(
                                        run_id=event.run_id,
                                        path=event.path,
                                        call_id=event.call_id,
                                        item=ToolUse(id=f"fc_{block.id}", name=block.name, input=""),
                                    )
                        yield event

            return events()

    session.agent = _Streaming()


class TestCommittedHangupEndsTheTurn:
    """Once ``commit_hangup`` is True the model is not asked for another step."""

    async def test_a_call_streamed_under_another_id_is_counted_once(self) -> None:
        model = _Model(_tool_only(), "Goodbye.")
        session, stt, _tts = _session(
            model, tools=[Tool(name="end_call", handler=_committing_end_call)], hangup_grace_secs=0.05
        )
        _stream_tool_calls_under_item_ids(session)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert len(model.calls) == 1
        assert session._turn_tool_batch_inflight == 0

    async def test_no_llm_call_after_the_hangup_tool_returns(self) -> None:
        model = _Model(_tool_only(), "Goodbye.")
        session, stt, tts = _session(
            model, tools=[Tool(name="end_call", handler=_committing_end_call)], hangup_grace_secs=0.05
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert len(model.calls) == 1
        assert tts.synthesized_texts.count("Goodbye.") == 1

    async def test_a_refused_hangup_still_asks_the_model(self) -> None:
        model = _Model(_tool_only(), "Claro, dígame.")
        session, stt, _tts = _session(
            model, tools=[Tool(name="end_call", handler=_committing_end_call)], hangup_grace_secs=0.05
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Vamos, vámonos."))
        await _until(lambda: "Claro, dígame." in _texts(events))
        assert not session.closed
        assert len(model.calls) == 2
        assert _tool_results(model.calls[1]) == {"c1": "Do not hang up. Keep helping."}
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_a_parallel_tool_finishes_before_the_turn_stops(self) -> None:
        saved: list[str] = []

        async def apply_pending() -> str:
            await asyncio.sleep(0.25)
            saved.append("ticket")
            return "Saved."

        both = Message(
            role="assistant",
            content=[
                ToolUseContent(id="c1", name="end_call", input={}),
                ToolUseContent(id="c2", name="apply_pending", input={}),
            ],
            stop_reason="tool_use",
        )
        model = _Model(both, "Goodbye.")
        session, stt, _tts = _session(
            model,
            tools=[Tool(name="end_call", handler=_committing_end_call), apply_pending],
            hangup_grace_secs=0.05,
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Adiós."))
        await asyncio.wait_for(call, timeout=3.0)
        assert session.closed
        assert saved == ["ticket"]
        assert len(model.calls) == 1

    async def test_a_retract_after_the_commit_continues_from_the_tool_result(self) -> None:
        model = _Model(_tool_only(), "Sure, what else?")
        session, stt, _tts = _session(
            model, tools=[Tool(name="end_call", handler=_committing_end_call)], hangup_grace_secs=0.05
        )
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        await stt.inject(TranscriptEvent(type="committed", text="Bye."))
        await _until(lambda: session._end_call_task is not None)
        await stt.inject(TranscriptEvent(type="committed", text="Wait. One more thing."))
        await _until(lambda: "Sure, what else?" in _texts(events))
        await asyncio.sleep(0.6)
        assert not session.closed
        assert len(model.calls) == 2
        history = model.calls[1]
        assert _tool_results(history) == {"c1": "Ending the call."}
        assert history[-1].role == "user" and "One more thing" in history[-1].collect_text()
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)
