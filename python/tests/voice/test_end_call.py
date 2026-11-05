"""``VoiceSession.end_call`` — the agent hangs up after its goodbye has been heard."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from timbal.types.content import ToolUseContent
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

from .test_session import FakePlaybackTracker, _FakeEndpointer
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
