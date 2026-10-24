"""``VoiceSession.end_call`` — the agent hangs up after its goodbye has been heard."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import SessionStarted, TranscriptEvent, VoiceSessionEvent, current_voice_session

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
