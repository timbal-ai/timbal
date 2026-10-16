"""``VoiceSession.end_call`` — the agent hangs up after its goodbye has been heard."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import SessionStarted, TranscriptEvent, VoiceSessionEvent, current_voice_session

from .test_session import FakePlaybackTracker
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
