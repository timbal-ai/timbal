"""``VoiceSession.notify`` — event-initiated agent turns.

The agent speaks up because something happened (background work finished),
not because the caller said something. Driven through ``session.run()`` with
the scripted STT/TTS fakes from ``test_session``; the model records what it
was asked so each test can check what the note looked like to it.
"""

# ruff: noqa: ARG001  (tool and driver signatures are fixed by their callers)
from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers import JsonlTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    AgentTextDone,
    AudioOutput,
    FillerConfig,
    FillerSpoken,
    SessionInterrupted,
    SessionStarted,
    TextToSpeech,
    TranscriptCommitted,
    TranscriptEvent,
    VoiceSession,
    VoiceSessionEvent,
    current_voice_session,
)
from timbal.voice.playback import PlaybackTracker
from timbal.voice.session import VOICE_EVENT_KIND

from .test_session import DelayedMockSTT, FakePlaybackTracker, MockTTS, SlowMockTTS, _FakeEndpointer

NOTE = "[event] Background task 'inversiones' finished: the fund is up 4% this quarter."
REPLY = "Ya tengo lo de las inversiones, ¿te lo resumo?"


class _Model:
    """Scripted replies, one per LLM call; keeps every call's input messages."""

    def __init__(self, *replies: str | Message) -> None:
        self.calls: list[list[Message]] = []
        self._replies = list(replies)
        self.model = TestModel(handler=self._reply)

    def _reply(self, messages: list[Message]) -> str | Message:
        self.calls.append(list(messages))
        return self._replies.pop(0) if self._replies else ""


def _session(
    model: _Model,
    *,
    tts: TextToSpeech | None = None,
    tracker: PlaybackTracker | None = None,
    tools: list | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[VoiceSession, DelayedMockSTT, TextToSpeech]:
    agent = Agent(name="voice_notify", model=model.model, tools=tools or [], **(agent_kwargs or {}))
    stt = DelayedMockSTT()
    tts = tts or MockTTS()
    session = VoiceSession(agent=agent, stt=stt, tts=tts, turn_detector="heuristic", playback_tracker=tracker, **kwargs)
    return session, stt, tts


async def _run(
    session: VoiceSession,
    stt: DelayedMockSTT,
    drive: Callable[[list[VoiceSessionEvent]], Awaitable[None]],
    *,
    timeout: float = 10.0,
) -> list[VoiceSessionEvent]:
    """Run ``session`` until ``drive`` returns, then end the STT stream."""
    events: list[VoiceSessionEvent] = []

    async def _no_audio() -> AsyncIterator[bytes]:
        return
        yield  # noqa: RET504 — make it an async generator

    async def _driver() -> None:
        await _until(lambda: any(isinstance(e, SessionStarted) for e in events))
        await drive(events)
        await stt.finish()

    async def _go() -> None:
        async with aclosing(session.run(_no_audio())) as stream:
            driver = asyncio.create_task(_driver())
            async for event in stream:
                events.append(event)
            await driver

    await asyncio.wait_for(_go(), timeout=timeout)
    return events


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def _done(events: list[VoiceSessionEvent]) -> list[str]:
    return [e.text for e in events if isinstance(e, AgentTextDone)]


def _event_note(messages: list[Message]) -> Message:
    [note] = [m for m in messages if m.metadata.get("kind") == VOICE_EVENT_KIND]
    return note


class TestEventTurn:
    async def test_idle_notify_speaks_and_is_not_the_callers_words(self) -> None:
        model = _Model(REPLY)
        session, stt, tts = _session(model)
        results: list[bool] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            results.append(await session.notify(NOTE))
            await _until(lambda: REPLY in _done(events))

        events = await _run(session, stt, drive)

        assert results == [True]
        [call] = model.calls
        note = call[-1]
        assert note.role == "user"
        assert note.is_runtime()
        assert note.metadata["kind"] == VOICE_EVENT_KIND
        assert note.metadata["delivery"] == "own_turn"
        assert f"<voice_event>\n{NOTE}\n</voice_event>" in note.collect_text()
        # Spoken like any reply...
        assert tts.synthesized_texts == [REPLY]
        assert [(e.role, e.text) for e in session.transcript] == [("assistant", REPLY)]
        # ...but nothing on the wire or in the transcript says the caller spoke.
        assert not [e for e in events if isinstance(e, TranscriptCommitted)]
        assert session.metrics[-1].event is True
        assert session.metrics[-1].user_text_chars == 0

    async def test_later_turns_still_see_the_note(self) -> None:
        model = _Model(REPLY, "Claro: sube un 4%.")
        session, stt, _tts = _session(model)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await session.notify(NOTE)
            await _until(lambda: REPLY in _done(events))
            await stt.inject(TranscriptEvent(type="committed", text="Sí, resúmemelo."))
            await _until(lambda: len(_done(events)) == 2)

        await _run(session, stt, drive)

        second = model.calls[1]
        assert [m.role for m in second[-3:]] == ["user", "assistant", "user"]
        assert second[-3].is_runtime() and NOTE in second[-3].collect_text()
        assert second[-1].collect_text() == "Sí, resúmemelo."
        assert [e.role for e in session.transcript] == ["assistant", "user", "assistant"]

    async def test_nothing_worth_saying_speaks_nothing_and_no_filler(self) -> None:
        """A slow tool in a caller turn gets the filler; the same in an event
        turn that ends with no text says nothing at all."""

        async def lookup(q: str) -> str:
            await asyncio.sleep(0.3)
            return "42"

        def tool_call(call_id: str) -> Message:
            return Message(
                role="assistant",
                content=[ToolUseContent(id=call_id, name="lookup", input={"q": "x"})],
                stop_reason="tool_use",
            )

        model = _Model(tool_call("c1"), "Listo.", tool_call("c2"), "")
        filler = FillerConfig(phrases=["Un momento."], delay_secs=0.0, think_delay_secs=0.0)
        session, stt, tts = _session(model, tools=[lookup], filler=filler)
        fillers_before_event: list[int] = []
        spoken_before_event: list[list[str]] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="¿Cuánto es?"))
            await _until(lambda: "Listo." in _done(events))
            fillers_before_event.append(sum(isinstance(e, FillerSpoken) for e in events))
            spoken_before_event.append(list(tts.synthesized_texts))
            assert await session.notify(NOTE)
            await _until(lambda: len(_done(events)) == 2)

        events = await _run(session, stt, drive)

        assert fillers_before_event[0] >= 1  # control: this config does speak fillers
        assert sum(isinstance(e, FillerSpoken) for e in events) == fillers_before_event[0]
        assert tts.synthesized_texts == spoken_before_event[0]
        assert _done(events)[-1] == ""
        assert session.transcript[-1].text == "Listo."
        assert session.metrics[-1].event is True and session.metrics[-1].filler_count == 0


class TestWhenIdle:
    async def test_waits_for_playback_to_end_instead_of_interrupting(self) -> None:
        tracker = FakePlaybackTracker()
        model = _Model("Hola, dime.", REPLY)
        session, stt, _tts = _session(model, tracker=tracker)
        pending: list[asyncio.Task[bool]] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="Hola"))
            await _until(lambda: "Hola, dime." in _done(events))
            await _until(lambda: session._current_turn_task.done())
            tracker.playing = True  # the reply is still coming out of the speaker
            pending.append(asyncio.create_task(session.notify(NOTE)))
            await asyncio.sleep(0.3)
            assert not pending[0].done()
            assert len(model.calls) == 1
            tracker.playing = False
            assert await pending[0] is True
            await _until(lambda: REPLY in _done(events))

        events = await _run(session, stt, drive)

        assert not [e for e in events if isinstance(e, SessionInterrupted)]
        assert [(e.role, e.text) for e in session.transcript] == [
            ("user", "Hola"),
            ("assistant", "Hola, dime."),
            ("assistant", REPLY),
        ]

    async def test_waits_while_the_caller_is_speaking(self) -> None:
        model = _Model("Dime.", REPLY)
        session, stt, _tts = _session(model)
        pending: list[asyncio.Task[bool]] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="partial", text="Oye, una cosa"))
            await _until(lambda: session._last_partial_at > 0)
            pending.append(asyncio.create_task(session.notify(NOTE)))
            await asyncio.sleep(0.3)
            assert not pending[0].done()
            assert model.calls == []
            await stt.inject(TranscriptEvent(type="committed", text="Oye, una cosa que quería preguntarte"))
            assert await pending[0] is True
            await _until(lambda: REPLY in _done(events))

        events = await _run(session, stt, drive)

        first, second = model.calls
        assert first[-1].collect_text() == "Oye, una cosa que quería preguntarte"
        assert second[-1].is_runtime()
        assert _done(events) == ["Dime.", REPLY]
        assert not [e for e in events if isinstance(e, SessionInterrupted)]

    async def test_caller_can_barge_in_on_the_event_turn(self) -> None:
        model = _Model(REPLY + " Son buenas noticias para el trimestre.", "Vale, dime.")
        tts = SlowMockTTS(delay=0.05, chunk=b"\x00\x01" * 16_000, num_chunks=4)
        session, stt, _tts = _session(model, tts=tts)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            assert await session.notify(NOTE)
            await _until(lambda: any(isinstance(e, AudioOutput) for e in events))
            await stt.inject(TranscriptEvent(type="committed", text="Espera, antes quería preguntarte otra cosa"))
            await _until(lambda: "Vale, dime." in _done(events))

        events = await _run(session, stt, drive)

        assert any(isinstance(e, SessionInterrupted) for e in events)
        assert model.calls[1][-1].collect_text() == "Espera, antes quería preguntarte otra cosa"

    @pytest.mark.parametrize("provider", ["in_memory", "jsonl"])
    async def test_overdue_note_rides_the_callers_next_turn(self, provider: str, tmp_path: Path) -> None:
        """``jsonl`` reloads the next turn's history from the persisted dump,
        so the note must reach the file, not only the live memory list."""
        agent_kwargs: dict[str, Any] = {}
        if provider == "jsonl":
            agent_kwargs["tracing_provider"] = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        tracker = FakePlaybackTracker()
        model = _Model("Hola, dime.", "Nada más. Por cierto, lo de las inversiones ya está.")
        session, stt, _tts = _session(model, tracker=tracker, agent_kwargs=agent_kwargs)
        results: list[bool] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="Hola"))
            await _until(lambda: "Hola, dime." in _done(events))
            await _until(lambda: session._current_turn_task.done())
            tracker.playing = True
            results.append(await session.notify(NOTE, key="task:inversiones", max_wait_secs=0.2))
            tracker.playing = False
            await asyncio.sleep(0.2)
            assert len(model.calls) == 1  # deferred, not spoken on its own
            await stt.inject(TranscriptEvent(type="committed", text="¿Algo más?"))
            await _until(lambda: len(_done(events)) == 2)

        events = await _run(session, stt, drive)

        assert results == [False]
        second = model.calls[1]
        note, said = second[-2], second[-1]
        assert note.is_runtime() and NOTE in note.collect_text()
        assert note.metadata["delivery"] == "next_turn"
        assert note.metadata["keys"] == ["task:inversiones"]
        assert said.collect_text() == "¿Algo más?" and not said.is_runtime()
        assert [e.text for e in events if isinstance(e, TranscriptCommitted)] == ["Hola", "¿Algo más?"]
        assert all(NOTE not in e.text for e in session.transcript)

    async def test_overdue_note_before_any_turn_goes_to_that_turns_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str | None] = []
        original = Agent.__call__

        def spy(self: Agent, **kwargs: Any) -> Any:
            seen.append(kwargs.get("system_prompt"))
            return original(self, **kwargs)

        monkeypatch.setattr(Agent, "__call__", spy)
        tracker = FakePlaybackTracker()
        tracker.playing = True
        model = _Model("Hola.")
        session, stt, _tts = _session(model, tracker=tracker)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            assert await session.notify(NOTE, max_wait_secs=0.1) is False
            tracker.playing = False
            await stt.inject(TranscriptEvent(type="committed", text="Hola"))
            await _until(lambda: "Hola." in _done(events))

        await _run(session, stt, drive)

        assert seen and NOTE in (seen[0] or "")
        assert all(not m.is_runtime() for m in model.calls[0])


class TestPolicyAndCoalescing:
    async def test_now_interrupts_the_reply(self) -> None:
        model = _Model("Te cuento todo sobre el fondo de inversión, empezando por el principio.", REPLY)
        tts = SlowMockTTS(delay=0.05, chunk=b"\x00\x01" * 16_000, num_chunks=4)
        session, stt, _tts = _session(model, tts=tts)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="Cuéntame del fondo"))
            await _until(lambda: any(isinstance(e, AudioOutput) for e in events))
            assert await session.notify(NOTE, policy="now") is True
            await _until(lambda: REPLY in _done(events))

        events = await _run(session, stt, drive)

        interrupted = next(i for i, e in enumerate(events) if isinstance(e, SessionInterrupted))
        reply = next(i for i, e in enumerate(events) if isinstance(e, AgentTextDone) and e.text == REPLY)
        assert interrupted < reply
        assert model.calls[1][-1].is_runtime()

    async def test_same_key_replaces_the_waiting_note(self) -> None:
        tracker = FakePlaybackTracker()
        tracker.playing = True
        model = _Model(REPLY)
        session, stt, _tts = _session(model, tracker=tracker)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            first = asyncio.create_task(session.notify("Task 1: 50% done.", key="task:1"))
            await asyncio.sleep(0.05)
            second = asyncio.create_task(session.notify("Task 1: finished.", key="task:1"))
            assert await asyncio.wait_for(first, timeout=1.0) is False
            tracker.playing = False
            assert await second is True
            await _until(lambda: REPLY in _done(events))

        await _run(session, stt, drive)

        [call] = model.calls
        text = _event_note(call).collect_text()
        assert "Task 1: finished." in text
        assert "50% done" not in text

    async def test_waiting_notes_are_delivered_in_one_turn(self) -> None:
        tracker = FakePlaybackTracker()
        tracker.playing = True
        model = _Model(REPLY)
        session, stt, _tts = _session(model, tracker=tracker)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            waiting = [
                asyncio.create_task(session.notify("Task A finished.")),
                asyncio.create_task(session.notify("Task B finished.", key="b")),
            ]
            await asyncio.sleep(0.1)
            tracker.playing = False
            assert await asyncio.gather(*waiting) == [True, True]
            await _until(lambda: REPLY in _done(events))
            await asyncio.sleep(0.2)

        events = await _run(session, stt, drive)

        [call] = model.calls
        note = _event_note(call)
        text = note.collect_text()
        assert text.count("<voice_event>") == 2
        assert text.index("Task A finished.") < text.index("Task B finished.")
        assert note.metadata["keys"] == ["b"]
        assert _done(events) == [REPLY]


class TestLifecycle:
    async def test_a_note_sent_before_the_call_starts_waits_for_it(self) -> None:
        model = _Model(REPLY)
        session, stt, _tts = _session(model)
        early = asyncio.create_task(session.notify(NOTE))
        await asyncio.sleep(0.1)
        assert not early.done()

        async def drive(events: list[VoiceSessionEvent]) -> None:
            assert await early is True
            await _until(lambda: REPLY in _done(events))

        await _run(session, stt, drive)

        [call] = model.calls
        assert call[-1].is_runtime()

    async def test_close_settles_a_waiting_note(self) -> None:
        tracker = FakePlaybackTracker()
        tracker.playing = True
        model = _Model(REPLY)
        session, stt, _tts = _session(model, tracker=tracker)
        pending: list[asyncio.Task[bool]] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            pending.append(asyncio.create_task(session.notify(NOTE, max_wait_secs=None)))
            await asyncio.sleep(0.1)

        await _run(session, stt, drive)

        assert await asyncio.wait_for(pending[0], timeout=1.0) is False
        assert model.calls == []
        assert await session.notify(NOTE) is False

    async def test_rejects_an_unknown_policy(self) -> None:
        session, _stt, _tts = _session(_Model())
        with pytest.raises(ValueError, match="policy"):
            await session.notify(NOTE, policy="later")  # type: ignore[arg-type]

    async def test_tools_reach_the_live_session(self) -> None:
        seen: list[VoiceSession | None] = []

        async def where_am_i() -> str:
            seen.append(current_voice_session())
            return "ok"

        model = _Model(
            Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="where_am_i", input={})],
                stop_reason="tool_use",
            ),
            "Hecho.",
        )
        session, stt, _tts = _session(model, tools=[where_am_i])

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="Hazlo"))
            await _until(lambda: "Hecho." in _done(events))

        await _run(session, stt, drive)

        assert seen == [session]
        assert current_voice_session() is None


class TestBlocker:
    """What keeps an event turn from starting, checked without running a call."""

    def _started(self) -> VoiceSession:
        session, _stt, _tts = _session(_Model(), tracker=FakePlaybackTracker())
        session.started_at = time.time()
        session._session_context = contextvars.copy_context()
        return session

    def test_quiet_line_is_free(self) -> None:
        assert self._started()._event_turn_blocker() is None

    def test_not_started(self) -> None:
        session, _stt, _tts = _session(_Model())
        assert session._event_turn_blocker() == "not_started"

    def test_local_vad_hears_the_caller(self) -> None:
        session = self._started()
        session._endpointer = _FakeEndpointer(speech_secs=0.4)  # type: ignore[assignment]
        assert session._event_turn_blocker() == "user_speaking"
        session._endpointer.speech_secs = 0.0  # type: ignore[union-attr]
        assert session._event_turn_blocker() is None

    def test_held_fragment(self) -> None:
        session = self._started()
        session._held_user_text = "y luego"
        assert session._event_turn_blocker() == "user_holding"
        assert session._event_turn_blocker(urgent=True) is None

    def test_a_partial_left_stranded_stops_counting(self) -> None:
        session = self._started()
        session._last_partial_at = time.monotonic()
        assert session._event_turn_blocker() == "user_speaking"
        session._last_partial_at = time.monotonic() - session.NOTIFY_STRANDED_PARTIAL_SECS - 1
        assert session._event_turn_blocker() is None

    def test_run_parked_on_the_caller_blocks_even_now(self) -> None:
        session = self._started()
        root = SimpleNamespace(status=SimpleNamespace(code="cancelled", reason="input_required"))
        session._last_run_context = SimpleNamespace(root_span=lambda: root)  # type: ignore[assignment]
        assert session._event_turn_blocker() == "run_awaiting_caller"
        assert session._event_turn_blocker(urgent=True) == "run_awaiting_caller"
        root.status = SimpleNamespace(code="success", reason=None)
        assert session._event_turn_blocker() is None
