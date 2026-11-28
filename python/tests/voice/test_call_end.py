"""The end of a call, for the application: ``on_close``, ``wait_until_heard`` and ``transfer_call``."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from structlog.testing import capture_logs
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import SessionStarted, TranscriptEvent, VoiceSessionEvent, current_voice_session
from timbal.voice.events import TranscriptEntry

from .test_end_call import GOODBYE, _call, _hangs_up_with, end_call
from .test_session import FakePlaybackTracker
from .test_voice_events import _Model, _session, _texts, _until

HANDOFF = "Le paso con un compañero."
NUMBER = "+34930000000"


class _Line:
    """The transport's side of a transfer: what it was asked to dial, and whether it takes it."""

    def __init__(self, *, refuses: bool = False) -> None:
        self.dialled: list[str] = []
        self.refuses = refuses

    async def transfer(self, to: str) -> None:
        self.dialled.append(to)
        if self.refuses:
            raise RuntimeError("HTTP 422 number not allowed")


async def transfer() -> str:
    session = current_voice_session()
    assert session is not None
    session.transfer_call(NUMBER)
    return "ok"


def _transfers_with(reply: str, *later: str | Message) -> _Model:
    return _Model(
        Message(role="assistant", content=[ToolUseContent(id="c1", name="transfer", input={})], stop_reason="tool_use"),
        reply,
        *later,
    )


async def _turn_over(session, events: list[VoiceSessionEvent], reply: str) -> None:
    """The turn that said ``reply`` has ended. Until then a short commit is a crumb of the caller's last words."""
    await _until(lambda: reply in _texts(events) and not session._is_speaking and not session._active_turn_user_text)


def _outcome(session) -> asyncio.Future[str]:
    future = session._transfer_outcome
    assert future is not None, "no transfer is waiting"
    return future


class TestOnClose:
    async def test_runs_once_when_the_call_is_closed(self) -> None:
        session, _stt, _tts = _session(_Model())
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        seen: list[bool] = []
        session.on_close(lambda: seen.append(session.closed))

        await session.close()
        await session.close()
        await asyncio.wait_for(call, timeout=3.0)

        assert seen == [True]

    async def test_close_waits_for_an_async_callback(self) -> None:
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])
        done: list[str] = []

        async def save() -> None:
            await asyncio.sleep(0.05)
            done.append("saved")

        session.on_close(save)
        await session.close()

        assert done == ["saved"]
        await asyncio.wait_for(call, timeout=3.0)

    async def test_the_run_does_not_end_while_a_callback_is_running(self) -> None:
        """The agent hangs up from the session's own task while the transport reads ``run()`` to its end. A
        transport that tears the call down when ``run()`` ends must not do so in the middle of a callback."""
        session, stt, _tts = _session(_hangs_up_with(GOODBYE), tools=[end_call])
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        release, seen = asyncio.Event(), []

        async def save() -> None:
            seen.append("started")
            await release.wait()
            seen.append("saved")

        session.on_close(save)
        await stt.inject(TranscriptEvent(type="committed", text="Vale, vamos a terminar."))
        await _until(lambda: seen == ["started"])
        await asyncio.sleep(0.3)
        assert session.closed and not call.done(), "the run ended in the middle of an on_close callback"

        release.set()
        await asyncio.wait_for(call, timeout=3.0)
        assert seen == ["started", "saved"]

    async def test_a_cancelled_close_does_not_take_the_callback_with_it(self) -> None:
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])
        release, seen = asyncio.Event(), []

        async def save() -> None:
            seen.append("started")
            await release.wait()
            seen.append("saved")

        session.on_close(save)
        closing = asyncio.create_task(session.close())
        await _until(lambda: seen == ["started"])
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        release.set()
        await asyncio.wait_for(call, timeout=3.0)
        assert seen == ["started", "saved"]

    async def test_runs_when_the_run_ends_without_close(self) -> None:
        """A consumer that stops reading: nobody called ``close()``, and the call is over all the same."""
        session, _stt, _tts = _session(_Model())
        seen: list[str] = []
        session.on_close(lambda: seen.append("over"))

        async def _no_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504 — make it an async generator

        async with aclosing(session.run(_no_audio())) as stream:
            async for event in stream:
                if isinstance(event, SessionStarted):
                    break

        assert seen == ["over"]
        await session.close()
        assert seen == ["over"]

    async def test_a_failing_callback_does_not_stop_the_others_or_the_close(self) -> None:
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])
        seen: list[str] = []

        def broken() -> None:
            raise RuntimeError("database is down")

        session.on_close(broken)
        session.on_close(lambda: seen.append("second"))
        with capture_logs() as logs:
            await session.close()

        assert seen == ["second"]
        [failure] = [entry for entry in logs if entry["event"] == "voice_on_close_failed"]
        assert "database is down" in failure["error"]
        await asyncio.wait_for(call, timeout=3.0)

    async def test_a_callback_that_never_returns_does_not_hold_the_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])
        monkeypatch.setattr(type(session), "ON_CLOSE_TIMEOUT_SECS", 0.05)
        seen: list[str] = []

        async def stuck() -> None:
            await asyncio.Event().wait()

        session.on_close(stuck)
        session.on_close(lambda: seen.append("second"))
        with capture_logs() as logs:
            await asyncio.wait_for(session.close(), timeout=2.0)

        assert seen == ["second"]
        assert [entry["event"] for entry in logs if entry["event"].startswith("voice_on_close")] == [
            "voice_on_close_timeout"
        ]
        await asyncio.wait_for(call, timeout=3.0)

    async def test_registered_after_the_call_ended_runs_straight_away(self) -> None:
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])
        await session.close()
        await asyncio.wait_for(call, timeout=3.0)
        seen: list[str] = []

        session.on_close(lambda: seen.append("late"))
        await _until(lambda: seen == ["late"])

    async def test_runs_in_the_context_it_was_registered_in(self) -> None:
        """Registered from a tool, the callback still has that call's context once the call is over."""
        where: contextvars.ContextVar[str] = contextvars.ContextVar("where", default="nowhere")
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])
        seen: list[str] = []

        async def after_an_await() -> None:
            await asyncio.sleep(0)
            seen.append(where.get())

        def register() -> None:
            where.set("the tool")
            session.on_close(lambda: seen.append(where.get()))
            session.on_close(after_an_await)

        contextvars.copy_context().run(register)
        await session.close()

        assert seen == ["the tool", "the tool"]
        await asyncio.wait_for(call, timeout=3.0)

    async def test_only_a_callable_is_taken(self) -> None:
        session, _stt, _tts = _session(_Model())
        with pytest.raises(TypeError):
            session.on_close("not a function")  # type: ignore[arg-type]


class TestWaitUntilHeard:
    async def test_true_once_the_reply_has_played(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_Model("Un momento."), tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Hola, quería una cita."))
        await _until(lambda: "Un momento." in _texts(events))
        heard = asyncio.create_task(session.wait_until_heard())
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert not heard.done(), "said it was heard while it was still playing"

        tracker.playing = False
        assert await asyncio.wait_for(heard, timeout=3.0) is True
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_false_when_the_caller_speaks_first(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_Model("Un momento.", "Dime."), tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Hola, quería una cita."))
        await _until(lambda: "Un momento." in _texts(events))
        heard = asyncio.create_task(session.wait_until_heard())
        await asyncio.sleep(0.1)
        await stt.inject(TranscriptEvent(type="committed", text="Espera, otra cosa."))

        assert await asyncio.wait_for(heard, timeout=3.0) is False
        tracker.playing = False
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_a_word_the_session_drops_is_not_the_caller_speaking(self) -> None:
        """«Vale.» over the reply is listening, not taking the floor: the session drops it and so does the wait."""
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_Model("Un momento."), tracker=tracker, ignore_backchannels=True)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Hola, quería una cita."))
        await _turn_over(session, events, "Un momento.")
        heard = asyncio.create_task(session.wait_until_heard())
        await asyncio.sleep(0.1)
        await stt.inject(TranscriptEvent(type="committed", text="Vale."))
        await _until(lambda: session._dropped_commit_text == "Vale.")
        await asyncio.sleep(0.2)
        assert not heard.done()

        tracker.playing = False
        assert await asyncio.wait_for(heard, timeout=3.0) is True
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_false_when_the_call_ends_first(self) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_Model("Un momento."), tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Hola, quería una cita."))
        await _until(lambda: "Un momento." in _texts(events))
        heard = asyncio.create_task(session.wait_until_heard())
        await asyncio.sleep(0.1)
        await session.close()

        assert await asyncio.wait_for(heard, timeout=3.0) is False
        await asyncio.wait_for(call, timeout=3.0)

    async def test_gives_up_on_audio_that_never_drains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracker = FakePlaybackTracker()
        session, stt, _tts = _session(_Model("Un momento."), tracker=tracker)
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        monkeypatch.setattr(type(session), "END_CALL_MAX_WAIT_SECS", 0.3)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Hola, quería una cita."))
        await _until(lambda: "Un momento." in _texts(events))

        assert await asyncio.wait_for(session.wait_until_heard(), timeout=3.0) is False
        tracker.playing = False
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)


class TestTransferCall:
    async def test_transfers_once_the_handoff_has_been_heard_and_ends_the_session(self) -> None:
        tracker, line = FakePlaybackTracker(), _Line()
        session, stt, _tts = _session(_transfers_with(HANDOFF), tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)
        assert session.can_transfer

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        outcome = _outcome(session)
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.3)
        assert line.dialled == [] and not session.closed, "transferred over its own sentence"

        tracker.playing = False
        await asyncio.wait_for(call, timeout=3.0)
        assert line.dialled == [NUMBER]
        assert outcome.result() == "transferred"
        assert session.closed and not session.can_transfer

    async def test_a_new_request_keeps_the_call_with_the_agent(self) -> None:
        tracker, line = FakePlaybackTracker(), _Line()
        session, stt, _tts = _session(_transfers_with(HANDOFF, "Dime."), tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        outcome = _outcome(session)
        await stt.inject(TranscriptEvent(type="committed", text="Espera, antes dime el horario del taller."))
        await _until(lambda: "Dime." in _texts(events))
        tracker.playing = False
        await asyncio.sleep(session.END_CALL_TAIL_SECS + 0.5)

        assert await asyncio.wait_for(outcome, timeout=1.0) == "cancelled"
        assert line.dialled == [] and not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_an_acknowledgement_does_not_cancel_it(self) -> None:
        """«Vale, gracias» answers "I'll put you through". It does not take the request back."""
        tracker, line = FakePlaybackTracker(), _Line()
        session, stt, _tts = _session(_transfers_with(HANDOFF, "A usted."), tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _turn_over(session, events, HANDOFF)
        outcome = _outcome(session)
        await stt.inject(TranscriptEvent(type="committed", text="Vale, gracias."))
        await _until(lambda: "A usted." in _texts(events))
        await asyncio.sleep(0.3)
        assert line.dialled == []

        tracker.playing = False
        await asyncio.wait_for(call, timeout=5.0)
        assert line.dialled == [NUMBER] and outcome.result() == "transferred"

    async def test_an_acknowledgement_the_session_drops_does_not_cancel_it(self) -> None:
        """«Vale.» over "I'll put you through" starts no turn, so it is in no transcript row. The row before it
        is the request that asked for the transfer, and read in its place it took the transfer back."""
        tracker, line = FakePlaybackTracker(), _Line()
        model = _transfers_with(HANDOFF)
        session, stt, _tts = _session(model, tools=[transfer], tracker=tracker, ignore_backchannels=True)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _turn_over(session, events, HANDOFF)
        outcome = _outcome(session)
        await stt.inject(TranscriptEvent(type="committed", text="Vale."))
        await _until(lambda: session._dropped_commit_text == "Vale.")
        await asyncio.sleep(0.3)
        assert len(model.calls) == 2, "the acknowledgement started a turn"
        assert line.dialled == [] and not outcome.done()

        tracker.playing = False
        await asyncio.wait_for(call, timeout=5.0)
        assert line.dialled == [NUMBER] and outcome.result() == "transferred"

    async def test_a_refused_transfer_leaves_the_call_with_the_agent(self) -> None:
        tracker, line = FakePlaybackTracker(), _Line(refuses=True)
        session, stt, _tts = _session(_transfers_with(HANDOFF), tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        outcome = _outcome(session)
        with capture_logs() as logs:
            assert await asyncio.wait_for(outcome, timeout=5.0) == "failed"

        assert line.dialled == [NUMBER]
        assert not session.closed and session.can_transfer
        [failure] = [entry for entry in logs if entry["event"] == "call_transfer_failed"]
        assert "422" in failure["error"]
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_the_agent_can_be_told_a_transfer_failed(self) -> None:
        """What the docs suggest after ``"failed"``: the caller was promised a person, so the agent says otherwise."""
        tracker, line = FakePlaybackTracker(), _Line(refuses=True)
        sorry = "Lo siento, no he podido pasarle. ¿Le tomo nota?"
        model = _transfers_with(HANDOFF, sorry)
        session, stt, _tts = _session(model, tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        if await _outcome(session) == "failed":
            await session.notify("The transfer did not go through. Apologise and offer to take a message.")
        await _until(lambda: sorry in _texts(events))

        assert "The transfer did not go through" in model.calls[-1][-1].collect_text()
        assert not session.closed
        await stt.finish()
        await asyncio.wait_for(call, timeout=3.0)

    async def test_a_hangup_asked_for_meanwhile_does_not_drop_the_caller(self) -> None:
        """The caller was told they are being put through. A model that answers their «gracias, adiós» with its
        hangup tool must not drop them instead."""
        tracker, line = FakePlaybackTracker(), _Line()
        hangs_up = Message(
            role="assistant", content=[ToolUseContent(id="c2", name="end_call", input={})], stop_reason="tool_use"
        )
        session, stt, _tts = _session(
            _transfers_with(HANDOFF, hangs_up, "Adiós."), tools=[transfer, end_call], tracker=tracker
        )
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        outcome = _outcome(session)
        await stt.inject(TranscriptEvent(type="committed", text="Vale, gracias, adiós."))
        await _until(lambda: "Adiós." in _texts(events))
        assert session._transfer_outcome is outcome and not outcome.done()

        tracker.playing = False
        await asyncio.wait_for(call, timeout=5.0)
        assert line.dialled == [NUMBER] and outcome.result() == "transferred"

    async def test_asked_again_it_is_the_same_transfer_to_the_last_number(self) -> None:
        tracker, line = FakePlaybackTracker(), _Line()
        session, stt, _tts = _session(_transfers_with(HANDOFF), tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        outcome = _outcome(session)
        assert session.transfer_call(" sip:cola@centralita.example ") is outcome

        tracker.playing = False
        await asyncio.wait_for(call, timeout=3.0)
        assert line.dialled == ["sip:cola@centralita.example"] and outcome.result() == "transferred"

    async def test_the_caller_hanging_up_first_ends_it_as_closed(self) -> None:
        tracker, line = FakePlaybackTracker(), _Line()
        session, stt, _tts = _session(_transfers_with(HANDOFF), tools=[transfer], tracker=tracker)
        session.call_transferer = line.transfer
        events: list[VoiceSessionEvent] = []
        call = await _call(session, events)

        tracker.playing = True
        await stt.inject(TranscriptEvent(type="committed", text="Quiero hablar con una persona."))
        await _until(lambda: HANDOFF in _texts(events))
        outcome = _outcome(session)
        await session.close()

        assert await asyncio.wait_for(outcome, timeout=1.0) == "closed"
        assert line.dialled == []
        await asyncio.wait_for(call, timeout=3.0)

    async def test_a_call_that_cannot_be_transferred_says_so(self) -> None:
        session, _stt, _tts = _session(_Model())
        call = await _call(session, [])

        assert not session.can_transfer
        with pytest.raises(RuntimeError, match="cannot be transferred"):
            session.transfer_call(NUMBER)
        session.call_transferer = _Line().transfer
        with pytest.raises(ValueError, match="number or SIP address"):
            session.transfer_call("  ")

        await session.close()
        await asyncio.wait_for(call, timeout=3.0)
        assert await session.transfer_call(NUMBER) == "closed"


class TestWhatTheCallerSaidMeanwhile:
    """What keeps the call up while a hangup or a transfer waits, when the session dropped the commit."""

    REQUEST = "Quiero hablar con una persona."

    async def _waiting(self):
        session, _stt, _tts = _session(_Model())
        session._transcript.append(TranscriptEntry(role="user", text=self.REQUEST))
        session._stamp_taken_commit()
        session._transcript.append(TranscriptEntry(role="assistant", text=HANDOFF))
        await asyncio.sleep(0.02)
        return session, session._last_commit_at

    @pytest.mark.parametrize(
        "dropped,keeps_the_call",
        [
            ("Vale.", False),
            ("Adiós.", False),
            ("Espera.", True),
            ("¿Cuánto va a tardar?", True),
            (REQUEST, False),  # a late twin of the request itself: nothing new
            (HANDOFF, False),  # the agent's own line coming back through the caller's microphone
        ],
    )
    async def test_a_dropped_commit_is_read_by_its_own_words(self, dropped: str, keeps_the_call: bool) -> None:
        session, baseline = await self._waiting()
        session._settle_consumed_commit(dropped)
        assert session._last_commit_at > baseline
        assert session._caller_retracted_since(baseline) is keeps_the_call

    async def test_a_request_taken_before_the_dropped_word_still_counts(self) -> None:
        session, baseline = await self._waiting()
        session._transcript.append(TranscriptEntry(role="user", text="Espera, una cosa más."))
        session._stamp_taken_commit()
        session._settle_consumed_commit("Vale.")
        assert session._caller_retracted_since(baseline) is True

    async def test_nothing_said_is_nothing_new(self) -> None:
        session, baseline = await self._waiting()
        assert session._caller_retracted_since(baseline) is False
