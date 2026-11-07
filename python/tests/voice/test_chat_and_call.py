"""One conversation typed in a ChatSession and spoken on a VoiceSession call at the same time."""

import asyncio

from timbal import Agent
from timbal.core.chat_session import ChatSession
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.voice import TranscriptEvent, VoiceSession, VoiceSessionEvent

from .test_session import DelayedMockSTT, MockTTS
from .test_voice_events import _GatedModel, _in_order, _Model, _run, _seen, _texts, _until


def _agent(model: _Model) -> Agent:
    return Agent(name="assistant", model=model.model, tracing_provider=InMemoryTracingProvider)


def _call(model: _Model, **kwargs) -> tuple[VoiceSession, DelayedMockSTT]:
    stt = DelayedMockSTT()
    return VoiceSession(agent=_agent(model), stt=stt, tts=MockTTS(), turn_detector="heuristic", **kwargs), stt


async def _turns(chat: ChatSession, n: int) -> None:
    await _until(lambda: chat.turns >= n and not chat.busy)


class TestCallJoinsAChat:
    async def test_one_line_across_both_channels(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Otra vez por chat.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _until(lambda: "Hola por voz." in _texts(events))
                await _until(lambda: call._current_turn_task is None or call._current_turn_task.done())
                chat.send("escrito 2")
                await _turns(chat, 2)

            await _run(call, stt, drive)

        assert _seen(model.calls[1])[-3:] == ["escrito 1", "Hola por chat.", "hablado 1"]
        assert _seen(model.calls[2])[-3:] == ["hablado 1", "Hola por voz.", "escrito 2"]


def _owner(chat: ChatSession):
    return chat.store.continuation_owner()


class TestTheCallHasTheSessionWhileLive:
    async def test_events_are_spoken_and_the_chat_gets_the_session_back(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "El informe está listo.", "Otro aviso por chat.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _until(lambda: "Hola por voz." in _texts(events))
                await _until(lambda: _owner(chat) is call._scheduler)
                assert chat._scheduler.lent
                await call.notify("informe listo")
                await _until(lambda: "El informe está listo." in _texts(events))

            await _run(call, stt, drive)
            assert chat.turns == 1  # the call spoke the event; the chat started nothing
            # Hung up: the session is the chat's again, continuing from the call's last turn.
            await _until(lambda: _owner(chat) is chat._scheduler)
            assert not chat._scheduler.lent and chat.head == call._last_run_context.id
            chat.store.post_event("otro aviso")
            await _turns(chat, 2)

        assert _in_order(_seen(model.calls[3]), "hablado 1", "Hola por voz.", "El informe está listo.")

    async def test_a_message_typed_during_the_call_is_a_chat_turn(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Respondido por chat.", "Visto por voz.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _until(lambda: "Hola por voz." in _texts(events))
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.send("escrito 2")
                await _turns(chat, 2)
                assert "Respondido por chat." not in _texts(events)  # answered in the chat, not spoken
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: "Visto por voz." in _texts(events))

            await _run(call, stt, drive)

        assert chat.last_status["code"] == "success"
        assert _seen(model.calls[2])[-3:] == ["hablado 1", "Hola por voz.", "escrito 2"]
        assert _seen(model.calls[3])[-3:] == ["escrito 2", "Respondido por chat.", "hablado 2"]


class TestBothBusy:
    async def test_the_caller_waits_for_a_running_chat_turn(self) -> None:
        # Call 1 (the chat's second turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Respuesta larga por chat.", "Hola por voz.", gate_at=1)
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                chat.send("escrito 2")
                await asyncio.wait_for(model.entered.wait(), 2)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await asyncio.sleep(0.3)
                assert len(model.calls) == 1  # the caller's turn waits for the chat's
                model.opened.set()
                await _until(lambda: "Hola por voz." in _texts(events))

            await _run(call, stt, drive)

        assert _seen(model.calls[2])[-3:] == ["escrito 2", "Respuesta larga por chat.", "hablado 1"]

    async def test_a_message_typed_during_a_call_turn_waits_for_it(self) -> None:
        # Call 1 (the call's turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Visto por chat.", gate_at=1)
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await asyncio.wait_for(model.entered.wait(), 2)
                chat.send("escrito 2")
                await asyncio.sleep(0.3)
                assert len(model.calls) == 1  # the chat's turn waits for the caller's
                model.opened.set()
                await _until(lambda: "Hola por voz." in _texts(events))
                await _turns(chat, 2)

            await _run(call, stt, drive)

        assert _seen(model.calls[2])[-3:] == ["hablado 1", "Hola por voz.", "escrito 2"]


class TestChatOpensOnACall:
    async def test_a_chat_on_a_call_that_owns_its_store(self) -> None:
        model = _Model("Hola por voz.", "Hola por chat.", "Aviso por voz.")
        call, stt = _call(model)
        chats: list[ChatSession] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await call.notify("dato de la app", announce="silent")  # the call owns a store now
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _until(lambda: "Hola por voz." in _texts(events))
            await _until(lambda: call._current_turn_task.done())
            chat = ChatSession(_agent(model), parent_id=call._last_run_context.id)
            chats.append(chat)
            assert chat._scheduler.lent and _owner(chat) is call._scheduler
            chat.send("escrito 1")
            await _turns(chat, 1)
            await call.notify("aviso")
            await _until(lambda: "Aviso por voz." in _texts(events))

        await _run(call, stt, drive)
        chat = chats[0]
        try:
            assert chat.turns == 1
            await _until(lambda: _owner(chat) is chat._scheduler)  # the call ended: the chat has it now
        finally:
            await chat.close()
        assert _seen(model.calls[1])[-3:] == ["hablado 1", "Hola por voz.", "escrito 1"]

    async def test_a_chat_on_a_live_call(self) -> None:
        model = _Model("Hola por voz.", "Hola por chat.", "Otra vez por voz.")
        call, stt = _call(model)
        seen_chat = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _until(lambda: "Hola por voz." in _texts(events))
            await _until(lambda: call._current_turn_task.done())
            async with ChatSession(_agent(model), parent_id=call._last_run_context.id) as chat:
                chat.send("escrito 1")
                await _turns(chat, 1)
                seen_chat.append(chat.last_status)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: "Otra vez por voz." in _texts(events))

        await _run(call, stt, drive)
        assert _seen(model.calls[1])[-3:] == ["hablado 1", "Hola por voz.", "escrito 1"]
        assert _seen(model.calls[2])[-3:] == ["escrito 1", "Hola por chat.", "hablado 2"]
