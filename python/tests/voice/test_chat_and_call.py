"""One conversation typed in a ChatSession and spoken on a VoiceSession call at the same time."""

import asyncio
import contextvars

import pytest
from timbal import Agent
from timbal.core.chat_session import RECOVERY_KIND, ChatSession, _Wake
from timbal.core.tool import Tool
from timbal.state.chat_store import InMemoryChatSessionStore
from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.voice import AgentApproval, TranscriptEvent, VoiceSession, VoiceSessionEvent

from .test_session import DelayedMockSTT, MockTTS
from .test_voice_events import _GatedModel, _in_order, _Model, _run, _seen, _texts, _tool_call, _until


def _agent(model: _Model, tools: list | None = None, provider: type = InMemoryTracingProvider) -> Agent:
    return Agent(name="assistant", model=model.model, tools=tools or [], tracing_provider=provider)


def _call(
    model: _Model, tools: list | None = None, provider: type = InMemoryTracingProvider, **kwargs
) -> tuple[VoiceSession, DelayedMockSTT]:
    stt = DelayedMockSTT()
    agent = _agent(model, tools, provider)
    return VoiceSession(agent=agent, stt=stt, tts=MockTTS(), turn_detector="heuristic", **kwargs), stt


async def _turns(chat: ChatSession, n: int) -> None:
    await _until(lambda: chat.turns >= n and not chat.busy)


async def _said(call: VoiceSession, events: list[VoiceSessionEvent], text: str, timeout: float = 5.0) -> None:
    """The call said ``text``, and that turn is over.

    Not the text alone: it is announced before the turn has ended (its run recorded as the call's last,
    the session free for the next utterance), and how long that takes differs by machine. A step that
    follows on the text may find the turn still going: an utterance injected then is not a new turn, and
    ``_last_run_context`` is still the turn before.
    """
    await _until(lambda: text in _texts(events), timeout)
    await _until(lambda: call._current_turn_task is None or call._current_turn_task.done(), timeout)


class TestCallJoinsAChat:
    async def test_one_line_across_both_channels(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Otra vez por chat.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: call._current_turn_task is None or call._current_turn_task.done())
                chat.send("escrito 2")
                await _turns(chat, 2)

            await _run(call, stt, drive)

        assert _seen(model.calls[1])[-3:] == ["escrito 1", "Hola por chat.", "hablado 1"]
        assert _seen(model.calls[2])[-3:] == ["hablado 1", "Hola por voz.", "escrito 2"]


class TestWhoAnswersAUserMessage:
    """A message sent as the user's (``source="user"``): the chat's while one shares the session, else the call's."""

    async def test_a_call_alone_answers_it(self) -> None:
        model = _Model("Son las tres.")
        call, stt = _call(model)
        receipts = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            receipts.append(await call.notify("¿qué hora es?", source="user", announce="when_idle"))
            await _said(call, events, "Son las tres.")

        await _run(call, stt, drive)
        assert receipts[0].state == "delivered"
        assert _seen(model.calls[0])[-1] == "¿qué hora es?"

    async def test_the_call_answers_it_again_once_the_chat_is_gone(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Son las tres.")
        chat = ChatSession(_agent(model))
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await chat.close()
                # Nobody is left to answer what is typed: the call does, as a call alone.
                await call.notify("¿qué hora es?", source="user", announce="when_idle")
                await _said(call, events, "Son las tres.")

            await _run(call, stt, drive)
        finally:
            await chat.close()
        assert _seen(model.calls[2])[-1] == "¿qué hora es?"

    async def test_a_chat_closed_to_be_restored_keeps_what_was_typed(self) -> None:
        # Call 2 (the call's second turn) holds until opened: a message typed meanwhile waits for it.
        model = _GatedModel(
            "Hola por chat.", "Hola por voz.", "Larga por voz.", "Son las tres.", "Visto por chat.", gate_at=2
        )
        store = InMemoryChatSessionStore()
        agent = _agent(model)
        chat = ChatSession(agent, session_store=store)
        restored: list[ChatSession] = []
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                typed = chat.send("escrito 2")
                await chat.flush()
                # Kept: the session lives on in its store, with this message. Not the call's to answer.
                await chat.close(keep=True)
                model.opened.set()
                await _said(call, events, "Larga por voz.")
                # Once the line is quiet the call speaks what is its own to answer. This is not.
                await _until(lambda: call._event_turn_blocker() is None)
                await asyncio.sleep(0.3)
                assert typed.state == "pending" and len(model.calls) == 3
                # What is sent to the call from now on is in no snapshot: the call's, as on a call alone.
                later = await call.notify("¿qué hora es?", source="user", announce="when_idle")
                await _said(call, events, "Son las tres.")
                assert later.state == "delivered" and typed.state == "pending"
                restored.append(await ChatSession.from_store(agent, store, chat.id))
                await _turns(restored[0], 2)

            await _run(call, stt, drive)
        finally:
            for session in (chat, *restored):
                await session.close()
        # Each answered once: the later one by the call, the kept one by the restored chat.
        last = [_seen(seen)[-1] for seen in model.calls]
        assert last.count("escrito 2") == 1 and last.count("¿qué hora es?") == 1 and len(model.calls) == 5

    async def test_another_chat_on_the_call_leaves_a_kept_chat_s_message_alone(self) -> None:
        # Call 2 (the call's second turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Larga por voz.", "Visto por chat.", gate_at=2)
        store = InMemoryChatSessionStore()
        agent = _agent(model)
        chat = ChatSession(agent, session_store=store)
        opened: list[ChatSession] = []
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                typed = chat.send("escrito 2")
                await chat.flush()
                await chat.close(keep=True)
                # A client that opens a new session on the conversation, not the one it had.
                other = ChatSession(_agent(model), parent_id=call._last_run_context.id)
                opened.append(other)
                model.opened.set()
                await _said(call, events, "Larga por voz.")
                await _until(lambda: call._event_turn_blocker() is None)
                await asyncio.sleep(0.3)
                # The message is the kept session's (it is in its snapshot): not this chat's either.
                assert typed.state == "pending" and other.turns == 0 and len(model.calls) == 3
                await other.close()
                opened.append(await ChatSession.from_store(agent, store, chat.id))
                await _turns(opened[1], 2)

            await _run(call, stt, drive)
        finally:
            for session in (chat, *opened):
                await session.close()
        assert [_seen(seen)[-1] for seen in model.calls].count("escrito 2") == 1 and len(model.calls) == 4

    @pytest.mark.parametrize("source", ["user", "app"])
    async def test_what_arrives_after_the_chat_let_go_is_not_in_its_snapshot(self, source: str) -> None:
        # The snapshot of a session closed to be restored is written after it has let go of the session
        # (here: behind a save still out). What reaches the call meanwhile is the call's: a user message
        # it answers, an application's event it speaks. Stored with the chat too, it would come back with
        # the chat and be delivered a second time.
        # Call 2 (the call's second turn) holds until opened.
        model = _GatedModel(
            "Hola por chat.", "Hola por voz.", "Larga por voz.", "Son las tres.", "Visto por chat.", gate_at=2
        )
        store = _GatedStore()
        agent = _agent(model)
        chat = ChatSession(agent, session_store=store)
        restored: list[ChatSession] = []
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            await chat.flush()
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await chat.flush()
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                store.hold = True
                chat.send("escrito 2")
                await asyncio.wait_for(store.entered.wait(), 2)  # its save is out, and held
                closing = asyncio.create_task(chat.close(keep=True))
                await _until(lambda: call._scheduler.successor(chat.store) is None)  # the chat has let go
                if source == "user":
                    later = await call.notify("¿qué hora es?", source="user", announce="when_idle")
                else:
                    later = await call.notify("¿qué hora es?")
                store.gate.set()
                await closing
                stored = [event["text"] for event in (await store.load(chat.id)).pending]
                assert stored == ["escrito 2"]
                model.opened.set()
                await _said(call, events, "Larga por voz.")
                await _said(call, events, "Son las tres.")
                assert later.state == "delivered"
                restored.append(await ChatSession.from_store(agent, store, chat.id))
                await _turns(restored[0], 2)

            await _run(call, stt, drive)
        finally:
            for session in (chat, *restored):
                await session.close()
        last = [seen[-1].collect_text() for seen in model.calls]
        assert sum("escrito 2" in text for text in last) == 1 and sum("¿qué hora es?" in text for text in last) == 1
        assert len(model.calls) == 5

    async def test_a_restore_that_fails_does_not_claim_what_was_sent_to_the_call_since(self) -> None:
        # The kept session is restored while its trace store is away: the restore fails and lets go
        # again. A message sent to the call in between was never the chat's, and is not marked as kept.
        _Flaky.down = False
        # Call 2 (the chat's second turn) holds until it is cut, before the model has answered it.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Son las tres.", gate_at=2)
        store = InMemoryChatSessionStore()
        agent = _agent(model, provider=_Flaky)
        chat = ChatSession(agent, session_store=store)
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, provider=_Flaky, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.send("escrito 2")
                await asyncio.wait_for(model.entered.wait(), 2)
                await chat.flush()
                await chat.close(keep=True)  # cuts the turn: the restore has to read its run
                model.opened.set()  # the gate is for that turn only
                later = await call.notify("¿qué hora es?", source="user", announce="when_idle")
                _Flaky.down = True
                with pytest.raises(ConnectionError):
                    await ChatSession.from_store(agent, store, chat.id)
                _Flaky.down = False
                assert later.kept_for is None
                await _said(call, events, "Son las tres.")
                assert later.state == "delivered"

            await _run(call, stt, drive)
        finally:
            _Flaky.down = False
            await chat.close()

    async def test_a_second_kept_chat_does_not_take_over_the_first_one_s_message(self) -> None:
        # Two sessions on one conversation, both closed to be restored. The message is the first one's: the
        # second neither marks it as its own nor stores it, so it is answered once, by the first.
        # Call 2 (the call's second turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Larga por voz.", "Visto por chat.", gate_at=2)
        store = InMemoryChatSessionStore()
        agent = _agent(model)
        chat = ChatSession(agent, session_store=store)
        opened: list[ChatSession] = []
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                typed = chat.send("escrito 2")
                await chat.flush()
                await chat.close(keep=True)
                second = ChatSession(agent, parent_id=call._last_run_context.id, session_store=store)
                opened.append(second)
                assert second.pending_messages() == []
                await second.flush()
                await second.close(keep=True)
                assert typed.kept_for == chat.id and (await store.load(second.id)).pending == []
                model.opened.set()
                await _said(call, events, "Larga por voz.")
                opened.append(await ChatSession.from_store(agent, store, second.id))
                await asyncio.sleep(0.3)
                assert typed.state == "pending" and len(model.calls) == 3  # not the second session's to answer
                await opened[1].close(keep=True)
                opened.append(await ChatSession.from_store(agent, store, chat.id))
                await _turns(opened[2], 2)

            await _run(call, stt, drive)
        finally:
            for session in (chat, *opened):
                await session.close()
        assert [_seen(seen)[-1] for seen in model.calls].count("escrito 2") == 1 and len(model.calls) == 4

    async def test_a_chat_opened_on_a_call_takes_the_message_the_call_was_holding(self) -> None:
        # Call 1 (the call's second turn) holds until opened: a user message sent to the call waits for it.
        model = _GatedModel("Hola por voz.", "Sigo por voz.", "Son las tres.", gate_at=1)
        call, stt = _call(model)
        chats: list[ChatSession] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await call.notify("dato", announce="silent")  # the call owns a store
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _said(call, events, "Hola por voz.")
            await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
            await asyncio.wait_for(model.entered.wait(), 2)
            receipt = await call.notify("¿qué hora es?", source="user", announce="when_idle")
            chat = ChatSession(_agent(model), parent_id=call._last_run_context.id)
            chats.append(chat)
            model.opened.set()
            await _said(call, events, "Sigo por voz.")
            # The chat's from the moment it is there, though nothing was sent to it: answered as a chat turn.
            await _turns(chat, 1)
            assert receipt.state == "delivered" and "Son las tres." not in _texts(events)

        await _run(call, stt, drive)
        await chats[0].close()
        assert _seen(model.calls[2])[-1] == "¿qué hora es?"


class TestTheCallsEventsStayTheCalls:
    """While the call has the session, a chat turn answers what was typed and takes nothing else."""

    async def test_a_call_turn_that_steers_leaves_what_was_typed_to_the_chat(self) -> None:
        # An agent that always steers (steering=True): the call's turn takes the events posted for the
        # turn in flight before it ends. Not what the user typed: that is the chat's turn.
        # Call 1 (the call's turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Visto por chat.", gate_at=1)

        def steering_agent() -> Agent:
            return Agent(name="assistant", model=model.model, steering=True, tracing_provider=InMemoryTracingProvider)

        async with ChatSession(steering_agent()) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            stt = DelayedMockSTT()
            call = VoiceSession(
                agent=steering_agent(), stt=stt, tts=MockTTS(), turn_detector="heuristic", parent_run_id=chat.head
            )
            typed = []

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await asyncio.wait_for(model.entered.wait(), 2)
                typed.append(chat.send("escrito 2"))  # steer: for the turn in flight, if it were the chat's
                await asyncio.sleep(0.2)
                model.opened.set()
                await _said(call, events, "Hola por voz.")
                await _turns(chat, 2)

            await _run(call, stt, drive)
            assert typed[0].run_id != call._last_run_context.id

        assert len(model.calls) == 3 and _seen(model.calls[2])[-1] == "escrito 2"

    async def test_a_call_turn_the_chat_admitted_leaves_what_was_typed_to_the_chat(self) -> None:
        # The chat has the session here (the call had no store and meets the chat's with its turn), and
        # the agent always steers. What is typed during the call's turn is still the chat's turn.
        # Call 1 (the call's second turn) holds until opened.
        model = _GatedModel("Hola por voz.", "Sigo por voz.", "Visto por chat.", gate_at=1)

        def steering_agent() -> Agent:
            return Agent(name="assistant", model=model.model, steering=True, tracing_provider=InMemoryTracingProvider)

        stt = DelayedMockSTT()
        call = VoiceSession(agent=steering_agent(), stt=stt, tts=MockTTS(), turn_detector="heuristic")
        typed = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _said(call, events, "Hola por voz.")
            async with ChatSession(steering_agent(), parent_id=call._last_run_context.id) as chat:
                assert _owner(chat) is chat._scheduler
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                typed.append(chat.send("escrito 2"))
                await asyncio.sleep(0.2)
                model.opened.set()
                await _said(call, events, "Sigo por voz.")
                await _turns(chat, 1)
                assert typed[0].run_id != call._last_run_context.id

        await _run(call, stt, drive)
        assert len(model.calls) == 3 and _seen(model.calls[2])[-1] == "escrito 2"

    async def test_an_event_the_call_spoke_is_not_spoken_again_when_a_kept_chat_comes_back(self) -> None:
        # The chat's snapshot lists the event (it was pending when the chat closed to be restored), and
        # the call then speaks it. Restored in the same process, the chat does not queue it a second time.
        # Call 2 (the call's second turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Larga por voz.", "Aviso uno.", "Aviso dos.", gate_at=2)
        store = InMemoryChatSessionStore()
        agent = _agent(model)
        chat = ChatSession(agent, session_store=store)
        restored: list[ChatSession] = []
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                receipt = await call.notify("informe listo")
                await chat.flush()
                await chat.close(keep=True)
                assert [event["text"] for event in (await store.load(chat.id)).pending] == ["informe listo"]
                model.opened.set()
                await _said(call, events, "Larga por voz.")
                await _said(call, events, "Aviso uno.")
                assert receipt.state == "delivered"
                restored.append(await ChatSession.from_store(agent, store, chat.id))
                await _until(lambda: call._event_turn_blocker() is None)
                await asyncio.sleep(0.4)
                assert "Aviso dos." not in _texts(events) and len(model.calls) == 4

            await _run(call, stt, drive)
        finally:
            for session in (chat, *restored):
                await session.close()

    async def test_a_restore_whose_snapshot_never_lands_does_not_forget_what_the_call_spoke(self) -> None:
        # As above, and the session store is away while the chat is restored: that session closes again
        # with nothing of its own stored, so the snapshot is still the one that lists the event.
        # Call 2 (the call's second turn) holds until opened.
        model = _GatedModel("Hola por chat.", "Hola por voz.", "Larga por voz.", "Aviso uno.", "Aviso dos.", gate_at=2)
        store = _AwayStore()
        agent = _agent(model)
        chat = ChatSession(agent, session_store=store)
        restored: list[ChatSession] = []
        try:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                receipt = await call.notify("informe listo")
                await chat.flush()
                await chat.close(keep=True)
                model.opened.set()
                await _said(call, events, "Larga por voz.")
                await _said(call, events, "Aviso uno.")
                assert receipt.state == "delivered"
                store.away = True
                restored.append(await ChatSession.from_store(agent, store, chat.id))
                await restored[0].close(keep=True, store_timeout=0.3)
                store.away = False
                assert [event["text"] for event in (await store.load(chat.id)).pending] == ["informe listo"]
                restored.append(await ChatSession.from_store(agent, store, chat.id))
                await _until(lambda: call._event_turn_blocker() is None)
                await asyncio.sleep(0.4)
                assert "Aviso dos." not in _texts(events) and len(model.calls) == 4

            await _run(call, stt, drive)
        finally:
            store.away = False
            for session in (chat, *restored):
                await session.close()

    async def test_a_chat_turn_does_not_take_an_event_posted_as_it_starts(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Visto por chat.", "Aviso por voz.")
        entered, gate, hold = asyncio.Event(), asyncio.Event(), [False]

        async def prompt() -> str:
            if hold[0]:
                hold[0] = False
                entered.set()
                await gate.wait()
            return "Eres útil."

        chat_agent = Agent(
            name="assistant", model=model.model, system_prompt=prompt, tracing_provider=InMemoryTracingProvider
        )
        async with ChatSession(chat_agent) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)
            receipts = []

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                hold[0] = True
                chat.send("escrito 2")
                await asyncio.wait_for(entered.wait(), 3)  # the chat's turn has started, its prompt not built yet
                receipts.append(await call.notify("informe listo"))
                gate.set()
                await _turns(chat, 2)
                # The call's to speak, not the chat's to fold into its answer.
                await _said(call, events, "Aviso por voz.")

            await _run(call, stt, drive)

        assert not any("informe listo" in text for text in _seen(model.calls[2]))
        assert "informe listo" in model.calls[3][-1].collect_text()
        assert receipts[0].state == "delivered" and receipts[0].run_id == call._last_run_context.id

    async def test_a_chat_turn_does_not_take_an_event_between_its_tool_batches(self) -> None:
        release, started = asyncio.Event(), asyncio.Event()

        async def slow() -> str:
            """Slow."""
            started.set()
            await release.wait()
            return "hecho"

        tools = [Tool(name="slow", handler=slow)]
        model = _Model("Hola por chat.", "Hola por voz.", _tool_call("slow"), "Visto por chat.", "Aviso por voz.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)
            receipts = []

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.send("escrito 2")
                await asyncio.wait_for(started.wait(), 3)
                # For the turn in flight (lane="next"), but not the user's words: the call's to speak.
                receipts.append(chat.store.post_event("informe listo", lane="next"))
                release.set()
                await _turns(chat, 2)
                await _said(call, events, "Aviso por voz.")

            await _run(call, stt, drive)

        assert not any("informe listo" in text for text in _seen(model.calls[3]))
        assert "informe listo" in model.calls[4][-1].collect_text()
        assert receipts[0].run_id == call._last_run_context.id


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
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                assert chat._scheduler.lent
                await call.notify("informe listo")
                await _said(call, events, "El informe está listo.")

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
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.send("escrito 2")
                await _turns(chat, 2)
                assert "Respondido por chat." not in _texts(events)  # answered in the chat, not spoken
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _said(call, events, "Visto por voz.")

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
                await _said(call, events, "Hola por voz.")

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
                # Not started: what the caller's turn ends on (a question) decides whether it may.
                assert not chat.busy and len(chat.pending_messages()) == 1
                model.opened.set()
                await _said(call, events, "Hola por voz.")
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
            await _said(call, events, "Hola por voz.")
            await _until(lambda: call._current_turn_task.done())
            chat = ChatSession(_agent(model), parent_id=call._last_run_context.id)
            chats.append(chat)
            assert chat._scheduler.lent and _owner(chat) is call._scheduler
            chat.send("escrito 1")
            await _turns(chat, 1)
            await call.notify("aviso")
            await _said(call, events, "Aviso por voz.")

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
            await _said(call, events, "Hola por voz.")
            await _until(lambda: call._current_turn_task.done())
            async with ChatSession(_agent(model), parent_id=call._last_run_context.id) as chat:
                chat.send("escrito 1")
                await _turns(chat, 1)
                seen_chat.append(chat.last_status)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _said(call, events, "Otra vez por voz.")

        await _run(call, stt, drive)
        assert _seen(model.calls[1])[-3:] == ["hablado 1", "Hola por voz.", "escrito 1"]
        assert _seen(model.calls[2])[-3:] == ["escrito 1", "Hola por chat.", "hablado 2"]


def _wire() -> list[Tool]:
    """A tool that asks before it runs, and counts its runs."""
    runs: list[str] = []

    def wire_money() -> str:
        """Wire money."""
        runs.append("wired")
        return "wired"

    tool = Tool(name="wire_money", handler=wire_money, requires_approval=True)
    tool.runs = runs  # type: ignore[attr-defined]
    return [tool]


def _two_gated() -> list[Tool]:
    """``_wire()`` plus a second tool that asks: a run that answers one approval and asks another."""

    def send_mail() -> str:
        """Send mail."""
        return "sent"

    return [*_wire(), Tool(name="send_mail", handler=send_mail, requires_approval=True)]


def _approval(chat: ChatSession) -> str:
    [pending] = chat.state()["pending_approvals"]
    return pending["approval_id"]


def _asked(events: list[VoiceSessionEvent]) -> list[AgentApproval]:
    return [event for event in events if isinstance(event, AgentApproval)]


class TestAQuestionWaitsInTheChat:
    """A chat turn that ended on an approval: only the chat can answer it, so the session stays with it."""

    async def test_the_call_does_not_take_the_session_and_its_event_is_not_swallowed(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"), "Hecho.", "El informe está listo.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("paga")
            await _until(lambda: chat.parked)
            call, stt = _call(model, tools, parent_run_id=chat.head)
            receipts = []

            async def drive(events: list[VoiceSessionEvent]) -> None:
                receipts.append(await call.notify("informe listo"))
                # Until the chat is told the call is there (the call asks when it looks for its store).
                await _until(lambda: chat._scheduler._borrower is not None)
                assert _owner(chat) is chat._scheduler and not chat._scheduler.lent and chat.parked
                assert receipts[0].state == "pending" and len(model.calls) == 1
                chat.resume({_approval(chat): True})
                await _turns(chat, 2)
                # Answered: the call has the session now, and speaks the event the question held back.
                await _said(call, events, "El informe está listo.")
                assert _owner(chat) is call._scheduler and chat._scheduler.lent

            await _run(call, stt, drive)

        assert tools[0].runs == ["wired"] and receipts[0].state == "delivered"
        assert "informe listo" in model.calls[2][-1].collect_text()

    async def test_what_the_caller_says_is_answered_once_the_chat_has_its_answer(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"), "Hecho.", "Hola por voz.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("paga")
            await _until(lambda: chat.parked)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                # The caller's turn queues behind the question: started on that run it would be taken for
                # its answer (prompt dropped, the tool asked again on the call).
                await _until(lambda: chat._scheduler.other_turn_active)
                assert len(model.calls) == 1 and chat.parked and not _asked(events)
                assert _owner(chat) is chat._scheduler
                chat.resume({_approval(chat): True})
                await _said(call, events, "Hola por voz.")
                assert not _asked(events)

            await _run(call, stt, drive)

        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[2])[-2:] == ["Hecho.", "hablado 1"]

    async def test_a_chat_turn_that_asks_during_the_call_takes_the_session_back(self) -> None:
        tools = _wire()
        model = _Model("Hola por chat.", "Hola por voz.", _tool_call("wire_money"), "Hecho.", "Visto por voz.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.send("paga")
                await _until(lambda: chat.parked)
                # The question is the chat's: so is the session while it waits.
                assert _owner(chat) is chat._scheduler and not chat._scheduler.lent
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert len(model.calls) == 3 and not _asked(events)
                chat.resume({_approval(chat): True})
                await _said(call, events, "Visto por voz.")
                # Answered, and the call's turn has run: lent again.
                await _until(lambda: _owner(chat) is call._scheduler)

            await _run(call, stt, drive)

        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[4])[-2:] == ["Hecho.", "hablado 2"]

    async def test_a_chat_that_closes_leaves_its_question_to_the_call(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"))
        chat = ChatSession(_agent(model, tools))
        try:
            chat.send("paga")
            await _until(lambda: chat.parked)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert not _asked(events)
                await chat.close()
                # Nobody is left to answer it in the chat: the call asks (as it does on any parked run).
                await _until(lambda: bool(_asked(events)))

            await _run(call, stt, drive)
        finally:
            await chat.close()
        assert tools[0].runs == []

    async def test_a_chat_restored_on_a_question_keeps_the_session(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"), "Hecho.", "Hola por voz.")
        agent = _agent(model, tools)
        async with ChatSession(agent) as first:
            first.send("paga")
            await _until(lambda: first.parked)
            head = first.head
        async with ChatSession(agent, parent_id=head) as chat:
            # No turn of this session ended on the question: it read it from the stored run. Its store is
            # new (as after a restart): a call continuing that run still finds the session.
            assert chat.parked and chat.store is not first.store
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert _owner(chat) is chat._scheduler and len(model.calls) == 1 and not _asked(events)
                chat.resume({_approval(chat): True})
                await _said(call, events, "Hola por voz.")

            await _run(call, stt, drive)

        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[2])[-2:] == ["Hecho.", "hablado 1"]

    async def test_a_chat_that_closes_hands_the_waiting_call_a_session_that_still_waits(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"))
        chat = ChatSession(_agent(model, tools))
        try:
            chat.send("paga")
            await _until(lambda: chat.parked)
            call, stt = _call(model, tools, parent_run_id=chat.head)
            receipts = []

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await _until(lambda: chat._scheduler._borrower is not None)
                await chat.close()
                # The call has the session now, and knows its head waits for an answer: an event turn
                # started on it would be taken for that answer (delivered, and never seen by the model).
                assert chat.store.continuation_owner() is call._scheduler and call._scheduler.head_parked
                receipts.append(await call.notify("informe listo"))
                await asyncio.sleep(0.3)
                assert receipts[0].state == "pending" and call._event_turn_blocker() == "run_awaiting_caller"
                assert len(model.calls) == 1 and not _asked(events)

            await _run(call, stt, drive)
        finally:
            await chat.close()


class TestACallWithNoStoreOfItsOwn:
    """A call that never spawned or was notified has no task store: it meets the chat's with its turn."""

    async def test_its_caller_waits_for_the_question_of_a_chat_opened_on_it(self) -> None:
        tools = _wire()
        model = _Model("Hola por voz.", _tool_call("wire_money"), "Hecho.", "Visto por voz.")
        call, stt = _call(model, tools)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _said(call, events, "Hola por voz.")
            await _until(lambda: call._current_turn_task.done())
            assert call._scheduler.store is None
            async with ChatSession(_agent(model, tools), parent_id=call._last_run_context.id) as chat:
                chat.send("paga")
                await _until(lambda: chat.parked)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert chat.parked and not _asked(events) and len(model.calls) == 2
                chat.resume({_approval(chat): True})
                await _said(call, events, "Visto por voz.")

        await _run(call, stt, drive)
        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[3])[-2:] == ["Hecho.", "hablado 2"]

    async def test_its_caller_waits_for_a_chat_restored_on_a_question_after_the_call_began(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"), "Hecho.", "Hola por voz.")
        agent = _agent(model, tools)
        async with ChatSession(agent) as first:
            first.send("paga")
            await _until(lambda: first.parked)
            head = first.head
        call, stt = _call(model, tools, parent_run_id=head)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            async with ChatSession(agent, parent_id=head) as chat:
                assert chat.parked
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert chat.parked and not _asked(events) and len(model.calls) == 1
                chat.resume({_approval(chat): True})
                await _said(call, events, "Hola por voz.")

        await _run(call, stt, drive)
        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[2])[-2:] == ["Hecho.", "hablado 1"]

    async def test_its_first_event_joins_the_chat_s_session(self) -> None:
        model = _Model("Hola por voz.", "Hola por chat.", "Aviso por voz.", "Visto por chat.")
        call, stt = _call(model)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _said(call, events, "Hola por voz.")
            await _until(lambda: call._current_turn_task.done())
            async with ChatSession(_agent(model), parent_id=call._last_run_context.id) as chat:
                chat.send("escrito 1")
                await _turns(chat, 1)
                await call.notify("aviso")
                await _said(call, events, "Aviso por voz.")
                # One session, not a second store for the call next to the chat's.
                assert call._scheduler.store is chat.store
                chat.send("escrito 2")
                await _turns(chat, 2)

        await _run(call, stt, drive)
        assert _in_order(_seen(model.calls[2]), "hablado 1", "Hola por voz.", "escrito 1", "Hola por chat.")
        assert _in_order(_seen(model.calls[3]), "Hola por chat.", "Aviso por voz.", "escrito 2")

    async def test_a_chat_opened_a_run_behind_it_does_not_take_the_call_back(self) -> None:
        model = _Model("Voz uno.", "Voz dos.", "Aviso por voz.")
        call, stt = _call(model)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _said(call, events, "Voz uno.")
            await _until(lambda: call._current_turn_task.done())
            first_run = call._last_run_context.id
            await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
            await _said(call, events, "Voz dos.")
            await _until(lambda: call._current_turn_task.done())
            # A client that had only seen the first reply: its chat branches from there, as a plain
            # run on that id would. The call goes on from its own last turn.
            async with ChatSession(_agent(model), parent_id=first_run) as chat:
                await call.notify("aviso")
                await _said(call, events, "Aviso por voz.")
                assert call._scheduler.store is not chat.store and _owner(chat) is chat._scheduler

        await _run(call, stt, drive)
        assert _in_order(_seen(model.calls[2]), "hablado 1", "Voz uno.", "hablado 2", "Voz dos.")


class TestAQuestionWaitsOnTheCall:
    """A call turn that ended on an approval: what is typed waits for its answer, as for any parked run."""

    async def test_what_is_typed_waits_for_the_call_s_answer(self) -> None:
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Hecho.", "Visto por chat.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())
                assert _owner(chat) is call._scheduler
                chat.send("escrito 2")
                await asyncio.sleep(0.3)
                # Not started: chained onto the call's run it would be taken for the approval's answer.
                assert chat.turns == 1 and len(model.calls) == 2 and len(chat.pending_messages()) == 1
                [asked] = _asked(events)
                resumed = await call.agent(
                    prompt="paga", parent_id=asked.run_id, resume={asked.approval_id: True}
                ).collect()
                assert resumed.status.code == "success"
                await _turns(chat, 2)

            await _run(call, stt, drive)

        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[3])[-2:] == ["Hecho.", "escrito 2"]

    async def test_what_is_typed_waits_for_a_second_question_too(self) -> None:
        # The run that answers the call's approval asks another one: the head is parked again, on a run
        # that is neither channel's own. What was typed still waits for it.
        tools = _two_gated()
        # Call 2 (the resume's second step, which asks again) holds until opened.
        model = _GatedModel(
            "Hola por chat.",
            _tool_call("wire_money", "c1"),
            _tool_call("send_mail", "c2"),
            "Hecho.",
            "Visto por chat.",
            gate_at=2,
        )
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())
                chat.send("escrito 2")
                [asked] = _asked(events)
                first = asyncio.create_task(
                    _collect(call.agent(prompt="paga", parent_id=asked.run_id, resume={asked.approval_id: True}))
                )
                await asyncio.wait_for(model.entered.wait(), 2)
                await asyncio.sleep(0.2)
                # The answer's run is running: the chat does not start behind it, where a second
                # question would find its turn already started.
                assert not chat.busy and len(chat.pending_messages()) == 1
                model.opened.set()
                output, [second] = await asyncio.wait_for(first, 3)
                assert output.status.reason == "approval_required"
                await asyncio.sleep(0.3)
                assert chat.turns == 1 and len(chat.pending_messages()) == 1 and not chat.parked
                answered = await call.agent(
                    prompt="paga", parent_id=output.run_id, resume={second.approval_id: True}
                ).collect()
                assert answered.status.code == "success"
                await _turns(chat, 2)

            await _run(call, stt, drive)

        assert _seen(model.calls[4])[-2:] == ["Hecho.", "escrito 2"]

    async def test_hanging_up_on_it_leaves_the_question_to_the_chat(self) -> None:
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Hecho.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())

            await _run(call, stt, drive)
            await _until(lambda: _owner(chat) is chat._scheduler)
            # The session came back with the call's question: the chat's to answer now (its card, its
            # resume), not a run only the phone that hung up could resume.
            await _until(lambda: chat.parked)
            assert chat.head == call._last_run_context.id and chat.state()["pending_approvals"]
            chat.send("escrito 2")
            await asyncio.sleep(0.2)
            assert chat.turns == 1 and len(model.calls) == 2 and len(chat.pending_messages()) == 1
            chat.resume({_approval(chat): True})
            await _turns(chat, 2)
            assert chat.last_status["code"] == "success" and not chat.parked

        assert tools[0].runs == ["wired"]
        # The answer's turn ran the tool, then took what was typed while the question waited.
        assert _seen(model.calls[2])[-1] == "escrito 2" and len(model.calls) == 3

    async def test_hanging_up_on_it_the_chat_can_drop_the_question(self) -> None:
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Otra cosa, entonces.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())

            await _run(call, stt, drive)
            await _until(lambda: chat.parked)
            chat.send("déjalo, otra cosa", mode="interrupt")
            await _until(lambda: len(model.calls) == 3 and not chat.busy)
            assert chat.last_status["code"] == "success" and not chat.parked

        assert tools[0].runs == []  # abandoned: the gated tool never ran
        assert _seen(model.calls[2])[-1] == "déjalo, otra cosa"

    async def test_a_message_that_already_dropped_it_goes_ahead_once_the_chat_has_the_question(self) -> None:
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Otra cosa, entonces.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())
                # Sent while the question is still the call's: it waits (the chat cannot drop it yet).
                chat.send("déjalo, otra cosa", mode="interrupt")
                await asyncio.sleep(0.2)
                assert len(model.calls) == 2 and len(chat.pending_messages()) == 1

            await _run(call, stt, drive)
            await _until(lambda: len(model.calls) == 3 and not chat.busy)
            assert chat.last_status["code"] == "success" and not chat.parked

        assert tools[0].runs == []
        assert _seen(model.calls[2])[-1] == "déjalo, otra cosa"

    async def test_a_second_question_asked_by_its_answer_is_still_the_call_s(self) -> None:
        tools = _two_gated()
        model = _Model("Hola por chat.", _tool_call("wire_money", "c1"), _tool_call("send_mail", "c2"), "Hecho.")
        second: list[ApprovalEvent] = []
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())
                [asked] = _asked(events)
                output, asks = await _collect(
                    call.agent(prompt="paga", parent_id=asked.run_id, resume={asked.approval_id: True})
                )
                # The phone answered and was asked again.
                assert output.status.reason == "approval_required" and _owner(chat) is call._scheduler
                second.extend(asks)

            await _run(call, stt, drive)
            # Hung up on it: the chat's to answer now, like the first question would have been.
            await _until(lambda: chat.parked)
            assert _approval(chat) == second[0].approval_id
            chat.resume({_approval(chat): True})
            await _turns(chat, 2)
            assert chat.last_status["code"] == "success"

    async def test_a_chat_opened_on_that_second_question_leaves_the_session_with_the_call(self) -> None:
        tools = _two_gated()
        model = _Model(_tool_call("wire_money", "c1"), _tool_call("send_mail", "c2"), "Hecho.")
        call, stt = _call(model, tools)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="paga"))
            await _until(lambda: bool(_asked(events)))
            await _until(lambda: call._current_turn_task.done())
            [asked] = _asked(events)
            output, _asks = await _collect(
                call.agent(prompt="paga", parent_id=asked.run_id, resume={asked.approval_id: True})
            )
            assert output.status.reason == "approval_required"
            async with ChatSession(_agent(model, tools), parent_id=output.run_id) as chat:
                # The call's question line: the call keeps the session, and the chat may answer.
                assert chat.parked and _owner(chat) is call._scheduler and chat._scheduler.lent
                chat.resume({_approval(chat): True})
                await _turns(chat, 1)
                assert chat.last_status["code"] == "success"

        await _run(call, stt, drive)

    async def test_a_second_chat_on_a_session_lent_to_a_call_is_refused(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                # One chat per session, as when the chat owns it: the call hands it back to one channel.
                with pytest.raises(RuntimeError):
                    ChatSession(_agent(model), parent_id=call._last_run_context.id)

            await _run(call, stt, drive)
            await _until(lambda: _owner(chat) is chat._scheduler)  # handed back to the chat it was lent by

    async def test_a_chat_opened_on_the_call_s_question_leaves_the_session_with_the_call(self) -> None:
        tools = _wire()
        model = _Model(_tool_call("wire_money"), "Hecho.")
        call, stt = _call(model, tools)
        chats: list[ChatSession] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="paga"))
            await _until(lambda: bool(_asked(events)))
            await _until(lambda: call._current_turn_task.done())
            chat = ChatSession(_agent(model, tools), parent_id=call._last_run_context.id)
            chats.append(chat)
            await chat.restore()
            # The call asked: it keeps the session (its caller is asked again, as on a call alone).
            # The chat sees the question and may answer it.
            assert _owner(chat) is call._scheduler and chat._scheduler.lent and chat.parked
            chat.resume({_approval(chat): True})
            await _turns(chat, 1)
            assert _owner(chat) is call._scheduler

        await _run(call, stt, drive)
        await chats[0].close()
        assert tools[0].runs == ["wired"] and chats[0].last_status["code"] == "success"


class _GatedStore(InMemoryChatSessionStore):
    """A session store whose next save, once ``hold`` is set, waits for ``gate``."""

    def __init__(self) -> None:
        super().__init__()
        self.hold = False
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def save(self, snapshot, *, expect_version, owner=None):
        if self.hold:
            self.hold = False
            self.entered.set()
            await self.gate.wait()
        return await super().save(snapshot, expect_version=expect_version, owner=owner)


class _AwayStore(InMemoryChatSessionStore):
    """A session store whose saves fail while ``away`` (reads and leases still answer)."""

    def __init__(self) -> None:
        super().__init__()
        self.away = False

    async def save(self, snapshot, *, expect_version, owner=None):
        if self.away:
            raise ConnectionError("session store away")
        return await super().save(snapshot, expect_version=expect_version, owner=owner)


class _Elsewhere(InMemoryTracingProvider):
    """A trace store of its own: what is stored here, a session reading the default one does not find."""

    _storage: dict = {}


class _Flaky(InMemoryTracingProvider):
    """A trace store whose strict reads fail while ``down``."""

    _storage: dict = {}
    down = False

    @classmethod
    async def get_strict(cls, run_context):
        if cls.down:
            raise ConnectionError("trace store away")
        return await super().get_strict(run_context)


class TestAQuestionTheChatCannotRead:
    """The call hung up on its question, and the chat cannot load the run that asked."""

    async def test_a_run_it_does_not_find_does_not_hold_it(self) -> None:
        # The call's runs are stored where the chat does not read. Nothing the chat starts can be taken
        # for that run's answer (its own turns read the same store and find no question either), and
        # nobody could answer it from here: the session goes on.
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Visto por chat.")
        async with ChatSession(_agent(model, tools)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, provider=_Elsewhere, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())

            await _run(call, stt, drive)
            await _until(lambda: _owner(chat) is chat._scheduler)
            chat.send("escrito 2")
            await _turns(chat, 2)
            assert not chat.parked and chat.last_status["code"] == "success"

        assert tools[0].runs == [] and _seen(model.calls[2])[-1] == "escrito 2"

    async def test_a_run_it_cannot_read_yet_is_asked_for_again(self) -> None:
        _Flaky.down = False
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Hecho.")
        async with ChatSession(_agent(model, tools, provider=_Flaky)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, tools, provider=_Flaky, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)))
                await _until(lambda: call._current_turn_task.done())
                _Flaky.down = True

            await _run(call, stt, drive)
            await _until(lambda: _owner(chat) is chat._scheduler)
            chat.send("escrito 2")
            await asyncio.sleep(0.3)
            # Unreadable is not "no question": a turn now could be taken for the approval's answer.
            assert chat.turns == 1 and not chat.parked and len(chat.pending_messages()) == 1
            _Flaky.down = False
            # Nothing new arrives: the session asks again by itself, and has the question.
            await _until(lambda: chat.parked)
            chat.resume({_approval(chat): True})
            await _turns(chat, 2)

        assert tools[0].runs == ["wired"] and _seen(model.calls[2])[-1] == "escrito 2"

    async def test_a_run_stored_late_is_its_question_again_before_a_turn_starts_on_it(self, tmp_path) -> None:
        # The trace store refuses every save during the call's last turn, so the chat finds nothing of
        # the run that asked and goes on. Its end is stored afterwards (a provider sends a failed save
        # again): the next turn would read it parked and be taken for the answer.
        slow = 30.0
        provider = _LateJsonl.configured(_path=tmp_path / "traces.jsonl")
        provider.failing = False
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Hecho.")
        async with ChatSession(_agent(model, tools, provider=provider)) as chat:
            chat.send("escrito 1")
            await _until(lambda: chat.turns >= 1 and not chat.busy, slow)
            call, stt = _call(model, tools, provider=provider, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                provider.failing = True
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)), slow)
                await _until(lambda: call._current_turn_task.done(), slow)

            await _run(call, stt, drive, timeout=4 * slow)
            provider.failing = False
            await _until(lambda: _owner(chat) is chat._scheduler and not chat._scheduler.head_parked, slow)
            assert not chat.parked
            await call._last_run_context._save_trace(barrier=True)  # the save that failed, sent again
            chat.send("escrito 2")
            # Read again before the turn: the question is there now, and is the chat's to answer.
            await _until(lambda: chat.parked, slow)
            assert chat.turns == 1 and len(chat.pending_messages()) == 1
            chat.resume({_approval(chat): True})
            await _until(lambda: chat.turns >= 2 and not chat.busy, slow)

        assert tools[0].runs == ["wired"] and _seen(model.calls[2])[-1] == "escrito 2"

    async def test_a_run_stored_late_brings_the_session_back_from_the_next_call(self, tmp_path) -> None:
        # As above, and a second call has the session by the time the run's end is stored: what is
        # typed would chain onto that run at the call. The chat reads it first and takes the session
        # back for the question.
        slow = 30.0
        provider = _LateJsonl.configured(_path=tmp_path / "traces.jsonl")
        provider.failing = False
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Hecho.")
        async with ChatSession(_agent(model, tools, provider=provider)) as chat:
            chat.send("escrito 1")
            await _until(lambda: chat.turns >= 1 and not chat.busy, slow)
            first, stt = _call(model, tools, provider=provider, parent_run_id=chat.head)

            async def asks(events: list[VoiceSessionEvent]) -> None:
                provider.failing = True
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)), slow)
                await _until(lambda: first._current_turn_task.done(), slow)

            await _run(first, stt, asks, timeout=4 * slow)
            provider.failing = False
            await _until(lambda: _owner(chat) is chat._scheduler and not chat._scheduler.head_parked, slow)
            second, stt2 = _call(model, tools, provider=provider, parent_run_id=chat.head)

            async def drive(_events: list[VoiceSessionEvent]) -> None:
                await second.notify("dato", announce="silent")  # the second call asks for the session
                await _until(lambda: _owner(chat) is second._scheduler, slow)
                await first._last_run_context._save_trace(barrier=True)  # the save that failed, sent again
                chat.send("escrito 2")
                await _until(lambda: chat.parked, slow)
                assert _owner(chat) is chat._scheduler and chat.turns == 1 and len(chat.pending_messages()) == 1
                chat.resume({_approval(chat): True})
                await _until(lambda: chat.turns >= 2 and not chat.busy, slow)

            # In a context of its own, as a call is: the first call ran in this task and left its last
            # run as the ambient one, which a second call started here would take for its own.
            await asyncio.create_task(_run(second, stt2, drive, timeout=4 * slow), context=contextvars.Context())

        assert tools[0].runs == ["wired"] and "escrito 2" in _seen(model.calls[2])

    async def test_a_re_read_that_fails_leaves_the_session_with_the_live_call(self, tmp_path) -> None:
        # The question was dropped (nothing stored of its run) and a second call has the session. The
        # chat's read before its next turn fails: that is "not known yet", not "the question appeared".
        # The chat's turn waits and asks again; the call keeps the session, and nothing is parked.
        slow = 30.0
        provider = _LateJsonl.configured(_path=tmp_path / "traces.jsonl")
        provider.failing = provider.unreadable = False
        tools = _wire()
        model = _Model("Hola por chat.", _tool_call("wire_money"), "Visto por chat.")
        async with ChatSession(_agent(model, tools, provider=provider)) as chat:
            chat.send("escrito 1")
            await _until(lambda: chat.turns >= 1 and not chat.busy, slow)
            first, stt = _call(model, tools, provider=provider, parent_run_id=chat.head)

            async def asks(events: list[VoiceSessionEvent]) -> None:
                provider.failing = True
                await stt.inject(TranscriptEvent(type="committed", text="paga"))
                await _until(lambda: bool(_asked(events)), slow)
                await _until(lambda: first._current_turn_task.done(), slow)

            await _run(first, stt, asks, timeout=4 * slow)
            provider.failing = False
            await _until(lambda: _owner(chat) is chat._scheduler and not chat._scheduler.head_parked, slow)
            second, stt2 = _call(model, tools, provider=provider, parent_run_id=chat.head)

            async def drive(_events: list[VoiceSessionEvent]) -> None:
                await second.notify("dato", announce="silent")  # the second call asks for the session
                await _until(lambda: _owner(chat) is second._scheduler, slow)
                provider.unreadable = True
                chat.send("escrito 2")
                await asyncio.sleep(0.8)  # more than the first wait before it asks again
                assert _owner(chat) is second._scheduler and chat._scheduler.lent
                assert not second._scheduler.head_parked and not chat.parked
                assert chat.turns == 1 and len(chat.pending_messages()) == 1
                provider.unreadable = False
                await _until(lambda: chat.turns >= 2 and not chat.busy, slow)
                assert _owner(chat) is second._scheduler

            await asyncio.create_task(_run(second, stt2, drive, timeout=4 * slow), context=contextvars.Context())

        assert tools[0].runs == [] and "escrito 2" in _seen(model.calls[2])

    async def test_only_a_wait_that_ran_out_makes_the_next_one_longer(self) -> None:
        # Nothing tells the session when the store is back, so it asks again after a wait that grows.
        # A wait cut short by a message is followed by a read at once and does not count: messages typed
        # during an outage must not push the next try out to the longest wait.
        async with ChatSession(_agent(_Model())) as chat:
            chat._work.set()
            await asyncio.wait_for(chat._wait_to_read_again(), 0.3)
            assert chat._question_failures == 0
            chat._work.clear()
            await asyncio.wait_for(chat._wait_to_read_again(), 2)  # the whole first wait (0.5 s)
            assert chat._question_failures == 1


class _LateJsonl(JsonlTracingProvider):
    """A trace file that refuses every save while ``failing``, and every strict read while ``unreadable``."""

    failing = False
    unreadable = False

    @classmethod
    async def _store(cls, run_context) -> None:
        if cls.failing:
            raise ConnectionError("503")
        await super()._store(run_context)

    @classmethod
    async def get_strict(cls, run_context):
        if cls.unreadable:
            raise ConnectionError("trace store away")
        return await super().get_strict(run_context)


async def _collect(stream) -> tuple[OutputEvent, list[ApprovalEvent]]:
    """A run's final event and the approvals it asked for."""
    events = [event async for event in stream]
    return events[-1], [event for event in events if isinstance(event, ApprovalEvent)]


class TestOnlyWhatIsTypedWhileLent:
    async def test_a_wake_waits_for_the_call_to_end(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Rutina hecha.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.wake("rutina de la mañana")
                await asyncio.sleep(0.3)
                # The application's turn, not something typed: it is not the chat's to run during the call.
                assert chat.turns == 1 and len(model.calls) == 2 and chat.due

            await _run(call, stt, drive)
            # Hung up: the wake runs now, on the call's last turn.
            await _turns(chat, 2)

        assert _in_order(_seen(model.calls[2]), "hablado 1", "Hola por voz.")
        assert "rutina de la mañana" in model.calls[2][-1].collect_text()

    async def test_a_call_that_ends_before_it_is_lent_the_session_never_gets_it(self) -> None:
        # Call 1 (the chat's second turn) holds until opened: the call joins and leaves meanwhile.
        model = _GatedModel("Hola por chat.", "Respuesta larga por chat.", "Aviso por chat.", gate_at=1)
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            chat.send("escrito 2")
            await asyncio.wait_for(model.entered.wait(), 2)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(_events: list[VoiceSessionEvent]) -> None:
                # The call asks for the session as it starts: the chat is busy, so it waits to be lent it.
                await _until(lambda: chat._scheduler._borrower is not None)
                assert _owner(chat) is chat._scheduler

            await _run(call, stt, drive)
            assert chat._scheduler._borrower is None  # hung up: no longer waiting to be lent it
            model.opened.set()
            await _turns(chat, 2)
            assert _owner(chat) is chat._scheduler and not chat._scheduler.lent
            chat.store.post_event("aviso")
            await _turns(chat, 3)  # an event is the chat's to answer again: nobody else has the session

    async def test_a_recovery_turn_still_goes_first_during_the_call(self) -> None:
        model = _Model("Hola por chat.", "Hola por voz.", "Revisado.", "Visto por chat.")
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                # The session's own cut turn: settled before what is typed, also while the call has it.
                chat._wakes.append(
                    _Wake(
                        text="revisa el turno cortado",
                        key=None,
                        tool_policy="read_only",
                        silent_ok=False,
                        kind=RECOVERY_KIND,
                    )
                )
                chat.send("escrito 2")
                await _turns(chat, 3)

            await _run(call, stt, drive)

        assert "revisa el turno cortado" in model.calls[2][-1].collect_text()
        assert _seen(model.calls[3])[-1] == "escrito 2"

    async def test_a_recovery_turn_goes_right_after_the_call_s_turn_not_after_what_was_typed(self) -> None:
        # Call 2 (the call's second turn) holds until opened.
        model = _GatedModel(
            "Hola por chat.", "Hola por voz.", "Sigo por voz.", "Revisado.", "Visto por chat.", gate_at=2
        )
        async with ChatSession(_agent(model)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await asyncio.wait_for(model.entered.wait(), 2)
                chat._wakes.append(
                    _Wake(
                        text="revisa el turno cortado",
                        key=None,
                        tool_policy="read_only",
                        silent_ok=False,
                        kind=RECOVERY_KIND,
                    )
                )
                chat.send("escrito 2")
                await asyncio.sleep(0.2)
                model.opened.set()
                await _turns(chat, 3)

            await _run(call, stt, drive)

        assert "revisa el turno cortado" in model.calls[3][-1].collect_text()
        assert _seen(model.calls[4])[-1] == "escrito 2"


def _ends_the_run_of(run_context, prompt: str | None) -> bool:
    """``run_context``'s run has ended and was started with ``prompt`` (the save under way is its last)."""
    root = run_context.root_span()
    return prompt is not None and root is not None and root.t1 is not None and prompt in str(root.input)


class _SlowEnd(InMemoryTracingProvider):
    """Holds the save that ends the run of the prompt ``hold``, once: a turn can be cut exactly there.

    By prompt, not "the next run that ends": the call saves its own turn once more as that turn finishes,
    which may come after the fault is armed.
    """

    _storage: dict = {}
    hold: str | None = None
    entered: asyncio.Event
    gate: asyncio.Event

    @classmethod
    async def _store(cls, run_context) -> None:
        await super()._store(run_context)
        if _ends_the_run_of(run_context, cls.hold):
            cls.hold = None
            cls.entered.set()
            await cls.gate.wait()


class TestATurnCutAsItEnds:
    async def test_a_chat_turn_interrupted_during_its_last_save_does_not_wedge_the_session(self) -> None:
        _SlowEnd.entered, _SlowEnd.gate, _SlowEnd.hold = asyncio.Event(), asyncio.Event(), None
        model = _Model("Hola por chat.", "Hola por voz.", "Cortado.", "Visto por chat.", "Otra vez por voz.")
        async with ChatSession(_agent(model, provider=_SlowEnd)) as chat:
            chat.send("escrito 1")
            await _turns(chat, 1)
            call, stt = _call(model, provider=_SlowEnd, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.")
                await _until(lambda: _owner(chat) is call._scheduler)
                _SlowEnd.hold = "escrito 2"
                chat.send("escrito 2")
                await asyncio.wait_for(_SlowEnd.entered.wait(), 3)
                chat.interrupt()  # lands in the save that ends the chat's run, admitted by the call
                await _until(lambda: not chat.busy)
                _SlowEnd.gate.set()
                # The call no longer counts that run as running: both channels go on.
                assert not call._scheduler.other_turn_active
                chat.send("escrito 3")
                await _turns(chat, 3)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _said(call, events, "Otra vez por voz.")

            await _run(call, stt, drive)


class _SlowJsonl(JsonlTracingProvider):
    """Holds the save that ends the run of the prompt ``hold``, once, before anything of it is written."""

    hold: str | None = None
    entered: asyncio.Event
    gate: asyncio.Event

    @classmethod
    async def _store(cls, run_context) -> None:
        if _ends_the_run_of(run_context, cls.hold):
            cls.hold = None
            cls.entered.set()
            await cls.gate.wait()
        await super()._store(run_context)


class TestATurnCutAsItEndsOnAStoredTrace:
    async def test_the_turn_waiting_behind_it_reads_what_it_answered(self, tmp_path) -> None:
        # Every save goes to a file: deadlines sized for a slow one (TIMBAL_TEST_SLOW_IO), not for a tick.
        slow = 30.0
        provider = _SlowJsonl.configured(_path=tmp_path / "traces.jsonl")
        provider.entered, provider.gate, provider.hold = asyncio.Event(), asyncio.Event(), None
        model = _Model("Hola por chat.", "Hola por voz.", "Respondido por chat.", "Visto por voz.")
        async with ChatSession(_agent(model, provider=provider)) as chat:
            chat.send("escrito 1")
            await _until(lambda: chat.turns >= 1 and not chat.busy, slow)
            call, stt = _call(model, provider=provider, parent_run_id=chat.head)

            async def drive(events: list[VoiceSessionEvent]) -> None:
                await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
                await _said(call, events, "Hola por voz.", slow)
                await _until(lambda: _owner(chat) is call._scheduler, slow)
                provider.hold = "escrito 2"
                chat.send("escrito 2")
                await asyncio.wait_for(provider.entered.wait(), slow)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: call._scheduler.busy, slow)
                chat.interrupt()  # lands in the save that ends the chat's run: nothing of it is stored yet
                provider.gate.set()
                await _said(call, events, "Visto por voz.", slow)

            await _run(call, stt, drive, timeout=4 * slow)

        # The call's turn was let in only once the cut run was stored: it continues from the chat's answer.
        assert _seen(model.calls[3])[-3:] == ["escrito 2", "Respondido por chat.", "hablado 2"]
