"""One conversation typed in a ChatSession and spoken on a VoiceSession call at the same time."""

import asyncio

import pytest
from timbal import Agent
from timbal.core.chat_session import RECOVERY_KIND, ChatSession, _Wake
from timbal.core.tool import Tool
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
                # Not started: what the caller's turn ends on (a question) decides whether it may.
                assert not chat.busy and len(chat.pending_messages()) == 1
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
                await _until(lambda: "El informe está listo." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
                await _until(lambda: _owner(chat) is call._scheduler)
                chat.send("paga")
                await _until(lambda: chat.parked)
                # The question is the chat's: so is the session while it waits.
                assert _owner(chat) is chat._scheduler and not chat._scheduler.lent
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert len(model.calls) == 3 and not _asked(events)
                chat.resume({_approval(chat): True})
                await _until(lambda: "Visto por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))

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
            await _until(lambda: "Hola por voz." in _texts(events))
            await _until(lambda: call._current_turn_task.done())
            assert call._scheduler.store is None
            async with ChatSession(_agent(model, tools), parent_id=call._last_run_context.id) as chat:
                chat.send("paga")
                await _until(lambda: chat.parked)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: chat._scheduler.other_turn_active)
                assert chat.parked and not _asked(events) and len(model.calls) == 2
                chat.resume({_approval(chat): True})
                await _until(lambda: "Visto por voz." in _texts(events))

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
                await _until(lambda: "Hola por voz." in _texts(events))

        await _run(call, stt, drive)
        assert tools[0].runs == ["wired"]
        assert _seen(model.calls[2])[-2:] == ["Hecho.", "hablado 1"]

    async def test_its_first_event_joins_the_chat_s_session(self) -> None:
        model = _Model("Hola por voz.", "Hola por chat.", "Aviso por voz.", "Visto por chat.")
        call, stt = _call(model)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="hablado 1"))
            await _until(lambda: "Hola por voz." in _texts(events))
            await _until(lambda: call._current_turn_task.done())
            async with ChatSession(_agent(model), parent_id=call._last_run_context.id) as chat:
                chat.send("escrito 1")
                await _turns(chat, 1)
                await call.notify("aviso")
                await _until(lambda: "Aviso por voz." in _texts(events))
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
            await _until(lambda: "Voz uno." in _texts(events))
            await _until(lambda: call._current_turn_task.done())
            first_run = call._last_run_context.id
            await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
            await _until(lambda: "Voz dos." in _texts(events))
            await _until(lambda: call._current_turn_task.done())
            # A client that had only seen the first reply: its chat branches from there, as a plain
            # run on that id would. The call goes on from its own last turn.
            async with ChatSession(_agent(model), parent_id=first_run) as chat:
                await call.notify("aviso")
                await _until(lambda: "Aviso por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
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
                await _until(lambda: "Hola por voz." in _texts(events))
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
                await _until(lambda: "Otra vez por voz." in _texts(events))

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
                await _until(lambda: "Hola por voz." in _texts(events), slow)
                await _until(lambda: _owner(chat) is call._scheduler, slow)
                provider.hold = "escrito 2"
                chat.send("escrito 2")
                await asyncio.wait_for(provider.entered.wait(), slow)
                await stt.inject(TranscriptEvent(type="committed", text="hablado 2"))
                await _until(lambda: call._scheduler.busy, slow)
                chat.interrupt()  # lands in the save that ends the chat's run: nothing of it is stored yet
                provider.gate.set()
                await _until(lambda: "Visto por voz." in _texts(events), slow)

            await _run(call, stt, drive, timeout=4 * slow)

        # The call's turn was let in only once the cut run was stored: it continues from the chat's answer.
        assert _seen(model.calls[3])[-3:] == ["escrito 2", "Respondido por chat.", "hablado 2"]
