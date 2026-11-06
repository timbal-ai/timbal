"""Per-tool interruption_mode on a voice call.

Modes follow ElevenLabs Agents (allow / disable_during_tool /
disable_during_tool_and_turn; strictest mode covers a parallel batch). Speech
during a protected window is kept for the next turn, which is LiveKit's
allow_interruptions=False with discard_audio_if_uninterruptible=False.
"""

from __future__ import annotations

import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    AgentStatus,
    AudioOutput,
    SessionInterrupted,
    TranscriptEvent,
    VoiceSession,
    VoiceSessionEvent,
)

from .test_session import DelayedMockSTT, FakePlaybackTracker, SlowMockTTS
from .test_voice_events import _run, _session, _texts, _until

REPLY = "Hello world how are you doing today my friend"
DURING_TOOL = "also check order 102"
DURING_REPLY = "email me the receipt please"
BARGE = "Actually I want to ask about something else entirely"


def _call(name: str, call_id: str) -> ToolUseContent:
    return ToolUseContent(id=call_id, name=name, input={})


def _tool_message(*calls: ToolUseContent) -> Message:
    return Message(role="assistant", content=list(calls), stop_reason="tool_use")


class _Script:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.model = TestModel(handler=self._reply)

    def _reply(self, messages: list[Message]) -> str | Message:
        self.calls.append(list(messages))
        last = messages[-1]
        text = last.collect_text() if last.role == "user" else ""
        if DURING_TOOL in text or DURING_REPLY in text or BARGE in text:
            return "Noted."
        if any(block.type == "tool_result" for message in messages for block in message.content):
            return REPLY
        names = [block.name for block in last.content if isinstance(block, ToolUseContent)]
        if not names:
            # First user turn: call every tool the agent was given.
            return _tool_message(
                *(_call(name, f"c{index}") for index, name in enumerate(self._tool_names(messages), start=1))
            )
        return REPLY

    @staticmethod
    def _tool_names(messages: list[Message]) -> list[str]:
        # The handler only sees messages, not the agent. Tests with one tool
        # call "lookup"; the parallel test calls both by name from the user text.
        user = messages[-1].collect_text()
        if "both" in user:
            return ["confirm", "note"]
        return ["lookup"]


def _voice(
    script: _Script,
    tools: list,
    *,
    tracker: FakePlaybackTracker | None = None,
    tts: SlowMockTTS | None = None,
) -> tuple[VoiceSession, DelayedMockSTT]:
    session, stt, _tts = _session(
        script,  # type: ignore[arg-type]
        tools=tools,
        tracker=tracker,
        tts=tts or SlowMockTTS(delay=0.02, chunk=b"\x00" * 100, num_chunks=4),
    )
    return session, stt


async def _status(events: list[VoiceSessionEvent], name: str) -> None:
    await _until(lambda: any(isinstance(event, AgentStatus) and name in event.text for event in events))


async def _idle(session: VoiceSession) -> None:
    """Wait until the carried turn and its playback have finished.

    ``close()`` interrupts audio that is still queued, and that emits
    ``SessionInterrupted``. Tests that assert no barge-in have to outlast it.
    """
    await _until(
        lambda: (
            (session._current_turn_task is None or session._current_turn_task.done())
            and not session._is_speaking
            and not session.playback.is_playing
            and not session._deferred_user_text
        )
    )


@pytest.mark.asyncio
async def test_allow_cancels_the_tool_on_barge_in() -> None:
    """Default mode: a commit while the tool runs cancels it."""
    cancelled = asyncio.Event()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def lookup() -> str:
        started.set()
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finished.set()
        return "shipped"

    script = _Script()
    session, stt = _voice(script, [lookup])

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="check 101"))
        await _until(lambda: started.is_set())
        await stt.inject(TranscriptEvent(type="committed", text=BARGE))
        await _until(lambda: cancelled.is_set())
        await _until(lambda: "Noted." in _texts(events))

    events = await _run(session, stt, drive)
    assert finished.is_set() is False
    assert any(isinstance(event, SessionInterrupted) for event in events)
    assert any(entry.role == "user" and BARGE in entry.text for entry in session.transcript)


@pytest.mark.asyncio
async def test_disable_during_tool_finishes_and_carries_speech() -> None:
    """Barge-in during the tool does not cancel it. The words open the next turn."""
    finished = asyncio.Event()

    async def lookup() -> str:
        await asyncio.sleep(0.25)
        finished.set()
        return "101 shipped"

    tool = Tool(handler=lookup, interruption_mode="disable_during_tool")
    script = _Script()
    session, stt = _voice(script, [tool])

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="check 101"))
        await _status(events, "lookup")
        await stt.inject(TranscriptEvent(type="committed", text=DURING_TOOL))
        await _until(lambda: finished.is_set(), timeout=2)
        # The reply after the tool is spoken in full, then the carried words.
        await _until(lambda: _texts(events)[:2] == [REPLY, "Noted."], timeout=4)
        await _idle(session)

    events = await _run(session, stt, drive, timeout=8)
    assert not any(isinstance(event, SessionInterrupted) for event in events)
    user_lines = [entry.text for entry in session.transcript if entry.role == "user"]
    assert user_lines[:2] == ["check 101", DURING_TOOL]
    carried = script.calls[-1]
    assert carried[-1].role == "user" and carried[-1].collect_text() == DURING_TOOL
    assert any(message.role == "assistant" and message.collect_text() == REPLY for message in carried)
    assert session.agent._tool_interruption_modes == {"lookup": "disable_during_tool"}


@pytest.mark.asyncio
async def test_barge_in_after_disable_during_tool_truncates_to_the_heard_prefix() -> None:
    """Once the tool has returned, barge-in works again and memory keeps the heard prefix."""
    finished = asyncio.Event()

    async def lookup() -> str:
        finished.set()
        return "101 shipped"

    tool = Tool(handler=lookup, interruption_mode="disable_during_tool")
    script = _Script()
    tracker = FakePlaybackTracker()
    session, stt = _voice(
        script,
        [tool],
        tracker=tracker,
        tts=SlowMockTTS(delay=0.05, chunk=b"\x00\x01" * 50, num_chunks=6),
    )

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="check 101"))
        await _until(lambda: finished.is_set())
        await _until(lambda: any(isinstance(event, AudioOutput) for event in events))
        tracker.playing = True
        tracker.played = 50
        await stt.inject(TranscriptEvent(type="committed", text=BARGE))
        await _until(lambda: any(isinstance(event, SessionInterrupted) for event in events))
        await _until(lambda: "Noted." in _texts(events))

    events = await _run(session, stt, drive)
    interrupted = next(event for event in events if isinstance(event, SessionInterrupted))
    assert interrupted.heard_text
    assert interrupted.heard_text != REPLY
    assert REPLY.startswith(interrupted.heard_text)
    memory = session._last_run_context.root_span().memory
    heard = next(
        message
        for message in memory
        if message.role == "assistant" and message.collect_text() == interrupted.heard_text
    )
    assert heard.metadata["interrupted"] is True


@pytest.mark.asyncio
async def test_disable_during_tool_and_turn_blocks_barge_in_and_carries_speech() -> None:
    """A partial and a commit during the reply do not cut it. The commit opens the next turn."""
    finished = asyncio.Event()

    async def lookup() -> str:
        await asyncio.sleep(0.05)
        finished.set()
        return "confirmed"

    tool = Tool(handler=lookup, interruption_mode="disable_during_tool_and_turn")
    script = _Script()
    tracker = FakePlaybackTracker()
    session, stt = _voice(
        script,
        [tool],
        tracker=tracker,
        tts=SlowMockTTS(delay=0.05, chunk=b"\x00" * 80, num_chunks=4),
    )

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="confirm the payment"))
        await _until(lambda: finished.is_set())
        await _until(lambda: any(isinstance(event, AudioOutput) for event in events))
        tracker.playing = True
        await stt.inject(TranscriptEvent(type="partial", text="please stop talking now"))
        await asyncio.sleep(0.05)
        assert not any(isinstance(event, SessionInterrupted) for event in events)
        await stt.inject(TranscriptEvent(type="committed", text=DURING_REPLY))
        await asyncio.sleep(0.05)
        assert not any(isinstance(event, SessionInterrupted) for event in events)
        # Playback of the protected reply has to finish before the carried turn.
        tracker.playing = False
        await _until(lambda: _texts(events)[:2] == [REPLY, "Noted."], timeout=4)
        await _idle(session)

    events = await _run(session, stt, drive, timeout=8)
    assert not any(isinstance(event, SessionInterrupted) for event in events)
    assert [entry.text for entry in session.transcript if entry.role == "user"][:2] == [
        "confirm the payment",
        DURING_REPLY,
    ]
    assert script.calls[-1][-1].collect_text() == DURING_REPLY
    # The protected reply is stored whole — nothing was cut back to a heard prefix.
    reply = next(
        message for message in script.calls[-1] if message.role == "assistant" and message.collect_text() == REPLY
    )
    assert reply.metadata.get("interrupted") is not True


@pytest.mark.asyncio
async def test_parallel_batch_uses_the_strictest_mode() -> None:
    """An allow tool running beside disable_during_tool stays protected until both finish."""
    order: list[str] = []

    async def confirm() -> str:
        await asyncio.sleep(0.05)
        order.append("confirm")
        return "confirmed"

    async def note() -> str:
        await asyncio.sleep(0.4)
        order.append("note")
        return "noted"

    tools = [
        Tool(handler=confirm, interruption_mode="disable_during_tool"),
        Tool(handler=note),
    ]
    script = _Script()
    session, stt = _voice(script, tools)

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="run both now"))
        await _until(lambda: "confirm" in order)
        await stt.inject(TranscriptEvent(type="committed", text=DURING_TOOL))
        await _until(lambda: "note" in order, timeout=2)
        await _until(lambda: _texts(events)[:2] == [REPLY, "Noted."], timeout=4)
        await _idle(session)

    events = await _run(session, stt, drive, timeout=8)
    assert order == ["confirm", "note"]
    assert not any(isinstance(event, SessionInterrupted) for event in events)


@pytest.mark.asyncio
async def test_speech_during_tool_is_merged_into_a_later_barge_in() -> None:
    """Words from the protected window are not dropped when the reply is interrupted later."""
    finished = asyncio.Event()

    async def lookup() -> str:
        await asyncio.sleep(0.2)
        finished.set()
        return "101 shipped"

    tool = Tool(handler=lookup, interruption_mode="disable_during_tool")
    script = _Script()
    tracker = FakePlaybackTracker()
    session, stt = _voice(script, [tool], tracker=tracker)

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="check 101"))
        await _status(events, "lookup")
        await stt.inject(TranscriptEvent(type="committed", text=DURING_TOOL))
        await _until(lambda: finished.is_set())
        await _until(lambda: any(isinstance(event, AudioOutput) for event in events))
        tracker.playing = True
        tracker.played = 40
        await stt.inject(TranscriptEvent(type="committed", text=BARGE))
        await _until(lambda: "Noted." in _texts(events))

    await _run(session, stt, drive)
    assert script.calls[-1][-1].collect_text() == f"{DURING_TOOL} {BARGE}"


@pytest.mark.asyncio
async def test_sub_agent_used_as_a_tool_can_block_barge_in() -> None:
    """A nested agent's own tools do not drive the call. The nested agent's mode does."""
    started = asyncio.Event()
    finished = asyncio.Event()

    async def quote() -> str:
        started.set()
        await asyncio.sleep(0.25)
        finished.set()
        return "42 dollars"

    pricing = Agent(
        name="pricing",
        description="Price a job",
        model=TestModel(responses=[_tool_message(_call("quote", "q1")), "The job is 42 dollars."]),
        tools=[quote],
        interruption_mode="disable_during_tool",
    )

    class _Outer:
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []
            self.model = TestModel(handler=self._reply)

        def _reply(self, messages: list[Message]) -> str | Message:
            self.calls.append(list(messages))
            text = messages[-1].collect_text() if messages[-1].role == "user" else ""
            if DURING_TOOL in text:
                return "Noted."
            if any(block.type == "tool_result" for message in messages for block in message.content):
                return REPLY
            # An agent used as a tool takes a prompt, unlike a bare function.
            return Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="pricing", input={"prompt": "price it"})],
                stop_reason="tool_use",
            )

    session, stt = _voice(_Outer(), [pricing])

    async def drive(events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="price the job"))
        await _until(lambda: started.is_set())
        await stt.inject(TranscriptEvent(type="committed", text=DURING_TOOL))
        await _until(lambda: finished.is_set(), timeout=2)
        await _until(lambda: _texts(events)[:2] == [REPLY, "Noted."], timeout=4)
        await _idle(session)

    events = await _run(session, stt, drive, timeout=8)
    assert not any(isinstance(event, SessionInterrupted) for event in events)
    assert session.agent._tool_interruption_modes == {"pricing": "disable_during_tool"}


def test_and_turn_pause_waits_for_the_turn_that_finishes_the_tool() -> None:
    """The wait stays open. A deferred turn does not spend the protection."""
    script = _Script()
    session, _stt = _voice(script, [])
    session.agent._tool_interruption_modes = {"wire": "disable_during_tool_and_turn"}
    session._is_speaking = True
    session._note_tool_call("wire", "c1")
    assert session._interruptions_suppressed()
    session._note_tool_output("voice_events.wire", paused=True)
    # Parked: the caller can speak, including over leftover playback.
    session.playback._playing_until = session.playback._clock() + 5
    assert not session._interruptions_suppressed()
    # Words captured during the tool open their own turn and leave the flag.
    session._reset_tool_interruption()
    session._is_speaking = True
    assert session._and_turn_after_pause
    assert not session._interruptions_suppressed()
    # The turn that finishes the tool protects the reply, and only that reply.
    session._note_tool_call("wire", "c2")
    session._note_tool_output("voice_events.wire", paused=False)
    assert session._interruptions_suppressed()
    assert not session._and_turn_after_pause


def test_unfinished_tool_batch_does_not_lock_the_call() -> None:
    script = _Script()
    session, _stt = _voice(script, [])
    session.agent._tool_interruption_modes = {"wire": "disable_during_tool"}
    session._is_speaking = True
    session._note_tool_call("wire", "c1")
    assert session._interruptions_suppressed()
    session._release_unfinished_tool_batch()
    session._is_speaking = False
    assert not session._interruptions_suppressed()


@pytest.mark.asyncio
async def test_hangup_keeps_speech_held_during_a_protected_window() -> None:
    script = _Script()
    session, _stt = _voice(script, [])
    session._deferred_user_text = "also check"
    session._held_user_text = DURING_TOOL
    await session.close()
    # The hold is a refinement of the deferred fragment, not a second copy.
    assert [entry.text for entry in session.transcript if entry.role == "user"] == [DURING_TOOL]


def test_dict_tool_definition_accepts_interruption_mode() -> None:
    def lookup() -> str:
        """Look up an order."""
        return "ok"

    agent = Agent(
        name="desk",
        model=TestModel(responses=["ok"]),
        tools=[{"handler": lookup, "interruption_mode": "disable_during_tool"}],
    )
    assert agent.tools[0].interruption_mode == "disable_during_tool"


@pytest.mark.asyncio
async def test_deferred_speech_is_kept_on_hangup() -> None:
    started = asyncio.Event()

    async def lookup() -> str:
        started.set()
        await asyncio.sleep(2)
        return "late"

    tool = Tool(handler=lookup, interruption_mode="disable_during_tool")
    script = _Script()
    session, stt = _voice(script, [tool])

    async def drive(_events: list[VoiceSessionEvent]) -> None:
        await stt.inject(TranscriptEvent(type="committed", text="check 101"))
        await _until(lambda: started.is_set())
        await stt.inject(TranscriptEvent(type="committed", text=DURING_TOOL))
        await _until(lambda: session._deferred_user_text == DURING_TOOL)
        await session.close()

    await _run(session, stt, drive)
    assert any(entry.role == "user" and entry.text == DURING_TOOL for entry in session.transcript)
