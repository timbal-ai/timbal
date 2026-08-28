"""Cross-modality continuity: a voice call and a text session are one run chain.

Voice → text: ``AgentTextDone.run_id`` hands the client a current pointer after
every completed turn. Text → voice: ``VoiceSession(parent_run_id=...)`` makes
the call's first turn a child of an existing run, after which the session's own
``_last_run_context`` chaining takes over unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import get_run_context
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    AgentTextDone,
    SessionStarted,
    TranscriptEvent,
    VoiceSession,
    VoiceSessionEvent,
)

from .test_session import DelayedMockSTT, MockSTT, MockTTS, _collect_events, _HungAgent


async def _text_turn(agent: Agent, prompt: str):
    """Run a text turn in its own task so its RunContext cannot leak into the
    test task's contextvars — a real voice server never has the text session's
    context ambient, and leakage would let fork-chaining mask a broken seed."""
    return await asyncio.create_task(agent(prompt=prompt).collect())


def _make_session(
    *,
    stt_script: list[TranscriptEvent] | None = None,
    agent: Agent | None = None,
    parent_run_id: str | None = None,
) -> VoiceSession:
    return VoiceSession(
        agent=agent or Agent(name="continuity", model=TestModel(responses=["Hello back!"]), tools=[]),
        stt=MockSTT(script=stt_script),
        tts=MockTTS(),
        turn_detector="heuristic",
        parent_run_id=parent_run_id,
    )


class TestRunIdOnAgentTextDone:
    async def test_done_carries_the_run_that_produced_the_turn(self) -> None:
        session = _make_session(stt_script=[TranscriptEvent(type="committed", text="Hi")])
        events = await _collect_events(session)

        done = next(e for e in events if isinstance(e, AgentTextDone))
        assert done.run_id is not None
        # The same run the trace was written under — what a client passes as
        # parent_id to continue this conversation over HTTP.
        assert done.run_id == session._last_run_context.id

    async def test_two_turns_yield_distinct_chained_run_ids(self) -> None:
        agent = Agent(name="continuity", model=TestModel(responses=["One.", "Two."]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(agent=agent, stt=stt, tts=MockTTS(), turn_detector="heuristic")

        events: list[VoiceSessionEvent] = []
        run_ids: list[str | None] = []
        parent_ids: list[str | None] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            prev = session._last_run_context
            for text in ("First question", "Second question"):
                await stt.inject(TranscriptEvent(type="committed", text=text))
                # AgentTextDone is emitted *before* the turn's finally assigns
                # _last_run_context — wait on the assignment itself, or a fast
                # scheduler (windows CI) reads the previous turn's value.
                while session._last_run_context is prev:
                    await asyncio.sleep(0.01)
                prev = session._last_run_context
                run_ids.append(prev.id)
                parent_ids.append(prev.parent_id)
                await asyncio.sleep(0.05)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty_audio())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)

        dones = [e for e in events if isinstance(e, AgentTextDone)]
        assert [d.run_id for d in dones] == run_ids
        assert run_ids[0] != run_ids[1]
        # The second turn is a child of the first — the thread is walkable.
        assert parent_ids[1] == run_ids[0]

    async def test_the_wire_payload_carries_it(self) -> None:
        from timbal.server.voice import event_to_payloads

        (payload,) = event_to_payloads(AgentTextDone(text="hi", run_id="r-1"), session=None, meta={})
        assert payload == {"type": "agent_text_done", "text": "hi", "run_id": "r-1"}

    def test_text_with_no_run_behind_it_has_none(self) -> None:
        # The greeting and realtime construct the event without a run_id; the
        # default must stay None so those paths need no knowledge of runs.
        assert AgentTextDone(text="hello").run_id is None


class TestParentRunSeed:
    async def test_first_turn_is_a_child_of_the_seed(self) -> None:
        """Text → voice: the call joins an existing conversation, memory included."""
        text_agent = Agent(name="continuity", model=TestModel(responses=["Noted: blue."]), tools=[])
        first = await _text_turn(text_agent, "my favorite color is blue")
        assert first.status.code == "success"

        message_counts: list[int] = []

        def _handler(messages):
            message_counts.append(len(messages))
            return "Your color is blue."

        voice_agent = Agent(name="continuity", model=TestModel(handler=_handler), tools=[])
        session = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="what is my color?")],
            agent=voice_agent,
            parent_run_id=first.run_id,
        )
        events = await _collect_events(session)

        assert session._last_run_context.parent_id == first.run_id
        # The seed is not a bare pointer: turn 1 resolved the text session's
        # memory, so the model saw the prior exchange, not a fresh thread.
        assert message_counts[0] == 3
        done = next(e for e in events if isinstance(e, AgentTextDone))
        assert done.run_id == session._last_run_context.id
        assert done.run_id != first.run_id

    async def test_second_turn_chains_from_the_first_not_the_seed(self) -> None:
        text_agent = Agent(name="continuity", model=TestModel(responses=["ok"]), tools=[])
        first = await _text_turn(text_agent, "hello from text")

        agent = Agent(name="continuity", model=TestModel(responses=["One.", "Two."]), tools=[])
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=agent, stt=stt, tts=MockTTS(), turn_detector="heuristic", parent_run_id=first.run_id
        )

        events: list[VoiceSessionEvent] = []
        chain: list[tuple[str, str | None]] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            prev = session._last_run_context
            for text in ("turn one", "turn two"):
                await stt.inject(TranscriptEvent(type="committed", text=text))
                # Wait on the _last_run_context assignment (turn finally), not
                # on AgentTextDone, which the turn emits before retiring.
                while session._last_run_context is prev:
                    await asyncio.sleep(0.01)
                prev = session._last_run_context
                chain.append((prev.id, prev.parent_id))
                await asyncio.sleep(0.05)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty_audio())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)

        assert chain[0][1] == first.run_id, "turn 1 joins the seeded conversation"
        assert chain[1][1] == chain[0][0], "turn 2 chains from turn 1, not from the seed"

    async def test_without_a_seed_the_first_turn_is_a_root(self) -> None:
        session = _make_session(stt_script=[TranscriptEvent(type="committed", text="Hi")])
        await _collect_events(session)
        assert session._last_run_context.parent_id is None

    async def test_seed_and_call_context_ride_the_same_context(self) -> None:
        """The empty-trace reuse path must carry both: identity on the session
        bag and the parent on the context, or one would silently drop the other."""
        text_agent = Agent(name="continuity", model=TestModel(responses=["ok"]), tools=[])
        first = await _text_turn(text_agent, "hi")

        session = _make_session(agent=Agent(name="continuity", model=TestModel(responses=["ok"]), tools=[]))
        session.call_context = {"rep_id": "R001"}
        session.parent_run_id = first.run_id
        await session._seed_call_context()

        ctx = get_run_context()
        assert ctx is not None
        assert ctx.parent_id == first.run_id
        assert (await ctx.get_session())["rep_id"] == "R001"
        assert not ctx._trace, "empty trace is what lets runnable.py reuse this context on turn 1"

    def test_blank_seed_is_normalized_to_none(self) -> None:
        session = _make_session(parent_run_id="   ")
        assert session.parent_run_id is None


class TestSeededPreRunTimeout:
    async def test_a_turn_that_never_started_does_not_report_the_seed(self) -> None:
        """The seed context exists before any run does. A turn that dies before
        the first ``__anext__`` must report run_id None — the seed's id points
        at nothing persisted — and must NOT adopt the seed as _last_run_context,
        or the retry's genuine run (which reuses the empty-trace seed) would be
        identical to _last_run_context and report None itself."""
        text_agent = Agent(name="continuity", model=TestModel(responses=["ok"]), tools=[])
        first = await _text_turn(text_agent, "hello from text")

        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=_HungAgent(),  # type: ignore[arg-type]
            stt=stt,
            tts=MockTTS(),
            turn_detector="heuristic",
            turn_timeout_secs=0.15,
            turn_timeout_fallback="Sorry, try again.",
            parent_run_id=first.run_id,
        )
        live_agent = Agent(name="continuity", model=TestModel(responses=["Recovered."]), tools=[])

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Are you there?"))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)
            while session._current_turn_task is not None and not session._current_turn_task.done():
                await asyncio.sleep(0.01)
            # No run happened — the failed turn must not have adopted the seed.
            assert session._last_run_context is None

            session.agent = live_agent
            await stt.inject(TranscriptEvent(type="committed", text="Hello again"))
            while session._last_run_context is None:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty_audio())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)

        dones = [e for e in events if isinstance(e, AgentTextDone)]
        assert dones[0].text == "Sorry, try again."
        assert dones[0].run_id is None, "no run started — the seed id points at nothing persisted"
        # The retry ran for real, on the surviving seed: it reports its run and
        # still joined the conversation the dial named.
        assert dones[1].run_id is not None
        assert dones[1].run_id == session._last_run_context.id
        assert session._last_run_context.parent_id == first.run_id


class TestTimeoutFallbackRunId:
    async def test_fallback_done_still_names_the_hung_run(self) -> None:
        """The apology is session-synthesized, but the timed-out run persisted —
        its id still points at the thread, so 'continue in chat' works even for
        the turn that went wrong."""

        async def stall() -> str:
            """Hang forever."""
            await asyncio.sleep(30)
            return "never"

        tool_call = Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name="stall", input={})],
            stop_reason="tool_use",
        )
        agent = Agent(name="continuity", model=TestModel(responses=[tool_call, "done"]), tools=[stall])
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=agent,
            stt=stt,
            tts=MockTTS(),
            turn_detector="heuristic",
            turn_timeout_secs=0.2,
            turn_timeout_fallback="Sorry, try again.",
        )

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Are you there?"))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)
            while session._current_turn_task is not None and not session._current_turn_task.done():
                await asyncio.sleep(0.01)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty_audio())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)

        done = next(e for e in events if isinstance(e, AgentTextDone))
        assert done.text == "Sorry, try again."
        assert done.run_id is not None
        assert done.run_id == session._last_run_context.id
