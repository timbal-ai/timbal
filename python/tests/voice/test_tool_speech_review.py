"""Regression cases from the interruption and tool-boundary review of #231."""

import asyncio
from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message
from timbal.voice import VoiceSession
from timbal.voice.playback import map_played_bytes_to_text

from .test_session import MockSTT, MockTTS, MockTTSStream, StreamingMockTTS


def make_session(tts=None, **kwargs):
    return VoiceSession(
        agent=Agent(name="desk", model=TestModel(responses=["done"])),
        stt=MockSTT(),
        tts=tts or MockTTS(),
        turn_detector="heuristic",
        **kwargs,
    )


class HungStream(MockTTSStream):
    def __init__(self):
        super().__init__()
        self.end_started = asyncio.Event()

    async def end(self):
        self.ended = True
        self.end_started.set()
        # No final audio: the consumer remains blocked until abort().


@pytest.mark.parametrize("at_boundary", [False, True])
async def test_interrupt_aborts_stream_and_pump_while_draining(at_boundary):
    session = make_session(tool_beat_secs=0.1)
    stream = HungStream()
    session._turn_tts_stream = stream
    session._turn_tts_pump = asyncio.create_task(session._pump_tts_stream(stream, ["Checking", 0]))
    pump = session._turn_tts_pump
    session._turn_tts_scheduled_text = "Checking"
    if at_boundary:
        session._schedule_tool_boundary()
    session._current_turn_task = asyncio.create_task(session._await_tts_chain())
    try:
        await asyncio.wait_for(stream.end_started.wait(), 1)
        await asyncio.wait_for(session.interrupt(), 1)
        assert stream.aborted
        assert pump.done()
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.parametrize("during_feed", [False, True])
async def test_timeout_preserves_stream_until_cleanup_without_second_drain(during_feed):
    import time

    session = make_session()
    stream = HungStream()
    session._turn_tts_stream = stream
    session._turn_tts_pump = asyncio.create_task(session._pump_tts_stream(stream, ["Checking", 0]))
    pump = session._turn_tts_pump
    if during_feed:
        session._tts_tail = asyncio.create_task(asyncio.Event().wait())
    session._turn_deadline = time.monotonic() + 0.02

    async def expire():
        with pytest.raises(TimeoutError):
            await session._await_turn_deadline(session._await_tts_chain())

    task = asyncio.create_task(expire())
    try:
        done, _ = await asyncio.wait([task], timeout=0.5)
        assert task in done, "expired turn entered another stream drain"
        await task
        await session._abort_tts_stream()
        assert stream.aborted
        assert pump.done()
    finally:
        task.cancel()
        pump.cancel()
        if session._tts_tail is not None:
            session._tts_tail.cancel()
            await asyncio.gather(session._tts_tail, return_exceptions=True)
        await asyncio.gather(task, pump, return_exceptions=True)


@pytest.mark.parametrize("heard_post", [0, 10, 30])
async def test_multistep_memory_keeps_only_heard_text_per_iteration(heard_post):
    session = make_session()
    pre = Message(
        role="assistant", content=[TextContent(text="Checking…"), ToolUseContent(id="lookup", name="lookup", input={})]
    )
    result = Message(role="tool", content=[TextContent(text="tool result")])
    post = Message(role="assistant", content=[TextContent(text="Order has shipped tomorrow.")])
    root = SimpleNamespace(memory=[pre, result, post], output=post)
    session._turn_assistant_text = "Checking… Order has shipped tomorrow."
    session._turn_iteration_prefix = "Checking… "
    session._turn_completed_replies = [(pre, ""), (post, "Checking… ")]
    session._turn_tts_segment_records = [["Checking…", 10], [" Order has shipped tomorrow.", 30]]
    session._turn_heard_bytes = 10 + heard_post
    assert session._apply_interruption_truncation(SimpleNamespace(root_span=lambda: root))
    expected = map_played_bytes_to_text([(" Order has shipped tomorrow.", 30)], heard_post)
    assert root.memory[0].collect_text() == "Checking…"
    assert root.memory[0].content[-1] is pre.content[-1]
    assert root.output.collect_text() == expected
    if heard_post:
        assert root.memory[-1].collect_text() == expected
    else:
        assert root.memory[-1] is result
    assert post.collect_text() == "Order has shipped tomorrow.", "LLM span must retain its original output"


class FlushStream(MockTTSStream):
    def __init__(self):
        super().__init__()
        self.flushes = 0
        self.drained = asyncio.Event()

    async def flush(self):
        self.flushes += 1
        return True

    async def drain(self):
        await self.drained.wait()
        return True


@pytest.mark.parametrize("failed", [False, True])
async def test_boundary_survives_failed_or_cancelled_predecessor(failed):
    session = make_session()
    stream = FlushStream()
    stream.drained.set()
    session._turn_tts_stream = stream
    session._turn_tts_scheduled_text = "Checking"

    async def predecessor():
        if failed:
            raise RuntimeError("filler failed")
        raise asyncio.CancelledError

    prev = asyncio.create_task(predecessor())
    await asyncio.gather(prev, return_exceptions=True)
    session._tts_tail = prev
    session._schedule_tool_boundary()
    await asyncio.gather(session._tts_tail, return_exceptions=True)
    assert stream.flushes == 1


async def test_silent_second_tool_does_not_schedule_another_boundary():
    session = make_session()
    stream = FlushStream()
    stream.drained.set()
    session._turn_tts_stream = stream
    session._turn_tts_scheduled_text = "Checking"
    session._schedule_tool_boundary()
    await session._tts_tail
    session._schedule_tool_boundary()
    await session._tts_tail
    assert stream.flushes == 1


async def test_new_iteration_does_not_mark_unemitted_text_as_heard():
    class DelayedPostStream(FlushStream):
        async def feed(self, text):
            if not self.fed:
                await super().feed(text)
            else:
                self.fed.append(text)  # post-tool audio has not arrived yet

    class TTS(StreamingMockTTS):
        def open_stream(self):
            stream = DelayedPostStream()
            self.streams.append(stream)
            return stream

    tts = TTS()
    session = make_session(tts)
    session._schedule_tts("Checking…")
    await session._tts_tail
    while not session._turn_audio_bytes:
        await asyncio.sleep(0)
    before = session._turn_audio_bytes
    tts.streams[0].drained.set()
    session._schedule_tool_boundary()
    await session._tts_tail
    session._schedule_tts(" Order has shipped.")
    await session._tts_tail
    try:
        assert len(tts.streams) == 1
        assert map_played_bytes_to_text(session._turn_tts_segment_records, before) == "Checking…"
    finally:
        await session._abort_tts_stream()


async def test_beat_does_not_restart_at_late_poll_wakeup(monkeypatch):
    import timbal.voice.session as mod

    now = [0.0]
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay + (0.04 if len(sleeps) == 1 else 0)

    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    class Playback:
        @property
        def is_playing(self):
            return now[0] < 0.01

    session = make_session(tool_beat_secs=0.6)
    session.playback = Playback()
    session._turn_tts_scheduled_text = "Checking"
    session._turn_audio_bytes = 320
    session._turn_audio_playing_until = 0.01
    session._schedule_tool_boundary()
    await session._tts_tail
    assert now[0] == pytest.approx(0.61)


async def test_punctuated_text_flushes_before_tool_arguments_finish():
    from contextlib import aclosing

    from timbal.types.events.delta import DeltaEvent, TextDelta, ToolUse
    from timbal.types.events.output import OutputEvent

    flushed = asyncio.Event()

    class Stream(MockTTSStream):
        async def flush(self):
            flushed.set()
            return True

    class TTS(StreamingMockTTS):
        def open_stream(self):
            stream = Stream()
            self.streams.append(stream)
            return stream

    class StreamingAgent(Agent):
        def __call__(self, **kwargs):
            source = super().__call__(**kwargs)

            async def events():
                async with aclosing(source):
                    async for event in source:
                        if isinstance(event, OutputEvent) and event.path == "desk.llm":
                            for block in event.output.content:
                                item = (
                                    TextDelta(id="text", text_delta=block.text)
                                    if isinstance(block, TextContent)
                                    else ToolUse(id=block.id, name=block.name)
                                )
                                yield DeltaEvent(run_id=event.run_id, path=event.path, call_id=event.call_id, item=item)
                                if isinstance(item, ToolUse):
                                    # Model is still generating arguments: no OutputEvent yet.
                                    await asyncio.wait_for(flushed.wait(), 1)
                        yield event

            return events()

    async def lookup():
        return "shipped"

    agent = StreamingAgent(
        name="desk",
        tools=[lookup],
        model=TestModel(
            responses=[
                Message(
                    role="assistant",
                    content=[
                        TextContent(text="Let me check that for you."),
                        ToolUseContent(id="lookup", name="lookup", input={}),
                    ],
                    stop_reason="tool_use",
                ),
                "It shipped.",
            ]
        ),
    )
    session = VoiceSession(agent=agent, stt=MockSTT(), tts=TTS(), turn_detector="heuristic")
    await asyncio.wait_for(session._run_turn("check it"), 3)
    assert flushed.is_set()
    assert session._last_run_context.root_span().output.collect_text() == "It shipped."


async def test_cancelled_turn_cannot_enter_a_new_drain():
    session = make_session()
    stream = HungStream()
    session._turn_tts_stream = stream

    async def cancelled_turn():
        asyncio.current_task().cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass  # An upstream provider swallowed the deadline cancellation.
        await session._await_tts_chain()

    task = asyncio.create_task(cancelled_turn())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not stream.end_started.is_set()
    assert session._turn_tts_stream is stream
    await session._abort_tts_stream()
