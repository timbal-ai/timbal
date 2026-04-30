"""Tests for VoiceSession lifecycle, transcript accumulation, audio recording, and error propagation."""
# ruff: noqa: ARG002

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice.session import (
    AgentTextDelta,
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    SpeechToText,
    TextToSpeech,
    TranscriptCommitted,
    TranscriptEvent,
    TranscriptPartial,
    VoiceSession,
    VoiceSessionEvent,
    _strip_markdown,
)

# ---------------------------------------------------------------------------
# Mock STT / TTS
# ---------------------------------------------------------------------------


class MockSTT(SpeechToText):
    """STT that replays a scripted sequence of TranscriptEvents.

    After events() exhausts, VoiceSession._process_stt_events calls session.close()
    automatically, so no special teardown logic is needed here.
    """

    def __init__(self, script: list[TranscriptEvent] | None = None) -> None:
        self._script = list(script or [])
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._connected = False
        self._closed = False

    async def connect(self, config: AudioInputConfig) -> None:
        self._connected = True
        for ev in self._script:
            await self._queue.put(ev)
        await self._queue.put(None)

    async def push_audio(self, chunk: bytes) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.type == "error":
                raise RuntimeError(item.text)
            if item.text:
                yield item

    async def close(self) -> None:
        self._closed = True


class SlowMockTTS(TextToSpeech):
    """TTS that introduces a delay between chunks to simulate real synthesis latency."""

    def __init__(self, delay: float = 0.2, chunk: bytes = b"\x00\x01" * 16, num_chunks: int = 2) -> None:
        self._delay = delay
        self._chunk = chunk
        self._num_chunks = num_chunks
        self._connected = False
        self._closed = False

    async def connect(self, config: AudioOutputConfig) -> None:
        self._connected = True

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for _ in range(self._num_chunks):
            await asyncio.sleep(self._delay)
            yield self._chunk

    async def close(self) -> None:
        self._closed = True


class DelayedMockSTT(SpeechToText):
    """STT whose events are pushed externally, allowing precise timing control."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._connected = False
        self._closed = False

    async def connect(self, config: AudioInputConfig) -> None:
        self._connected = True

    async def push_audio(self, chunk: bytes) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def inject(self, event: TranscriptEvent) -> None:
        await self._queue.put(event)

    async def finish(self) -> None:
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.type == "error":
                raise RuntimeError(item.text)
            if item.text:
                yield item

    async def close(self) -> None:
        self._closed = True


class MockTTS(TextToSpeech):
    """TTS that returns fixed PCM chunks per synthesize call."""

    def __init__(self, chunk: bytes = b"\x00\x01" * 16, num_chunks: int = 2) -> None:
        self._chunk = chunk
        self._num_chunks = num_chunks
        self._connected = False
        self._closed = False
        self.synthesized_texts: list[str] = []

    async def connect(self, config: AudioOutputConfig) -> None:
        self._connected = True

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized_texts.append(text)
        for _ in range(self._num_chunks):
            yield self._chunk

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PCM = b"\x00\x01" * 16


async def _collect_events(session: VoiceSession) -> list[VoiceSessionEvent]:
    """Run the session with an empty audio stream and collect all events."""

    async def _empty_audio() -> AsyncIterator[bytes]:
        return
        yield  # noqa: RET504 — make it an async generator

    events: list[VoiceSessionEvent] = []
    async with aclosing(session.run(_empty_audio())) as stream:
        async for ev in stream:
            events.append(ev)
    return events


def _make_session(
    stt_script: list[TranscriptEvent] | None = None,
    responses: list[str] | None = None,
    tts_chunk: bytes = _FAKE_PCM,
    tts_num_chunks: int = 2,
    record_audio: bool = False,
) -> tuple[VoiceSession, MockSTT, MockTTS]:
    agent = Agent(
        name="test_voice",
        model=TestModel(responses=responses or ["Hello back!"]),
        tools=[],
    )
    stt = MockSTT(script=stt_script)
    tts = MockTTS(chunk=tts_chunk, num_chunks=tts_num_chunks)
    session = VoiceSession(
        agent=agent,
        stt=stt,
        tts=tts,
        record_audio=record_audio,
    )
    return session, stt, tts


# ---------------------------------------------------------------------------
# Tests: basic lifecycle
# ---------------------------------------------------------------------------


class TestVoiceSessionLifecycle:
    async def test_empty_session_emits_started_and_ended(self) -> None:
        session, stt, tts = _make_session(stt_script=[])
        events = await _collect_events(session)

        types = [e.type for e in events]
        assert types[0] == "session_started"
        assert types[-1] == "session_ended"
        assert stt._connected
        assert tts._connected

    async def test_close_is_idempotent(self) -> None:
        session, _, _ = _make_session(stt_script=[])
        await _collect_events(session)
        await session.close()
        await session.close()


# ---------------------------------------------------------------------------
# Tests: transcript accumulation
# ---------------------------------------------------------------------------


class TestTranscript:
    async def test_single_turn_transcript(self) -> None:
        session, _, _ = _make_session(
            stt_script=[
                TranscriptEvent(type="committed", text="Hi there"),
            ],
            responses=["Hello!"],
        )
        await _collect_events(session)

        assert session.transcript[0].role == "user"
        assert session.transcript[0].text == "Hi there"
        assert session.transcript[1].role == "assistant"
        assert "Hello" in session.transcript[1].text

    async def test_multi_turn_transcript(self) -> None:
        session, stt, _ = _make_session(
            stt_script=[
                TranscriptEvent(type="committed", text="First question"),
            ],
            responses=["Answer one", "Answer two"],
        )
        await _collect_events(session)

        assert len(session.transcript) >= 2
        assert session.transcript[0].role == "user"
        assert session.transcript[0].text == "First question"

    async def test_partial_events_not_in_transcript(self) -> None:
        session, _, _ = _make_session(
            stt_script=[
                TranscriptEvent(type="partial", text="Hi th"),
                TranscriptEvent(type="partial", text="Hi there"),
                TranscriptEvent(type="committed", text="Hi there"),
            ],
            responses=["Hello!"],
        )
        events = await _collect_events(session)

        event_types = [e.type for e in events]
        assert "transcript_partial" in event_types

        for entry in session.transcript:
            assert entry.role in ("user", "assistant")

    async def test_transcript_entries_have_timestamps(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="test")],
            responses=["ok"],
        )
        await _collect_events(session)
        for entry in session.transcript:
            assert entry.timestamp > 0


# ---------------------------------------------------------------------------
# Tests: audio recording
# ---------------------------------------------------------------------------


class TestAudioRecording:
    async def test_output_audio_recorded_when_enabled(self) -> None:
        chunk = b"\x01\x02" * 8
        session, _, tts = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Say something")],
            responses=["Hi"],
            tts_chunk=chunk,
            tts_num_chunks=3,
            record_audio=True,
        )
        await _collect_events(session)

        assert len(session.output_audio) == len(chunk) * 3
        assert session.output_audio == chunk * 3

    async def test_output_audio_empty_when_disabled(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Say something")],
            responses=["Hi"],
            record_audio=False,
        )
        await _collect_events(session)
        assert session.output_audio == b""

    async def test_input_audio_recorded(self) -> None:
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = MockSTT(script=[])
        tts = MockTTS()
        session = VoiceSession(agent=agent, stt=stt, tts=tts, record_audio=True)

        audio_chunks = [b"\x00\x01" * 10, b"\x02\x03" * 10]

        async def _audio_source() -> AsyncIterator[bytes]:
            for c in audio_chunks:
                yield c

        async with aclosing(session.run(_audio_source())) as stream:
            async for _ in stream:
                pass

        assert session.input_audio == b"".join(audio_chunks)

    async def test_input_audio_empty_when_disabled(self) -> None:
        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        stt = MockSTT(script=[])
        tts = MockTTS()
        session = VoiceSession(agent=agent, stt=stt, tts=tts, record_audio=False)

        async def _audio_source() -> AsyncIterator[bytes]:
            yield b"\x00\x01" * 10

        async with aclosing(session.run(_audio_source())) as stream:
            async for _ in stream:
                pass

        assert session.input_audio == b""


# ---------------------------------------------------------------------------
# Tests: event types emitted
# ---------------------------------------------------------------------------


class TestEventTypes:
    async def test_committed_text_emits_transcript_committed(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Hello")],
            responses=["Hi!"],
        )
        events = await _collect_events(session)
        committed = [e for e in events if isinstance(e, TranscriptCommitted)]
        assert len(committed) == 1
        assert committed[0].text == "Hello"

    async def test_agent_response_emits_text_deltas_and_done(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Tell me something")],
            responses=["Sure thing!"],
        )
        events = await _collect_events(session)
        deltas = [e for e in events if isinstance(e, AgentTextDelta)]
        dones = [e for e in events if isinstance(e, AgentTextDone)]
        assert len(deltas) >= 1
        assert len(dones) == 1

    async def test_tts_emits_audio_output(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Speak")],
            responses=["Words"],
            tts_num_chunks=2,
        )
        events = await _collect_events(session)
        audio_events = [e for e in events if isinstance(e, AudioOutput)]
        assert len(audio_events) == 2

    async def test_partial_events_emitted(self) -> None:
        session, _, _ = _make_session(
            stt_script=[
                TranscriptEvent(type="partial", text="Hel"),
                TranscriptEvent(type="partial", text="Hello"),
                TranscriptEvent(type="committed", text="Hello"),
            ],
            responses=["Hi"],
        )
        events = await _collect_events(session)
        partials = [e for e in events if isinstance(e, TranscriptPartial)]
        assert len(partials) == 2


# ---------------------------------------------------------------------------
# Tests: error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    async def test_stt_error_event_becomes_session_error(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="error", text="STT auth failed")],
        )
        events = await _collect_events(session)
        errors = [e for e in events if isinstance(e, SessionError)]
        assert len(errors) >= 1
        assert "STT" in errors[0].message


# ---------------------------------------------------------------------------
# Tests: _strip_markdown
# ---------------------------------------------------------------------------


class TestStripMarkdown:
    def test_bold(self) -> None:
        assert _strip_markdown("This is **bold** text") == "This is bold text"

    def test_headers(self) -> None:
        assert _strip_markdown("## Title\nBody") == "Title\nBody"
        assert _strip_markdown("# H1\n## H2\n### H3") == "H1\nH2\nH3"

    def test_numbered_list(self) -> None:
        assert _strip_markdown("1. First\n2. Second") == "First\nSecond"

    def test_bullet_list(self) -> None:
        assert _strip_markdown("- Item A\n- Item B") == "Item A\nItem B"
        assert _strip_markdown("* Star A\n* Star B") == "Star A\nStar B"

    def test_combined(self) -> None:
        md = "## **Bold Title**\n1. First item\n- Bullet"
        result = _strip_markdown(md)
        assert "**" not in result
        assert "##" not in result
        assert "1." not in result

    def test_plain_text_unchanged(self) -> None:
        plain = "Hello, how are you doing today?"
        assert _strip_markdown(plain) == plain


# ---------------------------------------------------------------------------
# Tests: multi-turn conversation (DelayedMockSTT)
# ---------------------------------------------------------------------------


class TestMultiTurn:
    async def test_two_sequential_turns_produce_four_transcript_entries(self) -> None:
        """Two committed events with enough spacing for each to complete independently."""
        agent = Agent(name="t", model=TestModel(responses=["Answer 1", "Answer 2"]), tools=[])
        stt = DelayedMockSTT()
        tts = MockTTS()
        session = VoiceSession(agent=agent, stt=stt, tts=tts)

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)

            await stt.inject(TranscriptEvent(type="committed", text="First question"))
            while sum(1 for e in events if isinstance(e, AgentTextDone)) < 1:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)

            await stt.inject(TranscriptEvent(type="committed", text="Second question"))
            while sum(1 for e in events if isinstance(e, AgentTextDone)) < 2:
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

        assert len(session.transcript) == 4
        assert session.transcript[0].role == "user"
        assert session.transcript[0].text == "First question"
        assert session.transcript[1].role == "assistant"
        assert session.transcript[2].role == "user"
        assert session.transcript[2].text == "Second question"
        assert session.transcript[3].role == "assistant"

    async def test_interrupt_emitted_when_second_commit_during_turn(self) -> None:
        """A new committed while TTS audio is still 'playing' triggers SessionInterrupted."""
        agent = Agent(name="t", model=TestModel(responses=["First reply", "Second reply"]), tools=[])
        stt = DelayedMockSTT()
        # Large chunks (32 000 bytes = 1s of PCM16@16kHz each) keep _playback_end_estimate
        # far in the future so _assistant_active stays True after the agent turn finishes.
        big_chunk = b"\x00\x01" * 16_000
        tts = SlowMockTTS(delay=0.05, chunk=big_chunk, num_chunks=3)
        session = VoiceSession(agent=agent, stt=stt, tts=tts)

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)

            await stt.inject(TranscriptEvent(type="committed", text="Hello there"))
            while not any(isinstance(e, AudioOutput) for e in events):
                await asyncio.sleep(0.01)

            # Text must be >30 chars to bypass the continuation-combine heuristic.
            await stt.inject(TranscriptEvent(type="committed", text="Actually I want to ask about something else entirely"))
            # First turn was interrupted before AgentTextDone, so only the second emits one.
            while sum(1 for e in events if isinstance(e, AgentTextDone)) < 1:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty_audio())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)

        types = [type(e) for e in events]
        assert SessionInterrupted in types


# ---------------------------------------------------------------------------
# Tests: STT/TTS providers are closed
# ---------------------------------------------------------------------------


class TestProviderCleanup:
    async def test_stt_and_tts_closed_after_session(self) -> None:
        session, stt, tts = _make_session(stt_script=[])
        await _collect_events(session)
        await session.close()
        assert stt._closed
        assert tts._closed
