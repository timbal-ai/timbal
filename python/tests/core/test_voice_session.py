"""Tests for VoiceSession lifecycle, transcript accumulation, audio recording, and error propagation."""
# ruff: noqa: ARG002

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice.metrics import TurnMetricsEvent
from timbal.voice.playback import PlaybackTracker
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
# Tests: per-turn latency metrics
# ---------------------------------------------------------------------------


class TestTurnMetrics:
    async def test_single_turn_emits_metrics_event(self) -> None:
        chunk = b"\x00\x01" * 16
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Hello")],
            responses=["Hi there!"],
            tts_chunk=chunk,
            tts_num_chunks=2,
        )
        events = await _collect_events(session)

        metrics_events = [e for e in events if isinstance(e, TurnMetricsEvent)]
        assert len(metrics_events) == 1
        m = metrics_events[0].metrics

        assert m.turn_index == 1
        assert m.user_text_chars == len("Hello")
        assert m.interrupted is False
        assert m.eou_to_llm_first_token_ms is not None and m.eou_to_llm_first_token_ms >= 0
        assert m.eou_to_first_audio_ms is not None and m.eou_to_first_audio_ms >= 0
        assert m.eou_to_tts_first_byte_ms == m.eou_to_first_audio_ms
        assert m.llm_total_ms is not None and m.llm_total_ms >= 0
        assert m.tts_total_ms is not None and m.tts_total_ms >= 0
        assert m.turn_total_ms >= m.llm_total_ms
        assert m.tts_segments >= 1
        assert m.audio_bytes == len(chunk) * 2

        assert session.metrics == [m]

    async def test_final_metrics_persisted_by_serializing_provider(self, tmp_path) -> None:
        """Regression: the agent run's trace is saved (provider put) when the
        generator exhausts — before the turn's finally builds final metrics.
        Serializing providers (jsonl, platform) captured that provisional
        snapshot with null llm_total_ms; the turn finally must re-save."""
        import json

        from timbal.state.tracing.providers import JsonlTracingProvider

        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        agent = Agent(
            name="t",
            model=TestModel(responses=["Hello there friend"]),
            tools=[],
            tracing_provider=provider,
        )
        stt = MockSTT(script=[TranscriptEvent(type="committed", text="What time is it?")])
        session = VoiceSession(agent=agent, stt=stt, tts=MockTTS())
        await _collect_events(session)

        assert len(session.metrics) == 1
        live = session.metrics[0]
        assert live.llm_total_ms is not None

        records = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
        assert len(records) == 1
        persisted = records[0]["spans"][0]["metadata"]["voice_turn_metrics"]
        assert persisted == live.model_dump()

    async def test_metrics_emitted_after_agent_text_done(self) -> None:
        session, _, _ = _make_session(
            stt_script=[TranscriptEvent(type="committed", text="Hello")],
            responses=["Hi!"],
        )
        events = await _collect_events(session)
        types = [e.type for e in events]
        assert types.index("metrics") > types.index("agent_text_done")

    async def test_empty_session_has_no_metrics(self) -> None:
        session, _, _ = _make_session(stt_script=[])
        events = await _collect_events(session)
        assert not any(isinstance(e, TurnMetricsEvent) for e in events)
        assert session.metrics == []

    async def test_interrupted_turn_flagged(self) -> None:
        """Barge-in during playback → the first turn's metrics carry interrupted=True."""
        agent = Agent(name="t", model=TestModel(responses=["First reply", "Second reply"]), tools=[])
        stt = DelayedMockSTT()
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
            await stt.inject(TranscriptEvent(type="committed", text="Actually I want to ask about something else entirely"))
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

        assert len(session.metrics) == 2
        assert session.metrics[0].interrupted is True
        assert session.metrics[1].interrupted is False


# ---------------------------------------------------------------------------
# Tests: interruption truncation (what the user actually heard)
# ---------------------------------------------------------------------------


class FakePlaybackTracker(PlaybackTracker):
    """Playback tracker with externally controlled position for deterministic tests."""

    def __init__(self) -> None:
        self.emitted_bytes = 0
        self.played = 0
        self.playing = False
        self.interrupt_calls = 0

    def on_audio_emitted(self, num_bytes: int) -> None:
        self.emitted_bytes += num_bytes

    def on_interrupted(self) -> None:
        # Deliberately does not reset ``playing`` — tests control it explicitly.
        self.interrupt_calls += 1

    @property
    def played_bytes(self) -> int:
        return self.played

    @property
    def is_playing(self) -> bool:
        return self.playing


class TestInterruptionTruncation:
    """Invariants under test (voice barge-in / HOLD / CONTINUE):

    * transcript assistant text == heard prefix (or absent if 0 bytes)
    * root.memory last assistant matches that prefix (or both dropped)
    * serializing providers re-save after post-AgentTextDone truncation
    * CONTINUE_TURN: one merged user entry; no stray heard-assistant in
      transcript *or* parent memory
    * interrupt() cancels orphan ``_current_turn_task`` before ``_is_speaking``
    * HOLD expiry must not wipe a refined/re-armed hold
    """

    RESPONSE = "Hello world how are you doing today my friend"
    BARGE_IN = "Actually I want to ask about something else entirely"

    async def _run_barge_in_session(
        self,
        tracker: FakePlaybackTracker,
        *,
        played_bytes: int,
        interrupt_after_done: bool,
    ) -> tuple[VoiceSession, list[VoiceSessionEvent]]:
        """One turn, then a barge-in commit with ``tracker`` reporting ``played_bytes`` heard."""
        # Handler-based model: TestModel(responses=...) picks by assistant-message
        # count in history, which the truncation itself changes (dropping the
        # unheard reply would replay response[0]). The handler advances per call.
        replies = iter([self.RESPONSE, "Second reply", "Third reply"])
        agent = Agent(name="t", model=TestModel(handler=lambda _messages: next(replies)), tools=[])
        stt = DelayedMockSTT()
        chunk = b"\x00\x01" * 50  # 100 bytes per chunk
        tts = SlowMockTTS(delay=0.03, chunk=chunk, num_chunks=4) if not interrupt_after_done else MockTTS(chunk=chunk, num_chunks=4)
        session = VoiceSession(agent=agent, stt=stt, tts=tts, playback_tracker=tracker)

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Hello there"))
            if interrupt_after_done:
                while not any(isinstance(e, AgentTextDone) for e in events):
                    await asyncio.sleep(0.01)
            else:
                while not any(isinstance(e, AudioOutput) for e in events):
                    await asyncio.sleep(0.01)
            tracker.playing = True
            tracker.played = played_bytes
            await stt.inject(TranscriptEvent(type="committed", text=self.BARGE_IN))
            while sum(1 for e in events if isinstance(e, AgentTextDone)) < (2 if interrupt_after_done else 1):
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
        return session, events

    async def test_mid_turn_barge_in_records_heard_prefix(self) -> None:
        tracker = FakePlaybackTracker()
        # 100 bytes per chunk; heard 50 bytes — always a strict, non-empty
        # prefix regardless of how many chunks were emitted before the cancel.
        session, events = await self._run_barge_in_session(
            tracker, played_bytes=50, interrupt_after_done=False
        )

        assistant_entries = [e for e in session.transcript if e.role == "assistant"]
        # First assistant entry is the truncated heard prefix, not the full reply.
        heard = assistant_entries[0].text
        assert heard
        assert heard != self.RESPONSE
        assert self.RESPONSE.startswith(heard)

        interrupted_events = [e for e in events if isinstance(e, SessionInterrupted)]
        assert interrupted_events[0].heard_text == heard

    async def test_mid_turn_barge_in_truncates_memory(self) -> None:
        tracker = FakePlaybackTracker()
        session, _ = await self._run_barge_in_session(
            tracker, played_bytes=50, interrupt_after_done=False
        )
        # The next turn resolved memory from the truncated context: the second
        # turn's run context memory must contain the heard prefix, not the full reply.
        ctx = session._last_run_context
        root = ctx.root_span()
        assistant_msgs = [m for m in root.memory if m.role == "assistant"]
        first_reply_text = assistant_msgs[0].collect_text()
        assert first_reply_text != self.RESPONSE
        assert self.RESPONSE.startswith(first_reply_text)

    async def test_nothing_heard_drops_assistant_from_memory_and_transcript(self) -> None:
        tracker = FakePlaybackTracker()
        session, events = await self._run_barge_in_session(
            tracker, played_bytes=0, interrupt_after_done=False
        )
        # Nothing was heard: no truncated transcript entry for the first reply.
        assistant_entries = [e for e in session.transcript if e.role == "assistant"]
        assert all(e.text != self.RESPONSE and not self.RESPONSE.startswith(e.text) for e in assistant_entries)

        ctx = session._last_run_context
        root = ctx.root_span()
        assistant_msgs = [m for m in root.memory if m.role == "assistant"]
        # Only the second turn's reply may be in memory.
        assert all(not self.RESPONSE.startswith(m.collect_text() or "x") for m in assistant_msgs)

        interrupted_events = [e for e in events if isinstance(e, SessionInterrupted)]
        assert interrupted_events[0].heard_text is None or interrupted_events[0].heard_text == ""

    async def test_completed_turn_barge_in_rewrites_committed_entry(self) -> None:
        tracker = FakePlaybackTracker()
        session, _ = await self._run_barge_in_session(
            tracker, played_bytes=200, interrupt_after_done=True
        )
        assistant_entries = [e for e in session.transcript if e.role == "assistant"]
        heard = assistant_entries[0].text
        assert heard != self.RESPONSE
        assert self.RESPONSE.startswith(heard)
        # Second turn's reply is intact.
        assert assistant_entries[1].text == "Second reply"

    async def test_completed_turn_barge_in_resaves_serialized_trace(self) -> None:
        """Regression: interrupt() after AgentTextDone must re-persist the
        truncated memory. Serializing providers already saved the full reply in
        the turn's finally; without a re-save the stored snapshot stays wrong
        even though session_transcript / live memory are truncated."""
        from copy import deepcopy

        from timbal.state.tracing.providers import InMemoryTracingProvider

        class SnapshotProvider(InMemoryTracingProvider):
            """Capture a deep copy on each store — mimics jsonl/platform."""

            _snapshots: dict = {}

            @classmethod
            async def _store(cls, run_context) -> None:
                await super()._store(run_context)
                cls._snapshots[run_context.id] = deepcopy(run_context._trace)

        SnapshotProvider._snapshots = {}
        SnapshotProvider._storage = {}

        tracker = FakePlaybackTracker()
        replies = iter([self.RESPONSE, "Second reply"])
        agent = Agent(
            name="t",
            model=TestModel(handler=lambda _messages: next(replies)),
            tools=[],
            tracing_provider=SnapshotProvider,
        )
        stt = DelayedMockSTT()
        tts = MockTTS(chunk=b"\x00\x01" * 50, num_chunks=4)
        session = VoiceSession(agent=agent, stt=stt, tts=tts, playback_tracker=tracker)

        events: list[VoiceSessionEvent] = []
        first_run_id: str | None = None

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            nonlocal first_run_id
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Hello there"))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)
            first_run_id = session._last_run_context.id if session._last_run_context else None
            tracker.playing = True
            tracker.played = 200  # half of 400 emitted bytes
            await stt.inject(TranscriptEvent(type="committed", text=self.BARGE_IN))
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

        assert first_run_id is not None
        snap = SnapshotProvider._snapshots.get(first_run_id)
        assert snap is not None
        # Find the root span memory in the serialized snapshot.
        root = next(s for s in snap.values() if getattr(s, "path", None) and "." not in s.path)
        memory = root.memory or []
        assistant_texts = [
            m.collect_text() for m in memory if getattr(m, "role", None) == "assistant"
        ]
        assert assistant_texts, "expected truncated assistant message in stored trace"
        heard = assistant_texts[0]
        assert heard != self.RESPONSE
        assert self.RESPONSE.startswith(heard)

    async def test_barge_in_during_final_tts_wait_rewrites_not_appends(self) -> None:
        """Regression: ``_await_tts_chain`` swallows the cancellation of TTS
        tasks, so a barge-in landing while the turn sits in the *final*
        pre-commit wait lets the turn resume and commit the full reply with
        ``_cancel_turn`` already set. The finally's truncation must then
        rewrite the committed entry in place — append mode would leave the
        full entry plus a duplicate heard-prefix entry. TestModel never
        streams deltas (no pending tail segment), so the window is simulated
        with a held fake TTS task registered in ``_tts_tasks``."""

        class _RaceSession(VoiceSession):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._chain_calls = 0
                self.holding = asyncio.Event()

            async def _await_tts_chain(self) -> None:
                await super()._await_tts_chain()
                self._chain_calls += 1
                # Call 1 drains TTS after the llm OUTPUT event; call 2 is the
                # final wait right before the transcript commit — hold a fake
                # pending segment there until interrupt() cancels it.
                if self._turn_index == 1 and self._chain_calls == 2:
                    waiter = asyncio.ensure_future(asyncio.Event().wait())
                    self._tts_tasks.add(waiter)
                    self.holding.set()
                    try:
                        await waiter
                    except (asyncio.CancelledError, Exception):
                        pass

        tracker = FakePlaybackTracker()
        replies = iter([self.RESPONSE, "Second reply"])
        agent = Agent(name="t", model=TestModel(handler=lambda _messages: next(replies)), tools=[])
        stt = DelayedMockSTT()
        tts = MockTTS(chunk=b"\x00\x01" * 50, num_chunks=4)
        session = _RaceSession(agent=agent, stt=stt, tts=tts, playback_tracker=tracker)

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Hello there"))
            await asyncio.wait_for(session.holding.wait(), timeout=5)
            tracker.playing = True
            tracker.played = 200  # half of the 400 emitted bytes
            await stt.inject(TranscriptEvent(type="committed", text=self.BARGE_IN))
            while not any(isinstance(e, AgentTextDone) and e.text == "Second reply" for e in events):
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

        assistant_entries = [e for e in session.transcript if e.role == "assistant"]
        # Exactly one entry for the first reply (rewritten in place) — append
        # mode would produce two (full + heard prefix).
        assert len(assistant_entries) == 2
        heard = assistant_entries[0].text
        assert heard != self.RESPONSE
        assert self.RESPONSE.startswith(heard)
        assert assistant_entries[1].text == "Second reply"

    async def test_barge_in_during_completed_turn_finally_still_truncates(self) -> None:
        """Regression (Windows CI): the barge-in can land while the *finished*
        turn's finally is still persisting the trace. The task is not done()
        yet, so interrupt() cancels it (swallowed by the shielded aclose/save)
        and the old ``elif _turn_finalized_ok`` branch never ran — no
        truncation at all. A slow tracing provider makes the window
        deterministic."""
        from timbal.state.tracing.providers import InMemoryTracingProvider

        class SlowSaveProvider(InMemoryTracingProvider):
            _storage = {}

            @classmethod
            async def _store(cls, run_context) -> None:
                await asyncio.sleep(0.15)
                cls._storage[run_context.id] = run_context._trace

        tracker = FakePlaybackTracker()
        agent = Agent(
            name="t",
            model=TestModel(responses=[self.RESPONSE, "Second reply"]),
            tools=[],
            tracing_provider=SlowSaveProvider,
        )
        stt = DelayedMockSTT()
        tts = MockTTS(chunk=b"\x00\x01" * 50, num_chunks=4)
        session = VoiceSession(agent=agent, stt=stt, tts=tts, playback_tracker=tracker)

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Hello there"))
            # AgentTextDone is emitted *before* the turn's finally re-saves the
            # trace (0.15s sleep) — inject the barge-in inside that window so
            # interrupt() sees a finalized turn whose task is not done() yet.
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.005)
            tracker.playing = True
            tracker.played = 200  # half of the 400 emitted bytes
            await stt.inject(TranscriptEvent(type="committed", text=self.BARGE_IN))
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

        assistant_entries = [e for e in session.transcript if e.role == "assistant"]
        heard = assistant_entries[0].text
        assert heard != self.RESPONSE
        assert self.RESPONSE.startswith(heard)
        assert assistant_entries[1].text == "Second reply"

    async def test_continuation_with_heard_audio_merges_user_entry(self) -> None:
        """Regression: on CONTINUE_TURN, interrupt() runs before the transcript
        merge, and the aborted fragment turn's truncation can append a heard
        assistant entry. The old merge only replaced the last entry when it was
        ``user``, so it appended a *second* user line and left the spurious
        assistant fragment between them."""
        from timbal.voice.turn_detection import (
            CommitAction,
            CommitDecision,
            PartialDecision,
            TurnDetector,
        )

        class _ScriptedDetector(TurnDetector):
            def __init__(self, decisions: list[CommitDecision]) -> None:
                self._decisions = list(decisions)

            async def on_partial(self, text, state):  # noqa: ARG002
                return PartialDecision.IGNORE

            async def on_committed(self, text, state):  # noqa: ARG002
                return self._decisions.pop(0)

        fragment = "hello can"
        combined = "hello can you help me with something"
        detector = _ScriptedDetector(
            [
                CommitDecision(action=CommitAction.NEW_TURN, text=fragment, reason="test"),
                CommitDecision(action=CommitAction.CONTINUE_TURN, text=combined, reason="test"),
            ]
        )
        tracker = FakePlaybackTracker()
        # Handler (not responses=): CONTINUE memory cleanup drops the fragment
        # assistant message, so assistant-count-based TestModel would replay
        # response[0] for the merged turn.
        replies = iter([self.RESPONSE, "Second reply"])
        agent = Agent(name="t", model=TestModel(handler=lambda _m: next(replies)), tools=[])
        stt = DelayedMockSTT()
        tts = SlowMockTTS(delay=0.03, chunk=b"\x00\x01" * 50, num_chunks=4)
        session = VoiceSession(
            agent=agent, stt=stt, tts=tts, turn_detector=detector, playback_tracker=tracker
        )

        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text=fragment))
            while not any(isinstance(e, AudioOutput) for e in events):
                await asyncio.sleep(0.01)
            # Part of the fragment reply was heard when the user keeps talking.
            tracker.playing = True
            tracker.played = 50
            await stt.inject(TranscriptEvent(type="committed", text=combined))
            while not any(isinstance(e, AgentTextDone) for e in events):
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

        # One merged user entry, one assistant reply — no duplicate user line,
        # no stray assistant fragment in between.
        assert [e.role for e in session.transcript] == ["user", "assistant"]
        assert session.transcript[0].text == combined
        assert session.transcript[1].text == "Second reply"

    async def test_align_continue_memory_drops_heard_assistant(self) -> None:
        """Unit: CONTINUE rewrite must pop assistant:heard and rewrite user."""
        from timbal.state.context import RunContext
        from timbal.state.tracing.span import Span
        from timbal.types.content import TextContent
        from timbal.types.message import Message

        agent = Agent(name="t", model=TestModel(responses=["x"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(), tts=MockTTS())
        ctx = RunContext(tracing_provider=None, platform_config=None)
        root = Span(
            path="t",
            call_id="c1",
            parent_call_id=None,
            t0=0,
            memory=[
                Message(role="user", content=[TextContent(text="hello can")]),
                Message(role="assistant", content=[TextContent(text="Hello wor")]),
            ],
        )
        ctx._trace[root.call_id] = root
        session._last_run_context = ctx

        await session._align_continue_memory(
            fragment_user_text="hello can",
            combined_user_text="hello can you help me with something",
        )
        mem = root.memory
        assert len(mem) == 1
        assert mem[0].role == "user"
        assert mem[0].collect_text() == "hello can you help me with something"

    async def test_interrupt_cancels_orphan_turn_before_speaking(self) -> None:
        """``interrupt()`` must cancel a scheduled turn even if ``_is_speaking``
        is still False — otherwise HOLD expiry / commit races double-run agents."""
        gate = asyncio.Event()

        class _GatedSession(VoiceSession):
            async def _run_turn(self, user_text: str) -> None:
                await gate.wait()
                await super()._run_turn(user_text)

        agent = Agent(name="t", model=TestModel(responses=["should not speak", "ok"]), tools=[])
        stt = DelayedMockSTT()
        session = _GatedSession(agent=agent, stt=stt, tts=MockTTS())
        events: list[VoiceSessionEvent] = []

        async def _empty_audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Hello there"))
            # Wait until the turn task exists but has not set _is_speaking yet.
            for _ in range(100):
                task = session._current_turn_task
                if task is not None and not task.done() and not session._is_speaking:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("orphan turn never scheduled")
            assert session._assistant_active is False
            await session.interrupt()
            assert session._current_turn_task is None or session._current_turn_task.done()
            gate.set()  # would let the orphan proceed if cancel failed
            await asyncio.sleep(0.05)
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_empty_audio())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=5)
        assert not any(isinstance(e, AgentTextDone) for e in events)

    async def test_close_does_not_truncate_completed_turn(self) -> None:
        tracker = FakePlaybackTracker()
        agent = Agent(name="t", model=TestModel(responses=[self.RESPONSE]), tools=[])
        stt = MockSTT(script=[TranscriptEvent(type="committed", text="Hello there, how are you doing?")])
        tts = MockTTS(chunk=b"\x00\x01" * 50, num_chunks=4)
        session = VoiceSession(agent=agent, stt=stt, tts=tts, playback_tracker=tracker)
        # Simulate audio still playing (and only partially heard) at close time:
        # close() must NOT rewrite the committed transcript entry.
        tracker.playing = True
        tracker.played = 200
        await _collect_events(session)

        assistant_entries = [e for e in session.transcript if e.role == "assistant"]
        assert assistant_entries[0].text == self.RESPONSE


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
