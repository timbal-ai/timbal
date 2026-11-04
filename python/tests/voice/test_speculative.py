# ruff: noqa: ARG002
"""Speculative generation: confirm, cancel, transcript change, barge-in, tools.

The session starts the agent on Flux ``EagerEndOfTurn`` or a Smart Turn score,
holds audio until the committed transcript matches (LiveKit's word comparison),
and keeps ``defer_until_eot`` tools from running until then.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    SessionInterrupted,
    SpeculativeGeneration,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
    TurnMetricsEvent,
    VadEndpointer,
    VoiceConfig,
    VoiceSession,
)
from timbal.voice.predictive_eou import flux_eager_threshold
from timbal.voice.speculative import transcripts_equivalent

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedSTT(SpeechToText):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

    async def connect(self, config: AudioInputConfig) -> None:
        return None

    async def push_audio(self, chunk: bytes) -> None:
        return None

    async def commit(self) -> None:
        return None

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
            if item.type == "resumed" or item.text:
                yield item

    async def close(self) -> None:
        return None


class ScriptedTTS(TextToSpeech):
    def __init__(self, *, chunk: bytes = b"\x00\x01" * 16, num_chunks: int = 2) -> None:
        self.synthesized_texts: list[str] = []
        self._chunk = chunk
        self._num_chunks = num_chunks

    async def connect(self, config: AudioOutputConfig) -> None:
        return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized_texts.append(text)
        for _ in range(self._num_chunks):
            yield self._chunk

    async def close(self) -> None:
        return None


def _tool_call(name: str, call_id: str = "c1") -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id, name=name, input={})],
        stop_reason="tool_use",
    )


def _session(
    responses: list,
    *,
    speculative: SpeculativeGeneration | bool = True,
    tools: list | None = None,
    chunk: bytes = b"\x00\x01" * 16,
) -> tuple[VoiceSession, ScriptedSTT, ScriptedTTS, TestModel]:
    model = TestModel(responses=responses)
    agent = Agent(name="speculative_voice", model=model, tools=tools or [])
    stt = ScriptedSTT()
    tts = ScriptedTTS(chunk=chunk)
    session = VoiceSession(
        agent=agent,
        stt=stt,
        tts=tts,
        turn_detector="heuristic",
        speculative_generation=speculative,
    )
    return session, stt, tts, model


async def _until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


class _Drive:
    def __init__(self, session: VoiceSession, stt: ScriptedSTT) -> None:
        self.session = session
        self.stt = stt
        self.events: list = []
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> _Drive:
        async def _audio() -> AsyncIterator[bytes]:
            if False:
                yield b""

        async def _consume() -> None:
            async for event in self.session.run(_audio()):
                self.events.append(event)

        self._task = asyncio.create_task(_consume())
        await _until(lambda: any(getattr(e, "type", None) == "session_started" for e in self.events))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stt.finish()
        assert self._task is not None
        await self._task

    def audio(self) -> list:
        return [e for e in self.events if isinstance(e, AudioOutput)]

    def assistant_text(self) -> str:
        return " ".join(e.text for e in self.session.transcript if e.role == "assistant")


# ---------------------------------------------------------------------------
# Transcript comparison (LiveKit _transcripts_equivalent)
# ---------------------------------------------------------------------------


class TestTranscriptsEquivalent:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("book a table", "book a table"),
            ("Book a table.", "book a table"),
            ("HELLO!", "hello"),
            ("你好世界", "你好世界"),
        ],
    )
    def test_equivalent(self, left: str, right: str) -> None:
        assert transcripts_equivalent(left, right)
        assert transcripts_equivalent(right, left)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("book a table", "book a table for four"),
            ("hello", "hello there"),
            ("", "hello"),
            ("book a table", None),
            ("你好", "你好世界"),
        ],
    )
    def test_not_equivalent(self, left: str, right: str | None) -> None:
        assert not transcripts_equivalent(left, right)


class TestFluxEagerQuery:
    def test_explicit_value_wins_otherwise_stays_under_eot(self) -> None:
        assert flux_eager_threshold({}) == 0.5
        assert flux_eager_threshold({"eager_eot_threshold": 0.7}) == 0.7
        assert flux_eager_threshold({"eager_eot_threshold": "0.4"}) == 0.4
        # No 0.3 floor. eager stays strictly below eot, including an explicit value.
        assert flux_eager_threshold({"eot_threshold": 0.5}) == 0.4
        assert flux_eager_threshold({"eot_threshold": "0.35"}) == pytest.approx(0.25)
        assert flux_eager_threshold({"eot_threshold": 0.8, "eager_eot_threshold": 0.9}) == pytest.approx(0.7)
        assert flux_eager_threshold({"eager_eot_threshold": "nope"}) == 0.5

    def test_voice_config_coerces_bool_and_stays_off_the_client_hello(self) -> None:
        from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS

        assert VoiceConfig().speculative_generation.enabled is False
        assert VoiceConfig(speculative_generation=True).speculative_generation.enabled is True
        assert VoiceConfig(speculative_generation=False).speculative_generation.enabled is False
        tuned = VoiceConfig(speculative_generation={"eou_threshold": 0.6})
        assert tuned.speculative_generation.enabled is True
        assert tuned.speculative_generation.eou_threshold == 0.6
        held_off = VoiceConfig(speculative_generation={"enabled": False, "eou_threshold": 0.6})
        assert held_off.speculative_generation.enabled is False
        assert held_off.speculative_generation.eou_threshold == 0.6
        assert "speculative_generation" not in CLIENT_SETTABLE_VOICE_FIELDS
        # An empty block is omitted, so it does not clobber an inherited value.
        assert "speculative_generation" not in VoiceConfig.model_validate({}).model_fields_set


# ---------------------------------------------------------------------------
# Session paths
# ---------------------------------------------------------------------------


class TestSpeculativeSession:
    async def test_off_by_default_ignores_eager(self) -> None:
        session, stt, _tts, model = _session(["Hello."], speculative=False)
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await asyncio.sleep(0.05)
            assert model.call_count == 0
            assert drive.audio() == []
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await _until(lambda: drive.audio())
        assert session.speculative_generations_started == 0
        assert session.transcript[0].text == "book a table"

    async def test_confirm_reuses_attempt_and_holds_audio(self) -> None:
        session, stt, tts, model = _session(["The blue one."])
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await asyncio.sleep(0.05)
            assert drive.audio() == []
            assert tts.synthesized_texts == []
            assert not any(getattr(e, "type", None) == "agent_text_delta" for e in drive.events)
            await stt.inject(TranscriptEvent(type="committed", text="Book a table."))
            await _until(lambda: drive.audio())
            await _until(lambda: any(isinstance(e, TurnMetricsEvent) for e in drive.events))
        assert model.call_count == 1
        assert session.speculative_generations_started == 1
        assert session.speculative_generations_discarded == 0
        assert session.transcript[0].text == "Book a table."
        assert "blue" in drive.assistant_text()
        metrics = [e.metrics for e in drive.events if isinstance(e, TurnMetricsEvent)]
        assert metrics[-1].speculative is True
        assert metrics[-1].speculative_generations_started == 1
        assert metrics[-1].speculative_generations_discarded == 0
        assert metrics[-1].eou_to_first_audio_ms is not None
        assert metrics[-1].eou_to_first_audio_ms >= 0

    async def test_turn_resumed_discards_and_a_later_commit_starts_over(self) -> None:
        # TestModel picks a response from assistant turns already in the run, so a
        # fresh run after a discard would repeat the first string. Key off the prompt.
        def reply(messages):
            text = messages[-1].collect_text()
            return "The real reply." if "four" in text else "Too soon."

        session, stt, _tts, model = _session([], speculative=True)
        model.handler = reply
        model.responses = None
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await stt.inject(TranscriptEvent(type="resumed", text="book a table"))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await asyncio.sleep(0.05)
            assert drive.audio() == []
            assert model.call_count == 1
            await stt.inject(TranscriptEvent(type="committed", text="book a table for four"))
            await _until(lambda: model.call_count == 2)
            await _until(lambda: drive.audio())
        assert "real reply" in drive.assistant_text()
        assert "Too soon" not in drive.assistant_text()
        assert session.transcript[0].text == "book a table for four"
        assert session.speculative_generations_started == 1
        assert session.speculative_generations_discarded == 1

    async def test_changed_transcript_discards_and_restarts(self) -> None:
        def reply(messages):
            text = messages[-1].collect_text()
            return "For four." if "four" in text else "Too soon."

        session, stt, _tts, model = _session([], speculative=True)
        model.handler = reply
        model.responses = None
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await asyncio.sleep(0.05)
            assert drive.audio() == []
            await stt.inject(TranscriptEvent(type="committed", text="book a table for four"))
            await _until(lambda: model.call_count == 2)
            await _until(lambda: "For four" in drive.assistant_text() or drive.audio())
            await _until(lambda: any(isinstance(e, TurnMetricsEvent) and e.metrics.speculative for e in drive.events))
        assert model.call_count == 2
        assert session.speculative_generations_discarded == 1
        assert session.transcript[0].text == "book a table for four"
        assert "For four" in drive.assistant_text()
        assert "Too soon" not in drive.assistant_text()

    async def test_resumed_speech_partial_cancels_before_any_audio(self) -> None:
        session, stt, _tts, model = _session(["Too soon."])
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await stt.inject(TranscriptEvent(type="partial", text="book a table for four"))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await asyncio.sleep(0.05)
            assert drive.audio() == []
            assert not any(isinstance(e, SessionInterrupted) for e in drive.events)
            assert drive.assistant_text() == ""

    async def test_barge_in_after_confirm_interrupts_playback(self) -> None:
        # One second of PCM so the playback estimate is still "playing" when the
        # partial arrives. Heuristic barge-in needs audio in flight and 3 words.
        session, stt, _tts, model = _session(
            ["The blue one is ready now."],
            chunk=b"\x00" * 32_000,
        )
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: drive.audio())
            assert session._assistant_audio_playing
            await stt.inject(TranscriptEvent(type="partial", text="wait stop please now"))
            await _until(lambda: any(isinstance(e, SessionInterrupted) for e in drive.events))
        assert session.speculative_generations_discarded == 0

    async def test_read_tool_runs_before_confirm_and_side_effect_waits(self) -> None:
        lookup_ran = asyncio.Event()
        charge_ran = asyncio.Event()

        async def lookup() -> str:
            """Look up the table."""
            lookup_ran.set()
            return "blue"

        async def charge() -> str:
            """Charge the card."""
            charge_ran.set()
            return "charged"

        session, stt, _tts, model = _session(
            [_tool_call("lookup"), _tool_call("charge", "c2"), "All set."],
            tools=[
                Tool(handler=lookup),
                Tool(handler=charge, defer_until_eot=True),
            ],
        )
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await asyncio.wait_for(lookup_ran.wait(), timeout=2)
            await asyncio.sleep(0.05)
            assert not charge_ran.is_set()
            assert drive.audio() == []
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await asyncio.wait_for(charge_ran.wait(), timeout=2)
            await _until(lambda: "All set" in drive.assistant_text() or drive.audio())

    async def test_deferred_tool_is_dropped_when_the_turn_resumes(self) -> None:
        ran = asyncio.Event()

        async def charge() -> str:
            """Charge the card."""
            ran.set()
            return "charged"

        session, stt, _tts, _model = _session(
            [_tool_call("charge"), "Done."],
            tools=[Tool(handler=charge, defer_until_eot=True)],
        )
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="eager", text="charge it"))
            await _until(lambda: session.speculative_generations_started == 1)
            await asyncio.sleep(0.05)
            assert not ran.is_set()
            await stt.inject(TranscriptEvent(type="resumed", text=""))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await asyncio.sleep(0.05)
            assert not ran.is_set()

    async def test_smart_turn_score_starts_and_speech_resume_cancels(self) -> None:
        session, stt, _tts, model = _session(["Hello."])
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="partial", text="book a table"))
            await session._on_likely_eou(0.2, False)
            await asyncio.sleep(0.05)
            assert model.call_count == 0
            await session._on_likely_eou(0.8, True)
            await asyncio.sleep(0.05)
            assert model.call_count == 0
            await session._on_likely_eou(0.8, False)
            await _until(lambda: model.call_count == 1)
            assert session.speculative_generations_started == 1
            session._on_vad_speech_resume()
            await _until(lambda: session.speculative_generations_discarded == 1)

    async def test_max_retries_skips_further_attempts(self) -> None:
        session, stt, _tts, model = _session(
            ["One.", "Two."],
            speculative=SpeculativeGeneration(enabled=True, max_retries=1),
        )
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count == 1)
            await stt.inject(TranscriptEvent(type="resumed", text=""))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await stt.inject(TranscriptEvent(type="eager", text="book a table for four"))
            await asyncio.sleep(0.05)
            assert model.call_count == 1
            assert session.speculative_generations_started == 1
            await stt.inject(TranscriptEvent(type="committed", text="book a table for four"))
            await _until(lambda: model.call_count == 2)
            await _until(lambda: drive.audio())

    async def test_preemptive_tts_synthesizes_before_emitting(self) -> None:
        session, stt, tts, model = _session(
            ["The blue one."],
            speculative=SpeculativeGeneration(enabled=True, preemptive_tts=True),
        )
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: tts.synthesized_texts)
            await asyncio.sleep(0.05)
            assert model.call_count == 1
            assert drive.audio() == []
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: drive.audio())

    async def test_long_utterance_is_not_speculated(self) -> None:
        session, stt, _tts, model = _session(["Hello."])
        async with _Drive(session, stt) as drive:
            session._caller_speech_started_at = time.monotonic() - 30
            await stt.inject(TranscriptEvent(type="eager", text="a very long dictation"))
            await asyncio.sleep(0.05)
            assert model.call_count == 0
            await stt.inject(TranscriptEvent(type="committed", text="a very long dictation"))
            await _until(lambda: drive.audio())
        assert session.speculative_generations_started == 0


class _FakeVad:
    async def start(self, *, sample_rate: int) -> None:
        return None

    def reset(self) -> None:
        return None

    def process(self, chunk: bytes) -> list[float]:
        return [1.0 if byte == ord("S") else 0.0 for byte in chunk]


class TestEndpointScoreHook:
    async def test_audio_score_and_speech_resume_callbacks(self) -> None:
        scores: list[tuple[float, bool]] = []
        resumes: list[str] = []

        async def score() -> float:
            return 0.9

        async def commit() -> None:
            return None

        endpointer = VadEndpointer(
            _FakeVad(),
            stop_silence_secs=0.064,
            min_speech_secs=0.064,
            min_delay_secs=10.0,
            min_commit_interval_secs=0.0,
        )
        endpointer.bind(
            score=score,
            commit=commit,
            should_commit=lambda: True,
            on_score=lambda p, incomplete: _record(scores, (p, incomplete)),
            on_speech_resume=lambda: resumes.append("resume"),
        )
        await endpointer.start(sample_rate=16_000)
        endpointer.push(b"SS")
        endpointer.push(b"..")
        await _until(lambda: scores)
        assert scores[0][0] == pytest.approx(0.9)
        assert scores[0][1] is False
        endpointer.push(b"SS")
        await _until(lambda: resumes)
        await endpointer.close()


async def _record(target: list, value: tuple[float, bool]) -> None:
    target.append(value)
