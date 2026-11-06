# ruff: noqa: ARG002
"""Speculative generation: confirm, cancel, transcript change, barge-in, tools.

The session starts the agent on Flux ``EagerEndOfTurn`` or a Smart Turn score,
holds audio until the committed transcript matches (LiveKit's word comparison),
and does not start any tool call until that confirmation.
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


class TestToolHoldGate:
    """The runnable gate, without a voice session. Speculation off is no gate."""

    async def test_confirm_runs_handler_and_side_effects(self) -> None:
        effects: list[str] = []
        tool = _side_effect_tool(effects)
        gate, token = _install_gate()
        try:
            task = asyncio.create_task(tool().collect())
            await asyncio.sleep(0.05)
            assert effects == []
            gate.confirm()
            result = await asyncio.wait_for(task, timeout=2)
        finally:
            _reset_gate(token)
        assert result.status.code == "success"
        assert effects == ["default", "pre", "handler", "card"]

    async def test_discard_runs_nothing(self) -> None:
        effects: list[str] = []
        tool = _side_effect_tool(effects)
        gate, token = _install_gate()
        try:
            task = asyncio.create_task(tool().collect())
            await asyncio.sleep(0.05)
            assert effects == []
            gate.discard()
            result = await asyncio.wait_for(task, timeout=2)
        finally:
            _reset_gate(token)
        assert effects == []
        assert result.status.code == "cancelled"

    async def test_no_gate_runs_immediately(self) -> None:
        effects: list[str] = []
        result = await _side_effect_tool(effects, defer=True)().collect()
        assert result.status.code == "success"
        assert effects == ["default", "pre", "handler", "card"]


def _side_effect_tool(effects: list[str], *, defer: bool = False) -> Tool:
    def leak() -> str:
        effects.append("default")
        return "card"

    def pre() -> None:
        effects.append("pre")

    async def charge(note: str) -> str:
        """Charge the card."""
        effects.append("handler")
        effects.append(note)
        return "charged"

    return Tool(handler=charge, pre_hook=pre, default_params={"note": leak}, defer_until_eot=defer)


def _install_gate():
    from timbal.core.speculative_gate import SpeculativeGate, install_speculative_gate

    gate = SpeculativeGate()
    return gate, install_speculative_gate(gate)


def _reset_gate(token) -> None:
    from timbal.core.speculative_gate import reset_speculative_gate

    reset_speculative_gate(token)


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

    async def test_speculation_off_runs_tool_on_commit_only(self) -> None:
        effects: list[str] = []

        def leak() -> str:
            effects.append("default")
            return "card"

        def pre() -> None:
            effects.append("pre")

        async def charge(note: str) -> str:
            """Charge the card."""
            effects.append("handler")
            effects.append(note)
            return "charged"

        session, stt, _tts, model = _session(
            [_tool_call("charge"), "Done."],
            speculative=False,
            tools=[Tool(handler=charge, pre_hook=pre, default_params={"note": leak}, defer_until_eot=True)],
        )
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="charge it"))
            await asyncio.sleep(0.05)
            assert model.call_count == 0
            assert effects == []
            await stt.inject(TranscriptEvent(type="committed", text="charge it"))
            await _until(lambda: effects == ["default", "pre", "handler", "card"])
            await _until(lambda: drive.audio() or "Done" in drive.assistant_text())
        assert session.speculative_generations_started == 0
        assert effects == ["default", "pre", "handler", "card"]

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

    async def test_tools_wait_until_confirm(self) -> None:
        effects: list[str] = []

        def pre() -> None:
            effects.append("pre")

        async def lookup() -> str:
            """Look up the table."""
            effects.append("lookup")
            return "blue"

        async def charge() -> str:
            """Charge the card."""
            effects.append("charge")
            return "charged"

        session, stt, _tts, model = _session(
            [_tool_call("lookup"), _tool_call("charge", "c2"), "All set."],
            tools=[
                Tool(handler=lookup, pre_hook=pre),
                Tool(handler=charge, defer_until_eot=True),
            ],
        )
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count >= 1)
            await asyncio.sleep(0.05)
            assert effects == []
            assert drive.audio() == []
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: effects == ["pre", "lookup", "charge"])
            await _until(lambda: "All set" in drive.assistant_text() or drive.audio())

    async def test_turn_resumed_does_not_leave_a_half_run_tool(self) -> None:
        effects: list[str] = []
        release = asyncio.Event()

        def leak() -> str:
            effects.append("default")
            return "card"

        def pre() -> None:
            effects.append("pre")

        async def charge(note: str) -> str:
            """Charge the card."""
            effects.append("entered")
            try:
                await asyncio.wait_for(release.wait(), timeout=0.2)
            except TimeoutError:
                effects.append("finished")
            effects.append(note)
            return "charged"

        session, stt, _tts, model = _session(
            [_tool_call("charge"), "After."],
            tools=[Tool(handler=charge, pre_hook=pre, default_params={"note": leak})],
        )
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="eager", text="charge it"))
            await _until(lambda: model.call_count >= 1)
            await asyncio.sleep(0.05)
            assert effects == []
            await stt.inject(TranscriptEvent(type="resumed", text=""))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await asyncio.sleep(0.05)
            assert effects == []
        assert effects == []

    async def test_long_utterance_does_not_start_a_tool(self) -> None:
        effects: list[str] = []

        async def lookup() -> str:
            """Look up the table."""
            effects.append("lookup")
            return "blue"

        session, stt, _tts, model = _session(
            [_tool_call("lookup"), "Done."],
            tools=[Tool(handler=lookup)],
        )
        async with _Drive(session, stt) as drive:
            session._caller_speech_started_at = time.monotonic() - 30
            await stt.inject(TranscriptEvent(type="eager", text="a very long dictation"))
            await asyncio.sleep(0.05)
            assert model.call_count == 0
            assert effects == []
            assert session.speculative_generations_started == 0
            await stt.inject(TranscriptEvent(type="committed", text="a very long dictation"))
            await _until(lambda: effects == ["lookup"])
            await _until(lambda: drive.audio() or "Done" in drive.assistant_text())
        assert effects == ["lookup"]

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

    async def test_interrupt_drops_the_hold_so_the_next_turn_speaks(self) -> None:
        session, stt, _tts, model = _session(["Held.", "After."])

        def reply(_messages):
            return "After."

        model.handler = reply
        model.responses = None
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count >= 1)
            await session.interrupt()
            assert session.speculative_generations_discarded == 1
            assert not session._speculative_unconfirmed()
            await stt.inject(TranscriptEvent(type="committed", text="something else entirely"))
            await _until(lambda: drive.audio())
        assert "After" in drive.assistant_text()

    async def test_discarded_run_does_not_reclaim_session_head(self) -> None:
        session, stt, _tts, model = _session(["Held.", "After."])
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: session._turn_run_context is not None)
            attempt = session._speculative_attempt
            assert attempt is not None
            prior = attempt.prior_head
            discarded = session._turn_run_context.id
            assert discarded != prior
            # Abandon rewinds the head, then the still-running turn notes its
            # run and takes the head back. The turn finally has to undo that.
            session._abandon_speculative_attempt(reason="interrupted")
            session._scheduler.note_run(discarded)
            assert session._scheduler.head == discarded
            task = session._current_turn_task
            assert task is not None and not task.done()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            assert session._scheduler.head == prior
            await stt.inject(TranscriptEvent(type="committed", text="something else entirely"))
            await _until(
                lambda: session._last_run_context is not None and session._last_run_context.id != discarded
            )
            await _until(lambda: drive.audio())
        assert session._last_run_context is not None
        assert session._last_run_context.parent_id != discarded
        assert session._scheduler.head != discarded
        assert model.call_count >= 2

    async def test_cancel_before_start_does_not_mark_the_next_turn_discarded(self) -> None:
        session, stt, _tts, model = _session(["After."])
        original = session._spawn_session_task

        def _cancel_speculative(coro, *, name: str | None = None):
            task = original(coro, name=name)
            if name == "speculative-turn":
                task.cancel()
            return task

        session._spawn_session_task = _cancel_speculative
        async with _Drive(session, stt):
            prior = session._scheduler.head
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await stt.inject(TranscriptEvent(type="resumed", text=""))
            await _until(lambda: session.speculative_generations_discarded == 1)
            await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
            # Abandon sets the flag before it joins the task. Wait out that gap.
            await _until(lambda: session._drop_speculative_context is False)
            await stt.inject(TranscriptEvent(type="committed", text="something else entirely"))
            await _until(lambda: session._scheduler.head not in (None, prior) and model.call_count >= 1)
            real = session._scheduler.head
            assert session._speculative_discarded_run_id != real
            await session.interrupt()
            assert session._scheduler.head == real

    async def test_confirm_appends_when_the_caller_repeats_the_previous_line(self) -> None:
        session, stt, _tts, model = _session(["First.", "Second."])

        def reply(messages):
            return "Second." if messages[-1].collect_text().lower().startswith("book") else "First."

        model.handler = reply
        model.responses = None
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: model.call_count >= 1, timeout=5.0)
            await _until(
                lambda: session._current_turn_task is None or session._current_turn_task.done(),
                timeout=5.0,
            )
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            # Windows CI is slow to start the speculative retry and to append
            # the confirmed user line. 2s is not enough there.
            await _until(lambda: model.call_count >= 2, timeout=5.0)
            await stt.inject(TranscriptEvent(type="committed", text="Book a table."))
            await _until(lambda: sum(1 for row in session.transcript if row.role == "user") >= 2, timeout=5.0)
        user_lines = [row.text for row in session.transcript if row.role == "user"]
        assert user_lines == ["book a table", "Book a table."]

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

    async def test_stt_end_discards_an_unconfirmed_turn(self) -> None:
        effects: list[str] = []

        async def lookup() -> str:
            """Look up the table."""
            effects.append("lookup")
            return "blue"

        session, stt, _tts, model = _session(
            [_tool_call("lookup"), "Done."],
            tools=[Tool(handler=lookup)],
        )
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count >= 1)
            await asyncio.sleep(0.05)
            assert effects == []
            await stt.finish()
            await asyncio.wait_for(_until(lambda: session._closed), timeout=2)
        assert effects == []
        assert session.speculative_generations_discarded == 1

    async def test_commit_after_discard_still_starts_a_turn(self) -> None:
        session, stt, _tts, model = _session(["Held."])
        original = session.turn_detector.on_committed

        async def on_committed(text, state):
            decision = await original(text, state)
            session._abandon_speculative_attempt(reason="speech_resumed")
            return decision

        session.turn_detector.on_committed = on_committed
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count >= 1)
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: model.call_count >= 2)
            await _until(lambda: drive.audio())
        assert "Held" in drive.assistant_text()
        assert session.speculative_generations_discarded >= 1

    async def test_confirm_during_detector_does_not_start_a_second_turn(self) -> None:
        session, stt, _tts, model = _session(["Held."])
        original = session.turn_detector.on_committed

        async def on_committed(text, state):
            decision = await original(text, state)
            async with session._user_turn_lock:
                if session._speculative_matches(text):
                    await session._confirm_speculative(text, vad_endpointed=False)
            return decision

        session.turn_detector.on_committed = on_committed
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count >= 1)
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: drive.audio())
            await asyncio.sleep(0.05)
        user_lines = [row.text for row in session.transcript if row.role == "user"]
        assert user_lines == ["book a table"]
        assert model.call_count == 1

    async def test_typed_turn_during_detector_is_not_replaced(self) -> None:
        session, stt, _tts, model = _session(["Held."])
        original = session.turn_detector.on_committed

        async def on_committed(text, state):
            decision = await original(text, state)
            await session.submit_user_text("typed instead")
            return decision

        session.turn_detector.on_committed = on_committed
        async with _Drive(session, stt) as drive:
            await stt.inject(TranscriptEvent(type="eager", text="book a table"))
            await _until(lambda: model.call_count >= 1)
            await stt.inject(TranscriptEvent(type="committed", text="book a table"))
            await _until(lambda: drive.audio())
            await asyncio.sleep(0.05)
        user_lines = [row.text for row in session.transcript if row.role == "user"]
        assert user_lines == ["typed instead"]


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
