"""Opt-in provider continuation: never rewind heard speech or tool work."""

import asyncio
from copy import deepcopy

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import TranscriptEvent, VoiceSession
from timbal.voice.config import VoiceConfig
from timbal.voice.turn_detection import CommitAction, PartialDecision, ProviderTurnDetector

from .test_session import FakePlaybackTracker, MockSTT, MockTTS
from .test_turn_detection import _state

FIRST = "Pump three, I think."
REST = "The one next to the cooling unit."


def candidate(**kwargs):
    return _state(
        **{
            "assistant_active": True,
            "audio_playing": True,
            "active_user_text": FIRST,
            "user_turn_active": True,
            "reply_heard_bytes": 0,
            "tool_started": False,
            "seconds_since_last_commit": 0.5,
            **kwargs,
        }
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_window(value):
    with pytest.raises(ValueError):
        ProviderTurnDetector(merge_resumed_secs=value)
    with pytest.raises(ValueError):
        VoiceConfig(merge_resumed_secs=value)


async def test_defaults_do_not_merge_or_delay_partials():
    detector = ProviderTurnDetector()
    assert (await detector.on_committed(REST, candidate())).action == CommitAction.NEW_TURN
    assert await detector.on_partial(REST, candidate()) == PartialDecision.BARGE_IN


async def test_merge_and_cumulative_refinement():
    detector = ProviderTurnDetector(merge_resumed_secs=2)
    assert await detector.on_partial(REST, candidate()) == PartialDecision.IGNORE
    result = await detector.on_committed(REST, candidate())
    assert result.action == CommitAction.CONTINUE_TURN
    assert result.text == f"{FIRST} {REST}"
    cumulative = await detector.on_committed(f"{FIRST} {REST}", candidate())
    assert cumulative.text == result.text
    assert (await detector.on_committed(FIRST, candidate())).action == CommitAction.IGNORE


@pytest.mark.parametrize(
    "changes",
    [
        {"reply_heard_bytes": 1},
        {"reply_heard_bytes": None},
        {"tool_started": True},
        {"tool_started": None},
        {"user_turn_active": False},
        {"holding": True},
        {"seconds_since_last_commit": 2.0},
        {"seconds_since_last_commit": -1},
        {"active_user_text": ""},
    ],
)
async def test_unsafe_candidates_are_new_turns(changes):
    result = await ProviderTurnDetector(merge_resumed_secs=2).on_committed(REST, candidate(**changes))
    assert result.action == CommitAction.NEW_TURN
    assert result.text == REST


@pytest.mark.parametrize("text", ["Sí.", "D'acord.", "Vale.", "Yeah.", "No.", "Espera.", "Stop."])
async def test_no_acknowledgment_word_filter(text):
    result = await ProviderTurnDetector(merge_resumed_secs=2).on_committed(text, candidate(user_turn_active=False))
    assert result.action == CommitAction.NEW_TURN


class KnownPlayback(FakePlaybackTracker):
    @property
    def ack_received(self):
        return True


def session_for(model=None, *, tracing_provider=InMemoryTracingProvider, **kwargs):
    return VoiceSession(
        Agent(name="desk", model=model or TestModel(), tracing_provider=tracing_provider),
        MockSTT(),
        MockTTS(),
        turn_detector="provider",
        **kwargs,
    )


def test_reply_playback_distinguishes_filler_and_unknown():
    tracker = KnownPlayback()
    session = session_for(playback_tracker=tracker)
    session._turn_played_baseline = 0
    session._turn_tts_segment_records = [["", 100], ["Reply.", 200]]
    tracker.played = 100
    assert session._reply_heard_bytes() == 0
    tracker.played = 101
    assert session._reply_heard_bytes() == 1
    session._turn_heard_bytes = 120
    tracker.played = 200
    assert session._reply_heard_bytes() == 20
    session.playback = FakePlaybackTracker()
    assert session._reply_heard_bytes() is None
    session._turn_tts_segment_records = [["", 100]]
    assert session._reply_heard_bytes() == 0


@pytest.mark.parametrize("fillers", [0, 2])
@pytest.mark.parametrize("persisted", [False, True])
async def test_resumed_commit_replaces_prompt_transcript_and_memory(fillers, persisted, tmp_path):
    started = asyncio.Event()
    calls = []

    class PausedModel(TestModel):
        async def stream(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()
            async for item in super().stream(messages, **kwargs):
                yield item

    provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl") if persisted else None
    session = session_for(
        PausedModel(responses=["Understood."]),
        merge_resumed_secs=2,
        tracing_provider=provider or InMemoryTracingProvider,
    )
    try:
        await session._handle_committed(FIRST)
        await asyncio.wait_for(started.wait(), 2)
        assert session._turn_state().tool_started is False
        for i in range(fillers):
            await session._speak_filler(f"Thinking filler {i}.", "")
        await session._handle_committed(REST)
        await asyncio.wait_for(session._current_turn_task, 2)
        merged = f"{FIRST} {REST}"
        assert [m.collect_text() for m in calls[-1] if m.role == "user"] == [merged]
        assert [e.text for e in session.transcript if e.role == "user"] == [merged]
        assert [m.collect_text() for m in session._last_run_context.root_span().memory if m.role == "user"] == [merged]
        commits = [e for e in list(session._event_queue._queue) if e.type == "transcript_committed"]
        assert commits[-1].replace is True
        assert len([e for e in session.transcript if e.filler]) == fillers
        if provider is not None:
            stored = await provider.get(RunContext(parent_id=session._last_run_context.id, tracing_provider=provider))
            root = next(span for span in stored.values() if span.parent_call_id is None)
            assert [Message.validate(m).collect_text() for m in root.memory if Message.validate(m).role == "user"] == [
                merged
            ]
    finally:
        await session.close()


async def test_tool_started_while_detector_waits_preserves_history():
    model_started = asyncio.Event()
    let_tool_start = asyncio.Event()
    tool_started = asyncio.Event()
    calls = []
    executions = []

    class ToolModel(TestModel):
        async def stream(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            if len(calls) == 1:
                model_started.set()
                await let_tool_start.wait()
            async for item in super().stream(messages, **kwargs):
                yield item

    async def lookup():
        executions.append("lookup")
        tool_started.set()
        await asyncio.Event().wait()

    responses = iter(
        [Message(role="assistant", content=[ToolUseContent(id="c1", name="lookup", input={})]), "Understood."]
    )

    class RacingDetector(ProviderTurnDetector):
        async def on_committed(self, text, state):
            result = await super().on_committed(text, state)
            if result.action == CommitAction.CONTINUE_TURN:
                let_tool_start.set()
                await asyncio.wait_for(tool_started.wait(), 2)
            return result

    session = VoiceSession(
        Agent(name="desk", model=ToolModel(handler=lambda _: next(responses)), tools=[lookup]),
        MockSTT(),
        MockTTS(),
        turn_detector=RacingDetector(merge_resumed_secs=2),
    )
    try:
        await session._handle_committed(FIRST)
        await asyncio.wait_for(model_started.wait(), 2)
        await session._handle_committed(REST)
        await asyncio.wait_for(session._current_turn_task, 2)
        assert executions == ["lookup"]
        assert [e.text for e in session.transcript if e.role == "user"] == [FIRST, REST]
        assert [m.collect_text() for m in calls[-1] if m.role == "user"] == [FIRST, REST]
    finally:
        await session.close()


async def test_first_partial_overlap_is_retained_through_commit(monkeypatch):
    tracker = KnownPlayback()
    tracker.playing = True
    stt = MockSTT(
        script=[
            TranscriptEvent(type="partial", text="One"),
            TranscriptEvent(type="partial", text="One more"),
            TranscriptEvent(type="committed", text="One more thing."),
        ]
    )
    session = session_for(playback_tracker=tracker)
    session.stt = stt
    seen = []

    async def partial(_text, state):
        seen.append(state.caller_partial_started_during_playback)
        tracker.playing = False
        return PartialDecision.IGNORE

    async def committed(_text):
        seen.append(session._turn_state().caller_partial_started_during_playback)

    monkeypatch.setattr(session.turn_detector, "on_partial", partial)
    monkeypatch.setattr(session, "_handle_committed_inner", committed)
    await stt.connect(session.audio_input)
    await asyncio.wait_for(session._process_stt_events(), 2)
    assert seen == [True, True, True]
    assert session._turn_state().caller_partial_started_during_playback is None


def test_window_requires_provider_policy():
    with pytest.raises(ValueError, match="provider"):
        VoiceSession(
            Agent(name="desk", model=TestModel()), MockSTT(), MockTTS(), turn_detector="heuristic", merge_resumed_secs=2
        )


async def test_tool_evidence_during_interrupt_cannot_rewrite_user(monkeypatch):
    started = asyncio.Event()
    calls = []

    class PausedModel(TestModel):
        async def stream(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()
            async for item in super().stream(messages, **kwargs):
                yield item

    session = session_for(PausedModel(responses=["Understood."]), merge_resumed_secs=2)
    original = session.interrupt

    async def racing_interrupt(**kwargs):
        await original(**kwargs)
        session._turn_tool_call_observed = True

    try:
        await session._handle_committed(FIRST)
        await asyncio.wait_for(started.wait(), 2)
        monkeypatch.setattr(session, "interrupt", racing_interrupt)
        await session._handle_committed(REST)
        await asyncio.wait_for(session._current_turn_task, 2)
        assert [e.text for e in session.transcript if e.role == "user"] == [FIRST, REST]
        assert [m.collect_text() for m in calls[-1] if m.role == "user"] == [FIRST, REST]
    finally:
        await session.close()


@pytest.mark.parametrize(
    "previous,incoming,expected",
    [
        ("Book a cab.", "Book a cab to the station.", "Book a cab to the station."),
        ("Pump 3.", "Pump 30 next to it.", "Pump 3. Pump 30 next to it."),
        ("Vull una cita.", "Vull una cita demà.", "Vull una cita demà."),
        ("Necesito una cita.", "Necesito una cita mañana.", "Necesito una cita mañana."),
    ],
)
async def test_cumulative_transcripts_use_token_boundaries(previous, incoming, expected):
    result = await ProviderTurnDetector(merge_resumed_secs=2).on_committed(
        incoming, candidate(active_user_text=previous)
    )
    assert result.text == expected


def test_late_playback_correction_makes_heard_position_unknown():
    tracker = KnownPlayback()
    session = session_for(playback_tracker=tracker)
    session._turn_played_baseline = 1000
    session._turn_tts_segment_records = [["Answer.", 500]]
    tracker.played = 900
    assert session._reply_heard_bytes() is None
    session._turn_heard_bytes = 0  # interrupt's clamped snapshot cannot restore certainty.
    assert session._reply_heard_bytes() is None


@pytest.mark.parametrize("location", ["output", "memory"])
def test_provider_tool_calls_without_child_span_block_merge(location):
    from types import SimpleNamespace

    tool_message = Message(
        role="assistant", content=[ToolUseContent(id="server1", name="search", input={}, is_server_tool_use=True)]
    )
    root = SimpleNamespace(path="desk", output=None, memory=[tool_message, Message(role="user", content=[])])
    llm = SimpleNamespace(path="desk.llm", output=None)
    ctx = SimpleNamespace(_trace=SimpleNamespace(data={"root": root, "llm": llm}), root_span=lambda: root)
    session = session_for()
    session._turn_resume_memory_start = 2
    assert session._turn_has_tool_work(ctx) is False  # Earlier turn's tool history is fine.
    if location == "output":
        llm.output = tool_message
    else:
        root.memory.append(tool_message)
    assert session._turn_has_tool_work(ctx) is True


async def test_redacted_input_falls_back_without_duplicating_combined_prompt():
    from timbal.guardrails import DetectPII

    started = asyncio.Event()
    calls = []

    class PausedModel(TestModel):
        async def stream(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()
            async for item in super().stream(messages, **kwargs):
                yield item

    first = "Send the report to alice@example.com."
    rest = "The quarterly report."
    session = VoiceSession(
        Agent(
            name="desk",
            model=PausedModel(responses=["Understood."]),
            guardrails=[DetectPII(on_input="redact", types=["email"])],
        ),
        MockSTT(),
        MockTTS(),
        turn_detector="provider",
        merge_resumed_secs=2,
    )
    try:
        await session._handle_committed(first)
        await asyncio.wait_for(started.wait(), 2)
        assert session.turn_detector.can_merge_resumed(session._turn_state())
        await session._handle_committed(rest)
        await asyncio.wait_for(session._current_turn_task, 2)
        user_prompts = [m.collect_text() for m in calls[-1] if m.role == "user"]
        assert len(user_prompts) == 2
        assert "alice@example.com" not in user_prompts[0]
        assert user_prompts[-1] == rest
        assert [e.text for e in session.transcript if e.role == "user"] == [first, rest]
    finally:
        await session.close()


async def test_partial_inside_window_allows_commit_after_window_elapsed():
    detector = ProviderTurnDetector(merge_resumed_secs=2)
    # Caller starts speaking at 0.5s (< 2.0s), then finishes the clause at 2.4s.
    assert (
        await detector.on_partial(REST, candidate(seconds_since_last_commit=0.5, partials_since_last_commit=1))
        == PartialDecision.IGNORE
    )
    assert (
        await detector.on_partial(REST, candidate(seconds_since_last_commit=2.1, partials_since_last_commit=2))
        == PartialDecision.IGNORE
    )
    result = await detector.on_committed(REST, candidate(seconds_since_last_commit=2.4, partials_since_last_commit=2))
    assert result.action == CommitAction.CONTINUE_TURN
    assert result.text == f"{FIRST} {REST}"
    # Once consumed, a subsequent commit outside the window without an in-window partial does not merge.
    after = await detector.on_committed(REST, candidate(seconds_since_last_commit=2.5, partials_since_last_commit=1))
    assert after.action == CommitAction.NEW_TURN


async def test_resumed_partial_silences_filler_playback_without_cancelling_turn():
    started = asyncio.Event()
    calls = []

    class PausedModel(TestModel):
        async def stream(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()
            async for item in super().stream(messages, **kwargs):
                yield item

    tracker = KnownPlayback()
    stt = MockSTT(
        script=[
            TranscriptEvent(type="partial", text=REST),
            TranscriptEvent(type="committed", text=REST),
        ]
    )
    session = session_for(PausedModel(responses=["Understood."]), merge_resumed_secs=2, playback_tracker=tracker)
    session.stt = stt
    playing_at_commit = []
    orig_committed = session._handle_committed

    async def capture_committed(text):
        playing_at_commit.append((tracker.interrupt_calls, not session._current_turn_task.done()))
        await orig_committed(text)

    try:
        await session._handle_committed(FIRST)
        await asyncio.wait_for(started.wait(), 2)
        await session._speak_filler("Let me check that.", "")
        tracker.playing = True
        session._handle_committed = capture_committed
        await stt.connect(session.audio_input)
        await asyncio.wait_for(session._process_stt_events(), 2)
        assert playing_at_commit[-1] == (1, True)
        merged = f"{FIRST} {REST}"
        assert [m.collect_text() for m in calls[-1] if m.role == "user"] == [merged]
        assert [e.text for e in session.transcript if e.role == "user"] == [merged]
    finally:
        await session.close()


async def test_echo_and_stale_partials_do_not_latch_overlap_flag():
    tracker = KnownPlayback()
    tracker.playing = True
    session = session_for(playback_tracker=tracker)
    session._turn_assistant_text = "During a service we inspect the brakes."
    session._turn_tts_scheduled_text = session._turn_assistant_text
    session._turn_tts_segment_records = [[session._turn_assistant_text, 1000]]
    tracker.played = 1000

    stt = MockSTT(
        script=[
            # Echo partial during playback should not latch caller_partial_started_during_playback.
            TranscriptEvent(type="partial", text="inspect the brakes"),
        ]
    )
    session.stt = stt
    await stt.connect(session.audio_input)
    await asyncio.wait_for(session._process_stt_events(), 2)
    assert session._turn_state().caller_partial_started_during_playback is None

    # A stale uncommitted partial during playback must expire before the next utterance in silence.
    session._caller_partial_started_during_playback = True
    session._last_partial_at = 1.0
    assert session._turn_state().caller_partial_started_during_playback is None


async def test_empty_turn_retry_preserves_resumed_merge():
    retry_started = asyncio.Event()
    calls = []

    class EmptyThenPausedModel(TestModel):
        async def stream(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            if len(calls) == 2:
                retry_started.set()
                await asyncio.Event().wait()
            async for item in super().stream(messages, **kwargs):
                yield item

    session = session_for(EmptyThenPausedModel(responses=["", "Understood."]), merge_resumed_secs=2)
    try:
        await session._handle_committed(FIRST)
        await asyncio.wait_for(retry_started.wait(), 2)
        assert session._turn_state().tool_started is False
        await session._handle_committed(REST)
        await asyncio.wait_for(session._current_turn_task, 2)
        merged = f"{FIRST} {REST}"
        assert [m.collect_text() for m in calls[-1] if m.role == "user"] == [merged]
        assert [e.text for e in session.transcript if e.role == "user"] == [merged]
    finally:
        await session.close()

