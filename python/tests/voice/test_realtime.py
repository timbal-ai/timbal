"""Tests for the speech-to-speech seam (RealtimeModel / RealtimeSession).

No provider exists yet — a scripted FakeRealtimeModel proves the contract:
provider events map onto the same VoiceSessionEvent vocabulary the cascaded
VoiceSession emits, and the interrupt/truncate path keeps transcript and
provider state aligned with what the user actually heard.
"""

import asyncio

import pytest
from timbal.voice import (
    AgentTextDelta,
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    BufferedPlaybackTracker,
    RealtimeEvent,
    RealtimeModel,
    RealtimeSession,
    SessionEnded,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    TranscriptCommitted,
    TranscriptPartial,
    TurnMetricsEvent,
)

SAMPLE_RATE = 16_000
BPS = SAMPLE_RATE * 2  # PCM16 mono


class FakeRealtimeModel(RealtimeModel):
    """Scripted provider: replays a fixed RealtimeEvent sequence.

    ``gates`` lets a test hold the stream between events (e.g. to advance a
    fake clock mid-turn before the interruption arrives).
    """

    def __init__(self, script: list[RealtimeEvent], gates: dict[int, asyncio.Event] | None = None):
        self.script = script
        self.gates = gates or {}
        self.connected = False
        self.closed = False
        self.sent_audio: list[bytes] = []
        self.truncate_calls: list[float] = []

    async def connect(self, audio_input: AudioInputConfig, audio_output: AudioOutputConfig) -> None:
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.connected = True

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def events(self):
        for i, event in enumerate(self.script):
            gate = self.gates.get(i)
            if gate is not None:
                await gate.wait()
            yield event

    async def truncate(self, played_ms: float) -> None:
        self.truncate_calls.append(played_ms)

    async def close(self) -> None:
        self.closed = True


async def _mic(chunks: list[bytes]):
    for c in chunks:
        yield c
    # Keep the uplink open until the session ends (like a live microphone).
    await asyncio.sleep(3600)


async def _collect(session: RealtimeSession, mic_chunks: list[bytes] | None = None, timeout: float = 5.0):
    events = []

    async def drive():
        async for event in session.run(_mic(mic_chunks or [])):
            events.append(event)

    await asyncio.wait_for(drive(), timeout)
    return events


class TestHappyPath:
    async def test_event_flow_matches_cascaded_vocabulary(self):
        pcm = b"\x01\x02" * 1600  # 0.1s
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_partial", text="what's the"),
                RealtimeEvent(type="input_transcript_committed", text="What's the weather?"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="It's "),
                RealtimeEvent(type="output_audio", data=pcm),
                RealtimeEvent(type="output_text_delta", text="sunny."),
                RealtimeEvent(type="output_audio", data=pcm),
                RealtimeEvent(type="turn_done"),
            ]
        )
        session = RealtimeSession(model, record_audio=True)
        events = await _collect(session, mic_chunks=[b"\x00\x00" * 160])

        types = [type(e) for e in events]
        assert types == [
            SessionStarted,
            TranscriptPartial,
            TranscriptCommitted,
            AgentTextDelta,
            AudioOutput,
            AgentTextDelta,
            AudioOutput,
            AgentTextDone,
            TurnMetricsEvent,
            SessionEnded,
        ]

        assert model.connected and model.closed
        assert model.sent_audio == [b"\x00\x00" * 160]

        done = next(e for e in events if isinstance(e, AgentTextDone))
        assert done.text == "It's sunny."

        assert [(t.role, t.text) for t in session.transcript] == [
            ("user", "What's the weather?"),
            ("assistant", "It's sunny."),
        ]
        assert session.input_audio == b"\x00\x00" * 160
        assert session.output_audio == pcm + pcm

    async def test_metrics_subset(self):
        pcm = b"\x01\x02" * 1600
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="Hola"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="¡Hola!"),
                RealtimeEvent(type="output_audio", data=pcm),
                RealtimeEvent(type="turn_done"),
            ]
        )
        session = RealtimeSession(model)
        events = await _collect(session)

        (metrics_event,) = [e for e in events if isinstance(e, TurnMetricsEvent)]
        m = metrics_event.metrics
        assert m.turn_index == 1
        assert m.user_text_chars == len("Hola")
        assert m.eou_to_first_audio_ms is not None and m.eou_to_first_audio_ms >= 0
        # Provider does not expose internal LLM/TTS boundaries.
        assert m.llm_total_ms is None
        assert m.tts_total_ms is None
        assert m.interrupted is False
        assert m.audio_bytes == len(pcm)
        assert session.metrics == [m]

    async def test_turn_done_canonical_text_wins(self):
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="hi"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="Hello"),
                RealtimeEvent(type="turn_done", text="Hello there!"),
            ]
        )
        session = RealtimeSession(model)
        events = await _collect(session)
        done = next(e for e in events if isinstance(e, AgentTextDone))
        assert done.text == "Hello there!"
        assert session.transcript[-1].text == "Hello there!"

    async def test_audio_before_turn_started_opens_turn(self):
        # Providers don't all send an explicit turn_started; first output opens it.
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="hi"),
                RealtimeEvent(type="output_audio", data=b"\x01\x02" * 100),
                RealtimeEvent(type="turn_done"),
            ]
        )
        session = RealtimeSession(model)
        events = await _collect(session)
        (metrics_event,) = [e for e in events if isinstance(e, TurnMetricsEvent)]
        assert metrics_event.metrics.turn_index == 1


class TestInterruption:
    async def test_provider_interrupt_truncates_to_heard_prefix(self):
        clock_now = 0.0
        tracker = BufferedPlaybackTracker(bytes_per_second=BPS, clock=lambda: clock_now)

        gate = asyncio.Event()
        two_seconds = b"\x01\x02" * (SAMPLE_RATE * 2)  # 64000 bytes = 2s at 32kB/s
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="tell me a story"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="Once upon a time there was a fox"),
                RealtimeEvent(type="output_audio", data=two_seconds),
                RealtimeEvent(type="interrupted"),  # gated below
            ],
            gates={4: gate},
        )
        session = RealtimeSession(model, playback_tracker=tracker)

        events = []

        async def drive():
            nonlocal clock_now
            async for event in session.run(_mic([])):
                events.append(event)
                if isinstance(event, AudioOutput):
                    # The user heard 1s of the 2s reply, then barged in.
                    clock_now += 1.0
                    gate.set()
                if isinstance(event, SessionInterrupted):
                    await session.close()

        await asyncio.wait_for(drive(), 5)

        interrupted = next(e for e in events if isinstance(e, SessionInterrupted))
        # 1s of 2s → first half of the text, snapped back to a word boundary.
        assert interrupted.heard_text == "Once upon a"

        # Transcript records only what was heard.
        assert [(t.role, t.text) for t in session.transcript] == [
            ("user", "tell me a story"),
            ("assistant", "Once upon a"),
        ]

        # Provider was told the turn-relative heard position.
        assert len(model.truncate_calls) == 1
        assert model.truncate_calls[0] == pytest.approx(1000.0, abs=1.0)

        # Metrics emitted for the interrupted turn.
        (metrics_event,) = [e for e in events if isinstance(e, TurnMetricsEvent)]
        assert metrics_event.metrics.interrupted is True

    async def test_client_interrupt_path(self):
        clock_now = 0.0
        tracker = BufferedPlaybackTracker(bytes_per_second=BPS, clock=lambda: clock_now)

        gate = asyncio.Event()  # never set: the stream stalls after audio
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="hi"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="A very long reply indeed"),
                RealtimeEvent(type="output_audio", data=b"\x01\x02" * (SAMPLE_RATE * 2)),  # 2s
                RealtimeEvent(type="turn_done"),
            ],
            gates={4: gate},
        )
        session = RealtimeSession(model, playback_tracker=tracker)

        events = []

        async def drive():
            nonlocal clock_now
            async for event in session.run(_mic([])):
                events.append(event)
                if isinstance(event, AudioOutput):
                    clock_now += 1.0  # heard half
                    await session.interrupt()
                if isinstance(event, SessionInterrupted):
                    await session.close()

        await asyncio.wait_for(drive(), 5)

        interrupted = next(e for e in events if isinstance(e, SessionInterrupted))
        assert interrupted.heard_text == "A very long"
        assert model.truncate_calls == [pytest.approx(1000.0, abs=1.0)]

    async def test_barge_in_after_turn_done_truncates_committed_entry(self):
        # turn_done landed (full reply committed, metrics emitted) but the 2s of
        # buffered audio is still draining client-side. A barge-in in that
        # window must rewrite the committed entry to the heard prefix — not
        # silently skip truncation because the turn is no longer active.
        clock_now = 0.0
        tracker = BufferedPlaybackTracker(bytes_per_second=BPS, clock=lambda: clock_now)

        gate = asyncio.Event()  # never set: stream stalls after turn_done
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="hi"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="A very long reply indeed"),
                RealtimeEvent(type="output_audio", data=b"\x01\x02" * (SAMPLE_RATE * 2)),  # 2s
                RealtimeEvent(type="turn_done"),
                RealtimeEvent(type="turn_started"),  # gated, never delivered
            ],
            gates={5: gate},
        )
        session = RealtimeSession(model, playback_tracker=tracker)

        events = []

        async def drive():
            nonlocal clock_now
            async for event in session.run(_mic([])):
                events.append(event)
                if isinstance(event, AgentTextDone):
                    clock_now += 1.0  # user heard half, then barged in
                    await session.interrupt()
                if isinstance(event, SessionInterrupted):
                    await session.close()

        await asyncio.wait_for(drive(), 5)

        interrupted = next(e for e in events if isinstance(e, SessionInterrupted))
        assert interrupted.heard_text == "A very long"
        # Committed entry rewritten in place — no duplicate heard-prefix entry.
        assert [(t.role, t.text) for t in session.transcript] == [
            ("user", "hi"),
            ("assistant", "A very long"),
        ]
        assert model.truncate_calls == [pytest.approx(1000.0, abs=1.0)]
        # Metrics went out once, at normal completion.
        metrics_events = [e for e in events if isinstance(e, TurnMetricsEvent)]
        assert len(metrics_events) == 1
        assert metrics_events[0].metrics.interrupted is False

    async def test_barge_in_after_turn_done_nothing_heard_pops_entry(self):
        clock_now = 0.0
        tracker = BufferedPlaybackTracker(bytes_per_second=BPS, clock=lambda: clock_now)

        gate = asyncio.Event()
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="hi"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="Unheard reply"),
                RealtimeEvent(type="output_audio", data=b"\x01\x02" * SAMPLE_RATE),  # 1s
                RealtimeEvent(type="turn_done"),
                RealtimeEvent(type="turn_started"),  # gated, never delivered
            ],
            gates={5: gate},
        )
        session = RealtimeSession(model, playback_tracker=tracker)

        events = []

        async def drive():
            async for event in session.run(_mic([])):
                events.append(event)
                if isinstance(event, AgentTextDone):
                    await session.interrupt()  # clock never advances: 0 bytes heard
                if isinstance(event, SessionInterrupted):
                    await session.close()

        await asyncio.wait_for(drive(), 5)

        assert [(t.role, t.text) for t in session.transcript] == [("user", "hi")]
        assert model.truncate_calls == [pytest.approx(0.0, abs=1.0)]

    async def test_new_turn_baseline_excludes_previous_turns_draining_tail(self):
        # Turn A's 2s reply is still draining when turn B starts and is barged
        # into. B's heard_bytes / truncate must be relative to B's own audio,
        # not include A's unplayed tail (the playhead is behind the axis
        # position where B's audio begins).
        clock_now = 0.0
        tracker = BufferedPlaybackTracker(bytes_per_second=BPS, clock=lambda: clock_now)

        two_seconds = b"\x01\x02" * (SAMPLE_RATE * 2)
        gate = asyncio.Event()  # never set: stream stalls after B's audio
        model = FakeRealtimeModel(
            [
                RealtimeEvent(type="input_transcript_committed", text="first"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="Turn A full reply text"),
                RealtimeEvent(type="output_audio", data=two_seconds),
                RealtimeEvent(type="turn_done"),
                RealtimeEvent(type="input_transcript_committed", text="second"),
                RealtimeEvent(type="turn_started"),
                RealtimeEvent(type="output_text_delta", text="Turn B reply is this"),
                RealtimeEvent(type="output_audio", data=two_seconds),
                RealtimeEvent(type="turn_started"),  # gated, never delivered
            ],
            gates={9: gate},
        )
        session = RealtimeSession(model, playback_tracker=tracker)

        events = []
        audio_chunks_seen = 0

        async def drive():
            nonlocal clock_now, audio_chunks_seen
            async for event in session.run(_mic([])):
                events.append(event)
                if isinstance(event, AudioOutput):
                    audio_chunks_seen += 1
                    if audio_chunks_seen == 2:
                        # A played fully (2s) + 1s into B's 2s reply.
                        clock_now += 3.0
                        await session.interrupt()
                if isinstance(event, SessionInterrupted):
                    await session.close()

        await asyncio.wait_for(drive(), 5)

        interrupted = next(e for e in events if isinstance(e, SessionInterrupted))
        # 1s of B's 2s → first half of B's text, snapped to a word boundary.
        # (With the old playhead baseline, heard_bytes covered A's tail too and
        # the full "Turn B reply is this" would have been reported.)
        assert interrupted.heard_text == "Turn B"
        assert [(t.role, t.text) for t in session.transcript] == [
            ("user", "first"),
            ("assistant", "Turn A full reply text"),
            ("user", "second"),
            ("assistant", "Turn B"),
        ]
        # Turn-relative: 1000ms into B, not 3000ms from session start.
        assert model.truncate_calls == [pytest.approx(1000.0, abs=1.0)]
        interrupted_metrics = [
            e.metrics for e in events if isinstance(e, TurnMetricsEvent) and e.metrics.interrupted
        ]
        assert len(interrupted_metrics) == 1
        assert interrupted_metrics[0].turn_index == 2
        assert interrupted_metrics[0].heard_bytes == BPS  # 1s of B

    async def test_interrupt_when_idle_is_noop(self):
        model = FakeRealtimeModel(
            [RealtimeEvent(type="input_transcript_committed", text="hi")],
            gates={0: asyncio.Event()},  # stall immediately
        )
        session = RealtimeSession(model)

        events = []

        async def drive():
            started = False
            async for event in session.run(_mic([])):
                events.append(event)
                if isinstance(event, SessionStarted) and not started:
                    started = True
                    await session.interrupt()  # nothing playing, no turn
                    await session.close()

        await asyncio.wait_for(drive(), 5)
        assert not any(isinstance(e, SessionInterrupted) for e in events)
        assert model.truncate_calls == []


class TestLifecycle:
    async def test_model_error_event_surfaces(self):
        model = FakeRealtimeModel(
            [RealtimeEvent(type="error", text="provider exploded")],
        )
        session = RealtimeSession(model)
        events = await _collect(session)
        assert any(isinstance(e, SessionError) and "provider exploded" in e.message for e in events)
        # Stream end still closes the session cleanly.
        assert isinstance(events[-1], SessionEnded)
        assert model.closed

    async def test_stream_end_closes_session(self):
        model = FakeRealtimeModel([])
        session = RealtimeSession(model)
        events = await _collect(session)
        assert [type(e) for e in events] == [SessionStarted, SessionEnded]
        assert model.closed
