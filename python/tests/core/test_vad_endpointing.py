"""Tests for the VAD endpointing fast path.

Covers :func:`endpointing_delay`, the :class:`VadEndpointer` state machine
(with a fake VAD — no onnxruntime needed), and the full VoiceSession
integration: Silero speech-stop → audio EOU score → forced ``stt.commit()`` →
normal committed-transcript turn flow.
"""
# ruff: noqa: ARG002

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice import (
    AudioInputConfig,
    LocalAudioTurnDetector,
    VadEndpointer,
    VoiceSession,
    endpointing_delay,
)
from timbal.voice.eou import AudioEouModel
from timbal.voice.metrics import TurnMetricsEvent
from timbal.voice.session import (
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVad:
    """One VAD frame per input byte: ``S`` → speech (1.0), anything else → 0.0."""

    def __init__(self) -> None:
        self.started_with: int | None = None

    async def start(self, *, sample_rate: int) -> None:
        self.started_with = sample_rate

    def reset(self) -> None:
        pass

    def process(self, chunk: bytes) -> list[float]:
        return [1.0 if b == ord("S") else 0.0 for b in chunk]


class _FixedEou(AudioEouModel):
    def __init__(self, p: float = 0.95) -> None:
        self.p = p
        self.calls = 0

    async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:
        self.calls += 1
        return self.p


def _endpointer(vad: FakeVad | None = None, **knobs) -> VadEndpointer:
    defaults = dict(
        stop_silence_secs=0.064,  # 2 frames
        min_speech_secs=0.064,  # 2 frames
        min_delay_secs=0.0,
        max_delay_secs=0.05,
        min_commit_interval_secs=0.0,
    )
    defaults.update(knobs)
    return VadEndpointer(vad or FakeVad(), **defaults)


class _Recorder:
    """Bind targets that record calls and return scripted values."""

    def __init__(
        self,
        *,
        p: float | None = 0.95,
        gate: bool = True,
        p_text: float | None = None,
    ) -> None:
        self.p = p
        self.gate = gate
        self.p_text = p_text
        self.score_calls = 0
        self.commit_calls = 0
        self.text_score_calls = 0

    async def score(self) -> float | None:
        self.score_calls += 1
        return self.p

    async def text_score(self) -> float | None:
        self.text_score_calls += 1
        return self.p_text

    async def commit(self) -> None:
        self.commit_calls += 1

    def should_commit(self) -> bool:
        return self.gate


async def _started(ep: VadEndpointer, rec: _Recorder) -> VadEndpointer:
    ep.bind(
        score=rec.score,
        commit=rec.commit,
        should_commit=rec.should_commit,
        text_score=rec.text_score if rec.p_text is not None else None,
    )
    await ep.start(sample_rate=16_000)
    return ep


async def _settle(secs: float = 0.15) -> None:
    await asyncio.sleep(secs)


# ---------------------------------------------------------------------------
# endpointing_delay
# ---------------------------------------------------------------------------


class TestEndpointingDelay:
    def test_confident_maps_to_min(self):
        assert endpointing_delay(1.0, min_delay=0.0, max_delay=3.0) == 0.0
        assert endpointing_delay(1.0) == VadEndpointer.MIN_DELAY_SECS

    def test_zero_maps_to_max(self):
        assert endpointing_delay(0.0, min_delay=0.0, max_delay=3.0) == 3.0
        assert endpointing_delay(0.0) == 3.0

    def test_monotonic_decreasing(self):
        delays = [endpointing_delay(p / 10) for p in range(11)]
        assert delays == sorted(delays, reverse=True)

    def test_clamps_out_of_range(self):
        assert endpointing_delay(1.7) == endpointing_delay(1.0)
        assert endpointing_delay(-0.5) == endpointing_delay(0.0)

    def test_incomplete_exceeds_provider_debounce(self):
        # p below the completion threshold must compute a delay longer than
        # the ~1.2s ElevenLabs VAD debounce — the provider commit wins and the
        # existing HOLD machinery handles the incomplete utterance.
        assert endpointing_delay(0.4, min_delay=0.0) > 1.0

    def test_confident_respects_livekit_floor(self):
        # LiveKit min_delay shape: even p≈1 never collapses to ~0.
        assert endpointing_delay(0.95) >= VadEndpointer.MIN_DELAY_SECS - 1e-9
        assert VadEndpointer.MIN_DELAY_SECS >= 0.4

    def test_curve_still_fast_without_floor(self):
        assert endpointing_delay(0.95, min_delay=0.0) < 0.05


# ---------------------------------------------------------------------------
# VadEndpointer state machine
# ---------------------------------------------------------------------------


class TestVadEndpointer:
    async def test_speech_then_silence_commits(self):
        rec = _Recorder(p=0.95)
        ep = await _started(_endpointer(), rec)
        ep.push(b"SSS")  # 3 speech frames ≥ min_speech
        ep.push(b"\x00\x00")  # 2 silence frames ≥ stop_silence
        await _settle()
        assert rec.score_calls == 1
        assert rec.commit_calls == 1

    async def test_phoneme_dips_still_accumulate_speech(self):
        # Silero routinely dips below threshold between phonemes. A short word
        # ("work.") is 0.1–0.3s of speech with dips — the min_speech
        # accumulator must survive them (live failure: "Um, work." never
        # armed, session stalled until the user spoke again).
        rec = _Recorder(p=0.9)
        ep = await _started(_endpointer(min_speech_secs=0.12, stop_silence_secs=0.096), rec)
        # 4 speech frames (0.128s total) interleaved with 1-frame dips.
        ep.push(b"S\x00S\x00S\x00S")
        ep.push(b"\x00\x00\x00")  # 3 silence frames ≥ stop_silence
        await _settle()
        assert rec.score_calls == 1
        assert rec.commit_calls == 1

    async def test_separated_blips_do_not_accumulate_into_phantom_utterance(self):
        # Isolated noise blips separated by real silence must not sum up to
        # min_speech across utterance boundaries.
        rec = _Recorder(p=0.9)
        ep = await _started(_endpointer(min_speech_secs=0.12, stop_silence_secs=0.096), rec)
        for _ in range(4):
            ep.push(b"S")  # one frame of "speech"
            ep.push(b"\x00" * 6)  # real silence resets the accumulator
        await _settle()
        assert rec.score_calls == 0
        assert rec.commit_calls == 0

    async def test_short_blip_never_endpoints(self):
        rec = _Recorder()
        ep = await _started(_endpointer(min_speech_secs=0.2), rec)
        ep.push(b"S")  # 1 frame (0.032s) < min_speech
        ep.push(b"\x00\x00\x00")
        await _settle()
        assert rec.score_calls == 0
        assert rec.commit_calls == 0

    async def test_one_attempt_per_speech_stop(self):
        rec = _Recorder(p=0.95)
        ep = await _started(_endpointer(), rec)
        ep.push(b"SSS")
        ep.push(b"\x00" * 20)  # long silence — still one endpoint
        await _settle()
        assert rec.score_calls == 1
        assert rec.commit_calls == 1

    async def test_resumed_speech_cancels_pending(self):
        rec = _Recorder(p=0.1)  # incomplete → long delay
        ep = await _started(_endpointer(max_delay_secs=5.0), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle(0.05)  # score happened, now sleeping the long delay
        assert rec.score_calls == 1
        ep.push(b"SS")  # user resumes → pending cancelled
        await _settle(0.05)
        assert rec.commit_calls == 0

    async def test_text_incomplete_bumps_delay(self):
        """Audio-complete + hedge text → delay floored at TEXT_INCOMPLETE_DELAY,
        so a continuation can cancel before commit (live: "I don't know" → story)."""
        rec = _Recorder(p=0.98, p_text=0.2)
        ep = await _started(
            _endpointer(
                min_delay_secs=0.0,
                max_delay_secs=0.05,
                text_incomplete_delay_secs=0.4,
            ),
            rec,
        )
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle(0.1)  # past audio-only delay (0.05), still in text bump
        assert rec.score_calls == 1
        assert rec.text_score_calls == 1
        assert rec.commit_calls == 0
        ep.push(b"SS")  # continuation cancels during the bumped wait
        await _settle(0.05)
        assert rec.commit_calls == 0

    async def test_single_noise_frame_does_not_cancel(self):
        rec = _Recorder(p=0.5)
        # delay(0.5) = 0.25 * max → 0.1s with max 0.4: long enough to inject a blip.
        ep = await _started(_endpointer(max_delay_secs=0.4), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle(0.02)
        ep.push(b"S")  # 1 frame < SPEECH_RESUME_SECS (2 frames) — must not cancel
        await _settle(0.3)
        assert rec.commit_calls == 1

    async def test_session_gate_blocks_commit(self):
        rec = _Recorder(p=0.95, gate=False)
        ep = await _started(_endpointer(), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle()
        assert rec.score_calls == 0  # gated before scoring
        assert rec.commit_calls == 0

    async def test_none_score_never_commits(self):
        rec = _Recorder(p=None)
        ep = await _started(_endpointer(), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle()
        assert rec.score_calls == 1
        assert rec.commit_calls == 0

    async def test_notify_committed_cancels_pending(self):
        rec = _Recorder(p=0.1)
        ep = await _started(_endpointer(max_delay_secs=5.0), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle(0.05)
        ep.notify_committed()  # provider debounce beat us
        await _settle(0.05)
        assert rec.commit_calls == 0

    async def test_min_commit_interval_suppresses_second(self):
        rec = _Recorder(p=0.95)
        ep = await _started(_endpointer(min_commit_interval_secs=60.0), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle()
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle()
        assert rec.commit_calls == 1

    async def test_close_stops_everything(self):
        rec = _Recorder(p=0.1)
        ep = await _started(_endpointer(max_delay_secs=5.0), rec)
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle(0.05)
        await ep.close()
        ep.push(b"SSS")
        ep.push(b"\x00\x00")
        await _settle(0.05)
        assert rec.commit_calls == 0

    async def test_start_requires_bind(self):
        ep = _endpointer()
        with pytest.raises(RuntimeError):
            await ep.start(sample_rate=16_000)


class TestSpeechWindow:
    """``speech_secs_in_window`` — the barge-in hallucination-veto signal."""

    _FRAME = 512 / 16_000

    async def test_none_before_start(self):
        ep = _endpointer()
        assert ep.speech_secs_in_window(2.0) is None

    async def test_none_when_no_frames_processed(self):
        """Started but starved (no mic audio) → no evidence, must not veto."""
        ep = await _started(_endpointer(), _Recorder())
        assert ep.speech_secs_in_window(2.0) is None
        await ep.close()

    async def test_counts_recent_speech_frames(self):
        ep = await _started(_endpointer(), _Recorder())
        ep.push(b"SSSSSSSS")  # 8 speech frames
        assert ep.speech_secs_in_window(2.0) == pytest.approx(8 * self._FRAME)
        await ep.close()

    async def test_silence_counts_zero(self):
        """Frames flowing but all sub-threshold → 0.0 (veto-able), not None."""
        ep = await _started(_endpointer(), _Recorder())
        ep.push(b"\x00" * 8)
        assert ep.speech_secs_in_window(2.0) == 0.0
        await ep.close()


# ---------------------------------------------------------------------------
# VoiceSession integration
# ---------------------------------------------------------------------------


class EndpointSTT(SpeechToText):
    """STT with externally injected events; ``commit()`` finalizes ``pending_text``."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self.pending_text = ""
        self.commit_calls = 0

    async def connect(self, config: AudioInputConfig) -> None:
        pass

    async def push_audio(self, chunk: bytes) -> None:
        pass

    async def commit(self) -> None:
        self.commit_calls += 1
        if self.pending_text:
            await self._queue.put(TranscriptEvent(type="committed", text=self.pending_text))
            self.pending_text = ""

    async def inject_partial(self, text: str) -> None:
        self.pending_text = text
        await self._queue.put(TranscriptEvent(type="partial", text=text))

    async def finish(self) -> None:
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.text:
                yield item

    async def close(self) -> None:
        pass


class _SilentTTS(TextToSpeech):
    async def connect(self, config: AudioOutputConfig) -> None:
        pass

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        yield b"\x00\x01" * 8

    async def close(self) -> None:
        pass


class TestVoiceSessionEndpointing:
    async def _run_endpointed_session(self, eou_p: float = 0.95):
        stt = EndpointSTT()
        detector = LocalAudioTurnDetector(audio_eou=_FixedEou(eou_p))
        endpointer = _endpointer(FakeVad())
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["Sure thing."]), tools=[]),
            stt=stt,
            tts=_SilentTTS(),
            turn_detector=detector,
            vad_endpointing=endpointer,
        )

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def _audio() -> AsyncIterator[bytes]:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    return
                yield chunk

        events = []

        async def _consume() -> None:
            async for ev in session.run(_audio()):
                events.append(ev)

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0.05)  # session + endpointer started

        # Enough buffered PCM for score_recent_audio (≥ sample_rate bytes),
        # all mapped to VAD silence by FakeVad.
        await audio_queue.put(b"\x00" * 20_000)
        # The gate requires a partial newer than the last commit.
        await stt.inject_partial("what's the weather like today?")
        await asyncio.sleep(0.02)
        # Speech, then silence → endpoint fires.
        await audio_queue.put(b"S" * 4)
        await audio_queue.put(b"\x00" * 4)

        # Give the endpointer time to score + commit and the turn to run.
        for _ in range(100):
            await asyncio.sleep(0.02)
            if any(isinstance(e, TurnMetricsEvent) for e in events):
                break

        await stt.finish()
        await audio_queue.put(None)
        await asyncio.wait_for(consumer, timeout=5.0)
        return session, stt, events

    async def test_endpointer_forces_commit_and_runs_turn(self):
        session, stt, events = await self._run_endpointed_session(eou_p=0.95)
        assert stt.commit_calls >= 1
        user_entries = [e for e in session.transcript if e.role == "user"]
        assert [e.text for e in user_entries] == ["what's the weather like today?"]
        metrics = [e.metrics for e in events if isinstance(e, TurnMetricsEvent)]
        assert metrics and metrics[0].vad_endpointed is True

    async def test_incomplete_score_defers_to_provider(self):
        # p=0.1 → delay ≈ 0.81 * max(3.0) ≈ 2.4s — far longer than this test
        # waits, so no forced commit happens.
        stt = EndpointSTT()
        detector = LocalAudioTurnDetector(audio_eou=_FixedEou(0.1))
        endpointer = _endpointer(FakeVad(), max_delay_secs=3.0)
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
            stt=stt,
            tts=_SilentTTS(),
            turn_detector=detector,
            vad_endpointing=endpointer,
        )

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def _audio() -> AsyncIterator[bytes]:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    return
                yield chunk

        async def _consume() -> None:
            async for _ in session.run(_audio()):
                pass

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0.05)
        await audio_queue.put(b"\x00" * 20_000)
        await stt.inject_partial("so I was thinking about")
        await asyncio.sleep(0.02)
        await audio_queue.put(b"S" * 4)
        await audio_queue.put(b"\x00" * 4)
        await asyncio.sleep(0.3)

        assert stt.commit_calls == 0  # incomplete → endpointer stays quiet

        await stt.finish()
        await audio_queue.put(None)
        await asyncio.wait_for(consumer, timeout=5.0)

    async def test_auto_mode_off_without_audio_eou(self):
        # Default heuristic detector has no audio EOU → endpointer never arms,
        # even when explicitly requested.
        stt = EndpointSTT()
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
            stt=stt,
            tts=_SilentTTS(),
            vad_endpointing=True,
        )

        async def _audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _consume() -> None:
            async for _ in session.run(_audio()):
                pass

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0.05)
        assert session._endpointer is None
        await stt.finish()
        await asyncio.wait_for(consumer, timeout=5.0)

    async def test_disabled_by_config(self):
        stt = EndpointSTT()
        detector = LocalAudioTurnDetector(audio_eou=_FixedEou(0.95))
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
            stt=stt,
            tts=_SilentTTS(),
            turn_detector=detector,
            vad_endpointing=False,
        )

        async def _audio() -> AsyncIterator[bytes]:
            return
            yield  # noqa: RET504

        async def _consume() -> None:
            async for _ in session.run(_audio()):
                pass

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0.05)
        assert session._endpointer is None
        await stt.finish()
        await asyncio.wait_for(consumer, timeout=5.0)
