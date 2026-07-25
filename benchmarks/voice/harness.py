"""Replay harness: drives a real ``VoiceSession`` from a scripted scenario.

The harness is a **fake browser**. No session changes are needed because the
three seams already exist:

===============  ==========================================  ==================
Seam             Session API                                 Harness role
===============  ==========================================  ==================
audio in         ``session.run(audio_in)``                   paced PCM feeder
events out       yields ``VoiceSessionEvent``                script driver
playback         ``session.playback.on_playback_ack(ms)``    ack pump
===============  ==========================================  ==================

The ack pump is not optional: without acks ``playback_acks_received`` is False
and interruption truncation runs on the wall-clock estimate, which is a
*different code path* than production.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field

from scenario import (
    AwaitAssistantAudio,
    AwaitAssistantDone,
    AwaitCommit,
    Say,
    Scenario,
    Silence,
)
from synth import (
    ASSISTANT_VOICE_ID,
    BYTES_PER_SECOND,
    FRAME_SECS,
    HERE,
    SAMPLE_RATE,
    SILENCE_FRAME,
    TTS_MODEL,
    frames,
    write_wav,
)
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.voice import (
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    TranscriptCommitted,
    TranscriptPartial,
    TurnMetrics,
    TurnMetricsEvent,
    VoiceSession,
    VoiceSessionEvent,
    resolve_stt,
)
from timbal.voice.elevenlabs import ElevenLabsStreamTTS

DUMP_DIR = HERE / "results" / "dumps"

Logger = Callable[[str, str], None]


@dataclass(frozen=True)
class HarnessConfig:
    stt: str = "deepgram-flux"
    detector: str = "provider"
    language: str = "en"
    dump: bool = False

    @property
    def label(self) -> str:
        return f"{self.stt}/{self.detector}"


@dataclass
class RunResult:
    scenario_id: str
    stt: str
    detector: str
    committed: list[str] = field(default_factory=list)
    replies_spoken: list[str] = field(default_factory=list)
    interrupted: bool = False
    heard_text: str | None = None
    latencies_ms: list[float] = field(default_factory=list)
    metrics: list[TurnMetrics] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    audio_chunks: int = 0
    audio_bytes: int = 0
    wall_secs: float = 0.0
    failures: list[str] = field(default_factory=list)
    # Wall time from the user falling silent to the turn being accepted — the
    # dead air a caller actually sits through, and the half of the picture
    # `latencies_ms` cannot see. `eou→first audio` starts at the *accepted*
    # commit, so every second a hold spends deciding is invisible to it. That
    # blindness let a tier-removal experiment read as 11 fixes and 0
    # regressions while adding 2.6s to six barge-in cells.
    dead_air_ms: list[float] = field(default_factory=list)
    # Monotonic timestamp of the most recent speech end, consumed by the next
    # commit. Not an output; internal to the measurement.
    speech_ended_at: float | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# Fake browser internals
# ---------------------------------------------------------------------------


class PlaybackSim:
    """Playhead over a gapless client-side queue, acked back at browser cadence.

    Caveat: this re-derives the same wall-clock schedule
    ``BufferedPlaybackTracker`` computes internally, so it validates the *ack code
    path* (``ack_received`` true, extrapolation branch live) rather than the
    accuracy of the estimate. Real accuracy needs real playback hardware.
    """

    def __init__(self, bytes_per_second: int = BYTES_PER_SECOND) -> None:
        self._bps = bytes_per_second
        self._scheduled = 0
        self._playing_until = 0.0

    def on_emit(self, num_bytes: int) -> None:
        now = time.monotonic()
        self._playing_until = max(now, self._playing_until) + num_bytes / self._bps
        self._scheduled += num_bytes

    def on_interrupted(self) -> None:
        self._scheduled = self.played_bytes
        self._playing_until = time.monotonic()

    @property
    def played_bytes(self) -> int:
        remaining = max(0.0, self._playing_until - time.monotonic()) * self._bps
        return max(0, int(self._scheduled - remaining))

    @property
    def played_ms(self) -> float:
        return self.played_bytes / self._bps * 1000


class ScriptFeeder:
    """Paced PCM source: clip audio when speaking, silence the rest of the time."""

    def __init__(self) -> None:
        self._pending: deque[bytes] = deque()
        self._stopped = False

    def push(self, pcm: bytes) -> None:
        self._pending.extend(frames(pcm))

    async def drain(self) -> None:
        while self._pending:
            await asyncio.sleep(FRAME_SECS)

    def stop(self) -> None:
        self._stopped = True

    async def stream(self) -> AsyncIterator[bytes]:
        """20ms frames, continuously.

        The stream never gaps: STT sockets and Silero cannot tell a pause in the
        script from a stalled connection. Pacing follows an absolute schedule
        because ``sleep(FRAME_SECS)`` per iteration accumulates drift and slowly
        desynchronizes the replay from real time.
        """
        next_at = time.monotonic()
        while not self._stopped:
            yield self._pending.popleft() if self._pending else SILENCE_FRAME
            next_at += FRAME_SECS
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))


def stt_config(stt: str, language: str) -> AudioInputConfig:
    if stt.startswith("deepgram"):
        return AudioInputConfig(language=language, sample_rate=SAMPLE_RATE)
    return AudioInputConfig(
        model="scribe_v2_realtime",
        language=language,
        sample_rate=SAMPLE_RATE,
        extra={
            "commit_strategy": "vad",
            "min_speech_duration_ms": 100,
            "vad_silence_threshold_secs": 1.2,
            "vad_threshold": 0.4,
        },
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_scenario(
    scenario: Scenario,
    clips: dict[str, bytes],
    config: HarnessConfig,
    *,
    log: Logger | None = None,
) -> RunResult:
    """Replay one scenario against a live session and check its expectations."""
    result = RunResult(scenario_id=scenario.id, stt=config.stt, detector=config.detector)

    # Tracing must stay on: the session chains parent_id across turns for memory,
    # and disabling it makes every turn after the first log "Parent trace not
    # found". Pinned to in-memory so a platform config in the environment can't
    # drag the bench into a real tracing backend.
    agent = Agent(
        name="bench",
        model=TestModel(responses=scenario.replies),
        tools=[],
        tracing_provider=InMemoryTracingProvider,
    )
    session = VoiceSession(
        agent=agent,
        stt=resolve_stt(config.stt),
        tts=ElevenLabsStreamTTS(),
        audio_input=stt_config(config.stt, config.language),
        audio_output=AudioOutputConfig(model=TTS_MODEL, voice=ASSISTANT_VOICE_ID, sample_rate=SAMPLE_RATE),
        turn_detector=config.detector,
        record_audio=config.dump,
    )

    feeder = ScriptFeeder()
    playback = PlaybackSim()
    assistant_speaking = asyncio.Event()
    # session.close() interrupts any reply still playing. That teardown interrupt
    # is not a barge-in and must not be observed as one.
    closing = asyncio.Event()
    started = time.monotonic()

    def emit(kind: str, detail: str = "") -> None:
        if log is not None:
            log(kind, detail)

    async def ack_loop() -> None:
        """Report the playhead the way the playground does — every 250ms."""
        while True:
            await asyncio.sleep(0.25)
            session.playback.on_playback_ack(playback.played_ms)

    async def wait_for(predicate, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                result.errors.append(f"timed out waiting for {what}")
                emit("timeout", what)
                return
            await asyncio.sleep(FRAME_SECS)

    async def drive() -> None:
        # Baselines are taken when speech *starts*, not when the wait step is
        # reached: a fast commit can land while the clip is still draining, and a
        # wait armed after the fact would sit there until it timed out.
        commits_at_say = 0
        replies_at_say = 0

        for step in scenario.script:
            if isinstance(step, Say):
                emit("say", f'"{step.text}"')
                commits_at_say = len(result.committed)
                replies_at_say = len(result.replies_spoken)
                feeder.push(clips[step.clip_key])
                await feeder.drain()
                # drain() returns once the last frame is pushed, so this is the
                # instant the speaker fell silent. A later part of the same
                # fluent utterance overwrites it, leaving the final speech end.
                result.speech_ended_at = time.monotonic()
            elif isinstance(step, Silence):
                emit("silence", f"{step.secs:.1f}s")
                await asyncio.sleep(step.secs)
            elif isinstance(step, AwaitAssistantAudio):
                emit("await_audio", f"{step.offset_ms:.0f}ms into reply")
                await wait_for(assistant_speaking.is_set, step.timeout, "assistant audio")
                start = playback.played_ms
                await wait_for(
                    lambda start=start, offset=step.offset_ms: playback.played_ms - start >= offset,
                    step.timeout,
                    f"{step.offset_ms:.0f}ms of playback",
                )
            elif isinstance(step, AwaitCommit):
                emit("await_commit")
                await wait_for(lambda base=commits_at_say: len(result.committed) > base, step.timeout, "commit")
            elif isinstance(step, AwaitAssistantDone):
                emit("await_reply")
                await wait_for(lambda base=replies_at_say: len(result.replies_spoken) > base, step.timeout, "reply")
        await asyncio.sleep(0.5)  # let trailing events land before teardown
        closing.set()
        feeder.stop()
        await session.close()

    async with aclosing(session.run(feeder.stream())) as stream:
        driver: asyncio.Task[None] | None = None
        acker: asyncio.Task[None] | None = None
        try:
            async for event in stream:
                _observe(event, result, playback, assistant_speaking, closing, emit)
                if isinstance(event, SessionStarted):
                    started = time.monotonic()
                    acker = asyncio.create_task(ack_loop())
                    driver = asyncio.create_task(drive())
        finally:
            for task in (driver, acker):
                if task is not None and not task.done():
                    task.cancel()
            for task in (driver, acker):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)

    result.wall_secs = time.monotonic() - started

    if config.dump:
        for label, pcm in (("in", session.input_audio), ("out", session.output_audio)):
            if pcm:
                path = DUMP_DIR / f"{scenario.id}-{config.stt}-{config.detector}-{label}.wav"
                write_wav(path, pcm)
                emit("dump", str(path))

    result.failures = [
        message
        for expectation in scenario.expectations_for(config.detector, config.stt)
        if (message := expectation.check(result, scenario)) is not None
    ]
    return result


def _observe(
    event: VoiceSessionEvent,
    result: RunResult,
    playback: PlaybackSim,
    assistant_speaking: asyncio.Event,
    closing: asyncio.Event,
    emit: Logger,
) -> None:
    if isinstance(event, AudioOutput):
        # Individual chunks are far too chatty to log; the playhead is what matters.
        playback.on_emit(len(event.data))
        assistant_speaking.set()
        result.audio_chunks += 1
        result.audio_bytes += len(event.data)
    elif isinstance(event, SessionStarted):
        emit("session_started")
    elif isinstance(event, TranscriptPartial):
        emit("partial", f'"{event.text}"')
    elif isinstance(event, TranscriptCommitted):
        detail = f'"{event.text}"{"  (replace)" if event.replace else ""}'
        if result.speech_ended_at is not None:
            dead_air = (time.monotonic() - result.speech_ended_at) * 1000
            result.dead_air_ms.append(dead_air)
            result.speech_ended_at = None
            detail += f"  dead air {dead_air:.0f}ms"
        emit("committed", detail)
        if event.replace and result.committed:
            result.committed[-1] = event.text
        else:
            result.committed.append(event.text)
    elif isinstance(event, AgentTextDone):
        emit("agent_done", f'"{event.text}"')
        result.replies_spoken.append(event.text)
        assistant_speaking.clear()
    elif isinstance(event, SessionInterrupted):
        playback.on_interrupted()
        if closing.is_set():
            emit("teardown", "interrupt from close(), not counted")
        else:
            result.interrupted = True
            result.heard_text = event.heard_text
            emit("interrupted", f"heard: {event.heard_text!r}")
    elif isinstance(event, TurnMetricsEvent):
        metrics = event.metrics
        result.metrics.append(metrics)
        if metrics.eou_to_first_audio_ms is not None:
            result.latencies_ms.append(metrics.eou_to_first_audio_ms)
        emit(
            "metrics",
            f"eou→audio {metrics.eou_to_first_audio_ms}ms  segments {metrics.tts_segments}  "
            f"acks {metrics.playback_acks_received}  vad_eou {metrics.vad_endpointed}",
        )
    elif isinstance(event, SessionError):
        result.errors.append(event.message)
        emit("error", event.message)
