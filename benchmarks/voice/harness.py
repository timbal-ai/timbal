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

import array
import asyncio
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from degrade import MicPath, active_rms, telephone_state
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
    FRAME_BYTES,
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
    resolve_turn_detector,
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
    # Detector attributes to override per run, e.g.
    # {"text_complete_hold_timeout_secs": 1.0}. Exists so a parameter sweep can
    # ask "what is the right value" instead of editing product constants and
    # re-running by hand, which is how the hold tier came to be tested at 0.35
    # and 3.0 and nowhere in between. Applied after construction, so only
    # instance attributes — class-level constants must have a matching
    # instance attribute to be reachable.
    detector_params: Mapping[str, Any] = field(default_factory=dict)
    # Provider STT knobs to override per run, merged over the defaults in
    # `stt_config`. The detector is not the only endpointer in the pipeline:
    # ElevenLabs ships `vad_silence_threshold_secs=1.2` and Nova ships
    # `endpointing=300`, and until these were sweepable the 4x asymmetry
    # between them read as a provider property rather than a setting.
    stt_extra: Mapping[str, Any] = field(default_factory=dict)
    # Fraction of the assistant's own output mixed back into the mic, standing
    # in for imperfect echo cancellation. 0.0 is every run before this one.
    aec_leak: float = 0.0
    # Noise floor and telephone band between the speaker and the STT. The default
    # is the identity — studio-clean 16kHz TTS, which is what every number in the
    # README was measured on and the main reason to be careful quoting them for
    # real callers.
    mic_path: MicPath = field(default_factory=MicPath)

    @property
    def label(self) -> str:
        suffix = self.mic_path.label
        if self.aec_leak:
            suffix += f"[leak={self.aec_leak}]"
        for params in (self.detector_params, self.stt_extra):
            if params:
                suffix += "[" + ",".join(f"{k}={v}" for k, v in sorted(params.items())) + "]"
        return f"{self.stt}/{self.detector}{suffix}"


@dataclass
class RunResult:
    scenario_id: str
    stt: str
    detector: str
    committed: list[str] = field(default_factory=list)
    # Arrival times, parallel to `committed` / `replies_spoken`. A wait for "one
    # more commit" cannot tell a commit for the speech just fed from one for
    # earlier speech that arrived late, and where a provider's commit lags the
    # audio by more than the gap between fragments, that is every pause
    # scenario it has. See `AwaitCommit` in `drive`.
    committed_at: list[float] = field(default_factory=list)
    replies_spoken: list[str] = field(default_factory=list)
    replied_at: list[float] = field(default_factory=list)
    # Last time the session said anything at all — partial, commit or reply. What
    # "the session has finished with the audio I fed it" is actually made of,
    # since neither a count nor a single timestamp can express it.
    last_event_at: float = 0.0
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
    # Arrival of the newest partial. A partial newer than every commit means the
    # provider is still transcribing something — a commit (or a deliberate
    # suppression) is pending — and a wait must not read the session as settled.
    # See `_settled` in `run_scenario`. Internal to the measurement.
    last_partial_at: float = 0.0

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


def mix_pcm16(base: bytes, leak: bytes, gain: float) -> bytes:
    """Sum two PCM16 frames with ``leak`` attenuated, clipping at int16 bounds."""
    out = array.array("h", base)
    other = array.array("h", leak)
    for i in range(min(len(out), len(other))):
        out[i] = max(-32768, min(32767, int(out[i] + other[i] * gain)))
    return out.tobytes()


class ScriptFeeder:
    """Paced PCM source: clip audio when speaking, silence the rest of the time.

    With ``leak_gain`` above zero it also mixes the assistant's own output back
    into the mic, standing in for an echo canceller that does not fully cancel.
    Every run before this fed clean user-only audio, which means
    ``_likely_stt_echo`` — the guard that stops the assistant interrupting
    itself on speaker bleed — had never once been exercised.

    ``mic_path`` adds a noise floor and the telephone band. It applies here, at the
    last moment before the frame leaves, rather than to the clips: a real mic path
    degrades *everything* it carries — the user, the silence between them, and the
    echo — and the noise has to keep running through the pauses, since noise that
    stops when the speaker does would tell the endpointer where the turn ended.
    """

    # Cap on buffered assistant audio. TTS generates faster than realtime, so an
    # uncapped buffer would drift seconds behind and leak the assistant's voice
    # into silence long after it stopped speaking — a harsher and less realistic
    # test than the thing being simulated.
    MAX_LEAK_SECS = 1.0

    def __init__(self, leak_gain: float = 0.0, mic_path: MicPath | None = None) -> None:
        self._pending: deque[bytes] = deque()
        self._leak = bytearray()
        self._leak_gain = leak_gain
        self._max_leak_bytes = int(BYTES_PER_SECOND * self.MAX_LEAK_SECS)
        self._stopped = False
        self._mic_path = mic_path or MicPath()
        # One filter state for the whole session, so the codec has no per-frame
        # transient, and one read position so the bed plays as continuous noise
        # rather than restarting every 20ms.
        self._codec_state = telephone_state()
        self._bed = b""
        self._bed_pos = 0

    def set_noise_bed(self, bed: bytes) -> None:
        self._bed = bed

    def _next_noise(self) -> bytes | None:
        if not self._bed:
            return None
        if self._bed_pos + FRAME_BYTES > len(self._bed):
            self._bed_pos = 0
        frame = self._bed[self._bed_pos : self._bed_pos + FRAME_BYTES]
        self._bed_pos += FRAME_BYTES
        return frame

    def push(self, pcm: bytes) -> None:
        self._pending.extend(frames(pcm))

    def push_leak(self, pcm: bytes) -> None:
        """Queue assistant output to bleed into the mic."""
        if self._leak_gain <= 0:
            return
        self._leak.extend(pcm)
        if len(self._leak) > self._max_leak_bytes:
            del self._leak[: len(self._leak) - self._max_leak_bytes]

    def _next_leak(self) -> bytes | None:
        if self._leak_gain <= 0 or len(self._leak) < FRAME_BYTES:
            return None
        frame = bytes(self._leak[:FRAME_BYTES])
        del self._leak[:FRAME_BYTES]
        return frame

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
            frame = self._pending.popleft() if self._pending else SILENCE_FRAME
            if (leak := self._next_leak()) is not None:
                frame = mix_pcm16(frame, leak, self._leak_gain)
            # Speech and echo first, then the room, then the line: the noise is
            # picked up by the microphone, so it goes through the codec too.
            if (noise := self._next_noise()) is not None:
                frame = mix_pcm16(frame, noise, 1.0)
            if self._mic_path.telephone:
                frame = self._mic_path.apply(frame, self._codec_state)
            yield frame
            next_at += FRAME_SECS
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))


# STT knobs `--stt-param` may vary, per backend. An allowlist because providers
# ignore unknown query params silently: a typo would sweep one value under
# several labels and every number in the table would agree, convincingly.
SWEEPABLE_STT_KEYS: dict[str, frozenset[str]] = {
    "elevenlabs": frozenset(
        {"commit_strategy", "min_speech_duration_ms", "vad_silence_threshold_secs", "vad_threshold"}
    ),
    "deepgram-nova": frozenset({"endpointing", "utterance_end_ms", "smart_format", "punctuate", "interim_results"}),
    "deepgram-flux": frozenset({"eot_timeout_ms", "eot_threshold", "eager_eot_threshold"}),
}


_HANDSHAKE_REJECTION = re.compile(r"server rejected WebSocket connection: HTTP (4\d\d)")


def config_rejection(errors: Sequence[str]) -> str | None:
    """The provider refused the configuration, as opposed to the run going badly.

    ``SWEEPABLE_STT_KEYS`` validates key *names*, so a value the provider rejects
    still runs: `eot_threshold=0.4` is outside Flux's accepted range and every
    session dies on the handshake, which the sweep then reported as a 0% row
    indistinguishable from a setting that merely merges nothing. Encoding each
    provider's accepted ranges here would go stale the first time one changed
    theirs; noticing the refusal does not.

    Only 4xx. A rejected handshake is deterministic for a given config, so
    repeating it 77 more times cannot learn anything — whereas a timeout or a 5xx
    is exactly the transient the repeats exist to average over.
    """
    for e in errors:
        if m := _HANDSHAKE_REJECTION.search(e):
            return f"provider rejected the connection (HTTP {m.group(1)}): {e}"
    return None


def coerce_param(raw: str) -> float | int | bool | str:
    """Best-effort literal from a command-line value (`0.3` -> float, `true` -> bool)."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def stt_config(stt: str, language: str, extra: Mapping[str, Any] | None = None) -> AudioInputConfig:
    """Provider defaults for a cell, with `extra` merged over them.

    The ElevenLabs values mirror ``timbal.voice.server.default_voice_config_from_env``
    rather than inventing benchmark-only ones — measuring a configuration the
    product does not ship would report dead air nobody experiences.
    """
    overrides = dict(extra or {})
    if unknown := set(overrides) - SWEEPABLE_STT_KEYS.get(stt, frozenset()):
        raise ValueError(f"{stt} has no sweepable STT param(s) {sorted(unknown)}")
    if stt.startswith("deepgram"):
        return AudioInputConfig(language=language, sample_rate=SAMPLE_RATE, extra=overrides)
    return AudioInputConfig(
        model="scribe_v2_realtime",
        language=language,
        sample_rate=SAMPLE_RATE,
        extra={
            "commit_strategy": "vad",
            "min_speech_duration_ms": 100,
            "vad_silence_threshold_secs": 1.2,
            "vad_threshold": 0.4,
            **overrides,
        },
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _build_detector(config: HarnessConfig) -> Any:
    """The detector name, or a configured instance when params are overridden.

    Rejects unknown attribute names rather than silently setting them: a typo in
    a sweep would otherwise run the default configuration under a label claiming
    it was something else, and every number in the sweep would be wrong in a way
    nothing could detect afterwards.
    """
    if not config.detector_params:
        return config.detector
    detector = resolve_turn_detector(config.detector)
    for name, value in config.detector_params.items():
        if not hasattr(detector, name):
            raise ValueError(
                f"{type(detector).__name__} has no attribute {name!r}; cannot override it for {config.label}"
            )
        setattr(detector, name, value)
    return detector


def _settle_secs(config: HarnessConfig) -> float:
    """How long the session must say nothing before a wait accepts what it has.

    Must exceed the longest HOLD the detector can arm, and this is the whole reason
    the value is derived rather than chosen. A hold emits nothing while it debounces
    — that is what it is for — so a quiescence rule shorter than the hold cannot tell
    "the session is finished" from "the session is deliberately waiting", and the
    harness tears down mid-hold. It reads as the *product* dropping a turn.

    Measured on the ElevenLabs barge-ins, where the interrupting utterance commits as
    a 1.5s `lexical_hold` against a flat 1.0s settle: 19/24 at 1.0s, 24/24 once the
    settle clears the hold. Those five were previously written off as provider
    flakiness, which is what a harness bug looks like from the outside when it only
    bites the slowest provider.

    Floors at 1.0s to cover the gap between speech ending and the first partial for
    it (~0.4s on ElevenLabs, the slowest measured here) on detectors that never hold.
    """
    detector = resolve_turn_detector(_build_detector(config))
    holds = [
        getattr(detector, name, None)
        # DEFAULT_ included because LexicalTurnDetector passes its class constant
        # straight into the decision without ever binding an instance attribute.
        for name in (
            "hold_timeout_secs",
            "DEFAULT_HOLD_TIMEOUT_SECS",
            "text_complete_hold_timeout_secs",
            "text_incomplete_hold_timeout_secs",
        )
    ]
    longest = max((h for h in holds if isinstance(h, int | float)), default=0.0)
    # The provider's own endpointer is a hold too, and the floor has to clear it:
    # ElevenLabs ships vad_silence_threshold_secs=1.2, so on a holdless detector a
    # 1.0s quiescence rule declared "settled, with something in hand" while the
    # commit for the newest speech was ~0.2s from landing — and the something in
    # hand was an *older* utterance's commit. Same reasoning as the detector
    # holds above: derived rather than chosen, because the failure mode is
    # invisible and reads as the provider dropping a turn.
    extra = stt_config(config.stt, config.language, config.stt_extra).extra
    provider_secs = max(
        float(extra.get("vad_silence_threshold_secs", 0.0)),
        float(extra.get("endpointing", 0.0)) / 1000,
        float(extra.get("utterance_end_ms", 0.0)) / 1000,
        float(extra.get("eot_timeout_ms", 0.0)) / 1000,
    )
    return max(1.0, longest + 0.5, provider_secs + 0.5)


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
        audio_input=stt_config(config.stt, config.language, config.stt_extra),
        audio_output=AudioOutputConfig(model=TTS_MODEL, voice=ASSISTANT_VOICE_ID, sample_rate=SAMPLE_RATE),
        turn_detector=_build_detector(config),
        record_audio=config.dump,
    )

    feeder = ScriptFeeder(leak_gain=config.aec_leak, mic_path=config.mic_path)
    if config.mic_path.snr_db is not None:
        # Level the noise against this scenario's own speech, so the same nominal
        # SNR means the same thing in a scenario that is mostly pause as in one
        # that is wall-to-wall talking.
        spoken = b"".join(
            clips[s.clip_key] for s in scenario.script if isinstance(s, Say) and s.clip_key in clips
        )
        feeder.set_noise_bed(config.mic_path.bed(active_rms(spoken)))
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

    settle_secs = _settle_secs(config)

    def _settled(events: list[float], after: float) -> bool:
        """Whether the session is done with the speech fed up to ``after``.

        Anything arriving after that instant is *usually* the answer, since mic
        audio is paced at realtime and the provider cannot have been sent a clip
        it has not received yet. Usually, because the converse event looks
        identical: a commit for *earlier* speech can land that late too —
        ElevenLabs commits ~1.6s after speech ends, longer than the fragment gap
        in every pause scenario — and resolving on it tears the session down
        with the newest speech's commit still in flight, the same
        phantom-content-loss bias the timed wait replaced the watermark to fix.
        In flight is observable: the provider streams partials for speech it has
        not committed yet, so a partial newer than everything in hand keeps the
        wait open until the commit lands or the partial goes stale for
        ``settle_secs`` — past that it is a stranded partial, the product's
        problem to report rather than the harness's to wait on.

        Otherwise the provider may simply have answered early — it can have all
        the audio it needs before the harness finishes handing the clip over, and
        `food_simple` commits without the final "?" for exactly that reason. Then
        nothing will ever arrive "after", so a wait for one hangs for its whole
        timeout. Falling back to a quiet session with something already in hand
        covers that, and covers a last `Say` that correctly produces nothing at
        all, which used to depend on a stale event to avoid timing out.
        """
        now = time.monotonic()
        if result.last_partial_at > max([after, *events]) and now - result.last_partial_at < settle_secs:
            return False
        if any(t > after for t in events):
            return True
        if not events:
            return False
        return now - max(after, result.last_event_at) >= settle_secs

    async def drive() -> None:
        last_say_ended = 0.0

        for step in scenario.script:
            if isinstance(step, Say):
                emit("say", f'"{step.text}"')
                feeder.push(clips[step.clip_key])
                await feeder.drain()
                # drain() returns once the last frame is pushed, so this is the
                # instant the speaker fell silent. A later part of the same
                # fluent utterance overwrites it, leaving the final speech end.
                result.speech_ended_at = time.monotonic()
                last_say_ended = result.speech_ended_at
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
                await wait_for(
                    lambda after=last_say_ended: _settled(result.committed_at, after),
                    step.timeout,
                    "commit",
                )
            elif isinstance(step, AwaitAssistantDone):
                emit("await_reply")
                await wait_for(
                    lambda after=last_say_ended: _settled(result.replied_at, after),
                    step.timeout,
                    "reply",
                )
        await asyncio.sleep(0.5)  # let trailing events land before teardown
        closing.set()
        feeder.stop()
        await session.close()

    async with aclosing(session.run(feeder.stream())) as stream:
        driver: asyncio.Task[None] | None = None
        acker: asyncio.Task[None] | None = None
        try:
            async for event in stream:
                _observe(event, result, playback, feeder, assistant_speaking, closing, emit)
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
    feeder: ScriptFeeder,
    assistant_speaking: asyncio.Event,
    closing: asyncio.Event,
    emit: Logger,
) -> None:
    if isinstance(event, AudioOutput):
        # Individual chunks are far too chatty to log; the playhead is what matters.
        playback.on_emit(len(event.data))
        feeder.push_leak(event.data)
        assistant_speaking.set()
        result.audio_chunks += 1
        result.audio_bytes += len(event.data)
    elif isinstance(event, SessionStarted):
        emit("session_started")
    elif isinstance(event, TranscriptPartial):
        result.last_event_at = result.last_partial_at = time.monotonic()
        emit("partial", f'"{event.text}"')
    elif isinstance(event, TranscriptCommitted):
        result.last_event_at = time.monotonic()
        detail = f'"{event.text}"{"  (replace)" if event.replace else ""}'
        if result.speech_ended_at is not None:
            dead_air = (time.monotonic() - result.speech_ended_at) * 1000
            result.dead_air_ms.append(dead_air)
            result.speech_ended_at = None
            detail += f"  dead air {dead_air:.0f}ms"
        emit("committed", detail)
        if event.replace and result.committed:
            result.committed[-1] = event.text
            result.committed_at[-1] = time.monotonic()
        else:
            result.committed.append(event.text)
            result.committed_at.append(time.monotonic())
    elif isinstance(event, AgentTextDone):
        result.last_event_at = time.monotonic()
        emit("agent_done", f'"{event.text}"')
        result.replies_spoken.append(event.text)
        result.replied_at.append(time.monotonic())
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
