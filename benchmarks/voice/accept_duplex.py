"""Opt-in full-duplex acceptance: scripted caller PCM in, agent PCM out.

This is not the replay harness. ``harness.py`` feeds a real STT and a real TTS
a ``TestModel`` script. Here the session is Deepgram Flux, ``openai/gpt-5.4-mini``,
and ElevenLabs, the same stack Albert retested. No microphone: caller audio is
cached ElevenLabs PCM paced at 20ms, and playback acks go back the way the
playground does.

The default pytest run does not execute it. Set ``TIMBAL_VOICE_ACCEPTANCE=1``
and have ``OPENAI_API_KEY``, ``DEEPGRAM_API_KEY``, and ``ELEVENLABS_API_KEY``.
Missing keys are named and the run fails; it does not skip a requested gate.
Analysis is PCM energy and event timing. Nothing here listens.

    TIMBAL_VOICE_ACCEPTANCE=1 uv run python benchmarks/voice/accept_duplex.py
"""

from __future__ import annotations

import array
import asyncio
import json
import math
import os
import re
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from synth import (
    ASSISTANT_VOICE_ID,
    BYTES_PER_SECOND,
    FRAME_BYTES,
    FRAME_SECS,
    SAMPLE_RATE,
    SILENCE_FRAME,
    TTS_MODEL,
    frames,
    synthesize_clips,
    write_wav,
)
from timbal import Agent
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.voice import (
    AgentTextDelta,
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    FillerConfig,
    FillerSpoken,
    SessionEnded,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    SpeculativeGeneration,
    TranscriptCommitted,
    TranscriptPartial,
    TurnMetricsEvent,
    VoiceSession,
    current_voice_session,
    resolve_stt,
)
from timbal.voice.elevenlabs import ElevenLabsStreamTTS
from timbal.voice.session import _hangup_utterance_kind

LLM_MODEL = "openai/gpt-5.4-mini"
ACCEPTANCE_FLAG = "TIMBAL_VOICE_ACCEPTANCE"
PROVIDER_KEYS = ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
RESULTS_DIR = HERE / "results" / "acceptance"

# Energy floor for "this frame is speech". Silence is ~0; a normal TTS frame
# sits in the thousands. Analysis only — not a hearing model.
SPEECH_RMS = 500.0
_FAREWELL = re.compile(r"\b(?:good\s*bye|adiós|adios|au revoir|arrivederci|adeus)\b", re.IGNORECASE)

HANGUP_PROMPT = (
    "You are a phone agent. "
    "When the caller asks you to hang up, call end_call and do not say any words. "
    "If end_call says the caller spoke again, say exactly: Sure, go ahead. Do not call any tool. "
    "If they say goodbye after that, say exactly: Goodbye. Do not call any tool."
)
ORDER_PROMPT = (
    "You are a phone agent. One short spoken sentence, no markdown. "
    "When they ask for an order and a balance together, call check_order and check_balance "
    "in the same step, say nothing until both return, then say only what they returned. "
    "When they ask to look up a ticket, call fail_lookup and then tell them it failed in one sentence."
)
STORY_PROMPT = (
    "You are a phone agent. When asked for a story, tell a long story of at least six sentences "
    "about a train. Do not call tools. No markdown."
)
TOOL_BARGE_PROMPT = (
    "You are a phone agent. When they ask about an order, call check_order and say nothing "
    "until it returns, then say only what it returned."
)


def acceptance_skip_reason() -> str | None:
    """None when the opt-in flag is set. The default CI run is not a skip of a failed gate."""
    if os.environ.get(ACCEPTANCE_FLAG) != "1":
        return f"set {ACCEPTANCE_FLAG}=1 to run the provider duplex acceptance"
    return None


def missing_provider_keys() -> list[str]:
    return [name for name in PROVIDER_KEYS if not os.environ.get(name)]


def pcm_rms(pcm: bytes) -> float:
    usable = len(pcm) - (len(pcm) % 2)
    if usable < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples))


def speech_regions(
    pcm: bytes,
    *,
    threshold: float = SPEECH_RMS,
    min_secs: float = 0.18,
    gap_secs: float = 0.35,
) -> list[dict[str, float]]:
    """Loud spans in agent PCM. A gap shorter than ``gap_secs`` stays one span.

    This is an energy detector. It does not decide what was said.
    """
    if len(pcm) < FRAME_BYTES:
        return []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for offset in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        if pcm_rms(pcm[offset : offset + FRAME_BYTES]) >= threshold:
            if start is None:
                start = offset
        elif start is not None:
            spans.append((start, offset))
            start = None
    if start is not None:
        spans.append((start, len(pcm) - (len(pcm) % 2)))
    gap_bytes = int(gap_secs * BYTES_PER_SECOND)
    min_bytes = int(min_secs * BYTES_PER_SECOND)
    merged: list[list[int]] = []
    for span_start, span_end in spans:
        if merged and span_start - merged[-1][1] <= gap_bytes:
            merged[-1][1] = span_end
        else:
            merged.append([span_start, span_end])
    regions: list[dict[str, float]] = []
    for span_start, span_end in merged:
        if span_end - span_start < min_bytes:
            continue
        chunk = pcm[span_start:span_end]
        regions.append(
            {
                "start_secs": round(span_start / BYTES_PER_SECOND, 3),
                "end_secs": round(span_end / BYTES_PER_SECOND, 3),
                "dur_secs": round((span_end - span_start) / BYTES_PER_SECOND, 3),
                "rms": round(pcm_rms(chunk), 1),
            }
        )
    return regions


def farewell_count(text: str) -> int:
    return len(_FAREWELL.findall(text))


class PlaybackSim:
    """Same wall-clock playhead the replay harness acks. Not a speaker."""

    def __init__(self) -> None:
        self._scheduled = 0
        self._playing_until = 0.0

    def on_emit(self, num_bytes: int) -> None:
        now = time.monotonic()
        self._playing_until = max(now, self._playing_until) + num_bytes / BYTES_PER_SECOND
        self._scheduled += num_bytes

    def on_interrupted(self) -> None:
        self._scheduled = self.played_bytes
        self._playing_until = time.monotonic()

    @property
    def played_bytes(self) -> int:
        remaining = max(0.0, self._playing_until - time.monotonic()) * BYTES_PER_SECOND
        return max(0, int(self._scheduled - remaining))

    @property
    def played_ms(self) -> float:
        return self.played_bytes / BYTES_PER_SECOND * 1000


class PacedFeeder:
    """20ms frames. Silence when the script has nothing to say."""

    def __init__(self) -> None:
        self._pending: deque[bytes] = deque()
        self._stopped = False

    def push(self, pcm: bytes) -> None:
        self._pending.extend(frames(pcm))

    async def drain(self) -> None:
        while self._pending and not self._stopped:
            await asyncio.sleep(FRAME_SECS)

    def stop(self) -> None:
        self._stopped = True

    async def stream(self):
        next_at = time.monotonic()
        while not self._stopped:
            frame = self._pending.popleft() if self._pending else SILENCE_FRAME
            yield frame
            next_at += FRAME_SECS
            delay = next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)


@dataclass
class CallState:
    name: str
    t0: float
    audio: bytearray = field(default_factory=bytearray)
    partials: list[dict[str, Any]] = field(default_factory=list)
    commits: list[dict[str, Any]] = field(default_factory=list)
    deltas: list[dict[str, Any]] = field(default_factory=list)
    dones: list[dict[str, Any]] = field(default_factory=list)
    fillers: list[dict[str, Any]] = field(default_factory=list)
    interrupts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    ended: bool = False
    script_finished: bool = False
    ended_at: float | None = None
    last_audio_at: float | None = None
    playhead_bytes_at_end: int | None = None
    stt_name: str = ""
    tts_name: str = ""

    def offset(self, stamp: float | None) -> float | None:
        if stamp is None:
            return None
        return round(stamp - self.t0, 3)

    def spoken(self) -> str:
        return "".join(item["text"] for item in self.deltas)

    def tool_events(self, name: str, phase: str) -> list[dict[str, Any]]:
        return [item for item in self.tool_log if item["tool"] == name and item["phase"] == phase]


def observe(event: Any, state: CallState, playback: PlaybackSim) -> None:
    now = time.monotonic()
    if isinstance(event, TranscriptPartial):
        text = event.text or ""
        state.partials.append({"text": text, "kind": _hangup_utterance_kind(text), "t": round(now - state.t0, 3)})
    elif isinstance(event, TranscriptCommitted):
        text = event.text or ""
        state.commits.append(
            {
                "text": text,
                "kind": _hangup_utterance_kind(text),
                "replace": event.replace,
                "t": round(now - state.t0, 3),
            }
        )
    elif isinstance(event, AgentTextDelta):
        state.deltas.append({"text": event.text, "t": round(now - state.t0, 3)})
    elif isinstance(event, AgentTextDone):
        state.dones.append({"text": event.text, "t": round(now - state.t0, 3)})
    elif isinstance(event, FillerSpoken):
        state.fillers.append({"text": event.text, "t": round(now - state.t0, 3)})
    elif isinstance(event, AudioOutput):
        state.audio.extend(event.data)
        state.last_audio_at = now
        playback.on_emit(len(event.data))
    elif isinstance(event, SessionInterrupted):
        playback.on_interrupted()
        state.interrupts.append({"heard_text": event.heard_text, "t": round(now - state.t0, 3)})
    elif isinstance(event, SessionError):
        state.errors.append(event.message)
    elif isinstance(event, TurnMetricsEvent):
        state.metrics.append(event.metrics.model_dump())
    elif isinstance(event, SessionEnded):
        state.ended = True
        state.ended_at = now
        state.playhead_bytes_at_end = playback.played_bytes
        # Time still on the playback schedule. Emitted-minus-played is not this:
        # an interrupt discards audio that was already handed to the transport.
        state.extra["schedule_lag_secs"] = round(max(0.0, playback._playing_until - now), 3)


async def _wait(pred: Callable[[], bool], timeout: float, what: str, state: CallState) -> bool:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            state.timeouts.append(what)
            return False
        await asyncio.sleep(0.05)
    return True


async def _say(feeder: PacedFeeder, pcm: bytes) -> None:
    feeder.push(pcm)
    await feeder.drain()


def _session(agent: Agent, *, hangup_grace: float | None) -> VoiceSession:
    return VoiceSession(
        agent=agent,
        stt=resolve_stt("deepgram-flux"),
        tts=ElevenLabsStreamTTS(),
        audio_input=AudioInputConfig(language="en", sample_rate=SAMPLE_RATE),
        audio_output=AudioOutputConfig(model=TTS_MODEL, voice=ASSISTANT_VOICE_ID, sample_rate=SAMPLE_RATE),
        turn_detector="provider",
        ignore_backchannels=True,
        entity_patience=True,
        max_tts_pause_secs=0.2,
        hangup_grace_secs=hangup_grace,
        filler=FillerConfig(enabled=True, soft_timeout_secs=0.8, message="One moment."),
        speculative_generation=SpeculativeGeneration(enabled=True, preemptive_tts=True),
        record_audio=True,
    )


def _agent(name: str, prompt: str, tools: list[Any]) -> Agent:
    return Agent(
        name=name,
        model=LLM_MODEL,
        system_prompt=prompt,
        tools=tools,
        temperature=0,
        max_tokens=180,
        # The class, not an instance. An instance is rejected at turn start.
        tracing_provider=InMemoryTracingProvider,
    )


def _hangup_tools(log: list[dict[str, Any]]) -> list[Any]:
    async def end_call() -> str:
        """Hang up only if the caller stays quiet. If this says they spoke again, keep helping."""
        session = current_voice_session()
        entered = time.monotonic()
        log.append({"tool": "end_call", "phase": "enter", "t": entered})
        if session is None:
            log.append({"tool": "end_call", "phase": "exit", "ok": False, "t": time.monotonic()})
            return "No live call. Do not hang up."
        ok = await session.commit_hangup()
        log.append({"tool": "end_call", "phase": "exit", "ok": bool(ok), "t": time.monotonic()})
        if not ok:
            return "The caller spoke again. Do not hang up. Keep helping."
        return "Ending the call."

    return [end_call]


def _order_tools(log: list[dict[str, Any]], *, order_secs: float = 4.0) -> list[Any]:
    async def check_order(order_id: str = "1001") -> str:
        """Look up an order. Takes several seconds. Status only."""
        log.append({"tool": "check_order", "phase": "enter", "order_id": order_id, "t": time.monotonic()})
        try:
            await asyncio.sleep(order_secs)
        except asyncio.CancelledError:
            log.append({"tool": "check_order", "phase": "cancelled", "t": time.monotonic()})
            raise
        log.append({"tool": "check_order", "phase": "exit", "t": time.monotonic()})
        return f"Order {order_id} is packed. Status is in transit."

    async def check_balance() -> str:
        """Return the account balance. Instant."""
        log.append({"tool": "check_balance", "phase": "enter", "t": time.monotonic()})
        log.append({"tool": "check_balance", "phase": "exit", "t": time.monotonic()})
        return "The balance is forty dollars."

    async def fail_lookup() -> str:
        """Look up a ticket. Always fails."""
        log.append({"tool": "fail_lookup", "phase": "enter", "t": time.monotonic()})
        await asyncio.sleep(1.2)
        log.append({"tool": "fail_lookup", "phase": "raise", "t": time.monotonic()})
        raise RuntimeError("lookup failed")

    return [check_order, check_balance, fail_lookup]


def _slow_tool(log: list[dict[str, Any]]) -> list[Any]:
    async def check_order(order_id: str = "1001") -> str:
        """Look up an order. Takes several seconds."""
        log.append({"tool": "check_order", "phase": "enter", "t": time.monotonic()})
        try:
            await asyncio.sleep(6.0)
        except asyncio.CancelledError:
            log.append({"tool": "check_order", "phase": "cancelled", "t": time.monotonic()})
            raise
        log.append({"tool": "check_order", "phase": "exit", "t": time.monotonic()})
        return f"Order {order_id} is packed. Status is in transit."

    return [check_order]


async def _run_call(
    name: str,
    agent: Agent,
    script: Callable[[CallState, PacedFeeder, PlaybackSim, VoiceSession], Awaitable[None]],
    *,
    tool_log: list[dict[str, Any]],
    hangup_grace: float | None,
    out_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    state = CallState(name=name, t0=time.monotonic(), tool_log=tool_log)
    feeder = PacedFeeder()
    playback = PlaybackSim()
    session = _session(agent, hangup_grace=hangup_grace)
    state.stt_name = type(session.stt).__name__
    state.tts_name = type(session.tts).__name__

    async def ack_loop() -> None:
        while True:
            await asyncio.sleep(0.25)
            session.playback.on_playback_ack(playback.played_ms)

    async def drive() -> None:
        try:
            await script(state, feeder, playback, session)
            state.script_finished = True
        finally:
            feeder.stop()
            if not session.closed:
                await session.close()

    driver: asyncio.Task[None] | None = None
    acker: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(timeout):
            async with aclosing(session.run(feeder.stream())) as stream:
                async for event in stream:
                    observe(event, state, playback)
                    if isinstance(event, SessionStarted) and driver is None:
                        acker = asyncio.create_task(ack_loop())
                        driver = asyncio.create_task(drive())
    except TimeoutError:
        state.timeouts.append(f"scenario wall {timeout:.0f}s")
    finally:
        feeder.stop()
        for task in (driver, acker):
            if task is not None and not task.done():
                task.cancel()
        for task in (driver, acker):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        if not session.closed:
            await session.close()

    pcm = bytes(state.audio)
    wav_path = out_dir / f"{name}.wav"
    write_wav(wav_path, pcm)
    regions = speech_regions(pcm)
    shortfall = None
    if state.playhead_bytes_at_end is not None:
        shortfall = round((len(pcm) - state.playhead_bytes_at_end) / BYTES_PER_SECOND, 3)
    return {
        "name": name,
        "providers": {"stt": state.stt_name, "tts": state.tts_name, "llm": LLM_MODEL},
        "execution": "live_network",
        "wall_secs": round(time.monotonic() - state.t0, 3),
        "timeouts": state.timeouts,
        "errors": state.errors,
        "partials": state.partials,
        "commits": state.commits,
        "assistant_deltas": state.deltas,
        "assistant_spoken": state.spoken(),
        "assistant_done": state.dones,
        "fillers": state.fillers,
        "interrupts": state.interrupts,
        "metrics": state.metrics,
        "tool_log": [_relativize_tool(item, state.t0) for item in state.tool_log],
        "marks": {key: round(value - state.t0, 3) for key, value in state.marks.items()},
        "extra": state.extra,
        "ended": state.ended,
        "script_finished": state.script_finished,
        "ended_t": state.offset(state.ended_at),
        "last_audio_t": state.offset(state.last_audio_at),
        "playhead_shortfall_secs": shortfall,
        "schedule_lag_secs": state.extra.get("schedule_lag_secs"),
        "audio": {
            "bytes": len(pcm),
            "secs": round(len(pcm) / BYTES_PER_SECOND, 3),
            "rms": round(pcm_rms(pcm), 1),
            "regions": regions,
            "wav": str(wav_path),
        },
    }


def _relativize_tool(item: dict[str, Any], t0: float) -> dict[str, Any]:
    out = dict(item)
    if "t" in out:
        out["t"] = round(out["t"] - t0, 3)
    return out


async def _scenario(
    name: str,
    agent: Agent,
    script: Callable[[CallState, PacedFeeder, PlaybackSim, VoiceSession], Awaitable[None]],
    *,
    tool_log: list[dict[str, Any]],
    hangup_grace: float | None,
    out_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    return await _run_call(
        name,
        agent,
        script,
        tool_log=tool_log,
        hangup_grace=hangup_grace,
        out_dir=out_dir,
        timeout=timeout,
    )


def _timeouts(report: dict[str, Any], failures: list[str]) -> None:
    if report.get("timeouts"):
        failures.append(f"timed out waiting for {report['timeouts']}")


def _providers_ok(report: dict[str, Any], failures: list[str]) -> None:
    providers = report.get("providers") or {}
    if providers.get("stt") != "DeepgramFluxSTT" or providers.get("tts") != "ElevenLabsStreamTTS":
        failures.append(f"not the live voice stack: {providers}")
    if providers.get("llm") != LLM_MODEL:
        failures.append(f"llm is {providers.get('llm')}")
    if report.get("execution") != "live_network":
        failures.append("scenario did not record live network execution")


def _hangup_audio_ok(report: dict[str, Any], failures: list[str]) -> None:
    if report.get("fillers"):
        failures.append(f"filler on hangup: {report['fillers']}")
    if report.get("errors"):
        failures.append(f"session error: {report['errors']}")
    # schedule_lag is unplayed audio still queued at teardown. The byte shortfall
    # also counts audio an earlier interrupt already discarded, so it is not the
    # disconnect check.
    lag = report.get("schedule_lag_secs")
    if lag is None or lag > 0.2:
        failures.append(f"disconnect while audio was still scheduled (lag {lag}s)")
    ended = report.get("ended_t")
    last_audio = report.get("last_audio_t")
    if ended is None or last_audio is None:
        failures.append("missing end or audio timestamp")
    elif ended - last_audio > 15:
        failures.append(f"teardown lagged last audio emit by {ended - last_audio:.1f}s")


def _check_alright(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    _timeouts(report, failures)
    _providers_ok(report, failures)
    exits = [item for item in report.get("tool_log", []) if item["tool"] == "end_call" and item["phase"] == "exit"]
    if not exits:
        failures.append("end_call did not return")
    elif exits[0].get("ok") is not True:
        failures.append(f"Alright cancelled the hangup: {exits}")
    if not report.get("ended"):
        failures.append("session stayed up after Alright")
    spoken = report.get("assistant_spoken") or ""
    if farewell_count(spoken) != 1:
        failures.append(f"farewell count {farewell_count(spoken)} in {spoken!r}")
    regions = (report.get("audio") or {}).get("regions") or []
    if len(regions) != 1:
        failures.append(f"expected one audible span, got {regions}")
    _hangup_audio_ok(report, failures)
    alright_t = (report.get("marks") or {}).get("alright_start")
    during = [item for item in report.get("partials") or [] if alright_t is not None and item["t"] >= alright_t]
    prefixes = [item["text"] for item in during if item.get("kind") == "pending"]
    report["outcomes"] = {
        "partials_during_grace": [item["text"] for item in during],
        "unfinished_ack_partials": prefixes,
        "flux_emitted_unfinished_prefix": bool(prefixes),
    }
    return failures


def _check_retract_rearm(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    _timeouts(report, failures)
    _providers_ok(report, failures)
    enters = [item for item in report.get("tool_log", []) if item["tool"] == "end_call" and item["phase"] == "enter"]
    exits = [item for item in report.get("tool_log", []) if item["tool"] == "end_call" and item["phase"] == "exit"]
    if not exits or exits[0].get("ok") is not False:
        failures.append(f"Wait did not cancel the hangup: {exits}")
    bye_t = (report.get("marks") or {}).get("bye_start")
    ended = report.get("ended_t")
    if bye_t is None:
        failures.append("bye clip was not sent")
    elif ended is not None and ended < bye_t:
        failures.append("line dropped before the later goodbye")
    if not report.get("ended"):
        failures.append("later goodbye did not drop the line")
    later = [item["text"] for item in report.get("assistant_deltas") or [] if bye_t is not None and item["t"] >= bye_t]
    later_text = "".join(later)
    # The retracted attempt may already have started "Goodbye." The later
    # goodbye is the one that has to be the only farewell after "bye".
    if farewell_count(later_text) != 1:
        failures.append(f"farewell count after bye {farewell_count(later_text)} in {later_text!r}")
    _hangup_audio_ok(report, failures)
    spoken = report.get("assistant_spoken") or ""
    report["outcomes"] = {
        "end_call_invocations": len(enters),
        "rearm_without_second_tool": len(enters) == 1 and bool(exits) and exits[0].get("ok") is False,
        "model_said_sure": "sure, go ahead" in spoken.lower(),
    }
    if len(enters) > 1:
        report["outcomes"]["note"] = "model called end_call again; this run did not isolate automatic re-arm"
    elif len(enters) == 0:
        report["outcomes"]["note"] = "model never called end_call"
    return failures


def _check_tools(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    _timeouts(report, failures)
    _providers_ok(report, failures)
    log = report.get("tool_log") or []
    order_enter = next((item for item in log if item["tool"] == "check_order" and item["phase"] == "enter"), None)
    order_exit = next((item for item in log if item["tool"] == "check_order" and item["phase"] == "exit"), None)
    balance = [item for item in log if item["tool"] == "check_balance"]
    failed = [item for item in log if item["tool"] == "fail_lookup" and item["phase"] == "raise"]
    if order_enter is None or order_exit is None:
        failures.append(f"check_order did not finish: {log}")
    else:
        took = order_exit["t"] - order_enter["t"]
        if not 3.4 <= took <= 5.5:
            failures.append(f"check_order took {took:.2f}s, expected ~4s")
        fillers = report.get("fillers") or []
        during_order = [item for item in fillers if order_enter["t"] <= item["t"] <= order_exit["t"] + 0.3]
        if len(during_order) != 1 or during_order[0].get("text") != "One moment.":
            failures.append(f"expected one 'One moment.' during check_order, got {fillers}")
        else:
            lag = during_order[0]["t"] - order_enter["t"]
            if not 0.4 <= lag <= 2.5:
                failures.append(f"filler lag {lag:.2f}s after check_order, expected ~0.8s")
        fail_enter = next((item for item in log if item["tool"] == "fail_lookup" and item["phase"] == "enter"), None)
        fail_raise = next((item for item in log if item["tool"] == "fail_lookup" and item["phase"] == "raise"), None)
        if fail_enter and fail_raise:
            # 1.2s is past the 0.8s soft timeout, so this failure gets its own line.
            during_fail = [item for item in fillers if fail_enter["t"] <= item["t"] <= fail_raise["t"] + 0.3]
            if len(during_fail) != 1:
                failures.append(f"expected one filler during fail_lookup, got {during_fail}")
    if len(balance) < 2:
        failures.append("check_balance did not run")
    spoken = (report.get("assistant_spoken") or "").lower()
    if "transit" not in spoken and "packed" not in spoken:
        failures.append(f"slow-tool result was not spoken: {report.get('assistant_spoken')!r}")
    if "forty" not in spoken:
        failures.append(f"balance was not spoken: {report.get('assistant_spoken')!r}")
    if not failed:
        failures.append("fail_lookup did not fail")
    if report.get("ended") and not report.get("script_finished"):
        failures.append("tool session closed before the script finished")
    if report.get("errors"):
        failures.append(f"session error: {report['errors']}")
    overlap = False
    if order_enter and order_exit and balance:
        balance_enter = balance[0]["t"]
        overlap = order_enter["t"] <= balance_enter <= order_exit["t"]
    report["outcomes"] = {"parallel_tools": overlap, "filler_count": len(report.get("fillers") or [])}
    return failures


def _check_barge(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    _timeouts(report, failures)
    _providers_ok(report, failures)
    if not report.get("interrupts"):
        failures.append("Stop did not interrupt playback")
    audio = report.get("audio") or {}
    if not audio.get("bytes"):
        failures.append("no agent audio to interrupt")
    growth = (report.get("extra") or {}).get("audio_growth_after_interrupt")
    if growth is None:
        failures.append("interrupt settle was not measured")
    elif growth > BYTES_PER_SECOND:
        failures.append(f"agent kept emitting {growth} bytes after the interrupt")
    if report.get("errors"):
        failures.append(f"session error: {report['errors']}")
    return failures


def _check_tool_barge(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    _timeouts(report, failures)
    _providers_ok(report, failures)
    log = report.get("tool_log") or []
    cancelled = any(item["tool"] == "check_order" and item["phase"] == "cancelled" for item in log)
    exited = any(item["tool"] == "check_order" and item["phase"] == "exit" for item in log)
    if not cancelled:
        failures.append(f"barge-in did not cancel check_order: {log}")
    if exited:
        failures.append("check_order finished after the caller interrupted it")
    spoken = (report.get("assistant_spoken") or "").lower()
    if "transit" in spoken or "packed" in spoken:
        failures.append(f"interrupted tool result was spoken: {report.get('assistant_spoken')!r}")
    if not report.get("interrupts"):
        failures.append("no interrupt event")
    if report.get("ended") and not report.get("script_finished"):
        failures.append("barge-in closed the session")
    return failures


async def _script_alright(
    clips: dict[str, bytes], state: CallState, feeder: PacedFeeder, playback: PlaybackSim, session: VoiceSession
) -> None:
    del session
    await _say(feeder, clips["Please hang up now."])
    await _wait(lambda: bool(state.tool_events("end_call", "enter")), 25, "end_call enter", state)
    await _wait(lambda: len(state.audio) > 1600, 12, "farewell audio", state)
    await _wait(lambda: playback._playing_until - time.monotonic() < 0.35, 8, "farewell drain", state)
    state.marks["alright_start"] = time.monotonic()
    await _say(feeder, clips["Alright."])
    await _wait(lambda: bool(state.tool_events("end_call", "exit")), 10, "end_call return", state)
    await _wait(lambda: state.ended, 8, "session end", state)


async def _script_retract(
    clips: dict[str, bytes], state: CallState, feeder: PacedFeeder, playback: PlaybackSim, session: VoiceSession
) -> None:
    del playback, session
    await _say(feeder, clips["Please hang up now."])
    await _wait(lambda: bool(state.tool_events("end_call", "enter")), 25, "end_call enter", state)
    state.marks["wait_start"] = time.monotonic()
    await _say(feeder, clips["Wait, one more thing."])
    await _wait(lambda: bool(state.tool_events("end_call", "exit")), 12, "end_call return", state)
    await _wait(lambda: any("sure" in item["text"].lower() for item in state.dones), 15, "retract reply", state)
    await asyncio.sleep(0.4)
    state.marks["bye_start"] = time.monotonic()
    await _say(feeder, clips["Bye bye."])
    await _wait(lambda: state.ended, 20, "rearm disconnect", state)


async def _script_tools(
    clips: dict[str, bytes], state: CallState, feeder: PacedFeeder, playback: PlaybackSim, session: VoiceSession
) -> None:
    del playback, session
    await _say(feeder, clips["Please check the order and my balance together."])
    await _wait(lambda: bool(state.tool_events("check_order", "exit")), 20, "check_order done", state)
    await _wait(lambda: bool(state.dones), 15, "tool reply", state)
    await asyncio.sleep(0.6)
    replies_before = len(state.dones)
    await _say(feeder, clips["Look up the ticket."])
    await _wait(lambda: len(state.dones) > replies_before, 15, "failure reply", state)
    await asyncio.sleep(0.4)


async def _script_barge(
    clips: dict[str, bytes], state: CallState, feeder: PacedFeeder, playback: PlaybackSim, session: VoiceSession
) -> None:
    del session
    await _say(feeder, clips["Tell me a long story about a train."])
    await _wait(lambda: len(state.audio) > 8000, 25, "story audio", state)
    start_ms = playback.played_ms
    await _wait(lambda: playback.played_ms - start_ms >= 600, 15, "600ms of playback", state)
    state.marks["stop_start"] = time.monotonic()
    await _say(feeder, clips["Stop."])
    await _wait(lambda: bool(state.interrupts), 10, "interrupt", state)
    await asyncio.sleep(0.6)
    frozen = len(state.audio)
    await asyncio.sleep(1.0)
    state.extra["audio_growth_after_interrupt"] = len(state.audio) - frozen


async def _script_tool_barge(
    clips: dict[str, bytes], state: CallState, feeder: PacedFeeder, playback: PlaybackSim, session: VoiceSession
) -> None:
    del playback, session
    await _say(feeder, clips["What is the status of order one thousand one?"])
    await _wait(lambda: bool(state.tool_events("check_order", "enter")), 20, "check_order enter", state)
    await asyncio.sleep(0.4)
    state.marks["barge_start"] = time.monotonic()
    await _say(feeder, clips["Stop, never mind."])
    await _wait(
        lambda: bool(state.interrupts) or bool(state.tool_events("check_order", "cancelled")), 8, "tool cancel", state
    )
    await asyncio.sleep(0.8)


def _attach(report: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    report["failures"] = failures
    report["passed"] = not failures
    return report


async def run_acceptance(out_dir: Path | None = None) -> dict[str, Any]:
    """Run the bounded live scenarios. Raises if the flag is on and a key is missing."""
    missing = missing_provider_keys()
    if missing:
        raise RuntimeError(f"{ACCEPTANCE_FLAG}=1 but missing {', '.join(missing)}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir or (RESULTS_DIR / stamp)
    dest.mkdir(parents=True, exist_ok=True)
    clips = await synthesize_clips(
        [
            "Please hang up now.",
            "Alright.",
            "Wait, one more thing.",
            "Bye bye.",
            "Please check the order and my balance together.",
            "Look up the ticket.",
            "Tell me a long story about a train.",
            "Stop.",
            "What is the status of order one thousand one?",
            "Stop, never mind.",
        ]
    )

    hangup_log: list[dict[str, Any]] = []
    retract_log: list[dict[str, Any]] = []
    order_log: list[dict[str, Any]] = []
    barge_tool_log: list[dict[str, Any]] = []

    alright = await _scenario(
        "hangup_alright",
        _agent("accept_hangup", HANGUP_PROMPT, _hangup_tools(hangup_log)),
        lambda state, feeder, playback, session: _script_alright(clips, state, feeder, playback, session),
        tool_log=hangup_log,
        hangup_grace=1.2,
        out_dir=dest,
        timeout=50,
    )
    _attach(alright, _check_alright(alright))

    retract = await _scenario(
        "hangup_wait_then_bye",
        _agent("accept_retract", HANGUP_PROMPT, _hangup_tools(retract_log)),
        lambda state, feeder, playback, session: _script_retract(clips, state, feeder, playback, session),
        tool_log=retract_log,
        hangup_grace=1.2,
        out_dir=dest,
        timeout=55,
    )
    _attach(retract, _check_retract_rearm(retract))

    tools = await _scenario(
        "tools_slow_parallel_fail",
        _agent("accept_tools", ORDER_PROMPT, _order_tools(order_log)),
        lambda state, feeder, playback, session: _script_tools(clips, state, feeder, playback, session),
        tool_log=order_log,
        hangup_grace=1.2,
        out_dir=dest,
        timeout=55,
    )
    _attach(tools, _check_tools(tools))

    barge = await _scenario(
        "barge_in",
        _agent("accept_story", STORY_PROMPT, []),
        lambda state, feeder, playback, session: _script_barge(clips, state, feeder, playback, session),
        tool_log=[],
        hangup_grace=None,
        out_dir=dest,
        timeout=45,
    )
    _attach(barge, _check_barge(barge))

    tool_barge = await _scenario(
        "tool_interrupted",
        _agent("accept_tool_barge", TOOL_BARGE_PROMPT, _slow_tool(barge_tool_log)),
        lambda state, feeder, playback, session: _script_tool_barge(clips, state, feeder, playback, session),
        tool_log=barge_tool_log,
        hangup_grace=None,
        out_dir=dest,
        timeout=40,
    )
    _attach(tool_barge, _check_tool_barge(tool_barge))

    scenarios = [alright, retract, tools, barge, tool_barge]
    report = {
        "generated_at": stamp,
        "flag": ACCEPTANCE_FLAG,
        "listening": False,
        "analysis": "pcm rms speech regions and session events; not human listening",
        "stack": "deepgram-flux + openai/gpt-5.4-mini + elevenlabs",
        "distinct_from": "benchmarks/voice/harness.py uses TestModel, not this LLM",
        "scenarios": scenarios,
        "passed": all(item["passed"] for item in scenarios),
        "failures": {item["name"]: item["failures"] for item in scenarios if item["failures"]},
    }
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    reason = acceptance_skip_reason()
    if reason:
        print(json.dumps({"passed": None, "skipped": reason}))  # noqa: T201
        return 0
    missing = missing_provider_keys()
    if missing:
        print(json.dumps({"passed": False, "missing_keys": missing}))  # noqa: T201
        return 2
    report = asyncio.run(run_acceptance())
    summary = {
        "passed": report["passed"],
        "failures": report["failures"],
        "scenarios": [
            {
                "name": item["name"],
                "passed": item["passed"],
                "failures": item["failures"],
                "outcomes": item.get("outcomes"),
                "wav": (item.get("audio") or {}).get("wav"),
            }
            for item in report["scenarios"]
        ],
    }
    print(json.dumps(summary, indent=2))  # noqa: T201
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
