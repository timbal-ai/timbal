"""How many concurrent voice sessions can one process carry?

This exists to replace the guess in ``timbal/server/capacity.py``::

    _PER_CPU = 2.0  # "a starting point, not a measurement"

**What actually binds.** A voice session is soft real time. Mic audio arrives
every 20ms and has to be turned around before the next frame; TTS has to be
handed to the transport before the playout buffer drains. So the ceiling is not
"when does the box hit 100% CPU", it is "when does the event loop stop being
punctual". Those are different numbers, and the second is always the smaller
one, because a loop at 80% utilization already has a fat tail.

Concretely, per session, *inline on the event loop*:

* Silero VAD — one ONNX inference per 32ms of audio (``voice/vad.py``:
  ``FRAME_SAMPLES=512``, ``intra_op_num_threads=1``), synchronous in
  ``VadEndpointer.push``.
* ``CallRecorder.add_mic`` — numpy mix + MP3 encode, synchronous, when
  recording is on (``voice/recording.py``: "call from the session's event loop
  only").
* Per-chunk bookkeeping: detector buffer append, STT socket write.

And *off* the loop, in the default executor: Smart Turn (~50-100ms) and Namo,
once per utterance boundary rather than per frame.

That shape is why one scalar is suspect. The inline work scales with the *event
loop*, and a worker has exactly one — so handing a single-worker process four
cores does not let it do 4x the VAD. The executor work does scale with cores.
This script therefore reports two ceilings and takes the lower:

* ``sessions/loop`` — ramp until p99 loop lag exceeds the frame budget.
* ``sessions/cpu``  — measured CPU-seconds per session-second, inverted.

**On synthetic audio.** The mic signal is generated, not recorded speech, and
STT/TTS/LLM are local mocks. That is deliberate and it is *sound for CPU*:
Silero's cost is a fixed graph over a fixed 512-sample frame, so it does not
care whether the samples are speech. It would be unsound for accuracy — this
script says nothing about turn-taking quality. Use ``cli.py`` for that.

**What the mocks leave out**, and therefore what this number is optimistic
about: a real STT/TTS pair adds a websocket per session with per-message JSON
parsing and, for ElevenLabs, base64 audio decode — all on this same loop. Read
the result as an upper bound on sessions and re-measure against real providers
before spending the headroom.

Usage::

    uv run python benchmarks/voice/bench_cpu.py --quick
    uv run python benchmarks/voice/bench_cpu.py --detector local --recording
    uv run python benchmarks/voice/bench_cpu.py --sessions 8   # one data point

The honest number needs a cgroup, because ``auto`` sizes from the quota::

    docker run --rm --cpus=1 -v "$PWD:/w" -w /w python:3.11 \
        sh -c 'pip install -q uv && uv run python benchmarks/voice/bench_cpu.py'
"""

# The STT/TTS mocks below implement provider ABCs, so they take arguments they
# have no use for (`config`, `text`) — same as the fakes in python/tests.
# ruff: noqa: ARG002
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import math
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

os.environ.setdefault("TIMBAL_LOG_LEVEL", "ERROR")

from timbal import Agent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.logs import setup_logging  # noqa: E402
from timbal.server.capacity import available_cpus  # noqa: E402
from timbal.voice import (  # noqa: E402
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
    VoiceSession,
)
from timbal.voice.metrics import TurnMetricsEvent  # noqa: E402

SAMPLE_RATE = 16_000
FRAME_SECS = 0.02
FRAME_BYTES = int(SAMPLE_RATE * 2 * FRAME_SECS)

# The audio frame period is the deadline that matters: a loop late by more than
# this has already missed a frame, whatever the CPU meter says.
DEFAULT_LAG_BUDGET_MS = 20.0
# Some lateness is unavoidable (GC, the ramp's own bookkeeping). Fail a rung
# only when frames are *persistently* late.
DEFAULT_LATE_FRAME_PCT = 1.0


# ---------------------------------------------------------------------------
# Synthetic mic audio
# ---------------------------------------------------------------------------


def speech_like(secs: float, *, seed: int = 0) -> bytes:
    """Amplitude-modulated formant-ish noise: silence, then bursts, then silence.

    Shaped so Silero sees real onsets and offsets (and therefore so the
    endpointer's speech-stop → score → commit path actually fires), without
    needing a network round trip to a TTS provider. The *cost* of a VAD frame
    is input-independent; only the decisions are fake.
    """
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * secs)
    t = np.arange(n) / SAMPLE_RATE
    # Three formants under a 4Hz syllable envelope, plus a little breath noise.
    voiced = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((120, 0.5), (440, 0.3), (1800, 0.12)))
    syllables = (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)) ** 2
    signal = voiced * syllables + rng.normal(0, 0.01, n)
    # Utterance gating: ~1.6s of speech, ~0.9s of silence.
    period = 2.5
    phase = (t % period) / period
    signal *= np.where(phase < 0.64, 1.0, 0.0)
    return (np.clip(signal, -1, 1) * 12000).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Mocks: no network, so the measurement is the session's own CPU
# ---------------------------------------------------------------------------


class BenchSTT(SpeechToText):
    """Stays open for the whole run and commits a transcript every ``every``s.

    Unlike the unit-test mocks it must not exhaust its event stream — that would
    make ``VoiceSession`` close itself and end the session mid-measurement.
    """

    def __init__(self, *, every: float = 2.5, text: str = "what is the status of my order") -> None:
        self._every = every
        self._text = text
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.pushed_bytes = 0

    async def connect(self, config: AudioInputConfig) -> None:
        self._task = asyncio.create_task(self._script())

    async def _script(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._every * 0.6)
                await self._queue.put(TranscriptEvent(type="partial", text=self._text[:12]))
                await asyncio.sleep(self._every * 0.4)
                await self._queue.put(TranscriptEvent(type="committed", text=self._text))
        except asyncio.CancelledError:
            pass

    async def push_audio(self, chunk: bytes) -> None:
        self.pushed_bytes += len(chunk)

    async def commit(self) -> None:
        pass

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._queue.put(None)


class BenchTTS(TextToSpeech):
    """Emits real-time-paced PCM, so downlink bookkeeping costs what it costs."""

    def __init__(self, *, secs_per_reply: float = 2.0) -> None:
        self._secs = secs_per_reply

    async def connect(self, config: AudioOutputConfig) -> None:
        pass

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        chunk = b"\x00\x01" * (FRAME_BYTES // 2)
        for _ in range(int(self._secs / FRAME_SECS)):
            await asyncio.sleep(FRAME_SECS)
            yield chunk

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


class LoopLagMonitor:
    """Sample how late the loop is running.

    A task that asks for ``interval`` and gets ``interval + d`` back has been
    kept waiting ``d`` by whatever else was on the loop. That delay *is* the
    audio jitter a caller hears, which is why it, and not CPU%, is the ceiling.
    """

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.lags_ms: list[float] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        try:
            while True:
                t0 = time.perf_counter()
                await asyncio.sleep(self.interval)
                self.lags_ms.append((time.perf_counter() - t0 - self.interval) * 1000)
        except asyncio.CancelledError:
            pass

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q / 100 * len(ordered))) - 1))
    return ordered[idx]


@dataclass
class Rung:
    sessions: int
    wall_secs: float
    cpu_secs: float
    loop_lag_p50: float
    loop_lag_p99: float
    loop_lag_max: float
    late_frames_pct: float
    frame_late_p99: float
    turns: int
    first_audio_p95: float | None
    llm_first_token_p95: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def cores_used(self) -> float:
        return self.cpu_secs / self.wall_secs if self.wall_secs else 0.0

    @property
    def cores_per_session(self) -> float:
        return self.cores_used / self.sessions if self.sessions else 0.0

    def healthy(
        self,
        lag_budget_ms: float,
        late_pct: float,
        *,
        latency_budget_ms: float | None = None,
    ) -> bool:
        """Punctual *and* still answering promptly.

        Loop lag alone is too lenient: a run can hold p99 lag under the frame
        budget while the tail (a 100ms+ stall) pushes the time from
        end-of-utterance to first audio out by an order of magnitude. Since
        that delay is the whole reason a caller notices an overloaded box, it
        is a first-class failure and not a footnote.
        """
        if self.errors or self.loop_lag_p99 > lag_budget_ms or self.late_frames_pct > late_pct:
            return False
        if (
            latency_budget_ms is not None
            and self.first_audio_p95 is not None
            and self.first_audio_p95 > latency_budget_ms
        ):
            return False
        return True


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------


def build_session(args: argparse.Namespace, idx: int, recording_dir: Path | None):
    agent = Agent(
        name=f"bench_{idx}",
        model=TestModel(responses=["Your order shipped this morning and arrives Thursday."]),
        tools=[],
    )
    recorder = None
    if recording_dir is not None:
        from timbal.voice.recording import CallRecorder

        recorder = CallRecorder(
            path=recording_dir / f"s{idx}.mp3",
            sample_rate=SAMPLE_RATE,
        )
    return VoiceSession(
        agent=agent,
        stt=BenchSTT(every=args.turn_period),
        tts=BenchTTS(secs_per_reply=args.reply_secs),
        audio_input=AudioInputConfig(sample_rate=SAMPLE_RATE),
        audio_output=AudioOutputConfig(sample_rate=SAMPLE_RATE),
        turn_detector=args.detector,
        # True resolves the real Silero endpointer — the inline per-32ms cost
        # this whole script exists to price.
        vad_endpointing=not args.no_vad,
        recorder=recorder,
        session_id=f"bench-{idx}",
    )


@dataclass
class Window:
    """Whether we are inside the steady-state sampling window.

    Session start (provider connect, lazy model load, greeting) and teardown
    (close, recorder finalize) are one-time transients that scale with N and
    have nothing to do with how many calls a box can *hold*. Left in the
    sample they produce exactly the wrong shape of result: non-monotonic lag
    maxima and a "cliff" that is really 16 sessions booting at once.
    """

    active: bool = False
    late_frames: list[float] = field(default_factory=list)
    turn_latencies: list[float] = field(default_factory=list)
    # Split the turn so a regression is attributable: time to the first LLM
    # token is almost pure scheduling (TestModel answers instantly), so if that
    # is what grows, the loop is late getting the turn *started* rather than
    # slow doing the work. Anything left over is the TTS and emit path.
    llm_first_token: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    frames_fed: int = 0


class SessionRunner:
    """Drives one session; the caller owns the clock."""

    def __init__(self, session, pcm: bytes, window: Window) -> None:
        self.session = session
        self._pcm = pcm
        self._w = window
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _mic(self) -> AsyncIterator[bytes]:
        pos = 0
        # Absolute deadlines: `sleep(FRAME_SECS)` per iteration accumulates
        # drift and would quietly under-feed instead of reporting lateness.
        next_at = time.perf_counter()
        while not self._stop.is_set():
            if pos + FRAME_BYTES > len(self._pcm):
                pos = 0
            frame = self._pcm[pos : pos + FRAME_BYTES]
            pos += FRAME_BYTES
            next_at += FRAME_SECS
            delay = next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            elif self._w.active:
                # The loop owed us this frame already; record the debt.
                self._w.late_frames.append(-delay * 1000)
                next_at = time.perf_counter()
            else:
                next_at = time.perf_counter()
            if self._w.active:
                self._w.frames_fed += 1
            yield frame
        yield b""

    async def _consume(self) -> None:
        try:
            async for event in self.session.run(self._mic()):
                if isinstance(event, TurnMetricsEvent):
                    if self._w.active:
                        ms = event.metrics.eou_to_first_audio_ms
                        if ms is not None:
                            self._w.turn_latencies.append(ms)
                        llm = event.metrics.eou_to_llm_first_token_ms
                        if llm is not None:
                            self._w.llm_first_token.append(llm)
                elif isinstance(event, AudioOutput):
                    self.session.playback.on_audio_emitted(len(event.data))
        except Exception as e:  # noqa: BLE001 — a crashed session invalidates the rung
            self._w.errors.append(f"{type(e).__name__}: {e}")

    def start(self) -> None:
        self._task = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            await self.session.close()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5.0)


# ---------------------------------------------------------------------------
# One rung of the ramp
# ---------------------------------------------------------------------------


async def measure(args: argparse.Namespace, sessions: int, pcm: bytes) -> Rung:
    recording_dir = None
    if args.recording:
        recording_dir = Path(args.recording_dir or "/tmp/timbal-bench-cpu")
        recording_dir.mkdir(parents=True, exist_ok=True)

    # Warm up on a throwaway session: lazy model loads, first-inference
    # allocations and HF cache lookups are one-time process costs, not per
    # session steady state.
    warm_window = Window(active=False)
    warm = SessionRunner(build_session(args, -1, None), pcm, warm_window)
    warm.start()
    await asyncio.sleep(args.warmup)
    await warm.stop()

    window = Window()
    runners = [SessionRunner(build_session(args, i, recording_dir), pcm, window) for i in range(sessions)]
    for r in runners:
        r.start()

    # Let every session get through connect + greeting + a first turn before
    # anything counts.
    await asyncio.sleep(args.settle)

    monitor = LoopLagMonitor()
    window.active = True
    monitor.start()
    cpu0, wall0 = time.process_time(), time.perf_counter()
    await asyncio.sleep(args.duration)
    cpu_secs = time.process_time() - cpu0
    wall_secs = time.perf_counter() - wall0
    await monitor.stop()
    window.active = False

    # Teardown is outside the window on purpose — see Window.
    for r in runners:
        await r.stop()

    expected_frames = window.frames_fed or sessions * (args.duration / FRAME_SECS)
    late = window.late_frames
    return Rung(
        sessions=sessions,
        wall_secs=wall_secs,
        cpu_secs=cpu_secs,
        loop_lag_p50=pct(monitor.lags_ms, 50),
        loop_lag_p99=pct(monitor.lags_ms, 99),
        loop_lag_max=max(monitor.lags_ms) if monitor.lags_ms else 0.0,
        late_frames_pct=100 * len(late) / expected_frames if expected_frames else 0.0,
        frame_late_p99=pct(late, 99),
        turns=len(window.turn_latencies),
        first_audio_p95=pct(window.turn_latencies, 95) if window.turn_latencies else None,
        llm_first_token_p95=pct(window.llm_first_token, 95) if window.llm_first_token else None,
        errors=list(window.errors),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _failed(rung: Rung, *, args: argparse.Namespace, latency_ms: float | None) -> bool:
    return not rung.healthy(args.lag_budget, DEFAULT_LATE_FRAME_PCT, latency_budget_ms=latency_ms)


def latency_budget(args: argparse.Namespace, baseline_ms: float | None) -> float | None:
    """Turn-latency ceiling, anchored to this machine's single-session baseline.

    Absolute milliseconds are not portable — the mock TTS, the CPU, and the
    detector all move the floor. A multiple of the uncontended baseline asks the
    portable question instead: how much worse did *concurrency* make it? The
    floor keeps a fast baseline from making the bar unreachably tight.
    """
    if baseline_ms is None:
        return None
    return max(baseline_ms * args.latency_degradation, args.latency_floor)


def print_rung(r: Rung, budget: float, latency_ms: float | None, note: str = "") -> None:
    flag = "ok " if r.healthy(budget, DEFAULT_LATE_FRAME_PCT, latency_budget_ms=latency_ms) else "OVER"
    tail = f"  [{note}]" if note else ""
    latency = f"{r.first_audio_p95:6.0f}" if r.first_audio_p95 is not None else "     -"
    to_llm = f"{r.llm_first_token_p95:5.0f}" if r.llm_first_token_p95 is not None else "    -"
    print(  # noqa: T201
        f"  {flag} n={r.sessions:3d}  loop lag p50/p99/max "
        f"{r.loop_lag_p50:5.1f}/{r.loop_lag_p99:6.1f}/{r.loop_lag_max:7.1f} ms  "
        f"late frames {r.late_frames_pct:5.2f}%  "
        f"cores {r.cores_used:5.2f} ({r.cores_per_session:.3f}/session)  "
        f"turns {r.turns:3d}  eou→llm p95 {to_llm}  eou→audio p95 {latency} ms{tail}"
    )
    for err in r.errors[:3]:
        print(f"       ! {err}")  # noqa: T201


def report(args: argparse.Namespace, rungs: list[Rung], latency_ms: float | None) -> dict:
    cpus = available_cpus()

    def ok(r: Rung) -> bool:
        return r.healthy(args.lag_budget, DEFAULT_LATE_FRAME_PCT, latency_budget_ms=latency_ms)

    healthy = [r for r in rungs if ok(r)]
    per_loop = max((r.sessions for r in healthy), default=0)
    # A ceiling only means something if we actually hit it. If every rung passed
    # we know a lower bound, not a limit, and must not print it as one.
    exhausted = bool(rungs) and all(ok(r) for r in rungs)
    # Cost per session from the largest healthy rung: the smallest rungs are
    # dominated by fixed overhead and flatter the number.
    reference = max(healthy, key=lambda r: r.sessions, default=None)
    cores_each = reference.cores_per_session if reference else 0.0
    # Leave headroom: a loop sized to 100% of measured CPU has no room for a
    # GC pause, a reconnect, or a turn that lands on every session at once.
    per_cpu_throughput = (args.target_utilization / cores_each) if cores_each > 0 else 0.0

    print()  # noqa: T201
    print("=" * 78)  # noqa: T201
    print(  # noqa: T201
        f"detector={args.detector} vad={'off' if args.no_vad else 'silero'} "
        f"recording={'on' if args.recording else 'off'}  |  available_cpus={cpus:.2f}"
    )
    if latency_ms is not None:
        print(  # noqa: T201
            f"  turn-latency budget: {latency_ms:.0f} ms ({args.latency_degradation:.0f}x the single-session baseline)"
        )
    bound = ">=" if exhausted else "="
    print(  # noqa: T201
        f"  sessions per event loop (p99 lag <= {args.lag_budget:.0f}ms): {bound} {per_loop}"
    )
    if cores_each > 0:
        print(  # noqa: T201
            f"  CPU per session: {cores_each:.3f} cores"
            f"  ->  {per_cpu_throughput:.1f} sessions/cpu at {args.target_utilization:.0%} target"
        )
    if not per_loop:
        print("  no rung stayed inside the budget — raise --lag-budget or check --detector")  # noqa: T201
    elif exhausted:
        print(  # noqa: T201
            f"  every rung passed, so {per_loop} is a floor and not a ceiling — "
            "re-run with a larger --max-sessions, or under `docker --cpus=1`"
        )
        print(  # noqa: T201
            f"  _PER_CPU lower bound from CPU cost alone: {per_cpu_throughput / max(cpus, 0.1):.2f}"
        )
    else:
        binding = "the event loop" if per_loop <= per_cpu_throughput else "CPU"
        print(f"  binding constraint: {binding}")  # noqa: T201
        print(  # noqa: T201
            f"  _PER_CPU suggestion: {min(per_loop, per_cpu_throughput) / max(cpus, 0.1):.2f}"
            "   (sessions/loop is per *worker*, not per core — see the module docstring)"
        )
    print("=" * 78)  # noqa: T201

    return {
        "detector": args.detector,
        "vad": not args.no_vad,
        "recording": bool(args.recording),
        "available_cpus": cpus,
        "lag_budget_ms": args.lag_budget,
        "sessions_per_loop": per_loop,
        "sessions_per_loop_is_a_floor": exhausted,
        "turn_latency_budget_ms": latency_ms,
        "cores_per_session": cores_each,
        "sessions_per_cpu_at_target": per_cpu_throughput,
        "rungs": [
            {
                "sessions": r.sessions,
                "loop_lag_p50_ms": round(r.loop_lag_p50, 2),
                "loop_lag_p99_ms": round(r.loop_lag_p99, 2),
                "loop_lag_max_ms": round(r.loop_lag_max, 2),
                "late_frames_pct": round(r.late_frames_pct, 3),
                "cores_used": round(r.cores_used, 3),
                "cores_per_session": round(r.cores_per_session, 4),
                "turns": r.turns,
                "eou_to_llm_first_token_p95_ms": r.llm_first_token_p95,
                "eou_to_first_audio_p95_ms": r.first_audio_p95,
                "errors": r.errors,
            }
            for r in rungs
        ],
    }


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=int, default=None, help="Measure one N instead of ramping.")
    p.add_argument("--max-sessions", type=int, default=32, help="Ramp ceiling.")
    # Long enough for several complete turns per session: a rung shorter than a
    # few turn periods measures the idle path only, and the per-turn ONNX work
    # is exactly what we are trying to price.
    p.add_argument("--duration", type=float, default=30.0, help="Seconds sampled per rung.")
    p.add_argument("--warmup", type=float, default=8.0, help="Seconds of throwaway session first.")
    p.add_argument(
        "--settle",
        type=float,
        default=8.0,
        help="Seconds to run before sampling, so session startup stays out of the window.",
    )
    # `lexical` is the cheapest mode that still engages the Silero endpointer:
    # `heuristic`/`provider`/`raw` expose no EOU model, so the session logs
    # `vad_endpointing_unavailable` and skips the inline per-32ms VAD entirely.
    # Those modes are a real deployment shape, just a much cheaper one.
    p.add_argument("--detector", default="lexical", help="lexical | local | heuristic | provider | raw")
    p.add_argument("--no-vad", action="store_true", help="Disable the Silero endpointer.")
    p.add_argument("--recording", action="store_true", help="Enable inline MP3 recording.")
    p.add_argument("--recording-dir", default=None)
    p.add_argument("--turn-period", type=float, default=5.0, help="Seconds between committed turns.")
    p.add_argument("--reply-secs", type=float, default=2.0, help="Seconds of TTS per reply.")
    p.add_argument("--lag-budget", type=float, default=DEFAULT_LAG_BUDGET_MS)
    p.add_argument(
        "--latency-degradation",
        type=float,
        default=3.0,
        help="Fail a rung whose p95 eou->first-audio exceeds this multiple of the n=1 baseline.",
    )
    p.add_argument("--latency-floor", type=float, default=60.0, help="Never fail below this many ms.")
    p.add_argument(
        "--confirm",
        type=int,
        default=2,
        help="Re-run a failing rung this many times before accepting it as the ceiling.",
    )
    p.add_argument("--target-utilization", type=float, default=0.7)
    p.add_argument("--quick", action="store_true", help="Shorter rungs, lower ceiling.")
    p.add_argument("--json", default=None, help="Write the full result to this path.")
    args = p.parse_args()

    # Nothing configures structlog in a standalone script, so it defaults to
    # printing every debug line — which at n=48 is both unreadable and real
    # work being charged to the measurement.
    setup_logging()

    if args.quick:
        args.duration = min(args.duration, 12.0)
        args.warmup = min(args.warmup, 5.0)
        args.settle = min(args.settle, 5.0)
        args.max_sessions = min(args.max_sessions, 8)

    pcm = speech_like(10.0)
    ladder = [args.sessions] if args.sessions else _ladder(args.max_sessions)

    print(  # noqa: T201
        f"voice session CPU bench — detector={args.detector} "
        f"vad={'off' if args.no_vad else 'silero'} recording={'on' if args.recording else 'off'} "
        f"{args.duration:.0f}s/rung"
    )
    rungs: list[Rung] = []
    latency_ms: float | None = None
    for n in ladder:
        rung = await measure(args, n, pcm)
        # The first rung defines "uncontended", so it sets the bar every later
        # rung is judged against.
        if latency_ms is None:
            latency_ms = latency_budget(args, rung.first_audio_p95)
        print_rung(rung, args.lag_budget, latency_ms)

        failed = functools.partial(_failed, args=args, latency_ms=latency_ms)

        # Make a failure earn it. Pooled turn-latency p95 over a few hundred
        # turns is one bad GC pause away from tripping, and an unlucky rung
        # would otherwise become "the ceiling" — which is how a capacity
        # constant ends up anchored to a laptop's background noise. A real
        # ceiling reproduces; an outlier usually does not.
        for attempt in range(args.confirm if failed(rung) and not args.sessions else 0):
            retry = await measure(args, n, pcm)
            print_rung(retry, args.lag_budget, latency_ms, note=f"retry {attempt + 1}")
            # Keep the kinder run: we are looking for the point where it
            # *cannot* keep up, not where it once didn't.
            if not failed(retry):
                rung = retry
                break

        rungs.append(rung)
        # Once the loop is late, bigger rungs only tell us how much later.
        if not args.sessions and failed(rung):
            break

    result = report(args, rungs, latency_ms)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")  # noqa: T201


def _ladder(max_sessions: int) -> list[int]:
    out, n = [], 1
    while n <= max_sessions:
        out.append(n)
        n = n * 2 if n < 4 else n + 4
    return out


if __name__ == "__main__":
    asyncio.run(main())
