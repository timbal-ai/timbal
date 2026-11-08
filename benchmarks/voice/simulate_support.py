# ruff: noqa: T201
"""Simulated phone calls against ``examples/voice_support_demo.py::agent``.

The caller is ElevenLabs speech (a different voice from the agent), paced into
the session as 20ms microphone frames. Everything on the agent side is live:
Deepgram Flux, the demo's model, ElevenLabs, the demo's prompt, tools and
``voice_config``, built with the server's own ``build_voice_session``.

A simulated speaker plays agent audio in real time, acks the played position
the way the LiveKit transport does, and drops queued audio on an interrupt.
``--noise-dbfs`` adds a room-noise floor to the microphone and ``--echo-db``
leaks the speaker back into it (headphones are roughly no echo).

Event text decides what was said. Nothing here listens: ``call.wav`` per
scenario is a stereo recording (left: caller mic, right: agent playback) for a
person to listen to.

    uv run --frozen python benchmarks/voice/simulate_support.py
    uv run --frozen python benchmarks/voice/simulate_support.py adios_alright --noise-dbfs -45 --echo-db -30

Only provider keys are read from ``--env-file``. Traces stay in memory and
recording/upload env vars are cleared, so nothing leaves the machine except
the provider calls themselves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import unicodedata
import wave
from collections import deque
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "examples"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synth import (
    BYTES_PER_SECOND,
    FRAME_BYTES,
    FRAME_SECS,
    SAMPLE_RATE,
    SILENCE_FRAME,
    frames,
    synthesize_clips,
)

PROVIDER_KEYS = ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY")
REQUIRED_KEYS = ("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
_NO_UPLOAD = (
    "TIMBAL_VOICE_RECORDING_DIR",
    "TIMBAL_VOICE_RECORDING_UPLOAD",
    "TIMBAL_API_KEY",
    "TIMBAL_API_TOKEN",
    "TIMBAL_ORG_ID",
    "TIMBAL_APP_ID",
)
ORDER_1001 = re.compile(r"1001|mil uno|Ana|Barcelona|2B|empaquetad|embalad|preparad|tr[aá]nsito", re.IGNORECASE)
FAREWELL = re.compile(r"adi[oó]s|hasta luego|hasta pronto|que tenga|nos vemos|good\s*bye|bye", re.IGNORECASE)
REACTION_SECS = 0.6


def load_provider_keys(env_file: Path) -> dict[str, str]:
    """``KEY=VALUE`` reader with the launcher's quoting rules. Returns where each key came from."""
    from voice_support_demo_livekit import read_dotenv

    file_env = read_dotenv(env_file)
    sources: dict[str, str] = {}
    for key in PROVIDER_KEYS:
        if file_env.get(key):
            os.environ[key] = file_env[key]
            sources[key] = "file"
        elif os.environ.get(key):
            sources[key] = "shell"
        else:
            sources[key] = "missing"
    for key in _NO_UPLOAD:
        os.environ.pop(key, None)
    # No section by this name in ~/.timbal, so no platform credentials resolve.
    os.environ["TIMBAL_PROFILE"] = "simulate-support-offline"
    return sources


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def _arr(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32)


def noise_burst(secs: float, dbfs: float, seed: int) -> bytes:
    """A cough/clatter stand-in: low-passed noise under a Hann envelope. Not speech."""
    rng = np.random.default_rng(seed)
    n = int(secs * SAMPLE_RATE)
    raw = np.convolve(rng.standard_normal(n), np.ones(6) / 6, mode="same")
    raw *= np.hanning(n)
    peak = np.max(np.abs(raw)) or 1.0
    return _pcm(raw / peak * 32768 * 10 ** (dbfs / 20))


class Speaker:
    """Plays agent audio in real time. An interrupt drops what is still queued."""

    def __init__(self) -> None:
        self.queue = bytearray()
        self.played = 0

    def feed(self, pcm: bytes) -> None:
        self.queue.extend(pcm)

    def clear(self) -> None:
        self.queue.clear()

    def tick(self) -> bytes:
        if not self.queue:
            return SILENCE_FRAME
        n = min(FRAME_BYTES, len(self.queue))
        frame = bytes(self.queue[:n])
        del self.queue[:n]
        self.played += n
        return frame + b"\x00" * (FRAME_BYTES - n)

    @property
    def playing(self) -> bool:
        return bool(self.queue)

    @property
    def played_ms(self) -> float:
        return self.played / BYTES_PER_SECOND * 1000


class Line:
    """The caller's microphone: their speech, room noise, and speaker echo, 20ms at a time."""

    def __init__(self, speaker: Speaker, *, noise_dbfs: float | None, echo_db: float | None, seed: int = 7) -> None:
        self.speaker = speaker
        self.pending: deque[bytes] = deque()
        self.mic = bytearray()
        self.out = bytearray()
        self.stopped = False
        self._rng = np.random.default_rng(seed)
        self._noise_std = None if noise_dbfs is None else 32768 * 10 ** (noise_dbfs / 20)
        self._echo_gain = None if echo_db is None else 10 ** (echo_db / 20)
        self._echo: deque[bytes] = deque([SILENCE_FRAME] * 6)  # ~120ms acoustic + device delay

    def push(self, pcm: bytes) -> None:
        self.pending.extend(frames(pcm))

    @property
    def talking(self) -> bool:
        return bool(self.pending)

    async def drain(self) -> None:
        while self.pending and not self.stopped:
            await asyncio.sleep(FRAME_SECS)

    def _frame(self) -> bytes:
        played = self.speaker.tick()
        self.out.extend(played)
        self._echo.append(played)
        echo = self._echo.popleft()
        caller = self.pending.popleft() if self.pending else None
        if caller is None and self._noise_std is None and self._echo_gain is None:
            return SILENCE_FRAME
        mix = _arr(caller) if caller is not None else np.zeros(FRAME_BYTES // 2, dtype=np.float32)
        if self._noise_std is not None:
            mix = mix + self._rng.standard_normal(mix.size).astype(np.float32) * self._noise_std
        if self._echo_gain is not None:
            mix = mix + _arr(echo) * self._echo_gain
        return _pcm(mix)

    async def stream(self):
        next_at = time.monotonic()
        while not self.stopped:
            frame = self._frame()
            self.mic.extend(frame)
            yield frame
            next_at += FRAME_SECS
            delay = next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)


@dataclass
class Call:
    name: str
    t0: float
    wall0: float
    speaker: Speaker
    line: Line
    timeline: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    ended: bool = False

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def mark(self, kind: str, **data: Any) -> None:
        # Provider text can arrive decomposed ("o" + combining accent), which a
        # literal "ó" in a pattern never matches.
        data = {k: unicodedata.normalize("NFC", v) if isinstance(v, str) else v for k, v in data.items()}
        self.timeline.append({"t": self.now(), "kind": kind, **data})

    def items(self, kind: str, after: float = -1.0) -> list[dict[str, Any]]:
        return [e for e in self.timeline if e["kind"] == kind and e["t"] >= after]

    def spoken_after(self, after: float) -> str:
        return " ".join(e["text"] for e in self.items("agent_done", after) if e["text"])

    def commit_after(self, after: float) -> float | None:
        commits = self.items("commit", after)
        return commits[0]["t"] if commits else None

    def answer_to(self, said_at: float) -> str:
        """Agent text finished after the commit of what the caller said at ``said_at``.

        Text-done precedes playback, so a reply that was still playing when the
        caller spoke must not count as the answer to them.
        """
        commit = self.commit_after(said_at)
        return "" if commit is None else self.spoken_after(commit - 0.3)

    async def wait(self, pred: Callable[[], bool], timeout: float, what: str) -> bool:
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                self.timeouts.append(what)
                self.mark("timeout", what=what)
                return False
            await asyncio.sleep(0.05)
        return True

    async def say(self, clips: dict[str, bytes], text: str) -> float:
        start = self.now()
        if self.ended:
            self.mark("caller_says_after_hangup", text=text)
            return start
        self.mark("caller_says", text=text)
        self.line.push(clips[text])
        await self.line.drain()
        return start

    async def noise(self, pcm: bytes, label: str) -> float:
        start = self.now()
        if self.ended:
            return start
        self.mark("caller_noise", text=label)
        self.line.push(pcm)
        await self.line.drain()
        return start

    async def reply(self, after: float, what: str, timeout: float = 25.0) -> bool:
        """Wait for the answer to what the caller said at ``after`` to finish playing, then a human beat."""

        def answered() -> bool:
            commit = self.commit_after(after)
            return commit is not None and bool(self.items("agent_done", commit - 0.3)) and not self.speaker.playing

        ok = await self.wait(answered, timeout, what)
        await asyncio.sleep(REACTION_SECS)
        return ok

    async def agent_audible(self, after: float, ms: float, what: str, timeout: float = 15.0) -> bool:
        """Wait until the caller has heard ``ms`` of agent audio that started after ``after``."""
        baseline: dict[str, float] = {}

        def heard() -> bool:
            if self.now() < after:
                return False
            if "ms" not in baseline:
                if not self.speaker.playing:
                    return False
                baseline["ms"] = self.speaker.played_ms
            return self.speaker.played_ms - baseline["ms"] >= ms

        return await self.wait(heard, timeout, what)


def observe(event: Any, call: Call) -> None:
    from timbal.voice import (
        AgentTextDone,
        AudioOutput,
        FillerSpoken,
        SessionEnded,
        SessionError,
        SessionInterrupted,
        TranscriptCommitted,
        TranscriptPartial,
        TurnMetricsEvent,
    )

    if isinstance(event, AudioOutput):
        if not call.speaker.playing:
            call.mark("agent_audio_start")
        call.speaker.feed(event.data)
    elif isinstance(event, TranscriptPartial):
        call.mark("partial", text=event.text or "")
    elif isinstance(event, TranscriptCommitted):
        call.mark("commit", text=event.text or "", replace=event.replace)
    elif isinstance(event, AgentTextDone):
        call.mark("agent_done", text=event.text or "")
    elif isinstance(event, FillerSpoken):
        call.mark("filler", text=event.text or "")
    elif isinstance(event, SessionInterrupted):
        call.speaker.clear()
        call.mark("interrupted", heard=event.heard_text)
    elif isinstance(event, SessionError):
        call.mark("error", text=event.message)
    elif isinstance(event, TurnMetricsEvent):
        call.metrics.append(event.metrics.model_dump())
    elif isinstance(event, SessionEnded):
        call.ended = True
        call.mark("ended", unplayed_ms=round(len(call.speaker.queue) / BYTES_PER_SECOND * 1000))


def _record_provider_messages(stt: Any, call: Call) -> None:
    """Every Flux message, before the adapter filters it: what the provider actually sent."""
    handle = getattr(stt, "_handle_message", None)
    if handle is None:
        return

    async def recorded(msg: dict[str, Any]) -> None:
        call.mark("stt_raw", text=f"{msg.get('type')}/{msg.get('event', '')}: {msg.get('transcript', '')!r}")
        await handle(msg)

    stt._handle_message = recorded


def trace_summary(storage: dict[str, Any], wall0: float) -> list[dict[str, Any]]:
    """LLM and tool spans of every run, as offsets from the call start."""
    runs: list[dict[str, Any]] = []
    for run_id, trace in storage.items():
        spans = []
        for span in trace.as_records():
            depth = span.path.count(".")
            if not (span.path.endswith(".llm") or depth == 1):
                continue
            status = getattr(span, "status", None)
            is_llm = span.path.endswith(".llm")
            spans.append(
                {
                    "path": span.path,
                    "kind": "llm" if is_llm else "tool",
                    "t0": round(span.t0 / 1000 - wall0, 3),
                    "t1": None if span.t1 is None else round(span.t1 / 1000 - wall0, 3),
                    "status": None if status is None else f"{status.code}/{getattr(status, 'reason', None)}",
                    "ttft": (span.metadata or {}).get("ttft"),
                    "output": None if is_llm else str(span.output)[:120],
                }
            )
        spans.sort(key=lambda s: s["t0"])
        runs.append({"run_id": run_id, "spans": spans})
    runs.sort(key=lambda r: r["spans"][0]["t0"] if r["spans"] else 0)
    return runs


def llm_calls_after_committed_hangup(runs: list[dict[str, Any]]) -> int:
    count = 0
    for run in runs:
        done = [s for s in run["spans"] if s["path"].endswith("end_call") and "Ending the call" in (s["output"] or "")]
        if done:
            count += sum(1 for s in run["spans"] if s["kind"] == "llm" and s["t0"] >= done[0]["t1"])
    return count


def write_call_wav(path: Path, mic: bytes, out: bytes) -> None:
    n = min(len(mic), len(out)) // 2
    left, right = np.frombuffer(mic[: n * 2], np.int16), np.frombuffer(out[: n * 2], np.int16)
    stereo = np.empty(n * 2, dtype=np.int16)
    stereo[0::2], stereo[1::2] = left, right
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(stereo.tobytes())


# --------------------------------------------------------------------------- scenarios
# Each returns failures. They read what happened from events and the trace only.


async def sc_session_replay(call: Call, clips: dict[str, bytes]) -> list[str]:
    """pg-51d8f1239553 caller lines in order, ending with Adiós."""
    fails: list[str] = []
    for text in ("Hola.", "No sé cuánto me hago.", "Cuéntame algo, por favor.", "Gracias.", "Vamos, vámonos."):
        start = await call.say(clips, text)
        await call.reply(start, f"reply to {text!r}")
        if call.ended:
            return [f"call ended after {text!r}"]
    before_bye = call.now()
    if FAREWELL.search(call.spoken_after(0)):
        fails.append(f"farewell before the caller said goodbye: {call.spoken_after(0)!r}")
    await call.say(clips, "Adiós.")
    await call.wait(lambda: call.ended, 15, "hangup after Adiós")
    if len(FAREWELL.findall(call.spoken_after(before_bye))) != 1:
        fails.append(f"farewell count after Adiós: {call.spoken_after(before_bye)!r}")
    return fails


async def sc_adios_alright(call: Call, clips: dict[str, bytes]) -> list[str]:
    """Adiós, then 'Alright. Alright.' while the farewell is playing: one farewell, then hang up."""
    start = await call.say(clips, "Hola, ¿cómo va el pedido mil uno?")
    await call.reply(start, "order reply")
    bye = await call.say(clips, "Adiós.")
    await call.agent_audible(bye, 250, "farewell audible")
    ack = await call.say(clips, "Alright. Alright.")
    await call.wait(lambda: call.ended, 15, "hangup after ack")
    fails = []
    if not call.ended:
        fails.append("the acknowledgement kept the call up")
    farewells = FAREWELL.findall(call.spoken_after(bye))
    if len(farewells) != 1:
        fails.append(f"farewell count {len(farewells)}: {call.spoken_after(bye)!r}")
    answer = FAREWELL.sub("", call.answer_to(ack)).strip(" .,!¡")
    if answer:
        fails.append(f"agent answered the ack: {call.answer_to(ack)!r}")
    return fails


async def sc_adios_wait(call: Call, clips: dict[str, bytes]) -> list[str]:
    """Adiós, then 'Espera, una cosa más' during the farewell: stay up, answer, then hang up on a later goodbye."""
    start = await call.say(clips, "Hola.")
    await call.reply(start, "greeting reply")
    bye = await call.say(clips, "Adiós.")
    await call.agent_audible(bye, 250, "farewell audible")
    wait_t = await call.say(clips, "Espera, una cosa más.")
    await call.reply(wait_t, "reply to the retract")
    fails = []
    if call.ended:
        return ["hung up after 'Espera, una cosa más'"]
    ask = await call.say(clips, "¿Dónde está el pedido mil uno?")
    await call.reply(ask, "tracking reply")
    answer = call.answer_to(ask)
    if not ORDER_1001.search(answer):
        fails.append(f"order 1001 not answered after the retract: {answer!r}")
    if call.ended:
        fails.append("hung up before the second goodbye")
    second = await call.say(clips, "Vale, gracias. Adiós.")
    await call.wait(lambda: call.ended, 15, "hangup after second goodbye")
    if not call.ended:
        fails.append("second goodbye did not hang up")
    if len(FAREWELL.findall(call.spoken_after(second))) > 1:
        fails.append(f"more than one farewell at the end: {call.spoken_after(second)!r}")
    return fails


async def sc_slow_lookup_interrupted(call: Call, clips: dict[str, bytes]) -> list[str]:
    """Order 1002 is slow. Interrupt its waiting line with another request: answer that, never the stale one."""
    ask = await call.say(clips, "¿Cuál es el estado del pedido mil dos?")
    await call.wait(lambda: bool(call.items("filler", ask)), 12, "waiting line")
    await call.agent_audible(ask, 200, "waiting line audible")
    barge = await call.say(clips, "No, olvídalo. ¿Dónde está el pedido mil uno?")
    await call.reply(barge, "reply to the new request")
    await asyncio.sleep(SLOW_WAIT_SECS)
    fails = []
    after = call.spoken_after(barge)
    if re.search(r"Girona|Jordi|procesando|processing|etiqueta|label|mil dos|1002", after, re.IGNORECASE):
        fails.append(f"stale order 1002 result spoken after the interruption: {after!r}")
    if not ORDER_1001.search(call.answer_to(barge)):
        fails.append(f"new request not answered: {call.answer_to(barge)!r}")
    return fails


SLOW_WAIT_SECS = 5.0


async def sc_noise_then_barge(call: Call, clips: dict[str, bytes], bursts: list[bytes]) -> list[str]:
    """Noise bursts during a reply do not cut it. Two real interruptions in a row each do."""
    ask = await call.say(clips, "Explícame con detalle cómo funcionan los envíos de Norte Market y cuánto tardan.")
    await call.agent_audible(ask, 600, "first reply audible")
    noise_t = call.now()
    for i, burst in enumerate(bursts):
        await call.noise(burst, f"burst {i}")
        await asyncio.sleep(0.35)
    await asyncio.sleep(0.8)
    fails = []
    cut_by_noise = call.items("interrupted", noise_t)
    if cut_by_noise:
        fails.append(f"noise interrupted the reply: {cut_by_noise}")
    commits_from_noise = call.items("commit", noise_t)
    if commits_from_noise:
        fails.append(f"noise produced a commit: {commits_from_noise}")
    if not call.speaker.playing and not call.items("agent_done", ask):
        fails.append("first reply was not playing when the real interruption came")
    first = await call.say(clips, "Espera, tengo otra pregunta.")
    await call.wait(lambda: bool(call.items("interrupted", first)) or not call.speaker.playing, 8, "first barge")
    await call.reply(first, "reply to first barge")
    second_ask = await call.say(clips, "¿Hacen envíos a Girona y a Valencia, y cuánto cuestan?")
    await call.agent_audible(second_ask, 500, "second reply audible")
    second = await call.say(clips, "Perdona, otra cosa: ¿abren los domingos?")
    await call.reply(second, "reply to second barge")
    if not call.items("interrupted", first):
        fails.append("first real interruption did not cut playback (or the reply had already ended)")
    if not call.items("interrupted", second):
        fails.append("second real interruption did not cut playback (or the reply had already ended)")
    if not call.answer_to(second).strip():
        fails.append("no answer after the second interruption")
    return fails


SCENARIOS = {
    "session_replay": sc_session_replay,
    "adios_alright": sc_adios_alright,
    "adios_wait": sc_adios_wait,
    "slow_lookup_interrupted": sc_slow_lookup_interrupted,
    "noise_then_barge": sc_noise_then_barge,
}
CALLER_LINES = [
    "Hola.",
    "No sé cuánto me hago.",
    "Cuéntame algo, por favor.",
    "Gracias.",
    "Vamos, vámonos.",
    "Adiós.",
    "Hola, ¿cómo va el pedido mil uno?",
    "Alright. Alright.",
    "Espera, una cosa más.",
    "¿Dónde está el pedido mil uno?",
    "Vale, gracias. Adiós.",
    "¿Cuál es el estado del pedido mil dos?",
    "No, olvídalo. ¿Dónde está el pedido mil uno?",
    "Explícame con detalle cómo funcionan los envíos de Norte Market y cuánto tardan.",
    "Espera, tengo otra pregunta.",
    "¿Hacen envíos a Girona y a Valencia, y cuánto cuestan?",
    "Perdona, otra cosa: ¿abren los domingos?",
]


def _configure_logs(path: Path, level: str) -> Any:
    """Session logs as JSON lines. The handle stays open: closing tasks still log after the call."""
    handle = path.open("w")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(default=str),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.WriteLoggerFactory(file=handle),
        cache_logger_on_first_use=False,
    )
    return handle


async def run_scenario(name: str, clips: dict[str, bytes], args: argparse.Namespace, dest: Path) -> dict[str, Any]:
    import voice_support_demo as demo
    from timbal import Agent
    from timbal.server.voice import build_voice_session
    from timbal.state.tracing.providers import InMemoryTracingProvider
    from timbal.voice import SessionStarted, VoiceConfig

    demo.reset_demo_store()
    storage: dict[str, Any] = {}
    agent = Agent(
        name=demo.agent.name,
        model=demo._MODEL,
        system_prompt=demo._PROMPT,
        tools=[
            demo.lookup_order,
            demo.lookup_customer,
            demo.track_shipment,
            demo.stage_delivery_change,
            demo.stage_support_ticket,
            demo.apply_pending,
            demo.cancel_pending,
            demo.end_call,
        ],
        voice_config=demo.voice_config,
        tracing_provider=InMemoryTracingProvider.configured(_storage=storage),
    )
    defaults = VoiceConfig.model_validate(demo.voice_config)
    session, meta = build_voice_session(agent, defaults, {})

    speaker = Speaker()
    line = Line(speaker, noise_dbfs=args.noise_dbfs, echo_db=args.echo_db)
    call = Call(name=name, t0=time.monotonic(), wall0=time.time(), speaker=speaker, line=line)
    _record_provider_messages(session.stt, call)
    _configure_logs(dest / f"{name}.log.jsonl", args.log_level)
    bursts = [noise_burst(0.3, -14, seed) for seed in (1, 2)]
    fails: list[str] = []

    async def acks() -> None:
        while True:
            await asyncio.sleep(0.25)
            session.playback.on_playback_ack(speaker.played_ms)

    async def drive() -> None:
        script = SCENARIOS[name]
        try:
            if name == "noise_then_barge":
                fails.extend(await script(call, clips, bursts))
            else:
                fails.extend(await script(call, clips))
        finally:
            await asyncio.sleep(0.5)
            line.stopped = True
            if not session.closed:
                await session.close()

    tasks: list[asyncio.Task[Any]] = []
    try:
        async with asyncio.timeout(args.timeout):
            async with aclosing(session.run(line.stream())) as stream:
                async for event in stream:
                    observe(event, call)
                    if isinstance(event, SessionStarted) and not tasks:
                        call.mark("session_started")
                        tasks = [asyncio.create_task(acks()), asyncio.create_task(drive())]
            # A hangup ends the stream while the script is still checking what
            # happened. Its failures are the result, so let it finish.
            if len(tasks) == 2:
                await asyncio.wait({tasks[1]}, timeout=20)
                if not tasks[1].done():
                    call.timeouts.append("scenario script after session end")
    except TimeoutError:
        call.timeouts.append(f"scenario wall {args.timeout:.0f}s")
    finally:
        line.stopped = True
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not session.closed:
            await session.close()

    runs = trace_summary(storage, call.wall0)
    after_hangup = llm_calls_after_committed_hangup(runs)
    if after_hangup:
        fails.append(f"{after_hangup} LLM call(s) after a committed end_call")
    if call.timeouts:
        fails.append(f"timed out waiting for {call.timeouts}")
    errors = call.items("error")
    if errors:
        fails.append(f"session errors: {[e['text'] for e in errors]}")
    write_call_wav(dest / f"{name}.call.wav", bytes(line.mic), bytes(line.out))
    report = {
        "name": name,
        "passed": not fails,
        "failures": fails,
        "providers": {k: meta.get(k) for k in ("stt_provider", "stt_model", "tts_provider", "model")},
        "line": {"noise_dbfs": args.noise_dbfs, "echo_db": args.echo_db},
        "timeline": call.timeline,
        "turn_metrics": call.metrics,
        "runs": runs,
        "llm_calls_after_committed_hangup": after_hangup,
        "wav": str(dest / f"{name}.call.wav"),
        "log": str(dest / f"{name}.log.jsonl"),
    }
    (dest / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    return report


def print_report(report: dict[str, Any]) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(f"\n=== {report['name']}: {status}  {report['providers']}  line={report['line']}")
    for item in report["timeline"]:
        kind = item["kind"]
        if kind in ("partial", "agent_audio_start"):
            continue
        text = item.get("text", item.get("heard", item.get("what", item.get("unplayed_ms", ""))))
        print(f"  {item['t']:7.2f}  {kind:<14} {text}")
    for m in report["turn_metrics"]:
        print(
            f"  metrics turn={m['turn_index']} eou->first_token={m['eou_to_llm_first_token_ms']} "
            f"eou->first_audio={m['eou_to_first_audio_ms']} speculative={m['speculative']} "
            f"interrupted={m['interrupted']} filler={m['filler_spoken']}"
        )
    for run in report["runs"]:
        for s in run["spans"]:
            ttft = f" ttft={s['ttft']:.3f}" if isinstance(s["ttft"], float) else ""
            out = f" -> {s['output']!r}" if s["output"] else ""
            print(f"  span {s['t0']:7.2f}-{s['t1']}  {s['path']:<40} {s['status']}{ttft}{out}")
    for failure in report["failures"]:
        print(f"  FAIL: {failure}")
    print(f"  wav: {report['wav']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", help=f"any of {', '.join(SCENARIOS)} (default: all)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--noise-dbfs", type=float, default=None, help="room noise floor, e.g. -50 quiet, -40 busy")
    parser.add_argument("--echo-db", type=float, default=None, help="speaker leak into the mic, e.g. -30")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    parser.add_argument("--repeat", type=int, default=1, help="run each scenario this many times")
    args = parser.parse_args()
    unknown = [name for name in args.scenarios if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s) {unknown}")

    sources = load_provider_keys(args.env_file)
    print("keys:", {k: v for k, v in sources.items()})
    missing = [k for k in REQUIRED_KEYS if sources[k] == "missing"]
    if missing:
        print(f"missing {missing}")
        return 2
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    clips = await synthesize_clips(CALLER_LINES)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = HERE / "results" / "simulate_support" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.scenarios or list(SCENARIOS):
        for attempt in range(args.repeat):
            run_dest = dest / f"{name}-{attempt}" if args.repeat > 1 else dest
            run_dest.mkdir(parents=True, exist_ok=True)
            report = await run_scenario(name, clips, args, run_dest)
            report["attempt"] = attempt
            print_report(report)
            reports.append(report)
    summary = [
        {"name": r["name"], "attempt": r["attempt"], "passed": r["passed"], "failures": r["failures"]} for r in reports
    ]
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\nresults: {dest}")
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
