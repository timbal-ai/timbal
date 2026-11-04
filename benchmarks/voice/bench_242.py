# ruff: noqa: T201
"""Before/after measurement for the #242 voice work.

One command from the repo root, with the voice extra installed::

    uv run python benchmarks/voice/bench_242.py all

``before`` is ``origin/main`` at default ``VoiceConfig`` (Deepgram Flux when
``DEEPGRAM_API_KEY`` is set, else ElevenLabs Scribe). ``after`` is this tree
with the candidate best config. Ablations turn one knob off (tempo is turned
on). Caller audio is synthetic OpenAI ``gpt-4o-mini-tts`` PCM, resampled to
16 kHz and paced in real time through the cascaded session the server builds.

Raw rows land in ``benchmarks/voice/242-results/``. Nothing here is a unit
test and nothing is wired into pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import wave
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUDIO_DIR = HERE / "fixtures" / "242"
OUT_DIR = HERE / "242-results"
SAMPLE_RATE = 16_000
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "coral"
OPENAI_TTS_RATE = 24_000

# Candidate "best". Tempo stays off until a run shows it helps.
BEST: dict[str, Any] = {
    "ignore_backchannels": True,
    "hold_unfinished_secs": 1.2,
    "min_delay_secs": 0.30,
    "stop_silence_secs": 0.20,
    "first_flush_min_words": 4,
    "predict_during_speech": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
}

# One knob changed relative to BEST. ``tempo`` turns the feature on.
ABLATIONS: dict[str, dict[str, Any]] = {
    "no_backchannels": {"ignore_backchannels": False},
    "no_hold": {"hold_unfinished_secs": None},
    "default_floors": {"min_delay_secs": 0.45, "stop_silence_secs": 0.20},
    "no_first_flush": {"first_flush_min_words": None},
    "no_predict": {"predict_during_speech": False},
    "no_speculative": {"speculative_generation": {"enabled": False}},
    "speculative_hold_tts": {"speculative_generation": {"enabled": True, "preemptive_tts": False}},
    "no_pause_trim": {"max_tts_pause_secs": None},
    "tempo_1_05": {"tts_tempo": 1.05},
    "cache_bust": {},
}


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    lang: str
    kind: str
    # (text, role) role is "say" | "backchannel" | "barge" | "continuation"
    lines: tuple[tuple[str, str], ...]
    gap_before_continuation: float = 0.8
    # Groups this scenario is worth ablating. Empty = headline only.
    ablate: tuple[str, ...] = ()


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("en_statement", "en", "latency", (("The return window is thirty days, so I can still send it back.", "say"),)),
    ScenarioSpec("es_statement", "es", "latency", (("La ventana de devolución es de treinta días, así que todavía puedo enviarlo.", "say"),)),
    ScenarioSpec("en_question", "en", "latency", (("Can you check whether order four eight two one has already shipped?", "say"),)),
    ScenarioSpec("es_question", "es", "latency", (("¿Puedes mirar si el pedido cuatro ocho dos uno ya ha salido?", "say"),)),
    ScenarioSpec(
        "en_unfinished",
        "en",
        "unfinished",
        (("I want to book it for", "say"), ("Thursday afternoon", "continuation")),
        gap_before_continuation=0.8,
        ablate=("no_hold", "default_floors", "no_predict"),
    ),
    ScenarioSpec(
        "es_unfinished",
        "es",
        "unfinished",
        (("quiero reservar para", "say"), ("el jueves por la tarde", "continuation")),
        gap_before_continuation=0.8,
        ablate=("no_hold", "default_floors", "no_predict"),
    ),
    ScenarioSpec(
        "en_backchannel_okay",
        "en",
        "backchannel",
        (
            ("Explain the return policy in two sentences.", "say"),
            ("okay got it", "backchannel"),
        ),
        ablate=("no_backchannels",),
    ),
    ScenarioSpec(
        "en_backchannel_mhm",
        "en",
        "backchannel",
        (
            ("Explain the return policy in two sentences.", "say"),
            ("mhm", "backchannel"),
        ),
        ablate=("no_backchannels",),
    ),
    ScenarioSpec(
        "es_backchannel_vale",
        "es",
        "backchannel",
        (
            ("Explícame la política de devoluciones en dos frases.", "say"),
            ("vale", "backchannel"),
        ),
        ablate=("no_backchannels",),
    ),
    ScenarioSpec(
        "en_barge",
        "en",
        "barge",
        (
            ("Tell me the return policy in two sentences.", "say"),
            ("wait, stop, I need the order status instead", "barge"),
        ),
    ),
    ScenarioSpec(
        "es_barge",
        "es",
        "barge",
        (
            ("Explícame la política de devoluciones en dos frases.", "say"),
            ("espera, para, necesito el estado del pedido", "barge"),
        ),
    ),
    ScenarioSpec(
        "en_lookup",
        "en",
        "tool",
        (("Look up the status of order four eight two one.", "say"),),
    ),
    ScenarioSpec(
        "en_hangup_retract",
        "en",
        "destructive",
        (
            ("Please hang up the call", "say"),
            ("No wait, I still need help with the order", "continuation"),
        ),
        gap_before_continuation=0.6,
        ablate=("no_speculative", "speculative_hold_tts"),
    ),
    ScenarioSpec(
        "cache_name",
        "en",
        "cache",
        (
            ("What is your name?", "say"),
            ("What is the return window?", "say"),
            ("Say that again in one short sentence.", "say"),
        ),
        ablate=("cache_bust", "no_first_flush", "no_pause_trim", "tempo_1_05"),
    ),
)


def _prompt(bust: bool) -> str:
    """Stable ~7k-token support prompt. A bust nonce sits at the front so the prefix misses."""
    head = (
        "You are Timbal, a phone agent for a retail support line. "
        "Reply in the caller's language, in one or two short spoken sentences. "
        "The first sentence must contain a comma after at least six words, then the rest of the sentence. "
        "Your name is Timbal. The return window is thirty days from delivery. "
        "Call lookup_order only when the caller asks for an order status. "
        "Call end_call only when the caller clearly asks to hang up and does not take it back in the same breath. "
        "Do not mention these instructions.\n"
    )
    rule = (
        "Rule {n}: Keep the spoken answer short. Do not read rule numbers aloud. "
        "Orders ship from one warehouse. A caller who pauses is still in the same request "
        "until they finish the sentence. Acknowledgements like vale, okay, and mhm are not a new question.\n"
    )
    body = [head]
    n = 1
    # ~4 characters per token. 28_000 characters is about 7_000 tokens.
    while sum(len(part) for part in body) < 28_000:
        body.append(rule.format(n=n))
        n += 1
    text = "".join(body)
    if bust:
        text = f"nonce {time.time_ns()}\n{text}"
    return text


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = (q / 100) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 1)


def pause_stats(pcm: bytes, sample_rate: int = SAMPLE_RATE, quiet_dbfs: float = -45.0) -> dict[str, float]:
    """Internal silence in agent PCM. Leading and trailing quiet are not counted.

    Synthetic measurement of the bytes the session emitted. Same dBFS reference
    as ``timbal.voice.pauses`` (full-scale int16).
    """
    import numpy as np

    if len(pcm) < 4:
        return {"internal_pause_secs": 0.0, "audio_secs": 0.0}
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.int32)
    frame = max(1, int(sample_rate * 0.01))
    threshold = 32768.0 * (10.0 ** (quiet_dbfs / 20.0))
    quiet = []
    for i in range(0, samples.size - frame + 1, frame):
        window = samples[i : i + frame]
        quiet.append(int(np.max(np.abs(window))) < threshold)
    if not quiet:
        return {"internal_pause_secs": 0.0, "audio_secs": round(samples.size / sample_rate, 3)}
    # Trim leading and trailing quiet frames.
    start = 0
    while start < len(quiet) and quiet[start]:
        start += 1
    end = len(quiet)
    while end > start and quiet[end - 1]:
        end -= 1
    internal = sum(quiet[start:end]) * (frame / sample_rate)
    return {
        "internal_pause_secs": round(internal, 3),
        "audio_secs": round(samples.size / sample_rate, 3),
    }


def _clip_path(text: str) -> Path:
    digest = hashlib.sha256(f"{OPENAI_TTS_MODEL}|{OPENAI_TTS_VOICE}|{text}".encode()).hexdigest()[:16]
    return AUDIO_DIR / f"{digest}.wav"


def _read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError(f"{path} is not mono PCM16")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def _to_16k(pcm: bytes, rate: int) -> bytes:
    if rate == SAMPLE_RATE:
        return pcm
    from timbal.voice.telephony import PcmResampler

    resampler = PcmResampler(rate, SAMPLE_RATE)
    out = resampler.process(pcm) + resampler.flush()
    return out


def synthesize(texts: list[str]) -> dict[str, bytes]:
    """OpenAI TTS → 16 kHz PCM16. Cached on disk. Marked synthetic by the caller."""
    missing = [text for text in texts if not _clip_path(text).exists()]
    if missing:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; cannot synthesize caller audio")
        from openai import OpenAI

        client = OpenAI()
        for text in missing:
            print(f"synth {text!r}", flush=True)
            speech = client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text,
                response_format="wav",
            )
            raw_path = _clip_path(text).with_suffix(".src.wav")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(speech.read())
            pcm, rate = _read_wav(raw_path)
            _write_wav(_clip_path(text), _to_16k(pcm, rate), SAMPLE_RATE)
            raw_path.unlink(missing_ok=True)
    clips: dict[str, bytes] = {}
    for text in texts:
        pcm, rate = _read_wav(_clip_path(text))
        if rate != SAMPLE_RATE:
            pcm = _to_16k(pcm, rate)
        clips[text] = pcm
    return clips


def _scenario_texts() -> list[str]:
    seen: list[str] = []
    for spec in SCENARIOS:
        for text, _role in spec.lines:
            if text not in seen:
                seen.append(text)
    return seen


def _voice_config(overrides: dict[str, Any], lang: str, stt: str) -> tuple[Any, list[str]]:
    from timbal.voice.config import VoiceConfig

    data: dict[str, Any] = {"language": lang, "stt_provider": stt}
    data.update(overrides)
    allowed = set(VoiceConfig.model_fields)
    dropped = sorted(key for key in data if key not in allowed)
    kept = {key: value for key, value in data.items() if key in allowed}
    return VoiceConfig(**kept), dropped


def _make_agent(model: str, prompt: Any, log: list[dict[str, Any]]) -> Any:
    from timbal import Agent
    from timbal.core.tool import Tool
    from timbal.state.tracing.providers import InMemoryTracingProvider

    def lookup_order(order_id: str) -> str:
        """Look up an order by id. Read only. Does not change the order."""
        log.append({"tool": "lookup_order", "order_id": order_id, "t": time.monotonic()})
        return "Order 4821 shipped yesterday and arrives Thursday."

    def end_call() -> str:
        """Hang up. Only when the caller clearly asked to end the call and did not take it back."""
        log.append({"tool": "end_call", "t": time.monotonic()})
        from timbal.voice import current_voice_session

        session = current_voice_session()
        if session is not None:
            session.end_call()
        return "Ending the call."

    fields = set(getattr(Tool, "model_fields", {}))
    end_kwargs: dict[str, Any] = {"handler": end_call, "name": "end_call"}
    if "defer_until_eot" in fields:
        end_kwargs["defer_until_eot"] = True
    if "interruption_mode" in fields:
        end_kwargs["interruption_mode"] = "disable_during_tool"
    return Agent(
        name="voice_242",
        model=model,
        system_prompt=prompt,
        tools=[Tool(handler=lookup_order, name="lookup_order"), Tool(**end_kwargs)],
        tracing_provider=InMemoryTracingProvider,
    )


def _span_usage(ctx: Any) -> dict[str, Any]:
    usage: dict[str, int] = {}
    ttfts: list[float] = []
    trace = getattr(ctx, "_trace", None)
    if trace is None:
        return {"usage": usage, "ttft_ms": ttfts}
    for span in trace.values():
        for key, value in (getattr(span, "usage", None) or {}).items():
            usage[key] = usage.get(key, 0) + int(value)
        ttft = (span.metadata or {}).get("ttft")
        if isinstance(ttft, int | float) and ttft >= 0:
            ttfts.append(round(float(ttft) * 1000.0, 1))
    return {"usage": usage, "ttft_ms": ttfts}


def _token_bucket(usage: dict[str, int]) -> dict[str, int]:
    input_tokens = cached = output = 0
    for key, value in usage.items():
        if key.endswith(":input_cached_tokens") or key.endswith(":cache_read_input_tokens"):
            cached += value
        elif key.endswith(":input_tokens"):
            input_tokens += value
        elif key.endswith(":output_tokens"):
            output += value
    return {"input_tokens": input_tokens, "cached_tokens": cached, "output_tokens": output}


async def _run_one(
    spec: ScenarioSpec,
    clips: dict[str, bytes],
    *,
    model: str,
    stt: str,
    overrides: dict[str, Any],
    cache_bust: bool,
    side: str,
    profile: str,
    rep: int,
) -> dict[str, Any]:
    from harness import (  # noqa: PLC0415 — script-dir import, see cli.py
        FRAME_SECS,
        PlaybackSim,
        ScriptFeeder,
    )
    from timbal.server.voice import build_voice_session

    log: list[dict[str, Any]] = []
    stable = _prompt(False)
    prompt: Any = (lambda stable=stable: f"nonce {time.time_ns()}\n{stable}") if cache_bust else stable
    agent = _make_agent(model, prompt, log)
    config, dropped = _voice_config(overrides, spec.lang, stt)
    sig = inspect.signature(build_voice_session)
    extra: dict[str, Any] = {}
    if "call_context" in sig.parameters:
        extra["call_context"] = {"customer": "Acme", "user_number": "+15551212"}
    session, meta = build_voice_session(agent, config, {}, **extra)

    feeder = ScriptFeeder()
    playback = PlaybackSim()
    assistant_speaking = asyncio.Event()
    closing = asyncio.Event()
    state: dict[str, Any] = {
        "interrupted": 0,
        "premature": False,
        "continuation_fed": False,
        "pcm": bytearray(),
        "turn_pcm": bytearray(),
        "commits": [],
        "replies": [],
        "metrics": [],
        "errors": [],
        "wasted": [],
        "seen_discarded": 0,
        "first_audio_before_continuation": False,
        "awaiting_audio_from": None,
        "caller_silence_to_first_audio_ms": [],
    }

    def _note_discarded() -> None:
        discarded = int(getattr(session, "speculative_generations_discarded", 0) or 0)
        if discarded <= state["seen_discarded"]:
            return
        state["seen_discarded"] = discarded
        ctx = getattr(session, "_turn_run_context", None)
        rid = getattr(session, "_speculative_discarded_run_id", None)
        if ctx is not None and getattr(ctx, "id", None) == rid:
            state["wasted"].append(_span_usage(ctx))

    def observe(event: Any) -> None:
        from timbal.voice import (  # noqa: PLC0415
            AgentTextDone,
            AudioOutput,
            SessionError,
            SessionInterrupted,
            TranscriptCommitted,
            TurnMetricsEvent,
        )

        _note_discarded()
        if isinstance(event, AudioOutput):
            playback.on_emit(len(event.data))
            assistant_speaking.set()
            state["pcm"].extend(event.data)
            state["turn_pcm"].extend(event.data)
            if state["awaiting_audio_from"] is not None:
                # Wall clock from the last paced caller frame to the first
                # byte emitted. On Flux the TurnMetrics anchor is the last
                # partial, which lands near EndOfTurn and hides the silence
                # the caller already sat through.
                state["caller_silence_to_first_audio_ms"].append(
                    round((time.monotonic() - state["awaiting_audio_from"]) * 1000.0, 1)
                )
                state["awaiting_audio_from"] = None
            if spec.kind in {"unfinished", "destructive"} and not state["continuation_fed"]:
                state["first_audio_before_continuation"] = True
        elif isinstance(event, TranscriptCommitted):
            state["commits"].append({"text": event.text, "replace": bool(getattr(event, "replace", False)), "t": time.monotonic()})
        elif isinstance(event, AgentTextDone):
            state["replies"].append(event.text)
            assistant_speaking.clear()
        elif isinstance(event, SessionInterrupted):
            playback.on_interrupted()
            if not closing.is_set():
                state["interrupted"] += 1
        elif isinstance(event, TurnMetricsEvent):
            metrics = event.metrics
            snap = _span_usage(getattr(session, "_turn_run_context", None))
            pauses = pause_stats(bytes(state["turn_pcm"]))
            state["turn_pcm"].clear()
            dumped = metrics.model_dump() if hasattr(metrics, "model_dump") else {}
            dumped["usage"] = snap
            dumped["pause"] = pauses
            state["metrics"].append(dumped)
        elif isinstance(event, SessionError):
            state["errors"].append(event.message)

    async def wait_for(predicate, timeout: float, what: str, *, record: bool = True) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                if record:
                    state["errors"].append(f"timed out waiting for {what}")
                return
            await asyncio.sleep(FRAME_SECS)

    async def drive() -> None:
        for text, role in spec.lines:
            if role == "continuation":
                await asyncio.sleep(spec.gap_before_continuation)
            if role in {"backchannel", "barge"}:
                await wait_for(assistant_speaking.is_set, 25.0, "assistant audio")
                await wait_for(lambda: playback.played_ms >= 400, 20.0, "400ms of playback")
            feeder.push(clips[text])
            await feeder.drain()
            if role in {"say", "continuation"}:
                state["awaiting_audio_from"] = time.monotonic()
            if role == "continuation":
                state["continuation_fed"] = True
                # The follow-up has been handed to the STT. Give Flux its
                # silence timeout so a hold can merge it before we tear down.
                await asyncio.sleep(3.0)
            if role == "say" and spec.kind == "cache":
                await wait_for(lambda n=len(state["replies"]): len(state["replies"]) > n, 35.0, "reply")
        if spec.kind == "backchannel":
            await asyncio.sleep(3.0)
        elif spec.kind == "barge":
            await wait_for(lambda: state["interrupted"] > 0, 8.0, "barge-in", record=False)
        elif spec.kind != "cache":
            await wait_for(lambda: bool(state["replies"]) or bool(state["errors"]), 35.0, "any reply")
            await asyncio.sleep(0.8)
        closing.set()
        feeder.stop()
        await session.close()

    started = time.monotonic()
    async with aclosing(session.run(feeder.stream())) as stream:
        driver_task: asyncio.Task[None] | None = None
        acker: asyncio.Task[None] | None = None

        async def ack_loop() -> None:
            while True:
                await asyncio.sleep(0.25)
                session.playback.on_playback_ack(playback.played_ms)

        try:
            async for event in stream:
                from timbal.voice import SessionStarted

                observe(event)
                if isinstance(event, SessionStarted) and driver_task is None:
                    acker = asyncio.create_task(ack_loop())
                    driver_task = asyncio.create_task(drive())
        finally:
            for task in (driver_task, acker):
                if task is not None and not task.done():
                    task.cancel()
            for task in (driver_task, acker):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)

    if spec.kind in {"unfinished", "destructive"}:
        state["premature"] = bool(state["first_audio_before_continuation"])
    false_interrupt = spec.kind == "backchannel" and state["interrupted"] > 0
    missed_barge = spec.kind == "barge" and state["interrupted"] == 0
    end_calls = [row for row in log if row["tool"] == "end_call"]
    retractions = [row for row in state["commits"] if row["replace"] and not (row["text"] or "").strip()]
    discarded = int(getattr(session, "speculative_generations_discarded", 0) or 0)
    destructive_on_retracted = bool(end_calls) and (bool(retractions) or discarded > 0)
    wasted_tokens = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    for snap in state["wasted"]:
        bucket = _token_bucket(snap["usage"])
        for key in wasted_tokens:
            wasted_tokens[key] += bucket[key]
    return {
        "side": side,
        "profile": profile,
        "model": model,
        "stt": stt,
        "scenario": spec.id,
        "kind": spec.kind,
        "lang": spec.lang,
        "rep": rep,
        "audio": "synthetic-openai-tts",
        "wall_secs": round(time.monotonic() - started, 2),
        "config_dropped": dropped,
        "stt_resolved": (meta or {}).get("stt_provider"),
        "commits": state["commits"],
        "replies": state["replies"],
        "metrics": state["metrics"],
        "errors": state["errors"],
        "interrupted": state["interrupted"],
        "false_interrupt": false_interrupt,
        "missed_barge": missed_barge,
        "premature_cutoff": state["premature"],
        "tool_calls": log,
        "end_call_calls": len(end_calls),
        "retractions": len(retractions),
        "speculative_started": int(getattr(session, "speculative_generations_started", 0) or 0),
        "speculative_discarded": discarded,
        "destructive_on_retracted": destructive_on_retracted,
        "wasted_tokens": wasted_tokens,
        "caller_silence_to_first_audio_ms": state["caller_silence_to_first_audio_ms"],
        "prompt_chars": len(_prompt(False)),
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _metric_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for metrics in row.get("metrics") or []:
            value = metrics.get(field_name)
            if isinstance(value, int | float) and value >= 0 and not metrics.get("event"):
                values.append(float(value))
    return values


def _dist(values: list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "p50": _percentile(values, 50), "p90": _percentile(values, 90)}


def _caller_silence_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for value in row.get("caller_silence_to_first_audio_ms") or []:
            if isinstance(value, int | float) and value >= 0:
                values.append(float(value))
    return values


def _silence_values(rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        for metrics in row.get("metrics") or []:
            speech = metrics.get("speech_end_to_transcript_ms")
            audio = metrics.get("eou_to_first_audio_ms")
            if isinstance(speech, int | float) and isinstance(audio, int | float) and not metrics.get("event"):
                values.append(float(speech) + float(audio))
    return values


def _cache_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cold: list[float] = []
    warm: list[float] = []
    cached = input_tokens = 0
    for row in rows:
        turns = row.get("metrics") or []
        for index, metrics in enumerate(turns):
            bucket = _token_bucket((metrics.get("usage") or {}).get("usage") or {})
            ttfts = (metrics.get("usage") or {}).get("ttft_ms") or []
            ttft = ttfts[0] if ttfts else metrics.get("eou_to_llm_first_token_ms")
            cached += bucket["cached_tokens"]
            input_tokens += bucket["input_tokens"] + bucket["cached_tokens"]
            if isinstance(ttft, int | float):
                (cold if index == 0 else warm).append(float(ttft))
    rate = (cached / input_tokens) if input_tokens else None
    return {
        "ttft_cold": _dist(cold),
        "ttft_warm": _dist(warm),
        "cached_tokens": cached,
        "input_tokens_incl_cached": input_tokens,
        "hit_rate": None if rate is None else round(rate, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw rows. Uses #244 percentile math (same Hyndman–Fan type 7)."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("warmup"):
            continue
        key = (row.get("side"), row.get("profile"), row.get("model"), row.get("stt"), row.get("kind"))
        groups.setdefault(key, []).append(row)

    cells = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        side, profile, model, stt, kind = key
        pauses = []
        for row in group:
            for metrics in row.get("metrics") or []:
                pause = (metrics.get("pause") or {}).get("internal_pause_secs")
                if isinstance(pause, int | float):
                    pauses.append(float(pause))
        wasted = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        for row in group:
            for key_name in wasted:
                wasted[key_name] += int((row.get("wasted_tokens") or {}).get(key_name) or 0)
        cells.append(
            {
                "side": side,
                "profile": profile,
                "model": model,
                "stt": stt,
                "kind": kind,
                "runs": len(group),
                "caller_silence_to_first_audio_ms": _dist(_caller_silence_values(group)),
                "turn_metrics_silence_to_first_audio_ms": _dist(_silence_values(group)),
                "eou_to_first_audio_ms": _dist(_metric_values(group, "eou_to_first_audio_ms")),
                "eou_to_llm_first_token_ms": _dist(_metric_values(group, "eou_to_llm_first_token_ms")),
                "speech_end_to_transcript_ms": _dist(_metric_values(group, "speech_end_to_transcript_ms")),
                "cache": _cache_stats([row for row in group if row.get("kind") == "cache"] or group),
                "false_interrupts": sum(1 for row in group if row.get("false_interrupt")),
                "missed_barges": sum(1 for row in group if row.get("missed_barge")),
                "premature_cutoffs": sum(1 for row in group if row.get("premature_cutoff")),
                "destructive_on_retracted": sum(1 for row in group if row.get("destructive_on_retracted")),
                "end_call_calls": sum(int(row.get("end_call_calls") or 0) for row in group),
                "speculative_started": sum(int(row.get("speculative_started") or 0) for row in group),
                "speculative_discarded": sum(int(row.get("speculative_discarded") or 0) for row in group),
                "wasted_tokens": wasted,
                "internal_pause_secs": _dist(pauses),
                "errors": sum(len(row.get("errors") or []) for row in group),
            }
        )
    return {"cells": cells}


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "side",
        "profile",
        "model",
        "stt",
        "kind",
        "runs",
        "silence_p50",
        "silence_p90",
        "metrics_silence_p50",
        "eou_audio_p50",
        "eou_audio_p90",
        "llm_ttft_p50",
        "false_interrupts",
        "missed_barges",
        "premature_cutoffs",
        "destructive_on_retracted",
        "speculative_started",
        "speculative_discarded",
        "wasted_output_tokens",
        "pause_p50",
        "cache_hit_rate",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in summary["cells"]:
            writer.writerow(
                {
                    "side": cell["side"],
                    "profile": cell["profile"],
                    "model": cell["model"],
                    "stt": cell["stt"],
                    "kind": cell["kind"],
                    "runs": cell["runs"],
                    "silence_p50": cell["caller_silence_to_first_audio_ms"]["p50"],
                    "silence_p90": cell["caller_silence_to_first_audio_ms"]["p90"],
                    "metrics_silence_p50": cell["turn_metrics_silence_to_first_audio_ms"]["p50"],
                    "eou_audio_p50": cell["eou_to_first_audio_ms"]["p50"],
                    "eou_audio_p90": cell["eou_to_first_audio_ms"]["p90"],
                    "llm_ttft_p50": cell["eou_to_llm_first_token_ms"]["p50"],
                    "false_interrupts": cell["false_interrupts"],
                    "missed_barges": cell["missed_barges"],
                    "premature_cutoffs": cell["premature_cutoffs"],
                    "destructive_on_retracted": cell["destructive_on_retracted"],
                    "speculative_started": cell["speculative_started"],
                    "speculative_discarded": cell["speculative_discarded"],
                    "wasted_output_tokens": cell["wasted_tokens"]["output_tokens"],
                    "pause_p50": cell["internal_pause_secs"]["p50"],
                    "cache_hit_rate": cell["cache"]["hit_rate"],
                    "errors": cell["errors"],
                }
            )


def _jobs(side: str, reps: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    headline = [spec.id for spec in SCENARIOS]
    if side == "before":
        for spec_id in headline:
            for rep in range(reps):
                jobs.append(
                    {
                        "scenario": spec_id,
                        "model": "openai/gpt-5.4-mini",
                        "stt": "deepgram-flux",
                        "profile": "default",
                        "overrides": {},
                        "cache_bust": False,
                        "rep": rep,
                    }
                )
        for spec_id in ("en_statement", "es_question", "cache_name"):
            for rep in range(min(reps, 3)):
                jobs.append(
                    {
                        "scenario": spec_id,
                        "model": "groq/openai/gpt-oss-20b",
                        "stt": "deepgram-flux",
                        "profile": "default",
                        "overrides": {},
                        "cache_bust": False,
                        "rep": rep,
                    }
                )
        return jobs

    for spec_id in headline:
        for rep in range(reps):
            jobs.append(
                {
                    "scenario": spec_id,
                    "model": "openai/gpt-5.4-mini",
                    "stt": "deepgram-flux",
                    "profile": "best",
                    "overrides": BEST,
                    "cache_bust": False,
                    "rep": rep,
                }
            )
    for spec_id in ("en_statement", "es_question", "cache_name"):
        for rep in range(min(reps, 3)):
            jobs.append(
                {
                    "scenario": spec_id,
                    "model": "groq/openai/gpt-oss-20b",
                    "stt": "deepgram-flux",
                    "profile": "best",
                    "overrides": BEST,
                    "cache_bust": False,
                    "rep": rep,
                }
            )
    for name, patch in ABLATIONS.items():
        overrides = dict(BEST)
        overrides.update(patch)
        for spec in SCENARIOS:
            if name not in spec.ablate:
                continue
            for rep in range(min(reps, 3)):
                jobs.append(
                    {
                        "scenario": spec.id,
                        "model": "openai/gpt-5.4-mini",
                        "stt": "deepgram-flux",
                        "profile": name,
                        "overrides": overrides,
                        "cache_bust": name == "cache_bust",
                        "rep": rep,
                    }
                )
    # Floors only move the local endpointer. Flux forces it off, so this arm
    # is ElevenLabs Scribe, not the default STT.
    for profile, overrides in (
        ("scribe_default_floors", {"min_delay_secs": 0.45, "stop_silence_secs": 0.20}),
        ("scribe_tuned_floors", {"min_delay_secs": 0.30, "stop_silence_secs": 0.20, "predict_during_speech": True}),
    ):
        for spec_id in ("en_statement", "en_unfinished", "es_unfinished"):
            for rep in range(min(reps, 3)):
                jobs.append(
                    {
                        "scenario": spec_id,
                        "model": "openai/gpt-5.4-mini",
                        "stt": "elevenlabs",
                        "profile": profile,
                        "overrides": overrides,
                        "cache_bust": False,
                        "rep": rep,
                    }
                )
    return jobs


def _ensure_main_worktree() -> Path:
    path = Path("/tmp/timbal-main-242")
    if not path.exists():
        subprocess.check_call(["git", "worktree", "add", "--detach", str(path), "origin/main"], cwd=REPO)
    return path


def _run_jobs(jobs: list[dict[str, Any]], out: Path, side: str) -> None:
    clips = synthesize(_scenario_texts())
    by_id = {spec.id: spec for spec in SCENARIOS}
    for index, job in enumerate(jobs, start=1):
        spec = by_id[job["scenario"]]
        print(f"[{index}/{len(jobs)}] {side} {job['profile']} {job['model']} {spec.id} rep {job['rep']}", flush=True)
        try:
            row = asyncio.run(
                _run_one(
                    spec,
                    clips,
                    model=job["model"],
                    stt=job["stt"],
                    overrides=job["overrides"],
                    cache_bust=job["cache_bust"],
                    side=side,
                    profile=job["profile"],
                    rep=job["rep"],
                )
            )
        except Exception as exc:  # one failed call must not drop the rest of the matrix
            row = {
                "side": side,
                "profile": job["profile"],
                "model": job["model"],
                "stt": job["stt"],
                "scenario": spec.id,
                "kind": spec.kind,
                "rep": job["rep"],
                "audio": "synthetic-openai-tts",
                "errors": [f"{type(exc).__name__}: {exc}"],
                "metrics": [],
            }
            print(f"  ERROR {row['errors'][0]}", flush=True)
        else:
            silence = _silence_values([row])
            print(
                f"  silence {silence} interrupts {row.get('interrupted')} errors {row.get('errors')}",
                flush=True,
            )
        _append(out, row)


def _self_test() -> None:
    import numpy as np

    rate = 16_000
    tone = (np.sin(2 * np.pi * 220 * np.arange(rate) / rate) * 8000).astype("<i2")
    quiet = np.zeros(rate // 2, dtype="<i2")
    pcm = np.concatenate([quiet, tone, quiet, tone, quiet]).tobytes()
    stats = pause_stats(pcm, rate)
    # 0.5s of internal quiet between the tones. Leading and trailing are excluded.
    if not 0.4 <= stats["internal_pause_secs"] <= 0.6:
        raise SystemExit(f"pause self-test failed: {stats}")
    if _percentile([10, 20, 30, 40], 90) is None:
        raise SystemExit("percentile self-test failed")
    print("self-test ok", stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure #242 voice changes against main")
    parser.add_argument("command", choices=("all", "run", "synth", "summarize", "self-test"))
    parser.add_argument("--side", choices=("before", "after"), default="after")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "runs.jsonl")
    args = parser.parse_args()

    if args.command == "self-test":
        _self_test()
        return
    if args.command == "synth":
        synthesize(_scenario_texts())
        print(f"clips in {AUDIO_DIR}")
        return
    if args.command == "summarize":
        summary = summarize(_load_rows(args.out))
        summary_path = args.out.with_name("summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_csv(args.out.with_name("summary.csv"), summary)
        print(summary_path)
        return
    if args.command == "run":
        _run_jobs(_jobs(args.side, args.reps), args.out, args.side)
        return

    # all: synth, after on this tree, before on main, then summarize.
    synthesize(_scenario_texts())
    _run_jobs(_jobs("after", args.reps), args.out, "after")
    worktree = _ensure_main_worktree()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree / "python") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(
        [sys.executable, str(HERE / "bench_242.py"), "run", "--side", "before", "--reps", str(args.reps), "--out", str(args.out)],
        cwd=REPO,
        env=env,
    )
    summary = summarize(_load_rows(args.out))
    summary_path = args.out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.out.with_name("summary.csv"), summary)
    print(summary_path)


if __name__ == "__main__":
    main()
