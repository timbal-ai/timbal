"""Bounded dictation end-of-turn probe. Not a percentile benchmark.

Speech-end to Flux commit, then first token, then first audio. Names, digits,
and a hesitation are in the clips so a premature cut shows up as a missing
word in the committed transcript. ``entity_patience`` off and on use the same
clips. n below 20 does not get a p90 or a p99.

A separate call enables speculation. Flux emits ``EagerEndOfTurn`` only when
``eager_eot_threshold`` is on the socket; a direct session now sets that.
The call pauses and continues so a ``TurnResumed`` can discard the attempt.
Zero speculative starts is recorded as not exercised.

One extra clip is mixed with pink noise at 8 dB SNR. That is one sample.
PCM energy is not a transcript.

    TIMBAL_VOICE_DICTATION=1 uv run python benchmarks/voice/dictation_eot.py

Missing ``OPENAI_API_KEY``, ``DEEPGRAM_API_KEY``, or ``ELEVENLABS_API_KEY``
exits 2 and names the key. Groq and Anthropic are not called.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import accept_duplex as gate
from degrade import active_rms, noise_bed
from synth import SAMPLE_RATE, silence, synthesize_clips
from timbal.voice import (
    AudioInputConfig,
    AudioOutputConfig,
    SessionStarted,
    SpeculativeGeneration,
    VoiceSession,
    resolve_stt,
)
from timbal.voice.elevenlabs import ElevenLabsStreamTTS

FLAG = "TIMBAL_VOICE_DICTATION"
REPEATS = 2
ARTIFACT_DIR = Path("/opt/cursor/artifacts/voice-dictation")
PROMPT = (
    "You are a phone agent. When they dictate a number or spell a name, wait until they "
    "finish and read the whole thing back once. If they ask the time, say you have no clock. "
    "One short spoken sentence."
)
CLIPS = {
    "control": "What time is it?",
    "digits": "My number is four one five, um, five five five, oh one nine nine.",
    "name": "The name is J, O, N, um, E, S.",
}
EXPECTED = {
    "control": ("time",),
    "digits": ("four", "one", "five", "five", "five", "oh", "one", "nine", "nine"),
    "name": ("j", "o", "n", "e", "s"),
}
RESUME = ("My account number is", "four one five, actually nine.")


def summarize_stage(samples: list[float]) -> dict[str, Any]:
    """Min, median, max. Percentiles only once there are 20 samples."""
    ordered = sorted(samples)
    n = len(ordered)
    if n == 0:
        return {"n": 0, "min": None, "median": None, "max": None, "p90": None, "p99": None, "note": "no samples"}
    mid = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    row: dict[str, Any] = {
        "n": n,
        "min": round(ordered[0], 1),
        "median": round(mid, 1),
        "max": round(ordered[-1], 1),
        "p90": None,
        "p99": None,
    }
    if n >= 20:
        row["p90"] = round(ordered[min(n - 1, int(0.9 * (n - 1)))], 1)
        row["p99"] = round(ordered[min(n - 1, int(0.99 * (n - 1)))], 1)
        row["note"] = "percentile from this sample only"
    else:
        row["note"] = f"n={n} is not a percentile tail"
    return row


def recall(kind: str, transcript: str) -> float:
    words = transcript.casefold()
    wanted = EXPECTED[kind]
    if not wanted:
        return 1.0
    return round(sum(1 for word in wanted if word in words) / len(wanted), 3)


def at_snr(pcm: bytes, snr_db: float) -> bytes:
    """Pink noise under the speech. Deterministic. Not a telephony codec."""
    import array

    speech = active_rms(pcm) or 1.0
    noise_rms = speech / (10 ** (snr_db / 20))
    bed = noise_bed(len(pcm) / (SAMPLE_RATE * 2) + 0.5, noise_rms, kind="pink", seed=8)
    usable = len(pcm) - (len(pcm) % 2)
    out = array.array("h")
    out.frombytes(pcm[:usable])
    other = array.array("h")
    other.frombytes(bed[:usable])
    for i in range(min(len(out), len(other))):
        mixed = out[i] + other[i]
        out[i] = 32767 if mixed > 32767 else (max(mixed, -32768))
    return out.tobytes()


def _build(name: str, *, patience: bool, speculative: bool) -> VoiceSession:
    return VoiceSession(
        agent=gate._agent(name, PROMPT, []),
        stt=resolve_stt("deepgram-flux"),
        tts=ElevenLabsStreamTTS(),
        audio_input=AudioInputConfig(language="en", sample_rate=SAMPLE_RATE),
        audio_output=AudioOutputConfig(model=gate.TTS_MODEL, voice=gate.ASSISTANT_VOICE_ID, sample_rate=SAMPLE_RATE),
        turn_detector="provider",
        ignore_backchannels=True,
        entity_patience=patience,
        speculative_generation=SpeculativeGeneration(enabled=speculative, preemptive_tts=speculative),
    )


async def _one_session(
    name: str,
    turns: list[tuple[str, bytes]],
    *,
    patience: bool,
    speculative: bool,
    out_dir: Path,
) -> dict[str, Any]:
    state = gate.CallState(name=name, t0=time.monotonic())
    feeder = gate.PacedFeeder()
    playback = gate.PlaybackSim()
    session = _build(name, patience=patience, speculative=speculative)
    state.stt_name = type(session.stt).__name__
    state.tts_name = type(session.tts).__name__

    async def ack_loop() -> None:
        while True:
            await asyncio.sleep(0.25)
            session.playback.on_playback_ack(playback.played_ms)

    async def drive() -> None:
        try:
            for label, pcm in turns:
                before = len(state.metrics)
                state.marks[f"{label}_say"] = time.monotonic()
                await gate._say(feeder, pcm)
                state.marks[f"{label}_silent"] = time.monotonic()
                await gate._wait(lambda before=before: len(state.metrics) > before, 25, label, state)
                await asyncio.sleep(0.3)
            state.script_finished = True
        finally:
            feeder.stop()
            if not session.closed:
                await session.close()

    driver: asyncio.Task[None] | None = None
    acker: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(90):
            from contextlib import aclosing

            async with aclosing(session.run(feeder.stream())) as stream:
                async for event in stream:
                    gate.observe(event, state, playback)
                    if isinstance(event, SessionStarted) and driver is None:
                        acker = asyncio.create_task(ack_loop())
                        driver = asyncio.create_task(drive())
    except TimeoutError:
        state.timeouts.append("session wall 90s")
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
    wav = out_dir / f"{name}.wav"
    gate.write_wav(wav, pcm)
    return {
        "name": name,
        "patience": patience,
        "speculative": speculative,
        "eager_eot_threshold": session.audio_input.extra.get("eager_eot_threshold"),
        "commits": state.commits,
        "metrics": state.metrics,
        "timeouts": state.timeouts,
        "errors": state.errors,
        "wav": str(wav),
        "speculative_started": max((m.get("speculative_generations_started") or 0) for m in state.metrics)
        if state.metrics
        else 0,
        "speculative_discarded": max((m.get("speculative_generations_discarded") or 0) for m in state.metrics)
        if state.metrics
        else 0,
    }


def _rows(runs: list[dict[str, Any]], *, patience: bool) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in CLIPS}
    for run in runs:
        if run["patience"] is not patience or run["speculative"]:
            continue
        metrics = list(run["metrics"])
        commits = [c["text"] for c in run["commits"]]
        for index, kind in enumerate(CLIPS):
            if index >= len(metrics):
                continue
            metric = metrics[index]
            text = commits[index] if index < len(commits) else ""
            grouped[kind].append(
                {
                    "speech_end_to_transcript_ms": metric.get("speech_end_to_transcript_ms"),
                    "eou_to_llm_first_token_ms": metric.get("eou_to_llm_first_token_ms"),
                    "eou_to_first_audio_ms": metric.get("eou_to_first_audio_ms"),
                    "recall": recall(kind, text),
                    "transcript": text,
                }
            )
    summary = {}
    for kind, rows in grouped.items():
        summary[kind] = {
            "speech_end_to_transcript_ms": summarize_stage(
                [row["speech_end_to_transcript_ms"] for row in rows if row["speech_end_to_transcript_ms"] is not None]
            ),
            "eou_to_llm_first_token_ms": summarize_stage(
                [row["eou_to_llm_first_token_ms"] for row in rows if row["eou_to_llm_first_token_ms"] is not None]
            ),
            "eou_to_first_audio_ms": summarize_stage(
                [row["eou_to_first_audio_ms"] for row in rows if row["eou_to_first_audio_ms"] is not None]
            ),
            "recall": rows,
        }
    return summary


async def run_dictation(out_dir: Path | None = None) -> dict[str, Any]:
    missing = gate.missing_provider_keys()
    if missing:
        raise RuntimeError(f"{FLAG}=1 but missing {', '.join(missing)}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = out_dir or (gate.RESULTS_DIR.parent / "dictation" / stamp)
    dest.mkdir(parents=True, exist_ok=True)
    clips = await synthesize_clips(list(CLIPS.values()) + list(RESUME))
    turns = [(kind, clips[text]) for kind, text in CLIPS.items()]
    runs: list[dict[str, Any]] = []
    for patience in (False, True):
        for rep in range(REPEATS):
            runs.append(
                await _one_session(
                    f"{'patience' if patience else 'baseline'}_{rep}",
                    turns,
                    patience=patience,
                    speculative=False,
                    out_dir=dest,
                )
            )
    resume_pcm = clips[RESUME[0]] + silence(0.7) + clips[RESUME[1]]
    speculation = await _one_session(
        "speculation_resume",
        [("resume", resume_pcm)],
        patience=False,
        speculative=True,
        out_dir=dest,
    )
    noisy = await _one_session(
        "digits_snr8",
        [("digits", at_snr(clips[CLIPS["digits"]], 8.0))],
        patience=False,
        speculative=False,
        out_dir=dest,
    )
    report = {
        "generated_at": stamp,
        "listening": False,
        "analysis": "Stage timings and committed transcript text. PCM energy is not words. n is not a p99.",
        "repeats": REPEATS,
        "baseline": _rows(runs, patience=False),
        "entity_patience": _rows(runs, patience=True),
        "speculation": {
            "eager_eot_threshold": speculation.get("eager_eot_threshold"),
            "started": speculation.get("speculative_started"),
            "discarded": speculation.get("speculative_discarded"),
            "exercised": bool(speculation.get("speculative_started")),
            "commits": speculation.get("commits"),
            "metrics": speculation.get("metrics"),
            "timeouts": speculation.get("timeouts"),
            "errors": speculation.get("errors"),
        },
        "noise_snr8_n1": {
            "note": "one pink-noise clip at 8 dB SNR, not a noise-robustness claim",
            "commits": noisy.get("commits"),
            "metrics": noisy.get("metrics"),
            "recall": recall("digits", " ".join(c["text"] for c in noisy.get("commits") or [])),
        },
        "runs": runs,
    }
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        target = ARTIFACT_DIR / stamp
        target.mkdir(parents=True, exist_ok=True)
        for path in dest.iterdir():
            if path.suffix in {".json", ".wav"}:
                shutil.copy2(path, target / path.name)
        report["artifact_dir"] = str(target)
        (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except OSError as exc:
        report["artifact_error"] = str(exc)
    return report


def main() -> int:
    if os.environ.get(FLAG) != "1":
        print(json.dumps({"skipped": f"set {FLAG}=1"}))  # noqa: T201
        return 0
    missing = gate.missing_provider_keys()
    if missing:
        print(json.dumps({"passed": False, "missing_keys": missing}))  # noqa: T201
        return 2
    report = asyncio.run(run_dictation())
    exercised = report["speculation"]["exercised"]
    summary = {
        "speculation_exercised": exercised,
        "speculation_started": report["speculation"]["started"],
        "speculation_discarded": report["speculation"]["discarded"],
        "noise_recall": report["noise_snr8_n1"]["recall"],
        "baseline_digits": report["baseline"]["digits"]["speech_end_to_transcript_ms"],
        "patience_digits": report["entity_patience"]["digits"]["speech_end_to_transcript_ms"],
        "artifact_dir": report.get("artifact_dir"),
    }
    print(json.dumps(summary, indent=2))  # noqa: T201
    return 0 if exercised else 3


if __name__ == "__main__":
    raise SystemExit(main())
