"""CI foundation for voice latency and false-cut regressions.

The live replay harness under ``benchmarks/voice`` speaks real audio at a
provider and is not run in CI. This module replays the text half of that
corpus: scripted transcripts, the lexical turn detector, and a local
``TestModel``. A run fails when silence-to-first-audio p95 or
end-of-utterance-to-first-audio p95 exceeds the ceiling, or when a scripted
turn is cut that the corpus says must not be cut (and the reverse, a real
barge-in that no longer cuts).

False cut here means one of:

* the assistant starts speaking before the scripted utterance has finished
  (a false end of turn — the trailing-conjunction case in the replay corpus)
* a noise caption or a filled pause interrupts the reply
* a scripted barge-in does not interrupt

Ceilings are opt-in arguments. The committed corpus in the test suite passes
them. Nothing in a normal agent call turns this on.

This module is the CI replay, not a supported import. Gate traces you already
have with :func:`timbal.voice.latency.gate_silence_to_first_audio`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from timbal import Agent
from timbal.core.test_model import TestModel

from .eou import PunctuationEouPredictor
from .events import (
    AudioOutput,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    TranscriptCommitted,
    VoiceSessionEvent,
)
from .latency import gate_silence_to_first_audio, percentile, summarize_turn_metrics
from .metrics import TurnMetrics
from .providers import AudioInputConfig, AudioOutputConfig, SpeechToText, TextToSpeech, TranscriptEvent
from .session import VoiceSession
from .turn_detection import LexicalTurnDetector

# Replies are short on purpose: the suite measures the pipeline, not synthesis.
_REPLIES = ("Sure, I can help with that.", "Okay.", "One moment.")


@dataclass
class ReplayObservation:
    case_id: str
    user_turns: list[str] = field(default_factory=list)
    interrupted: bool = False
    false_cuts: list[str] = field(default_factory=list)
    metrics: list[TurnMetrics] = field(default_factory=list)
    error: str | None = None

    @property
    def eou_ms(self) -> list[float]:
        return [turn.eou_to_first_audio_ms for turn in self.metrics if turn.eou_to_first_audio_ms is not None]


def false_cuts_for(case: Mapping[str, Any], observation: ReplayObservation) -> list[str]:
    """Compare one replay to the corpus. Does not include latency."""
    if observation.error:
        return [f"{observation.case_id}: {observation.error}"]
    cuts = [f"{observation.case_id}: {cut}" for cut in observation.false_cuts]
    expected = case.get("expected_user_turns")
    if expected is not None and len(observation.user_turns) != expected:
        cuts.append(
            f"{observation.case_id}: expected {expected} user turn(s), "
            f"got {len(observation.user_turns)}: {observation.user_turns}"
        )
    for fragment in case.get("user_text_contains") or []:
        joined = " ".join(observation.user_turns)
        if fragment not in joined:
            cuts.append(f"{observation.case_id}: user text missing {fragment!r}: {observation.user_turns}")
    expect_interrupted = case.get("expect_interrupted")
    if expect_interrupted is True and not observation.interrupted:
        cuts.append(f"{observation.case_id}: expected a barge-in to cut the reply")
    if expect_interrupted is False and observation.interrupted:
        cuts.append(f"{observation.case_id}: false cut: the reply was interrupted")
    return cuts


def gate_voice_regression(
    observations: list[ReplayObservation],
    cases: list[Mapping[str, Any]],
    *,
    p95_eou_to_first_audio_ms: float,
    p95_silence_to_first_audio_ms: float,
    min_eou_samples: int,
) -> list[str]:
    """Failures for false cuts and for p95 latency. Empty means the corpus held."""
    by_id = {case["id"]: case for case in cases}
    failures: list[str] = []
    metrics: list[TurnMetrics] = []
    eou: list[float] = []
    for observation in observations:
        case = by_id.get(observation.case_id, {})
        failures.extend(false_cuts_for(case, observation))
        metrics.extend(observation.metrics)
        eou.extend(observation.eou_ms)
    if len(eou) < min_eou_samples:
        failures.append(f"end of utterance to first audio has {len(eou)} samples, need {min_eou_samples}")
    else:
        p95 = percentile(eou, 95)
        if p95 is not None and p95 > p95_eou_to_first_audio_ms:
            failures.append(
                f"end of utterance to first audio p95 {p95:.1f}ms is above {p95_eou_to_first_audio_ms:.1f}ms"
            )
    report = summarize_turn_metrics(metrics)
    failures.extend(gate_silence_to_first_audio(report, p95_ms=p95_silence_to_first_audio_ms))
    return failures


async def replay_cases(cases: list[Mapping[str, Any]], *, timeout_secs: float = 8.0) -> list[ReplayObservation]:
    """Run each scripted case through a fresh session. One failure does not stop the rest."""
    observations: list[ReplayObservation] = []
    for case in cases:
        try:
            observations.append(await asyncio.wait_for(replay_case(case), timeout=timeout_secs))
        except Exception as exc:  # noqa: BLE001 — one case must not abort the rest of the corpus
            observations.append(ReplayObservation(case_id=str(case.get("id")), error=f"{type(exc).__name__}: {exc}"))
    return observations


async def replay_case(case: Mapping[str, Any]) -> ReplayObservation:
    case_id = str(case["id"])
    stt = _ScriptedSTT()
    agent = Agent(name="voice_regression", model=TestModel(responses=list(_REPLIES)), tools=[])
    # Punctuation scorer, not Namo: the suite must run without timbal[voice] and without a model download.
    session = VoiceSession(
        agent=agent,
        stt=stt,
        tts=_InstantTTS(),
        turn_detector=LexicalTurnDetector(text_eou=PunctuationEouPredictor()),
    )
    events: list[VoiceSessionEvent] = []
    false_cuts: list[str] = []
    # close() interrupts a reply whose playback clock has not caught up. The
    # live harness ignores that teardown interrupt; so does this replay.
    closing = False

    async def drive() -> None:
        nonlocal closing
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not any(isinstance(event, SessionStarted) for event in events):
            await asyncio.sleep(0.01)
        if not any(isinstance(event, SessionStarted) for event in events):
            raise TimeoutError("session did not start")
        for step in case.get("steps") or []:
            op = step["op"]
            if step.get("forbid_audio_yet") and any(isinstance(event, AudioOutput) for event in events):
                false_cuts.append("false end of turn: audio started before the utterance finished")
            if op == "partial":
                await stt.inject(TranscriptEvent(type="partial", text=step["text"]))
            elif op == "commit":
                await stt.inject(TranscriptEvent(type="committed", text=step["text"]))
            elif op == "sleep":
                await asyncio.sleep(float(step["seconds"]))
            elif op == "wait_audio":
                limit = time.monotonic() + float(step.get("timeout", 2))
                while time.monotonic() < limit and not any(isinstance(event, AudioOutput) for event in events):
                    await asyncio.sleep(0.01)
                if not any(isinstance(event, AudioOutput) for event in events):
                    raise TimeoutError("no assistant audio")
            else:
                raise ValueError(f"unknown step {op}")
        closing = True
        await stt.finish()

    async def _empty() -> AsyncIterator[bytes]:
        if False:
            yield b""

    driver = asyncio.create_task(drive())
    interrupted = False
    try:
        async for event in session.run(_empty()):
            events.append(event)
            if isinstance(event, SessionInterrupted) and not closing:
                interrupted = True
            if isinstance(event, SessionError):
                false_cuts.append(f"session error: {event.message}")
        await driver
    except Exception:
        if not driver.done():
            driver.cancel()
        raise
    user_turns = _user_turns(events)
    return ReplayObservation(
        case_id=case_id,
        user_turns=user_turns,
        interrupted=interrupted,
        false_cuts=false_cuts,
        metrics=list(session.metrics),
    )


def _user_turns(events: list[VoiceSessionEvent]) -> list[str]:
    turns: list[str] = []
    for event in events:
        if not isinstance(event, TranscriptCommitted):
            continue
        if event.replace and turns:
            turns[-1] = event.text
        else:
            turns.append(event.text)
    return turns


class _ScriptedSTT(SpeechToText):
    """Transcripts injected by the driver, in order."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

    async def connect(self, config: AudioInputConfig) -> None:
        del config

    async def push_audio(self, chunk: bytes) -> None:
        del chunk

    async def commit(self) -> None:
        return None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.text:
                yield item

    async def close(self) -> None:
        return None

    async def inject(self, event: TranscriptEvent) -> None:
        await self._queue.put(event)

    async def finish(self) -> None:
        await self._queue.put(None)


class _InstantTTS(TextToSpeech):
    """First byte immediately, then a short hold so a later commit still lands mid-reply.

    End-of-utterance to first audio stays the time to the first chunk. The hold
    keeps ``assistant_active`` true long enough for a noise caption, a filled
    pause, or a real barge-in to be decided against a reply that is still going.
    """

    def __init__(self) -> None:
        self._chunk = b"\x00\x01" * 16

    async def connect(self, config: AudioOutputConfig) -> None:
        del config

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        del text
        yield self._chunk
        await asyncio.sleep(0.45)
        yield self._chunk

    async def close(self) -> None:
        return None
