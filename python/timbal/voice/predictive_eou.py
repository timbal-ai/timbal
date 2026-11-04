"""Predict end of turn while the caller is still speaking.

The default endpointer scores audio only after Silero has already collected
:data:`DEFAULT_STOP_SILENCE_SECS` of silence, then waits a further floor
(:data:`DEFAULT_MIN_DELAY_SECS` when the score is confident). This module
moves the scoring earlier and, when every signal already agrees the turn is
finished, confirms on a shorter silence.

What was copied, and what was not:

* **LiveKit** (``audio_recognition``): a turn model on the transcript,
  including interim results, chooses a short versus a long wait after speech
  ends. Interim text never commits a turn by itself. We score each partial
  as it arrives (Namo when installed, otherwise the lexical baseline) and
  keep that probability for the silence decision.
* **AssemblyAI Universal Streaming**: after ``min_turn_silence`` (160 ms in
  their aggressive preset) a confident semantic score ends the turn; otherwise
  silence runs out to ``max_turn_silence``. :data:`EAGER_SILENCE_SECS` is that
  short confirmation. A low score keeps today's floor.
* **Pipecat Smart Turn v3**: the checkpoint is trained on audio that ends at a
  VAD pause, not mid-phoneme. We still run it on a rolling window during
  speech (so a score is in flight), and the decision uses a fresh score kicked
  off at the first silent frames — the earliest window the model was trained
  for. A rolling score alone cannot authorize an early commit.
* **Deepgram Flux**: ``EagerEndOfTurn`` / ``TurnResumed`` are a provider
  prediction with a confidence, not a commit. ``TurnResumed`` vetoes the short
  wait. Neither event starts the assistant's reply.
* **ElevenLabs ``turn_v3``**: the target shape — decide from words, prosody,
  and a short pause together, instead of a fixed silence. Their model is not
  in this process. Smart Turn plus the text score plus Silero is the local
  stand-in, and it still confirms with :data:`EAGER_SILENCE_SECS` of silence.

Off unless a session sets ``predict_during_speech``. Nothing here starts an
agent turn; the endpointer still force-commits the STT, and the turn detector
can still hold an unfinished transcript.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from .eou import PunctuationEouPredictor
from .turn_detection import _ends_with_correction_marker, _is_hesitation_only

logger = structlog.get_logger("timbal.voice.predictive_eou")

# Same numbers as VadEndpointer. Duplicated so this module does not import
# endpointing (that module imports us). Tests assert the two stay equal.
DEFAULT_STOP_SILENCE_SECS = 0.20
DEFAULT_MIN_DELAY_SECS = 0.45
DEFAULT_MAX_DELAY_SECS = 3.0
DEFAULT_DELAY_CURVE = 2.0

# AssemblyAI's aggressive min_turn_silence, and the measured ElevenLabs
# turn_v3 decision (~0.16 s). The short path never confirms faster than this.
EAGER_SILENCE_SECS = 0.16
# Two 32 ms VAD frames: start the audio model as the pause begins, overlapping
# its ~50–100 ms inference with the confirmation silence.
PREDICT_SILENCE_SECS = 0.064
# How often to score the rolling window while speech is still active.
ROLL_INTERVAL_SECS = 0.32

# Smart Turn's own complete cut. Below this the pause is not "done".
AUDIO_COMPLETE = 0.5
# Effective text score (Namo, rescued by terminal punctuation). The lexical
# gate below is stricter: neutral unpunctuated text (~0.60) must not pass.
TEXT_COMPLETE = 0.5
LEXICAL_CONFIDENT = 0.9
FLUX_COMPLETE = 0.5

# Replay prior from benchmarks/voice/scenario.py. Smart Turn v3 scores the
# fluent mid-sentence cut "I want to return an item I bought" at 0.019, and
# the same words as their own sentence at 0.986. Intra-utterance pauses use
# the cut; finished turns use a confident complete. This is the script's own
# measurement applied as a prior, not a re-inference of the clips.
FLUENT_CUT_AUDIO_P = 0.019
FINISHED_AUDIO_P = 0.95

ScoreAudio = Callable[[], Awaitable[float | None]]
ScoreText = Callable[[str], Awaitable[float | None]]


def endpoint_curve(
    p: float,
    *,
    min_delay: float = DEFAULT_MIN_DELAY_SECS,
    max_delay: float = DEFAULT_MAX_DELAY_SECS,
    curve: float = DEFAULT_DELAY_CURVE,
) -> float:
    """Same mapping as :func:`timbal.voice.endpointing.endpointing_delay`."""
    p = min(1.0, max(0.0, p))
    return min_delay + (max_delay - min_delay) * (1.0 - p) ** curve


def baseline_decision_secs(p: float) -> float:
    """Silence from speech-end to commit when prediction does not fire.

    Stop-silence, then the curve. Inference overlaps the curve's floor, so it
    is not added again — matching :class:`~timbal.voice.endpointing.VadEndpointer`.
    """
    return DEFAULT_STOP_SILENCE_SECS + endpoint_curve(p)


@dataclass(frozen=True)
class EouSignals:
    """One snapshot of the signals gathered while the caller was speaking."""

    p_audio: float | None = None
    audio_fresh: bool = False
    """Pause-onset Smart Turn score, not the rolling mid-speech one."""
    p_text: float | None = None
    text_fresh: bool = False
    p_rolling: float | None = None
    p_flux: float | None = None
    flux_state: Literal["none", "eager", "resumed"] = "none"
    text_veto: bool = False
    """Lexical hedge, dangling token, or a trailing correction marker."""


def prediction_is_complete(signals: EouSignals) -> bool:
    """Whether the short confirmation may replace the silence floor.

    Every present signal has to agree. A missing audio score does not agree:
    text alone is how a punctuated fragment of an unfinished sentence gets cut.
    Flux ``TurnResumed`` disagrees. The rolling mid-speech score is not a vote.
    """
    if signals.text_veto or not signals.text_fresh or signals.p_text is None:
        return False
    if signals.p_text < TEXT_COMPLETE:
        return False
    if not signals.audio_fresh or signals.p_audio is None or signals.p_audio < AUDIO_COMPLETE:
        return False
    if signals.flux_state == "resumed":
        return False
    if signals.flux_state == "eager" and signals.p_flux is not None and signals.p_flux < FLUX_COMPLETE:
        return False
    return True


_LEXICAL = PunctuationEouPredictor()


async def text_blocks_early(text: str) -> bool:
    """True when the words themselves forbid the short wait.

    Terminal punctuation is the only lexical score that passes
    (:data:`LEXICAL_CONFIDENT`). Hedges, dangling tokens, trailing-off
    ellipses, and self-corrections ("… Sorry.") stay on the normal floor.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _is_hesitation_only(stripped) or _ends_with_correction_marker(stripped):
        return True
    return await _LEXICAL.predict_eou(stripped) < LEXICAL_CONFIDENT


class SpeechEouPredictor:
    """Caches text, audio, and Flux scores gathered before the turn is confirmed.

    ``score_audio`` reads the detector's rolling PCM window (Smart Turn).
    ``score_text`` scores one partial. Both are optional in the sense that a
    ``None`` return simply withholds that vote. The object never commits.
    """

    def __init__(
        self,
        score_audio: ScoreAudio,
        score_text: ScoreText,
        *,
        eager_silence_secs: float = EAGER_SILENCE_SECS,
        predict_silence_secs: float = PREDICT_SILENCE_SECS,
        roll_interval_secs: float = ROLL_INTERVAL_SECS,
        score_timeout_secs: float = 1.0,
    ) -> None:
        self._score_audio = score_audio
        self._score_text = score_text
        self.eager_silence_secs = eager_silence_secs
        self.predict_silence_secs = predict_silence_secs
        self.roll_interval_secs = roll_interval_secs
        self.score_timeout_secs = score_timeout_secs

        self._closed = False
        self._pause_gen = 0
        self._audio_gen = 0
        self._audio_kicked_gen = -1
        self._audio_gen_scored = -1
        self._p_audio: float | None = None
        self._audio_task: asyncio.Task[None] | None = None

        self._text = ""
        self._text_gen = 0
        self._text_gen_scored = -1
        self._scored_text = ""
        self._p_text: float | None = None
        self._text_task: asyncio.Task[None] | None = None

        self._p_rolling: float | None = None
        self._last_roll = 0.0
        self._roll_task: asyncio.Task[None] | None = None

        self._flux_state: Literal["none", "eager", "resumed"] = "none"
        self._p_flux: float | None = None

    async def close(self) -> None:
        self._closed = True
        for task in (self._audio_task, self._text_task, self._roll_task):
            if task is not None and not task.done():
                task.cancel()
        self._audio_task = None
        self._text_task = None
        self._roll_task = None

    def note_partial(self, text: str) -> None:
        """Score a partial transcript. The latest text wins."""
        if self._closed:
            return
        text = text.strip()
        if not text or text == self._text:
            return
        self._text = text
        self._text_gen += 1
        gen = self._text_gen
        self._text_task = asyncio.create_task(self._run_text(gen, text))

    def note_flux(self, event: Literal["eager", "resumed"], text: str, confidence: float | None) -> None:
        """Record a Flux eager prediction or its cancellation.

        ``resumed`` vetoes the short wait. ``eager`` is one more vote and, when
        it carries text, refreshes the partial. It does not commit.
        """
        if self._closed:
            return
        if event == "resumed":
            self._flux_state = "resumed"
            self._p_flux = confidence
            return
        self._flux_state = "eager"
        self._p_flux = confidence
        if text.strip():
            self.note_partial(text)

    def note_speech(self) -> None:
        """Rolling audio score while Silero still hears speech. Not a vote."""
        if self._closed:
            return
        now = time.monotonic()
        if self._roll_task is not None and not self._roll_task.done():
            return
        if self._last_roll and now - self._last_roll < self.roll_interval_secs:
            return
        self._last_roll = now
        self._roll_task = asyncio.create_task(self._run_roll())

    def kick_audio(self) -> None:
        """Score the window that ends at this pause. Once per pause."""
        if self._closed:
            return
        if self._audio_kicked_gen == self._pause_gen:
            return
        self._audio_kicked_gen = self._pause_gen
        self._audio_gen += 1
        gen = self._audio_gen
        pause = self._pause_gen
        self._audio_task = asyncio.create_task(self._run_audio(gen, pause))

    def on_speech_resumed(self) -> None:
        """The pause was not a turn. Drop the pause-onset score and any Flux flag.

        ``TurnResumed`` vetoes the pause it cancelled. It must not stick to the
        next pause, or one continued sentence would disable the short wait for
        the rest of the call. The in-flight endpoint is cancelled by the VAD
        at the same time, so clearing the flag here cannot authorize that pause.
        """
        self._pause_gen += 1
        self._audio_gen += 1
        task = self._audio_task
        self._audio_task = None
        if task is not None and not task.done():
            task.cancel()
        self._flux_state = "none"
        self._p_flux = None

    async def signals(self) -> EouSignals:
        """Wait briefly for the in-flight scores and return the snapshot."""
        await self._await_task(self._audio_task)
        await self._await_task(self._text_task)
        veto = await text_blocks_early(self._text) if self._text else True
        text_fresh = (
            self._text_gen_scored == self._text_gen and self._scored_text == self._text and self._p_text is not None
        )
        return EouSignals(
            p_audio=self._p_audio,
            audio_fresh=self._audio_gen_scored == self._audio_gen and self._p_audio is not None,
            p_text=self._p_text,
            text_fresh=text_fresh,
            p_rolling=self._p_rolling,
            p_flux=self._p_flux,
            flux_state=self._flux_state,
            text_veto=veto,
        )

    async def _await_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), self.score_timeout_secs)
        except TimeoutError:
            return

    async def _run_audio(self, gen: int, pause: int) -> None:
        p = await self._call_audio()
        if self._closed or gen != self._audio_gen or pause != self._pause_gen:
            return
        self._p_audio = p
        self._audio_gen_scored = gen

    async def _run_roll(self) -> None:
        p = await self._call_audio()
        if self._closed:
            return
        self._p_rolling = p

    async def _run_text(self, gen: int, text: str) -> None:
        try:
            p = await self._score_text(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("predictive_text_score_failed", error=str(exc))
            p = None
        if self._closed or gen != self._text_gen:
            return
        self._p_text = p
        self._scored_text = text
        self._text_gen_scored = gen

    async def _call_audio(self) -> float | None:
        try:
            return await self._score_audio()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("predictive_audio_score_failed", error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Replay of the voice scenario scripts (no audio, no network)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptPause:
    """One user span in a replay script.

    ``continues`` is an intra-utterance pause: committing before ``gap_secs``
    elapses is a false cut. A turn end has ``continues`` false and no gap.
    """

    text: str
    continues: bool
    gap_secs: float | None = None


@dataclass(frozen=True)
class ReplayReport:
    """Policy replay over script pauses. Latencies are silence-to-decision."""

    turns: int
    intra: int
    early_commits: int
    false_cuts: int
    baseline_false_cuts: int
    text_only_false_cuts: int
    qualified_turn_ends: int
    median_baseline_ms: float
    median_predictive_ms: float
    median_qualified_baseline_ms: float
    median_qualified_predictive_ms: float


def _step_kind(step: Any) -> str:
    if hasattr(step, "text") and hasattr(step, "part"):
        return "say"
    if hasattr(step, "secs"):
        return "silence"
    return "boundary"


def script_pauses(script: Sequence[Any]) -> list[ScriptPause]:
    """Intra-utterance pauses and turn ends from a replay script.

    A ``Say`` followed by silence and another slice of the same fluent
    utterance is a pause inside the turn. Anything else the speaker says is
    a turn end — the partial at that point is the whole utterance so far,
    which is what streaming STT actually delivers.
    """
    kinds = [_step_kind(step) for step in script]
    pauses: list[ScriptPause] = []
    parts: list[str] = []
    utterance: str | None = None
    for index, step in enumerate(script):
        if kinds[index] != "say":
            continue
        say_utterance = getattr(step, "utterance", None)
        if say_utterance != utterance:
            utterance = say_utterance
            parts = []
        parts.append(step.text)
        text = " ".join(parts)
        nxt = next((j for j in range(index + 1, len(script)) if kinds[j] == "say"), None)
        intra = False
        gap = 0.0
        if nxt is not None and say_utterance is not None and getattr(script[nxt], "utterance", None) == say_utterance:
            between = kinds[index + 1 : nxt]
            if between and all(kind == "silence" for kind in between):
                intra = True
                gap = sum(script[j].secs for j in range(index + 1, nxt))
        if intra:
            pauses.append(ScriptPause(text=text, continues=True, gap_secs=gap))
        else:
            pauses.append(ScriptPause(text=text, continues=False))
    return pauses


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


async def evaluate_replay(pauses: Sequence[ScriptPause]) -> ReplayReport:
    """Decision latency and false cuts for the predictive policy on these pauses.

    Audio is the script prior (:data:`FLUENT_CUT_AUDIO_P` inside a fluent
    pause, :data:`FINISHED_AUDIO_P` at a turn end). Text is the lexical
    baseline, the same gate the live short path uses. ``text_only_false_cuts``
    is the counterfactual that skips the audio vote.
    """
    baseline_ms: list[float] = []
    predictive_ms: list[float] = []
    qualified_base: list[float] = []
    qualified_pred: list[float] = []
    early_commits = 0
    false_cuts = 0
    baseline_false_cuts = 0
    text_only_false_cuts = 0
    intra = 0
    qualified = 0
    for pause in pauses:
        veto = await text_blocks_early(pause.text)
        p_text = await _LEXICAL.predict_eou(pause.text)
        p_audio = FLUENT_CUT_AUDIO_P if pause.continues else FINISHED_AUDIO_P
        early = prediction_is_complete(
            EouSignals(
                p_audio=p_audio,
                audio_fresh=True,
                p_text=p_text,
                text_fresh=True,
                text_veto=veto,
            )
        )
        text_only_early = not veto and p_text >= TEXT_COMPLETE
        base = baseline_decision_secs(p_audio)
        pred = EAGER_SILENCE_SECS if early else base
        if pause.continues:
            intra += 1
            gap = pause.gap_secs if pause.gap_secs is not None else 0.0
            if early and pred < gap:
                false_cuts += 1
            if base < gap:
                baseline_false_cuts += 1
            if text_only_early and EAGER_SILENCE_SECS < gap:
                text_only_false_cuts += 1
            continue
        baseline_ms.append(base * 1000.0)
        predictive_ms.append(pred * 1000.0)
        if early:
            early_commits += 1
            qualified += 1
            qualified_base.append(base * 1000.0)
            qualified_pred.append(pred * 1000.0)
    return ReplayReport(
        turns=len(pauses) - intra,
        intra=intra,
        early_commits=early_commits,
        false_cuts=false_cuts,
        baseline_false_cuts=baseline_false_cuts,
        text_only_false_cuts=text_only_false_cuts,
        qualified_turn_ends=qualified,
        median_baseline_ms=_median(baseline_ms),
        median_predictive_ms=_median(predictive_ms),
        median_qualified_baseline_ms=_median(qualified_base),
        median_qualified_predictive_ms=_median(qualified_pred),
    )
