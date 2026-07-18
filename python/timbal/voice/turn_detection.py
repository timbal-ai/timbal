"""Turn detection for :class:`~timbal.voice.VoiceSession`.

A :class:`TurnDetector` decides how the session reacts to STT output:

* ``on_partial`` — should a live partial transcript barge in (interrupt the
  assistant) or be ignored?
* ``on_committed`` — new turn, continuation, hold (wait), or ignore?

Detectors are pure w.r.t. the session: they receive a :class:`TurnState`
snapshot and return a decision. ``push_audio`` is a no-op by default so
audio-consuming detectors (Smart Turn / LiveKit v1-mini style) can slot in
without session changes.

**Modes** (see :func:`resolve_turn_detector`):

* ``heuristic`` (default) — :class:`HeuristicTurnDetector`, no extra deps
* ``provider`` — :class:`ProviderTurnDetector`, trust STT/realtime endpointing
* ``local`` — :class:`LocalAudioTurnDetector` + injectable :class:`AudioEouModel`
* ``lexical`` — :class:`LexicalTurnDetector`, optional zero-dep text overlay

Provider-native paths (OpenAI ``semantic_vad``, ElevenLabs Scribe VAD commits,
Deepgram Flux, Gemini Live) map to ``provider`` or a future realtime session
that skips local detection entirely. We deliberately do **not** chase
deprecated text-classifier ONNX models.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import deque
from difflib import SequenceMatcher
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict

from .eou import (
    _DANGLING_TOKENS,
    _WORD_RE,
    AudioEouModel,
    PunctuationEouPredictor,
    TextEouPredictor,
)

if TYPE_CHECKING:
    from .session import AudioInputConfig

logger = structlog.get_logger("timbal.voice.turn_detection")


class TurnState(BaseModel):
    """Snapshot the session hands the detector — keeps detectors pure and testable."""

    model_config = ConfigDict(extra="forbid")

    assistant_active: bool
    """Agent is generating OR audio is still playing client-side.

    Must not include a pending HOLD — that is :attr:`holding`. Folding HOLD into
    this flag breaks silence-hallucination filtering and mid-hold refinements.
    """
    audio_playing: bool
    """Client likely still has queued TTS audio to play."""
    assistant_text: str
    """Assistant text accumulated so far this turn."""
    active_user_text: str
    """User text of the in-flight turn, or the pending HOLD fragment when holding."""
    seconds_since_turn_start: float
    seconds_since_last_commit: float
    partials_since_last_commit: int
    """Partial transcripts seen since the previous committed transcript."""
    holding: bool = False
    """Session is debouncing an incomplete commit (:attr:`CommitAction.HOLD`)."""


class PartialDecision(StrEnum):
    IGNORE = "ignore"
    BARGE_IN = "barge_in"


class CommitAction(StrEnum):
    IGNORE = "ignore"
    NEW_TURN = "new_turn"
    CONTINUE_TURN = "continue"
    HOLD = "hold"
    """Incomplete utterance — don't start the agent yet; wait for more speech
    or the session's hold timeout (Pipecat/LiveKit endpointing pattern).
    Heuristic/provider detectors never emit this; only local/lexical opt-in."""


class CommitDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: CommitAction
    text: str
    """Final user text for the turn (merged text for ``CONTINUE_TURN`` / ``HOLD``)."""
    reason: str
    """Why, for debug logs ("echo", "refinement", "continuation", "hold", ...)."""
    hold_timeout_secs: float | None = None
    """Optional per-decision override for how long the session may HOLD."""


class TurnDetector(ABC):
    """Pluggable turn-taking policy.

    Lifecycle: ``start`` → ``on_partial`` / ``on_committed`` (and optionally
    ``push_audio``) → ``close``. ``start``, ``close``, and ``push_audio`` are
    no-ops by default so text-only detectors stay minimal.
    """

    async def start(self, config: AudioInputConfig) -> None:  # noqa: B027
        """Called once when the session starts, before any decisions."""

    async def close(self) -> None:  # noqa: B027
        """Called on session cleanup."""

    def push_audio(self, chunk: bytes) -> None:  # noqa: B027
        """Raw mic PCM, for detectors that run local VAD / audio models."""

    @abstractmethod
    async def on_partial(self, text: str, state: TurnState) -> PartialDecision: ...

    @abstractmethod
    async def on_committed(self, text: str, state: TurnState) -> CommitDecision: ...


# ---------------------------------------------------------------------------
# Heuristics (default implementation)
# ---------------------------------------------------------------------------

_NOISE_PATTERN = re.compile(r"^\s*\(.*\)\s*$")
# Word + closing paren only, no leading "(" — e.g. Scribe hallucinating "(music playing)" as "Music)"
_STT_ORPHAN_CLOSE_CAPTION = re.compile(r"^[A-Za-z][A-Za-z\s'-]{0,48}\)\s*$")
# Opens "(" but no ")" on this commit (split caption or noise)
_STT_INCOMPLETE_OPEN_PAREN = re.compile(r"^\([^)]*$")


def _is_noise(text: str) -> bool:
    """Filter STT artifacts like '(wind blowing)', '(silence)', '(music)', etc."""
    return bool(_NOISE_PATTERN.match(text))


def _is_garbage_commit(text: str) -> bool:
    """Filter silence/noise hallucinations: lone punctuation, broken captions, etc.

    Realtime models often emit parenthetical *sound* labels; on silence you can get
    fragments like ``(`` or ``Music)`` without the user speaking.
    """
    t = text.strip()
    if not t:
        return True
    # Single non-alphanumeric (e.g. "(" from a split "(music)")
    if len(t) == 1 and not t.isalnum():
        return True
    # Tiny all-punctuation runs
    if len(t) <= 3 and all(not c.isalnum() for c in t):
        return True
    if t in {")", "]", "}", "…"}:
        return True
    if _STT_ORPHAN_CLOSE_CAPTION.match(t):
        return True
    if _STT_INCOMPLETE_OPEN_PAREN.match(t):
        return True
    return False


def _normalize_echo(s: str) -> str:
    return " ".join(s.lower().split())


def _likely_stt_echo(committed: str, assistant_so_far: str) -> bool:
    """True if STT text is probably the assistant's own speech leaking into the mic."""
    c = _normalize_echo(committed)
    a = _normalize_echo(assistant_so_far)
    if not a:
        return False
    # Short leaks (punctuation / tail of TTS) often transcribe as 1–9 chars.
    if len(c) < 10:
        if len(c) >= 2 and c in a:
            return True
        return False
    if c in a:
        return True
    tail_len = min(len(a), max(len(c) * 3, 100))
    tail = a[-tail_len:]
    return SequenceMatcher(None, c, tail).ratio() >= 0.68


def _is_same_user_utterance_refinement(active: str, new: str) -> bool:
    """True if ``new`` is a longer/corrected transcript of the same user phrase.

    Realtime STT commonly emits multiple ``committed`` events per pause; the
    second is usually an extension, not intentional barge-in.
    """
    if active.strip() == new.strip():
        return True
    a, b = _normalize_echo(active), _normalize_echo(new)
    if not a or not b:
        return False
    if a == b:
        return True
    r = SequenceMatcher(None, a, b).ratio()
    # Near-duplicate punctuation / spacing (possibly seconds apart).
    if min(len(a), len(b)) >= 6 and r >= 0.84:
        return True
    if len(b) > len(a):
        if (len(a) >= 5 and b.startswith(a)) or (len(a) >= 8 and a in b) or r >= 0.74:
            return True
    elif len(a) > len(b) and len(b) >= 8:
        # Second pass shorter but same gist (rare).
        if a.startswith(b) or r >= 0.86:
            return True
    return False


def _looks_like_fresh_hold_utterance(text: str) -> bool:
    """True if ``text`` looks like its own utterance, not a VAD-split continuation.

    Used while HOLDing to avoid gluing ``stop`` / short new questions onto an
    incomplete held fragment. Function-word / lowercase multi-word starts
    (``the weather…``) stay eligible for merge.
    """
    n = text.strip()
    if not n:
        return False
    words = _WORD_RE.findall(n)
    if not words:
        return True
    first = words[0].lower()
    if first in _DANGLING_TOKENS:
        return False
    # Lowercase multi-word glue is usually the rest of the held phrase.
    if n[0].islower() and n[0].isalpha() and len(words) >= 2:
        return False
    return True


class HeuristicTurnDetector(TurnDetector):
    """Default detector: the regex/similarity heuristics tuned for ElevenLabs Scribe.

    Thresholds are class attributes so subclasses can tweak without copying logic.
    """

    MIN_BARGE_IN_PARTIAL_CHARS = 4
    HALLUCINATION_MIN_CHARS = 41
    EARLY_DUPLICATE_WINDOW_SECS = 1.5
    EARLY_DUPLICATE_MIN_CHARS = 5
    EARLY_DUPLICATE_RATIO = 0.58
    CONTINUATION_WINDOW_SECS = 3.0
    CONTINUATION_MAX_CHARS = 30

    async def on_partial(self, text: str, state: TurnState) -> PartialDecision:
        if not state.audio_playing or not text:
            return PartialDecision.IGNORE
        is_noise = _is_noise(text)
        is_echo = _likely_stt_echo(text, state.assistant_text) if not is_noise else False
        too_short = len(text) < self.MIN_BARGE_IN_PARTIAL_CHARS
        if not is_noise and not is_echo and not too_short:
            return PartialDecision.BARGE_IN
        logger.debug(
            "stt_partial_skipped",
            text_preview=text[:80],
            too_short=too_short,
            is_noise=is_noise,
            is_echo=is_echo,
            audio_playing=state.audio_playing,
        )
        return PartialDecision.IGNORE

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        if _is_noise(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="noise")
        if _is_garbage_commit(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="garbage")
        # A long commit with zero preceding partials while nothing is playing is
        # almost always an STT hallucination on silence. Uses assistant_active
        # only — pending HOLD must not flip that flag (see TurnState.holding).
        if (
            state.partials_since_last_commit == 0
            and len(text) >= self.HALLUCINATION_MIN_CHARS
            and not state.assistant_active
        ):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="hallucination")
        if state.assistant_active and _likely_stt_echo(text, state.assistant_text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="echo")
        # Pending HOLD: STT often re-commits a longer form of the same fragment.
        # Mid-turn "refinement" IGNORE would freeze the held text until timeout —
        # re-HOLD with the updated utterance instead (session trusts decision.text).
        # Do NOT glue every non-refinement onto the held fragment — a separate
        # utterance ("stop", a new question) must supersede the hold.
        if state.holding and state.active_user_text:
            if _is_same_user_utterance_refinement(state.active_user_text, text):
                better = text if len(text.strip()) >= len(state.active_user_text.strip()) else state.active_user_text
                return CommitDecision(action=CommitAction.HOLD, text=better, reason="hold_refinement")
            # Same gates as mid-turn VAD-split continuation.
            if (
                state.seconds_since_last_commit < self.CONTINUATION_WINDOW_SECS
                and len(text) < self.CONTINUATION_MAX_CHARS
            ):
                combined = state.active_user_text.rstrip(", ") + " " + text
                return CommitDecision(action=CommitAction.HOLD, text=combined, reason="hold_merge")
            return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="hold_supersede")
        if state.assistant_active and state.active_user_text:
            if _is_same_user_utterance_refinement(state.active_user_text, text):
                return CommitDecision(action=CommitAction.IGNORE, text=text, reason="refinement")
            # Very soon after turn start, VAD often double-fires near-identical commits.
            if state.seconds_since_turn_start < self.EARLY_DUPLICATE_WINDOW_SECS:
                a, b = _normalize_echo(state.active_user_text), _normalize_echo(text)
                if (
                    min(len(a), len(b)) >= self.EARLY_DUPLICATE_MIN_CHARS
                    and SequenceMatcher(None, a, b).ratio() >= self.EARLY_DUPLICATE_RATIO
                ):
                    return CommitDecision(action=CommitAction.IGNORE, text=text, reason="early_duplicate")
            # VAD split a single utterance into two fast commits (e.g. "Hola, ¿qué tal" + "estás?").
            # Combine and restart the turn instead of treating the fragment as a new query.
            if (
                state.seconds_since_last_commit < self.CONTINUATION_WINDOW_SECS
                and len(text) < self.CONTINUATION_MAX_CHARS
            ):
                combined = state.active_user_text.rstrip(", ") + " " + text
                return CommitDecision(action=CommitAction.CONTINUE_TURN, text=combined, reason="continuation")
        return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="new_turn")


class ProviderTurnDetector(TurnDetector):
    """Trust the STT / realtime provider's endpointing.

    Use when ElevenLabs Scribe VAD commits, Deepgram Flux, AssemblyAI, OpenAI
    ``semantic_vad``, or similar already decide turn boundaries. We only drop
    obvious noise/garbage (and assistant echo while speaking) — no heuristic
    continuation merging that fights the provider.

    Opt-in: ``resolve_turn_detector("provider")`` or
    ``VoiceSession(..., turn_detector=ProviderTurnDetector())``.
    """

    async def on_partial(self, text: str, state: TurnState) -> PartialDecision:
        # Provider already endpointed; partials are captions only unless clearly
        # a barge-in while audio is playing.
        if not state.audio_playing or not text:
            return PartialDecision.IGNORE
        if _is_noise(text) or _is_garbage_commit(text):
            return PartialDecision.IGNORE
        if _likely_stt_echo(text, state.assistant_text):
            return PartialDecision.IGNORE
        if len(text.strip()) < 4:
            return PartialDecision.IGNORE
        return PartialDecision.BARGE_IN

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        if _is_noise(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="noise")
        if _is_garbage_commit(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="garbage")
        if state.assistant_active and _likely_stt_echo(text, state.assistant_text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="echo")
        return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="provider")


class LexicalTurnDetector(HeuristicTurnDetector):
    """Heuristic + optional zero-dep text EOU (punctuation / dangling tokens).

    Not a substitute for audio Smart Turn. Useful when you want HOLD on
    obviously unfinished transcripts without installing ``timbal[voice]``.

    On a fresh (idle) commit that scores incomplete → :attr:`CommitAction.HOLD`.
    On a follow-up while holding / mid-turn with an incomplete active fragment →
    :attr:`CommitAction.CONTINUE_TURN`.
    """

    completion_threshold: float = 0.5
    CONTINUATION_WINDOW_SECS_LEXICAL = 6.0
    DEFAULT_HOLD_TIMEOUT_SECS = 1.5

    def __init__(self, text_eou: TextEouPredictor | None = None) -> None:
        self.text_eou = text_eou or PunctuationEouPredictor()

    async def start(self, config: AudioInputConfig) -> None:  # noqa: ARG002
        await self.text_eou.start()

    async def close(self) -> None:
        await self.text_eou.close()

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        decision = await super().on_committed(text, state)
        if decision.action is CommitAction.IGNORE:
            return decision
        if decision.action is CommitAction.CONTINUE_TURN:
            return decision

        candidate = decision.text or text

        # Parent glued a short non-refinement onto the held fragment (hold_merge).
        # If the new commit alone looks complete and isn't a function-word
        # continuation ("the weather…"), drop the held fragment — otherwise
        # "stop" / a short new question becomes garbled user text.
        if (
            state.holding
            and decision.action is CommitAction.HOLD
            and decision.reason == "hold_merge"
            and _looks_like_fresh_hold_utterance(text)
        ):
            p_new = await self.text_eou.predict_eou(text)
            if p_new >= self.completion_threshold:
                return CommitDecision(
                    action=CommitAction.NEW_TURN,
                    text=text,
                    reason="lexical_hold_supersede",
                )

        # Parent HOLD (hold_refinement / hold_merge) or an already-pending hold:
        # re-score the updated utterance — it may now look complete.
        if decision.action is CommitAction.HOLD or state.holding:
            p = await self.text_eou.predict_eou(candidate)
            if p < self.completion_threshold:
                return CommitDecision(
                    action=CommitAction.HOLD,
                    text=candidate,
                    reason="lexical_hold" if not state.holding else "lexical_hold_update",
                    hold_timeout_secs=self.DEFAULT_HOLD_TIMEOUT_SECS,
                )
            return CommitDecision(
                action=CommitAction.NEW_TURN,
                text=candidate,
                reason="lexical_hold_complete",
            )

        # Mid-turn incomplete fragment + new commit → merge (same as old semantic path).
        if state.assistant_active and state.active_user_text:
            if state.seconds_since_last_commit < self.CONTINUATION_WINDOW_SECS_LEXICAL:
                p_active = await self.text_eou.predict_eou(state.active_user_text)
                if p_active < self.completion_threshold:
                    combined = state.active_user_text.rstrip(", ") + " " + text
                    return CommitDecision(
                        action=CommitAction.CONTINUE_TURN,
                        text=combined,
                        reason="lexical_continuation",
                    )
            return decision

        # Idle: hold if this commit itself looks incomplete.
        p = await self.text_eou.predict_eou(candidate)
        if p < self.completion_threshold:
            return CommitDecision(
                action=CommitAction.HOLD,
                text=candidate,
                reason="lexical_hold",
                hold_timeout_secs=self.DEFAULT_HOLD_TIMEOUT_SECS,
            )
        return decision


# Back-compat alias from the short-lived Phase-1a name.
SemanticTurnDetector = LexicalTurnDetector


class LocalAudioTurnDetector(HeuristicTurnDetector):
    """Local audio EOU (Pipecat Smart Turn / LiveKit v1-mini shaped).

    Buffers mic PCM via :meth:`push_audio`. On each STT commit, after the usual
    noise/echo filters, scores the recent audio window with an injected
    :class:`~timbal.voice.eou.AudioEouModel`. Incomplete → :attr:`CommitAction.HOLD`
    (session debounce); complete → :attr:`CommitAction.NEW_TURN`.

    Without an ``audio_eou`` model this degrades to :class:`HeuristicTurnDetector`
    (no HOLD) so opting into ``"local"`` without ``timbal[voice]`` never breaks.
    """

    completion_threshold: float = 0.5
    DEFAULT_HOLD_TIMEOUT_SECS = 2.0
    AUDIO_WINDOW_SECS = 8.0

    def __init__(
        self,
        audio_eou: AudioEouModel | None = None,
        *,
        completion_threshold: float | None = None,
        hold_timeout_secs: float | None = None,
    ) -> None:
        self.audio_eou = audio_eou
        if completion_threshold is not None:
            self.completion_threshold = completion_threshold
        self.hold_timeout_secs = (
            hold_timeout_secs if hold_timeout_secs is not None else self.DEFAULT_HOLD_TIMEOUT_SECS
        )
        self._sample_rate = 16_000
        self._pcm: deque[bytes] = deque()
        self._pcm_bytes = 0
        self._max_pcm_bytes = 0

    async def start(self, config: AudioInputConfig) -> None:
        self._sample_rate = int(getattr(config, "sample_rate", None) or 16_000)
        # PCM16 mono: 2 bytes/sample.
        self._max_pcm_bytes = int(self._sample_rate * 2 * self.AUDIO_WINDOW_SECS)
        self._pcm.clear()
        self._pcm_bytes = 0
        if self.audio_eou is not None:
            await self.audio_eou.start(sample_rate=self._sample_rate)

    async def close(self) -> None:
        if self.audio_eou is not None:
            await self.audio_eou.close()
        self._pcm.clear()
        self._pcm_bytes = 0

    def push_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._pcm.append(chunk)
        self._pcm_bytes += len(chunk)
        while self._pcm_bytes > self._max_pcm_bytes and self._pcm:
            dropped = self._pcm.popleft()
            self._pcm_bytes -= len(dropped)

    def _recent_pcm(self) -> bytes:
        if not self._pcm:
            return b""
        return b"".join(self._pcm)

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        decision = await super().on_committed(text, state)
        if decision.action is CommitAction.IGNORE:
            return decision
        if decision.action is CommitAction.CONTINUE_TURN:
            return decision
        # Parent hold_merge of a self-contained commit → don't glue onto held text.
        if (
            state.holding
            and decision.action is CommitAction.HOLD
            and decision.reason == "hold_merge"
            and _looks_like_fresh_hold_utterance(text)
        ):
            decision = CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="hold_supersede")
        if self.audio_eou is None:
            return decision
        # Score fresh turns and hold updates (parent may HOLD a refined fragment).
        if decision.action not in (CommitAction.NEW_TURN, CommitAction.HOLD):
            return decision
        candidate = decision.text or text
        pcm = self._recent_pcm()
        if len(pcm) < self._sample_rate:  # < ~0.5s of audio — not enough signal
            return decision
        try:
            p = await self.audio_eou.predict_complete(pcm, sample_rate=self._sample_rate)
        except Exception as e:
            logger.warning("audio_eou_predict_failed", error=str(e))
            return decision
        if p >= self.completion_threshold:
            return CommitDecision(
                action=CommitAction.NEW_TURN,
                text=candidate,
                reason="audio_complete" if state.holding else (decision.reason or "new_turn"),
            )
        # Incomplete: mid-agent-turn → merge+restart; else HOLD (session debounce).
        if state.active_user_text and state.assistant_active:
            combined = state.active_user_text.rstrip(", ") + " " + text
            return CommitDecision(
                action=CommitAction.CONTINUE_TURN,
                text=combined,
                reason="audio_continuation",
            )
        return CommitDecision(
            action=CommitAction.HOLD,
            text=candidate,
            reason="audio_hold",
            hold_timeout_secs=self.hold_timeout_secs,
        )


def resolve_turn_detector(spec: Any = None) -> TurnDetector:
    """Build a :class:`TurnDetector` from a string name, instance, or ``None``.

    Accepted names: ``heuristic`` (default), ``provider``, ``local``, ``lexical``.
    Instances are returned unchanged. Unknown strings raise ``ValueError``.
    """
    if spec is None:
        return HeuristicTurnDetector()
    if isinstance(spec, TurnDetector):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in ("", "heuristic", "default"):
            return HeuristicTurnDetector()
        if key in ("provider", "stt"):
            return ProviderTurnDetector()
        if key in ("local", "audio", "smart_turn"):
            return LocalAudioTurnDetector()
        if key in ("lexical", "semantic", "punctuation"):
            return LexicalTurnDetector()
        raise ValueError(
            f"Unknown turn_detector {spec!r}; expected one of "
            "'heuristic', 'provider', 'local', 'lexical', or a TurnDetector instance"
        )
    raise TypeError(f"turn_detector must be str | TurnDetector | None, got {type(spec)!r}")
