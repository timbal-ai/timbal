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

import copy
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

    def clone(self) -> TurnDetector:
        """Per-session copy for shared configs (server ``voice_config``).

        Each :class:`~timbal.voice.VoiceSession` owns its detector's
        ``start``/``push_audio``/``close`` lifecycle, so a single instance must
        not be shared across concurrent sessions. Default is a shallow copy
        (fine for stateless detectors; injected models stay shared). Detectors
        with per-session mutable state must override (see
        :class:`LocalAudioTurnDetector`).
        """
        return copy.copy(self)

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


# Filled-pause vocalizations across the languages ElevenLabs Scribe commonly
# transcribes ("uh", "um", Spanish "eh"...). Only real words are excluded:
# "no", "sí", "ok", "ya" must never match.
_HESITATION_TOKENS = frozenset(
    {
        "uh", "uhh", "uhhh", "um", "umm", "ummm", "uhm", "uhum",
        "hm", "hmm", "hmmm", "mm", "mmm", "mhm", "mmhm",
        "er", "erm", "eh", "ehh", "ehm", "em",
        "ah", "ahh", "ahhh", "aah",
    }
)
_HESITATION_WORD_RE = re.compile(r"[a-zà-öø-ÿ']+")


def _is_hesitation_only(text: str) -> bool:
    """True when the utterance is nothing but filled pauses ("Uh...", "Um, hmm").

    Hesitations signal *more speech coming* — they must neither barge in on the
    agent nor start a turn of their own ("Uh..." → agent replies to "Uh...").
    """
    words = _HESITATION_WORD_RE.findall(text.lower())
    return bool(words) and all(w in _HESITATION_TOKENS for w in words)


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


def _join_held(held: str, text: str) -> str:
    """Held fragment + follow-up as one utterance (used by hold_supersede)."""
    held = held.strip()
    if not held:
        return text
    return held + " " + text


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
    # A single short partial while the assistant is speaking is far more often
    # a mic blip / mis-transcribed speaker echo (e.g. "Nice.") than a real
    # interruption — and a false barge-in cancels TTS mid-reply and truncates
    # the committed transcript to what was heard (possibly nothing). Require a
    # few words before interrupting on a *partial*; real short commands
    # ("Stop.") still interrupt ~1s later via their *commit* (NEW_TURN path).
    # Same knob as Pipecat's MinWordsInterruptionStrategy(min_words=3) and
    # LiveKit's min_interruption_words.
    MIN_BARGE_IN_PARTIAL_WORDS = 3
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
        is_hesitation = _is_hesitation_only(text)
        is_echo = _likely_stt_echo(text, state.assistant_text) if not is_noise else False
        too_short = (
            len(text) < self.MIN_BARGE_IN_PARTIAL_CHARS
            or len(text.split()) < self.MIN_BARGE_IN_PARTIAL_WORDS
        )
        if not is_noise and not is_hesitation and not is_echo and not too_short:
            return PartialDecision.BARGE_IN
        logger.debug(
            "stt_partial_skipped",
            text_preview=text[:80],
            too_short=too_short,
            is_noise=is_noise,
            is_hesitation=is_hesitation,
            is_echo=is_echo,
            audio_playing=state.audio_playing,
        )
        return PartialDecision.IGNORE

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        if _is_noise(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="noise")
        if _is_garbage_commit(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="garbage")
        if _is_hesitation_only(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="hesitation")
        # A long commit with zero preceding partials while nothing is playing is
        # almost always an STT hallucination on silence. Uses assistant_active
        # only — pending HOLD must not flip that flag (see TurnState.holding).
        # Also skip while HOLDing: the follow-up commit is the rest of an
        # incomplete utterance (often arrives with few/no partials after a
        # thinking pause) and must reach the hold_merge / hold_refinement path.
        if (
            state.partials_since_last_commit == 0
            and len(text) >= self.HALLUCINATION_MIN_CHARS
            and not state.assistant_active
            and not state.holding
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
            # Same gates as mid-turn VAD-split continuation — but still refuse
            # to glue a self-contained utterance ("stop", a short new question).
            if (
                state.seconds_since_last_commit < self.CONTINUATION_WINDOW_SECS
                and len(text) < self.CONTINUATION_MAX_CHARS
                and not _looks_like_fresh_hold_utterance(text)
            ):
                combined = state.active_user_text.rstrip(", ") + " " + text
                return CommitDecision(action=CommitAction.HOLD, text=combined, reason="hold_merge")
            # Supersede = start the turn NOW, but never DROP the held fragment:
            # it is speech the user actually said and was never answered. STT
            # capitalizes every committed segment, so a mid-thought
            # continuation after a forced/VAD split ("…thinking about" +
            # "Something.") systematically looks "fresh" — discarding the
            # fragment here loses half the utterance. Prepending is right for
            # true supersedes too ("what's the weather… actually tell me a
            # joke" reads as a natural self-correction).
            return CommitDecision(
                action=CommitAction.NEW_TURN,
                text=_join_held(state.active_user_text, text),
                reason="hold_supersede",
            )
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


class RawTurnDetector(TurnDetector):
    """No filtering at all — a debugging detector.

    Every committed transcript starts a NEW_TURN and every non-empty partial
    during playback barges in. None of the silence/noise/echo/refinement
    heuristics run, so you see exactly what STT emits (including the agent's
    own speech leaking back through the mic). Never use in production:
    without echo suppression an open-speaker setup will happily talk to
    itself in a loop — which is precisely what this mode makes visible.

    Opt-in: ``resolve_turn_detector("raw")`` or the playground dropdown.
    """

    async def on_partial(self, text: str, state: TurnState) -> PartialDecision:
        if state.audio_playing and text:
            return PartialDecision.BARGE_IN
        return PartialDecision.IGNORE

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:  # noqa: ARG002
        return CommitDecision(action=CommitAction.NEW_TURN, text=text, reason="raw")


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
        if _is_noise(text) or _is_garbage_commit(text) or _is_hesitation_only(text):
            return PartialDecision.IGNORE
        if _likely_stt_echo(text, state.assistant_text):
            return PartialDecision.IGNORE
        # Same min-words gate as HeuristicTurnDetector: one short partial while
        # the assistant is speaking is more often a mic blip than a barge-in.
        if len(text.strip()) < 4 or len(text.split()) < HeuristicTurnDetector.MIN_BARGE_IN_PARTIAL_WORDS:
            return PartialDecision.IGNORE
        return PartialDecision.BARGE_IN

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        if _is_noise(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="noise")
        if _is_garbage_commit(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="garbage")
        if _is_hesitation_only(text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="hesitation")
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
        # Parent already replaced the held fragment with a distinct utterance
        # ("stop", a new question). Do not re-HOLD just because state.holding
        # is still true — that defers the supersede until the old timer fires.
        if decision.reason == "hold_supersede":
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
                # Start now, but keep the held fragment (see the parent's
                # hold_supersede: never drop transcribed user speech).
                return CommitDecision(
                    action=CommitAction.NEW_TURN,
                    text=_join_held(state.active_user_text, text),
                    reason="lexical_hold_supersede",
                )

        # Parent HOLD (hold_refinement / hold_merge): re-score the updated
        # utterance — it may now look complete. Only enter on HOLD decisions;
        # ``state.holding`` alone must not override a parent NEW_TURN.
        if decision.action is CommitAction.HOLD:
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
    # Grace window after an incomplete-scored commit before the fragment runs
    # anyway. Reference points: LiveKit's max_endpointing_delay is 6.0s of
    # *total* silence when their EOU model says "not done"; Pipecat's
    # smart-turn fallback (stop_secs) is 3.0s of continued silence. The HOLD
    # only arms after the STT VAD silence (~1.2s with the server default), so
    # 3.0s here gives ~4.2s total — between the two. This is also the full
    # price of a *wrong* "incomplete" score (e.g. Smart Turn on a bare
    # "Thank you."), and no re-score can rescue those mid-hold: the backend
    # trims trailing silence before scoring, so waiting longer reproduces the
    # same window and the same score.
    DEFAULT_HOLD_TIMEOUT_SECS = 3.0
    # Confidence tier for the HOLD (LiveKit's min/max endpointing delay shape,
    # with the transcript as the confidence signal): when the audio model says
    # "incomplete" but the text looks finished (terminal punctuation — Smart
    # Turn systematically under-scores short closers like "Thank you." /
    # "I am David." / "Quite good."), the hold shrinks to this. Keep it short:
    # the VAD endpointer has usually already paid ~0.5–3s of silence delay
    # before the commit, so a second 1.2s tax felt like dead air on every
    # finished utterance Smart Turn under-scored. Both-signals-incomplete
    # keeps the full timeout.
    TEXT_COMPLETE_HOLD_TIMEOUT_SECS = 0.35
    # The tier needs *confidently* finished text (terminal punctuation scores
    # P_TERMINAL=0.95), not the predictor's complete-leaning neutral (0.60) —
    # unpunctuated text must not shorten the hold.
    TEXT_COMPLETE_TIER_THRESHOLD = 0.9
    # Inverse tier: audio says complete but text looks mid-thought (hedges
    # score P_HEDGE=0.2, dangling/continuing ~0.15). Don't NEW_TURN — short
    # HOLD so a continuation ("…tell me a story") can merge. Neutral (0.60)
    # must NOT trigger this, or every unpunctuated complete fires a hold.
    TEXT_INCOMPLETE_TIER_THRESHOLD = 0.4
    TEXT_INCOMPLETE_HOLD_TIMEOUT_SECS = 1.2
    # The model consumes the last 8s of *speech*; the EOU backend trims the
    # trailing silence (STT commit debounce, hold pauses) before scoring, so
    # buffer extra raw PCM to keep a full 8s of signal after the trim.
    # 12s of 16kHz PCM16 is ~384KB per session.
    AUDIO_WINDOW_SECS = 12.0

    def __init__(
        self,
        audio_eou: AudioEouModel | None = None,
        *,
        completion_threshold: float | None = None,
        hold_timeout_secs: float | None = None,
        text_complete_hold_timeout_secs: float | None = None,
        text_incomplete_hold_timeout_secs: float | None = None,
        fallback_text_eou: TextEouPredictor | None = None,
    ) -> None:
        self.audio_eou = audio_eou
        if completion_threshold is not None:
            self.completion_threshold = completion_threshold
        self.hold_timeout_secs = (
            hold_timeout_secs if hold_timeout_secs is not None else self.DEFAULT_HOLD_TIMEOUT_SECS
        )
        self.text_complete_hold_timeout_secs = (
            text_complete_hold_timeout_secs
            if text_complete_hold_timeout_secs is not None
            else self.TEXT_COMPLETE_HOLD_TIMEOUT_SECS
        )
        self.text_incomplete_hold_timeout_secs = (
            text_incomplete_hold_timeout_secs
            if text_incomplete_hold_timeout_secs is not None
            else self.TEXT_INCOMPLETE_HOLD_TIMEOUT_SECS
        )
        # Used when the buffered PCM is too short to score, and as the text
        # confidence signal for both HOLD tiers (complete-text shortens;
        # incomplete-text delays an audio-complete commit). Default is the
        # zero-dep lexical baseline; ``resolve_turn_detector("local")`` injects
        # Namo when ``timbal[voice]`` is installed.
        self.fallback_text_eou = fallback_text_eou or PunctuationEouPredictor()
        # Namo under-scores many finished questions (~0.0 on "How are you?" /
        # "What's two plus two?"). Lexical gate corroborates so we only pay the
        # incomplete-text tax when both agree the user is mid-thought.
        self._lexical_gate = PunctuationEouPredictor()
        self._sample_rate = 16_000
        self._pcm: deque[bytes] = deque()
        self._pcm_bytes = 0
        self._max_pcm_bytes = 0

    async def effective_text_eou(self, text: str) -> float:
        """``P(complete)`` for HOLD tiers / VAD delay — Namo with lexical rescue.

        Namo under-scores many finished questions (~0.0 on "How are you?").
        Only lift the score when the lexical baseline is *confidently* complete
        (terminal punct ≥ :attr:`TEXT_COMPLETE_TIER_THRESHOLD`) — a blunt
        ``max(namo, lexical)`` would also promote neutral unpunctuated
        mid-thoughts (lexical ~0.60) over the incomplete tier and neuter Namo.
        """
        p = await self.fallback_text_eou.predict_eou(text)
        if isinstance(self.fallback_text_eou, PunctuationEouPredictor):
            return p
        p_lex = await self._lexical_gate.predict_eou(text)
        if p_lex >= self.TEXT_COMPLETE_TIER_THRESHOLD:
            return max(p, p_lex)
        return p

    async def start(self, config: AudioInputConfig) -> None:
        self._sample_rate = int(getattr(config, "sample_rate", None) or 16_000)
        # PCM16 mono: 2 bytes/sample.
        self._max_pcm_bytes = int(self._sample_rate * 2 * self.AUDIO_WINDOW_SECS)
        self._pcm.clear()
        self._pcm_bytes = 0
        if self.audio_eou is not None:
            await self.audio_eou.start(sample_rate=self._sample_rate)
        await self.fallback_text_eou.start()

    async def close(self) -> None:
        if self.audio_eou is not None:
            await self.audio_eou.close()
        await self.fallback_text_eou.close()
        self._pcm.clear()
        self._pcm_bytes = 0

    def clone(self) -> LocalAudioTurnDetector:
        """Fresh PCM buffer per session; the ``audio_eou`` model stays shared."""
        return type(self)(
            audio_eou=self.audio_eou,
            completion_threshold=self.completion_threshold,
            hold_timeout_secs=self.hold_timeout_secs,
            text_complete_hold_timeout_secs=self.text_complete_hold_timeout_secs,
            text_incomplete_hold_timeout_secs=self.text_incomplete_hold_timeout_secs,
            fallback_text_eou=self.fallback_text_eou,
        )

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

    async def score_recent_audio(self) -> float | None:
        """``P(complete)`` for the buffered mic window, or ``None``.

        The VAD endpointing fast path (:class:`~timbal.voice.endpointing.VadEndpointer`)
        calls this right after Silero detects speech-stop — before any STT
        commit exists. Returns ``None`` when there is no audio EOU model, not
        enough buffered signal (< ~0.5s), or inference fails; the endpointer
        then does nothing and the provider debounce commits as usual.
        """
        if self.audio_eou is None:
            return None
        pcm = self._recent_pcm()
        if len(pcm) < self._sample_rate:  # < ~0.5s of PCM16 — not enough signal
            return None
        try:
            return await self.audio_eou.predict_complete(pcm, sample_rate=self._sample_rate)
        except Exception as e:
            logger.warning("audio_eou_predict_failed", error=str(e))
            return None

    async def on_committed(self, text: str, state: TurnState) -> CommitDecision:
        decision = await super().on_committed(text, state)
        if decision.action is CommitAction.IGNORE:
            return decision
        if decision.action is CommitAction.CONTINUE_TURN:
            return decision
        # Parent hold_merge of a self-contained commit → start the turn now
        # instead of re-holding, but keep the held fragment in the turn text
        # (never drop transcribed user speech; see parent hold_supersede).
        if (
            state.holding
            and decision.action is CommitAction.HOLD
            and decision.reason == "hold_merge"
            and _looks_like_fresh_hold_utterance(text)
        ):
            decision = CommitDecision(
                action=CommitAction.NEW_TURN,
                text=_join_held(state.active_user_text, text),
                reason="hold_supersede",
            )
        # The parent's continuation gates (<3s, <30 chars) are tuned for mid-turn
        # VAD splits. While HOLDing, the audio model has explicitly judged the
        # held fragment incomplete — a non-fresh follow-up is the rest of that
        # thought no matter its length or how long the user paused (the hold
        # window *is* the merge window). Merge and rescore the full utterance.
        if (
            state.holding
            and state.active_user_text
            and decision.reason == "hold_supersede"
            and self.audio_eou is not None
            and not _looks_like_fresh_hold_utterance(text)
        ):
            combined = state.active_user_text.rstrip(", ") + " " + text
            decision = CommitDecision(action=CommitAction.HOLD, text=combined, reason="hold_merge")
        # Distinct utterance already replaced the hold — start immediately even
        # if the audio EOU still scores the previous window as incomplete.
        if decision.reason == "hold_supersede":
            return decision
        if self.audio_eou is None:
            return decision
        # Score fresh turns and hold updates (parent may HOLD a refined fragment).
        if decision.action not in (CommitAction.NEW_TURN, CommitAction.HOLD):
            return decision
        candidate = decision.text or text
        pcm = self._recent_pcm()
        if len(pcm) < self._sample_rate:  # < ~0.5s of audio — not enough signal
            # Parent HOLD stands; but don't let a NEW_TURN through unchecked —
            # score the text lexically so an incomplete fast commit still HOLDs.
            if decision.action is not CommitAction.NEW_TURN:
                return decision
            p_text = await self.effective_text_eou(candidate)
            if p_text >= self.completion_threshold:
                return decision
            return CommitDecision(
                action=CommitAction.HOLD,
                text=candidate,
                reason="audio_short_lexical_hold",
                hold_timeout_secs=self.hold_timeout_secs,
            )
        try:
            p = await self.audio_eou.predict_complete(pcm, sample_rate=self._sample_rate)
        except Exception as e:
            logger.warning("audio_eou_predict_failed", error=str(e))
            return decision
        # INFO on purpose: the score is the whole point of "local" mode and
        # fires at most once per STT commit — without it there is no way to
        # tell "model said complete" apart from "hold expired" in server logs.
        logger.info(
            "audio_eou_score",
            p=round(p, 3),
            threshold=self.completion_threshold,
            complete=p >= self.completion_threshold,
            text_preview=candidate[:80],
        )
        if p >= self.completion_threshold:
            # Inverse tier (see TEXT_INCOMPLETE_HOLD_TIMEOUT_SECS): audio says
            # done but the transcript looks mid-thought — hold short so a
            # continuation can merge. Never overrule into IGNORE; just delay.
            try:
                p_text = await self.effective_text_eou(candidate)
            except Exception as e:
                logger.warning("text_eou_predict_failed", error=str(e))
                p_text = None
            if p_text is not None and p_text < self.TEXT_INCOMPLETE_TIER_THRESHOLD:
                # Mid-agent barge-in that looks incomplete still merges via
                # CONTINUE rather than parking a HOLD over the reply.
                if state.active_user_text and state.assistant_active:
                    combined = state.active_user_text.rstrip(", ") + " " + text
                    return CommitDecision(
                        action=CommitAction.CONTINUE_TURN,
                        text=combined,
                        reason="audio_complete_text_incomplete_continue",
                    )
                return CommitDecision(
                    action=CommitAction.HOLD,
                    text=candidate,
                    reason="audio_complete_text_incomplete",
                    hold_timeout_secs=min(
                        self.text_incomplete_hold_timeout_secs, self.hold_timeout_secs
                    ),
                )
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
        # Confidence tier (see TEXT_COMPLETE_HOLD_TIMEOUT_SECS): a transcript
        # that reads finished disagrees with the audio score — hold, but short.
        timeout = self.hold_timeout_secs
        reason = "audio_hold"
        try:
            p_text = await self.effective_text_eou(candidate)
        except Exception as e:
            logger.warning("text_eou_predict_failed", error=str(e))
            p_text = None
        if p_text is not None and p_text >= self.TEXT_COMPLETE_TIER_THRESHOLD:
            timeout = min(self.text_complete_hold_timeout_secs, self.hold_timeout_secs)
            reason = "audio_hold_text_complete"
        return CommitDecision(
            action=CommitAction.HOLD,
            text=candidate,
            reason=reason,
            hold_timeout_secs=timeout,
        )


def _default_audio_eou() -> AudioEouModel | None:
    """Smart Turn v3 when ``timbal[voice]`` is installed, else ``None``.

    A single shared instance: the ONNX session is stateless per call and the
    model load is expensive, so every ``"local"`` detector (and its per-session
    clones) reuses it.
    """
    global _DEFAULT_AUDIO_EOU
    if _DEFAULT_AUDIO_EOU is not _AUDIO_EOU_UNSET:
        return _DEFAULT_AUDIO_EOU
    try:
        from .smart_turn import SmartTurnEouModel
    except ImportError:
        logger.warning(
            "smart_turn_unavailable",
            hint="turn_detector='local' without timbal[voice]; degrading to heuristics. "
            "Install with: pip install 'timbal[voice]'",
        )
        _DEFAULT_AUDIO_EOU = None
        return None
    _DEFAULT_AUDIO_EOU = SmartTurnEouModel()
    return _DEFAULT_AUDIO_EOU


def _default_text_eou() -> TextEouPredictor:
    """Namo English DistilBERT when ``timbal[voice]`` is installed, else punctuation.

    Shared process-wide (same rationale as :func:`_default_audio_eou`).
    """
    global _DEFAULT_TEXT_EOU
    if _DEFAULT_TEXT_EOU is not _TEXT_EOU_UNSET:
        return _DEFAULT_TEXT_EOU
    try:
        from .namo import NamoTextEouPredictor
    except ImportError:
        logger.debug(
            "namo_text_eou_unavailable",
            hint="falling back to PunctuationEouPredictor; install timbal[voice] for Namo",
        )
        _DEFAULT_TEXT_EOU = PunctuationEouPredictor()
        return _DEFAULT_TEXT_EOU
    _DEFAULT_TEXT_EOU = NamoTextEouPredictor()
    return _DEFAULT_TEXT_EOU


_AUDIO_EOU_UNSET: Any = object()
_DEFAULT_AUDIO_EOU: AudioEouModel | None = _AUDIO_EOU_UNSET
_TEXT_EOU_UNSET: Any = object()
_DEFAULT_TEXT_EOU: TextEouPredictor | Any = _TEXT_EOU_UNSET


def resolve_turn_detector(spec: Any = None) -> TurnDetector:
    """Build a :class:`TurnDetector` from a name, instance, factory, or ``None``.

    Accepted names: ``heuristic`` (default), ``provider``, ``local``, ``lexical``,
    ``raw`` (debug: no silence/noise/echo filtering at all).
    ``local`` auto-loads Smart Turn (audio) + Namo (text) when the
    ``timbal[voice]`` extra is installed (heuristic / punctuation degradation
    otherwise). Instances are returned unchanged — callers that reuse one spec
    across concurrent sessions (server ``voice_config``) must
    :meth:`~TurnDetector.clone` per session, or pass a zero-arg factory
    callable instead.
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
            return LocalAudioTurnDetector(
                audio_eou=_default_audio_eou(),
                fallback_text_eou=_default_text_eou(),
            )
        if key in ("lexical", "semantic", "punctuation"):
            return LexicalTurnDetector(text_eou=_default_text_eou())
        if key in ("raw", "none", "off"):
            return RawTurnDetector()
        raise ValueError(
            f"Unknown turn_detector {spec!r}; expected one of "
            "'heuristic', 'provider', 'local', 'lexical', 'raw', a TurnDetector "
            "instance, or a zero-arg factory returning one"
        )
    if callable(spec):
        built = spec()
        if not isinstance(built, TurnDetector):
            raise TypeError(
                f"turn_detector factory must return a TurnDetector, got {type(built)!r}"
            )
        return built
    raise TypeError(f"turn_detector must be str | TurnDetector | callable | None, got {type(spec)!r}")
