"""Turn detection for :class:`~timbal.voice.VoiceSession`.

A :class:`TurnDetector` decides how the session reacts to STT output:

* ``on_partial`` — should a live partial transcript barge in (interrupt the
  assistant) or be ignored?
* ``on_committed`` — is a committed transcript a new turn, a continuation of
  the in-flight turn, or noise/echo/refinement to ignore?

Detectors are pure with respect to the session: they receive a
:class:`TurnState` snapshot and return a decision. This keeps them unit
testable and lets implementations range from regex heuristics
(:class:`HeuristicTurnDetector`, the default) to local VAD or semantic
end-of-turn models — ``push_audio`` exists (as a no-op here) so audio-consuming
detectors can slot in without changing the session.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .session import AudioInputConfig

logger = structlog.get_logger("timbal.voice.turn_detection")


class TurnState(BaseModel):
    """Snapshot the session hands the detector — keeps detectors pure and testable."""

    model_config = ConfigDict(extra="forbid")

    assistant_active: bool
    """Agent is generating OR audio is still playing client-side."""
    audio_playing: bool
    """Client likely still has queued TTS audio to play."""
    assistant_text: str
    """Assistant text accumulated so far this turn."""
    active_user_text: str
    """User text of the in-flight turn (empty string if idle)."""
    seconds_since_turn_start: float
    seconds_since_last_commit: float
    partials_since_last_commit: int
    """Partial transcripts seen since the previous committed transcript."""


class PartialDecision(StrEnum):
    IGNORE = "ignore"
    BARGE_IN = "barge_in"


class CommitAction(StrEnum):
    IGNORE = "ignore"
    NEW_TURN = "new_turn"
    CONTINUE_TURN = "continue"


class CommitDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: CommitAction
    text: str
    """Final user text for the turn (merged text for ``CONTINUE_TURN``)."""
    reason: str
    """Why, for debug logs ("echo", "refinement", "continuation", ...)."""


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
        # almost always an STT hallucination on silence.
        if (
            state.partials_since_last_commit == 0
            and len(text) >= self.HALLUCINATION_MIN_CHARS
            and not state.assistant_active
        ):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="hallucination")
        if state.assistant_active and _likely_stt_echo(text, state.assistant_text):
            return CommitDecision(action=CommitAction.IGNORE, text=text, reason="echo")
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
