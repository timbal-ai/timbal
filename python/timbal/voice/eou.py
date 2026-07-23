"""End-of-utterance (EOU) models for turn detection.

Two seams, matching how LiveKit / Pipecat actually ship turn-taking today:

* :class:`AudioEouModel` — **preferred local path.** Scores raw PCM
  (Pipecat Smart Turn, LiveKit v1-mini style). Used by
  :class:`~timbal.voice.LocalAudioTurnDetector` after a short VAD/STT pause.
* :class:`TextEouPredictor` — optional lightweight text overlay (punctuation /
  dangling tokens). Not a substitute for audio EOU; useful when you want a
  zero-dep lexical bias without installing ``timbal[voice]``.

Provider-native endpointing (OpenAI ``semantic_vad``, ElevenLabs Scribe VAD
commits, Deepgram Flux, etc.) does **not** go through these classes — use
:class:`~timbal.voice.ProviderTurnDetector` (or a future realtime session that
defers entirely to the provider).

``predict_*`` methods are async so backends can offload inference; ``start`` /
``close`` bracket model load/teardown against the session lifecycle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger("timbal.voice.eou")


class AudioEouModel(ABC):
    """Local audio end-of-turn model (Smart Turn / LiveKit v1-mini shaped).

    Implementations consume recent user PCM and return ``P(complete)``.
    Concrete backends (ONNX Smart Turn v3, etc.) live behind the optional
    ``timbal[voice]`` extra and are injected into
    :class:`~timbal.voice.LocalAudioTurnDetector`.
    """

    async def start(self, *, sample_rate: int) -> None:  # noqa: B027, ARG002
        """Load weights / warm up. ``sample_rate`` is the session mic rate."""

    async def close(self) -> None:  # noqa: B027
        """Release resources."""

    @abstractmethod
    async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:
        """Return ``P(complete)`` in ``[0, 1]`` from recent mono PCM16LE audio."""


class TextEouPredictor(ABC):
    """Optional text-only EOU score (lexical / punctuation).

    Kept separate from :class:`AudioEouModel` so the audio path never pretends
    a transcript classifier is equivalent to Smart Turn-style models.
    """

    async def start(self) -> None:  # noqa: B027
        """Load models / warm up."""

    async def close(self) -> None:  # noqa: B027
        """Release resources."""

    @abstractmethod
    async def predict_eou(self, text: str) -> float:
        """Return ``P(complete)`` in ``[0, 1]`` — higher means the user is done."""


# Back-compat alias used by early Phase-1a code / tests.
EouPredictor = TextEouPredictor


# ---------------------------------------------------------------------------
# Optional lexical baseline (no dependencies)
# ---------------------------------------------------------------------------

_TERMINAL_PUNCT = tuple(".?!…。？！")
_CONTINUING_PUNCT = tuple(",;:—–-")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_DANGLING_TOKENS = frozenset(
    {
        # English
        "and", "or", "but", "so", "because", "if", "when", "while", "that",
        "which", "who", "to", "of", "for", "with", "from", "in", "on", "at",
        "by", "the", "a", "an", "my", "your", "our", "their", "his", "her",
        "its", "i", "we", "you", "they", "it", "is", "are", "was", "were",
        "um", "uh", "like", "then", "as", "than", "into", "about", "over",
        # Spanish
        "y", "e", "o", "u", "pero", "porque", "que", "si", "cuando",
        "mientras", "de", "del", "para", "por", "con", "sin", "en",
        "la", "el", "los", "las", "un", "una", "unos", "unas", "mi", "tu",
        "su", "es", "está", "estoy", "eh", "este", "esto", "como", "más",
        "muy",
    }
)

# Thinking-pause / hedge phrases: often punctuated as complete by STT
# ("Uh, I don't know.") but the speaker continues. Short single-token hedges
# match only as the *entire* utterance; multi-word phrases also match as a
# trailing clause ("well i don't know"). Swappable later for a DistilBERT /
# Namo-class text EOU behind the same :class:`TextEouPredictor` interface.
_SHORT_HEDGES = frozenset(
    {
        "uh", "um", "uhm", "hmm", "mm", "mhm", "mmhmm", "well", "so", "like",
        "maybe", "perhaps", "idk", "pues", "este", "eh",
    }
)
_PHRASE_HEDGES = frozenset(
    {
        "i don't know",
        "i dont know",
        "i do not know",
        "not sure",
        "i'm not sure",
        "im not sure",
        "i am not sure",
        "let me think",
        "i mean",
        "you know",
        "i guess",
        "no se",
        "no sé",
        "a ver",
        "no lo se",
        "no lo sé",
    }
)
_PUNCT_STRIP_RE = re.compile(r"[.?!…。？！,;:—–\-]+")


def _normalize_utterance(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace, unify apostrophes."""
    t = text.lower().strip()
    for a in ("'", "'", "`"):
        t = t.replace(a, "'")
    t = _PUNCT_STRIP_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _looks_like_hedge(text: str) -> bool:
    """True when the utterance is (or ends in) a thinking-pause hedge."""
    norm = _normalize_utterance(text)
    if not norm:
        return False
    if norm in _SHORT_HEDGES or norm in _PHRASE_HEDGES:
        return True
    for phrase in _PHRASE_HEDGES:
        if norm.endswith(" " + phrase):
            return True
    return False


class PunctuationEouPredictor(TextEouPredictor):
    """Zero-dep lexical EOU: punctuation, dangling tokens, and hedges.

    Scores (tunable via subclass attributes):

    * thinking-pause hedge ("i don't know", bare "uh"/"well") → :attr:`P_HEDGE`
      — wins over terminal punctuation (STT writes "Uh, I don't know.")
    * ends with terminal punctuation → :attr:`P_TERMINAL`
    * ends with continuing punctuation / ellipsis → :attr:`P_CONTINUING`
    * last word is a dangling conjunction/preposition/filler → :attr:`P_DANGLING`
    * otherwise → :attr:`P_NEUTRAL` (slightly complete-leaning; STT often drops
      the final period and over-merging is worse than under-merging)

    English + Spanish dangling lists match the server's ``es`` default. This
    is the dumb baseline behind :class:`TextEouPredictor` — swap for a small
    ONNX text classifier later without changing turn-detection wiring.
    """

    P_TERMINAL: float = 0.95
    P_CONTINUING: float = 0.15
    P_DANGLING: float = 0.15
    P_HEDGE: float = 0.2
    P_NEUTRAL: float = 0.60

    async def predict_eou(self, text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 1.0
        # Hedges before terminal punct: "Uh, I don't know." must not score
        # complete just because STT stuck a period on a mid-thought pause.
        if _looks_like_hedge(stripped):
            return self.P_HEDGE
        # Ellipsis before the terminal check: STT writes "..." / "…" exactly
        # when the speaker trails off mid-thought — the opposite of terminal,
        # despite ending in ".".
        if stripped.endswith("...") or stripped.endswith("…"):
            return self.P_CONTINUING
        last_char = stripped[-1]
        if last_char in _TERMINAL_PUNCT:
            return self.P_TERMINAL
        if last_char in _CONTINUING_PUNCT:
            return self.P_CONTINUING
        words = _WORD_RE.findall(stripped.lower())
        if words and words[-1] in _DANGLING_TOKENS:
            return self.P_DANGLING
        return self.P_NEUTRAL
