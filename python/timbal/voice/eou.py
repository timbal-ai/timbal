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


class PunctuationEouPredictor(TextEouPredictor):
    """Zero-dep lexical EOU: terminal punctuation + trailing dangling tokens.

    Scores (tunable via subclass attributes):

    * ends with terminal punctuation → :attr:`P_TERMINAL`
    * ends with continuing punctuation → :attr:`P_CONTINUING`
    * last word is a dangling conjunction/preposition/filler → :attr:`P_DANGLING`
    * otherwise → :attr:`P_NEUTRAL` (slightly complete-leaning; STT often drops
      the final period and over-merging is worse than under-merging)

    English + Spanish dangling lists match the server's ``es`` default.
    """

    P_TERMINAL: float = 0.95
    P_CONTINUING: float = 0.15
    P_DANGLING: float = 0.15
    P_NEUTRAL: float = 0.60

    async def predict_eou(self, text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 1.0
        last_char = stripped[-1]
        if last_char in _TERMINAL_PUNCT:
            return self.P_TERMINAL
        if last_char in _CONTINUING_PUNCT:
            return self.P_CONTINUING
        words = _WORD_RE.findall(stripped.lower())
        if words and words[-1] in _DANGLING_TOKENS:
            return self.P_DANGLING
        return self.P_NEUTRAL
