"""Speculative (preemptive) reply generation for :class:`~timbal.voice.VoiceSession`.

Start the agent when a turn *looks* finished, hold every audio byte until it
is confirmed, and drop the attempt if the caller keeps talking.

Two triggers, both opt-in via :class:`SpeculativeGeneration`:

* Smart Turn scores a pause as probably complete (``P(complete)`` at or above
  ``eou_threshold``).
* Deepgram Flux ``EagerEndOfTurn``. ``TurnResumed`` cancels the attempt.
  Flux only emits those events when ``eager_eot_threshold`` is on the
  connection. The server sets it with
  :func:`timbal.voice.predictive_eou.flux_eager_threshold`: no 0.3 floor,
  strictly below ``eot_threshold``, and an explicit value lowered the same way.

Transcript reuse follows LiveKit Agents' ``_transcripts_equivalent``
(Apache-2.0, ``livekit/agents`` ``agent_activity.py``): exact match, otherwise
casefolded words with punctuation removed and CJK/Thai characters split.
A confirmed transcript that is not equivalent discards the attempt and the
turn starts over on the final text.

No tool call starts until confirmation. The model still runs, which is the
latency win, but the handler, its ``pre_hook``, the approval gate, argument
guardrails, and default-parameter callables wait. A discarded attempt
cancels that wait, so none of them start. ``defer_until_eot`` marks a tool
whose work cannot be undone; it is not an opt-out, and it does nothing when
speculative generation is off. The gate itself is
:class:`timbal.core.speculative_gate.SpeculativeGate`, so the tool path does
not import this package.

``max_speech_duration`` and ``max_retries`` match LiveKit's
``PreemptiveGenerationOptions`` (10s, 3 per user turn). Speculation is
skipped once the caller has already been speaking longer than
``max_speech_duration``, because a long utterance is still likely to change.
``preemptive_tts`` defaults to false, also matching LiveKit: the model runs
early, speech synthesis starts once the turn is confirmed. Audio that does
get produced (a filler, or synthesis when ``preemptive_tts`` is on) stays
buffered until confirmation.
"""

from __future__ import annotations

import asyncio
import re
import string
from dataclasses import dataclass, field
from typing import Any

from ..core.speculative_gate import (
    SpeculativeGate,
    install_speculative_gate,
    reset_speculative_gate,
    wait_for_turn_confirmation,
)

__all__ = [
    "AudioHold",
    "SpeculativeAttempt",
    "SpeculativeGate",
    "documented_speculative_preset",
    "install_speculative_gate",
    "reset_speculative_gate",
    "transcripts_equivalent",
    "wait_for_turn_confirmation",
]

# LiveKit ``split_words(..., split_character=True)`` treats these as their own
# words. Arabic and Indic stay whitespace-delimited; only CJK and Thai split.
_CHAR_WORD = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\u0E00-\u0E7F]")
# ``string.punctuation`` plus the marks LiveKit's word tokenizer strips.
_PUNCTUATION = "".join(
    (
        string.punctuation,
        "。",
        "，",
        "、",
        "？",
        "！",
        "；",
        "：",
        "…",
        "—",
        "–",
        "“",
        "”",
        "‘",
        "’",
    )
)
_PUNCT_TABLE = str.maketrans("", "", _PUNCTUATION)


def transcripts_equivalent(first: str, second: str | None) -> bool:
    """Whether a speculative transcript can be reused for the confirmed one.

    Adapted from LiveKit Agents ``_transcripts_equivalent`` (Apache-2.0):
    identical strings match; otherwise casefolded words, punctuation ignored,
    CJK and Thai split into characters. An empty word list never matches, so
    a blank attempt cannot be reused.
    """
    if first == second:
        return True
    if second is None:
        return False
    first_words = _words(first)
    return bool(first_words) and first_words == _words(second)


def _words(text: str) -> list[str]:
    words: list[str] = []
    word_start = 0

    def _add(start: int, end: int) -> None:
        word = text[start:end].translate(_PUNCT_TABLE)
        if word:
            words.append(word.casefold())

    for pos, char in enumerate(text):
        if char.isspace():
            _add(word_start, pos)
            word_start = pos + 1
        elif _CHAR_WORD.match(char):
            if word_start < pos:
                _add(word_start, pos)
            _add(pos, pos + 1)
            word_start = pos + 1
    _add(word_start, len(text))
    return words


@dataclass
class AudioHold:
    """PCM produced before the turn is confirmed.

    Chunks are ``(pcm, segment_record)``. ``release`` lets the session emit
    them; ``discard`` drops them. Synthesis after ``released`` emits directly.
    """

    chunks: list[tuple[bytes, list]] = field(default_factory=list)
    released: bool = False
    discarded: bool = False
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def hold(self, chunk: bytes, record: list) -> bool:
        """Buffer ``chunk``. Return True when the caller should emit it now."""
        if self.discarded:
            return False
        if self.released:
            return True
        self.chunks.append((chunk, record))
        return False

    def release(self) -> None:
        self.released = True
        self.ready.set()

    def discard(self) -> None:
        self.discarded = True
        self.chunks.clear()
        self.ready.set()

    def take(self) -> list[tuple[bytes, list]]:
        chunks = self.chunks
        self.chunks = []
        return chunks


@dataclass
class SpeculativeAttempt:
    """One in-flight speculative turn, addressed by the transcript it used."""

    text: str
    source: str
    gate: SpeculativeGate
    hold: AudioHold
    prior_context: Any
    prior_head: str | None = None
    settled: bool = False


def documented_speculative_preset(model: str):
    """Opt-in preset for one measured model. Not applied unless you pass it.

    ``openai/gpt-5.4-mini`` is the Flux ship block in
    ``benchmarks/voice/REPORT-242.md`` (enabled, preemptive TTS). That table is
    a small sample, not a percentile, and it is not evidence for other OpenAI
    ids. ``groq/openai/gpt-oss-20b`` in the same report moved silence p90 from
    3603.2 ms to 4935.1 ms at n=3, which is a reason not to copy the block
    onto that model, not a ranking of every provider. Every other id,
    including other OpenAI models, Groq, and Anthropic, stays off.
    ``VoiceConfig`` still defaults to off.
    """
    from .config import SpeculativeGeneration

    key = (model or "").strip().lower()
    if key == "openai/gpt-5.4-mini":
        return SpeculativeGeneration(enabled=True, preemptive_tts=True)
    return SpeculativeGeneration(enabled=False)
