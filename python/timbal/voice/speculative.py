"""Speculative (preemptive) reply generation for :class:`~timbal.voice.VoiceSession`.

Start the agent when a turn *looks* finished, hold every audio byte until it
is confirmed, and drop the attempt if the caller keeps talking.

Two triggers, both opt-in via :class:`SpeculativeGeneration`:

* Smart Turn scores a pause as probably complete (``P(complete)`` at or above
  ``eou_threshold``).
* Deepgram Flux ``EagerEndOfTurn``. ``TurnResumed`` cancels the attempt.
  Flux only emits those events when ``eager_eot_threshold`` is on the
  connection. The server sets it with
  :func:`timbal.voice.predictive_eou.flux_eager_threshold` (the same helper
  during-speech prediction uses). An explicit ``stt_extra`` value wins.

Transcript reuse follows LiveKit Agents' ``_transcripts_equivalent``
(Apache-2.0, ``livekit/agents`` ``agent_activity.py``): exact match, otherwise
casefolded words with punctuation removed and CJK/Thai characters split.
A confirmed transcript that is not equivalent discards the attempt and the
turn starts over on the final text.

Side-effecting tools (``Runnable.defer_until_eot``, Deepgram's
``defer_until_eot``) do not run until confirmation. Read-only tools run
immediately — that is where the latency win is. A discarded attempt cancels
the wait, so a deferred handler never starts.

``max_speech_duration`` and ``max_retries`` match LiveKit's
``PreemptiveGenerationOptions`` (10s, 3 per user turn). ``preemptive_tts``
defaults to false, also matching LiveKit: the model runs early, speech
synthesis starts once the turn is confirmed. Audio that does get produced
(a filler, or synthesis when ``preemptive_tts`` is on) stays buffered until
confirmation.
"""

from __future__ import annotations

import asyncio
import re
import string
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

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


class SpeculativeGate:
    """Blocks ``defer_until_eot`` tools until the voice turn is confirmed.

    ``confirm`` lets every waiter run the handler. ``discard`` cancels them
    so the handler never starts. Outside a speculative turn there is no gate
    and :func:`wait_for_turn_confirmation` returns immediately.
    """

    def __init__(self) -> None:
        self._confirmed = asyncio.Event()
        self._discarded = asyncio.Event()

    @property
    def confirmed(self) -> bool:
        return self._confirmed.is_set()

    def confirm(self) -> None:
        self._confirmed.set()

    def discard(self) -> None:
        self._discarded.set()

    async def wait(self) -> None:
        if self._confirmed.is_set():
            return
        if self._discarded.is_set():
            raise asyncio.CancelledError
        confirm_task = asyncio.create_task(self._confirmed.wait())
        discard_task = asyncio.create_task(self._discarded.wait())
        try:
            await asyncio.wait(
                {confirm_task, discard_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (confirm_task, discard_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(confirm_task, discard_task, return_exceptions=True)
        if self._discarded.is_set() and not self._confirmed.is_set():
            raise asyncio.CancelledError


_GATE: ContextVar[SpeculativeGate | None] = ContextVar("timbal_speculative_gate", default=None)


def install_speculative_gate(gate: SpeculativeGate) -> Token[SpeculativeGate | None]:
    return _GATE.set(gate)


def reset_speculative_gate(token: Token[SpeculativeGate | None]) -> None:
    _GATE.reset(token)


async def wait_for_turn_confirmation() -> None:
    """No-op without a speculative gate; otherwise block until confirm or discard."""
    gate = _GATE.get()
    if gate is None:
        return
    await gate.wait()


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
