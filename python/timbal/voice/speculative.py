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
import unicodedata
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
    "NumberSpan",
    "number_spans",
    "spoken_numbers",
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
    CJK and Thai split into characters. Numbers must match as values, sign,
    units and any separator that could change the quantity included
    (``spoken_numbers``): a changed value is a new turn, never a reused reply
    or tool call. An empty word list never matches,
    so a blank attempt cannot be reused.
    """
    if first == second:
        return True
    if second is None:
        return False
    first_words = _words(first)
    return bool(first_words) and first_words == _words(second) and spoken_numbers(first) == spoken_numbers(second)


# A number as written, sign and decimal / thousands separators included: ignoring punctuation must not make
# "1.50" the same as "150", or "-20" the same as "20". A separator counts only between digits ("20." is 20).
_NUMBER = re.compile(r"(?<![\w.,])[-+\u2212]?\d+(?:[.,]\d+)*")
# Units written against a number. Punctuation is ignored between words, so without them "$12" and "12%"
# would both be "12". A currency sign may sit before the number (after its sign, or before it) or after it,
# "%" and the like only after, with one space at most (any space character: "12\u00a0€" in French
# typography). A sign belongs to one number: "3 $5" is 3 and $5, "3$ 5" is 3$ and 5.
_CURRENCY = "$€£¥₹₩¢₽₺₪₫₱₦฿₴₲₵₡₭₮₸₼₾"
_SUFFIX_UNITS = "%‰‱°"
_SIGNS = "-+\u2212"
# Between two numbers, an operator says how they combine: "5 + 3" is not "5 - 3", "5 * 3" not "5 / 3".
_OPERATORS = {"+": "+", "-": "-", "\u2212": "-", "*": "*", "×": "*", "/": "/", "÷": "/", "=": "="}


@dataclass(frozen=True)
class NumberSpan:
    """One number in a transcript: where it is (units and sign included) and what it says.

    ``value`` is strict: two spans with the same ``value`` mean the same quantity in every locale.
    ``"1.250"`` and ``"1,000"`` keep their separator (thousands in one locale, three decimals in another),
    while ``"1,000,000"`` / ``"1.000.000"`` / ``"1000000"`` are one value and ``"1,50"`` is ``"1.50"``. It
    also carries the operator that joins it to the number before ("5 - 3": ``-`` then 3).
    ``spoken`` is the same words: it also reads a three-digit group as thousands ("1,000" is said like
    "1000"), and has no operator, for finding words again in a buffer an STT may have formatted differently.
    """

    start: int
    end: int
    value: str
    spoken: str


def _is_space(ch: str) -> bool:
    return unicodedata.category(ch) == "Zs"


def _starts_number(text: str, at: int) -> bool:
    """A number (or its sign) starts at ``at``: the unit there is that number's, not the one before."""
    if at < len(text) and text[at].isdigit():
        return True
    return at + 1 < len(text) and text[at] in _SIGNS and text[at + 1].isdigit()


def number_spans(text: str) -> list[NumberSpan]:
    """The numbers in ``text`` in order, each with its sign and the units written against it."""
    text = text or ""
    spans: list[NumberSpan] = []
    taken = 0  # where the previous span ends: a unit before it is that span's
    for match in _NUMBER.finditer(text):
        start, end = match.span()
        body = match.group(0)
        sign = ""
        if body[0] in _SIGNS:
            sign, body = ("+" if body[0] == "+" else "-"), body[1:]
        units: list[str] = []
        # Before: "$12", "$ 12", "-$12", "$-12".
        if start - 1 >= taken and text[start - 1] in _CURRENCY:
            units.append(text[start - 1])
            start -= 1
        elif start - 2 >= taken and _is_space(text[start - 1]) and text[start - 2] in _CURRENCY:
            units.append(text[start - 2])
            start -= 2
        if not sign and units and start - 1 >= taken and text[start - 1] in _SIGNS:
            sign = "+" if text[start - 1] == "+" else "-"
            start -= 1
        # After: "12%", "12 %", "12€", "12 €", unless that sign opens the next number ("3 $5").
        if end < len(text) and (text[end] in _SUFFIX_UNITS or text[end] in _CURRENCY):
            if text[end] in _SUFFIX_UNITS or not _starts_number(text, end + 1):
                units.append(text[end])
                end += 1
        elif (
            end + 1 < len(text)
            and _is_space(text[end])
            and (text[end + 1] in _SUFFIX_UNITS or text[end + 1] in _CURRENCY)
            and (text[end + 1] in _SUFFIX_UNITS or not _starts_number(text, end + 2))
        ):
            units.append(text[end + 1])
            end += 2
        unit = "".join(sorted(units))
        spoken = f"{sign}{_quantity(body, three_digit_group_is_thousands=True)}{unit}"
        operator = ""
        if spans:
            gap = text[spans[-1].end : start]
            if sign and not gap.strip():
                operator, sign = sign, ""  # "5 -3", "5 - 3" and "5-3" are one subtraction
            elif gap.strip() in _OPERATORS:
                operator = _OPERATORS[gap.strip()]
        quantity = _quantity(body, three_digit_group_is_thousands=False)
        spans.append(
            NumberSpan(
                start=start,
                end=end,
                value=f"{operator}~{quantity}{unit}" if operator else f"{sign}{quantity}{unit}",
                spoken=spoken,
            )
        )
        taken = end
    return spans


def _quantity(body: str, *, three_digit_group_is_thousands: bool) -> str:
    """Digits with ``.`` as the decimal point, or ``body`` as written when the separators are ambiguous."""
    seps = [ch for ch in body if ch in ".,"]
    if not seps:
        return body
    if len(set(seps)) == 2:
        # "1,000.50" / "1.000,50": the last separator is the decimal point, the other groups thousands.
        point = seps[-1]
        whole, frac = body.rsplit(point, 1)
        groups = whole.split("," if point == "." else ".")
        if point not in whole and _thousands(groups):
            return "".join(groups) + "." + frac
        return body
    groups = re.split(r"[.,]", body)
    if len(groups) > 2:
        # "1,000,000": a decimal point appears once, so repeated separators group thousands.
        return "".join(groups) if _thousands(groups) else body
    whole, frac = groups
    if len(frac) != 3:
        return f"{whole}.{frac}"  # "1.50", "1,5": no locale groups two digits
    if three_digit_group_is_thousands and _thousands(groups):
        return whole + frac
    return body  # "1,000", "1.250": thousands in one locale, three decimals in another


def _thousands(groups: list[str]) -> bool:
    return 1 <= len(groups[0]) <= 3 and all(len(group) == 3 for group in groups[1:])


def spoken_numbers(text: str) -> list[str]:
    """The numbers in ``text``, in order, each with its sign, units and decimal point (``−`` read as ``-``).

    Strict (:attr:`NumberSpan.value`): a separator that could change the quantity keeps the number apart
    ("1.250" is not "1250", "1,000" is not "1000"), and so does a unit ("$12" is not "12%" or "12").
    """
    return [span.value for span in number_spans(text)]


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
    them; ``discard`` drops them. Synthesis emits directly once the buffered
    chunks have been emitted (``drained``), never ahead of them.
    """

    chunks: list[tuple[bytes, list]] = field(default_factory=list)
    released: bool = False
    discarded: bool = False
    drained: bool = False
    """Released and every buffered chunk emitted: from then on synthesis emits directly. Until then a new
    chunk queues behind the buffered ones, so playback keeps the order they were synthesized in."""
    flushing: bool = False
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def hold(self, chunk: bytes, record: list) -> bool:
        """Buffer ``chunk``. Return True when the caller should emit it now."""
        if self.discarded:
            return False
        if self.drained:
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
    task: Any = None
    """The turn task running this attempt: the only one a discard cancels."""


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
