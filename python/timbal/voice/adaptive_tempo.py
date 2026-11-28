"""Slow the agent down for a caller who talks slowly, or who is upset.

A person on the phone matches the other's pace. A fixed ``tts_tempo`` cannot:
the caller who takes their time is answered as briskly as the one in a hurry,
and a complaint gets the same cheerful speed as an order.

``VoiceConfig.adaptive_tempo`` turns ``tts_tempo`` into a ceiling. Each reply
is spoken at ``tts_tempo`` times a factor between ``slowest`` and 1, decided
when the turn starts and fixed for it. Two things lower it:

* **How fast the caller talks.** Syllables of what they said over the seconds
  they spent saying it (:class:`~timbal.voice.mic_activity.MicActivity`),
  smoothed over their utterances. At ``normal_rate`` or faster the factor is
  1; at ``slow_rate`` or slower it is ``slowest``; in between, a straight
  line. An utterance of a few words, one with digits in it (a transcript
  writes "600 123 456" and nobody knows how it was said), or one that
  measures faster than anyone talks, says nothing.
* **Whether they are upset.** Words of a complaint in what they said
  (:data:`DEFAULT_UPSET_WORDS`: "harto", "inaceptable", "fed up", a swear
  word) hold the factor at ``upset_tempo`` for ``upset_turns`` replies. A
  word list is a blunt reading of a mood. It misses the caller who is angry
  in polite words, and it never speeds anything up, so what it gets wrong is
  a reply a little slower than it had to be.

The agent never talks faster than ``tts_tempo``.

Pure arithmetic over text and seconds. :class:`CallerPace` holds the state of
one call.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from .config import language_key

# Syllables per second of speech, pauses left out, as :func:`syllables` counts
# them: (slow, normal). People differ and so do voices: by
# ``benchmarks/voice/caller_pace.py`` one ElevenLabs model reads Spanish at 5.7
# and another at 7.6, and English at 5.3 and 7.4. ``normal`` sits at the slower
# reading and ``slow`` a quarter below it, so the full slowdown is for a caller
# who talks like that voice played at 0.7 to 0.8 of its speed. English has
# fewer, longer syllables.
DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "es": (4.4, 6.0),
    "ca": (4.4, 6.0),
    "it": (4.4, 6.0),
    "pt": (4.4, 6.0),
    "fr": (4.2, 5.6),
    "en": (3.6, 5.0),
    "de": (3.6, 5.0),
}
FALLBACK_RATES = (4.0, 5.5)

DEFAULT_UPSET_WORDS: dict[str, tuple[str, ...]] = {
    "es": (
        "harto", "harta", "hartos", "vergüenza", "vergonzoso", "inaceptable", "inadmisible", "indignado", "indignada",
        "indignante", "cabreado", "cabreada", "enfadado", "enfadada", "estafa", "estafando", "timo", "engaño",
        "reclamación", "denuncia", "denunciar", "tomadura de pelo", "tomando el pelo", "no puede ser", "ya está bien",
        "es increíble", "estoy flipando", "me parece fatal", "un desastre", "una vergüenza", "nadie me",
        "mierda", "joder", "coño", "hostia", "puta", "cojones",
    ),
    "ca": (
        "fart", "farta", "vergonya", "vergonyós", "inacceptable", "indignat", "indignada", "emprenyat", "emprenyada",
        "estafa", "engany", "reclamació", "denúncia", "denunciar", "presa de pèl", "no pot ser", "ja n'hi ha prou",
        "un desastre", "una vergonya", "ningú em", "merda", "collons",
    ),
    "en": (
        "fed up", "sick of", "ridiculous", "unacceptable", "outrageous", "furious", "angry", "frustrated",
        "frustrating", "terrible", "awful", "disgrace", "a joke", "scam", "rip-off", "ripped off", "complaint",
        "nobody has", "no one has", "worst", "shit", "fuck", "fucking", "damn", "bullshit", "crap",
    ),
}  # fmt: skip

MIN_SYLLABLES = 8
"""An utterance shorter than this says nothing about pace ("sí, claro")."""
MIN_SPEECH_SECS = 1.0
MAX_RATE = 9.5
"""Nobody talks faster. More means the seconds were not all measured (talk over the agent)."""
MIN_RATE = 1.5
"""Slower is not speech either: noise counted as talking."""
SMOOTHING = 0.5
"""Weight of the newest utterance in the caller's pace."""

_WORD = re.compile(r"[^\W\d_]+")
_VOWELS = re.compile(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûü]+")
# A stressed "í" or "ú" next to another vowel is a syllable of its own ("que-rí-a", "pa-ís").
_HIATUS = re.compile(r"(?<=[aeoàáèéòó])[íú]|[íú](?=[aeoàáèéòó])")


def syllables(text: str) -> int:
    """Syllables of ``text``: groups of vowels, one at least per word.

    Close enough for a ratio. Spanish "te-a-tro" is undercounted and English
    silent vowels ("made") overcounted. The rates above are in this unit.
    """
    return sum(max(1, len(_VOWELS.findall(word)) + len(_HIATUS.findall(word))) for word in _WORD.findall(text.lower()))


def _fold(text: str) -> str:
    stripped = "".join(ch for ch in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\w']+", " ", stripped).split())


def _upset_pattern(words: Sequence[str]) -> re.Pattern[str] | None:
    folded = sorted({_fold(word) for word in words if _fold(word)}, key=len, reverse=True)
    if not folded:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(word) for word in folded) + r")(?!\w)")


class CallerPace:
    """The factor on ``tts_tempo`` for the next reply of one call.

    :meth:`heard` once for each thing the caller said that gets a reply, with
    the seconds of speech since the one before. A transcript that repeats the
    last one and goes on ("quiero cambiar" then "quiero cambiar la cuenta") is
    the same utterance still growing: it replaces what the first said about
    the caller, with the seconds of both. :meth:`preview` is the factor a
    transcript would give without keeping it, for a reply started before its
    transcript is confirmed.
    """

    def __init__(
        self,
        *,
        slowest: float = 0.9,
        slow_rate: float | None = None,
        normal_rate: float | None = None,
        upset_tempo: float | None = 0.93,
        upset_turns: int = 3,
        upset_words: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.slowest = slowest
        self.slow_rate = slow_rate
        self.normal_rate = normal_rate
        self.upset_tempo = upset_tempo
        self.upset_turns = upset_turns
        self._upset_words = upset_words
        self._upset_patterns: dict[str, re.Pattern[str] | None] = {}
        self.rate: float | None = None
        """The caller's pace so far, in syllables per second. ``None`` until an utterance measured it."""
        self.upset_left = 0
        """Replies, the next one included, spoken at ``upset_tempo``."""
        self.last_measured: float | None = None
        """The rate of the utterance last heard, or ``None`` when it said nothing about pace."""
        # What was known before the utterance last heard, and that utterance:
        # a longer transcript of it starts again from here.
        self._before: tuple[float | None, int] = (None, 0)
        self._last: tuple[str, float] = ("", 0.0)

    # -- what the caller said ------------------------------------------------

    def rates(self, language: str | None) -> tuple[float, float]:
        """``(slow, normal)`` syllables per second for ``language``."""
        slow, normal = DEFAULT_RATES.get(language_key(language), FALLBACK_RATES)
        slow = self.slow_rate if self.slow_rate is not None else slow
        normal = self.normal_rate if self.normal_rate is not None else normal
        return (slow, normal) if normal > slow else (slow, slow + 0.1)

    @staticmethod
    def measure(text: str, speech_secs: float | None) -> float | None:
        """Syllables per second of one utterance, or ``None`` when it says nothing about pace."""
        if speech_secs is None or speech_secs < MIN_SPEECH_SECS or any(ch.isdigit() for ch in text):
            return None
        count = syllables(text)
        if count < MIN_SYLLABLES:
            return None
        rate = count / speech_secs
        return rate if MIN_RATE <= rate <= MAX_RATE else None

    def is_upset(self, text: str, language: str | None) -> bool:
        if self.upset_tempo is None or self.upset_turns <= 0:
            return False
        key = language_key(language)
        if key not in self._upset_patterns:
            words = self._upset_words
            if isinstance(words, Mapping):
                chosen = words.get(key)
                if chosen is None:
                    chosen = words.get("*")
            else:
                chosen = words
            if chosen is None:
                # A language without a list (or none pinned: the reply follows the caller) reads every list.
                chosen = DEFAULT_UPSET_WORDS.get(key) or [w for ws in DEFAULT_UPSET_WORDS.values() for w in ws]
            self._upset_patterns[key] = _upset_pattern(chosen)
        pattern = self._upset_patterns[key]
        return pattern is not None and pattern.search(_fold(text)) is not None

    # -- the factor -----------------------------------------------------------

    def _after(self, text: str, speech_secs: float | None, language: str | None):
        """State after hearing ``text``: ``(rate, upset_left, measured, before, last)``."""
        folded = _fold(text)
        previous_text, previous_secs = self._last
        before = (self.rate, self.upset_left)
        total = speech_secs
        if previous_text and folded != previous_text and folded.startswith(previous_text + " "):
            # The same utterance, longer: what its first part said is taken back.
            before = self._before
            total = None if speech_secs is None else previous_secs + speech_secs
        rate, upset_left = before
        measured = self.measure(text, total)
        if measured is not None:
            rate = measured if rate is None else rate + (measured - rate) * SMOOTHING
        upset_left = self.upset_turns if self.is_upset(text, language) else max(0, upset_left - 1)
        return rate, upset_left, measured, before, (folded, total or 0.0)

    def _factor(self, rate: float | None, upset_left: int, language: str | None) -> float:
        pace = 1.0
        if rate is not None:
            slow, normal = self.rates(language)
            share = min(1.0, max(0.0, (rate - slow) / (normal - slow)))
            pace = self.slowest + (1.0 - self.slowest) * share
        if upset_left > 0 and self.upset_tempo is not None:
            pace = min(pace, self.upset_tempo)
        return round(pace, 2)

    def heard(self, text: str, speech_secs: float | None, language: str | None) -> float:
        """The caller said ``text`` in ``speech_secs`` of speech. Returns the factor for the reply."""
        self.rate, self.upset_left, self.last_measured, self._before, self._last = self._after(
            text, speech_secs, language
        )
        return self.factor(language)

    def preview(self, text: str, speech_secs: float | None, language: str | None) -> float:
        """The factor :meth:`heard` would return, keeping nothing."""
        rate, upset_left, *_ = self._after(text, speech_secs, language)
        return self._factor(rate, upset_left, language)

    def factor(self, language: str | None) -> float:
        """The factor for a reply now, from what has been heard."""
        return self._factor(self.rate, self.upset_left, language)
