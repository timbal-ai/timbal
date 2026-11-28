"""A hesitation sound at the opening of some replies.

A person answering on the phone does not open every sentence cleanly. Now and
then the first thing out is a sound while the thought forms: "Eh... son diez
con sesenta." A model writes none of that unless it is prompted to, and then
unevenly. This adds the sound at a set rate, to the text sent to the voice and
nowhere else: the reply the client receives, the transcript and the model's
memory keep the model's own words.

Many replies open with a word or two of acknowledgement ("Perfecto.", "Vale,
Marta."). The sound then goes after it, where the thought starts: "Perfecto.
Eh... tu factura es de cuarenta euros." Without that, an agent whose replies
open that way never hesitated: measured on two demo agents, every reply of
their test calls did.

Pure text. :class:`Hesitator` holds one call's state. The session shows it the
first segment it speaks in each model step: the start of a turn, and the first
speech after a tool call. When that segment is only an acknowledgement, it
shows it the next one too.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .config import language_key

DEFAULT_OPENERS: dict[str, tuple[str, ...]] = {
    "es": ("eh...", "mm...", "pues...", "a ver,", "mm,", "eh,"),
    "en": ("um,", "well,", "so,", "mm,"),
    "ca": ("eh...", "mm...", "doncs...", "a veure,", "mm,", "eh,"),
}
"""The sounds used for a language that ``HesitationConfig.openers`` does not
cover. Written the way the voice should read them: an ellipsis is a longer
pause than a comma."""

SOUND_AUDIO_WEIGHT = 2
"""A sound takes about this many times the audio of as many characters of the
sentence behind it, because it carries its own pause. On ElevenLabs
``eleven_v4_turbo`` the sounds above added 0.33 to 0.93 s (median 0.58 s),
which is 0.8 to 3.5 times their share by characters (median 2.0), over 32
syntheses. The session uses it to tell how much of a segment's audio was the
sound when a reply is cut short."""

MIN_OPENING_WORDS = 3
"""The sound goes before a clause of at least this many words. One or two
words before a comma or a full stop are an acknowledgement, a greeting, a name
or a goodbye ("Vale,", "De acuerdo.", "Hasta luego,"), not a thought."""

MAX_ACKNOWLEDGEMENT_CLAUSES = 2
"""How many such short clauses the sound may go after ("Perfecto, Marta. Eh...
tu factura..."). A reply that opens with more of them is said clean."""

GREETING_WORDS = frozenset(
    {
        # es
        "hola", "buenas", "buenos", "adios", "hasta", "chao", "bienvenido", "bienvenida", "encantado", "encantada",
        # ca
        "adeu", "fins", "bon", "bona", "benvingut", "benvinguda", "encantat",
        # en
        "hello", "hi", "hey", "bye", "goodbye", "welcome", "morning", "afternoon", "evening", "night",
    }
)  # fmt: skip
"""Words that make a short opening clause a greeting or a goodbye, without
accents. Nothing is added after one: "Hola, eh... soy Laura" and "Hasta luego,
eh... que vaya bien" are not how a person hesitates."""

# Punctuation that closes a sentence. After any other clause end (a comma, a
# colon, a dash) the sound is in the middle of one and keeps its lower case.
_SENTENCE_END = ".!?…\n"

# Punctuation that closes the first clause. It needs a space or the end after
# it, so the comma in "10,60" and the dot in "10.60" do not count.
_CLAUSE_END = re.compile(r"[,.;:!?…—–](?=\s|$)|\n")
# A sound the model wrote on its own: "Mm,", "Ehh...", "Um.", "Hmm —". The
# punctuation is required because "um" and "er" are words in some languages.
_BARE_SOUND = re.compile(
    r"(?:e+(?:h+m*|m+)|u+(?:h+m*|m+)|h*m+|a+h+|e+r+m*)\s*(?:[,.…—–-]|$)",
    re.IGNORECASE,
)


class Hesitated(NamedTuple):
    """One segment with a hesitation sound added."""

    offset: int
    """Where the sound was inserted in the original segment (after its leading whitespace)."""
    opener: str
    """The sound as spoken, with the space that follows it: ``"Eh... "``."""
    spoken: str
    """The whole segment to send to the voice. Apart from the sound it is the
    original, character for character, except for the case of one letter."""


def _core(opener: str) -> str:
    return opener.rstrip(" ,.…").lower()


def _opens_with(body: str, core: str) -> bool:
    if not core or body[: len(core)].lower() != core:
        return False
    return len(body) == len(core) or not body[len(core)].isalpha()


def _fold(word: str) -> str:
    """``word`` in lower case, without accents or the punctuation around it."""
    plain = "".join(ch for ch in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(ch))
    return plain.strip("¡!¿?\"'()[]«»")


def _is_greeting(clause: str) -> bool:
    return any(_fold(word) in GREETING_WORDS for word in clause.split())


class Hesitator:
    """Decides, one reply opening at a time, whether it gets a hesitation sound.

    The sound goes before the first clause of :data:`MIN_OPENING_WORDS` words
    or more: at the very start, or after one or two short clauses of
    acknowledgement ("Vale,", "Perfecto, Marta."). An opening gets one when
    all of these hold:

    * the language has openers (``openers``, else :data:`DEFAULT_OPENERS`);
    * that clause does not already open with one, or with a bare sound and
      punctuation ("Mm,", "Ehh..."). Such an opening counts as hesitated;
    * it starts with a letter, so not with a figure, ``¿``, ``¡`` or a bracket;
    * the short clauses before it are at most
      :data:`MAX_ACKNOWLEDGEMENT_CLAUSES` and none is a greeting or a goodbye
      (:data:`GREETING_WORDS`);
    * the opening before it was not hesitated;
    * a random draw falls under ``rate``.

    The sound is never the one used the time before, when there is another.
    """

    def __init__(
        self,
        rate: float,
        openers: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.rate = rate
        self._openers = openers
        self._rng = random.Random(seed)
        self._previous = False
        self._last_opener = ""

    def openers(self, language: str | None) -> tuple[str, ...]:
        """The sounds for ``language``: the configured ones, else the built-in list, else none."""
        key = language_key(language)
        configured = self._openers
        if isinstance(configured, Mapping):
            found = configured.get(key) or configured.get("*")
            if found:
                return tuple(found)
        elif configured:
            return tuple(configured)
        return DEFAULT_OPENERS.get(key, ())

    def state(self) -> tuple[bool, str]:
        """What :meth:`restore` needs to undo the openings considered since."""
        return self._previous, self._last_opener

    def restore(self, state: tuple[bool, str]) -> None:
        """Forget the openings considered since :meth:`state`: nobody heard them."""
        self._previous, self._last_opener = state

    def consider(self, text: str, language: str | None, *, mid_sentence: bool = False) -> tuple[Hesitated | None, str]:
        """Return ``text`` with a sound added, or ``None``, and the reason either way.

        Reasons: ``added``, ``no_openers``, ``already``, ``not_a_word``,
        ``short_opening``, ``acknowledgement``, ``after_hesitation``, ``draw``.
        Blank text is ``not_a_word``. ``acknowledgement`` is a segment that is
        only short clauses ("Perfecto."): the opening is still to come, so the
        caller shows the next segment. Both leave the state alone.

        ``mid_sentence`` says ``text`` continues a sentence (the segment before
        it ended on a comma): a sound at its start keeps its lower case.
        """
        openers = self.openers(language)
        if not openers:
            return None, "no_openers"
        body = text.lstrip()
        if not body:
            return None, "not_a_word"
        start, sentence_start, found = self._opening(body, openers, sentence_start=not mid_sentence)
        if found == "already":
            self._previous = True
            return None, "already"
        if found == "acknowledgement":
            return None, "acknowledgement"
        after_hesitation, self._previous = self._previous, False
        if found:
            return None, found
        if after_hesitation:
            return None, "after_hesitation"
        if self._rng.random() >= self.rate:
            return None, "draw"
        choices = [opener for opener in openers if opener != self._last_opener] or list(openers)
        opener = self._rng.choice(choices)
        self._previous, self._last_opener = True, opener
        offset = len(text) - len(body) + start
        rest = body[start:]
        sound = opener + " "
        if sentence_start:
            # The sound opens the sentence now, so it takes the capital and the
            # next word gives it up. An acronym ("DNI") keeps its own.
            sound = sound[0].upper() + sound[1:]
            first = rest[0].lower()
            if len(rest) > 1 and rest[1].islower() and len(first) == 1:
                rest = first + rest[1:]
        return Hesitated(offset, sound, text[:offset] + sound + rest), "added"

    @staticmethod
    def _opening(body: str, openers: Sequence[str], *, sentence_start: bool) -> tuple[int, bool, str]:
        """Where in ``body`` the sound would go, whether a sentence starts there, and why not ("" when it can)."""
        start = 0
        skipped = 0
        while True:
            rest = body[start:]
            if _BARE_SOUND.match(rest) or any(_opens_with(rest, _core(opener)) for opener in openers):
                return start, sentence_start, "already"
            if not rest[0].isalpha():
                return start, sentence_start, "not_a_word"
            end = _CLAUSE_END.search(rest)
            clause = rest[: end.start()] if end else rest
            if len(clause.split()) >= MIN_OPENING_WORDS:
                return start, sentence_start, ""
            if end is None or _is_greeting(clause):
                return start, sentence_start, "short_opening"
            after = rest[end.end() :]
            if not after.strip():
                # Nothing but short clauses: the thought is in a later segment.
                return start, sentence_start, "acknowledgement"
            if skipped == MAX_ACKNOWLEDGEMENT_CLAUSES:
                return start, sentence_start, "short_opening"
            skipped += 1
            sentence_start = end.group() in _SENTENCE_END
            start += end.end() + len(after) - len(after.lstrip())
