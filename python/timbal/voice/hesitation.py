"""A hesitation sound at the opening of some replies.

A person answering on the phone does not open every sentence cleanly. Now and
then the first thing out is a sound while the thought forms: "Eh... son diez
con sesenta." A model writes none of that unless it is prompted to, and then
unevenly. This adds the sound at a set rate, to the text sent to the voice and
nowhere else: the reply the client receives, the transcript and the model's
memory keep the model's own words.

Pure text. :class:`Hesitator` holds one call's state. The session shows it the
first segment it speaks in each model step: the start of a turn, and the first
speech after a tool call.
"""

from __future__ import annotations

import random
import re
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
"""A reply whose first clause is shorter than this is not hesitated. One or
two words before the first comma or full stop are an acknowledgement, a
greeting, a name or a goodbye ("Vale,", "De acuerdo.", "Hasta luego,")."""

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


class Hesitator:
    """Decides, one reply opening at a time, whether it gets a hesitation sound.

    An opening gets one when all of these hold:

    * the language has openers (``openers``, else :data:`DEFAULT_OPENERS`);
    * it does not already open with one, or with a bare sound and punctuation
      ("Mm,", "Ehh..."). Such an opening counts as hesitated;
    * it starts with a letter, so not with a figure, ``¿``, ``¡`` or a bracket;
    * its first clause has at least :data:`MIN_OPENING_WORDS` words;
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

    def consider(self, text: str, language: str | None) -> tuple[Hesitated | None, str]:
        """Return ``text`` with a sound added, or ``None``, and the reason either way.

        Reasons: ``added``, ``no_openers``, ``already``, ``not_a_word``,
        ``short_opening``, ``after_hesitation``, ``draw``. Blank text is
        ``not_a_word`` and leaves the state alone.
        """
        openers = self.openers(language)
        if not openers:
            return None, "no_openers"
        body = text.lstrip()
        if not body:
            return None, "not_a_word"
        offset = len(text) - len(body)
        if _BARE_SOUND.match(body) or any(_opens_with(body, _core(opener)) for opener in openers):
            self._previous = True
            return None, "already"
        after_hesitation, self._previous = self._previous, False
        if not body[0].isalpha():
            return None, "not_a_word"
        if len(_CLAUSE_END.split(body, maxsplit=1)[0].split()) < MIN_OPENING_WORDS:
            return None, "short_opening"
        if after_hesitation:
            return None, "after_hesitation"
        if self._rng.random() >= self.rate:
            return None, "draw"
        choices = [opener for opener in openers if opener != self._last_opener] or list(openers)
        opener = self._rng.choice(choices)
        self._previous, self._last_opener = True, opener
        # The sound opens the sentence now, so it takes the capital and the
        # reply's first word gives it up. An acronym ("DNI") keeps its own.
        sound = opener[0].upper() + opener[1:] + " "
        first = body[0].lower()
        if len(body) > 1 and body[1].islower() and len(first) == 1:
            body = first + body[1:]
        return Hesitated(offset, sound, text[:offset] + sound + body), "added"
