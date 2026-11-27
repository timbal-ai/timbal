"""A spoken form for words the voice reads badly, in the text sent to the voice and nowhere else.

A neural voice decides take by take how to read an acronym. Measured on
ElevenLabs ``eleven_v3_conversational`` (``benchmarks/voice/tts_acronyms.py``,
18 takes of each, the time from the word before the term to the word after
it): "DNI" took a median of 0.95 s and anything from 0.54 to 1.90 s, "NIE"
1.32 s (0.56 to 2.24) and "CIF" 1.20 s (0.90 to 1.84). Written the way it is
said, "deneí" took 0.48 s (0.34 to 0.88), "níe" 0.43 s (0.26 to 0.72) and
"cif" 0.46 s (0.38 to 0.74). Other voices need other terms, or none:
``eleven_multilingual_v2`` read the same acronyms in 0.45 to 0.60 s as written.

``VoiceConfig.pronunciations`` maps a term to that spoken form. Reply deltas,
the transcript and the model's memory keep the model's own spelling.

Pure text. :class:`Pronouncer` holds the compiled terms of one call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .config import language_key

ANY_LANGUAGE = "*"


class Pronouncer:
    """Replaces whole terms, in the letter case they were given.

    ``pronunciations`` is ``{term: spoken}`` for every language, or
    ``{language: {term: spoken}}`` with ``"*"`` for all of them (a language's
    own entry for a term wins). A term matches when no letter, digit or
    underscore touches it, so ``"NIE"`` is replaced in ``"el NIE."`` and left
    alone in ``"NIEs"`` and ``"DNIE"``. The longest term wins where two overlap.
    """

    def __init__(self, pronunciations: Mapping[str, str] | Mapping[str, Mapping[str, str]] | None) -> None:
        self._by_language: dict[str, dict[str, str]] = {}
        for key, value in (pronunciations or {}).items():
            if isinstance(value, Mapping):
                self._by_language.setdefault(language_key(key) or ANY_LANGUAGE, {}).update(value)
            else:
                self._by_language.setdefault(ANY_LANGUAGE, {})[key] = value
        self._compiled: dict[str, tuple[re.Pattern[str], dict[str, str]] | None] = {}

    def __bool__(self) -> bool:
        return any(self._by_language.values())

    def terms(self, language: str | None) -> dict[str, str]:
        """The terms that apply to ``language``: the ones for every language, then its own."""
        terms = dict(self._by_language.get(ANY_LANGUAGE, {}))
        key = language_key(language)
        if key and key != ANY_LANGUAGE:
            terms.update(self._by_language.get(key, {}))
        return terms

    def apply(self, text: str, language: str | None) -> str:
        """``text`` with each term replaced by its spoken form."""
        key = language_key(language)
        if key not in self._compiled:
            terms = self.terms(language)
            pattern = None
            if terms:
                longest_first = sorted(terms, key=len, reverse=True)
                pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(term) for term in longest_first) + r")(?!\w)")
            self._compiled[key] = (pattern, terms) if pattern is not None else None
        compiled = self._compiled[key]
        if compiled is None or not text:
            return text
        pattern, terms = compiled
        return pattern.sub(lambda match: terms[match.group()], text)
