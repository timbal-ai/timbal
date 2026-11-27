"""Sentence and clause boundaries for streaming TTS.

The rules are the ones LiveKit Agents and Pipecat already run in production,
ported without their tokenizer dependencies (NLTK, sentencex, blingfire):

* LiveKit Agents ``tokenize/_basic_sent.py`` (Apache-2.0, Copyright 2023-2026
  LiveKit, Inc.): title and suffix abbreviations (``Mr.``, ``Dr.``, ``Inc.``),
  single-letter initials, dotted acronyms (``U.S.``, ``a.m.``, ``Ph.D.``),
  ellipsis, and dots that sit between digits (``3.14``, ``192.168.1.1``).
* Pipecat ``utils/string.py`` end-of-sentence cases (BSD-2-Clause): a comma is
  not a sentence end, decimals and ``a.m.`` / ``p.m.`` stay intact, and a
  thousands separator (``1,234`` or ``1, 234``) is not a clause break.
  Spanish titles (``Sr.``, ``Sra.``, ``Dra.``, ``Ud.``) and ``EE. UU.`` follow
  the same patterns.

Indexes returned here are positions in the original string, immediately after
the whitespace that follows the boundary, matching the flush prefix the
session already sends to TTS.
"""

from __future__ import annotations

import re

# Sentence terminators the flush already treated as clause ends, including
# Arabic ؟/؛ and fullwidth ？/！. "." is the only one that is ambiguous.
_SENTENCE_TERMINATORS = frozenset(".!?\n;:\u061f\u061b\uff1f\uff01")

# Commas that can open the first TTS segment. Semicolons and colons are
# already sentence terminators; a comma is not.
_CLAUSE_COMMAS = frozenset(",\uff0c\u060c")  # "," fullwidth "，" Arabic "،"

# Titles and suffixes whose period is not the end of a sentence.
# English list from LiveKit's prefixes/suffixes; Spanish is the same idea.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "capt",
        "cpt",
        "lt",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
        "co",
        "vs",
        "etc",
        "fig",
        "vol",
        "gen",
        "hon",
        "rev",
        "sra",
        "srta",
        "dra",
        "profa",
        "ud",
        "uds",
        "vd",
        "vds",
        "dña",
        "da",
        "don",
        "doña",
        "sto",
        "sta",
        "lic",
        "ing",
        "arq",
        "av",
        "art",
        "cap",
        "tel",
        "aprox",
        "atte",
        "dpto",
        "depto",
        "dto",
        "pág",
        "pags",
        "págs",
        "ej",
    }
)

# "No. 5" / "núm. 3" is a label; "No. Thanks." is a sentence.
_NUMBER_ABBREVIATIONS = frozenset({"no", "nos", "num", "núm", "nro", "nr"})

# Two or more letter-groups joined by periods: "U.S.", "a.m.", "Ph.D.",
# "N.A.S.A.", "EE. UU.", "p. ej.". One group ("OK.") is an ordinary sentence.
_DOTTED_ABBREV = re.compile(r"(?<![\w.])(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{1,3}\.\s*){2,}")


def word_count(text: str) -> int:
    """Whitespace-separated tokens that contain a letter or digit."""
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def sentence_boundary_ends(text: str) -> list[int]:
    """Indexes just after the whitespace following each real sentence end."""
    dotted = [m.span() for m in _DOTTED_ABBREV.finditer(text)]
    return _boundary_ends(text, dotted, commas=False)


def clause_boundary_ends(text: str) -> list[int]:
    """Indexes just after the whitespace following a comma clause break.

    The comma must have a non-digit lookahead, so ``12,`` and ``1,234`` stay
    one token until the next word shows the comma was a clause. No lookahead
    means "still streaming" — the caller flushes the tail when generation ends.
    """
    dotted = [m.span() for m in _DOTTED_ABBREV.finditer(text)]
    return _boundary_ends(text, dotted, commas=True)


def _boundary_ends(text: str, dotted: list[tuple[int, int]], *, commas: bool) -> list[int]:
    marks = _CLAUSE_COMMAS if commas else _SENTENCE_TERMINATORS
    ends: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch not in marks:
            i += 1
            continue
        if not commas and ch == "." and not _period_ends_sentence(text, i, dotted):
            i += 1
            continue
        j = i + 1
        if j >= n or not text[j].isspace():
            # "3.14" and "3:00" have no space. A comma glued to the next word
            # ("days,so") still needs a lookahead that isn't a digit.
            if commas and j < n and not text[j].isdigit():
                ends.append(j)
                i = j
                continue
            i += 1
            continue
        while j < n and text[j].isspace():
            j += 1
        if commas:
            if j >= n or text[j].isdigit():
                i += 1
                continue
        elif j >= n:
            i += 1
            continue
        ends.append(j)
        i = j
    return ends


def _period_ends_sentence(text: str, i: int, dotted: list[tuple[int, int]]) -> bool:
    # A run of dots ("Well...", "Wait...") is an ellipsis, not a sentence end.
    # LiveKit protects every dot in the run; only the last one used to leak
    # through here and flush "Well..." before a later comma could.
    if (i > 0 and text[i - 1] == ".") or (i + 1 < len(text) and text[i + 1] == "."):
        return False
    prev = text[i - 1] if i else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if prev.isdigit() and nxt.isdigit():
        return False
    if any(start <= i < end for start, end in dotted):
        return False
    token = _token_before(text, i)
    # "A." / "U." is an initial. "n't." is a contraction ("I don't."), not one.
    if len(token) == 1 and token.isalpha() and not (i >= 2 and text[i - 2] in "'’"):
        return False
    key = token.casefold()
    if key in _ABBREVIATIONS:
        return False
    if key in _NUMBER_ABBREVIATIONS:
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j].isdigit():
            return False
    return True


def _token_before(text: str, i: int) -> str:
    j = i
    while j > 0 and text[j - 1].isalnum():
        j -= 1
    return text[j:i]
