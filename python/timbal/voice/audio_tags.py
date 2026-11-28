"""Laughs and sighs the voice performs: ``[laughs]`` written inside the reply.

Some voices read a bracketed mark as a sound and not as words. Measured
through the adapters (``benchmarks/voice/tts_audio_tags.py``, Spanish
sentences, the marks in English): the ElevenLabs dialogue models (Eleven v3,
v3 conversational, v4 turbo) performed every one, a transcriber that labels
sounds heard a laugh, a sigh, a throat being cleared and none of the mark's
words. ElevenLabs Flash v2.5 and Multilingual v2 said them aloud:
"Chuckles. Claro que sí...", "Clears throat. Perdona...".

So ``VoiceConfig.audio_tags`` does two things, and which one depends on the
voice of the call:

* A voice that performs them (``TextToSpeech.performs_audio_tags``): the
  model is told, in the system prompt, which marks exist and to use them
  rarely. What it writes is limited before it reaches the voice: at most
  ``max_per_reply`` in a reply, and none in the ``rest_replies`` replies after
  one that had one. A person who laughs in every sentence is not relaxed.
* Any other voice: the model is told nothing, and a mark it writes anyway
  (it is in the agent's own prompt, or in the history of an earlier call) is
  taken out of the text sent to the voice.

A bracketed word that is not in the list is taken out for both: the list is
what the voice may do. Only the text sent to the voice changes. Reply deltas,
the transcript and the model's memory keep what the model wrote, so a client
that shows the reply can show the mark or drop it (:func:`strip_audio_tags`).

Pure text. :class:`AudioTagPolicy` holds the state of one call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_AUDIO_TAGS: tuple[str, ...] = ("laughs", "chuckles", "sighs", "exhales", "clears throat")
"""Measured as performed on the ElevenLabs dialogue models. Sounds a person on
the phone makes; delivery marks (``[whispers]``, ``[excited]``) change the
whole line and are left to whoever lists them."""

# A bracketed word or two: letters, spaces, apostrophes, hyphens. Not "[1]", not "[a, b]".
_TAG = re.compile(r"\[\s*([^\W\d_](?:[^\W\d_]|[ '’\-]){0,30}?)\s*\]")


def tag_name(raw: str) -> str:
    """``" Clears  Throat "`` → ``"clears throat"``."""
    return " ".join(raw.lower().split())


def audio_tags_note(tags: Sequence[str]) -> str:
    """The system-prompt lines for a voice that performs ``tags``."""
    marks = ", ".join(f"[{tag}]" for tag in tags)
    return (
        "The voice on this call performs these marks as sounds when you write them inside a reply, "
        f"square brackets included: {marks}. Write one only where a person on the phone would make that sound, "
        "in a few replies of the call at most and never twice in one reply. Write it exactly as listed, in English, "
        "whatever language you are speaking. Never in a greeting or a goodbye, never next to data "
        "(numbers, dates, codes, names), and never for a feeling the conversation does not have. "
        "Nothing else in brackets is spoken."
    )


def _without(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` minus ``spans``, without the doubled space a removed mark leaves."""
    out: list[str] = []
    position = 0
    for start, end in spans:
        out.append(text[position:start])
        position = end
        kept = "".join(out)
        if not kept or kept[-1].isspace():
            while position < len(text) and text[position] in " \t":
                position += 1
    out.append(text[position:])
    return "".join(out)


def strip_audio_tags(text: str) -> str:
    """``text`` without any bracketed mark: what was said in words. For captions and logs."""
    return _without(text, [match.span() for match in _TAG.finditer(text)])


class AudioTagPolicy:
    """Which marks of a reply reach the voice.

    :meth:`start_reply` once per reply, then :meth:`voiced` for each piece of
    it in the order it is spoken. :meth:`state` and :meth:`restore` rewind a
    reply nobody heard (a speculative turn that was dropped).
    """

    def __init__(self, tags: Sequence[str], *, max_per_reply: int = 1, rest_replies: int = 1) -> None:
        self.tags = tuple(dict.fromkeys(tag_name(tag) for tag in tags if tag_name(tag)))
        self.max_per_reply = max_per_reply
        self.rest_replies = rest_replies
        self._allowed = frozenset(self.tags)
        self._in_reply = 0
        # Replies still to go without a mark after one that had one.
        self._resting = 0

    def state(self) -> tuple[int, int]:
        return self._in_reply, self._resting

    def restore(self, state: tuple[int, int]) -> None:
        self._in_reply, self._resting = state

    def start_reply(self) -> None:
        """A new reply begins: the one before it counts as rested, or starts the rest."""
        if self._in_reply:
            self._resting = self.rest_replies
        elif self._resting:
            self._resting -= 1
        self._in_reply = 0

    def voiced(self, text: str, *, performed: bool, limited: bool = True) -> tuple[str, int]:
        """``text`` for the voice, and how many marks it keeps.

        ``performed`` is whether this call's voice makes the sound. ``limited``
        is False for a line the application wrote itself (a greeting, a filler
        phrase): its marks are its author's choice, and it is not a reply.
        """
        if "[" not in text:
            return text, 0
        dropped: list[tuple[int, int]] = []
        kept = 0
        for match in _TAG.finditer(text):
            allowed = performed and tag_name(match.group(1)) in self._allowed
            if allowed and limited:
                allowed = not self._resting and self._in_reply < self.max_per_reply
                if allowed:
                    self._in_reply += 1
            if allowed:
                kept += 1
            else:
                dropped.append(match.span())
        return (_without(text, dropped) if dropped else text), kept
