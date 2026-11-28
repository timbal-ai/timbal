"""Listening sounds: "ajá", "mm-hm", said while the caller is still talking.

Someone who listens on the phone is not silent. When the other person pauses
in the middle of a long explanation, or between two groups of a number they
are dictating, the listener says "ajá" and the speaker goes on. An agent that
says nothing until the turn is over leaves that pause empty, and callers fill
it themselves: "¿hola? ¿me oyes?".

``VoiceConfig.backchannel`` fills it. The sound is not a turn: no model is
called, nothing is added to the transcript or to the model's memory, and the
caller who keeps talking over it is not interrupting anything.

**When.** The caller has just gone quiet (``pause_secs`` of silence on the
mic, see :class:`~timbal.voice.mic_activity.MicActivity`), has said enough
since the last sound (``min_speech_secs``), and one of these says they are
not done:

* ``entity``: what they have said ends in a number, a spelling, an email or a
  code that is still open (:mod:`timbal.voice.entity_patience`).
* ``held``: the transcriber committed and the turn detector is holding the
  commit for the rest of the phrase.
* ``unfinished``: the caption ends on a word that leaves the phrase open
  ("y", "porque", "mi número es").
* ``long``: they have been talking for ``long_speech_secs`` and the caption
  does not end a sentence (a period, a question mark, an exclamation mark).

The first three are evidence that more is coming. The last is only that they
stopped in the middle of something. A long explanation that ends on a
finished sentence gets none: in real calls that sound fell half a second
before the endpointer ended the turn, and the caller heard "mm-hm" and then
the reply (``benchmarks/voice/backchannel_call.py``).

Never while the agent is speaking or about to (a turn running, a reply being
prepared ahead of its transcript, the greeting, a goodbye), never twice in
``min_gap_secs``, at most ``max_per_turn`` times before the agent's next
reply, and only for ``rate`` of the pauses that qualify: a listener who
answers every pause sounds like a machine too.

**What.** A phrase of the reply language, not the one said last. The clips are
synthesized once per call with the agent's own voice, trimmed, and played
quieter than a reply (``volume``), so the sound is there the instant the
pause is long enough. A sound spoken on demand would arrive after the caller
has started talking again.

:class:`Backchanneler` decides. The clip helpers work on PCM16 mono. Neither
knows about the session.
"""

from __future__ import annotations

import random
import sys
from array import array
from collections.abc import Mapping, Sequence

from .config import language_key

ANY_LANGUAGE = "*"

# Each was heard as itself in at least 10 of 12 takes by each ElevenLabs model
# tried (v3 conversational and v4 Turbo: every take; Flash v2.5 and
# Multilingual v2: 10 to 12) and in 6 of 6 by Gemini 3.8 Flash TTS, and no take
# was longer than 0.86 s (``benchmarks/voice/tts_backchannels.py``). Left out
# on purpose: "sí" and "yes", which a caller can take for an answer; "ya",
# which Multilingual v2 said as "yeah" or as a question in 5 of 6 takes;
# Catalan "d'acord", heard as that in 10 of 24 takes (as "decor" and "dot org"
# on Flash).
DEFAULT_BACKCHANNELS: dict[str, tuple[str, ...]] = {
    "es": ("Ajá.", "Mm-hm.", "Vale."),
    "ca": ("Ahà.", "Mm-hm."),
    "en": ("Mm-hm.", "Uh-huh.", "Right.", "Okay."),
    ANY_LANGUAGE: ("Mm-hm.",),
}

REASONS = ("entity", "held", "unfinished", "long")

PAUSE_WINDOW_SECS = 0.6
"""How long after ``pause_secs`` a sound can still start. Later, the moment has passed."""
ENTITY_MIN_SPEECH_SECS = 0.8
"""A group of a number is short ("seis cero cero"): it needs less speech than a phrase."""
MAX_CLIP_SECS = 1.2
"""A clip longer than this is not a listening sound: the voice said something else."""
MIN_CLIP_SECS = 0.08
CLIP_EDGE_SECS = 0.02
"""Silence kept on each side of a trimmed clip."""
CLIP_QUIET_DBFS = -42.0


class Backchanneler:
    """When a listening sound is due in one call, and which one."""

    def __init__(
        self,
        phrases: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        *,
        pause_secs: float = 0.45,
        min_speech_secs: float = 1.5,
        min_gap_secs: float = 3.0,
        max_per_turn: int = 3,
        rate: float = 0.7,
        long_speech_secs: float | None = 6.0,
        seed: int | None = None,
    ) -> None:
        self._phrases = phrases
        self.pause_secs = pause_secs
        self.min_speech_secs = min_speech_secs
        self.min_gap_secs = min_gap_secs
        self.max_per_turn = max_per_turn
        self.rate = rate
        self.long_speech_secs = long_speech_secs
        self._rng = random.Random(seed)
        self._last_phrase: str | None = None
        self._last_at: float | None = None
        self._decided_pause: object = None
        self.spoken_this_turn = 0
        self.spoken = 0

    # -- which ---------------------------------------------------------------

    def phrases(self, language: str | None) -> tuple[str, ...]:
        """The sounds for ``language``: the configured ones, else the defaults, else "Mm-hm."."""
        key = language_key(language)
        configured = self._phrases
        if configured is None:
            return DEFAULT_BACKCHANNELS.get(key) or DEFAULT_BACKCHANNELS[ANY_LANGUAGE]
        if isinstance(configured, Mapping):
            chosen = configured.get(key)
            if chosen is None:
                chosen = configured.get(ANY_LANGUAGE)
            if chosen is None:
                chosen = DEFAULT_BACKCHANNELS.get(key) or DEFAULT_BACKCHANNELS[ANY_LANGUAGE]
            return tuple(chosen)
        return tuple(configured)

    def _pick(self, choices: Sequence[str]) -> str:
        fresh = [phrase for phrase in choices if phrase != self._last_phrase] or list(choices)
        return self._rng.choice(fresh)

    # -- when ----------------------------------------------------------------

    def pause_due(self, quiet_secs: float | None) -> bool:
        """The caller has been quiet for long enough, and not for too long."""
        return quiet_secs is not None and self.pause_secs <= quiet_secs <= self.pause_secs + PAUSE_WINDOW_SECS

    def decide(
        self,
        *,
        now: float,
        pause: object,
        quiet_secs: float | None,
        speech_secs: float,
        utterance_secs: float,
        reason: str | None,
        choices: Sequence[str],
    ) -> tuple[str | None, str]:
        """``(phrase, why)``: the sound to say now, or ``None`` and why not.

        ``pause`` names the pause being looked at (anything that stays the
        same while it lasts): one pause gets one decision, so a draw that says
        no is not asked again 50 ms later. ``speech_secs`` is how long the
        caller has talked since the last sound or since they started,
        ``utterance_secs`` since they started.
        """
        if not self.pause_due(quiet_secs):
            return None, "no_pause"
        if pause == self._decided_pause:
            return None, "decided"
        if reason is None:
            return None, "no_reason"
        if reason == "long" and (self.long_speech_secs is None or utterance_secs < self.long_speech_secs):
            return None, "not_long"
        needed = min(self.min_speech_secs, ENTITY_MIN_SPEECH_SECS) if reason == "entity" else self.min_speech_secs
        if speech_secs < needed:
            return None, "little_said"
        if self.spoken_this_turn >= self.max_per_turn:
            return None, "turn_limit"
        if self._last_at is not None and now - self._last_at < self.min_gap_secs:
            return None, "too_soon"
        if not choices:
            return None, "no_clip"
        # From here the pause has had its chance, whatever the draw says.
        self._decided_pause = pause
        if self.rate < 1.0 and self._rng.random() >= self.rate:
            return None, "draw"
        phrase = self._pick(choices)
        self._last_phrase = phrase
        self._last_at = now
        self.spoken_this_turn += 1
        self.spoken += 1
        return phrase, reason

    def agent_replied(self) -> None:
        """The agent took its turn: the count for the caller's next one starts again."""
        self.spoken_this_turn = 0


# -- clips -------------------------------------------------------------------


def _samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _bytes(samples: array) -> bytes:
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def trim_silence(pcm: bytes, sample_rate: int, *, quiet_dbfs: float = CLIP_QUIET_DBFS) -> bytes:
    """``pcm`` without the silence before and after its sound, a short edge kept. Empty when it is all silence."""
    samples = _samples(pcm)
    window = max(1, sample_rate // 100)
    limit = 32768 * 10 ** (quiet_dbfs / 20)
    loud = [
        start
        for start in range(0, len(samples), window)
        if max(abs(sample) for sample in samples[start : start + window]) > limit
    ]
    if not loud:
        return b""
    edge = int(sample_rate * CLIP_EDGE_SECS)
    first = max(0, loud[0] - edge)
    last = min(len(samples), loud[-1] + window + edge)
    return _bytes(samples[first:last])


def scale_pcm(pcm: bytes, volume: float) -> bytes:
    """``pcm`` at ``volume`` times its level, clipped to 16 bits."""
    if volume == 1.0:
        return pcm[: len(pcm) - len(pcm) % 2]
    return _bytes(array("h", (max(-32768, min(32767, round(sample * volume))) for sample in _samples(pcm))))


def prepare_clip(pcm: bytes, sample_rate: int, *, volume: float = 1.0) -> bytes | None:
    """A synthesized listening sound ready to play, or ``None`` when it is not one.

    Not one: silence, or a take longer than :data:`MAX_CLIP_SECS` (the voice
    read the phrase as a sentence, or added to it).
    """
    trimmed = trim_silence(pcm, sample_rate)
    secs = len(trimmed) / 2 / sample_rate
    if not MIN_CLIP_SECS <= secs <= MAX_CLIP_SECS:
        return None
    return scale_pcm(trimmed, volume)
