"""Opt-in patience while a caller is dictating an entity.

Transcript-only. No model, no NER service. The patterns are language-agnostic
(digits, single-letter spelling, ``@`` emails, mixed letter/digit codes), the
same classes AssemblyAI's entity-aware endpointing holds open and ElevenLabs
waits on with ``spelling_patience`` / ``turn_eagerness: patient``.

While the tail is one of those, the session holds the commit
(:attr:`CommitAction.HOLD`) for ``entity_patience_secs`` and, on Deepgram
Flux, asks the socket to suppress natural end-of-turn (``eot_threshold`` 1.0,
silence backstop = the patience window) and drops ``EagerEndOfTurn``. Deepgram's
own guidance is to leave eager off for dictation and to raise ``eot_threshold``
/ ``eot_timeout_ms`` when the caller pauses inside a number. A finished tail
(a word after the entity, a complete email, ``?`` / ``!``) commits immediately.
The patience window expiring also commits — one budget, not a second wait on
top of a Flux timeout that already elapsed.

Pipecat's shape, without its LLM marker: a short timeout when the turn looks
done, a long one while it does not (`FilterIncompleteUserTurnStrategies`
``incomplete_short_timeout`` / ``incomplete_long_timeout``, Smart Turn's
``stop_secs`` fallback). An existing HOLD (unfinished-phrase, audio incomplete)
is kept; only its timeout is raised when the tail is also an open entity.

Sources (semantics, not copied code):

* https://www.assemblyai.com/docs/streaming/turn-detection
* https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow
* https://developers.deepgram.com/docs/flux/configure
* https://developers.deepgram.com/docs/flux/voice-agent-eager-eot
* https://deepgram.com/blog/turn-detection-voice-agents-silence-vad-semantic-models
* https://docs.pipecat.ai/api-reference/server/utilities/turn-management/filter-incomplete-turns
"""

from __future__ import annotations

import re
import unicodedata

from .turn_detection import (
    _HESITATION_TOKENS,
    CommitAction,
    CommitDecision,
    TurnState,
    _join_held,
)

# AssemblyAI's max_accuracy ``max_turn_silence`` is 2560 ms: how long an entity
# that may still continue is held before the turn is forced closed. ElevenLabs
# ``spelling_patience: auto`` is the same idea with no published duration.
DEFAULT_ENTITY_PATIENCE_SECS = 2.5

# Flux: 1.0 suppresses natural EndOfTurn; only ``eot_timeout_ms`` closes the
# turn. That is the dictation setting (Deepgram Configure docs: account
# numbers, OTPs). Eager stays a local ignore — Configure cannot unset it.
FLUX_PATIENCE_EOT_THRESHOLD = 1.0

_EMAIL_DONE = re.compile(r"[^\s@]+@[^\s@]+\.[A-Za-z0-9]{2,}\Z")
_DIGIT_RUN = re.compile(r"\d+(?:[.\-/,]\d+)*\Z")
_CODE_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*\Z")
_TOKEN_RE = re.compile(r"\S+")
_HESITATION_TAIL = re.compile(r"([^\s,.;:!?]+)(?:[,.;:!?]+)?\s*$")

# Spoken digits. AssemblyAI holds formatted entities; callers say the digits
# as words, and Flux then commits words rather than numerals. English and
# Spanish are both in the set because those are the languages the voice
# stack already endpoints. One word is not enough (see ``entity_capture``):
# a sentence that merely ends in "one" or "una" stays a normal turn.
_NUMBER_WORDS = frozenset(
    {
        "zero",
        "oh",
        "nought",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "cero",
        "uno",
        "una",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
    }
)

# Letter names that are not also everyday function words ("a", "i", "de",
# "en", "te"). A single one of these opens a spelling hold. Bare letters
# still need two, so a sentence ending in "o" or "a" does not.
_LETTER_NAMES = frozenset(
    {
        "aitch",
        "jay",
        "kay",
        "zee",
        "zed",
        "wye",
        "double",
        "jota",
        "hache",
        "ene",
        "ese",
        "efe",
        "ele",
        "eme",
        "eñe",
        "erre",
        "equis",
        "zeta",
        "uve",
        "ge",
        "ka",
        "cu",
    }
)


def entity_capture(text: str) -> str | None:
    """``"digits"`` / ``"spelling"`` / ``"email"`` / ``"code"`` when ``text`` ends in an open entity.

    Digits include numerals and spoken number words (English and Spanish).
    Spelling includes a letter name ("jota", "aitch") or two bare letters.

    ``None`` when it does not: ordinary prose, a complete email, a trailing
    ``?`` or ``!``, or a number/code the caller has already left (a word
    follows it). A trailing ``.`` does not finish digits, spelling, or codes —
    callers and STT both use periods between groups (``J. O. H.``, ``415.555``).
    """
    raw = _strip_trailing_hesitation((text or "").strip())
    if not raw or raw.endswith(("?", "!")):
        return None
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return None

    email = _email_state(tokens[-1])
    if email == "done":
        return None
    if email == "open":
        return "email"

    digits = 0
    letters = 0
    letter_names = 0
    saw_code = False
    consumed = 0
    for tok in reversed(tokens):
        atom = _atom(tok)
        if atom is None:
            break
        kind, d, n_letters = atom
        digits += d
        letters += n_letters
        if kind == "letter_name":
            letter_names += 1
        saw_code = saw_code or kind == "code"
        consumed += 1
    if consumed == 0:
        return None
    if saw_code or (digits >= 1 and letters >= 1):
        return "code"
    if digits >= 2:
        return "digits"
    # A named letter ("jota", "aitch") is already a spelling. Two bare letters
    # ("J O") are too. One bare letter alone is a word ("a", "I", "o"). A
    # bare letter after other words is how providers write a spelled name
    # ("apellido J" for "jota").
    if letter_names >= 1 or letters >= 2 or (letters == 1 and len(tokens) > consumed):
        return "spelling"
    return None


def apply_entity_patience(
    decision: CommitDecision,
    text: str,
    state: TurnState,
    *,
    enabled: bool,
    hold_secs: float,
    provider_patience_live: bool = False,
) -> CommitDecision:
    """Raise a hold while ``decision`` still ends inside an entity.

    ``enabled=False`` returns ``decision`` unchanged. ``provider_patience_live``
    means Flux already applied the long silence budget to this audio, so an
    EndOfTurn of an open entity *is* the timeout and must start the turn
    instead of waiting again. A hold the detector asked for on its own
    (unfinished phrase, audio incomplete) is never released here.
    """
    if not enabled or decision.action is CommitAction.IGNORE:
        return decision

    original = (decision.text or text).strip()
    piece = original
    held = (state.active_user_text or "").strip() if state.holding else ""
    if held and not _contains_held(held, original):
        piece = _join_held(held, (text or original).strip())

    kind = entity_capture(piece)
    if kind is None:
        return _release(decision, original, piece, held)

    if provider_patience_live and not state.holding and decision.action is CommitAction.NEW_TURN:
        return CommitDecision(
            action=CommitAction.NEW_TURN,
            text=piece,
            reason="entity_patience_provider_timeout",
        )
    # A continuation of a turn that is already being answered is the caller's
    # barge-in, not a dictation fragment parked before any reply. Leave it.
    if decision.action is CommitAction.CONTINUE_TURN and state.assistant_active and not state.holding:
        if piece == original:
            return decision
        return CommitDecision(action=CommitAction.CONTINUE_TURN, text=piece, reason=decision.reason)

    timeout = hold_secs
    if decision.action is CommitAction.HOLD and decision.hold_timeout_secs is not None:
        timeout = max(decision.hold_timeout_secs, hold_secs)
    reason = decision.reason if decision.action is CommitAction.HOLD else "entity_patience"
    return CommitDecision(
        action=CommitAction.HOLD,
        text=piece,
        reason=reason,
        hold_timeout_secs=timeout,
    )


def _release(decision: CommitDecision, original: str, piece: str, held: str) -> CommitDecision:
    if piece == original:
        return decision
    if decision.action is CommitAction.HOLD:
        return CommitDecision(
            action=CommitAction.HOLD,
            text=piece,
            reason=decision.reason,
            hold_timeout_secs=decision.hold_timeout_secs,
        )
    if decision.action is CommitAction.CONTINUE_TURN:
        return CommitDecision(action=CommitAction.CONTINUE_TURN, text=piece, reason=decision.reason)
    reason = "entity_patience_done" if held and entity_capture(held) else decision.reason
    return CommitDecision(action=CommitAction.NEW_TURN, text=piece, reason=reason)


def _contains_held(held: str, piece: str) -> bool:
    h = " ".join(held.split())
    p = " ".join(piece.split())
    return bool(h) and (p == h or p.startswith(h + " "))


def _strip_trailing_hesitation(text: str) -> str:
    """Drop trailing filled pauses so ``415 um`` is still an open number.

    Uses the same token set as turn detection. A hesitation-only utterance
    strips to empty and is not an entity (the detector already ignores it).
    """
    s = text.rstrip()
    while s:
        m = _HESITATION_TAIL.search(s)
        if m is None:
            break
        word = m.group(1).lower().strip("'")
        if word not in _HESITATION_TOKENS:
            break
        s = s[: m.start()].rstrip(" ,;:")
    return s


def _email_state(token: str) -> str | None:
    core = token.strip().strip(".,;:<>\"'()[]")
    if "@" not in core:
        return None
    if _EMAIL_DONE.fullmatch(core):
        return "done"
    return "open"


def _fold_token(token: str) -> str:
    core = token.strip().strip(".,;:|#*_/\"'")
    decomposed = unicodedata.normalize("NFKD", core)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _atom(token: str) -> tuple[str, int, int] | None:
    """``(kind, n_digits, n_letters)`` or ``None`` when the token ends the span."""
    core = token.strip().strip(".,;:|#*_/")
    if not core or "@" in core:
        return None
    if _DIGIT_RUN.fullmatch(core):
        return ("digit", sum(c.isdigit() for c in core), 0)
    folded = _fold_token(core)
    if folded in _NUMBER_WORDS:
        return ("digit", 1, 0)
    if folded in _LETTER_NAMES:
        return ("letter_name", 0, 1)
    if len(core) == 1 and core.isalpha():
        return ("letter", 0, 1)
    if _CODE_RUN.fullmatch(core) and any(c.isalpha() for c in core) and any(c.isdigit() for c in core):
        return ("code", sum(c.isdigit() for c in core), sum(c.isalpha() for c in core))
    return None
