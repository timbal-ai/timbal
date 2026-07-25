"""Scenario DSL for the voice replay harness.

A scenario is a *script* (what the user does, including silences and reactive
barge-ins) plus *expectations* (behavioral labels checked after the run).

Two rules learned the hard way, both encoded here:

* **Silence is the real input.** Turn detection keys on gap duration, so every
  pause is an explicit ``Silence`` with a duration we control rather than
  something we hope the TTS produces.
* **Never assert verbatim transcripts.** STT wobbles between runs. Text
  expectations compare normalized similarity against a threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from harness import RunResult


# ---------------------------------------------------------------------------
# Script primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Say:
    """Inject one synthesized clip."""

    text: str


@dataclass(frozen=True)
class Silence:
    """Inject exactly N seconds of silence."""

    secs: float


@dataclass(frozen=True)
class AwaitAssistantAudio:
    """Block until N ms of the assistant's reply has *played*, then continue.

    The seam that makes barge-in testable: the interruption lands at a known
    offset into the reply instead of wherever a fixed sleep happens to fall.
    """

    offset_ms: float
    timeout: float = 20.0


@dataclass(frozen=True)
class AwaitCommit:
    """Block until the session commits another user transcript."""

    timeout: float = 15.0


@dataclass(frozen=True)
class AwaitAssistantDone:
    """Block until the agent finishes generating a reply."""

    timeout: float = 20.0


Step = Say | Silence | AwaitAssistantAudio | AwaitCommit | AwaitAssistantDone


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


# ---------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------


class Expectation(Protocol):
    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        """Return ``None`` when satisfied, else a human-readable failure."""


@dataclass(frozen=True)
class UserTurns:
    """Exact number of committed user turns.

    The single most valuable assertion in the suite: a fragment wrongly split
    into two turns, or two utterances wrongly merged into one, both show up here.
    """

    count: int

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        if len(result.committed) != self.count:
            return f"expected {self.count} user turn(s), got {len(result.committed)}: {result.committed}"
        return None


@dataclass(frozen=True)
class Merged:
    """All spoken fragments arrived as one turn resembling ``text``."""

    text: str
    min_similarity: float = 0.85

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        if len(result.committed) != 1:
            return f"expected a single merged turn, got {len(result.committed)}: {result.committed}"
        score = similarity(self.text, result.committed[0])
        if score < self.min_similarity:
            return f"merged text similarity {score:.2f} < {self.min_similarity}: {result.committed[0]!r}"
        return None


@dataclass(frozen=True)
class NoGhostTurns:
    """Every committed turn resembles something the script actually said.

    Catches transcripts invented from echo, noise or a mis-fired watchdog.
    """

    min_similarity: float = 0.6

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        spoken = [s.text for s in scenario.script if isinstance(s, Say)]
        ghosts = [
            text
            for text in result.committed
            if not any(similarity(text, said) >= self.min_similarity for said in spoken)
        ]
        if ghosts:
            return f"turns not present in the script: {ghosts}"
        return None


@dataclass(frozen=True)
class Interrupted:
    """Whether a barge-in cancelled the assistant.

    Teardown interrupts from ``session.close()`` are excluded by the harness, so
    this only ever reflects a real barge-in.
    """

    expected: bool = True

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        if result.interrupted != self.expected:
            return f"expected interrupted={self.expected}, got {result.interrupted}"
        return None


@dataclass(frozen=True)
class HeardPrefix:
    """``heard_text`` is a genuine prefix of the reply, optionally short enough.

    Guards the truncation path: memory and transcript must match what the user
    actually heard, not the whole reply the agent generated.
    """

    max_chars: int | None = None

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        heard = result.heard_text
        if not heard:
            return f"expected heard text, got {heard!r}"
        if not any(normalize(reply).startswith(normalize(heard)) for reply in scenario.replies):
            return f"heard text is not a prefix of any reply: {heard!r}"
        if self.max_chars is not None and len(heard) > self.max_chars:
            return f"heard {len(heard)} chars > {self.max_chars}: {heard!r}"
        return None


@dataclass(frozen=True)
class NoAgentReply:
    """The agent never spoke — the utterance should not have started a turn."""

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        if result.replies_spoken:
            return f"expected no reply, agent said: {result.replies_spoken}"
        return None


@dataclass(frozen=True)
class MaxLatency:
    """Every turn's ``eou_to_first_audio_ms`` stayed under a ceiling."""

    ms: float

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        over = [v for v in result.latencies_ms if v > self.ms]
        if over:
            return f"eou→audio over {self.ms}ms: {[round(v) for v in over]}"
        return None


@dataclass(frozen=True)
class NoErrors:
    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        if result.errors:
            return f"session errors: {result.errors}"
        return None


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    id: str
    domain: str
    replies: list[str]
    """Fixed agent replies (``TestModel``). Reply *length* decides what a
    barge-in can land inside, so these are part of the test, not scaffolding."""
    script: list[Step]
    expect: list[Expectation]
    expect_by_detector: dict[str, list[Expectation]] = field(default_factory=dict)
    """Per-detector overrides. Necessary because the same observable behavior can
    be correct for different reasons: Deepgram Flux merges a mid-thought pause
    inside its own turn machine, while ``lexical`` merges it via Timbal's HOLD."""
    detectors: frozenset[str] | None = None
    """Detectors this scenario is meaningful for (``None`` = all)."""
    quick: bool = False
    """Part of the representative ``--quick`` subset (one per domain, plus a barge-in)."""
    known_failure: dict[str, str] = field(default_factory=dict)
    """Detector (or ``"*"``) -> why this cannot pass today.

    The scenario still runs and still reports, but it does not gate, and its
    *desired* expectations stay in the file rather than being rewritten to match
    broken behavior. If it starts passing, that's reported as an unexpected pass.
    """
    intermittent: bool = False
    """The known failure is nondeterministic, so passing sometimes means nothing.

    Suppresses the unexpected-pass report, which would otherwise fire on roughly
    half of all runs and train everyone to ignore it.
    """
    note: str = ""

    def expectations_for(self, detector: str) -> list[Expectation]:
        return self.expect_by_detector.get(detector, self.expect)

    def known_failure_reason(self, detector: str) -> str | None:
        return self.known_failure.get(detector) or self.known_failure.get("*")

    def applies_to(self, detector: str) -> bool:
        return self.detectors is None or detector in self.detectors

    def texts(self) -> list[str]:
        return [s.text for s in self.script if isinstance(s, Say)]


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

# Barge-in needs a reply long enough to interrupt *inside*, so reply length is
# part of the test rather than scaffolding.
_RETURN_POLICY_REPLY = (
    "Our return policy allows returns within thirty days of purchase, provided "
    "the item is unused and you still have the original receipt or a valid proof "
    "of purchase from one of our authorized retailers."
)
_MENU_REPLY = (
    "Today we have a roasted vegetable soup, a grilled chicken sandwich with "
    "house slaw, a mushroom risotto, and for dessert a lemon tart or a chocolate "
    "brownie served warm with vanilla ice cream."
)
_FEVER_REPLY = (
    "For a mild fever, rest and steady fluids are usually enough, and you can "
    "take paracetamol to bring your temperature down, but if it stays above "
    "thirty nine degrees for more than three days you should see a doctor."
)
_IMPORTS_REPLY = (
    "When Python imports a module it first checks the module cache in sys dot "
    "modules, then searches each entry on the path in order, compiles the source "
    "to bytecode, caches that bytecode on disk, and finally executes the module "
    "body exactly once."
)

# Bare fillers must not start a turn — except under `provider`, where trusting
# the provider's commit makes a commit the correct behavior.
_FILLER_DETECTORS = frozenset({"heuristic", "lexical", "local"})
_FILLER_NOTE = "Skipped for `provider`: trusting the provider's commit means a bare filler legitimately starts a turn."

# Measured, not assumed. The fragment Flux commits mid-pause is, in isolation, a
# grammatically complete sentence: "The pain started.", "I'll take a burger, some
# fries." Namo — the text EOU that `lexical` actually resolves to — scores both
# 1.00 complete, and scores them identically with and without the trailing period,
# so this is not an artifact of provider-added punctuation. Only prosody separates
# "The pain started [rising, continuing]" from "The pain started [falling, done]",
# and SmartTurn v3 under `--detector local` does not catch it either. The untapped
# signal is Flux's own `end_of_turn_confidence`, which Timbal currently logs and
# discards.
_TRAILING_MODIFIER_FAILURE = (
    "the committed fragment is a complete sentence in isolation, so text EOU scores it "
    "complete; only prosody marks the continuation, and the audio EOU misses it too"
)

# Under `provider` the session defers to the STT's endpointing entirely, so none of
# Timbal's hold logic runs and whatever Flux decides stands.
_PROVIDER_ENDPOINTING_FAILURE = (
    "`provider` defers to Flux's endpointing, which splits here; `lexical` merges it "
    "correctly because Namo scores the fragment 0.00 incomplete"
)

# Standard tail: wait for the turn to finish rather than sleeping a fixed guess,
# then leave a moment for any stray extra commit to show up.
_SETTLE = [AwaitCommit(), AwaitAssistantDone(), Silence(0.8)]

SCENARIOS: list[Scenario] = [
    # -- support: long replies (barge-in surface), policy jargon ---------------
    Scenario(
        id="support_simple",
        domain="support",
        replies=["Returns are accepted within thirty days of purchase."],
        script=[Say("What's your return policy?"), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(), Interrupted(False), NoErrors()],
        quick=True,
    ),
    Scenario(
        id="support_pause",
        domain="support",
        replies=["I can help with that. Do you have the order number?"],
        script=[
            Say("I want to return an item I bought"),
            Silence(1.5),
            Say("about three weeks ago."),
            *_SETTLE,
        ],
        expect=[
            Merged("I want to return an item I bought about three weeks ago."),
            NoGhostTurns(),
            Interrupted(False),
            NoErrors(),
        ],
        known_failure={"provider": _PROVIDER_ENDPOINTING_FAILURE},
        note=(
            "The one scenario where a detector choice changes correctness rather than "
            "latency: under `provider` it splits and the second fragment barges in on the "
            "reply to the first, under `lexical` it merges into a single clean turn."
        ),
    ),
    Scenario(
        id="support_barge_in",
        domain="support",
        replies=[_RETURN_POLICY_REPLY],
        script=[
            Say("Tell me about your return policy."),
            AwaitAssistantAudio(offset_ms=600),
            Say("Actually, cancel that."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), HeardPrefix(max_chars=80), NoGhostTurns(), NoErrors()],
        quick=True,
    ),
    Scenario(
        id="support_barge_in_late",
        domain="support",
        replies=[_RETURN_POLICY_REPLY],
        script=[
            Say("Tell me about your return policy."),
            AwaitAssistantAudio(offset_ms=2500),
            Say("Okay, thanks."),
            *_SETTLE,
        ],
        expect=[
            UserTurns(2),
            Interrupted(True),
            HeardPrefix(),  # later barge-in -> longer prefix, so no ceiling
            NoGhostTurns(),
            NoErrors(),
        ],
        note="Pairs with support_barge_in: same reply, later offset, measurably longer heard text.",
    ),
    # -- food_ordering: enumerations, mid-list pauses --------------------------
    Scenario(
        id="food_simple",
        domain="food_ordering",
        replies=["One large coffee and a croissant. Anything else?"],
        script=[Say("Can I get a large coffee and a croissant?"), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(), Interrupted(False), NoErrors()],
    ),
    Scenario(
        id="food_pause",
        domain="food_ordering",
        replies=["Sure — one large pepperoni. Anything else?"],
        script=[
            Say("I'd like to order a large"),
            Silence(1.5),
            Say("pepperoni pizza please."),
            *_SETTLE,
        ],
        expect=[
            Merged("I'd like to order a large pepperoni pizza please."),
            NoGhostTurns(),
            Interrupted(False),
            NoErrors(),
        ],
        quick=True,
        note=(
            "Under deepgram-flux + provider, Flux holds through the gap and merges inside "
            "its own turn machine — no Timbal HOLD involved. Under lexical it is our HOLD. "
            "Same expectation, different mechanism."
        ),
    ),
    Scenario(
        id="food_long_pause",
        domain="food_ordering",
        replies=["Got it — a large pepperoni pizza."],
        script=[
            Say("I'd like to order a large"),
            Silence(3.0),  # long enough that a split is the correct answer
            Say("pepperoni pizza please."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), NoGhostTurns(), NoErrors()],
        expect_by_detector={
            # Observed, with the mechanism deliberately not asserted: under `lexical`
            # this merges, and the event trace shows Flux never committing the first
            # fragment at all — it streams partials through the whole 3s gap and emits
            # one commit. So the merge happens inside Flux, not in Timbal's HOLD.
            # Why the same cached audio splits under `provider` is still unexplained;
            # `commit()` is a no-op for Flux, so it is not an obvious forced endpoint.
            "lexical": [
                Merged("I'd like to order a large pepperoni pizza please."),
                NoGhostTurns(),
                NoErrors(),
            ],
        },
        note=(
            "Splits under `provider`, merges under `lexical`, from byte-identical cached "
            "audio. Consistent across runs but not yet explained — see the comment above."
        ),
    ),
    Scenario(
        id="food_list_pause",
        domain="food_ordering",
        replies=["A burger, fries and a chocolate milkshake. Coming up."],
        script=[
            Say("I'll take a burger, some fries,"),
            Silence(1.4),  # mid-list breath, not an end of turn
            Say("and a chocolate milkshake."),
            *_SETTLE,
        ],
        expect=[
            Merged("I'll take a burger, some fries, and a chocolate milkshake."),
            NoGhostTurns(),
            NoErrors(),
        ],
        known_failure={"*": _TRAILING_MODIFIER_FAILURE},
        note="A pause after a comma mid-enumeration: prosody says continue even though the gap is long.",
    ),
    Scenario(
        id="food_barge_in",
        domain="food_ordering",
        replies=[_MENU_REPLY],
        script=[
            Say("Tell me what's on the menu today."),
            AwaitAssistantAudio(offset_ms=800),
            Say("Just the specials, please."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), HeardPrefix(), NoGhostTurns(), NoErrors()],
    ),
    # -- medical_intake: hesitation-heavy -------------------------------------
    Scenario(
        id="medical_simple",
        domain="medical_intake",
        replies=["How long has the headache lasted?"],
        script=[Say("I've had a headache since yesterday."), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(), Interrupted(False), NoErrors()],
    ),
    Scenario(
        id="medical_hesitation",
        domain="medical_intake",
        replies=["Take your time."],
        script=[Say("Um..."), Silence(2.5)],
        expect=[NoAgentReply(), NoErrors()],
        detectors=_FILLER_DETECTORS,
        quick=True,
        note=_FILLER_NOTE,
    ),
    Scenario(
        id="medical_hesitant_pause",
        domain="medical_intake",
        replies=["Thanks. Has it changed since then?"],
        script=[
            Say("The pain started"),
            Silence(1.5),
            Say("maybe on Tuesday."),
            *_SETTLE,
        ],
        expect=[Merged("The pain started maybe on Tuesday."), NoGhostTurns(), NoErrors()],
        known_failure={"*": _TRAILING_MODIFIER_FAILURE},
        note="Trailing off mid-recall, the archetypal case for holding instead of endpointing.",
    ),
    Scenario(
        id="medical_barge_in",
        domain="medical_intake",
        replies=[_FEVER_REPLY],
        script=[
            Say("What should I do about a fever?"),
            AwaitAssistantAudio(offset_ms=700),
            Say("Sorry, one more thing."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), HeardPrefix(), NoGhostTurns(), NoErrors()],
    ),
    # -- banking: digit strings and short confirmations ------------------------
    Scenario(
        id="banking_digits",
        domain="banking",
        replies=["Thank you. I've found the account."],
        # Spelled out, because that is what comes back: ElevenLabs speaks "447291"
        # digit by digit and Deepgram transcribes the words, so a numeral here would
        # look like a hallucinated turn to every text comparison.
        script=[Say("My account number is four four seven two nine one."), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(), Interrupted(False), NoErrors()],
        note="Digit strings are the STT-hostile case: no lexical context to fall back on.",
    ),
    Scenario(
        id="banking_digits_pause",
        domain="banking",
        replies=["Got it, account 447291."],
        script=[
            Say("My account number is four four seven"),
            Silence(1.4),
            Say("two nine one."),
            *_SETTLE,
        ],
        expect=[
            Merged("My account number is four four seven two nine one.", min_similarity=0.75),
            NoGhostTurns(min_similarity=0.4),
            NoErrors(),
        ],
        known_failure={"*": "splits into three turns, one of them a spurious single-token commit (observed: 'I')"},
        note=(
            "The hardest merge in the suite: a digit string broken across a pause, with no "
            "words to hint continuation. Also the only scenario so far to produce a genuine "
            "ghost turn, which is worth keeping precisely because it reproduces one."
        ),
    ),
    Scenario(
        id="banking_confirmation",
        domain="banking",
        replies=["Great, the transfer is scheduled."],
        script=[Say("Yes, that's correct."), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(), Interrupted(False), NoErrors()],
        quick=True,
        known_failure={
            "*": "intermittently splits into 'Yes.' + \"That's correct.\", and the second "
            "fragment barges in on the reply to the first"
        },
        intermittent=True,
        note=(
            "Observed passing and failing on consecutive runs of an unchanged tree. Short "
            'utterances have a history here: ElevenLabs once transcribed "yes." / "work." '
            "as partials that never committed at all, hanging the session."
        ),
    ),
    Scenario(
        id="banking_hesitation",
        domain="banking",
        replies=["No problem, take your time."],
        script=[Say("Uh..."), Silence(2.5)],
        expect=[NoAgentReply(), NoErrors()],
        detectors=_FILLER_DETECTORS,
        note=_FILLER_NOTE,
    ),
    # -- coding_help: technical tokens ---------------------------------------
    Scenario(
        id="coding_simple",
        domain="coding_help",
        replies=["It means you indexed something that doesn't support indexing."],
        script=[Say("What does TypeError not subscriptable mean?"), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(min_similarity=0.5), Interrupted(False), NoErrors()],
        quick=True,
        note="Technical tokens have no conversational prior, so STT leans harder on acoustics.",
    ),
    Scenario(
        id="coding_pause",
        domain="coding_help",
        replies=["Can you paste the traceback?"],
        script=[
            Say("I'm getting an error in my"),
            Silence(1.5),
            Say("async generator function."),
            *_SETTLE,
        ],
        expect=[
            Merged("I'm getting an error in my async generator function.", min_similarity=0.8),
            NoGhostTurns(min_similarity=0.5),
            NoErrors(),
        ],
    ),
    Scenario(
        id="coding_barge_in",
        domain="coding_help",
        replies=[_IMPORTS_REPLY],
        script=[
            Say("Explain how Python handles imports."),
            AwaitAssistantAudio(offset_ms=900),
            Say("Actually, never mind."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), HeardPrefix(), NoGhostTurns(), NoErrors()],
    ),
]


def select(
    ids: list[str] | None,
    detector: str,
    *,
    quick: bool = False,
) -> tuple[list[Scenario], list[Scenario]]:
    """Return ``(applicable, skipped)`` for a detector, filtered by ids or ``quick``."""
    chosen = SCENARIOS
    if ids:
        chosen = [s for s in chosen if s.id in set(ids)]
    if quick:
        chosen = [s for s in chosen if s.quick]
    return (
        [s for s in chosen if s.applies_to(detector)],
        [s for s in chosen if not s.applies_to(detector)],
    )
