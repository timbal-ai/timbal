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
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from harness import RunResult


# ---------------------------------------------------------------------------
# Script primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Say:
    """Inject one synthesized clip.

    ``utterance`` marks this as slice ``part`` of a longer sentence rendered in
    one piece — see :func:`fluent`.
    """

    text: str
    utterance: str | None = None
    part: int = 0

    @property
    def clip_key(self) -> str:
        return self.text if self.utterance is None else f"{self.utterance}#{self.part}"


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


def fluent(*parts: str | float) -> list[Step]:
    """Script one sentence the speaker pauses inside, keeping mid-sentence prosody.

    Takes alternating text and gap seconds — ``fluent("The pain started", 1.5,
    "maybe on Tuesday.")`` — and renders the joined sentence in a single TTS call,
    slicing it at the character timestamps.

    Synthesizing the fragments separately is what a naive harness does, and it is
    wrong in a way that silently changes the answer: ElevenLabs renders each
    fragment as its own sentence with a falling contour. Measured with Smart Turn
    v3 on identical words, "I want to return an item I bought" scores 0.986
    (finished) standalone and 0.019 (mid-thought) cut from the fluent render. Use
    this for any pause *inside* a sentence; use bare ``Say`` for genuinely separate
    utterances.
    """
    texts = [p for p in parts if isinstance(p, str)]
    utterance = " ".join(texts)
    steps: list[Step] = []
    index = 0
    for part in parts:
        if isinstance(part, str):
            steps.append(Say(part, utterance=utterance, part=index))
            index += 1
        else:
            steps.append(Silence(part))
    return steps


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------


_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}  # fmt: skip


def _spell_digits(match: re.Match[str]) -> str:
    return " ".join(_DIGIT_WORDS[d] for d in match.group())


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and render digit runs as spoken words.

    The digit pass exists because providers disagree about a spoken digit string
    with no bearing on turn-taking: Flux writes "four four seven two nine one"
    and Nova writes "447291" for the same audio. Without it, `banking_digits`
    fails in seven of twelve cells having committed exactly one turn in all
    twelve — a pure formatting difference reported as a turn-taking defect.

    Digits are spelled out one by one rather than parsed as numbers, matching how
    an account number is read aloud. "30 days" would normalize to "three zero
    days", which is wrong, but nothing in the suite says a number that way.
    """
    spelled = re.sub(r"\d+", _spell_digits, text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", spelled)).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def best_window(haystack: str, needle: str) -> float:
    """Similarity between ``needle`` and the closest same-length word window of ``haystack``.

    Whole-string similarity cannot see a dropped fragment inside a long utterance;
    this can, because a missing part matches no window.
    """
    hay, ned = normalize(haystack).split(), normalize(needle).split()
    if not ned:
        return 1.0
    if not hay:
        return 0.0
    joined = " ".join(ned)
    windows = (" ".join(hay[i : i + len(ned)]) for i in range(max(1, len(hay) - len(ned) + 1)))
    return max(SequenceMatcher(None, joined, w).ratio() for w in windows)


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
        # Whole-string similarity alone lets a dropped fragment through: ElevenLabs
        # committed coding_double_pause without "inside a test" and still scored 0.84
        # against the full sentence, so the scenario reported a clean merge on a third
        # of the utterance being lost. A merge that loses a part is not a merge.
        if (fault := AllPartsHeard().check(result, scenario)) is not None:
            return fault
        score = similarity(self.text, result.committed[0])
        if score < self.min_similarity:
            return f"merged text similarity {score:.2f} < {self.min_similarity}: {result.committed[0]!r}"
        return None


@dataclass(frozen=True)
class ContentPreserved:
    """Everything spoken reached the agent, across however many turns it took.

    The turn-count-agnostic counterpart to :class:`Merged`. Use it when *whether*
    an utterance splits is a defensible product choice but losing half of it is
    not — at a long pause, holding and splitting are both reasonable, and a
    scenario that demands one of them is asserting a policy it cannot justify.
    This still catches the failure that actually hurts: ElevenLabs committing
    "I'd like to order a large" and silently dropping the pizza.
    """

    min_similarity: float = 0.85

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        if (fault := AllPartsHeard().check(result, scenario)) is not None:
            return fault
        spoken = " ".join(scenario.texts())
        heard = " ".join(result.committed)
        score = similarity(spoken, heard)
        if score < self.min_similarity:
            return f"content similarity {score:.2f} < {self.min_similarity}: {heard!r} vs {spoken!r}"
        return None


@dataclass(frozen=True)
class NoGhostTurns:
    """Every committed turn resembles something the script actually said.

    Catches transcripts invented from echo, noise or a mis-fired watchdog.
    """

    min_similarity: float = 0.6

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        spoken = scenario.texts()
        ghosts = [
            text
            for text in result.committed
            if not any(similarity(text, said) >= self.min_similarity for said in spoken)
        ]
        if ghosts:
            return f"turns not present in the script: {ghosts}"
        return None


@dataclass(frozen=True)
class AllPartsHeard:
    """Every scripted fragment reached the agent, wherever the turns fell.

    Similarity over a whole utterance is blind to a dropped tail, and the longer
    the utterance the blinder it gets. ElevenLabs committed `coding_double_pause`
    as "The build fails when I import the module" — "inside a test" simply gone —
    and that still scores 0.84 against the full sentence, clearing the 0.8 bar on
    `Merged`. The scenario reported a clean merge while a third of it was lost.
    Matching each fragment against its best window catches the drop no matter how
    much correct text surrounds it.
    """

    min_similarity: float = 0.75

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        heard = " ".join(result.committed)
        missing = [t for t in scenario.texts() if best_window(heard, t) < self.min_similarity]
        if missing:
            return f"fragments never heard: {missing} — got {result.committed}"
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
    """Every turn's ``eou_to_first_audio_ms`` stayed under a ceiling.

    Measures the *pipeline* after a turn is accepted. It cannot see a hold,
    because its clock starts at the accepted commit — use :class:`MaxDeadAir`
    for anything that changes turn-taking timing.
    """

    ms: float

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        over = [v for v in result.latencies_ms if v > self.ms]
        if over:
            return f"eou→audio over {self.ms}ms: {[round(v) for v in over]}"
        return None


@dataclass(frozen=True)
class MaxDeadAir:
    """Time from the user falling silent to the turn being accepted.

    The assertion that makes hold policy falsifiable. Every other metric here is
    blind to it: `eou→audio` starts at the accepted commit, so a hold that sits
    for three seconds before committing costs nothing measurable, and pass/fail
    only ever saw whether the turn eventually merged.

    That blindness had teeth. Disabling the text-complete hold tier reads as 11
    scenario-cells fixed and zero regressions, and on that evidence it looks
    like an obvious win — while adding 2.6s before the assistant answers an
    ordinary question in six barge-in cells, 2.6s to three closers, and 8.2s to
    a long run-on it "fixed". Use this on any scenario where the speaker has
    genuinely finished, so buying a merge with dead air has to be declared.
    """

    ms: float

    def check(self, result: RunResult, scenario: Scenario) -> str | None:
        over = [v for v in result.dead_air_ms if v > self.ms]
        if over:
            return f"dead air over {self.ms}ms: {[round(v) for v in over]}"
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
    """Overrides keyed by ``"detector"``, ``"stt/detector"`` or ``"*"``, most
    specific first. Necessary because the same observable behavior can be correct
    for different reasons: Deepgram Flux merges a mid-thought pause inside its own
    turn machine, while ``lexical`` merges it via Timbal's HOLD."""
    detectors: frozenset[str] | None = None
    """Detectors this scenario is meaningful for (``None`` = all)."""
    quick: bool = False
    """Part of the representative ``--quick`` subset (one per domain, plus a barge-in)."""
    known_failure: dict[str, str] = field(default_factory=dict)
    """``"detector"`` / ``"stt/detector"`` / ``"*"`` -> why this cannot pass today.

    The scenario still runs and still reports, but it does not gate, and its
    *desired* expectations stay in the file rather than being rewritten to match
    broken behavior. If it starts passing, that's reported as an unexpected pass.
    """
    known_failure_under_leak: dict[str, str] = field(default_factory=dict)
    """Same key grammar, but only when the run injects echo (``--aec-leak`` > 0).

    Echo is a separate axis, not a harder version of the clean one: a scenario can
    be sound under user-only audio and fail purely because the assistant hears
    itself. Folding the two into one table would either excuse a clean-run failure
    or make every leak run report nine regressions it already knows about.

    Always treated as intermittent, whatever :attr:`intermittent` says, because
    under-suppression depends on how the STT happens to mis-hear a given phrase
    rather than on the gain — so a pass at any single gain proves nothing.
    """
    intermittent: bool = False
    """The known failure is nondeterministic, so passing sometimes means nothing.

    Suppresses the unexpected-pass report, which would otherwise fire on roughly
    half of all runs and train everyone to ignore it.
    """
    note: str = ""

    def _scoped(self, table: dict[str, Any], detector: str, stt: str | None) -> Any | None:
        """Look up ``table`` by cell, then STT, then detector, then ``"*"``.

        Both overrides and known failures need this. A finding is usually specific
        to one *cell*, not to a detector everywhere: `food_list_pause` merges under
        `lexical` on Flux and splits under `lexical` on Nova, because on Flux the
        provider merged it before any detector saw it. Keying by detector alone
        would let a Flux observation silently excuse a Nova failure.

        The ``"deepgram-nova/*"`` rung exists because some failures are the STT's
        alone and land identically under all four detectors — Nova committing a
        "mm-hmm", ElevenLabs merging two finished sentences. Writing those out
        per cell repeats one fact four times and hides that it is one fact.
        """
        rungs = [f"{stt}/{detector}", f"{stt}/*"] if stt is not None else []
        for key in (*rungs, detector, "*"):
            if (hit := table.get(key)) is not None:
                return hit
        return None

    def expectations_for(self, detector: str, stt: str | None = None) -> list[Expectation]:
        return self._scoped(self.expect_by_detector, detector, stt) or self.expect

    def known_failure_reason(self, detector: str, stt: str | None = None, aec_leak: float = 0.0) -> str | None:
        """The leak table wins when echo is injected: it is the more specific claim,
        and a scenario marked for both is failing for two different reasons."""
        if aec_leak and (hit := self._scoped(self.known_failure_under_leak, detector, stt)) is not None:
            return hit
        return self._scoped(self.known_failure, detector, stt)

    def is_intermittent(self, detector: str, stt: str | None = None, aec_leak: float = 0.0) -> bool:
        if aec_leak and self._scoped(self.known_failure_under_leak, detector, stt) is not None:
            return True
        return self.intermittent

    def applies_to(self, detector: str) -> bool:
        return self.detectors is None or detector in self.detectors

    def texts(self) -> list[str]:
        """Everything the user says, fluent parts included — what was *spoken*."""
        return [s.text for s in self.script if isinstance(s, Say)]

    def standalone_texts(self) -> list[str]:
        """Clips synthesized on their own; fluent parts are sliced from one render."""
        return [s.text for s in self.script if isinstance(s, Say) and s.utterance is None]

    def fluent_groups(self) -> list[tuple[str, tuple[str, ...]]]:
        """``(whole utterance, parts)`` for each sentence spoken across pauses."""
        groups: dict[str, list[str]] = {}
        for step in self.script:
            if isinstance(step, Say) and step.utterance is not None:
                groups.setdefault(step.utterance, []).append(step.text)
        return [(utterance, tuple(parts)) for utterance, parts in groups.items()]


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
# grammatically complete sentence — "The pain started." before "maybe on Tuesday" —
# and Namo, the text EOU that `lexical` resolves to, scores it 1.00 complete with or
# without the trailing period. Only prosody marks the continuation, and while the
# audio EOU does hear it (Smart Turn scores this fragment 0.054 on fluent audio), a
# text-confidence tier shrinks the resulting hold to 0.35s against a 1.5s pause, so
# `local` splits it too — intermittently; it merges roughly one run in five. Flux's
# own `end_of_turn_confidence` is no help either: sampled across the suite,
# fragments (0.690-0.920) and true ends (0.701-0.904) overlap almost entirely.
#
# Every other scenario in this class stopped failing once fluent synthesis landed
# (see `fluent`), which is the measure of how much the old per-`Say` fixture was
# testing TTS phrasing rather than turn-taking.
_TRAILING_MODIFIER_FAILURE = (
    "the committed fragment is a complete sentence in isolation, so text EOU scores it "
    "complete; the audio EOU does hear the continuation but the text-complete tier cuts "
    "its hold to 0.35s"
)

# The announced half of the family above: "...447. Sorry." and "...blue pill. No, wait."
# say outright that a correction is coming, and `local` now holds on that marker alone
# (trailing_correction_marker). What remains here is timing, not judgement — the hold is
# armed, and the correction either arrives inside it or does not. Retired the markers on
# flux/local (banking, 6/6) and nova/local (medical, 6/6) outright.
_CORRECTION_MARKER_HELD = (
    "the correction marker now arms a hold, so this is a race rather than a misjudgement: "
    "the retraction has to arrive inside the short tier"
)

# Off Flux, Namo alone is not enough to hold a mid-sentence pause. Nova commits a
# fragment that reads as a finished sentence and `lexical` has no second opinion to
# contradict it — the audio EOU is exactly what `local` adds, and the gap between
# them on this family is 20% vs 60% (Nova) and 70% vs 80% (ElevenLabs). Marked per
# cell rather than per detector because the same detector merges these on Flux,
# where the provider held the fragment before any detector saw it.
_TEXT_ONLY_PAUSE_SHORTFALL = (
    "text EOU alone cannot hold this pause off Flux: the committed fragment reads finished and nothing contradicts it"
)

# Even with Smart Turn, some pauses are past what either signal resolves. Kept
# separate from the text-only reason so a fix to one is not credited to the other.
_AUDIO_PAUSE_SHORTFALL = "audio EOU hears the continuation but the hold does not survive this gap"

# `heuristic` and `provider` have no HOLD at all, so a pause the STT commits at is a
# turn boundary and nothing downstream can merge it. Not a defect but the control
# group: it measures what the hold path is worth, which is 65-69% against 98-100% on
# the same STT.
#
# Marked per cell rather than per detector, because whether it bites is an STT ×
# detector interaction and not a property of either. `deepgram-flux/heuristic` scores
# 90% on the identical detector, since Flux holds the fragment provider-side before
# any detector sees it — so a blanket "holdless detector cannot merge" rule would
# falsely mark nine scenarios it passes and hide regressions there. Nova and
# ElevenLabs commit each fragment, and their two holdless cells fail the same family.
_NO_HOLD_TO_MERGE = (
    "`heuristic` and `provider` have no hold, so each fragment the STT commits becomes its own turn"
)

# ElevenLabs' endpointer waits longer than Flux's, which helps mid-sentence pauses
# and hurts here: two finished sentences 0.9s apart come back as one turn, in all
# four cells, with no detector given a chance to disagree.
_EL_OVERMERGE = "ElevenLabs merges the two sentences into one turn before any detector sees them"

# Under `provider` the session defers to the STT's endpointing entirely, so none of
# Timbal's hold logic runs and whatever Flux decides stands.
# Under `provider` there is no Timbal hold to override the STT, so any pause Flux
# chooses to end its turn at becomes a split. Flux usually holds through these two
# and occasionally does not; a single run reads as a clean pass, and only --repeat
# shows the 2-in-3. Both fragments here end at the "--" Flux writes for a hesitation.
_FLUX_PROVIDER_SPLIT = (
    "Flux intermittently ends its turn at the pause rather than holding through it, and under "
    "`provider` nothing in Timbal runs to override that — merged 2/3 at --repeat 3"
)

# A spurious single-token "I" turn committed just after a correctly merged utterance.
# Long attributed to `banking_digits_pause` and assumed to be something about digit
# strings; `medical_filler_midway` reproduced it verbatim on ordinary words, so it is
# the Flux + `provider` path, not the content. Roughly one run in four to six, which is
# why it took a wider suite to catch twice.
_FLUX_PROVIDER_GHOST_I = (
    "Flux + `provider` intermittently commits a spurious single-token 'I' turn after the real "
    "one; not specific to this scenario"
)

# What survives echo suppression once the suppressor actually works.
#
# Thirteen scenarios carried this marker when the leak axis first ran, all failing the
# same way: the assistant heard its own bleed, interrupted itself, and committed a
# fragment of its own reply. Eleven were retired by fixing `_likely_stt_echo` to score
# against a same-length window, and the twelfth by refusing to let the stale-partial
# watchdog resurrect a commit that had already been suppressed as echo.
#
# This one is different in kind and will not yield to a better threshold. The STT hands
# over a *single* commit containing both the echo tail and the user's real word —
# "retailers. Stop." out of "...authorized retailers." plus "Stop." — so suppression and
# admission are the same decision applied to two utterances. Dropping it loses the
# interruption the scenario exists to test; keeping it scores a ghost turn. Splitting
# the commit is the only real fix, and that needs the echo span located within it
# rather than a verdict on the whole string.
_ECHO_MIXED_COMMIT = (
    "the STT merges the echo tail and the user's word into one commit "
    "('retailers. Stop.'), so it cannot be suppressed without losing the barge-in"
)
_ECHO_LEAK_CELL = "deepgram-flux/local"

# Digit readback is the worst case for echo, from both directions at once.
#
# Garbled echo is only suppressed once a session has proved it leaks, by producing one
# verbatim echo first (`TurnDetector.echo_verdict`). Digits never get there: the leak
# is quiet, so the STT garbles every copy of it, and a run of digits garbles into other
# digits that still resemble the reply. Nothing verbatim ever arrives to arm the filter.
#
# And the obvious repair — trust resemblance here, since a readback obviously echoes —
# is exactly backwards. The reply *is* the user's digits, so a genuine correction
# ("Four four eight." against "…account four four seven.") scores 0.75, inside the range
# real echo occupies. Digit readback is where resemblance is least informative and most
# tempting, which is why these two are marked rather than tuned around.
#
# Both are gated on measurement, not marked: everything else that fails under leak is
# intermittent and the baseline now records it at its true rate. Only these two fail
# every repeat.
_ECHO_GARBLED_DIGITS = (
    "garbled echo of a digit readback: the session never produces verbatim echo to arm "
    "the resemblance check, and resemblance cannot be trusted when the reply is the digits"
)

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
        note=(
            "One question, one reply, no pauses — and at a 0.15 echo leak the assistant "
            "interrupts itself on 3 of 4 repeats. The simplest scenario in the suite is enough "
            "to show that a mediocre echo canceller does not degrade turn-taking at the "
            "margins, it breaks the ordinary case."
        ),
    ),
    Scenario(
        id="support_pause",
        domain="support",
        replies=["I can help with that. Do you have the order number?"],
        script=[
            *fluent("I want to return an item I bought", 1.5, "about three weeks ago."),
            *_SETTLE,
        ],
        expect=[
            Merged("I want to return an item I bought about three weeks ago."),
            NoGhostTurns(),
            Interrupted(False),
            NoErrors(),
        ],
        known_failure={
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        note=(
            "Used to split under `provider` and `local` and was marked a known failure for "
            "both. Fluent synthesis fixed it outright: rendered standalone this fragment "
            "scores 0.986 finished, and cut from the whole sentence it scores 0.019.\n\n"
            "The last marker, `deepgram-nova/lexical`, went the same way once holds could be "
            "extended on mic energy instead of only on new partials: a text-only detector has "
            "no prosody to hold on, so it was expiring mid-pause. Passes 3/3 there now, and "
            "the marker was never flagged intermittent, so it had been failing reliably."
        ),
    ),
    Scenario(
        id="support_closer",
        domain="support",
        replies=["Happy to help. Have a good day."],
        script=[Say("That's all."), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(), Interrupted(False), MaxDeadAir(3500), NoErrors()],
        note=(
            "The case the text-complete hold tier exists for, and the only one in the suite: "
            "Smart Turn under-scores this closer (0.115 — 13 of 14 closers measured score "
            "above 0.5 and never reach the tier) while the text reads finished, so the hold "
            "is armed on an utterance that is actually over. Disabling the tier costs 2.7s of "
            "dead air here (speech end to reply: 0.91s with it, 3.59s without), which is why "
            "the tier survives even though it is what splits medical_hesitant_pause. "
            "Carries MaxDeadAir rather than MaxLatency, because eou→audio is measured from "
            "the *accepted* commit and reads ~200ms whether the hold ran or not. Dead air "
            "sits at 1.9s median / 2.8s worst with the tier; removing it puts this at ~4.5s, "
            "which is what the 3500ms ceiling is placed to catch. It is a tripwire on a "
            "6-sample distribution, not a tight bound — the per-cell dead-air gate in "
            "score.py is the real instrument."
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
        expect=[UserTurns(2), Interrupted(True), HeardPrefix(max_chars=100), NoGhostTurns(), NoErrors()],
        quick=True,
        note=(
            "The ceiling was 80 and sat inside the harness's own resolution: playback acks "
            "arrive every 250ms, several characters of speech per tick, so runs landed at 82 "
            "and 85 with the interrupt working perfectly and a genuine prefix heard. It read "
            "as a barge-in regression on a run that had none. 100 leaves room for ack "
            "granularity while still separating this from support_barge_in_late (~77+ chars "
            "at a 2500ms offset is the contrast being drawn, and that one carries no ceiling)."
        ),
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
    Scenario(
        id="support_barge_in_instant",
        domain="support",
        replies=[_RETURN_POLICY_REPLY],
        script=[
            Say("Tell me about your return policy."),
            AwaitAssistantAudio(offset_ms=150),
            Say("Actually, cancel that."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), NoGhostTurns(), NoErrors()],
        note=(
            "Barge-in before the assistant is really speaking, racing playback start against "
            "the interrupt. Deliberately asserts no HeardPrefix: at 150ms there may honestly "
            "be nothing heard yet, and failing on that would be testing the expectation."
        ),
    ),
    Scenario(
        id="support_barge_in_one_word",
        domain="support",
        replies=[_RETURN_POLICY_REPLY],
        script=[
            Say("Tell me about your return policy."),
            AwaitAssistantAudio(offset_ms=900),
            Say("Stop."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), NoGhostTurns(), NoErrors()],
        known_failure_under_leak={_ECHO_LEAK_CELL: _ECHO_MIXED_COMMIT},
        note=(
            "The most important barge-in in voice UX and the one the code is built to ignore: "
            "both HeuristicTurnDetector and ProviderTurnDetector drop partials shorter than "
            "MIN_BARGE_IN_PARTIAL_WORDS (2) as mic blips, and 'Stop.' is one word. Asserts the "
            "desired behavior rather than the implemented one — if this fails everywhere, the "
            "gate is the finding."
        ),
    ),
    Scenario(
        id="support_pause_short",
        domain="support",
        replies=["Of course. What's the order number?"],
        script=[
            *fluent("I need help with", 0.6, "my recent order."),
            *_SETTLE,
        ],
        expect=[
            Merged("I need help with my recent order."),
            NoGhostTurns(),
            Interrupted(False),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/provider": _FLUX_PROVIDER_SPLIT,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "The floor of the pause family: 0.6s is shorter than any endpointer's silence "
            "window, so every configuration should merge it. Exists to prove the harder pause "
            "scenarios are measuring difficulty rather than a broken fixture — and it earns "
            "that, merging in eleven of twelve cells. The exception is instructive: `provider` "
            "on Flux splits even 0.6s about one run in three, so the floor is not the "
            "detector's, it is whatever the STT decided."
        ),
    ),
    Scenario(
        id="support_trailing_conjunction",
        domain="support",
        replies=["I'm sorry to hear that. Let me check the tracking."],
        script=[
            *fluent("I ordered a lamp last week and", 1.5, "it still hasn't arrived."),
            *_SETTLE,
        ],
        expect=[
            Merged("I ordered a lamp last week and it still hasn't arrived."),
            NoGhostTurns(),
            NoErrors(),
        ],
        known_failure={
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/lexical": _TEXT_ONLY_PAUSE_SHORTFALL + " — merged 3/4",
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "A dangling coordinating conjunction is the strongest continuation cue in text, so "
            "this is the pause case text EOU should get right without any help from prosody — "
            "the complement to medical_hesitant_pause, where the fragment reads complete and "
            "only prosody says otherwise."
        ),
    ),
    Scenario(
        id="support_echo_silence",
        domain="support",
        replies=[_RETURN_POLICY_REPLY],
        script=[
            Say("Tell me about your return policy."),
            # The user says nothing for the whole reply. Anything committed here
            # came out of the assistant's own mouth.
            AwaitAssistantDone(),
            Silence(4.0),
        ],
        expect=[UserTurns(1), Interrupted(False), NoGhostTurns(), NoErrors()],
        note=(
            "Run with --aec-leak to bleed the assistant's output back into the mic. "
            "`_likely_stt_echo` exists so speaker bleed does not make the assistant "
            "interrupt itself, and until this scenario nothing had ever fed it echo: every "
            "run was clean user-only audio. coding_barge_in_echo proves the suppressor does "
            "not fire on genuine speech; this is the other half.\n\n"
            "Measured on deepgram-flux/local: clean at 0.0, intermittent from 0.20, and 4-5 "
            "failures in 6 at 0.30 — where the assistant cuts itself off mid-reply and commits "
            "its own words as a user turn. Real barge-in still works at 0.30, so the suppressor "
            "is letting echo through rather than over-suppressing.\n\n"
            "0.15 was recorded as clean here and is not: it fails ~1 in 3. The original reading "
            "came from too few repeats on the one scenario the leak axis was ever run against. "
            "The axis now carries its own baseline, so the whole suite's leak behaviour is "
            "recorded at measured rates instead of being marked scenario by scenario.\n\n"
            "The mechanism is visible in what gets committed: 'Our retailers.', 'Arise "
            "retailers.', 'has aroused retailers.' — all manglings of the reply's own tail, "
            "'...authorized retailers.' The suppressor is a text-similarity check against "
            "what the assistant said, so echo that the STT transcribes *badly* stops "
            "resembling its source and passes the very filter built to catch it. Louder, "
            "cleaner echo would be caught; garbled echo is not."
        ),
    ),
    Scenario(
        id="support_silence_only",
        domain="support",
        replies=["Hello?"],
        script=[Silence(4.0)],
        expect=[UserTurns(0), NoAgentReply(), NoErrors()],
        note=(
            "Four seconds of nothing. No scenario previously asserted that an open mic with no "
            "speech produces no turn, which is the purest form of the ghost-turn bug and the "
            "one a VAD or watchdog misfire would cause."
        ),
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
            *fluent("I'd like to order a large", 1.5, "pepperoni pizza please."),
            *_SETTLE,
        ],
        expect=[
            Merged("I'd like to order a large pepperoni pizza please."),
            NoGhostTurns(),
            Interrupted(False),
            NoErrors(),
        ],
        known_failure={
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/lexical": _TEXT_ONLY_PAUSE_SHORTFALL + " — merged 2/3",
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
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
            # long enough that a split is the correct answer
            *fluent("I'd like to order a large", 3.0, "pepperoni pizza please."),
            *_SETTLE,
        ],
        expect=[ContentPreserved(), NoGhostTurns(), NoErrors()],
        note=(
            "The one scenario whose desired outcome is genuinely ambiguous, so it asserts "
            "content rather than turn count. At a 3.0s gap both answers are defensible — "
            "holding costs 3s of dead air, splitting sends a fragment to the LLM — and the "
            "suite has no basis for calling either wrong. Observed: Flux merges under "
            "`lexical`/`local` and splits under `heuristic`/`provider`, Nova splits under all "
            "four, ElevenLabs splits under `lexical`. Flux's merge is not Timbal's: "
            "the trace shows Flux streaming partials through the whole gap and emitting one "
            "commit, so no fragment ever reaches a detector. It previously demanded a merge "
            "wherever one had been seen, which quietly turned observations into requirements "
            "and failed Nova for behaving reasonably.\n\n"
            "It also carried a known failure blaming ElevenLabs for dropping the second "
            "fragment instead of splitting. That was the harness: `AwaitCommit` resolved on "
            "the *first* fragment's commit, which on ElevenLabs lands ~1.6s late and so "
            "arrived after the wait was armed, leaving only the 0.8s tail for a commit that "
            "needed ~1.0s. The turn was lost to teardown, not by the provider — and one "
            "observed turn instead of two is also why this note used to say ElevenLabs merged "
            "here. It splits, 4/4, content intact."
        ),
    ),
    Scenario(
        id="food_list_pause",
        domain="food_ordering",
        replies=["A burger, fries and a chocolate milkshake. Coming up."],
        script=[
            # mid-list breath, not an end of turn
            *fluent("I'll take a burger, some fries,", 1.4, "and a chocolate milkshake."),
            *_SETTLE,
        ],
        expect=[
            Merged("I'll take a burger, some fries, and a chocolate milkshake."),
            NoGhostTurns(),
            NoErrors(),
        ],
        known_failure={
            "deepgram-nova/local": _AUDIO_PAUSE_SHORTFALL,
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/local": _AUDIO_PAUSE_SHORTFALL + " — merged 2/3",
            "elevenlabs/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "A pause after a comma mid-enumeration: prosody says continue even though the "
            "gap is long. Failed on every detector until fluent synthesis; now passes on all "
            "three."
        ),
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
    Scenario(
        id="food_backchannel",
        domain="food_ordering",
        replies=[_MENU_REPLY],
        script=[
            Say("Tell me what's on the menu today."),
            AwaitAssistantAudio(offset_ms=1200),
            Say("Mm-hmm."),
            # No AwaitAssistantDone: the reply to turn 1 has already been generated and a
            # correct run produces no second one, so waiting for one always times out.
            Silence(3.0),
        ],
        expect=[UserTurns(1), Interrupted(False), NoGhostTurns(), NoErrors()],
        known_failure={
            "deepgram-nova/*": "Nova transcribes the backchannel as 'Mhmm.' and commits it, "
            "which cuts the assistant off; no detector sees anything to distinguish it from a "
            "one-word barge-in",
            "deepgram-flux/*": "same commit, intermittently — Flux usually swallows the sound "
            "but lets it through often enough to fail roughly one run in three, under both "
            "`local` and `lexical`",
        },
        intermittent=True,
        note=(
            "Listeners say 'mm-hmm' while you talk and do not mean stop. Nothing in the suite "
            "distinguished a backchannel from a barge-in before this, and the two are "
            "acoustically similar and semantically opposite. The split is by STT, not "
            "detector: Flux and ElevenLabs swallow the sound, Nova transcribes it and all four "
            "detectors then interrupt on it. Timbal has no notion of a backchannel, so on any "
            "STT that transcribes one, an acknowledgement silences the assistant. Nova fails "
            "it every run; Flux leaks it roughly one run in three, which is how it passed "
            "three straight repeats under `lexical` before failing a re-baseline."
        ),
    ),
    Scenario(
        id="food_trailing_preposition",
        domain="food_ordering",
        replies=["Sure, I can deliver there."],
        script=[
            *fluent("Can you deliver it to", 1.5, "the office on Main Street?"),
            *_SETTLE,
        ],
        expect=[
            Merged("Can you deliver it to the office on Main Street?"),
            NoGhostTurns(),
            NoErrors(),
        ],
        known_failure={
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/local": _AUDIO_PAUSE_SHORTFALL + " — merged 2/3",
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "A stranded preposition: syntactically incomplete, so text EOU should hold without "
            "prosody. It does on Flux and under `local` on Nova — the off-Flux `lexical` failure "
            "is the one that should be fixable on text alone, and is not."
        ),
    ),
    Scenario(
        id="food_rapid_fire",
        domain="food_ordering",
        replies=["Two coffees, got it."],
        script=[
            Say("I'll have a coffee."),
            Silence(0.9),
            Say("Actually, make it two."),
            *_SETTLE,
        ],
        expect=[UserTurns(2), NoGhostTurns(), NoErrors()],
        known_failure={"elevenlabs/*": _EL_OVERMERGE},
        note=(
            "The inverse of every pause scenario: two genuinely finished sentences separated by "
            "a short gap, which must *not* merge. The suite is full of cases punishing a split "
            "and had none punishing a merge, so a detector could have scored well by simply "
            "holding everything."
        ),
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
            *fluent("The pain started", 1.5, "maybe on Tuesday."),
            *_SETTLE,
        ],
        expect=[Merged("The pain started maybe on Tuesday."), NoGhostTurns(), NoErrors()],
        known_failure={"*": _TRAILING_MODIFIER_FAILURE},
        intermittent=True,
        note=(
            "Trailing off mid-recall, the archetypal case for holding instead of "
            "endpointing, and the last of its class still failing after fluent synthesis. "
            "`local` merges it about one run in five, so passing proves nothing."
        ),
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
    Scenario(
        id="medical_barge_in_twice",
        domain="medical_intake",
        replies=[_FEVER_REPLY],
        script=[
            Say("What should I do about a fever?"),
            AwaitAssistantAudio(offset_ms=600),
            Say("Wait, hold on."),
            AwaitAssistantAudio(offset_ms=600),
            Say("Okay, never mind."),
            *_SETTLE,
        ],
        expect=[UserTurns(3), Interrupted(True), NoGhostTurns(), NoErrors()],
        note=(
            "Two interruptions in one session. Every other barge-in scenario stops after the "
            "first, so nothing checked that the session recovers enough to be interrupted "
            "again — a stuck playback or un-cleared interrupt flag would pass all of them."
        ),
    ),
    Scenario(
        id="medical_self_correction",
        domain="medical_intake",
        replies=["Noted, the white one."],
        script=[
            *fluent("I take the blue pill, no wait,", 1.2, "the white one."),
            *_SETTLE,
        ],
        expect=[
            Merged("I take the blue pill, no wait, the white one.", min_similarity=0.8),
            NoGhostTurns(min_similarity=0.5),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/*": _TRAILING_MODIFIER_FAILURE,
            "elevenlabs/local": _CORRECTION_MARKER_HELD + " — merged 5/6",
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "Self-repair mid-utterance. 'No wait,' is a complete clause boundary that reads "
            "finished to text EOU while guaranteeing more speech follows — and committing here "
            "sends the LLM the retracted answer, which is worse than a split usually is. Fails "
            "0/3 in every Flux cell and passes on nova/lexical and three ElevenLabs cells, so "
            "the ceiling is the STT's segmentation rather than the detector's judgement."
        ),
    ),
    Scenario(
        id="medical_filler_midway",
        domain="medical_intake",
        replies=["Thanks. Anything else?"],
        script=[
            *fluent("The pain is, um,", 1.3, "mostly at night."),
            *_SETTLE,
        ],
        expect=[
            Merged("The pain is, um, mostly at night.", min_similarity=0.8),
            NoGhostTurns(min_similarity=0.5),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/provider": _FLUX_PROVIDER_GHOST_I,
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "A filler immediately before the gap, which is where hedges actually occur in "
            "speech. medical_hesitation covers a bare 'Um...' at idle; this covers one "
            "mid-sentence, where the hedge logic has to hold rather than suppress. It merges "
            "the pause correctly everywhere and is marked only for the ghost 'I' — which it "
            "was the first scenario other than banking_digits_pause to produce, retiring the "
            "theory that the ghost had anything to do with digits."
        ),
    ),
    Scenario(
        id="medical_long_utterance",
        domain="medical_intake",
        replies=["That's helpful, thank you."],
        script=[
            Say(
                "I've been having headaches every morning for about two weeks and they "
                "usually fade after breakfast but yesterday it lasted the whole day and "
                "I felt dizzy whenever I stood up too quickly."
            ),
            *_SETTLE,
        ],
        expect=[UserTurns(1), ContentPreserved(min_similarity=0.8), NoGhostTurns(), NoErrors()],
        known_failure={
            "deepgram-nova/lexical": "Nova endpoints at the clause boundaries inside this run-on "
            "and text EOU scores each piece finished, so it commits three turns",
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        note=(
            "Twelve seconds of unbroken speech with several clause boundaries an endpointer "
            "could mistake for an ending. The longest utterance in the suite by a wide margin; "
            "everything else is short enough that a premature commit has nowhere to happen."
        ),
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
        known_failure_under_leak={_ECHO_LEAK_CELL: _ECHO_GARBLED_DIGITS},
        note="Digit strings are the STT-hostile case: no lexical context to fall back on.",
    ),
    Scenario(
        id="banking_digits_pause",
        domain="banking",
        replies=["Got it, account 447291."],
        script=[
            *fluent("My account number is four four seven", 1.4, "two nine one."),
            *_SETTLE,
        ],
        expect=[
            Merged("My account number is four four seven two nine one.", min_similarity=0.75),
            NoGhostTurns(min_similarity=0.4),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/provider": _FLUX_PROVIDER_GHOST_I,
            "deepgram-nova/local": _AUDIO_PAUSE_SHORTFALL + " — merged 2/3",
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "elevenlabs/local": _AUDIO_PAUSE_SHORTFALL + " — merged 2/3",
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
        },
        known_failure_under_leak={_ECHO_LEAK_CELL: _ECHO_GARBLED_DIGITS},
        intermittent=True,
        note=(
            "The hardest merge in the suite: a digit string broken across a pause, with no "
            "words to hint continuation. `lexical` and `local` merge it once the fragment "
            "carries mid-sentence prosody. `provider` mostly merges it too, but drops to a "
            "split plus a ghost 'I' about one run in six, visible only under --repeat. That "
            "ghost was assumed to be about digit strings until medical_filler_midway produced "
            "the identical turn on ordinary words.\n\n"
            "On deepgram-nova/lexical this used to merge and now splits, which is a real "
            "change and the right one: the merge was an artifact of Nova taking 7.5s to "
            "finalize, long enough to swallow the 1.4s gap whole. Once the endpointer armed "
            "for text-only detectors and commits landed at ~1.0s, the pause became visible "
            "and that cell splits it — consistent with its 20% pause-merge rate everywhere "
            "else. A merge bought with 6.5s of dead air was never worth having."
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
    Scenario(
        id="banking_short_reject",
        domain="banking",
        replies=["Understood, I've cancelled it."],
        script=[Say("No."), *_SETTLE],
        expect=[UserTurns(1), NoGhostTurns(min_similarity=0.3), Interrupted(False), NoErrors()],
        note=(
            "The shortest possible real turn. Pairs with banking_confirmation, which is "
            "intermittent for exactly this reason: ElevenLabs has been seen transcribing "
            "one-word answers as partials that never commit, hanging the session outright. A "
            "'No.' that goes unheard is a worse failure than a split."
        ),
    ),
    Scenario(
        id="banking_correction",
        domain="banking",
        replies=["Transferring to account four four eight."],
        script=[
            *fluent("Send it to account four four seven, sorry,", 1.3, "four four eight."),
            *_SETTLE,
        ],
        expect=[
            Merged("Send it to account four four seven, sorry, four four eight.", min_similarity=0.75),
            NoGhostTurns(min_similarity=0.4),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/provider": _TRAILING_MODIFIER_FAILURE,
            "deepgram-nova/local": _CORRECTION_MARKER_HELD + " — merged 4/6",
            "deepgram-nova/lexical": _TRAILING_MODIFIER_FAILURE,
            "elevenlabs/local": _CORRECTION_MARKER_HELD + " — merged 5/6",
            "deepgram-flux/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "Self-correction on a digit string: no lexical context, and committing early sends "
            "the LLM the wrong account number rather than merely a fragment. The one scenario "
            "where a split has a concrete, wrong consequence — the agent acts on 447 when the "
            "caller said 448. `lexical` merges it 3/3 on Flux where `local` and `provider` "
            "never do, the clearest case in the suite of Namo beating the audio EOU outright."
        ),
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
            *fluent("I'm getting an error in my", 1.5, "async generator function."),
            *_SETTLE,
        ],
        expect=[
            Merged("I'm getting an error in my async generator function.", min_similarity=0.8),
            NoGhostTurns(min_similarity=0.5),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/provider": _FLUX_PROVIDER_SPLIT,
            "deepgram-nova/lexical": _TEXT_ONLY_PAUSE_SHORTFALL,
            "deepgram-flux/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "Baselined as a clean pass until --repeat 3 caught `provider` splitting it once in "
            "three. Nothing changed in the product; the suite had only ever run it once per "
            "cell. A worthwhile reminder that a green single run is weak evidence."
        ),
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
    Scenario(
        id="coding_barge_in_echo",
        domain="coding_help",
        replies=[_IMPORTS_REPLY],
        script=[
            Say("Explain how Python handles imports."),
            AwaitAssistantAudio(offset_ms=1400),
            Say("Sorry, the module cache?"),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(True), NoGhostTurns(min_similarity=0.5), NoErrors()],
        note=(
            "A real interruption made of words the assistant is in the middle of saying, which "
            "is what asking someone to repeat themselves sounds like. `_likely_stt_echo` "
            "suppresses partials resembling the assistant's own text to survive a leaky AEC, "
            "and cannot tell this apart from the echo it exists to drop. Measures that "
            "suppressor's false-positive rate on genuine speech."
        ),
    ),
    Scenario(
        id="coding_double_pause",
        domain="coding_help",
        replies=["Sounds like an import cycle. Can you share the traceback?"],
        script=[
            *fluent("The build fails when I", 1.2, "import the module", 1.2, "inside a test."),
            *_SETTLE,
        ],
        expect=[
            Merged("The build fails when I import the module inside a test.", min_similarity=0.8),
            NoGhostTurns(min_similarity=0.5),
            NoErrors(),
        ],
        known_failure={
            "deepgram-flux/*": "holds through the first gap and splits at the second; the "
            "fragment it commits reads as a finished sentence, so nothing argues for holding",
            "elevenlabs/heuristic": _NO_HOLD_TO_MERGE,
            "elevenlabs/provider": _NO_HOLD_TO_MERGE,
            "deepgram-nova/heuristic": _NO_HOLD_TO_MERGE,
            "deepgram-nova/provider": _NO_HOLD_TO_MERGE,
        },
        intermittent=True,
        note=(
            "Two gaps in one sentence, and the first is the one that gets held: every Flux cell "
            "merges 'The build fails when I' with 'import the module' and then commits, leaving "
            "'inside a test' as its own turn. A hold that fires once and latches would look "
            "identical, which is why one-gap scenarios could not have found this. ElevenLabs "
            "merges all three — but see AllPartsHeard: it used to drop 'inside a test' entirely "
            "and still score 0.84 similarity, and this scenario is what exposed that."
        ),
    ),
    Scenario(
        id="coding_followup_after_reply",
        domain="coding_help",
        replies=["A segfault means bad memory access."],
        script=[
            Say("What does a segmentation fault mean?"),
            AwaitCommit(),
            AwaitAssistantDone(),
            # AwaitAssistantDone fires when generation ends, not when playback drains, so
            # speaking here would barge in on the tail. This waits out a ~2.5s reply.
            Silence(5.0),
            Say("And how do I debug it?"),
            *_SETTLE,
        ],
        expect=[UserTurns(2), Interrupted(False), NoGhostTurns(), NoErrors()],
        note=(
            "A second turn taken after the assistant has finished speaking, uninterrupted. "
            "This is a regression test for a bug we shipped: STT not resuming once the "
            "assistant stopped, so the follow-up was never heard. Every other multi-turn "
            "scenario barges in, which exercises the opposite path and would keep passing."
        ),
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
