"""Scoring, result persistence and regression gating for the voice replay harness.

Three ideas do the work here:

* **Records, not assertions.** Every run serializes to one JSONL line, so a run
  is analyzable after the fact rather than only pass/fail at the time.
* **Deltas, not absolutes.** Gating compares against a committed baseline.
  Absolute thresholds rot within a week because STT providers drift under us —
  yesterday's 400ms ceiling is tomorrow's false alarm in both directions.
* **Flaky is a finding, not noise.** STT is nondeterministic, so a scenario that
  passes some repeats and fails others is surfaced by name. Those are usually a
  real race, not a bad test.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harness import RunResult
from scenario import Scenario, best_window, normalize, similarity

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
# Not under results/ (which is gitignored) — the baseline is meant to be committed.
BASELINE_PATH = HERE / "baseline.json"

# Gate thresholds. Deliberately loose: this catches direction changes, not noise.
LATENCY_REGRESSION_RATIO = 1.15
# Latency gating needs a distribution, not a handful of samples. At suite sizes
# below this, p95 is effectively the maximum and a single slow turn trips the
# gate while every scenario passes (observed: 394ms -> 477ms on an unchanged
# tree). Below the floor, latency is reported but never gated, and the gate uses
# p50 — robust to one outlier — rather than p95.
MIN_LATENCY_SAMPLES = 20


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One scenario × config × repeat."""

    scenario: str
    domain: str
    stt: str
    detector: str
    repeat: int
    passed: bool
    failures: list[str]
    user_turns: int
    ghost_turns: int
    interrupted: bool
    heard_chars: int | None
    latencies_ms: list[float]
    dead_air_ms: list[float]
    vad_endpointed: list[bool]
    errors: list[str]
    wall_secs: float
    known_failure: str = ""
    """Non-empty when this scenario is expected to fail for this detector."""
    intermittent: bool = False
    jobs: int = 1
    """Concurrency this run was measured under. Latency is only comparable at equal
    concurrency, so it is recorded per run rather than inferred later."""
    aec_leak: float = 0.0
    """Echo gain fed back into the mic. Recorded because it changes what the run means
    — a leak run and a clean one were previously indistinguishable in the JSONL, so a
    file could only be read correctly by whoever remembered the command."""
    label: str = ""
    """Full cell label, axis suffixes included (``[leak=0.15]``, swept params).

    Scoring and baselining key off this rather than rebuilding ``stt/detector``, which
    silently dropped every non-default axis: a leak or sweep run matched no cell at
    all, so it printed no scorecard, gated nothing, and its ``--update-baseline`` was
    a no-op. Once matched, the same stripping would have let a leak cell overwrite the
    clean baseline of the same stt/detector."""

    @property
    def xfail(self) -> bool:
        return bool(self.known_failure)

    @property
    def xpass(self) -> bool:
        return self.xfail and self.passed and not self.intermittent

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# Below this, window matching is not evidence: a one- or two-word needle finds a
# passable window in almost any script, and a spurious lone "I" turn (Flux emits
# them) is a real defect that has to stay countable.
MIN_GHOST_WINDOW_WORDS = 3


def count_ghost_turns(result: RunResult, scenario: Scenario, min_similarity: float = 0.6) -> int:
    """Committed turns whose content the script never said.

    Tracked as a metric independent of whether a scenario declared the matching
    expectation — a hallucinated transcript is worth knowing about everywhere.

    Matched against a *window* of the whole script rather than any single spoken
    entry, because turn boundaries are the thing under test and rarely line up
    with script boundaries. Comparing per entry mis-scored both directions: a
    correct three-fragment merge ("The build fails when I import the module.
    Inside a test.") is ~2.5x longer than any fragment of `coding_double_pause`
    and resembled none of them, so five cells reported a hallucinated turn for
    the merge the scenario asserts — while a run-on split into three commits
    counted two ghosts for the same reason inverted. Both were gating.
    """
    spoken = scenario.texts()
    script = " ".join(spoken)
    ghosts = 0
    for text in result.committed:
        if len(normalize(text).split()) >= MIN_GHOST_WINDOW_WORDS:
            attributable = best_window(script, text) >= min_similarity
        else:
            attributable = any(similarity(text, said) >= min_similarity for said in spoken)
        ghosts += not attributable
    return ghosts


def record(
    scenario: Scenario,
    result: RunResult,
    repeat: int = 0,
    jobs: int = 1,
    aec_leak: float = 0.0,
    label: str = "",
) -> RunRecord:
    return RunRecord(
        scenario=scenario.id,
        domain=scenario.domain,
        stt=result.stt,
        detector=result.detector,
        repeat=repeat,
        passed=result.passed,
        failures=list(result.failures),
        user_turns=len(result.committed),
        ghost_turns=count_ghost_turns(result, scenario),
        interrupted=result.interrupted,
        heard_chars=None if result.heard_text is None else len(result.heard_text),
        latencies_ms=[round(v, 1) for v in result.latencies_ms],
        dead_air_ms=[round(v, 1) for v in result.dead_air_ms],
        vad_endpointed=[m.vad_endpointed for m in result.metrics],
        errors=list(result.errors),
        wall_secs=round(result.wall_secs, 2),
        known_failure=scenario.known_failure_reason(result.detector, result.stt, aec_leak) or "",
        intermittent=scenario.is_intermittent(result.detector, result.stt, aec_leak),
        jobs=jobs,
        aec_leak=aec_leak,
        label=label or f"{result.stt}/{result.detector}",
    )


def write_jsonl(records: list[RunRecord], path: Path | None = None) -> Path:
    path = path or RESULTS_DIR / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{r.to_json()}\n" for r in records))
    return path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile. ``q`` in [0, 1].

    Rounded: these land in a committed baseline, and float noise like
    ``382.97999999999996`` makes review diffs unreadable.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 1)


@dataclass
class Scorecard:
    label: str
    stt: str
    detector: str
    runs: int
    """Gated runs only — known failures are counted separately."""
    passed: int
    pass_rate: float
    ghost_turns: int
    errors: int
    latency_p50: float | None
    latency_p95: float | None
    latency_min: float | None = None
    latency_max: float | None = None
    latency_samples: int = 0
    # Speech end -> turn accepted. Gated alongside latency because it is the only
    # metric that can see a hold: `eou→audio` starts counting at the accepted
    # commit, so hold policy is invisible to it by construction.
    dead_air_p50: float | None = None
    dead_air_p95: float | None = None
    dead_air_samples: int = 0
    per_scenario: dict[str, float] = field(default_factory=dict)
    """Gated scenario id -> pass rate across repeats. Known failures are excluded, so
    marking one cannot quietly bake broken behavior into the baseline."""
    flaky: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)
    unexpected_passes: list[str] = field(default_factory=list)
    wall_secs: float = 0.0
    jobs: int = 1
    """Concurrency the cell ran at. Latency gating is suppressed unless this matches
    the baseline's, so a baseline and its runs must agree; the concurrency itself is
    not the problem — see the note in `_compare_latency`."""


def build_scorecard(records: list[RunRecord]) -> Scorecard:
    if not records:
        raise ValueError("no records to score")

    gated = [r for r in records if not r.xfail]
    by_scenario: dict[str, list[RunRecord]] = {}
    for r in gated:
        by_scenario.setdefault(r.scenario, []).append(r)

    per_scenario = {name: sum(r.passed for r in runs) / len(runs) for name, runs in by_scenario.items()}
    # Latency describes the stack, not an expectation, so known failures still count.
    latencies = [v for r in records for v in r.latencies_ms]
    dead_air = [v for r in records for v in r.dead_air_ms]
    passed = sum(r.passed for r in gated)

    return Scorecard(
        # Fallback keeps result files written before labels were recorded scoreable.
        label=records[0].label or f"{records[0].stt}/{records[0].detector}",
        stt=records[0].stt,
        detector=records[0].detector,
        runs=len(gated),
        passed=passed,
        pass_rate=passed / len(gated) if gated else 1.0,
        ghost_turns=sum(r.ghost_turns for r in gated),
        errors=sum(len(r.errors) for r in gated),
        latency_p50=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        latency_min=min(latencies) if latencies else None,
        latency_max=max(latencies) if latencies else None,
        latency_samples=len(latencies),
        dead_air_p50=percentile(dead_air, 0.5),
        dead_air_p95=percentile(dead_air, 0.95),
        dead_air_samples=len(dead_air),
        per_scenario=dict(sorted(per_scenario.items())),
        flaky=sorted(name for name, rate in per_scenario.items() if 0.0 < rate < 1.0),
        known_failures=sorted({r.scenario for r in records if r.xfail}),
        unexpected_passes=sorted({r.scenario for r in records if r.xpass}),
        wall_secs=round(sum(r.wall_secs for r in records), 1),
        jobs=max(r.jobs for r in records),
    )


def format_scorecard(card: Scorecard) -> str:
    lines = [
        f"{card.label}: {card.passed}/{card.runs} passed ({card.pass_rate:.0%})",
        f"  ghost turns: {card.ghost_turns}   session errors: {card.errors}",
    ]
    if card.latency_p50 is not None:
        gated = "" if card.latency_samples >= MIN_LATENCY_SAMPLES else "  (ungated: too few samples)"
        lines.append(
            f"  eou→audio:   p50 {card.latency_p50:.0f}ms   p95 {card.latency_p95:.0f}ms   "
            f"range {card.latency_min:.0f}-{card.latency_max:.0f}ms   "
            f"n={card.latency_samples}{gated}"
        )
    if card.dead_air_p50 is not None:
        gated = "" if card.dead_air_samples >= MIN_LATENCY_SAMPLES else "  (ungated: too few samples)"
        lines.append(
            f"  dead air:    p50 {card.dead_air_p50:.0f}ms   p95 {card.dead_air_p95:.0f}ms   "
            f"n={card.dead_air_samples}{gated}   (speech end → turn accepted)"
        )
    failing = [name for name, rate in card.per_scenario.items() if rate < 1.0]
    if failing:
        lines.append(f"  failing:     {', '.join(failing)}")
    if card.flaky:
        lines.append(f"  FLAKY:       {', '.join(card.flaky)}  (passed some repeats, not others)")
    if card.known_failures:
        lines.append(f"  known fail:  {', '.join(card.known_failures)}  (reported, not gated)")
    if card.unexpected_passes:
        lines.append(
            f"  XPASS:       {', '.join(card.unexpected_passes)}  — now passing, drop the known_failure marker"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# Grid glyphs. Deliberately distinguishes "failed" from "expected to fail": a cell
# full of x is a documented limitation, a single ✗ is a regression.
_PASS, _FAIL, _XFAIL, _XPASS, _FLAKY, _ABSENT = "✓", "✗", "x", "!", "~", "·"


def format_grid(cards: list[Scorecard], records: list[RunRecord]) -> str:
    """Scenario × cell grid — the question the matrix exists to answer.

    Reading down a column compares scenarios within one configuration; reading
    across a row shows which configurations handle a given situation, which is the
    only way to tell a Timbal bug from a provider quirk.
    """
    labels = [c.label for c in cards]
    by_cell: dict[str, dict[str, list[RunRecord]]] = {label: {} for label in labels}
    for r in records:
        cell = by_cell.get(f"{r.stt}/{r.detector}")
        if cell is not None:
            cell.setdefault(r.scenario, []).append(r)

    scenarios = sorted({r.scenario for r in records})
    width = max((len(s) for s in scenarios), default=0)
    columns = [str(i + 1) for i in range(len(labels))]

    def glyph(runs: list[RunRecord]) -> str:
        if not runs:
            return _ABSENT
        rate = sum(r.passed for r in runs) / len(runs)
        if runs[0].xfail:
            return _XPASS if all(r.xpass for r in runs) else _XFAIL
        if rate == 1.0:
            return _PASS
        return _FAIL if rate == 0.0 else _FLAKY

    lines = ["", "matrix  (rows: scenarios, columns: cells)", ""]
    lines.append(f"  {'':<{width}}  " + "  ".join(f"{c:>3}" for c in columns))
    for name in scenarios:
        row = "  ".join(f"{glyph(by_cell[label].get(name, [])):>3}" for label in labels)
        lines.append(f"  {name:<{width}}  {row}")

    lines.append("")
    for i, card in enumerate(cards):
        rate = f"{card.passed}/{card.runs}"
        p50 = f"{card.latency_p50:.0f}ms" if card.latency_p50 is not None else "-"
        lines.append(f"  {i + 1:>3}  {card.label:<28} {rate:>7}  p50 {p50:>7}")
    lines.append("")
    lines.append(
        f"  {_PASS} pass   {_FAIL} FAIL   {_XFAIL} known failure   {_XPASS} XPASS   {_FLAKY} flaky   {_ABSENT} not run"
    )
    return "\n".join(lines)


def cross_cell_flaky(records: list[RunRecord]) -> list[str]:
    """Scenarios that are flaky *within* at least one cell.

    Deliberately not "differs across cells" — that is the matrix working as
    intended, since detectors are supposed to behave differently. Flakiness inside
    a single cell is the one that means a race.
    """
    by_cell_scenario: dict[tuple[str, str], list[RunRecord]] = {}
    for r in records:
        by_cell_scenario.setdefault((f"{r.stt}/{r.detector}", r.scenario), []).append(r)
    flaky = set()
    for (label, scenario), runs in by_cell_scenario.items():
        if len(runs) < 2:
            continue
        rate = sum(r.passed for r in runs) / len(runs)
        if 0.0 < rate < 1.0:
            flaky.add(f"{scenario} [{label}] {sum(r.passed for r in runs)}/{len(runs)}")
    return sorted(flaky)


# ---------------------------------------------------------------------------
# Baseline + gating
# ---------------------------------------------------------------------------


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_baseline(card: Scorecard, path: Path = BASELINE_PATH) -> None:
    """Merge one config's scorecard into the baseline, leaving the others alone."""
    baseline = load_baseline(path)
    baseline[card.label] = asdict(card)
    path.write_text(json.dumps(dict(sorted(baseline.items())), indent=2) + "\n")


@dataclass
class Comparison:
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Informational: movement seen but deliberately not gated."""

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(card: Scorecard, baseline: dict[str, dict], partial: bool = False) -> Comparison:
    """Gate on movement away from the baseline, not on absolute thresholds.

    ``partial`` marks a filtered run (``-s`` / ``--quick``). Per-scenario rates stay
    comparable because they are matched by name, but every *aggregate* is computed over
    a different population than the baseline's, so comparing them invents movement that
    is only a change of subject. Measured: the barge-in subset alone carries most of the
    suite's ghost turns, so running just those reported "ghost turns 1 → 3" against a
    full-suite baseline while being clean — and the reverse is worse, since a subset
    that excludes them looks like an improvement. Latency and dead air are the same
    story, a p50 over eight scenarios against a p50 over thirty-nine.
    """
    out = Comparison()
    previous = baseline.get(card.label)
    if previous is None:
        out.notes.append(f"no baseline for {card.label} yet — run with --update-baseline")
        return out

    was: dict[str, float] = previous.get("per_scenario", {})
    for name, rate in card.per_scenario.items():
        before = was.get(name)
        if before is None:
            continue
        if rate < before:
            out.regressions.append(f"{name}: pass rate {before:.0%} → {rate:.0%}")
        elif rate > before:
            out.improvements.append(f"{name}: pass rate {before:.0%} → {rate:.0%}")

    new_scenarios = sorted(set(card.per_scenario) - set(was))
    if new_scenarios:
        out.notes.append(f"new scenarios, nothing to compare: {', '.join(new_scenarios)}")

    for name in card.unexpected_passes:
        out.improvements.append(f"{name}: known failure now passes — drop the known_failure marker")

    if partial:
        out.notes.append(
            "filtered run: per-scenario rates gated, aggregates (ghost turns, latency, dead air) not comparable"
        )
        return out

    if card.ghost_turns > previous.get("ghost_turns", 0):
        out.regressions.append(f"ghost turns {previous.get('ghost_turns', 0)} → {card.ghost_turns}")

    _compare_timing(out, card, previous, "latency", "eou→audio p50", card.latency_p50, card.latency_samples)
    _compare_timing(out, card, previous, "dead_air", "dead air p50", card.dead_air_p50, card.dead_air_samples)

    return out


def _compare_timing(
    out: Comparison,
    card: Scorecard,
    previous: dict,
    key: str,
    label: str,
    now: float | None,
    samples: int,
) -> None:
    """Gate one timing percentile against its baseline entry.

    Both timings gate identically, but they answer different questions and a
    change can move one without the other: removing the text-complete hold tier
    left `eou→audio` flat — it starts at the accepted commit — while adding
    2.6s of dead air to six barge-in cells.
    """
    before = previous.get(f"{key}_p50")
    if not before or not now:
        return
    delta = f"{label} {before:.0f}ms → {now:.0f}ms"
    # Only compare like with like. Measured on deepgram-nova/local, --quick,
    # 3 repeats, contention is not detectable at --jobs 6: p50 278ms vs 280ms
    # serial, p95 407ms vs 453ms — better, because eou→audio is mostly waiting
    # on STT and TTS sockets rather than competing for CPU, and the serial run's
    # one 962ms sample is first-run model warmup weighing on a smaller n. The
    # guard stays because it is free and the equality is what makes it sound.
    before_jobs = previous.get("jobs", 1)
    if card.jobs != before_jobs:
        out.notes.append(f"{delta} — not gated, measured at --jobs {card.jobs} against a --jobs {before_jobs} baseline")
        return
    moved = now / before
    if moved > LATENCY_REGRESSION_RATIO:
        worse = f"{delta} (>{(LATENCY_REGRESSION_RATIO - 1) * 100:.0f}% worse)"
        if min(samples, previous.get(f"{key}_samples", 0)) >= MIN_LATENCY_SAMPLES:
            out.regressions.append(worse)
        else:
            out.notes.append(f"{worse} — not gated, fewer than {MIN_LATENCY_SAMPLES} samples")
    elif moved < 1 / LATENCY_REGRESSION_RATIO:
        out.improvements.append(delta)
