"""Parameter sweep: score one setting on merges gained against dead air added.

The hold tier is the worked example. ``TEXT_COMPLETE_HOLD_TIMEOUT_SECS`` shipped at
0.35s having been measured at exactly two values: 0.35 (splits four
trailing-modifier scenarios) and 3.0 (fixes 11 scenario-cells and adds 2.6s of
dead air to six barge-in cells). Nobody had tried 0.8, or 1.2 — which is what it
ships at now. The interesting question — is there a setting that buys most of the
merges for a fraction of the silence — was unanswerable until dead air became
measurable, because a configuration that merged everything by holding forever
looked free.

Two metrics, deliberately. Optimizing pass rate alone reproduces the mistake this
harness was built to catch::

    uv run python benchmarks/voice/sweep.py \\
        --param text_complete_hold_timeout_secs --values 0.35,0.8,1.2,2.0,3.0 \\
        --stt deepgram-nova,elevenlabs --detector local --repeat 3

``stt.``-prefixed params vary the provider's own endpointing instead, which is the
other half of the pipeline and the half nobody had tuned::

    uv run python benchmarks/voice/sweep.py \\
        --param stt.vad_silence_threshold_secs --values 0.4,0.6,0.8,1.2 \\
        --stt elevenlabs --detector local,lexical --repeat 3

Defaults to the pause-merge family (every scenario asserting ``Merged``), since
that is the only family where detectors differ at all — barge-in, plain turns and
silence sit at 100% in all twelve cells and would only add runtime and noise.

Results are indicative, not conclusive: 39 English scenarios on three backends,
18 of them carrying a known failure somewhere, and real intermittency. Treat a
winner as a candidate to confirm against the full grid and the Flux gate, never as
a decision.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics as st
import sys
import time
from dataclasses import dataclass, field

import structlog
from dotenv import load_dotenv
from harness import HarnessConfig, coerce_param, config_rejection, run_scenario
from scenario import SCENARIOS, Merged, Scenario
from synth import synthesize_clips, synthesize_fluent


def pause_family() -> list[Scenario]:
    """Scenarios that assert a merge — the only discriminating family."""
    return [s for s in SCENARIOS if any(isinstance(e, Merged) for e in s.expect)]


@dataclass
class Outcome:
    """What one parameter value scored across every cell and repeat."""

    value: str
    passed: int = 0
    total: int = 0
    dead_air: list[float] = field(default_factory=list)
    per_scenario: dict[str, list[bool]] = field(default_factory=dict)
    wall_secs: float = 0.0

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        air = f"{st.median(self.dead_air):>6.0f}" if self.dead_air else "     -"
        p95 = f"{sorted(self.dead_air)[int(0.95 * len(self.dead_air))]:>6.0f}" if self.dead_air else "     -"
        return f"{self.value:>10}  {self.passed:>3}/{self.total:<3} {self.rate:>5.0%}   {air}ms  {p95}ms"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--param",
        required=True,
        help="attribute to vary: a detector attribute, or 'stt.<key>' for a provider STT knob "
        "(see harness.SWEEPABLE_STT_KEYS)",
    )
    parser.add_argument("--values", required=True, help="comma-separated values to try")
    parser.add_argument("--stt", default="deepgram-nova,elevenlabs")
    parser.add_argument("--detector", default="local")
    parser.add_argument("-s", "--scenario", action="append", help="override the default pause family")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every scenario, not just the pause family — needed to price a winner, since the "
        "cost of holding longer lands on closers and barge-ins that the pause family excludes",
    )
    parser.add_argument("--repeat", type=int, default=3, help="runs per scenario per cell (default 3)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    values = [coerce_param(v.strip()) for v in args.values.split(",") if v.strip()]
    param_field, param_key = (
        ("stt_extra", args.param.removeprefix("stt."))
        if args.param.startswith("stt.")
        else ("detector_params", args.param)
    )
    stts = [v.strip() for v in args.stt.split(",") if v.strip()]
    detectors = [v.strip() for v in args.detector.split(",") if v.strip()]

    scenarios = list(SCENARIOS) if args.all else pause_family()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in SCENARIOS if s.id in wanted]
    if not scenarios:
        print("no scenarios selected")
        return 2

    clips = await synthesize_clips({t for s in scenarios for t in s.standalone_texts()})
    clips.update(await synthesize_fluent([g for s in scenarios for g in s.fluent_groups()]))

    cells = [(stt, det) for stt in stts for det in detectors]
    runs = len(values) * len(cells) * len(scenarios) * args.repeat
    print(f"\n{runs} runs: {len(values)} values x {len(cells)} cell(s) x {len(scenarios)} scenarios x {args.repeat}")
    print(f"varying {args.param} over {values}")
    print(f"scenarios: {', '.join(s.id for s in scenarios)}\n")

    semaphore = asyncio.Semaphore(max(1, args.jobs))
    outcomes = {str(v): Outcome(value=str(v)) for v in values}
    started = time.monotonic()

    rejected: dict[str, str] = {}

    async def one(value: object, stt: str, det: str, scenario: Scenario) -> None:
        if str(value) in rejected:
            return
        config = HarnessConfig(stt=stt, detector=det, **{param_field: {param_key: value}})
        async with semaphore:
            result = await run_scenario(scenario, clips, config)
        if why := config_rejection(result.errors):
            rejected.setdefault(str(value), why)
            return
        out = outcomes[str(value)]
        out.total += 1
        out.passed += result.passed
        # Only the final commit. Pooling every dead-air sample rewards splitting:
        # a merge yields one long wait, a split yields two short ones, so the
        # configuration that fragments the utterance scores *better* on silence.
        # Final speech end -> final commit is the number a caller actually feels
        # and is comparable across both outcomes.
        if result.dead_air_ms:
            out.dead_air.append(result.dead_air_ms[-1])
        out.wall_secs += result.wall_secs
        out.per_scenario.setdefault(scenario.id, []).append(result.passed)

    await asyncio.gather(
        *(
            one(value, stt, det, scenario)
            for value in values
            for stt, det in cells
            for scenario in scenarios
            for _ in range(args.repeat)
        )
    )

    print(f"{args.param:>10}  {'merges':<10} {'':>4}  {'dead air p50':>10}  {'p95':>6}")
    for value in values:
        if why := rejected.get(str(value)):
            print(f"  {value!s:>10}  {why}")
            continue
        print("  " + outcomes[str(value)].summary())

    print("\nper scenario (pass rate across cells and repeats)\n")
    header = "".join(f"{v!s:>9}" for v in values)
    print(f"  {'scenario':<30}{header}")
    for scenario in scenarios:
        row = ""
        for value in values:
            if str(value) in rejected:
                row += f"{'-':>8} "
                continue
            runs_ = outcomes[str(value)].per_scenario.get(scenario.id, [])
            row += f"{(sum(runs_) / len(runs_) if runs_ else 0):>8.0%} "
        print(f"  {scenario.id:<30}{row}")

    print(f"\n  elapsed: {time.monotonic() - started:.0f}s")
    print("\n  A winner here is a candidate, not a decision: confirm it against the full")
    print("  12-cell grid and the Flux gate before believing it.")
    # Non-zero on a rejected value: the sweep did not measure what it was asked to.
    return 1 if rejected else 0


if __name__ == "__main__":
    load_dotenv()
    # A sweep is hundreds of sessions, and at DEBUG each one emits a few hundred
    # lines: the result table lands tens of megabytes below the scroll and gets
    # dropped outright by anything that caps captured output. Same default as
    # `cli.py`, which had it from the start.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    sys.exit(asyncio.run(main()))
