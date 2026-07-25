"""Runner for the voice replay harness.

Usage (from repo root)::

    export ELEVENLABS_API_KEY=...
    export DEEPGRAM_API_KEY=...

    uv run python benchmarks/voice/cli.py                            # all scenarios
    uv run python benchmarks/voice/cli.py -s barge_in -s pause       # a subset
    uv run python benchmarks/voice/cli.py --detector lexical         # Timbal's HOLD path
    uv run python benchmarks/voice/cli.py --repeat 3                 # variance + flaky detection
    uv run python benchmarks/voice/cli.py --update-baseline          # accept current behavior
    uv run python benchmarks/voice/cli.py --quiet                    # results only
    uv run python benchmarks/voice/cli.py --dump                     # WAVs per run

Replay runs in real time — the STT provider keeps its own wall clock and Silero
needs an unbroken stream — so a scenario costs roughly what the conversation
would cost a human.

Exit code is non-zero when an expectation fails or a regression is gated.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import structlog
from dotenv import load_dotenv
from harness import HarnessConfig, RunResult, run_scenario
from scenario import SCENARIOS, select
from score import (
    build_scorecard,
    compare,
    format_scorecard,
    load_baseline,
    record,
    save_baseline,
    write_jsonl,
)
from synth import synthesize_clips, synthesize_fluent


def _report(result: RunResult, known_failure: str | None, intermittent: bool = False) -> None:
    print()
    print(f"  turns:       {result.committed}")
    if result.interrupted:
        print(f"  heard:       {result.heard_text!r}")
    if result.latencies_ms:
        print(f"  eou→audio:   {', '.join(f'{v:.0f}ms' for v in result.latencies_ms)}")
    print(f"  audio:       {result.audio_chunks} chunks, {result.audio_bytes} bytes")
    print(f"  wall:        {result.wall_secs:.1f}s")
    for failure in result.failures:
        print(f"    ✗ {failure}")
    if known_failure and not result.passed:
        print(f"  KNOWN FAIL (not gated): {known_failure}")
    elif known_failure and intermittent:
        print("  PASS (known intermittent failure — passing proves nothing)")
    elif known_failure:
        print("  XPASS — known failure now passes, drop the known_failure marker")
    else:
        print(f"  {'PASS' if result.passed else 'FAIL'}")


def _list_scenarios() -> int:
    domain = ""
    for s in SCENARIOS:
        if s.domain != domain:
            domain = s.domain
            print(f"\n{domain}")
        scope = "" if s.detectors is None else f"  detectors={','.join(sorted(s.detectors))}"
        print(f"  {'*' if s.quick else ' '} {s.id:<24}{scope}")
        for detector, reason in s.known_failure.items():
            kind = "intermittent" if s.intermittent else "known failure"
            print(f"      [{kind}, {detector}] {reason}")
        if s.note:
            print(f"      {s.note}")
    known = sum(1 for s in SCENARIOS if s.known_failure)
    print(f"\n{len(SCENARIOS)} scenarios, {known} known failures; * = --quick subset")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--scenario", action="append", help="scenario id (repeatable)")
    parser.add_argument("--stt", default="deepgram-flux", help="elevenlabs | deepgram-flux | deepgram-nova")
    parser.add_argument("--detector", default="provider", help="heuristic | provider | lexical | local")
    parser.add_argument("--language", default="en")
    parser.add_argument("--repeat", type=int, default=1, help="runs per scenario (variance / flaky detection)")
    parser.add_argument("--dump", action="store_true", help="write input/output WAVs per run")
    parser.add_argument("--quick", action="store_true", help="representative subset (one per domain)")
    parser.add_argument("--quiet", action="store_true", help="hide the per-event stream")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--update-baseline", action="store_true", help="accept these results as the baseline")
    parser.add_argument("--no-gate", action="store_true", help="report regressions without failing")
    parser.add_argument("--verbose", action="store_true", help="keep timbal DEBUG logs")
    args = parser.parse_args()

    if args.list:
        return _list_scenarios()

    load_dotenv(override=True)
    # Quiet by default because session/agent INFO logs bury the event stream; DEBUG
    # under --verbose because that is where provider internals like Flux's
    # end_of_turn_confidence are reported.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if args.verbose else logging.WARNING)
    )

    config = HarnessConfig(stt=args.stt, detector=args.detector, language=args.language, dump=args.dump)
    selected, skipped = select(args.scenario, config.detector, quick=args.quick)
    if not selected:
        print(f"nothing to run for detector={config.detector}; scenarios: {[s.id for s in SCENARIOS]}")
        return 2

    if args.update_baseline and (args.scenario or args.quick):
        # A baseline entry is replaced wholesale, so accepting a filtered run would
        # silently drop every scenario it didn't cover and disarm their gates.
        print("refusing to update the baseline from a filtered run: it would drop the other scenarios")
        return 2

    clips = await synthesize_clips([text for s in selected for text in s.standalone_texts()])
    clips |= await synthesize_fluent([g for s in selected for g in s.fluent_groups()])

    records = []
    started = time.monotonic()
    for repeat in range(max(1, args.repeat)):
        for scenario in selected:
            suffix = f"  repeat {repeat + 1}/{args.repeat}" if args.repeat > 1 else ""
            print(f"\n=== {scenario.id}  ({config.label}){suffix} ===")
            t0 = time.monotonic()

            def log(kind: str, detail: str = "", t0=t0) -> None:
                if not args.quiet:
                    print(f"  [+{time.monotonic() - t0:6.2f}s] {kind:<18}{detail}")

            result = await run_scenario(scenario, clips, config, log=log)
            _report(result, scenario.known_failure_reason(config.detector), scenario.intermittent)
            records.append(record(scenario, result, repeat=repeat))

    for scenario in skipped:
        print(f"\n=== {scenario.id}  SKIPPED (not meaningful for {config.detector}) ===")
        if scenario.note:
            print(f"  {scenario.note}")

    card = build_scorecard(records)
    path = write_jsonl(records)

    print(f"\n{'─' * 72}")
    print(format_scorecard(card))
    for name in card.known_failures:
        reason = next(s.known_failure_reason(config.detector) for s in selected if s.id == name)
        print(f"    known: {name} — {reason}")
    print(f"  elapsed:     {time.monotonic() - started:.0f}s   results: {path}")

    if args.update_baseline:
        if card.pass_rate < 1.0:
            # Saving here would record the failure as the accepted status quo and
            # disarm its gate. Either fix it, or mark it known_failure with a reason.
            print(
                "\n  refusing to update the baseline: "
                f"{card.runs - card.passed} run(s) failed and are not marked known_failure"
            )
            return 1
        save_baseline(card)
        print(f"  baseline updated for {card.label}")
        return 0

    comparison = compare(card, load_baseline())
    if comparison.notes:
        for line in comparison.notes:
            print(f"\n  note: {line}")
    if comparison.improvements:
        print("\n  improvements vs baseline:")
        for line in comparison.improvements:
            print(f"    + {line}")
    if comparison.regressions:
        print("\n  REGRESSIONS vs baseline:")
        for line in comparison.regressions:
            print(f"    - {line}")

    failed_expectations = card.pass_rate < 1.0
    gated = comparison.regressions and not args.no_gate
    return 1 if (failed_expectations or gated) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
