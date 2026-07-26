"""Runner for the voice replay harness.

Usage (from repo root)::

    export ELEVENLABS_API_KEY=...
    export DEEPGRAM_API_KEY=...

    uv run python benchmarks/voice/cli.py                            # all scenarios
    uv run python benchmarks/voice/cli.py -s barge_in -s pause       # a subset
    uv run python benchmarks/voice/cli.py --detector lexical         # Timbal's HOLD path
    uv run python benchmarks/voice/cli.py --detector local,lexical,provider --jobs 4
    uv run python benchmarks/voice/cli.py --repeat 3                 # variance + flaky detection
    uv run python benchmarks/voice/cli.py --update-baseline          # accept current behavior
    uv run python benchmarks/voice/cli.py --quiet                    # results only
    uv run python benchmarks/voice/cli.py --dump                     # WAVs per run

``--stt`` and ``--detector`` take comma-separated lists and are crossed into a
matrix, one scorecard and one baseline entry per cell. A row of the resulting grid
answers the question a single cell cannot: whether a scenario fails because Timbal
is wrong or because one provider is.

Replay runs in real time — the STT provider keeps its own wall clock and Silero
needs an unbroken stream — so a scenario costs roughly what the conversation
would cost a human. ``--jobs`` overlaps runs to claw that back, at the cost of
latency fidelity: concurrent sessions contend for the same ONNX inference, so
those numbers are reported but never gated (§ ``score.compare``).

Exit code is non-zero when an expectation fails or a regression is gated.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass

import structlog
from dotenv import load_dotenv
from harness import HarnessConfig, RunResult, coerce_param, run_scenario
from scenario import SCENARIOS, Scenario, select
from score import (
    RunRecord,
    build_scorecard,
    compare,
    cross_cell_flaky,
    format_grid,
    format_scorecard,
    load_baseline,
    record,
    save_baseline,
    write_jsonl,
)
from synth import synthesize_clips, synthesize_fluent


def _report(result: RunResult, known_failure: str | None, intermittent: bool = False) -> list[str]:
    lines = ["", f"  turns:       {result.committed}"]
    if result.interrupted:
        lines.append(f"  heard:       {result.heard_text!r}")
    if result.latencies_ms:
        lines.append(f"  eou→audio:   {', '.join(f'{v:.0f}ms' for v in result.latencies_ms)}")
    lines.append(f"  audio:       {result.audio_chunks} chunks, {result.audio_bytes} bytes")
    lines.append(f"  wall:        {result.wall_secs:.1f}s")
    lines.extend(f"    ✗ {failure}" for failure in result.failures)
    if known_failure and not result.passed:
        lines.append(f"  KNOWN FAIL (not gated): {known_failure}")
    elif known_failure and intermittent:
        lines.append("  PASS (known intermittent failure — passing proves nothing)")
    elif known_failure:
        lines.append("  XPASS — known failure now passes, drop the known_failure marker")
    else:
        lines.append(f"  {'PASS' if result.passed else 'FAIL'}")
    return lines


def _values(flags: list[str] | None, default: str) -> list[str]:
    """Flatten repeated and comma-separated flags, order preserved, deduplicated."""
    if not flags:
        return [default]
    out = [part.strip() for flag in flags for part in flag.split(",") if part.strip()]
    return list(dict.fromkeys(out))


def _parse_params(flags: list[str] | None) -> dict[str, object]:
    """``KEY=VALUE`` flags to a coerced mapping."""
    params: dict[str, object] = {}
    for flag in flags or []:
        key, _, raw = flag.partition("=")
        if not _ or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {flag!r}")
        params[key.strip()] = coerce_param(raw.strip())
    return params


@dataclass
class _Job:
    config: HarnessConfig
    scenario: Scenario
    repeat: int
    repeats: int

    @property
    def header(self) -> str:
        suffix = f"  repeat {self.repeat + 1}/{self.repeats}" if self.repeats > 1 else ""
        return f"\n=== {self.scenario.id}  ({self.config.label}){suffix} ==="


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
        for detector, reason in s.known_failure_under_leak.items():
            print(f"      [under --aec-leak, {detector}] {reason}")
        if s.note:
            print(f"      {s.note}")
    known = sum(1 for s in SCENARIOS if s.known_failure)
    leak_known = sum(1 for s in SCENARIOS if s.known_failure_under_leak)
    print(
        f"\n{len(SCENARIOS)} scenarios, {known} known failures, "
        f"{leak_known} more under --aec-leak; * = --quick subset"
    )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--scenario", action="append", help="scenario id (repeatable)")
    parser.add_argument("--stt", action="append", help="elevenlabs | deepgram-flux | deepgram-nova (comma-separated)")
    parser.add_argument("--detector", action="append", help="heuristic | provider | lexical | local (comma-separated)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--repeat", type=int, default=1, help="runs per scenario (variance / flaky detection)")
    parser.add_argument(
        "--jobs", type=int, default=1, help="concurrent runs; latency gates whenever this matches the baseline's"
    )
    parser.add_argument("--dump", action="store_true", help="write input/output WAVs per run")
    parser.add_argument(
        "--aec-leak",
        type=float,
        default=0.0,
        metavar="GAIN",
        help="mix the assistant's own output back into the mic at this gain (0.1-0.3 is a "
        "realistic imperfect echo canceller); exercises the echo suppressor, which clean "
        "user-only audio never does",
    )
    # Reproducing one cell of a sweep with the event stream visible was the missing
    # step between "this value scores worse" and knowing why.
    parser.add_argument(
        "--stt-param",
        action="append",
        metavar="KEY=VALUE",
        help="provider STT knob, e.g. vad_silence_threshold_secs=0.3 (see harness.SWEEPABLE_STT_KEYS)",
    )
    parser.add_argument(
        "--detector-param",
        action="append",
        metavar="KEY=VALUE",
        help="detector attribute, e.g. text_complete_hold_timeout_secs=1.2",
    )
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

    jobs = max(1, args.jobs)
    repeats = max(1, args.repeat)
    try:
        stt_extra = _parse_params(args.stt_param)
        detector_params = _parse_params(args.detector_param)
    except ValueError as e:
        print(e)
        return 2
    cells = [
        HarnessConfig(
            stt=stt,
            detector=detector,
            language=args.language,
            dump=args.dump,
            aec_leak=args.aec_leak,
            stt_extra=stt_extra,
            detector_params=detector_params,
        )
        for stt in _values(args.stt, "deepgram-flux")
        for detector in _values(args.detector, "provider")
    ]

    if args.update_baseline and (args.scenario or args.quick):
        # A baseline entry is replaced wholesale, so accepting a filtered run would
        # silently drop every scenario it didn't cover and disarm their gates.
        print("refusing to update the baseline from a filtered run: it would drop the other scenarios")
        return 2
    # A parallel baseline used to be refused here, on the grounds that contention
    # would bake the machine's load into the latency the gate compares against.
    # Measured, that does not happen: deepgram-nova/local, --quick, 3 repeats, is
    # p50 278ms / p95 407ms at --jobs 6 against 280ms / 453ms serial — no worse,
    # because eou→audio is mostly waiting on STT and TTS sockets rather than
    # competing for CPU. What keeps this sound is that the baseline records its own
    # concurrency and `_compare_latency` declines to compare across a mismatch, so
    # a parallel baseline is only ever read by equally parallel runs. The refusal
    # bought no protection against a busy machine at --jobs 1 either, and cost a
    # ~6x slower baseline, which is the kind of price that stops people taking one.

    queue: list[_Job] = []
    skipped: list[tuple[HarnessConfig, Scenario]] = []
    for config in cells:
        selected, cell_skipped = select(args.scenario, config.detector, quick=args.quick)
        skipped.extend((config, s) for s in cell_skipped)
        queue.extend(_Job(config, scenario, repeat, repeats) for repeat in range(repeats) for scenario in selected)
    if not queue:
        detectors = ", ".join(sorted({c.detector for c in cells}))
        print(f"nothing to run for detector(s) {detectors}; scenarios: {[s.id for s in SCENARIOS]}")
        return 2

    wanted = {job.scenario.id: job.scenario for job in queue}.values()
    clips = await synthesize_clips([text for s in wanted for text in s.standalone_texts()])
    clips |= await synthesize_fluent([g for s in wanted for g in s.fluent_groups()])

    if len(cells) > 1 or jobs > 1:
        print(f"\n{len(queue)} run(s) across {len(cells)} cell(s) at --jobs {jobs}")

    # Serial keeps the live event stream, which is the whole debugging story for a
    # single scenario. Concurrent runs buffer it, because interleaved streams from
    # four sessions are unreadable and worse than none.
    live = jobs == 1 and not args.quiet
    semaphore = asyncio.Semaphore(jobs)
    started = time.monotonic()

    async def run(job: _Job) -> RunRecord:
        async with semaphore:
            if live:
                print(job.header)
            t0 = time.monotonic()
            buffer: list[str] = []

            def log(kind: str, detail: str = "") -> None:
                if args.quiet:
                    return
                line = f"  [+{time.monotonic() - t0:6.2f}s] {kind:<18}{detail}"
                print(line) if live else buffer.append(line)

            result = await run_scenario(job.scenario, clips, job.config, log=log)
            leak = job.config.aec_leak
            known = job.scenario.known_failure_reason(job.config.detector, job.config.stt, leak)
            report = _report(result, known, job.scenario.is_intermittent(job.config.detector, job.config.stt, leak))
            if live:
                print("\n".join(report))
            else:
                print("\n".join([job.header, *buffer, *report]))
            return record(
                job.scenario, result, repeat=job.repeat, jobs=jobs, aec_leak=leak, label=job.config.label
            )

    records = list(await asyncio.gather(*(run(job) for job in queue)))

    for config, scenario in skipped:
        print(f"\n=== {scenario.id}  ({config.label})  SKIPPED (not meaningful for {config.detector}) ===")
        if scenario.note:
            print(f"  {scenario.note}")

    path = write_jsonl(records)
    baseline = load_baseline()
    cards = []
    exit_code = 0

    for config in cells:
        cell_records = [r for r in records if r.label == config.label]
        if not cell_records:
            continue
        card = build_scorecard(cell_records)
        cards.append(card)

        print(f"\n{'─' * 72}")
        print(format_scorecard(card))
        for name in card.known_failures:
            reason = next(r.known_failure for r in cell_records if r.scenario == name)
            print(f"    known: {name} — {reason}")

        if args.update_baseline:
            if card.pass_rate < 1.0:
                # Saving here would record the failure as the accepted status quo
                # and disarm its gate. Either fix it, or mark it known_failure.
                print(
                    "\n  refusing to update the baseline: "
                    f"{card.runs - card.passed} run(s) failed and are not marked known_failure"
                )
                exit_code = 1
                continue
            save_baseline(card)
            print(f"  baseline updated for {card.label}")
            continue

        comparison = compare(card, baseline)
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
        if card.pass_rate < 1.0 or (comparison.regressions and not args.no_gate):
            exit_code = 1

    if len(cards) > 1:
        print(f"\n{'─' * 72}")
        print(format_grid(cards, records))

    flaky = cross_cell_flaky(records)
    if flaky:
        print("\n  FLAKY (passed some repeats, not others):")
        for line in flaky:
            print(f"    ~ {line}")

    print(f"\n  elapsed:     {time.monotonic() - started:.0f}s   results: {path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
