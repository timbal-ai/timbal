#!/usr/bin/env python3
"""Measure Runnable timeout overhead with identical token-like streams.

Run from the repository root:
    uv run python benchmarks/timeout/bench_streaming.py --quick
    uv run python benchmarks/timeout/bench_streaming.py
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import platform
import statistics
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import structlog
from timbal import Tool
from timbal.state import set_run_context
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
from timbal.types.events.delta import TextDelta


async def tokens(chunks: int) -> AsyncGenerator[TextDelta, None]:
    for _ in range(chunks):
        yield TextDelta(id="text", text_delta="x")


async def measure(tool: Tool, chunks: int, iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        # Start independent runs and avoid growing trace/session history. Let the
        # loop discard cancelled timer handles between runs, outside the timing.
        set_run_context(None)
        InMemoryTracingProvider._storage.clear()
        await asyncio.sleep(0)
        started = time.perf_counter_ns()
        result = await tool(chunks=chunks).collect()
        elapsed_us = (time.perf_counter_ns() - started) / 1_000
        if result.status.code != "success" or result.error is not None:
            raise RuntimeError(f"Benchmark run failed: {result.status}: {result.error}")
        samples.append(elapsed_us)
    return samples


async def run(args: argparse.Namespace) -> dict:
    tools = {
        "unset": Tool(name="tokens", handler=tokens, tracing_provider=InMemoryTracingProvider),
        "enabled": Tool(name="tokens", handler=tokens, timeout=120, tracing_provider=InMemoryTracingProvider),
    }
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timeout_seconds": 120,
        "rounds": args.rounds,
        "iterations_per_round": args.iterations,
        "tracing": "InMemoryTracingProvider",
        "scenarios": [],
    }
    for chunks in args.chunks:
        rounds = []
        for tool in tools.values():
            await measure(tool, chunks, 10)
        for index in range(args.rounds):
            row = {}
            # Alternate order to reduce systematic warmup/thermal bias.
            modes = ("unset", "enabled") if index % 2 == 0 else ("enabled", "unset")
            for mode in modes:
                gc.collect()
                samples = await measure(tools[mode], chunks, args.iterations)
                row[mode] = {
                    "p50_us": statistics.median(samples),
                    "p95_us": sorted(samples)[int((len(samples) - 1) * 0.95)],
                }
            rounds.append(row)
        unset = statistics.median(row["unset"]["p50_us"] for row in rounds)
        enabled = statistics.median(row["enabled"]["p50_us"] for row in rounds)
        scenario = {
            "chunks": chunks,
            "unset_p50_us": unset,
            "enabled_p50_us": enabled,
            "overhead_us_per_run": enabled - unset,
            "overhead_us_per_chunk": (enabled - unset) / chunks,
            "overhead_percent": (enabled / unset - 1) * 100,
            "rounds": rounds,
        }
        report["scenarios"].append(scenario)
        print(
            f"{chunks:>6} chunks | unset {unset:>9.1f} us | timeout {enabled:>9.1f} us | "
            f"{scenario['overhead_percent']:>+6.1f}% | "
            f"{scenario['overhead_us_per_chunk']:>6.2f} extra us/chunk"
        )
    InMemoryTracingProvider._storage.clear()
    set_run_context(None)
    return report


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--chunks", nargs="+", type=positive_int, default=[1, 100, 1000])
    parser.add_argument("--rounds", type=positive_int)
    parser.add_argument("--iterations", type=positive_int)
    parser.add_argument("--json", type=Path, help="Save raw per-round metrics and environment metadata")
    args = parser.parse_args()
    args.rounds = args.rounds or (3 if args.quick else 7)
    args.iterations = args.iterations or (20 if args.quick else 200)
    logging.disable(logging.CRITICAL)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
    print(f"Python {platform.python_version()} | {args.rounds} rounds x {args.iterations} runs | tracing enabled")
    report = asyncio.run(run(args))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
