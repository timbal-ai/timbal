#!/usr/bin/env python3
"""
Timbal Workflow vs Mastra Workflow — DAG benchmark (cross-language).

Three scenarios with trivial handler functions (no LLM calls):
  1. Sequential:    A → B → C → D
  2. Fan-out/in:    A → [B, C, D] → E
  3. Diamond:       A → [B, C] → D

Timbal runs on CPython/asyncio in this process; Mastra runs on Node/V8 as a
subprocess (bench_workflow.mjs) with the same procedure. Workflow construction
is excluded on both sides; a Mastra run is createRun() + start() — its normal
programmatic execution path. See README.md for cross-runtime caveats.

Run:
    uv run python benchmarks/mastra/bench_workflow.py
    uv run python benchmarks/mastra/bench_workflow.py --quick
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
warnings.filterwarnings("ignore")

import structlog  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

import asyncio  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--timbal-only", action="store_true", help="Skip Mastra, measure Timbal only")
parser.add_argument("--mastra-json", type=Path, default=None, help="Reuse a previous Mastra JSON result instead of running Node")
_args, _ = parser.parse_known_args()

N_ITERS = 50 if _args.quick else 200
N_WARMUP = 5 if _args.quick else 20
N_BURST = 100 if _args.quick else 500
N_MEM = 50 if _args.quick else 200
THROUGHPUT_OPS = 200 if _args.quick else 1000
CONCURRENCY_LEVELS = [1, 10, 50, 200]
WIDTH = 76

BENCH_DIR = Path(__file__).parent

# ── Display helpers ──────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"


def section(title: str) -> None:
    print()
    print(f"{BOLD}{CYAN}{'─' * WIDTH}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * WIDTH}{RESET}")


def subsection(title: str) -> None:
    print(f"\n  {BOLD}{title}{RESET}")


def fmt_us(us: float) -> str:
    if us >= 1_000:
        return f"{us / 1_000:>8.2f} ms"
    return f"{us:>8.1f} µs"


def pct(samples: list[float], p: float) -> float:
    idx = min(int(len(samples) * p / 100), len(samples) - 1)
    return sorted(samples)[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# MASTRA — spawn the Node-side benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_mastra_side(script: str) -> dict | None:
    if _args.timbal_only:
        return None
    if _args.mastra_json:
        return json.loads(_args.mastra_json.read_text())
    node = shutil.which("node")
    if node is None:
        print(f"{DIM}  node not found on PATH — running Timbal only.{RESET}")
        return None
    if not (BENCH_DIR / "node_modules").exists():
        print(f"{DIM}  benchmarks/mastra/node_modules missing — run `npm install` in benchmarks/mastra first.{RESET}")
        return None
    cmd = [node, "--expose-gc", script] + (["--quick"] if _args.quick else [])
    print(f"{DIM}  running Mastra side: {' '.join(cmd[1:])}…{RESET}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=BENCH_DIR, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"{DIM}  Mastra side failed (exit {proc.returncode}) — running Timbal only.{RESET}")
        return None
    return json.loads(proc.stdout)


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — Workflow factories (identical DAG shapes to the Mastra side)
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_traces():
    InMemoryTracingProvider._storage.clear()


def _timbal_sequential() -> Workflow:
    """A → B → C → D"""

    def step_a(x: int) -> int:
        return x + 1

    def step_b(x: int) -> int:
        return x * 2

    def step_c(x: int) -> int:
        return x + 10

    def step_d(x: int) -> int:
        return x - 3

    wf = Workflow(name="sequential")
    wf.step(step_a)
    wf.step(step_b, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(step_c, x=lambda: get_run_context().step_span("step_b").output)
    wf.step(step_d, x=lambda: get_run_context().step_span("step_c").output)
    return wf


def _timbal_fanout() -> Workflow:
    """A → [B, C, D] → E"""

    def step_a(x: int) -> int:
        return x + 1

    def branch_b(x: int) -> int:
        return x * 2

    def branch_c(x: int) -> int:
        return x * 3

    def branch_d(x: int) -> int:
        return x * 4

    def step_e(b: int, c: int, d: int) -> int:
        return b + c + d

    wf = Workflow(name="fanout")
    wf.step(step_a)
    wf.step(branch_b, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(branch_c, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(branch_d, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(
        step_e,
        b=lambda: get_run_context().step_span("branch_b").output,
        c=lambda: get_run_context().step_span("branch_c").output,
        d=lambda: get_run_context().step_span("branch_d").output,
    )
    return wf


def _timbal_diamond() -> Workflow:
    """A → [B, C] → D"""

    def step_a(x: int) -> int:
        return x + 1

    def path_b(x: int) -> int:
        return x + 10

    def path_c(x: int) -> int:
        return x * 5

    def combine(b: int, c: int) -> int:
        return b + c

    wf = Workflow(name="diamond")
    wf.step(step_a)
    wf.step(path_b, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(path_c, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(
        combine,
        b=lambda: get_run_context().step_span("path_b").output,
        c=lambda: get_run_context().step_span("path_c").output,
    )
    return wf


TIMBAL_FACTORIES = {
    "sequential": _timbal_sequential,
    "fanout": _timbal_fanout,
    "diamond": _timbal_diamond,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Timbal measurement (same procedure as the Node side)
# ═══════════════════════════════════════════════════════════════════════════════


async def measure_timbal(name: str) -> dict:
    wf = TIMBAL_FACTORIES[name]()

    async def run():
        await wf(x=3).collect()

    # Latency
    for _ in range(N_WARMUP):
        await run()
    _clear_traces()
    gc.collect()
    lat = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        await run()
        lat.append((time.perf_counter() - t0) * 1e6)
    _clear_traces()

    # Memory
    for _ in range(N_WARMUP):
        await run()
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(N_MEM):
        await run()
        _clear_traces()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Burst
    await asyncio.gather(*[run() for _ in range(10)])
    _clear_traces()
    gc.collect()
    burst: list[float] = []

    async def timed():
        t0 = time.perf_counter()
        await run()
        burst.append((time.perf_counter() - t0) * 1e6)

    await asyncio.gather(*[timed() for _ in range(N_BURST)])
    _clear_traces()

    # Throughput
    tp = {}
    for conc in CONCURRENCY_LEVELS:
        sem = asyncio.Semaphore(conc)

        async def bounded():
            async with sem:
                await run()

        gc.collect()
        t0 = time.perf_counter()
        await asyncio.gather(*[bounded() for _ in range(THROUGHPUT_OPS)])
        tp[str(conc)] = THROUGHPUT_OPS / (time.perf_counter() - t0)
        _clear_traces()

    return {
        "latency_us": {
            "mean": statistics.mean(lat),
            "p50": pct(lat, 50), "p75": pct(lat, 75), "p95": pct(lat, 95), "p99": pct(lat, 99),
            "max": max(lat),
        },
        "mem_per_run_bytes": peak / N_MEM,
        "burst_us": {
            "p50": pct(burst, 50), "p75": pct(burst, 75), "p95": pct(burst, 95), "p99": pct(burst, 99),
            "max": max(burst),
        },
        "throughput_ops_s": tp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Combined output
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_NAMES = {
    "sequential": "Sequential  (A → B → C → D)",
    "fanout": "Fan-out/in  (A → [B, C, D] → E)",
    "diamond": "Diamond  (A → [B, C] → D)",
}

COL_W = 12


def print_scenario(key: str, timbal: dict, mastra: dict | None) -> None:
    section(f"Scenario: {SCENARIO_NAMES[key]}")

    cols = ["Timbal"]
    frames = [timbal]
    if mastra:
        cols += ["Mastra", "Mastra+obs"]
        frames += [mastra["bare"], mastra["obs"]]

    hdr = f"  {'':>{COL_W}}" + "".join(f"  {c:>{COL_W}}" for c in cols)
    sep = f"  {'─' * COL_W}" + f"  {'─' * COL_W}" * len(cols)

    subsection(f"Latency  (×{N_ITERS} sequential runs)")
    print(hdr)
    print(sep)
    for label in ["mean", "p50", "p95", "p99"]:
        vals = [f["latency_us"][label] for f in frames]
        print(f"  {label:>{COL_W}}" + "".join(f"  {fmt_us(v):>{COL_W}}" for v in vals))

    subsection(f"Memory per run  (×{N_MEM} runs) — different instruments, do not ratio across runtimes")
    print(f"  {'framework':<{COL_W + 12}}  {'per run':>14}  {'method':<24}")
    print(f"  {'─' * (COL_W + 12)}  {'─' * 14}  {'─' * 24}")
    print(f"  {'Timbal':<{COL_W + 12}}  {timbal['mem_per_run_bytes']:>12,.0f} B  {'tracemalloc peak/N':<24}")
    if mastra:
        for name, f in [("Mastra", mastra["bare"]), ("Mastra+obs", mastra["obs"])]:
            print(f"  {name:<{COL_W + 12}}  {f['mem_per_run_bytes']:>12,.0f} B  {'V8 heap growth/N (≈)':<24}")

    subsection(f"Burst  ({N_BURST} concurrent runs)")
    print(hdr)
    print(sep)
    for label in ["p50", "p75", "p95", "p99", "max"]:
        vals = [f["burst_us"][label] for f in frames]
        print(f"  {label:>{COL_W}}" + "".join(f"  {fmt_us(v):>{COL_W}}" for v in vals))

    subsection(f"Throughput  ({THROUGHPUT_OPS} runs)")
    print(hdr)
    print(sep)
    for conc in CONCURRENCY_LEVELS:
        vals = [f["throughput_ops_s"][str(conc)] for f in frames]
        print(f"  {conc:>{COL_W}}" + "".join(f"  {v:>10.0f}/s" for v in vals))


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal Workflow vs Mastra Workflow — DAG benchmark (cross-language){RESET}")
    print(f"  {N_ITERS} iters · {N_BURST} burst · {N_MEM} mem runs · {THROUGHPUT_OPS} throughput ops")
    print(f"  {DIM}Trivial handlers (no LLM). Pure DAG scheduling overhead.{RESET}")
    print(f"  {DIM}Timbal: CPython {sys.version_info.major}.{sys.version_info.minor}/asyncio · Mastra: Node/V8 subprocess (createRun + start per run).{RESET}")
    print(f"  {DIM}Cross-runtime comparison — see README.md for what is and isn't comparable.{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print()

    mastra = run_mastra_side("bench_workflow.mjs")
    if mastra:
        m = mastra["meta"]
        print(f"{DIM}  Mastra side: node {m['node']} · @mastra/core {m['mastra_core']} · @mastra/observability {m['mastra_observability']} · ai {m['ai_sdk']}{RESET}")

    for key in ["sequential", "fanout", "diamond"]:
        timbal = await measure_timbal(key)
        print_scenario(key, timbal, mastra["scenarios"][key] if mastra else None)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print("  Both frameworks execute identical DAGs with trivial handlers.")
    print("  Workflow construction excluded; Mastra runs include createRun() + start().")
    print("  Each framework runs in its native runtime; numbers include runtime differences.")
    print("  Memory uses per-runtime instruments (tracemalloc vs V8 heap) — not cross-comparable.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
