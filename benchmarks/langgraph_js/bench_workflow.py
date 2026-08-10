#!/usr/bin/env python3
"""
Runtime calibration — LangGraph Python vs LangGraph.js (vs Timbal) DAG workflows.

Same StateGraph shapes on both LangGraph runtimes, plus Timbal Workflow for
reference. The LG.py÷LG.js ratio isolates CPython-vs-V8 for identical Pregel
scheduling — the calibration factor for reading cross-language DAG benchmarks
like benchmarks/mastra/bench_workflow.py.

Scenarios: sequential (A→B→C→D), fan-out/in (A→[B,C,D]→E), diamond (A→[B,C]→D).
Trivial handlers, no LLM.

Run:
    uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_workflow.py
    uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_workflow.py --quick
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
parser.add_argument("--timbal-only", action="store_true", help="Skip LangGraph on both runtimes")
parser.add_argument("--lgjs-json", type=Path, default=None, help="Reuse a previous LG.js JSON result instead of running Node")
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
# LANGGRAPH.JS — spawn the Node-side benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_lgjs_side(script: str) -> dict | None:
    if _args.timbal_only:
        return None
    if _args.lgjs_json:
        return json.loads(_args.lgjs_json.read_text())
    node = shutil.which("node")
    if node is None:
        print(f"{DIM}  node not found on PATH — skipping LangGraph.js.{RESET}")
        return None
    if not (BENCH_DIR / "node_modules").exists():
        print(f"{DIM}  benchmarks/langgraph_js/node_modules missing — run `npm install` there first.{RESET}")
        return None
    cmd = [node, "--expose-gc", script] + (["--quick"] if _args.quick else [])
    print(f"{DIM}  running LangGraph.js side: {' '.join(cmd[1:])}…{RESET}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=BENCH_DIR, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"{DIM}  LangGraph.js side failed (exit {proc.returncode}).{RESET}")
        return None
    return json.loads(proc.stdout)


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — Workflow factories (identical shapes)
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_traces():
    InMemoryTracingProvider._storage.clear()


def _timbal_sequential() -> Workflow:
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
# LANGGRAPH PYTHON — StateGraph factories (identical to benchmarks/langchain)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    HAS_LANGGRAPH_PY = True
except ImportError:
    HAS_LANGGRAPH_PY = False

if HAS_LANGGRAPH_PY:
    class SeqState(TypedDict):
        x: int
        a: int
        b: int
        c: int
        d: int

    def _lg_sequential():
        g = StateGraph(SeqState)
        g.add_node("A", lambda s: {"a": s["x"] + 1})
        g.add_node("B", lambda s: {"b": s["a"] * 2})
        g.add_node("C", lambda s: {"c": s["b"] + 10})
        g.add_node("D", lambda s: {"d": s["c"] - 3})
        g.add_edge(START, "A")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", END)
        return g.compile()

    class FanState(TypedDict):
        x: int
        a: int
        bb: int
        cc: int
        dd: int
        e: int

    def _lg_fanout():
        g = StateGraph(FanState)
        g.add_node("A", lambda s: {"a": s["x"] + 1})
        g.add_node("B", lambda s: {"bb": s["a"] * 2})
        g.add_node("C", lambda s: {"cc": s["a"] * 3})
        g.add_node("D", lambda s: {"dd": s["a"] * 4})
        g.add_node("E", lambda s: {"e": s["bb"] + s["cc"] + s["dd"]})
        g.add_edge(START, "A")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "E")
        g.add_edge("C", "E")
        g.add_edge("D", "E")
        g.add_edge("E", END)
        return g.compile()

    class DiamondState(TypedDict):
        x: int
        a: int
        b: int
        c: int
        combined: int

    def _lg_diamond():
        g = StateGraph(DiamondState)
        g.add_node("A", lambda s: {"a": s["x"] + 1})
        g.add_node("B", lambda s: {"b": s["a"] + 10})
        g.add_node("C", lambda s: {"c": s["a"] * 5})
        g.add_node("D", lambda s: {"combined": s["b"] + s["c"]})
        g.add_edge(START, "A")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        g.add_edge("D", END)
        return g.compile()

    LG_FACTORIES = {"sequential": _lg_sequential, "fanout": _lg_fanout, "diamond": _lg_diamond}


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement (same procedure as the Node side)
# ═══════════════════════════════════════════════════════════════════════════════


async def measure_python(run, clear=None) -> dict:
    clear = clear or (lambda: None)

    for _ in range(N_WARMUP):
        await run()
    clear()
    gc.collect()
    lat = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        await run()
        lat.append((time.perf_counter() - t0) * 1e6)
    clear()

    for _ in range(N_WARMUP):
        await run()
    clear()
    gc.collect()
    tracemalloc.start()
    for _ in range(N_MEM):
        await run()
        clear()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    await asyncio.gather(*[run() for _ in range(10)])
    clear()
    gc.collect()
    burst: list[float] = []

    async def timed():
        t0 = time.perf_counter()
        await run()
        burst.append((time.perf_counter() - t0) * 1e6)

    await asyncio.gather(*[timed() for _ in range(N_BURST)])
    clear()

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
        clear()

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


def print_scenario(key: str, timbal: dict, lgpy: dict | None, lgjs: dict | None) -> None:
    section(f"Scenario: {SCENARIO_NAMES[key]}")

    cols = ["Timbal"]
    frames = [timbal]
    if lgpy:
        cols.append("LG.py")
        frames.append(lgpy)
    if lgjs:
        cols.append("LG.js")
        frames.append(lgjs)

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
    if lgpy:
        print(f"  {'LG.py':<{COL_W + 12}}  {lgpy['mem_per_run_bytes']:>12,.0f} B  {'tracemalloc peak/N':<24}")
    if lgjs:
        print(f"  {'LG.js':<{COL_W + 12}}  {lgjs['mem_per_run_bytes']:>12,.0f} B  {'V8 heap growth/N (≈)':<24}")

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

    if lgpy and lgjs:
        subsection("Runtime calibration  (LG.py ÷ LG.js — same framework, so ≈ CPython/V8 factor)")
        p50 = lgpy["latency_us"]["p50"] / lgjs["latency_us"]["p50"]
        mean = lgpy["latency_us"]["mean"] / lgjs["latency_us"]["mean"]
        tp10 = lgjs["throughput_ops_s"]["10"] / lgpy["throughput_ops_s"]["10"]
        tp200 = lgjs["throughput_ops_s"]["200"] / lgpy["throughput_ops_s"]["200"]
        print(f"  latency p50: {p50:.2f}x  ·  mean: {mean:.2f}x  ·  throughput c=10: {tp10:.2f}x  ·  c=200: {tp200:.2f}x")


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Runtime calibration — LangGraph Python vs LangGraph.js (DAG workflows){RESET}")
    print(f"  {N_ITERS} iters · {N_BURST} burst · {N_MEM} mem runs · {THROUGHPUT_OPS} throughput ops")
    print(f"  {DIM}Same framework, both runtimes: the LG.py÷LG.js ratio isolates CPython vs V8.{RESET}")
    print(f"  {DIM}Timbal shown for reference (built-in tracing always on; LG columns are bare).{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print()

    if not HAS_LANGGRAPH_PY and not _args.timbal_only:
        print(f"{DIM}  langgraph (Python) not importable — run via:{RESET}")
        print(f"{DIM}  uv run --no-sync --with langgraph --with langchain-core python {Path(__file__).relative_to(Path.cwd())}{RESET}")

    lgjs = run_lgjs_side("bench_workflow.mjs")
    if lgjs:
        m = lgjs["meta"]
        print(f"{DIM}  LG.js side: node {m['node']} · @langchain/langgraph {m['langgraph']} · @langchain/core {m['langchain_core']}{RESET}")

    ratios = []
    for key in ["sequential", "fanout", "diamond"]:
        wf = TIMBAL_FACTORIES[key]()

        async def t_run(w=wf):
            await w(x=3).collect()

        timbal = await measure_python(t_run, clear=_clear_traces)

        lgpy = None
        if HAS_LANGGRAPH_PY and not _args.timbal_only:
            graph = LG_FACTORIES[key]()

            async def lg_run(g=graph):
                await g.ainvoke({"x": 3})

            lgpy = await measure_python(lg_run)

        lgjs_s = lgjs["scenarios"][key]["bare"] if lgjs else None
        print_scenario(key, timbal, lgpy, lgjs_s)
        if lgpy and lgjs_s:
            ratios.append(lgpy["latency_us"]["p50"] / lgjs_s["latency_us"]["p50"])

    print()
    print(f"{DIM}{'─' * WIDTH}")
    if ratios:
        print(f"  Calibration summary: LG.py ÷ LG.js latency p50 spans {min(ratios):.2f}-{max(ratios):.2f}x across")
        print("  identical DAGs (>1 = V8 faster). Apply that factor when reading any Timbal")
        print("  (Python) vs TypeScript-framework workflow table in benchmarks/.")
    print("  Graph compilation and Workflow construction excluded from timing.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
