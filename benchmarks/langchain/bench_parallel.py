#!/usr/bin/env python3
"""
Timbal vs LangGraph — Wide Parallel Fan-out benchmark.

Tests scheduling overhead and true parallelism across N parallel branches.

  Topology:  root → [N async branches] → sink

  Scenario A (trivial):    each branch returns immediately (pure overhead)
  Scenario B (async work): each branch does asyncio.sleep(0.001) — 1 ms

Branch widths tested: N = 4, 8, 16, 32, 64

Scenario B is the definitive parallelism test:
  - Truly parallel → p50 ≈ 1 ms flat regardless of N
  - Sequential     → p50 ≈ N × 1 ms

Run:
    uv run python benchmarks/langchain/bench_parallel.py
    uv run python benchmarks/langchain/bench_parallel.py --quick
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import operator
import os
import statistics
import time
import tracemalloc
import warnings
from typing import Annotated, Any

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
warnings.filterwarnings("ignore")

import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
_args, _ = parser.parse_known_args()

N_ITERS = 50 if _args.quick else 200
N_WARMUP = 5 if _args.quick else 20
N_BURST = 50 if _args.quick else 200
WIDTHS = [4, 8, 16] if _args.quick else [4, 8, 16, 32, 64]
DISPLAY_W = 88

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
GREEN = "\033[32m"


def section(title: str) -> None:
    print()
    print(f"{BOLD}{CYAN}{'─' * DISPLAY_W}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * DISPLAY_W}{RESET}")


def subsection(title: str) -> None:
    print(f"\n  {BOLD}{title}{RESET}")


def fmt_us(us: float) -> str:
    if us >= 10_000:
        return f"{us / 1_000:>8.1f} ms"
    if us >= 1_000:
        return f"{us / 1_000:>8.2f} ms"
    return f"{us:>8.1f} µs"


def pct(s: list[float], p: float) -> float:
    return sorted(s)[min(int(len(s) * p / 100), len(s) - 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal() -> None:
    InMemoryTracingProvider._storage.clear()


def _make_root_getter():
    """Async getter for root's output."""
    async def get_root() -> Any:
        return get_run_context().step_span("root").output
    return get_root


def _make_all_branches_getter(n: int):
    """Async getter that collects all N branch outputs as a list."""
    async def get_all_branches() -> list:
        ctx = get_run_context()
        return [ctx.step_span(f"branch_{i}").output for i in range(n)]
    return get_all_branches


def _make_branch_fn(i: int, async_work: bool):
    """Factory: captures i via closure — no idx param in the function signature."""
    if async_work:
        async def branch(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * (i + 2)
    else:
        async def branch(x: int) -> int:
            return x * (i + 2)
    branch.__name__ = f"branch_{i}"
    return branch


def _timbal_wide(n: int, async_work: bool = False) -> Workflow:
    """root → [N parallel async branches] → sink"""
    wf = Workflow(name=f"wide_{n}")

    # Root
    async def root(x: int) -> int:
        return x + 1

    wf.step(root)

    # N parallel branches — all async so they yield to the event loop
    for i in range(n):
        branch = _make_branch_fn(i, async_work)
        # depends_on explicit — AST analyzer won't resolve closure variable step names
        wf.step(branch, depends_on=["root"], x=_make_root_getter())

    # Sink receives all branch outputs as a single list — avoids **kwargs Pydantic edge case
    async def sink(results: list) -> int:
        return sum(results)

    wf.step(
        sink,
        depends_on=[f"branch_{i}" for i in range(n)],
        results=_make_all_branches_getter(n),
    )
    return wf


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH
# ═══════════════════════════════════════════════════════════════════════════════

from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402


try:
    from unittest.mock import MagicMock
    from langchain_core.tracers.langchain import LangChainTracer

    def _make_ls_tracer() -> LangChainTracer:
        t = LangChainTracer()
        t._persist_run_single = MagicMock(return_value=None)
        return t

    HAS_LANGSMITH = True
except Exception:
    HAS_LANGSMITH = False


# Module-level state schema (Annotated list reducer collects parallel branch outputs)
class _WideState(TypedDict):
    x: int
    root_out: int
    results: Annotated[list[int], operator.add]
    final: int


def _lg_wide(n: int, async_work: bool = False):
    """root → [N parallel async branches] → sink"""
    g = StateGraph(_WideState)

    async def lg_root(state: _WideState) -> dict:
        return {"root_out": state["x"] + 1}

    g.add_node("root", lg_root)

    for i in range(n):
        if async_work:
            async def lg_branch(state: _WideState, _i: int = i) -> dict:
                await asyncio.sleep(0.001)
                return {"results": [state["root_out"] * (_i + 2)]}
        else:
            async def lg_branch(state: _WideState, _i: int = i) -> dict:
                return {"results": [state["root_out"] * (_i + 2)]}
        lg_branch.__name__ = f"lg_branch_{i}"
        g.add_node(f"branch_{i}", lg_branch)

    async def lg_sink(state: _WideState) -> dict:
        return {"final": sum(state["results"])}

    g.add_node("sink", lg_sink)

    g.add_edge(START, "root")
    for i in range(n):
        g.add_edge("root", f"branch_{i}")
        g.add_edge(f"branch_{i}", "sink")
    g.add_edge("sink", END)

    return g.compile()


def _lg_init() -> dict:
    return {"x": 5, "root_out": 0, "results": [], "final": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement harnesses
# ═══════════════════════════════════════════════════════════════════════════════


async def _latency(factory, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        await factory()
    _clear_timbal()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)
    _clear_timbal()
    return samples


async def _memory(factory, n: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        await factory()
    _clear_timbal()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await factory()
        _clear_timbal()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


async def _burst(factory, n: int) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(10, n))])
    _clear_timbal()
    gc.collect()
    samples: list[float] = []

    async def _timed():
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    await asyncio.gather(*[_timed() for _ in range(n)])
    _clear_timbal()
    return sorted(samples)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-scenario benchmark
# ═══════════════════════════════════════════════════════════════════════════════


async def run_scenario(label: str, async_work: bool) -> None:
    section(label)

    cols = ["Timbal", "LG (bare)"] + (["LG+Smith"] if HAS_LANGSMITH else [])
    col_w = 14

    # Build all workflow instances upfront (construction excluded from timing)
    instances: dict[int, dict] = {}
    for n in WIDTHS:
        t_wf = _timbal_wide(n, async_work)
        lg_wf = _lg_wide(n, async_work)
        ls_tracer = _make_ls_tracer() if HAS_LANGSMITH else None
        instances[n] = {
            "t_run": lambda _w=t_wf: _w(x=5).collect(),
            "lg_run": lambda _w=lg_wf: _w.ainvoke(_lg_init()),
            "ls_run": (lambda _w=lg_wf, _tr=ls_tracer: _w.ainvoke(_lg_init(), config={"callbacks": [_tr]}))
                if HAS_LANGSMITH else None,
        }

    # Collect latency samples
    data: dict[int, dict] = {}
    for n in WIDTHS:
        inst = instances[n]
        t_s = await _latency(inst["t_run"], N_ITERS, N_WARMUP)
        lg_s = await _latency(inst["lg_run"], N_ITERS, N_WARMUP)
        ls_s = await _latency(inst["ls_run"], N_ITERS, N_WARMUP) if inst["ls_run"] else None
        t_b = await _burst(inst["t_run"], N_BURST)
        lg_b = await _burst(inst["lg_run"], N_BURST)
        ls_b = await _burst(inst["ls_run"], N_BURST) if inst["ls_run"] else None
        data[n] = {"t_s": t_s, "lg_s": lg_s, "ls_s": ls_s, "t_b": t_b, "lg_b": lg_b, "ls_b": ls_b}

    # ── Latency table ──────────────────────────────────────────────────────────
    hdr = f"  {'N':>6}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = f"  {'─'*6}" + f"  {'─'*col_w}" * len(cols)

    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Latency {stat_label}  (×{N_ITERS} sequential runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_s"], stat_p), pct(r["lg_s"], stat_p)]
            if r["ls_s"]:
                vals.append(pct(r["ls_s"], stat_p))
            print(f"  {n:>6}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    # ── Scaling overhead ───────────────────────────────────────────────────────
    subsection("Overhead per extra branch  (slope of p50 latency vs N)")
    print(f"  {DIM}Estimated by linear regression over the N values above{RESET}")
    if len(WIDTHS) >= 2:
        for label_s, key in [("Timbal", "t_s"), ("LG (bare)", "lg_s")] + ([("LG+Smith", "ls_s")] if HAS_LANGSMITH else []):
            xs = WIDTHS
            ys = [pct(data[n][key], 50) for n in WIDTHS if data[n][key]]
            if len(ys) < 2:
                continue
            x_mean = statistics.mean(xs)
            y_mean = statistics.mean(ys)
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum((x - x_mean) ** 2 for x in xs)
            print(f"  {label_s:<14}  {slope:>+.1f} µs / branch")

    # ── Burst table — N as rows, percentile bands as sub-headers ──────────────
    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Burst {stat_label}  ({N_BURST} concurrent runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_b"], stat_p), pct(r["lg_b"], stat_p)]
            if r["ls_b"]:
                vals.append(pct(r["ls_b"], stat_p))
            print(f"  {n:>6}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    if async_work:
        print(f"\n  {DIM}Theoretical serial: N × 1000 µs  |  Theoretical parallel: ~1000 µs flat{RESET}")
        print(f"  {DIM}Both frameworks use asyncio tasks — expect near-flat latency across widths{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")
    print(f"{BOLD}  Timbal vs LangGraph — Wide Parallel Fan-out{RESET}")
    print(f"  Topology: root → [N async branches] → sink")
    print(f"  Widths: {WIDTHS}  |  {N_ITERS} iters  |  {N_BURST} burst")
    if HAS_LANGSMITH:
        print(f"  {DIM}LangSmith: mock tracer (no network){RESET}")
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")

    # ── Correctness check ─────────────────────────────────────────────────────
    print(f"\n  {DIM}Verifying correctness (N=4, trivial)...{RESET}")
    t_wf = _timbal_wide(4, async_work=False)
    lg_wf = _lg_wide(4, async_work=False)
    t_result = await t_wf(x=5).collect()
    lg_result = await lg_wf.ainvoke(_lg_init())
    # root: 5+1=6, branches i=0..3: 6*(2,3,4,5)=12,18,24,30, sink sum=84
    t_final = t_result.output
    lg_final = lg_result["final"]
    match = t_final == lg_final
    print(f"  N=4: Timbal={t_final}  LangGraph={lg_final}  {'✓ match' if match else '✗ MISMATCH'}")
    if not match:
        print(f"  WARNING: outputs differ — benchmark may not be comparing equivalent work")

    print(f"\n  {DIM}Verifying async work (N=4, async_work=True)...{RESET}")
    t_wf_a = _timbal_wide(4, async_work=True)
    lg_wf_a = _lg_wide(4, async_work=True)
    t_result_a = await t_wf_a(x=5).collect()
    lg_result_a = await lg_wf_a.ainvoke(_lg_init())
    t_final_a = t_result_a.output
    lg_final_a = lg_result_a["final"]
    match_a = t_final_a == lg_final_a
    print(f"  N=4: Timbal={t_final_a}  LangGraph={lg_final_a}  {'✓ match' if match_a else '✗ MISMATCH'}")

    await run_scenario(
        "Scenario A — Trivial branches (async, no sleep)  ← pure scheduling overhead",
        async_work=False,
    )
    await run_scenario(
        "Scenario B — 1 ms async sleep per branch  ← parallelism verification",
        async_work=True,
    )

    print()
    print(f"{DIM}{'─' * DISPLAY_W}")
    print("  Scenario A: measures pure per-branch scheduling cost (task creation, span")
    print("  lifecycle, param resolution). Both frameworks pay this on every parallel step.")
    print()
    print("  Scenario B: verifies true concurrency. asyncio.sleep() yields to the event")
    print("  loop so all N branches run concurrently. Latency should be ~1 ms flat.")
    print("  If latency scales with N, the framework is serialising parallel branches.")
    print()
    print("  Timbal uses one asyncio.Task per step. Overhead scales linearly with N.")
    print("  LangGraph uses a Pregel superstep model. Different per-branch cost structure.")
    print(f"{'─' * DISPLAY_W}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
