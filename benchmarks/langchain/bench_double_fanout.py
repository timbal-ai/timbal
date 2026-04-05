#!/usr/bin/env python3
"""
Timbal vs LangGraph — Double Fan-out benchmark.

Topology:  root → [N branches phase-1] → aggregator → [N branches phase-2] → sink

This topology exercises:
  1. Two full fan-out / fan-in scheduling cycles per invocation
  2. Wide parallelism at N = 16, 32, 64, 128
  3. True concurrency with 1 ms async work per branch

Total async steps per run:  2N + 3  (root, N×p1, aggregator, N×p2, sink)
At N=128: 259 async tasks per invocation.

Run:
    uv run python benchmarks/langchain/bench_double_fanout.py
    uv run python benchmarks/langchain/bench_double_fanout.py --quick
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
N_BURST = 30 if _args.quick else 100
WIDTHS = [16, 32, 64] if _args.quick else [16, 32, 64, 128]
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
    async def get_root() -> Any:
        return get_run_context().step_span("root").output
    return get_root


def _make_phase1_getter(n: int):
    async def get_phase1() -> list:
        ctx = get_run_context()
        return [ctx.step_span(f"p1_{i}").output for i in range(n)]
    return get_phase1


def _make_aggregator_getter():
    async def get_agg() -> Any:
        return get_run_context().step_span("aggregator").output
    return get_agg


def _make_phase2_getter(n: int):
    async def get_phase2() -> list:
        ctx = get_run_context()
        return [ctx.step_span(f"p2_{i}").output for i in range(n)]
    return get_phase2


def _make_p1_branch(i: int, async_work: bool):
    if async_work:
        async def branch(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * (i + 2)
    else:
        async def branch(x: int) -> int:
            return x * (i + 2)
    branch.__name__ = f"p1_{i}"
    return branch


def _make_p2_branch(i: int, async_work: bool):
    if async_work:
        async def branch(y: int) -> int:
            await asyncio.sleep(0.001)
            return y * (i + 2)
    else:
        async def branch(y: int) -> int:
            return y * (i + 2)
    branch.__name__ = f"p2_{i}"
    return branch


def _timbal_double(n: int, async_work: bool = False, tracing_provider=InMemoryTracingProvider) -> Workflow:
    """root → [N×p1] → aggregator → [N×p2] → sink"""
    wf = Workflow(name=f"double_{n}", tracing_provider=tracing_provider)

    async def root(x: int) -> int:
        return x + 1

    wf.step(root)

    # Phase 1 — fan out from root
    for i in range(n):
        wf.step(_make_p1_branch(i, async_work), depends_on=["root"], x=_make_root_getter())

    # Aggregator — fan in, sum all phase-1 outputs
    async def aggregator(results: list) -> int:
        return sum(results)

    wf.step(
        aggregator,
        depends_on=[f"p1_{i}" for i in range(n)],
        results=_make_phase1_getter(n),
    )

    # Phase 2 — fan out from aggregator
    for i in range(n):
        wf.step(_make_p2_branch(i, async_work), depends_on=["aggregator"], y=_make_aggregator_getter())

    # Sink — fan in, sum all phase-2 outputs
    async def sink(results: list) -> int:
        return sum(results)

    wf.step(
        sink,
        depends_on=[f"p2_{i}" for i in range(n)],
        results=_make_phase2_getter(n),
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


class _DoubleState(TypedDict):
    x: int
    root_out: int
    phase1_results: Annotated[list[int], operator.add]
    agg_out: int
    phase2_results: Annotated[list[int], operator.add]
    final: int


def _lg_double(n: int, async_work: bool = False):
    """root → [N×p1] → aggregator → [N×p2] → sink"""
    g = StateGraph(_DoubleState)

    async def lg_root(state: _DoubleState) -> dict:
        return {"root_out": state["x"] + 1}

    g.add_node("root", lg_root)

    for i in range(n):
        if async_work:
            async def lg_p1(state: _DoubleState, _i: int = i) -> dict:
                await asyncio.sleep(0.001)
                return {"phase1_results": [state["root_out"] * (_i + 2)]}
        else:
            async def lg_p1(state: _DoubleState, _i: int = i) -> dict:
                return {"phase1_results": [state["root_out"] * (_i + 2)]}
        lg_p1.__name__ = f"lg_p1_{i}"
        g.add_node(f"p1_{i}", lg_p1)

    async def lg_aggregator(state: _DoubleState) -> dict:
        return {"agg_out": sum(state["phase1_results"])}

    g.add_node("aggregator", lg_aggregator)

    for i in range(n):
        if async_work:
            async def lg_p2(state: _DoubleState, _i: int = i) -> dict:
                await asyncio.sleep(0.001)
                return {"phase2_results": [state["agg_out"] * (_i + 2)]}
        else:
            async def lg_p2(state: _DoubleState, _i: int = i) -> dict:
                return {"phase2_results": [state["agg_out"] * (_i + 2)]}
        lg_p2.__name__ = f"lg_p2_{i}"
        g.add_node(f"p2_{i}", lg_p2)

    async def lg_sink(state: _DoubleState) -> dict:
        return {"final": sum(state["phase2_results"])}

    g.add_node("sink", lg_sink)

    g.add_edge(START, "root")
    for i in range(n):
        g.add_edge("root", f"p1_{i}")
        g.add_edge(f"p1_{i}", "aggregator")
    for i in range(n):
        g.add_edge("aggregator", f"p2_{i}")
        g.add_edge(f"p2_{i}", "sink")
    g.add_edge("sink", END)

    return g.compile()


def _lg_init() -> dict:
    return {"x": 5, "root_out": 0, "phase1_results": [], "agg_out": 0, "phase2_results": [], "final": 0}


def _expected(n: int) -> int:
    """Compute expected final value for x=5, N branches per phase."""
    root_out = 5 + 1  # 6
    phase1_sum = sum(root_out * (i + 2) for i in range(n))
    phase2_sum = sum(phase1_sum * (i + 2) for i in range(n))
    return phase2_sum


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


async def _seq_peak_memory(factory, n: int, warmup: int) -> int:
    """Peak memory across N sequential runs, cleared between each — measures per-run cost."""
    for _ in range(warmup):
        await factory()
    _clear_timbal()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await factory()
        _clear_timbal()
        gc.collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


async def _burst_peak_memory(factory, n: int) -> int:
    """Peak memory while N runs execute concurrently — no clear between them.
    This is where InMemoryTracingProvider accumulates all traces simultaneously."""
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_timbal()
    gc.collect()
    tracemalloc.start()
    await asyncio.gather(*[factory() for _ in range(n)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _clear_timbal()
    return peak


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

    instances: dict[int, dict] = {}
    for n in WIDTHS:
        t_wf = _timbal_double(n, async_work)
        lg_wf = _lg_double(n, async_work)
        ls_tracer = _make_ls_tracer() if HAS_LANGSMITH else None
        instances[n] = {
            "t_run": lambda _w=t_wf: _w(x=5).collect(),
            "lg_run": lambda _w=lg_wf: _w.ainvoke(_lg_init()),
            "ls_run": (lambda _w=lg_wf, _tr=ls_tracer: _w.ainvoke(_lg_init(), config={"callbacks": [_tr]}))
                if HAS_LANGSMITH else None,
        }

    data: dict[int, dict] = {}
    for n in WIDTHS:
        inst = instances[n]
        print(f"  {DIM}Benchmarking N={n} ({2*n+3} steps)...{RESET}", flush=True)
        t_s = await _latency(inst["t_run"], N_ITERS, N_WARMUP)
        lg_s = await _latency(inst["lg_run"], N_ITERS, N_WARMUP)
        ls_s = await _latency(inst["ls_run"], N_ITERS, N_WARMUP) if inst["ls_run"] else None
        t_b = await _burst(inst["t_run"], N_BURST)
        lg_b = await _burst(inst["lg_run"], N_BURST)
        ls_b = await _burst(inst["ls_run"], N_BURST) if inst["ls_run"] else None
        data[n] = {"t_s": t_s, "lg_s": lg_s, "ls_s": ls_s, "t_b": t_b, "lg_b": lg_b, "ls_b": ls_b}

    hdr = f"  {'N':>6}  {'steps':>7}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = f"  {'─'*6}  {'─'*7}" + f"  {'─'*col_w}" * len(cols)

    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Latency {stat_label}  (×{N_ITERS} sequential runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_s"], stat_p), pct(r["lg_s"], stat_p)]
            if r["ls_s"]:
                vals.append(pct(r["ls_s"], stat_p))
            print(f"  {n:>6}  {2*n+3:>7}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    subsection("Overhead per extra branch-pair  (slope of p50 vs N, 2 branches per unit)")
    print(f"  {DIM}Each N increment adds 2 async branches (one per phase){RESET}")
    if len(WIDTHS) >= 2:
        for label_s, key in [("Timbal", "t_s"), ("LG (bare)", "lg_s")] + ([("LG+Smith", "ls_s")] if HAS_LANGSMITH else []):
            xs = WIDTHS
            ys = [pct(data[n][key], 50) for n in WIDTHS if data[n][key]]
            if len(ys) < 2:
                continue
            x_mean = statistics.mean(xs)
            y_mean = statistics.mean(ys)
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum((x - x_mean) ** 2 for x in xs)
            print(f"  {label_s:<14}  {slope:>+.1f} µs / branch-pair  ({slope/2:>+.1f} µs per branch)")

    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Burst {stat_label}  ({N_BURST} concurrent runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_b"], stat_p), pct(r["lg_b"], stat_p)]
            if r["ls_b"]:
                vals.append(pct(r["ls_b"], stat_p))
            print(f"  {n:>6}  {2*n+3:>7}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    if async_work:
        print(f"\n  {DIM}Each phase: theoretical serial = N×1000 µs, parallel ≈ 1000 µs flat{RESET}")
        print(f"  {DIM}Two phases: expect ~2000 µs floor + scheduling overhead{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Memory section
# ═══════════════════════════════════════════════════════════════════════════════


def fmt_kb(b: float) -> str:
    kb = b / 1024
    if kb >= 1024:
        return f"{kb / 1024:>8.1f} MB"
    return f"{kb:>8.1f} KB"


async def run_memory() -> None:
    section("Memory — InMemoryTracingProvider vs tracing_provider=None vs LangGraph")

    cols = ["T (InMemory)", "T (no trace)", "LG (bare)"]
    col_w = 14

    N_MEM_SEQ = 20 if _args.quick else 100
    N_MEM_BURST = N_BURST

    print(f"  {DIM}Sequential: {N_MEM_SEQ} runs, cleared between each  |  Burst: {N_MEM_BURST} concurrent{RESET}")

    hdr = f"  {'N':>6}  {'steps':>7}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = f"  {'─'*6}  {'─'*7}" + f"  {'─'*col_w}" * len(cols)

    for sub, n_runs, harness_label in [
        ("Sequential peak  (per-run, cleared between)", N_MEM_SEQ, "seq"),
        (f"Burst peak  ({N_MEM_BURST} concurrent, no GC between)", N_MEM_BURST, "burst"),
    ]:
        subsection(sub)
        print(hdr)
        print(sep)
        for n in WIDTHS:
            t_inmem_wf  = _timbal_double(n, async_work=False, tracing_provider=InMemoryTracingProvider)
            t_none_wf   = _timbal_double(n, async_work=False, tracing_provider=None)
            lg_wf       = _lg_double(n, async_work=False)

            t_inmem_run = lambda _w=t_inmem_wf: _w(x=5).collect()
            t_none_run  = lambda _w=t_none_wf:  _w(x=5).collect()
            lg_run      = lambda _w=lg_wf:       _w.ainvoke(_lg_init())

            if harness_label == "seq":
                t_im  = await _seq_peak_memory(t_inmem_run, n_runs, warmup=3)
                t_no  = await _seq_peak_memory(t_none_run,  n_runs, warmup=3)
                lg_m  = await _seq_peak_memory(lg_run,      n_runs, warmup=3)
            else:
                t_im  = await _burst_peak_memory(t_inmem_run, n_runs)
                t_no  = await _burst_peak_memory(t_none_run,  n_runs)
                lg_m  = await _burst_peak_memory(lg_run,      n_runs)

            print(f"  {n:>6}  {2*n+3:>7}" + "".join(f"  {fmt_kb(v):>{col_w}}" for v in [t_im, t_no, lg_m]))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")
    print(f"{BOLD}  Timbal vs LangGraph — Double Fan-out{RESET}")
    print(f"  Topology: root → [N×p1] → aggregator → [N×p2] → sink")
    print(f"  Widths: {WIDTHS}  |  {N_ITERS} iters  |  {N_BURST} burst")
    if HAS_LANGSMITH:
        print(f"  {DIM}LangSmith: mock tracer (no network){RESET}")
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")

    # ── Correctness check ─────────────────────────────────────────────────────
    print(f"\n  {DIM}Verifying correctness (N=4, trivial)...{RESET}")
    expected = _expected(4)
    t_wf = _timbal_double(4, async_work=False)
    lg_wf = _lg_double(4, async_work=False)
    t_result = await t_wf(x=5).collect()
    lg_result = await lg_wf.ainvoke(_lg_init())
    t_final = t_result.output
    lg_final = lg_result["final"]
    t_ok = t_final == expected
    lg_ok = lg_final == expected
    print(f"  N=4: expected={expected}  Timbal={t_final} {'✓' if t_ok else '✗'}  LangGraph={lg_final} {'✓' if lg_ok else '✗'}")
    if not (t_ok and lg_ok):
        print(f"  WARNING: outputs differ from expected — benchmark may not be comparing equivalent work")

    print(f"\n  {DIM}Verifying async work (N=4, async_work=True)...{RESET}")
    t_wf_a = _timbal_double(4, async_work=True)
    lg_wf_a = _lg_double(4, async_work=True)
    t_result_a = await t_wf_a(x=5).collect()
    lg_result_a = await lg_wf_a.ainvoke(_lg_init())
    t_final_a = t_result_a.output
    lg_final_a = lg_result_a["final"]
    print(f"  N=4: expected={expected}  Timbal={t_final_a} {'✓' if t_final_a == expected else '✗'}  LangGraph={lg_final_a} {'✓' if lg_final_a == expected else '✗'}")

    await run_scenario(
        "Scenario A — Trivial branches (async, no sleep)  ← pure scheduling overhead",
        async_work=False,
    )
    await run_scenario(
        "Scenario B — 1 ms async sleep per branch  ← parallelism verification",
        async_work=True,
    )

    await run_memory()

    print()
    print(f"{DIM}{'─' * DISPLAY_W}")
    print(f"  Topology: root → [N parallel branches] → aggregator → [N parallel branches] → sink")
    print(f"  Total steps: 2N + 3  |  Two full fan-out / fan-in cycles per invocation")
    print()
    print(f"  Scenario A: measures pure scheduling cost at scale. Overhead accumulates")
    print(f"  across both phases. Per-branch slope × 2 ≈ total overhead per extra width unit.")
    print()
    print(f"  Scenario B: parallelism floor is ~2 ms (two sequential 1 ms phases).")
    print(f"  Latency growing beyond 2 ms means the framework is serialising branches.")
    print(f"  Burst tests reveal contention under concurrent workflow invocations.")
    print(f"{'─' * DISPLAY_W}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
