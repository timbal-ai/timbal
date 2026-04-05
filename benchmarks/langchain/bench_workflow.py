#!/usr/bin/env python3
"""
Timbal Workflow vs LangGraph StateGraph — DAG benchmark.

Three scenarios with trivial handler functions (no LLM calls):
  1. Sequential:    A → B → C → D
  2. Fan-out/in:    A → [B, C, D] → E
  3. Diamond:       A → [B, C] → D

Metrics: latency, memory, burst, throughput.

Run:
    uv run python benchmarks/langchain/bench_workflow.py
    uv run python benchmarks/langchain/bench_workflow.py --quick
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
import statistics  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
_args, _ = parser.parse_known_args()

N_ITERS = 50 if _args.quick else 200
N_WARMUP = 5 if _args.quick else 20
N_BURST = 100 if _args.quick else 500
N_MEM = 50 if _args.quick else 200
THROUGHPUT_OPS = 200 if _args.quick else 1000
CONCURRENCY_LEVELS = [1, 10, 50, 200]
WIDTH = 76

# ── Display helpers ──────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
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
# TIMBAL — Workflow factories
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal_traces():
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


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH — StateGraph factories
# ═══════════════════════════════════════════════════════════════════════════════

from langgraph.graph import StateGraph, START, END  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

try:
    from unittest.mock import MagicMock
    from langchain_core.tracers.langchain import LangChainTracer

    def _make_ls_tracer() -> LangChainTracer:
        tracer = LangChainTracer()
        tracer._persist_run_single = MagicMock(return_value=None)
        return tracer

    HAS_LANGSMITH = True
except (ImportError, Exception):
    HAS_LANGSMITH = False


# -- Sequential state --

class SeqState(TypedDict):
    x: int
    a: int
    b: int
    c: int
    d: int


def _lg_seq_a(state: SeqState) -> dict:
    return {"a": state["x"] + 1}


def _lg_seq_b(state: SeqState) -> dict:
    return {"b": state["a"] * 2}


def _lg_seq_c(state: SeqState) -> dict:
    return {"c": state["b"] + 10}


def _lg_seq_d(state: SeqState) -> dict:
    return {"d": state["c"] - 3}


def _lg_sequential():
    g = StateGraph(SeqState)
    g.add_node("A", _lg_seq_a)
    g.add_node("B", _lg_seq_b)
    g.add_node("C", _lg_seq_c)
    g.add_node("D", _lg_seq_d)
    g.add_edge(START, "A")
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    g.add_edge("D", END)
    return g.compile()


# -- Fan-out state --

class FanState(TypedDict):
    x: int
    a: int
    bb: int
    cc: int
    dd: int
    e: int


def _lg_fan_a(state: FanState) -> dict:
    return {"a": state["x"] + 1}


def _lg_fan_b(state: FanState) -> dict:
    return {"bb": state["a"] * 2}


def _lg_fan_c(state: FanState) -> dict:
    return {"cc": state["a"] * 3}


def _lg_fan_d(state: FanState) -> dict:
    return {"dd": state["a"] * 4}


def _lg_fan_e(state: FanState) -> dict:
    return {"e": state["bb"] + state["cc"] + state["dd"]}


def _lg_fanout():
    g = StateGraph(FanState)
    g.add_node("A", _lg_fan_a)
    g.add_node("B", _lg_fan_b)
    g.add_node("C", _lg_fan_c)
    g.add_node("D", _lg_fan_d)
    g.add_node("E", _lg_fan_e)
    g.add_edge(START, "A")
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("A", "D")
    g.add_edge(["B", "C", "D"], "E")
    g.add_edge("E", END)
    return g.compile()


# -- Diamond state --

class DiamondState(TypedDict):
    x: int
    a: int
    b: int
    c: int
    d: int


def _lg_dia_a(state: DiamondState) -> dict:
    return {"a": state["x"] + 1}


def _lg_dia_b(state: DiamondState) -> dict:
    return {"b": state["a"] + 10}


def _lg_dia_c(state: DiamondState) -> dict:
    return {"c": state["a"] * 5}


def _lg_dia_d(state: DiamondState) -> dict:
    return {"d": state["b"] + state["c"]}


def _lg_diamond():
    g = StateGraph(DiamondState)
    g.add_node("A", _lg_dia_a)
    g.add_node("B", _lg_dia_b)
    g.add_node("C", _lg_dia_c)
    g.add_node("D", _lg_dia_d)
    g.add_edge(START, "A")
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge(["B", "C"], "D")
    g.add_edge("D", END)
    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement harnesses
# ═══════════════════════════════════════════════════════════════════════════════


async def _latency_async(factory, n, warmup) -> list[float]:
    for _ in range(warmup):
        await factory()
    _clear_timbal_traces()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)
    return samples


def _latency_sync(factory, n, warmup) -> list[float]:
    for _ in range(warmup):
        factory()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)
    _clear_timbal_traces()
    return samples


async def _memory_async(factory, n, warmup) -> tuple[float, float]:
    for _ in range(warmup):
        await factory()
    _clear_timbal_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await factory()
        _clear_timbal_traces()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


def _memory_sync(factory, n, warmup) -> tuple[float, float]:
    for _ in range(warmup):
        factory()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        factory()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


async def _burst_async(factory, n) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(10, n))])
    _clear_timbal_traces()
    gc.collect()
    samples: list[float] = []

    async def timed():
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    await asyncio.gather(*[timed() for _ in range(n)])
    _clear_timbal_traces()
    return sorted(samples)


async def _throughput_async(factory, total, conc) -> float:
    sem = asyncio.Semaphore(conc)

    async def bounded():
        async with sem:
            await factory()

    t0 = time.perf_counter()
    await asyncio.gather(*[bounded() for _ in range(total)])
    elapsed = time.perf_counter() - t0
    _clear_timbal_traces()
    return total / elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# Per-scenario benchmark
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_NAMES = {
    1: "Sequential  (A → B → C → D)",
    2: "Fan-out/in  (A → [B, C, D] → E)",
    3: "Diamond  (A → [B, C] → D)",
}


async def bench_scenario(scenario: int) -> None:
    section(f"Scenario {scenario}: {SCENARIO_NAMES[scenario]}")

    # Build workflows (excluded from timing — one-time cost)
    if scenario == 1:
        t_wf = _timbal_sequential()
        lg_wf = _lg_sequential()
    elif scenario == 2:
        t_wf = _timbal_fanout()
        lg_wf = _lg_fanout()
    else:
        t_wf = _timbal_diamond()
        lg_wf = _lg_diamond()

    input_val = 5

    lg_run = lambda: lg_wf.ainvoke({"x": input_val})  # noqa: E731
    ls_tracer = _make_ls_tracer() if HAS_LANGSMITH else None
    ls_run = (lambda: lg_wf.ainvoke({"x": input_val}, config={"callbacks": [ls_tracer]})) if HAS_LANGSMITH else None

    cols = ["Timbal", "LG (bare)"] + (["LG+Smith"] if HAS_LANGSMITH else [])
    hdr  = f"  {'':>12}" + "".join(f"  {c:>12}" for c in cols)
    sep  = f"  {'─' * 12}" + "  ────────────" * len(cols)

    # ── Latency ──────────────────────────────────────────────────────────────
    subsection(f"Latency  (×{N_ITERS})")

    t_samples  = await _latency_async(lambda: t_wf(x=input_val).collect(), N_ITERS, N_WARMUP)
    lg_samples = await _latency_async(lg_run, N_ITERS, N_WARMUP)
    ls_samples = await _latency_async(ls_run, N_ITERS, N_WARMUP) if HAS_LANGSMITH else None

    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        fn = statistics.mean if p is None else lambda s: pct(s, p)
        vals = [fn(t_samples), fn(lg_samples)] + ([fn(ls_samples)] if ls_samples else [])
        print(f"  {label:>12}" + "".join(f"  {fmt_us(v)}" for v in vals))

    # ── Memory ───────────────────────────────────────────────────────────────
    subsection(f"Memory  (×{N_MEM} runs)")

    t_peak,  t_per  = await _memory_async(lambda: t_wf(x=input_val).collect(), N_MEM, N_WARMUP)
    lg_peak, lg_per = await _memory_async(lg_run, N_MEM, N_WARMUP)
    ls_peak, ls_per = (await _memory_async(ls_run, N_MEM, N_WARMUP)) if HAS_LANGSMITH else (None, None)

    fw_w = 24
    print(f"  {'framework':<{fw_w}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * fw_w}  {'─' * 12}  {'─' * 12}")
    print(f"  {'Timbal':<{fw_w}}  {t_peak / 1024:>10.1f} KB  {t_per:>10.0f}  B")
    print(f"  {'LG (bare)':<{fw_w}}  {lg_peak / 1024:>10.1f} KB  {lg_per:>10.0f}  B")
    if ls_peak is not None:
        print(f"  {'LG + LangSmith':<{fw_w}}  {ls_peak / 1024:>10.1f} KB  {ls_per:>10.0f}  B")

    # ── Burst ────────────────────────────────────────────────────────────────
    subsection(f"Burst  ({N_BURST} concurrent)")

    t_burst  = await _burst_async(lambda: t_wf(x=input_val).collect(), N_BURST)
    lg_burst = await _burst_async(lg_run, N_BURST)
    ls_burst = (await _burst_async(ls_run, N_BURST)) if HAS_LANGSMITH else None

    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [pct(t_burst, p), pct(lg_burst, p)] + ([pct(ls_burst, p)] if ls_burst else [])
        print(f"  {label:>12}" + "".join(f"  {fmt_us(v)}" for v in vals))

    wall = f"\n  {DIM}wall: Timbal {max(t_burst)/1000:.1f} ms  |  LG bare {max(lg_burst)/1000:.1f} ms"
    if ls_burst:
        wall += f"  |  LG+Smith {max(ls_burst)/1000:.1f} ms"
    print(wall + RESET)

    # ── Throughput ───────────────────────────────────────────────────────────
    subsection(f"Throughput  ({THROUGHPUT_OPS} runs)")

    print(hdr)
    print(sep)

    for conc in CONCURRENCY_LEVELS:
        t_ops  = await _throughput_async(lambda: t_wf(x=input_val).collect(), THROUGHPUT_OPS, conc)
        lg_ops = await _throughput_async(lg_run, THROUGHPUT_OPS, conc)
        line = f"  {conc:>12}  {t_ops:>10.0f}/s  {lg_ops:>10.0f}/s"
        if HAS_LANGSMITH:
            ls_ops = await _throughput_async(ls_run, THROUGHPUT_OPS, conc)
            line += f"  {ls_ops:>10.0f}/s"
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal Workflow vs LangGraph StateGraph — DAG benchmark{RESET}")
    print(f"  {N_ITERS} iters · {N_BURST} burst · {N_MEM} mem runs · {THROUGHPUT_OPS} throughput ops")
    print(f"  Trivial handlers (no LLM). Pure DAG scheduling overhead.")
    print(f"  {DIM}Both use native async (ainvoke / .collect()). Single shared instance.{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    # Verify correctness first
    print(f"\n  {DIM}Verifying correctness...{RESET}")
    for name, t_factory, lg_factory, inp in [
        ("sequential", _timbal_sequential, _lg_sequential, 5),
        ("fanout", _timbal_fanout, _lg_fanout, 5),
        ("diamond", _timbal_diamond, _lg_diamond, 5),
    ]:
        t_wf = t_factory()
        lg_wf = lg_factory()
        t_result = await t_wf(x=inp).collect()
        lg_result = await lg_wf.ainvoke({"x": inp})
        # Get the last meaningful value from LangGraph state
        lg_final = lg_result.get("d") or lg_result.get("e")
        t_final = t_result.output
        print(f"  {name}: Timbal={t_final}  LangGraph={lg_final}  {'ok' if t_final == lg_final else 'MISMATCH!'}")

    for scenario in [1, 2, 3]:
        await bench_scenario(scenario)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  Both frameworks execute identical DAGs with trivial handlers.")
    print(f"  Both run natively async (ainvoke / .collect()) on a single shared instance.")
    print(f"  Graph compilation and Workflow construction excluded from timing.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
