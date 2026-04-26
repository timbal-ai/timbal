#!/usr/bin/env python3
"""
Timbal Workflow vs Pydantic Graph — DAG-ish benchmark.

Three scenarios with trivial handler functions (no LLM calls):
  1. Sequential:    A → B → C → D
  2. Fan-out/in:    A → [B, C, D] → E
  3. Diamond:       A → [B, C] → D

Important: Pydantic Graph is a typed state-machine runner, not a DAG scheduler.
The sequential scenario is structurally comparable. The fan-out scenarios use the
closest idiomatic Pydantic Graph shape: one graph node manually computes the branch
work, then the graph continues to the join node.

Columns:
  Timbal        — built-in InMemory tracing always on
  PG bare       — Pydantic Graph with auto_instrument=False
  PG+Logfire    — Pydantic Graph with auto_instrument=True and Logfire configured
                  with send_to_logfire=False (span creation, no network I/O)

Run:
    uv run python benchmarks/pydantic/bench_workflow.py
    uv run python benchmarks/pydantic/bench_workflow.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import logging
import os
import statistics
import time
import tracemalloc
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
warnings.filterwarnings("ignore")

import structlog  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
_args, _ = parser.parse_known_args()

N_ITERS = 50 if _args.quick else 200
N_WARMUP = 5 if _args.quick else 20
N_BURST = 50 if _args.quick else 200
N_MEM = 50 if _args.quick else 200
THROUGHPUT_OPS = 100 if _args.quick else 500
CONCURRENCY_LEVELS = [1, 10, 50, 200]
WIDTH = 88

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
YELLOW = "\033[33m"


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
    return sorted(samples)[min(int(len(samples) * p / 100), len(samples) - 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal() -> None:
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
# PYDANTIC GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

HAS_PG = False
HAS_LOGFIRE = False

try:
    from pydantic_graph import BaseNode, End, Graph, GraphRunContext  # noqa: E402

    @dataclasses.dataclass
    class _SeqState:
        x: int = 0
        a: int = 0
        b: int = 0
        c: int = 0

    @dataclasses.dataclass
    class _SeqA(BaseNode[_SeqState, None, int]):
        async def run(self, ctx: GraphRunContext[_SeqState, None]) -> _SeqB:
            ctx.state.a = ctx.state.x + 1
            return _SeqB()

    @dataclasses.dataclass
    class _SeqB(BaseNode[_SeqState, None, int]):
        async def run(self, ctx: GraphRunContext[_SeqState, None]) -> _SeqC:
            ctx.state.b = ctx.state.a * 2
            return _SeqC()

    @dataclasses.dataclass
    class _SeqC(BaseNode[_SeqState, None, int]):
        async def run(self, ctx: GraphRunContext[_SeqState, None]) -> _SeqD:
            ctx.state.c = ctx.state.b + 10
            return _SeqD()

    @dataclasses.dataclass
    class _SeqD(BaseNode[_SeqState, None, int]):
        async def run(self, ctx: GraphRunContext[_SeqState, None]) -> End[int]:
            return End(ctx.state.c - 3)

    @dataclasses.dataclass
    class _FanState:
        x: int = 0
        a: int = 0
        b: int = 0
        c: int = 0
        d: int = 0

    @dataclasses.dataclass
    class _FanA(BaseNode[_FanState, None, int]):
        async def run(self, ctx: GraphRunContext[_FanState, None]) -> _FanBranches:
            ctx.state.a = ctx.state.x + 1
            return _FanBranches()

    @dataclasses.dataclass
    class _FanBranches(BaseNode[_FanState, None, int]):
        async def run(self, ctx: GraphRunContext[_FanState, None]) -> _FanE:
            x = ctx.state.a
            ctx.state.b = x * 2
            ctx.state.c = x * 3
            ctx.state.d = x * 4
            return _FanE()

    @dataclasses.dataclass
    class _FanE(BaseNode[_FanState, None, int]):
        async def run(self, ctx: GraphRunContext[_FanState, None]) -> End[int]:
            return End(ctx.state.b + ctx.state.c + ctx.state.d)

    @dataclasses.dataclass
    class _DiamondState:
        x: int = 0
        a: int = 0
        b: int = 0
        c: int = 0

    @dataclasses.dataclass
    class _DiamondA(BaseNode[_DiamondState, None, int]):
        async def run(self, ctx: GraphRunContext[_DiamondState, None]) -> _DiamondBranches:
            ctx.state.a = ctx.state.x + 1
            return _DiamondBranches()

    @dataclasses.dataclass
    class _DiamondBranches(BaseNode[_DiamondState, None, int]):
        async def run(self, ctx: GraphRunContext[_DiamondState, None]) -> _DiamondD:
            x = ctx.state.a
            ctx.state.b = x + 10
            ctx.state.c = x * 5
            return _DiamondD()

    @dataclasses.dataclass
    class _DiamondD(BaseNode[_DiamondState, None, int]):
        async def run(self, ctx: GraphRunContext[_DiamondState, None]) -> End[int]:
            return End(ctx.state.b + ctx.state.c)

    def _pg_sequential(auto_instrument: bool):
        return Graph(nodes=[_SeqA, _SeqB, _SeqC, _SeqD], name="pg_sequential", auto_instrument=auto_instrument)

    def _pg_fanout(auto_instrument: bool):
        return Graph(nodes=[_FanA, _FanBranches, _FanE], name="pg_fanout", auto_instrument=auto_instrument)

    def _pg_diamond(auto_instrument: bool):
        return Graph(
            nodes=[_DiamondA, _DiamondBranches, _DiamondD],
            name="pg_diamond",
            auto_instrument=auto_instrument,
        )

    HAS_PG = True
except ImportError as e:
    print(f"Pydantic Graph not found: {e}")


def _try_init_logfire() -> bool:
    global HAS_LOGFIRE
    try:
        import logfire as _logfire

        _logfire.configure(send_to_logfire=False, console=False)
        HAS_LOGFIRE = True
    except (ImportError, Exception):
        pass
    return HAS_LOGFIRE


# ═══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

Factory = Callable[[], Awaitable[Any]]


async def _latency(factory: Factory, n: int, warmup: int) -> list[float]:
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


async def _memory(factory: Factory, n: int, warmup: int) -> tuple[float, float]:
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


async def _burst(factory: Factory, n: int) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_timbal()
    gc.collect()
    samples: list[float] = []

    async def timed() -> None:
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    await asyncio.gather(*[timed() for _ in range(n)])
    _clear_timbal()
    return samples


async def _throughput(factory: Factory, total: int, conc: int) -> float:
    sem = asyncio.Semaphore(conc)

    async def bounded() -> None:
        async with sem:
            await factory()

    gc.collect()
    t0 = time.perf_counter()
    await asyncio.gather(*[bounded() for _ in range(total)])
    elapsed = time.perf_counter() - t0
    _clear_timbal()
    return total / elapsed


@dataclasses.dataclass
class RunData:
    latency: list[float]
    mem_peak: float
    mem_per: float
    burst: list[float]
    throughput: list[float]


async def _collect(label: str, factory: Factory) -> RunData:
    print(f"    {label} latency...", end=" ", flush=True)
    lat = await _latency(factory, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(lat, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory(factory, N_MEM, N_WARMUP)
    print(f"{mem_peak / 1024:.1f} KB peak", flush=True)

    print(f"    {label} burst ({N_BURST} concurrent)...", end=" ", flush=True)
    burst = await _burst(factory, N_BURST)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    tp = []
    for conc in CONCURRENCY_LEVELS:
        print(f"    {label} throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput(factory, THROUGHPUT_OPS, conc)
        tp.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=lat, mem_peak=mem_peak, mem_per=mem_per, burst=burst, throughput=tp)


SCENARIOS = {
    "sequential": "A → B → C → D",
    "fanout": "A → [B, C, D] → E",
    "diamond": "A → [B, C] → D",
}


def _make_timbal_factory(name: str) -> Factory:
    wf = {
        "sequential": _timbal_sequential,
        "fanout": _timbal_fanout,
        "diamond": _timbal_diamond,
    }[name]()
    return lambda: wf(x=5).collect()


def _make_pg_factory(name: str, auto_instrument: bool) -> Factory:
    graph = {
        "sequential": _pg_sequential,
        "fanout": _pg_fanout,
        "diamond": _pg_diamond,
    }[name](auto_instrument)
    state_factory = {
        "sequential": lambda: _SeqState(x=5),
        "fanout": lambda: _FanState(x=5),
        "diamond": lambda: _DiamondState(x=5),
    }[name]
    start_factory = {
        "sequential": _SeqA,
        "fanout": _FanA,
        "diamond": _DiamondA,
    }[name]
    return lambda: graph.run(start_factory(), state=state_factory())


def _print_scenario(name: str, rows: list[tuple[str, RunData]]) -> None:
    section(f"{name}: {SCENARIOS[name]}")
    col_w = 14
    hdr = f"  {'':>12}" + "".join(f"  {label:>{col_w}}" for label, _ in rows)
    sep = f"  {'─' * 12}" + f"  {'─' * col_w}" * len(rows)

    subsection(f"Latency  (×{N_ITERS} sequential runs)")
    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = []
        for _, d in rows:
            v = statistics.mean(d.latency) if p is None else pct(d.latency, p)
            vals.append(fmt_us(v))
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    subsection(f"Memory  (×{N_MEM} runs)")
    print(f"  {'framework':<20}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * 20}  {'─' * 12}  {'─' * 12}")
    for label, d in rows:
        print(f"  {label:<20}  {d.mem_peak / 1024:>10.1f} KB  {d.mem_per:>10.0f}  B")

    subsection(f"Burst  ({N_BURST} concurrent)")
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d.burst, p)) for _, d in rows]
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))
    wall = " | ".join(f"{label}: {max(d.burst) / 1e3:.1f} ms" for label, d in rows)
    print(f"\n  {DIM}wall: {wall}{RESET}")

    subsection(f"Throughput  ({THROUGHPUT_OPS} loops)")
    print(hdr)
    print(sep)
    for i, conc in enumerate(CONCURRENCY_LEVELS):
        vals = [f"{d.throughput[i]:>10.0f}/s" for _, d in rows]
        print(f"  {conc:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal Workflow vs Pydantic Graph — workflow benchmark{RESET}")
    print(f"  {N_ITERS} iters · {N_BURST} burst · {N_MEM} mem · {THROUGHPUT_OPS} throughput ops")
    if HAS_PG:
        print("  Pydantic Graph found — will measure bare + Logfire")
        print(f"  {DIM}Fan-out scenarios use manual branch work inside one Pydantic Graph node.{RESET}")
    else:
        print(f"  {YELLOW}Pydantic Graph not found. Install: uv pip install pydantic-ai logfire{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    print(f"\n  {DIM}Spot-checking correctness (sequential)...{RESET}", flush=True)
    t_res = await _make_timbal_factory("sequential")()
    t_ok = t_res.output == 19
    msg = f"  Timbal: {'✓' if t_ok else f'✗ (got {t_res.output!r})'}"
    if HAS_PG:
        p_res = await _make_pg_factory("sequential", auto_instrument=False)()
        p_ok = p_res.output == 19
        msg += f"  |  PG: {'✓' if p_ok else f'✗ (got {p_res.output!r})'}"
    print(msg, flush=True)

    all_rows: dict[str, list[tuple[str, RunData]]] = {}
    for scenario in SCENARIOS:
        print(f"\n  {DIM}[Phase 1 · {scenario}]{RESET}", flush=True)
        rows = [("Timbal", await _collect("Timbal", _make_timbal_factory(scenario)))]
        if HAS_PG:
            rows.append(("PG bare", await _collect("PG bare", _make_pg_factory(scenario, auto_instrument=False))))
        all_rows[scenario] = rows

    if HAS_PG:
        print(f"\n  {DIM}Activating Logfire (send_to_logfire=False)...{RESET}", flush=True)
        if _try_init_logfire():
            for scenario in SCENARIOS:
                print(f"\n  {DIM}[Phase 2 · {scenario}]{RESET}", flush=True)
                all_rows[scenario].append(
                    ("PG+Logfire", await _collect("PG+Logfire", _make_pg_factory(scenario, auto_instrument=True)))
                )
        else:
            print(f"  {YELLOW}Logfire not found — skipping obs column.{RESET}", flush=True)

    for scenario, rows in all_rows.items():
        _print_scenario(scenario, rows)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print("  Timbal: Workflow DAG with one span per step; fan-out branches are scheduled by the framework.")
    print("  PG bare: Pydantic Graph with auto_instrument=False.")
    print("  PG+Logfire: Pydantic Graph with built-in graph/node spans; no network export.")
    print("  Note: Pydantic Graph has no native DAG fan-out; branch work is manual inside one node.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
