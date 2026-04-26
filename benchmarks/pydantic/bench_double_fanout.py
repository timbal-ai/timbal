#!/usr/bin/env python3
"""
Timbal Workflow vs Pydantic Graph — double fan-out benchmark.

Target topology: root → [N×p1] → aggregator → [N×p2] → sink

Pydantic Graph has no first-class DAG fan-out, so its closest equivalent is:
root → phase1 node with manual asyncio.gather → aggregator → phase2 gather → sink.

The Pydantic Graph column is labelled "PG manual" for that reason. It is useful for
understanding the closest Pydantic Graph implementation, but it is not a per-branch
scheduler comparison.

Run:
    uv run python benchmarks/pydantic/bench_double_fanout.py
    uv run python benchmarks/pydantic/bench_double_fanout.py --quick
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
N_BURST = 30 if _args.quick else 100
WIDTHS = [16, 32, 64] if _args.quick else [16, 32, 64, 128]
DISPLAY_W = 88

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
YELLOW = "\033[33m"


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


def fmt_kb(value: float) -> str:
    kb = value / 1024
    if kb >= 1024:
        return f"{kb / 1024:>8.1f} MB"
    return f"{kb:>8.1f} KB"


def pct(samples: list[float], p: float) -> float:
    return sorted(samples)[min(int(len(samples) * p / 100), len(samples) - 1)]


from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal() -> None:
    InMemoryTracingProvider._storage.clear()


async def _get_root() -> Any:
    return get_run_context().step_span("root").output


async def _get_aggregator() -> Any:
    return get_run_context().step_span("aggregator").output


def _make_p1_branch(i: int, async_work: bool):
    async def branch(x: int) -> int:
        if async_work:
            await asyncio.sleep(0.001)
        return x * (i + 2)

    branch.__name__ = f"p1_{i}"
    return branch


def _make_p2_branch(i: int, async_work: bool):
    async def branch(y: int) -> int:
        if async_work:
            await asyncio.sleep(0.001)
        return y * (i + 2)

    branch.__name__ = f"p2_{i}"
    return branch


def _timbal_double(n: int, async_work: bool = False) -> Workflow:
    wf = Workflow(name=f"double_{n}")

    async def root(x: int) -> int:
        return x + 1

    wf.step(root)
    for i in range(n):
        wf.step(_make_p1_branch(i, async_work), depends_on=["root"], x=_get_root)

    async def aggregator(results: list[int]) -> int:
        return sum(results)

    wf.step(
        aggregator,
        depends_on=[f"p1_{i}" for i in range(n)],
        results=lambda: [get_run_context().step_span(f"p1_{i}").output for i in range(n)],
    )

    for i in range(n):
        wf.step(_make_p2_branch(i, async_work), depends_on=["aggregator"], y=_get_aggregator)

    async def sink(results: list[int]) -> int:
        return sum(results)

    wf.step(
        sink,
        depends_on=[f"p2_{i}" for i in range(n)],
        results=lambda: [get_run_context().step_span(f"p2_{i}").output for i in range(n)],
    )
    return wf


HAS_PG = False
HAS_LOGFIRE = False

try:
    from pydantic_graph import BaseNode, End, Graph, GraphRunContext  # noqa: E402

    @dataclasses.dataclass
    class _DoubleState:
        x: int = 0
        n: int = 0
        async_work: bool = False
        root_out: int = 0
        p1_results: list[int] = dataclasses.field(default_factory=list)
        agg_out: int = 0
        p2_results: list[int] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class _Root(BaseNode[_DoubleState, None, int]):
        async def run(self, ctx: GraphRunContext[_DoubleState, None]) -> _Phase1:
            ctx.state.root_out = ctx.state.x + 1
            return _Phase1()

    @dataclasses.dataclass
    class _Phase1(BaseNode[_DoubleState, None, int]):
        async def run(self, ctx: GraphRunContext[_DoubleState, None]) -> _Aggregator:
            async def branch(i: int) -> int:
                if ctx.state.async_work:
                    await asyncio.sleep(0.001)
                return ctx.state.root_out * (i + 2)

            ctx.state.p1_results = list(await asyncio.gather(*(branch(i) for i in range(ctx.state.n))))
            return _Aggregator()

    @dataclasses.dataclass
    class _Aggregator(BaseNode[_DoubleState, None, int]):
        async def run(self, ctx: GraphRunContext[_DoubleState, None]) -> _Phase2:
            ctx.state.agg_out = sum(ctx.state.p1_results)
            return _Phase2()

    @dataclasses.dataclass
    class _Phase2(BaseNode[_DoubleState, None, int]):
        async def run(self, ctx: GraphRunContext[_DoubleState, None]) -> _Sink:
            async def branch(i: int) -> int:
                if ctx.state.async_work:
                    await asyncio.sleep(0.001)
                return ctx.state.agg_out * (i + 2)

            ctx.state.p2_results = list(await asyncio.gather(*(branch(i) for i in range(ctx.state.n))))
            return _Sink()

    @dataclasses.dataclass
    class _Sink(BaseNode[_DoubleState, None, int]):
        async def run(self, ctx: GraphRunContext[_DoubleState, None]) -> End[int]:
            return End(sum(ctx.state.p2_results))

    def _pg_double(auto_instrument: bool):
        return Graph(
            nodes=[_Root, _Phase1, _Aggregator, _Phase2, _Sink],
            name="pg_double",
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


@dataclasses.dataclass
class RunData:
    trivial: list[float]
    async_work: list[float]
    mem_peak: float
    mem_per: float
    burst: list[float]


async def _collect(label: str, trivial: Factory, async_work: Factory) -> RunData:
    print(f"    {label} trivial...", end=" ", flush=True)
    trivial_samples = await _latency(trivial, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(trivial_samples, 50)).strip()}", flush=True)

    print(f"    {label} async work...", end=" ", flush=True)
    async_samples = await _latency(async_work, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(async_samples, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory(trivial, N_ITERS, N_WARMUP)
    print(f"{fmt_kb(mem_peak).strip()} peak", flush=True)

    print(f"    {label} burst ({N_BURST} concurrent, async work)...", end=" ", flush=True)
    burst = await _burst(async_work, N_BURST)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    return RunData(trivial=trivial_samples, async_work=async_samples, mem_peak=mem_peak, mem_per=mem_per, burst=burst)


def _timbal_factory(n: int, async_work: bool) -> Factory:
    wf = _timbal_double(n, async_work=async_work)
    return lambda: wf(x=5).collect()


def _pg_factory(n: int, async_work: bool, auto_instrument: bool) -> Factory:
    graph = _pg_double(auto_instrument=auto_instrument)
    return lambda: graph.run(_Root(), state=_DoubleState(x=5, n=n, async_work=async_work))


def _print_width(n: int, rows: list[tuple[str, RunData]]) -> None:
    section(f"Width N={n}  (2N+3 target steps; PG manual uses 5 graph nodes)")
    col_w = 14
    hdr = f"  {'':>12}" + "".join(f"  {label:>{col_w}}" for label, _ in rows)
    sep = f"  {'─' * 12}" + f"  {'─' * col_w}" * len(rows)

    for title, attr in [("Trivial branches", "trivial"), ("Async work branches (2× 1 ms fan-outs)", "async_work")]:
        subsection(f"{title}  (×{N_ITERS})")
        print(hdr)
        print(sep)
        for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
            vals = []
            for _, d in rows:
                samples = getattr(d, attr)
                v = statistics.mean(samples) if p is None else pct(samples, p)
                vals.append(fmt_us(v))
            print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    subsection(f"Burst  ({N_BURST} concurrent, async work)")
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d.burst, p)) for _, d in rows]
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))
    wall = " | ".join(f"{label}: {max(d.burst) / 1e3:.1f} ms" for label, d in rows)
    print(f"\n  {DIM}wall: {wall}{RESET}")

    subsection(f"Memory  (trivial, ×{N_ITERS})")
    print(f"  {'framework':<20}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * 20}  {'─' * 12}  {'─' * 12}")
    for label, d in rows:
        print(f"  {label:<20}  {fmt_kb(d.mem_peak)}  {d.mem_per:>10.0f}  B")


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")
    print(f"{BOLD}  Timbal Workflow vs Pydantic Graph — double fan-out benchmark{RESET}")
    print(f"  widths={WIDTHS} · {N_ITERS} iters · {N_BURST} burst")
    if HAS_PG:
        print("  Pydantic Graph found — will measure bare + Logfire")
        print(f"  {DIM}PG manual uses two asyncio.gather fan-outs inside graph nodes.{RESET}")
    else:
        print(f"  {YELLOW}Pydantic Graph not found. Install: uv pip install pydantic-ai logfire{RESET}")
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")

    all_rows: dict[int, list[tuple[str, RunData]]] = {}
    for n in WIDTHS:
        print(f"\n  {DIM}[Phase 1 · N={n}]{RESET}", flush=True)
        rows = [("Timbal", await _collect("Timbal", _timbal_factory(n, False), _timbal_factory(n, True)))]
        if HAS_PG:
            rows.append(("PG bare", await _collect("PG bare", _pg_factory(n, False, False), _pg_factory(n, True, False))))
        all_rows[n] = rows

    if HAS_PG:
        print(f"\n  {DIM}Activating Logfire (send_to_logfire=False)...{RESET}", flush=True)
        if _try_init_logfire():
            for n in WIDTHS:
                print(f"\n  {DIM}[Phase 2 · N={n}]{RESET}", flush=True)
                all_rows[n].append(
                    (
                        "PG+Logfire",
                        await _collect("PG+Logfire", _pg_factory(n, False, True), _pg_factory(n, True, True)),
                    )
                )
        else:
            print(f"  {YELLOW}Logfire not found — skipping obs column.{RESET}", flush=True)

    for n, rows in all_rows.items():
        _print_width(n, rows)

    print()
    print(f"{DIM}{'─' * DISPLAY_W}")
    print("  Timbal: 2N+3 first-class workflow steps, one task/span per step.")
    print("  PG manual: 5 graph nodes total, with branch concurrency owned by user code.")
    print("  Async-work latency should be near 2 ms when both fan-out phases are concurrent.")
    print(f"{'─' * DISPLAY_W}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
