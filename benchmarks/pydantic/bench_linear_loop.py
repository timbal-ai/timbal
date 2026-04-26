#!/usr/bin/env python3
"""
Timbal Workflow vs Pydantic Graph — linear loop benchmark.

This removes the branchy-control-flow caveats:
  - no if/else branch choice in the workload
  - no untaken branch steps
  - no workload-level asyncio.gather

Pydantic Graph uses a simple while-style loop:
  Step → Step → ... → End

Timbal Workflow uses the equivalent bounded loop unrolled into N sequential async
steps. This isolates per-executed-node/step overhead.

Run:
    uv run python benchmarks/pydantic/bench_linear_loop.py
    uv run python benchmarks/pydantic/bench_linear_loop.py --quick
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
N_MEM = 50 if _args.quick else 200
N_BURST = 30 if _args.quick else 100
THROUGHPUT_OPS = 100 if _args.quick else 500
CONCURRENCY_LEVELS = [1, 10, 50, 200]
ROUNDS = [8, 16, 32] if _args.quick else [8, 16, 32, 64]
WIDTH = 92

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


async def _step_output(name: str) -> Any:
    return get_run_context().step_span(name).output


def _output_param(name: str):
    async def resolve() -> Any:
        return await _step_output(name)

    return resolve


def _make_step(i: int):
    async def step(x: int) -> int:
        return x * 3 + i

    step.__name__ = f"step_{i}"
    return step


def _timbal_linear(rounds: int) -> Workflow:
    wf = Workflow(name=f"linear_{rounds}")
    previous_step: str | None = None

    for i in range(rounds):
        step_name = f"step_{i}"
        if previous_step is None:
            wf.step(_make_step(i))
        else:
            wf.step(_make_step(i), depends_on=[previous_step], x=_output_param(previous_step))
        previous_step = step_name

    return wf


HAS_PG = False
HAS_LOGFIRE = False

try:
    from pydantic_graph import BaseNode, End, Graph, GraphRunContext  # noqa: E402

    @dataclasses.dataclass
    class _LinearState:
        x: int = 0
        i: int = 0
        rounds: int = 0

    @dataclasses.dataclass
    class _Step(BaseNode[_LinearState, None, int]):
        async def run(self, ctx: GraphRunContext[_LinearState, None]) -> _Step | End[int]:
            ctx.state.x = ctx.state.x * 3 + ctx.state.i
            ctx.state.i += 1
            if ctx.state.i < ctx.state.rounds:
                return _Step()
            return End(ctx.state.x)

    def _pg_linear(auto_instrument: bool):
        return Graph(nodes=[_Step], name="pg_linear", auto_instrument=auto_instrument)

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


def _expected(x: int, rounds: int) -> int:
    for i in range(rounds):
        x = x * 3 + i
    return x


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
    latency = await _latency(factory, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(latency, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory(factory, N_MEM, N_WARMUP)
    print(f"{fmt_kb(mem_peak).strip()} peak", flush=True)

    print(f"    {label} burst ({N_BURST} concurrent)...", end=" ", flush=True)
    burst = await _burst(factory, N_BURST)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    throughput = []
    for conc in CONCURRENCY_LEVELS:
        print(f"    {label} throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput(factory, THROUGHPUT_OPS, conc)
        throughput.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=latency, mem_peak=mem_peak, mem_per=mem_per, burst=burst, throughput=throughput)


def _timbal_factory(rounds: int) -> Factory:
    wf = _timbal_linear(rounds)
    return lambda: wf(x=5).collect()


def _pg_factory(rounds: int, auto_instrument: bool) -> Factory:
    graph = _pg_linear(auto_instrument=auto_instrument)
    return lambda: graph.run(_Step(), state=_LinearState(x=5, rounds=rounds))


def _print_rounds(rounds: int, rows: list[tuple[str, RunData]]) -> None:
    section(f"{rounds} linear steps/nodes")
    col_w = 14
    hdr = f"  {'':>12}" + "".join(f"  {label:>{col_w}}" for label, _ in rows)
    sep = f"  {'─' * 12}" + f"  {'─' * col_w}" * len(rows)

    subsection(f"Latency  (×{N_ITERS})")
    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = []
        for _, d in rows:
            v = statistics.mean(d.latency) if p is None else pct(d.latency, p)
            vals.append(fmt_us(v))
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    subsection(f"Memory  (×{N_MEM})")
    print(f"  {'framework':<20}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * 20}  {'─' * 12}  {'─' * 12}")
    for label, d in rows:
        print(f"  {label:<20}  {fmt_kb(d.mem_peak)}  {d.mem_per:>10.0f}  B")

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
    print(f"{BOLD}  Timbal Workflow vs Pydantic Graph — linear loop benchmark{RESET}")
    print(f"  rounds={ROUNDS} · {N_ITERS} iters · {N_BURST} burst · no branches/skips/gather")
    if HAS_PG:
        print("  Pydantic Graph found — will measure bare + Logfire")
    else:
        print(f"  {YELLOW}Pydantic Graph not found. Install: uv pip install pydantic-ai logfire{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    first_rounds = ROUNDS[0]
    print(f"\n  {DIM}Spot-checking correctness ({first_rounds} rounds)...{RESET}", flush=True)
    expected = _expected(5, first_rounds)
    t_res = await _timbal_factory(first_rounds)()
    t_ok = t_res.output == expected
    msg = f"  Timbal: {'✓' if t_ok else f'✗ (got {t_res.output!r}, expected {expected!r})'}"
    if HAS_PG:
        p_res = await _pg_factory(first_rounds, auto_instrument=False)()
        p_ok = p_res.output == expected
        msg += f"  |  PG: {'✓' if p_ok else f'✗ (got {p_res.output!r}, expected {expected!r})'}"
    print(msg, flush=True)

    all_rows: dict[int, list[tuple[str, RunData]]] = {}
    for rounds in ROUNDS:
        print(f"\n  {DIM}[Phase 1 · rounds={rounds}]{RESET}", flush=True)
        rows = [("Timbal", await _collect("Timbal", _timbal_factory(rounds)))]
        if HAS_PG:
            rows.append(("PG bare", await _collect("PG bare", _pg_factory(rounds, auto_instrument=False))))
        all_rows[rounds] = rows

    if HAS_PG:
        print(f"\n  {DIM}Activating Logfire (send_to_logfire=False)...{RESET}", flush=True)
        if _try_init_logfire():
            for rounds in ROUNDS:
                print(f"\n  {DIM}[Phase 2 · rounds={rounds}]{RESET}", flush=True)
                all_rows[rounds].append(
                    ("PG+Logfire", await _collect("PG+Logfire", _pg_factory(rounds, auto_instrument=True)))
                )
        else:
            print(f"  {YELLOW}Logfire not found — skipping obs column.{RESET}", flush=True)

    for rounds, rows in all_rows.items():
        _print_rounds(rounds, rows)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print("  Timbal: bounded linear loop unrolled into explicit async workflow steps.")
    print("  PG: while-style graph loop repeatedly executing the same node class.")
    print("  No branch selection, skipped steps, or workload-level gather are involved.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
