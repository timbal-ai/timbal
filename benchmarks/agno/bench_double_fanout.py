#!/usr/bin/env python3
"""
Timbal Workflow vs Agno Workflow — double fan-out benchmark.

Topology: root → [N phase-1 branches] → aggregate → [N phase-2 branches] → sink

Agno uses real Workflow/Parallel phases with telemetry=False. Timbal uses Workflow
with built-in InMemory tracing always on.

Run:
    uv run python benchmarks/agno/bench_double_fanout.py
    uv run python benchmarks/agno/bench_double_fanout.py --quick
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
os.environ.setdefault("AGNO_TELEMETRY", "false")
warnings.filterwarnings("ignore")

import structlog  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
_args, _ = parser.parse_known_args()

N_ITERS = 30 if _args.quick else 100
N_WARMUP = 5 if _args.quick else 15
N_BURST = 30 if _args.quick else 100
WIDTHS = [4, 8, 16] if _args.quick else [4, 8, 16, 32]
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


def pct(samples: list[float], p: float) -> float:
    return sorted(samples)[min(int(len(samples) * p / 100), len(samples) - 1)]


from timbal import Workflow as TimbalWorkflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal() -> None:
    InMemoryTracingProvider._storage.clear()


async def _get_root() -> Any:
    return get_run_context().step_span("root").output


async def _get_agg() -> Any:
    return get_run_context().step_span("aggregate").output


def _phase1_fn(i: int, async_work: bool):
    async def phase1(x: int) -> int:
        if async_work:
            await asyncio.sleep(0.001)
        return x + i

    phase1.__name__ = f"phase1_{i}"
    return phase1


def _phase2_fn(i: int, async_work: bool):
    async def phase2(x: int) -> int:
        if async_work:
            await asyncio.sleep(0.001)
        return x * (i + 1)

    phase2.__name__ = f"phase2_{i}"
    return phase2


def _timbal_double(n: int, async_work: bool = False) -> TimbalWorkflow:
    wf = TimbalWorkflow(name=f"double_{n}")

    async def root(x: int) -> int:
        return x + 1

    async def aggregate(values: list[int]) -> int:
        return sum(values)

    async def sink(values: list[int]) -> int:
        return sum(values)

    wf.step(root)
    for i in range(n):
        wf.step(_phase1_fn(i, async_work), depends_on=["root"], x=_get_root)

    wf.step(
        aggregate,
        depends_on=[f"phase1_{i}" for i in range(n)],
        values=lambda: [get_run_context().step_span(f"phase1_{i}").output for i in range(n)],
    )

    for i in range(n):
        wf.step(_phase2_fn(i, async_work), depends_on=["aggregate"], x=_get_agg)

    wf.step(
        sink,
        depends_on=[f"phase2_{i}" for i in range(n)],
        values=lambda: [get_run_context().step_span(f"phase2_{i}").output for i in range(n)],
    )
    return wf


HAS_AGNO = False

try:
    from agno.utils.log import logger as agno_logger  # noqa: E402
    from agno.utils.log import workflow_logger
    from agno.workflow import Parallel  # noqa: E402
    from agno.workflow import Workflow as AgnoWorkflow
    from agno.workflow.types import StepInput, StepOutput  # noqa: E402

    agno_logger.setLevel(logging.CRITICAL)
    workflow_logger.setLevel(logging.CRITICAL)

    def _input_x(inp: StepInput) -> int:
        if isinstance(inp.input, dict):
            return int(inp.input["x"])
        return int(inp.input)

    async def ag_root(inp: StepInput) -> StepOutput:
        return StepOutput(content=_input_x(inp) + 1)

    def _ag_phase1_fn(i: int, async_work: bool):
        async def phase1(inp: StepInput) -> StepOutput:
            if async_work:
                await asyncio.sleep(0.001)
            return StepOutput(content=int(inp.previous_step_content) + i)

        phase1.__name__ = f"phase1_{i}"
        return phase1

    async def ag_aggregate(inp: StepInput) -> StepOutput:
        par = inp.previous_step_outputs["phase1"]
        assert par.steps is not None
        return StepOutput(content=sum(int(step.content) for step in par.steps))

    def _ag_phase2_fn(i: int, async_work: bool):
        async def phase2(inp: StepInput) -> StepOutput:
            if async_work:
                await asyncio.sleep(0.001)
            return StepOutput(content=int(inp.previous_step_content) * (i + 1))

        phase2.__name__ = f"phase2_{i}"
        return phase2

    async def ag_sink(inp: StepInput) -> StepOutput:
        par = inp.previous_step_outputs["phase2"]
        assert par.steps is not None
        return StepOutput(content=sum(int(step.content) for step in par.steps))

    def _agno_double(n: int, async_work: bool = False) -> AgnoWorkflow:
        phase1 = [_ag_phase1_fn(i, async_work) for i in range(n)]
        phase2 = [_ag_phase2_fn(i, async_work) for i in range(n)]
        return AgnoWorkflow(
            name=f"double_{n}",
            steps=[ag_root, Parallel(*phase1, name="phase1"), ag_aggregate, Parallel(*phase2, name="phase2"), ag_sink],
            telemetry=False,
            store_events=False,
            store_executor_outputs=False,
        )

    HAS_AGNO = True
except ImportError as e:
    print(f"Agno not found: {e}")


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
    print(f"{mem_peak / 1024:.1f} KB peak", flush=True)

    print(f"    {label} burst ({N_BURST} concurrent, async work)...", end=" ", flush=True)
    burst = await _burst(async_work, N_BURST)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    return RunData(trivial=trivial_samples, async_work=async_samples, mem_peak=mem_peak, mem_per=mem_per, burst=burst)


def _timbal_factory(n: int, async_work: bool) -> Factory:
    wf = _timbal_double(n, async_work=async_work)
    return lambda: wf(x=5).collect()


def _agno_factory(n: int, async_work: bool) -> Factory:
    wf = _agno_double(n, async_work=async_work)
    return lambda: wf.arun({"x": 5})


def _print_width(n: int, rows: list[tuple[str, RunData]]) -> None:
    section(f"Width N={n} per phase")
    col_w = 14
    hdr = f"  {'':>12}" + "".join(f"  {label:>{col_w}}" for label, _ in rows)
    sep = f"  {'─' * 12}" + f"  {'─' * col_w}" * len(rows)

    for title, attr in [("Trivial branches", "trivial"), ("Async work branches (1 ms sleep per branch)", "async_work")]:
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
        print(f"  {label:<20}  {d.mem_peak / 1024:>10.1f} KB  {d.mem_per:>10.0f}  B")


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal Workflow vs Agno Workflow — double fan-out benchmark{RESET}")
    print(f"  widths={WIDTHS} · {N_ITERS} iters · {N_BURST} burst")
    if HAS_AGNO:
        print("  Agno found — telemetry=False")
    else:
        print(f"  {YELLOW}Agno not found. Install: uv pip install agno{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    all_rows: dict[int, list[tuple[str, RunData]]] = {}
    for n in WIDTHS:
        print(f"\n  {DIM}[N={n} per phase]{RESET}", flush=True)
        rows = [("Timbal", await _collect("Timbal", _timbal_factory(n, False), _timbal_factory(n, True)))]
        if HAS_AGNO:
            rows.append(("Agno", await _collect("Agno", _agno_factory(n, False), _agno_factory(n, True))))
        all_rows[n] = rows

    for n, rows in all_rows.items():
        _print_width(n, rows)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print("  Timbal: two explicit first-class fan-out phases, one task/span per branch.")
    print("  Agno: two Workflow Parallel phases with telemetry=False.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
