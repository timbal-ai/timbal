#!/usr/bin/env python3
"""
Timbal Workflow vs Agno Workflow — DAG benchmark.

Three scenarios with trivial async handlers:
  1. Sequential:    A → B → C → D
  2. Fan-out/in:    A → [B, C, D] → E
  3. Diamond:       A → [B, C] → D

Agno uses its real Workflow/Parallel primitives with telemetry=False. Timbal uses
Workflow with built-in InMemory tracing always on.

Run:
    uv run python benchmarks/agno/bench_workflow.py
    uv run python benchmarks/agno/bench_workflow.py --quick
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


from timbal import Workflow as TimbalWorkflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal() -> None:
    InMemoryTracingProvider._storage.clear()


def _timbal_sequential() -> TimbalWorkflow:
    async def step_a(x: int) -> int:
        return x + 1

    async def step_b(x: int) -> int:
        return x * 2

    async def step_c(x: int) -> int:
        return x + 10

    async def step_d(x: int) -> int:
        return x - 3

    wf = TimbalWorkflow(name="sequential")
    wf.step(step_a)
    wf.step(step_b, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(step_c, x=lambda: get_run_context().step_span("step_b").output)
    wf.step(step_d, x=lambda: get_run_context().step_span("step_c").output)
    return wf


def _timbal_fanout() -> TimbalWorkflow:
    async def step_a(x: int) -> int:
        return x + 1

    async def branch_b(x: int) -> int:
        return x * 2

    async def branch_c(x: int) -> int:
        return x * 3

    async def branch_d(x: int) -> int:
        return x * 4

    async def step_e(b: int, c: int, d: int) -> int:
        return b + c + d

    wf = TimbalWorkflow(name="fanout")
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


def _timbal_diamond() -> TimbalWorkflow:
    async def step_a(x: int) -> int:
        return x + 1

    async def path_b(x: int) -> int:
        return x + 10

    async def path_c(x: int) -> int:
        return x * 5

    async def combine(b: int, c: int) -> int:
        return b + c

    wf = TimbalWorkflow(name="diamond")
    wf.step(step_a)
    wf.step(path_b, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(path_c, x=lambda: get_run_context().step_span("step_a").output)
    wf.step(
        combine,
        b=lambda: get_run_context().step_span("path_b").output,
        c=lambda: get_run_context().step_span("path_c").output,
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

    def _prev(inp: StepInput) -> Any:
        return inp.previous_step_content

    def _step(inp: StepInput, name: str) -> StepOutput | None:
        if not inp.previous_step_outputs:
            return None
        return inp.previous_step_outputs.get(name)

    async def ag_step_a(inp: StepInput) -> StepOutput:
        return StepOutput(content=_input_x(inp) + 1)

    async def ag_step_b(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) * 2)

    async def ag_step_c(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) + 10)

    async def ag_step_d(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) - 3)

    async def ag_branch_b(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) * 2)

    async def ag_branch_c(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) * 3)

    async def ag_branch_d(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) * 4)

    async def ag_step_e(inp: StepInput) -> StepOutput:
        par = _step(inp, "parallel")
        assert par is not None and par.steps is not None
        return StepOutput(content=sum(int(s.content) for s in par.steps))

    async def ag_path_b(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) + 10)

    async def ag_path_c(inp: StepInput) -> StepOutput:
        return StepOutput(content=int(_prev(inp)) * 5)

    async def ag_combine(inp: StepInput) -> StepOutput:
        par = _step(inp, "parallel")
        assert par is not None and par.steps is not None
        return StepOutput(content=sum(int(s.content) for s in par.steps))

    def _agno_sequential() -> AgnoWorkflow:
        return AgnoWorkflow(
            name="sequential",
            steps=[ag_step_a, ag_step_b, ag_step_c, ag_step_d],
            telemetry=False,
            store_events=False,
            store_executor_outputs=False,
        )

    def _agno_fanout() -> AgnoWorkflow:
        return AgnoWorkflow(
            name="fanout",
            steps=[ag_step_a, Parallel(ag_branch_b, ag_branch_c, ag_branch_d, name="parallel"), ag_step_e],
            telemetry=False,
            store_events=False,
            store_executor_outputs=False,
        )

    def _agno_diamond() -> AgnoWorkflow:
        return AgnoWorkflow(
            name="diamond",
            steps=[ag_step_a, Parallel(ag_path_b, ag_path_c, name="parallel"), ag_combine],
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
    print(f"{mem_peak / 1024:.1f} KB peak", flush=True)

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


SCENARIOS = {
    "sequential": "A → B → C → D",
    "fanout": "A → [B, C, D] → E",
    "diamond": "A → [B, C] → D",
}


def _timbal_factory(name: str) -> Factory:
    wf = {
        "sequential": _timbal_sequential,
        "fanout": _timbal_fanout,
        "diamond": _timbal_diamond,
    }[name]()
    return lambda: wf(x=5).collect()


def _agno_factory(name: str) -> Factory:
    wf = {
        "sequential": _agno_sequential,
        "fanout": _agno_fanout,
        "diamond": _agno_diamond,
    }[name]()
    return lambda: wf.arun({"x": 5})


def _print_scenario(name: str, rows: list[tuple[str, RunData]]) -> None:
    section(f"{name}: {SCENARIOS[name]}")
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
    print(f"{BOLD}  Timbal Workflow vs Agno Workflow — workflow benchmark{RESET}")
    print(f"  {N_ITERS} iters · {N_BURST} burst · {N_MEM} mem · {THROUGHPUT_OPS} throughput ops")
    if HAS_AGNO:
        print("  Agno found — telemetry=False")
    else:
        print(f"  {YELLOW}Agno not found. Install: uv pip install agno{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    print(f"\n  {DIM}Spot-checking correctness (sequential)...{RESET}", flush=True)
    t_res = await _timbal_factory("sequential")()
    t_ok = t_res.output == 19
    msg = f"  Timbal: {'✓' if t_ok else f'✗ (got {t_res.output!r})'}"
    if HAS_AGNO:
        a_res = await _agno_factory("sequential")()
        a_ok = a_res.content == 19
        msg += f"  |  Agno: {'✓' if a_ok else f'✗ (got {a_res.content!r})'}"
    print(msg, flush=True)

    all_rows: dict[str, list[tuple[str, RunData]]] = {}
    for scenario in SCENARIOS:
        print(f"\n  {DIM}[Scenario · {scenario}]{RESET}", flush=True)
        rows = [("Timbal", await _collect("Timbal", _timbal_factory(scenario)))]
        if HAS_AGNO:
            rows.append(("Agno", await _collect("Agno", _agno_factory(scenario))))
        all_rows[scenario] = rows

    for scenario, rows in all_rows.items():
        _print_scenario(scenario, rows)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print("  Timbal: Workflow DAG with built-in InMemory tracing always on.")
    print("  Agno: Workflow/Parallel primitives with telemetry=False.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
