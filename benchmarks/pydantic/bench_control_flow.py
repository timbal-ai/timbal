#!/usr/bin/env python3
"""
Timbal Workflow vs Pydantic Graph — branchy control-flow benchmark.

Topology per round:
  decision_i → (left_i OR right_i) → join_i

The same shape repeats for N rounds. There is no internal fan-out and no
asyncio.gather inside the workload. This is the case where Pydantic Graph cannot
compress many branches into one manually-owned gather node; it has to run and
instrument repeated graph nodes on the hot path.

Pydantic Graph uses a real while-style loop:
  Decision → Left/Right → Join → Decision ... → End

Timbal Workflow is a DAG, so the same bounded loop is unrolled into N explicit
decision/branch/join rounds.

Columns:
  Timbal        — built-in InMemory tracing always on
  PG bare       — Pydantic Graph with auto_instrument=False
  PG+Logfire    — Pydantic Graph with auto_instrument=True and Logfire configured
                  with send_to_logfire=False (span creation, no network I/O)

Run:
    uv run python benchmarks/pydantic/bench_control_flow.py
    uv run python benchmarks/pydantic/bench_control_flow.py --quick
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


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — bounded loop unrolled into an explicit DAG
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Workflow  # noqa: E402
from timbal.state import get_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_timbal() -> None:
    InMemoryTracingProvider._storage.clear()


async def _step_output(name: str, default: Any = None) -> Any:
    span = get_run_context().step_span(name, default=None)
    return default if span is None else span.output


def _make_decision(i: int):
    async def decision(x: int) -> dict[str, int | bool]:
        take_left = ((x + i) % 3) != 0
        return {"value": x, "take_left": take_left}

    decision.__name__ = f"decision_{i}"
    return decision


def _make_left(i: int):
    async def left(decision: dict[str, int | bool]) -> int:
        return int(decision["value"]) * 3 + i

    left.__name__ = f"left_{i}"
    return left


def _make_right(i: int):
    async def right(decision: dict[str, int | bool]) -> int:
        return int(decision["value"]) + 7 + i

    right.__name__ = f"right_{i}"
    return right


def _make_join(i: int):
    async def join(left: int | None, right: int | None) -> int:
        return left if left is not None else int(right)

    join.__name__ = f"join_{i}"
    return join


def _output_param(name: str, default: Any = None):
    async def resolve() -> Any:
        return await _step_output(name, default)

    return resolve


def _take_left(name: str):
    async def when() -> bool:
        return bool((await _step_output(name))["take_left"])

    return when


def _take_right(name: str):
    async def when() -> bool:
        return not bool((await _step_output(name))["take_left"])

    return when


def _timbal_control(rounds: int) -> Workflow:
    wf = Workflow(name=f"control_{rounds}")
    previous_step: str | None = None

    for i in range(rounds):
        decision_name = f"decision_{i}"
        left_name = f"left_{i}"
        right_name = f"right_{i}"
        join_name = f"join_{i}"

        if previous_step is None:
            wf.step(_make_decision(i))
        else:
            wf.step(_make_decision(i), depends_on=[previous_step], x=_output_param(previous_step))

        wf.step(
            _make_left(i),
            depends_on=[decision_name],
            when=_take_left(decision_name),
            decision=_output_param(decision_name),
        )
        wf.step(
            _make_right(i),
            depends_on=[decision_name],
            when=_take_right(decision_name),
            decision=_output_param(decision_name),
        )
        wf.step(
            _make_join(i),
            depends_on=[left_name, right_name],
            left=_output_param(left_name),
            right=_output_param(right_name),
        )
        previous_step = join_name

    return wf


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC GRAPH — actual while-style loop
# ═══════════════════════════════════════════════════════════════════════════════

HAS_PG = False
HAS_LOGFIRE = False

try:
    from pydantic_graph import BaseNode, End, Graph, GraphRunContext  # noqa: E402

    @dataclasses.dataclass
    class _ControlState:
        x: int = 0
        i: int = 0
        rounds: int = 0
        branch_value: int = 0

    @dataclasses.dataclass
    class _Decision(BaseNode[_ControlState, None, int]):
        async def run(self, ctx: GraphRunContext[_ControlState, None]) -> _Left | _Right:
            if ((ctx.state.x + ctx.state.i) % 3) != 0:
                return _Left()
            return _Right()

    @dataclasses.dataclass
    class _Left(BaseNode[_ControlState, None, int]):
        async def run(self, ctx: GraphRunContext[_ControlState, None]) -> _Join:
            ctx.state.branch_value = ctx.state.x * 3 + ctx.state.i
            return _Join()

    @dataclasses.dataclass
    class _Right(BaseNode[_ControlState, None, int]):
        async def run(self, ctx: GraphRunContext[_ControlState, None]) -> _Join:
            ctx.state.branch_value = ctx.state.x + 7 + ctx.state.i
            return _Join()

    @dataclasses.dataclass
    class _Join(BaseNode[_ControlState, None, int]):
        async def run(self, ctx: GraphRunContext[_ControlState, None]) -> _Decision | End[int]:
            ctx.state.x = ctx.state.branch_value
            ctx.state.i += 1
            if ctx.state.i < ctx.state.rounds:
                return _Decision()
            return End(ctx.state.x)

    def _pg_control(auto_instrument: bool):
        return Graph(
            nodes=[_Decision, _Left, _Right, _Join],
            name="pg_control",
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


def _expected(x: int, rounds: int) -> int:
    for i in range(rounds):
        if ((x + i) % 3) != 0:
            x = x * 3 + i
        else:
            x = x + 7 + i
    return x


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
    wf = _timbal_control(rounds)
    return lambda: wf(x=5).collect()


def _pg_factory(rounds: int, auto_instrument: bool) -> Factory:
    graph = _pg_control(auto_instrument=auto_instrument)
    return lambda: graph.run(_Decision(), state=_ControlState(x=5, rounds=rounds))


def _print_rounds(rounds: int, rows: list[tuple[str, RunData]]) -> None:
    section(f"{rounds} branch rounds  (~{rounds * 3} hot-path nodes/steps)")
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
    print(f"{BOLD}  Timbal Workflow vs Pydantic Graph — branchy control-flow benchmark{RESET}")
    print(f"  rounds={ROUNDS} · {N_ITERS} iters · {N_BURST} burst · no internal gather")
    if HAS_PG:
        print("  Pydantic Graph found — will measure bare + Logfire")
        print(f"  {DIM}PG uses a real while-style loop; Timbal uses the equivalent unrolled DAG.{RESET}")
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
    print("  Timbal: bounded branch loop unrolled into explicit workflow steps.")
    print("  PG: actual while-style graph loop with Decision → Left/Right → Join repeated.")
    print("  No workload-level asyncio.gather is used; every round is sequential control flow.")
    print("  The fair observable comparison is Timbal vs PG+Logfire.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
