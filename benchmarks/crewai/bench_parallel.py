#!/usr/bin/env python3
"""
Timbal vs CrewAI Flow — Wide Parallel Fan-out benchmark.

Tests scheduling overhead and true parallelism across N parallel branches.

  Topology:  root → [N async branches] → sink

  Scenario A (trivial):    each branch returns immediately (pure overhead)
  Scenario B (async work): each branch does asyncio.sleep(0.001) — 1 ms

Branch widths tested: N = 4, 8, 16  (quick) / 4, 8, 16, 32, 64  (full)

Two columns:
  Timbal        — one asyncio.Task + Span per branch (built-in tracing always on)
  Flow steps    — N individual @listen methods + OpenInference (N+3 spans per kickoff,
                  structurally equivalent to Timbal's per-step model)

Scenario B is the definitive parallelism test:
  - Truly parallel → p50 ≈ 1 ms flat regardless of N
  - Sequential     → p50 ≈ N × 1 ms

Run:
    uv run python benchmarks/crewai/bench_parallel.py
    uv run python benchmarks/crewai/bench_parallel.py --quick
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import statistics
import time
import tracemalloc
import warnings
from typing import Any

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
warnings.filterwarnings("ignore")

import structlog  # noqa: E402

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


def _make_all_branches_getter(n: int):
    async def get_all_branches() -> list:
        ctx = get_run_context()
        return [ctx.step_span(f"branch_{i}").output for i in range(n)]
    return get_all_branches


def _make_branch_fn(i: int, async_work: bool):
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

    async def root(x: int) -> int:
        return x + 1

    wf.step(root)

    for i in range(n):
        branch = _make_branch_fn(i, async_work)
        wf.step(branch, depends_on=["root"], x=_make_root_getter())

    async def sink(results: list) -> int:
        return sum(results)

    wf.step(
        sink,
        depends_on=[f"branch_{i}" for i in range(n)],
        results=_make_all_branches_getter(n),
    )
    return wf


# ═══════════════════════════════════════════════════════════════════════════════
# CREWAI FLOW
#
# _make_wide_listen_flow(n)  ("Flow steps") — N individual @listen methods built
#   dynamically via type(). Each branch is a first-class Flow step. OI instruments
#   each @start/@listen invocation → N+3 spans per kickoff (structurally equal to
#   Timbal). NullExporter: full span pipeline measured, no network I/O.
# ═══════════════════════════════════════════════════════════════════════════════

HAS_OPENINFERENCE = False
_oi_instrumentor = None

try:
    from openinference.instrumentation.crewai import CrewAIInstrumentor as _OIInstrumentor  # noqa: E402
    from opentelemetry.sdk.trace import TracerProvider as _OTelTracerProvider  # noqa: E402
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor as _SimpleSpanProcessor  # noqa: E402
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter  # noqa: E402
    from opentelemetry.sdk.trace.export import SpanExportResult as _SpanExportResult  # noqa: E402

    class _NullExporter(_SpanExporter):
        def export(self, spans):
            return _SpanExportResult.SUCCESS
        def shutdown(self):
            pass

    def _oi_instrument():
        global _oi_instrumentor
        provider = _OTelTracerProvider()
        provider.add_span_processor(_SimpleSpanProcessor(_NullExporter()))
        _oi_instrumentor = _OIInstrumentor()
        _oi_instrumentor.instrument(tracer_provider=provider)

    def _oi_uninstrument():
        global _oi_instrumentor
        if _oi_instrumentor is not None:
            _oi_instrumentor.uninstrument()
            _oi_instrumentor = None

    HAS_OPENINFERENCE = True

except ImportError:
    def _oi_instrument(): pass  # type: ignore[misc]
    def _oi_uninstrument(): pass  # type: ignore[misc]


HAS_CREWAI = False

try:
    from crewai.flow.flow import Flow, start, listen, and_  # noqa: E402
    from pydantic import BaseModel, Field as _Field  # noqa: E402

    # ── Flow steps (fair comparison: N @listen methods + OI → N+3 spans) ───────
    class _WideState(BaseModel):
        x: int = 0
        results: list = _Field(default_factory=list)

    _FlowWithWideState = Flow[_WideState]

    def _make_wide_listen_flow(n: int, async_work: bool = False):
        """Build a Flow class with root + N @listen branches + sink using type()."""
        async def root(self):
            return getattr(self.state, "x", 0) + 1
        root_m = start()(root)
        methods: dict = {"root": root_m, "__module__": __name__, "__qualname__": f"_WideListenFlow_{n}"}

        for i in range(n):
            async def branch_fn(self, root_val, i=i):
                if async_work:
                    await asyncio.sleep(0.001)
                result = root_val * (i + 2)
                self.state.results.append(result)
                return result
            branch_fn.__name__ = f"branch_{i}"
            methods[f"branch_{i}"] = listen(root_m)(branch_fn)

        branch_methods = [methods[f"branch_{i}"] for i in range(n)]

        async def sink(self, _=None, _n=n):
            return sum(self.state.results)
        methods["sink"] = listen(and_(*branch_methods))(sink)

        return type(f"_WideListenFlow_{n}", (_FlowWithWideState,), methods)

    HAS_CREWAI = True

except ImportError as e:
    print(f"CrewAI not found: {e}")


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

    cols = ["Timbal"]
    if HAS_CREWAI and HAS_OPENINFERENCE:
        cols.append("Flow steps")
    col_w = 14

    # Build all instances upfront (construction excluded from timing).
    # CrewAI Flow instances are NOT safe for concurrent reuse — fresh per call.
    instances: dict[int, dict] = {}
    for n in WIDTHS:
        t_wf = _timbal_wide(n, async_work)
        listen_cls = _make_wide_listen_flow(n, async_work) if HAS_CREWAI and HAS_OPENINFERENCE else None
        instances[n] = {
            "t_run": lambda _w=t_wf: _w(x=5).collect(),
            "steps_run": (lambda _cls=listen_cls: _cls(suppress_flow_events=True).kickoff_async(inputs={"x": 5}))
                if listen_cls is not None else None,
        }

    # Pass 1: Timbal (OI off)
    # Pass 2: Flow steps (instrument OI, run @listen flow, uninstrument)
    data: dict[int, dict] = {}
    for n in WIDTHS:
        inst = instances[n]
        t_s = await _latency(inst["t_run"], N_ITERS, N_WARMUP)
        t_b = await _burst(inst["t_run"], N_BURST)

        st_s = st_b = None
        if inst["steps_run"] is not None:
            _oi_instrument()
            st_s = await _latency(inst["steps_run"], N_ITERS, N_WARMUP)
            st_b = await _burst(inst["steps_run"], N_BURST)
            _oi_uninstrument()

        data[n] = {"t_s": t_s, "st_s": st_s, "t_b": t_b, "st_b": st_b}

    hdr = f"  {'N':>6}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = f"  {'─' * 6}" + f"  {'─' * col_w}" * len(cols)

    # ── Latency tables ─────────────────────────────────────────────────────────
    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Latency {stat_label}  (×{N_ITERS} sequential runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_s"], stat_p)]
            if r["st_s"] is not None:
                vals.append(pct(r["st_s"], stat_p))
            print(f"  {n:>6}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    # ── Scaling overhead ───────────────────────────────────────────────────────
    subsection("Overhead per extra branch  (slope of p50 latency vs N)")
    print(f"  {DIM}Estimated by linear regression over the N values above{RESET}")
    if len(WIDTHS) >= 2:
        slope_cols = [("Timbal", "t_s")]
        if HAS_CREWAI and HAS_OPENINFERENCE:
            slope_cols.append(("Flow steps", "st_s"))
        for label_s, key in slope_cols:
            xs = WIDTHS
            ys = [pct(data[n][key], 50) for n in WIDTHS if data[n].get(key) is not None]
            if len(ys) < 2:
                continue
            x_mean = statistics.mean(xs)
            y_mean = statistics.mean(ys)
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / \
                    sum((x - x_mean) ** 2 for x in xs)
            print(f"  {label_s:<14}  {slope:>+.1f} µs / branch")

    # ── Burst tables ────────────────────────────────────────────────────────────
    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Burst {stat_label}  ({N_BURST} concurrent runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_b"], stat_p)]
            if r["st_b"] is not None:
                vals.append(pct(r["st_b"], stat_p))
            print(f"  {n:>6}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    if async_work:
        print(f"\n  {DIM}Theoretical serial: N × 1000 µs  |  Theoretical parallel: ~1000 µs flat{RESET}")
        print(f"  {DIM}Both frameworks use asyncio.gather — expect near-flat latency across widths{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")
    print(f"{BOLD}  Timbal vs CrewAI Flow — Wide Parallel Fan-out{RESET}")
    print(f"  Topology: root → [N async branches] → sink")
    print(f"  Widths: {WIDTHS}  |  {N_ITERS} iters  |  {N_BURST} burst")
    if HAS_CREWAI:
        oi_str = " + OpenInference" if HAS_OPENINFERENCE else " (OpenInference not found)"
        print(f"  CrewAI Flow found{oi_str}")
        if HAS_OPENINFERENCE:
            print(f"  {DIM}Flow steps: N @listen methods + OI → N+3 spans/kickoff (structurally equal to Timbal).{RESET}")
    else:
        print(f"  {YELLOW}CrewAI not found — Timbal numbers only{RESET}")
        print(f"  Install: uv pip install crewai openinference-instrumentation-crewai")
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")

    # ── Correctness check ──────────────────────────────────────────────────────
    print(f"\n  {DIM}Verifying correctness (N=4, trivial)...{RESET}")
    t_wf = _timbal_wide(4, async_work=False)
    t_result = await t_wf(x=5).collect()
    t_final = t_result.output
    expected = 84  # root=6, branches: 6*(2,3,4,5)=12,18,24,30, sum=84
    t_ok = t_final == expected
    msg = f"  N=4: Timbal={t_final}"
    if HAS_CREWAI and HAS_OPENINFERENCE:
        ca_steps_cls = _make_wide_listen_flow(4, async_work=False)
        ca_steps = await ca_steps_cls(suppress_flow_events=True).kickoff_async(inputs={"x": 5})
        steps_ok = ca_steps == expected
        msg += f"  Flow steps={ca_steps}"
        msg += f"  {'✓ match' if t_ok and steps_ok else '✗ MISMATCH'}"
    else:
        msg += f"  {'✓ match' if t_ok else '✗ MISMATCH'}"
    print(msg)

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
    print(f"  Timbal: one asyncio.Task + Span per branch — full per-branch observability.")
    print()
    print(f"  Flow steps: N individual @listen methods built dynamically via type(). Each")
    print(f"  branch is a first-class Flow step. OI instruments _execute_method() → one span")
    print(f"  per @start/@listen invocation = N+3 spans per kickoff (root + N branches + sink")
    print(f"  + 1 kickoff span). Structurally equivalent to Timbal. NullExporter: full span")
    print(f"  pipeline measured, no network I/O. suppress_flow_events=True on all flows.")
    print()
    print(f"  Scenario B: asyncio.sleep yields so all N branches run concurrently.")
    print(f"  Latency should be ~1 ms flat for truly parallel execution.")
    print(f"{'─' * DISPLAY_W}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
