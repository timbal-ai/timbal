#!/usr/bin/env python3
"""
Timbal vs CrewAI Flow — Double Fan-out benchmark.

Topology:  root → [N branches phase-1] → aggregator → [N branches phase-2] → sink

This topology exercises:
  1. Two full fan-out / fan-in scheduling cycles per invocation
  2. Wide parallelism at N = 16, 32, 64, 128
  3. True concurrency with 1 ms async work per branch

Total async steps per run:  2N + 3  (root, N×p1, aggregator, N×p2, sink)
At N=128: 259 async tasks per invocation.

Two columns:
  Timbal        — one asyncio.Task + Span per step; 2N+3 spans per invocation
  Flow steps    — full @listen topology via type() + OI → 2N+3 spans per kickoff;
                  structurally equivalent to Timbal's per-step model

Run:
    uv run python benchmarks/crewai/bench_double_fanout.py
    uv run python benchmarks/crewai/bench_double_fanout.py --quick
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


def fmt_kb(b: float) -> str:
    kb = b / 1024
    if kb >= 1024:
        return f"{kb / 1024:>8.1f} MB"
    return f"{kb:>8.1f} KB"


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

    for i in range(n):
        wf.step(_make_p1_branch(i, async_work), depends_on=["root"], x=_make_root_getter())

    async def aggregator(results: list) -> int:
        return sum(results)

    wf.step(
        aggregator,
        depends_on=[f"p1_{i}" for i in range(n)],
        results=_make_phase1_getter(n),
    )

    for i in range(n):
        wf.step(_make_p2_branch(i, async_work), depends_on=["aggregator"], y=_make_aggregator_getter())

    async def sink(results: list) -> int:
        return sum(results)

    wf.step(
        sink,
        depends_on=[f"p2_{i}" for i in range(n)],
        results=_make_phase2_getter(n),
    )

    return wf


def _expected(n: int) -> int:
    """Compute expected final value for x=5, N branches per phase."""
    root_out = 5 + 1  # 6
    phase1_sum = sum(root_out * (i + 2) for i in range(n))
    phase2_sum = sum(phase1_sum * (i + 2) for i in range(n))
    return phase2_sum


# ═══════════════════════════════════════════════════════════════════════════════
# CREWAI FLOW
#
# _make_double_listen_flow(n)  ("Flow steps") — full @listen topology:
#   root @start → N @listen(root) p1 branches → aggregator @listen(and_(*p1))
#   → N @listen(agg) p2 branches → sink @listen(and_(*p2)).
#   OI instruments _execute_method() → 2N+3 spans per kickoff (structurally
#   equal to Timbal). NullExporter: full span pipeline, no network I/O.
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

    # ── Flow steps (fair: 2N+3 @listen methods + OI → 2N+3 spans) ─────────────
    class _DoubleState(BaseModel):
        x: int = 0
        p1_results: list = _Field(default_factory=list)
        p2_results: list = _Field(default_factory=list)

    _FlowWithDoubleState = Flow[_DoubleState]

    def _make_double_listen_flow(n: int, async_work: bool = False):
        """Build root → N×p1 → aggregator → N×p2 → sink using @listen + type()."""
        async def root(self):
            return getattr(self.state, "x", 0) + 1
        root_m = start()(root)
        methods: dict = {"root": root_m, "__module__": __name__, "__qualname__": f"_DoubleListenFlow_{n}"}

        # Phase-1 branches
        for i in range(n):
            async def p1_fn(self, root_val, i=i):
                if async_work:
                    await asyncio.sleep(0.001)
                result = root_val * (i + 2)
                self.state.p1_results.append(result)
                return result
            p1_fn.__name__ = f"p1_{i}"
            methods[f"p1_{i}"] = listen(root_m)(p1_fn)

        p1_methods = [methods[f"p1_{i}"] for i in range(n)]

        async def aggregator(self, _=None):
            return sum(self.state.p1_results)
        agg_m = listen(and_(*p1_methods))(aggregator)
        methods["aggregator"] = agg_m

        # Phase-2 branches
        for i in range(n):
            async def p2_fn(self, agg_val, i=i):
                if async_work:
                    await asyncio.sleep(0.001)
                result = agg_val * (i + 2)
                self.state.p2_results.append(result)
                return result
            p2_fn.__name__ = f"p2_{i}"
            methods[f"p2_{i}"] = listen(agg_m)(p2_fn)

        p2_methods = [methods[f"p2_{i}"] for i in range(n)]

        async def sink(self, _=None):
            return sum(self.state.p2_results)
        methods["sink"] = listen(and_(*p2_methods))(sink)

        return type(f"_DoubleListenFlow_{n}", (_FlowWithDoubleState,), methods)

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


async def _seq_peak_memory(factory, n: int, warmup: int) -> int:
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

    cols = ["Timbal"]
    if HAS_CREWAI and HAS_OPENINFERENCE:
        cols.append("Flow steps")
    col_w = 14

    instances: dict[int, dict] = {}
    for n in WIDTHS:
        t_wf = _timbal_double(n, async_work)
        steps_cls = _make_double_listen_flow(n, async_work) if HAS_CREWAI and HAS_OPENINFERENCE else None
        instances[n] = {
            "t_run": lambda _w=t_wf: _w(x=5).collect(),
            "steps_run": (lambda _cls=steps_cls: _cls(suppress_flow_events=True).kickoff_async(inputs={"x": 5}))
                if steps_cls is not None else None,
        }

    data: dict[int, dict] = {}
    for n in WIDTHS:
        print(f"  {DIM}Benchmarking N={n} ({2*n+3} steps)...{RESET}", flush=True)
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

    hdr = f"  {'N':>6}  {'steps':>7}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = f"  {'─' * 6}  {'─' * 7}" + f"  {'─' * col_w}" * len(cols)

    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Latency {stat_label}  (×{N_ITERS} sequential runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_s"], stat_p)]
            if r["st_s"] is not None:
                vals.append(pct(r["st_s"], stat_p))
            print(f"  {n:>6}  {2*n+3:>7}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    subsection("Overhead per extra branch-pair  (slope of p50 vs N, 2 branches per unit)")
    print(f"  {DIM}Each N increment adds 2 async branches (one per phase){RESET}")
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
            print(f"  {label_s:<14}  {slope:>+.1f} µs / branch-pair  ({slope / 2:>+.1f} µs per branch)")

    for stat_label, stat_p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        subsection(f"Burst {stat_label}  ({N_BURST} concurrent runs)")
        print(hdr)
        print(sep)
        for n in WIDTHS:
            r = data[n]
            vals = [pct(r["t_b"], stat_p)]
            if r["st_b"] is not None:
                vals.append(pct(r["st_b"], stat_p))
            print(f"  {n:>6}  {2*n+3:>7}" + "".join(f"  {fmt_us(v):>{col_w}}" for v in vals))

    if async_work:
        print(f"\n  {DIM}Each phase: theoretical serial = N×1000 µs, parallel ≈ 1000 µs flat{RESET}")
        print(f"  {DIM}Two phases: expect ~2000 µs floor + scheduling overhead{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Memory section
# ═══════════════════════════════════════════════════════════════════════════════


async def run_memory() -> None:
    section("Memory — InMemoryTracingProvider vs tracing_provider=None vs Flow steps")

    cols = ["T (InMemory)", "T (no trace)"] + \
           (["Flow steps"] if HAS_CREWAI and HAS_OPENINFERENCE else [])
    col_w = 14

    N_MEM_SEQ = 20 if _args.quick else 100
    N_MEM_BURST = N_BURST

    print(f"  {DIM}Sequential: {N_MEM_SEQ} runs, cleared between each  |  Burst: {N_MEM_BURST} concurrent{RESET}")

    hdr = f"  {'N':>6}  {'steps':>7}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = f"  {'─' * 6}  {'─' * 7}" + f"  {'─' * col_w}" * len(cols)

    for sub, n_runs, harness_label in [
        ("Sequential peak  (per-run, cleared between)", N_MEM_SEQ, "seq"),
        (f"Burst peak  ({N_MEM_BURST} concurrent, no GC between)", N_MEM_BURST, "burst"),
    ]:
        subsection(sub)
        print(hdr)
        print(sep)
        for n in WIDTHS:
            t_inmem_wf = _timbal_double(n, async_work=False, tracing_provider=InMemoryTracingProvider)
            t_none_wf = _timbal_double(n, async_work=False, tracing_provider=None)

            t_inmem_run = lambda _w=t_inmem_wf: _w(x=5).collect()
            t_none_run = lambda _w=t_none_wf: _w(x=5).collect()

            if harness_label == "seq":
                t_im = await _seq_peak_memory(t_inmem_run, n_runs, warmup=3)
                t_no = await _seq_peak_memory(t_none_run, n_runs, warmup=3)
            else:
                t_im = await _burst_peak_memory(t_inmem_run, n_runs)
                t_no = await _burst_peak_memory(t_none_run, n_runs)

            vals = [t_im, t_no]
            if HAS_CREWAI and HAS_OPENINFERENCE:
                st_cls = _make_double_listen_flow(n, async_work=False)
                st_run = lambda _cls=st_cls: _cls(suppress_flow_events=True).kickoff_async(inputs={"x": 5})
                _oi_instrument()
                if harness_label == "seq":
                    st_m = await _seq_peak_memory(st_run, n_runs, warmup=3)
                else:
                    st_m = await _burst_peak_memory(st_run, n_runs)
                _oi_uninstrument()
                vals.append(st_m)

            print(f"  {n:>6}  {2*n+3:>7}" + "".join(f"  {fmt_kb(v):>{col_w}}" for v in vals))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")
    print(f"{BOLD}  Timbal vs CrewAI Flow — Double Fan-out{RESET}")
    print(f"  Topology: root → [N×p1] → aggregator → [N×p2] → sink")
    print(f"  Widths: {WIDTHS}  |  {N_ITERS} iters  |  {N_BURST} burst")
    if HAS_CREWAI:
        oi_str = " + OpenInference" if HAS_OPENINFERENCE else " (OpenInference not found)"
        print(f"  CrewAI Flow found{oi_str}")
        if HAS_OPENINFERENCE:
            print(f"  {DIM}Flow steps: full @listen topology + OI → 2N+3 spans/kickoff (structurally equal to Timbal).{RESET}")
    else:
        print(f"  {YELLOW}CrewAI not found — Timbal numbers only{RESET}")
        print(f"  Install: uv pip install crewai openinference-instrumentation-crewai")
    print(f"{BOLD}{'═' * DISPLAY_W}{RESET}")

    # ── Correctness check ──────────────────────────────────────────────────────
    print(f"\n  {DIM}Verifying correctness (N=4, trivial)...{RESET}")
    expected = _expected(4)
    t_wf = _timbal_double(4, async_work=False)
    t_result = await t_wf(x=5).collect()
    t_final = t_result.output
    t_ok = t_final == expected
    msg = f"  N=4: expected={expected}  Timbal={t_final} {'✓' if t_ok else '✗'}"
    if HAS_CREWAI and HAS_OPENINFERENCE:
        st_cls = _make_double_listen_flow(4, async_work=False)
        st_result = await st_cls(suppress_flow_events=True).kickoff_async(inputs={"x": 5})
        st_ok = st_result == expected
        msg += f"  Flow steps={st_result} {'✓' if st_ok else '✗'}"
    print(msg)
    if not t_ok:
        print(f"  WARNING: Timbal output differs from expected")

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
    print(f"  Topology: root → [N×p1] → aggregator → [N×p2] → sink")
    print(f"  Total steps: 2N + 3  |  Two full fan-out / fan-in cycles per invocation")
    print()
    print(f"  Timbal: one asyncio.Task + Span per step — full per-branch observability.")
    print()
    print(f"  Flow steps: full @listen topology (root + N×p1 + aggregator + N×p2 + sink)")
    print(f"  built dynamically via type(). OI instruments _execute_method() → 2N+3 spans")
    print(f"  per kickoff. Structurally equivalent to Timbal. NullExporter: full span")
    print(f"  pipeline measured, no network I/O. suppress_flow_events=True on all flows.")
    print()
    print(f"  Scenario B: parallelism floor is ~2 ms (two sequential 1 ms gather phases).")
    print(f"  Latency growing beyond 2 ms means the framework is serialising branches.")
    print(f"{'─' * DISPLAY_W}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
