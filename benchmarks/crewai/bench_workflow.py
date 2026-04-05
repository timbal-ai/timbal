#!/usr/bin/env python3
"""
Timbal Workflow vs CrewAI Flow — DAG benchmark.

Three scenarios with trivial handler functions (no LLM calls):
  1. Sequential:    A → B → C → D
  2. Fan-out/in:    A → [B, C, D] → E
  3. Diamond:       A → [B, C] → D

Timbal uses Workflow with .step() chaining and get_run_context().
CrewAI uses Flow with @start / @listen decorators and and_() for fan-in.
Both support pure Python functions — no LLM required for either.

Fan-out note: CrewAI Flow fires multiple @listen(step_a) methods in parallel
via asyncio.gather internally. and_() waits for all before running the
aggregator, using shared self.state to pass branch results.

Columns:
  Timbal      — built-in InMemory tracing always on
  Flow bare   — CrewAI Flow, no observability
  Flow+OI     — CrewAI Flow + OpenInference (openinference-instrumentation-crewai)
                spans created + serialized, HTTP export mocked (NullExporter)

Collection order: Timbal → Flow bare → activate OI → Flow+OI → deactivate OI
OpenInference can be uninstrumented (standard OTel wrapt pattern), so we
instrument/uninstrument per scenario to keep bare and +OI runs cleanly separated.

Metrics: latency (mean/p50/p95/p99), memory, burst, throughput.

Run:
    uv run python benchmarks/crewai/bench_workflow.py
    uv run python benchmarks/crewai/bench_workflow.py --quick
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
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
N_BURST = 50 if _args.quick else 200
N_MEM = 50 if _args.quick else 200
THROUGHPUT_OPS = 100 if _args.quick else 500
CONCURRENCY_LEVELS = [1, 10, 50, 200]
WIDTH = 88

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


def _clear_timbal():
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
# CREWAI — Flow factories
# ═══════════════════════════════════════════════════════════════════════════════

HAS_CREWAI = False

try:
    from crewai.flow.flow import Flow, and_, listen, start  # noqa: E402

    class _CASequentialFlow(Flow):
        """A → B → C → D — pure Python, no LLM."""

        suppress_flow_events: bool = True

        @start()
        def step_a(self) -> int:
            return self.state["x"] + 1

        @listen(step_a)
        def step_b(self, val: int) -> int:
            return val * 2

        @listen(step_b)
        def step_c(self, val: int) -> int:
            return val + 10

        @listen(step_c)
        def step_d(self, val: int) -> int:
            return val - 3

    class _CAFanoutFlow(Flow):
        """A → [B, C, D] → E — branches fire in parallel via asyncio.gather."""

        suppress_flow_events: bool = True

        @start()
        def step_a(self) -> int:
            val = self.state["x"] + 1
            self.state["a"] = val
            return val

        @listen(step_a)
        def branch_b(self, val: int) -> int:
            self.state["b"] = val * 2
            return self.state["b"]

        @listen(step_a)
        def branch_c(self, val: int) -> int:
            self.state["c"] = val * 3
            return self.state["c"]

        @listen(step_a)
        def branch_d(self, val: int) -> int:
            self.state["d"] = val * 4
            return self.state["d"]

        @listen(and_(branch_b, branch_c, branch_d))
        def step_e(self) -> int:
            return self.state["b"] + self.state["c"] + self.state["d"]

    class _CADiamondFlow(Flow):
        """A → [B, C] → D — B and C fire in parallel, D waits for both."""

        suppress_flow_events: bool = True

        @start()
        def step_a(self) -> int:
            val = self.state["x"] + 1
            self.state["a"] = val
            return val

        @listen(step_a)
        def path_b(self, val: int) -> int:
            self.state["b"] = val + 10
            return self.state["b"]

        @listen(step_a)
        def path_c(self, val: int) -> int:
            self.state["c"] = val * 5
            return self.state["c"]

        @listen(and_(path_b, path_c))
        def combine(self) -> int:
            return self.state["b"] + self.state["c"]

    HAS_CREWAI = True

except ImportError as e:
    print(f"CrewAI not found: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# OPENINFERENCE — Flow instrumentation (openinference-instrumentation-crewai)
#
# Patches Flow._execute_method(), Flow.kickoff(), Flow.kickoff_async() via wrapt.
# Creates one OTel span per @start/@listen method invocation — full span lifecycle
# (creation, attribute serialization, processor pipeline) measured. Export is
# mocked via NullExporter so no network I/O is included.
#
# suppress_flow_events=True on Flow classes suppresses CrewAI's own event bus —
# not OI. OI patches _execute_method directly and fires regardless.
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
        """Accepts spans through the full OTel pipeline (creation + serialization)
        but discards on export — equivalent to mocking LangSmith's _persist_run_single.
        Hot-path overhead (span creation, attribute setting, processor) is measured;
        network I/O is not."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement harnesses
# ═══════════════════════════════════════════════════════════════════════════════


async def _latency_async(factory, n, warmup) -> list[float]:
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


async def _memory_async(factory, n, warmup) -> tuple[float, float]:
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


async def _burst_async(factory, n) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(10, n))])
    _clear_timbal()
    gc.collect()
    samples: list[float] = []

    async def timed():
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    await asyncio.gather(*[timed() for _ in range(n)])
    _clear_timbal()
    return sorted(samples)


async def _throughput_async(factory, total, conc) -> float:
    sem = asyncio.Semaphore(conc)

    async def bounded():
        async with sem:
            await factory()

    t0 = time.perf_counter()
    await asyncio.gather(*[bounded() for _ in range(total)])
    elapsed = time.perf_counter() - t0
    _clear_timbal()
    return total / elapsed


async def _collect_all(run_fn) -> dict:
    """Collect latency, memory, burst, and throughput for a single runner."""
    lat = await _latency_async(run_fn, N_ITERS, N_WARMUP)
    mem_peak, mem_per = await _memory_async(run_fn, N_MEM, N_WARMUP)
    burst = await _burst_async(run_fn, N_BURST)
    tp = [await _throughput_async(run_fn, THROUGHPUT_OPS, c) for c in CONCURRENCY_LEVELS]
    return {"lat": lat, "mem_peak": mem_peak, "mem_per": mem_per, "burst": burst, "tp": tp}


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

    if scenario == 1:
        t_wf = _timbal_sequential()
        ca_cls = _CASequentialFlow if HAS_CREWAI else None
    elif scenario == 2:
        t_wf = _timbal_fanout()
        ca_cls = _CAFanoutFlow if HAS_CREWAI else None
    else:
        t_wf = _timbal_diamond()
        ca_cls = _CADiamondFlow if HAS_CREWAI else None

    async def t_run():
        await t_wf(x=5).collect()

    async def ca_run():
        await ca_cls().kickoff_async(inputs={"x": 5})

    # ── Pass 1: Timbal ────────────────────────────────────────────────────────
    print(f"  {DIM}Collecting Timbal...{RESET}", flush=True)
    t_d = await _collect_all(t_run)

    # ── Pass 2: Flow bare ─────────────────────────────────────────────────────
    ca_d = None
    if ca_cls:
        print(f"  {DIM}Collecting Flow bare...{RESET}", flush=True)
        ca_d = await _collect_all(ca_run)

    # ── Pass 3: Flow+OI ───────────────────────────────────────────────────────
    # OI patches Flow._execute_method() via wrapt. instrument() wraps, uninstrument()
    # unwraps — safe to call per-scenario. Each @start/@listen invocation generates
    # one OTel span through the full pipeline (creation → attributes → NullExporter).
    oi_d = None
    if ca_cls and HAS_OPENINFERENCE:
        print(f"  {DIM}Collecting Flow+OI...{RESET}", flush=True)
        _oi_instrument()
        oi_d = await _collect_all(ca_run)
        _oi_uninstrument()

    # ── Build column list ─────────────────────────────────────────────────────
    col_w = 14
    cols: list[tuple[str, dict]] = [("Timbal", t_d)]
    if ca_d:
        cols.append(("Flow bare", ca_d))
    if oi_d:
        cols.append(("Flow+OI", oi_d))

    hdr = f"  {'':>12}" + "".join(f"  {c:>{col_w}}" for c, _ in cols)
    sep = f"  {'─' * 12}" + f"  {'─' * col_w}" * len(cols)

    # ── Latency ──────────────────────────────────────────────────────────────
    subsection(f"Latency  (×{N_ITERS})")
    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = []
        for _, d in cols:
            v = statistics.mean(d["lat"]) if p is None else pct(d["lat"], p)
            vals.append(fmt_us(v))
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    # ── Memory ───────────────────────────────────────────────────────────────
    subsection(f"Memory  (×{N_MEM} runs)")
    fw_w = 24
    print(f"  {'framework':<{fw_w}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * fw_w}  {'─' * 12}  {'─' * 12}")
    for name, d in cols:
        print(f"  {name:<{fw_w}}  {d['mem_peak']/1024:>10.1f} KB  {d['mem_per']:>10.0f}  B")

    # ── Burst ─────────────────────────────────────────────────────────────────
    subsection(f"Burst  ({N_BURST} concurrent)")
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d["burst"], p)) for _, d in cols]
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))
    wall_parts = [f"{name}: {max(d['burst'])/1000:.1f} ms" for name, d in cols]
    print(f"\n  {DIM}wall: {' | '.join(wall_parts)}{RESET}")

    # ── Throughput ────────────────────────────────────────────────────────────
    subsection(f"Throughput  ({THROUGHPUT_OPS} runs)")
    print(hdr)
    print(sep)
    for i, conc in enumerate(CONCURRENCY_LEVELS):
        vals = [f"{d['tp'][i]:>10.0f}/s" for _, d in cols]
        print(f"  {conc:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal Workflow vs CrewAI Flow — DAG benchmark{RESET}")
    print(f"  {N_ITERS} iters · {N_BURST} burst · {N_MEM} mem · {THROUGHPUT_OPS} throughput ops")
    if HAS_CREWAI:
        oi_str = " + OpenInference" if HAS_OPENINFERENCE else " (OpenInference not found)"
        print(f"  CrewAI Flow found{oi_str}")
    else:
        print(f"  {YELLOW}CrewAI not found — Timbal numbers only{RESET}")
        print(f"  Install: uv pip install crewai openinference-instrumentation-crewai")
    print(f"  {DIM}Timbal: built-in InMemory tracing always on.{RESET}")
    if HAS_OPENINFERENCE:
        print(f"  {DIM}Flow+OI: spans created + serialized via NullExporter (HTTP mocked).{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    # Verify correctness
    print(f"\n  {DIM}Verifying correctness...{RESET}")
    for name, t_factory, ca_flow_cls, inp, expected in [
        ("sequential", _timbal_sequential, _CASequentialFlow if HAS_CREWAI else None, 5, 19),
        ("fanout",     _timbal_fanout,     _CAFanoutFlow     if HAS_CREWAI else None, 5, 54),
        ("diamond",    _timbal_diamond,    _CADiamondFlow    if HAS_CREWAI else None, 5, 46),
    ]:
        t_wf = t_factory()
        t_result = await t_wf(x=inp).collect()
        t_ok = t_result.output == expected
        msg = f"  {name}: Timbal={t_result.output}"
        if ca_flow_cls is not None:
            ca_result = await ca_flow_cls().kickoff_async(inputs={"x": inp})
            ca_ok = ca_result == expected
            msg += f"  Flow={ca_result}  {'✓' if t_ok and ca_ok else '✗ MISMATCH'}"
        else:
            msg += f"  {'✓' if t_ok else '✗ MISMATCH'}"
        print(msg)

    for scenario in [1, 2, 3]:
        await bench_scenario(scenario)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  Timbal: built-in InMemory tracing on every run — spans recorded per step.")
    print(f"  Flow bare: @start/@listen dispatched via asyncio.gather; no instrumentation.")
    print(f"  Flow+OI: OpenInference patches Flow._execute_method() — one OTel span per")
    print(f"  @start/@listen invocation. NullExporter: span creation + serialization cost")
    print(f"  measured, export discarded (no network I/O). suppress_flow_events=True on")
    print(f"  all Flow classes — CrewAI's own event bus suppressed, OI patches directly.")
    print(f"  Graph construction / Flow class definition excluded from timing.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
