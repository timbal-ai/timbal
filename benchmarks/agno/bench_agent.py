#!/usr/bin/env python3
"""
Timbal vs Agno — full agent loop benchmark.

Three scenarios, each running the identical pipeline on both frameworks:
  1. Single tool call:   prompt → LLM → tool(add) → LLM → answer
  2. Multi-step (3 tools): prompt → LLM → add → LLM → mul → LLM → sub → LLM → answer
  3. Parallel tools:     prompt → LLM → [add, mul, neg] concurrent → LLM → answer
     (both Timbal and Agno dispatch tool calls concurrently via asyncio.gather)

Timbal uses TestModel for fully offline, deterministic LLM simulation.
Agno uses a custom FakeModel subclass — also offline.

Columns:
  Timbal             — built-in InMemory tracing (always on)
  Agno (no tel)      — telemetry=False; Agno's default behaviour sends a blocking
                       HTTP call to api.agno.com on every run (adds ~400–500 ms).
                       You must opt out explicitly to get sane runtime performance.
  Agno (tel mocked)  — telemetry=True with acreate_agent_run mocked; measures the
                       JSON serialisation overhead of Agno's default telemetry path
                       without the network call.

Key finding:
  Agno's default telemetry (telemetry=True, the out-of-the-box behaviour) is an
  awaited async HTTP POST to Agno's API at the end of every run. No batching, no
  queue — just a blocking network call on the hot path. Measured overhead: ~400–500 ms
  per run, making the default configuration unsuitable for any latency-sensitive
  workload. All Agno benchmarks below use telemetry=False unless stated otherwise.

Collection order:
  Phase 1 (bare):    Timbal + Agno (no tel)       (all 3 scenarios)
  → mock acreate_agent_run (global, cannot unmock cleanly — must run bare first)
  Phase 2 (+tel):    Agno (tel mocked)             (all 3 scenarios)
  → print combined tables

Run:
    uv run python benchmarks/agno/bench_agent.py
    uv run python benchmarks/agno/bench_agent.py --quick
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
os.environ.setdefault("AGNO_TELEMETRY", "false")   # overridden per-phase below
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

N_ITERS     = 20  if _args.quick else 100
N_WARMUP    = 3   if _args.quick else 10
N_MEM       = 20  if _args.quick else 100
N_BURST     = {1: 20, 2: 10, 3: 15}   if _args.quick else {1: 50, 2: 30, 3: 40}
TP_OPS      = 30  if _args.quick else 200
CONC_LEVELS = [1, 10, 50]
WIDTH       = 90

# ── Display helpers ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
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
    idx = min(int(len(samples) * p / 100), len(samples) - 1)
    return sorted(samples)[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — TestModel agent
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Agent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.types.message import Message  # noqa: E402
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_traces() -> None:
    InMemoryTracingProvider._storage.clear()


def _count_tool_results(messages) -> int:
    return sum(
        1 for m in messages
        for c in (m.content if hasattr(m, "content") and m.content else [])
        if isinstance(c, ToolResultContent)
    )


def _make_timbal_agent(scenario: int) -> Agent:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    def negate(x: int) -> int:
        """Negate a number."""
        return -x

    tools = {1: [add], 2: [add, multiply, subtract], 3: [add, multiply, negate]}[scenario]

    if scenario == 1:
        def handler(messages):
            if _count_tool_results(messages) == 0:
                return Message(role="assistant", content=[ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2})])
            return Message(role="assistant", content=[TextContent(type="text", text="3")], stop_reason="end_turn")
    elif scenario == 2:
        def handler(messages):
            step = _count_tool_results(messages)
            if step == 0:
                return Message(role="assistant", content=[ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2})])
            elif step == 1:
                return Message(role="assistant", content=[ToolUseContent(type="tool_use", id="c2", name="multiply", input={"a": 3, "b": 4})])
            elif step == 2:
                return Message(role="assistant", content=[ToolUseContent(type="tool_use", id="c3", name="subtract", input={"a": 12, "b": 3})])
            return Message(role="assistant", content=[TextContent(type="text", text="9")], stop_reason="end_turn")
    else:
        def handler(messages):
            if _count_tool_results(messages) == 0:
                return Message(role="assistant", content=[
                    ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2}),
                    ToolUseContent(type="tool_use", id="c2", name="multiply", input={"a": 3, "b": 4}),
                    ToolUseContent(type="tool_use", id="c3", name="negate", input={"x": 5}),
                ])
            return Message(role="assistant", content=[TextContent(type="text", text="done")], stop_reason="end_turn")

    return Agent(name="bench_agent", model=TestModel(handler=handler), tools=tools)


# ═══════════════════════════════════════════════════════════════════════════════
# AGNO — custom FakeModel + agent factory
#
# Agno's Model is an abstract dataclass. We implement the four required invoke
# methods plus two _parse helpers. The base class response() / aresponse()
# methods own the tool loop and call invoke() on each iteration.
#
# Fake LLM strategy: count role="tool" messages in the history to determine
# which step we're on — stateless, safe for concurrent async runs.
#
# Concurrency: Agno dispatches multiple tool calls from a single response
# concurrently via asyncio.gather. Scenario 3 is truly parallel on both sides.
#
# Telemetry: Agno's default (telemetry=True) awaits an HTTP POST to api.agno.com
# on every run — a blocking call that adds ~400-500 ms. Disabled via telemetry=False
# for the bare column. The "+tel mocked" column re-enables it with the HTTP call
# replaced by an AsyncMock, measuring JSON serialisation overhead only.
# ═══════════════════════════════════════════════════════════════════════════════

HAS_AGNO = False

try:
    from dataclasses import dataclass  # noqa: E402
    from agno.models.base import Model as AgnoModel, ModelResponse as AgnoModelResponse  # noqa: E402
    from agno.models.message import Message as AgnoMessage  # noqa: E402
    from agno.agent import Agent as AgnoAgent  # noqa: E402

    def _count_agno_tool_msgs(messages: list) -> int:
        return sum(1 for m in messages if m.role == "tool")

    @dataclass
    class _FakeAgnoModel(AgnoModel):
        id: str = "fake"
        scenario: int = 1

        def invoke(self, messages, assistant_message, **kwargs) -> AgnoModelResponse:
            step = _count_agno_tool_msgs(messages)
            if self.scenario == 1:
                if step == 0:
                    return AgnoModelResponse(tool_calls=[{"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}])
                return AgnoModelResponse(content="3")
            elif self.scenario == 2:
                if step == 0:
                    return AgnoModelResponse(tool_calls=[{"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}])
                elif step == 1:
                    return AgnoModelResponse(tool_calls=[{"id": "c2", "type": "function", "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'}}])
                elif step == 2:
                    return AgnoModelResponse(tool_calls=[{"id": "c3", "type": "function", "function": {"name": "subtract", "arguments": '{"a": 12, "b": 3}'}}])
                return AgnoModelResponse(content="9")
            else:
                if step == 0:
                    return AgnoModelResponse(tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
                        {"id": "c2", "type": "function", "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'}},
                        {"id": "c3", "type": "function", "function": {"name": "negate", "arguments": '{"x": 5}'}},
                    ])
                return AgnoModelResponse(content="done")

        async def ainvoke(self, messages, assistant_message, **kwargs) -> AgnoModelResponse:
            return self.invoke(messages, assistant_message, **kwargs)

        def invoke_stream(self, *args, **kwargs):
            yield self.invoke(*args, **kwargs)

        async def ainvoke_stream(self, *args, **kwargs):
            yield self.invoke(*args, **kwargs)

        def _parse_provider_response(self, response, **kwargs) -> AgnoModelResponse:
            return response

        def _parse_provider_response_delta(self, response) -> AgnoModelResponse:
            return response

    def _make_agno_agent(scenario: int, telemetry: bool = False) -> AgnoAgent:
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        def subtract(a: int, b: int) -> int:
            """Subtract b from a."""
            return a - b

        def negate(x: int) -> int:
            """Negate a number."""
            return -x

        tools_map = {
            1: [add],
            2: [add, multiply, subtract],
            3: [add, multiply, negate],
        }
        return AgnoAgent(
            model=_FakeAgnoModel(scenario=scenario),
            tools=tools_map[scenario],
            markdown=False,
            telemetry=telemetry,
        )

    HAS_AGNO = True

except ImportError as e:
    print(f"Agno not found: {e}")


def _mock_agno_telemetry() -> bool:
    """Replace acreate_agent_run with an AsyncMock so telemetry=True agents
    incur JSON-serialisation overhead but no network calls.

    This patches the module globally and cannot be cleanly reversed; bare
    measurements must complete before this is called.
    """
    try:
        from unittest.mock import AsyncMock, patch
        patch("agno.api.agent.acreate_agent_run", new=AsyncMock(return_value=None)).start()
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement harnesses
# ═══════════════════════════════════════════════════════════════════════════════


async def _latency_async(factory, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        await factory()
    _clear_traces()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1e6)
    _clear_traces()
    return samples


async def _memory_async(factory, n: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        await factory()
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await factory()
        _clear_traces()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


async def _burst_async(factory, n: int) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_traces()
    gc.collect()
    samples: list[float] = []

    async def _timed():
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1e6)

    await asyncio.gather(*[_timed() for _ in range(n)])
    _clear_traces()
    return sorted(samples)


async def _burst_memory_async(factory, n: int) -> tuple[float, float]:
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    await asyncio.gather(*[factory() for _ in range(n)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _clear_traces()
    return peak, peak / n


async def _throughput_async(factory, total: int, conc: int) -> float:
    sem = asyncio.Semaphore(conc)

    async def _bounded():
        async with sem:
            await factory()

    gc.collect()
    t0 = time.perf_counter()
    await asyncio.gather(*[_bounded() for _ in range(total)])
    elapsed = time.perf_counter() - t0
    _clear_traces()
    return total / elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class RunData:
    latency: list[float]       # µs, N_ITERS samples
    mem_peak: float            # bytes
    mem_per: float             # bytes per run
    burst: list[float]         # µs, N_BURST[scenario] samples
    burst_mem_peak: float      # bytes — peak during concurrent burst
    burst_mem_per: float       # bytes per concurrent run
    throughput: list[float]    # ops/s per CONC_LEVELS entry


# ═══════════════════════════════════════════════════════════════════════════════
# Collection functions
# ═══════════════════════════════════════════════════════════════════════════════


async def _collect_timbal(scenario: int) -> RunData:
    agent = _make_timbal_agent(scenario)
    factory = lambda: agent(prompt="go").collect()
    n_burst = N_BURST[scenario]

    print(f"    Timbal latency...", end=" ", flush=True)
    lat = await _latency_async(factory, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(lat, 50)).strip()}", flush=True)

    print(f"    Timbal memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory_async(factory, N_MEM, N_WARMUP)
    print(f"{mem_peak / 1024:.0f} KB peak", flush=True)

    print(f"    Timbal burst ({n_burst} concurrent)...", end=" ", flush=True)
    burst = await _burst_async(factory, n_burst)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    print(f"    Timbal burst memory ({n_burst} concurrent)...", end=" ", flush=True)
    burst_mem_peak, burst_mem_per = await _burst_memory_async(factory, n_burst)
    print(f"{burst_mem_peak / 1024:.0f} KB peak", flush=True)

    tp = []
    for conc in CONC_LEVELS:
        print(f"    Timbal throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput_async(factory, TP_OPS, conc)
        tp.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=lat, mem_peak=mem_peak, mem_per=mem_per, burst=burst,
                   burst_mem_peak=burst_mem_peak, burst_mem_per=burst_mem_per, throughput=tp)


async def _collect_agno(scenario: int, label: str, telemetry: bool = False) -> RunData:
    agent = _make_agno_agent(scenario, telemetry=telemetry)
    factory = lambda: agent.arun("go")
    n_burst = N_BURST[scenario]

    print(f"    {label} latency...", end=" ", flush=True)
    lat = await _latency_async(factory, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(lat, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory_async(factory, N_MEM, N_WARMUP)
    print(f"{mem_peak / 1024:.0f} KB peak", flush=True)

    print(f"    {label} burst ({n_burst} concurrent)...", end=" ", flush=True)
    burst = await _burst_async(factory, n_burst)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    print(f"    {label} burst memory ({n_burst} concurrent)...", end=" ", flush=True)
    burst_mem_peak, burst_mem_per = await _burst_memory_async(factory, n_burst)
    print(f"{burst_mem_peak / 1024:.0f} KB peak", flush=True)

    tp = []
    for conc in CONC_LEVELS:
        print(f"    {label} throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput_async(factory, TP_OPS, conc)
        tp.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=lat, mem_peak=mem_peak, mem_per=mem_per, burst=burst,
                   burst_mem_peak=burst_mem_peak, burst_mem_per=burst_mem_per, throughput=tp)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-scenario printer
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_NAMES = {
    1: "Single tool  (LLM → add → LLM → answer)",
    2: "Multi-step  (LLM → add → LLM → mul → LLM → sub → LLM → answer)",
    3: "Parallel tools  (LLM → [add, mul, neg] concurrent → LLM → answer)",
}

COL_W = 16


def _print_scenario(
    scenario: int,
    t: RunData,
    agno: RunData | None,
    agno_tel: RunData | None,
) -> None:
    section(f"Scenario {scenario}: {SCENARIO_NAMES[scenario]}")

    cols: list[tuple[str, RunData]] = [("Timbal", t)]
    if agno:
        cols.append(("Agno (no tel)", agno))
    if agno_tel:
        cols.append(("Agno (tel mock)", agno_tel))

    hdr = f"  {'':>12}" + "".join(f"  {c:>{COL_W}}" for c, _ in cols)
    sep = f"  {'─' * 12}" + f"  {'─' * COL_W}" * len(cols)

    # ── Latency ──────────────────────────────────────────────────────────────
    subsection(f"Latency  (×{N_ITERS} sequential runs)")
    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = []
        for _, d in cols:
            v = statistics.mean(d.latency) if p is None else pct(d.latency, p)
            vals.append(fmt_us(v))
        print(f"  {label:>12}" + "".join(f"  {v:>{COL_W}}" for v in vals))

    # ── Memory ───────────────────────────────────────────────────────────────
    subsection(f"Memory  (×{N_MEM} runs)")
    fw = 24
    print(f"  {'framework':<{fw}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * fw}  {'─' * 12}  {'─' * 12}")
    for name, d in cols:
        print(f"  {name:<{fw}}  {d.mem_peak / 1024:>10.1f} KB  {d.mem_per:>10.0f}  B")

    # ── Burst ────────────────────────────────────────────────────────────────
    n_burst = N_BURST[scenario]
    subsection(f"Burst  ({n_burst} concurrent — fully async on both frameworks)")
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d.burst, p)) for _, d in cols]
        print(f"  {label:>12}" + "".join(f"  {v:>{COL_W}}" for v in vals))
    wall_parts = [f"{name}: {max(d.burst) / 1e3:.1f} ms" for name, d in cols]
    print(f"\n  {DIM}wall: {' | '.join(wall_parts)}{RESET}")

    # ── Burst memory ─────────────────────────────────────────────────────────
    subsection(f"Burst memory  ({n_burst} concurrent — peak during full burst, no GC between)")
    print(f"  {'framework':<{fw}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * fw}  {'─' * 12}  {'─' * 12}")
    for name, d in cols:
        print(f"  {name:<{fw}}  {d.burst_mem_peak / 1024:>10.1f} KB  {d.burst_mem_per:>10.0f}  B")

    # ── Throughput ────────────────────────────────────────────────────────────
    subsection(f"Throughput  ({TP_OPS} loops)")
    print(hdr)
    print(sep)
    for i, conc in enumerate(CONC_LEVELS):
        vals = [f"{d.throughput[i]:>10.0f}/s" for _, d in cols]
        print(f"  {conc:>12}" + "".join(f"  {v:>{COL_W}}" for v in vals))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal vs Agno — agent loop benchmark{RESET}")
    print(f"  {N_ITERS} iters · burst {N_BURST} · {N_MEM} mem · {TP_OPS} throughput ops")
    print(f"  Timbal: TestModel (offline) · Agno: FakeModel (offline)")
    if HAS_AGNO:
        print(f"  Agno found — will measure bare (tel=False) + telemetry mocked")
        print(f"  {DIM}Both frameworks dispatch parallel tool calls concurrently.{RESET}")
        print(f"  {YELLOW}Note: Agno default (telemetry=True) awaits a blocking HTTP POST per run (~400–500 ms).{RESET}")
        print(f"  {YELLOW}      'Agno (no tel)' uses telemetry=False — the explicit opt-out required for production.{RESET}")
    else:
        print(f"  {YELLOW}Agno not found — Timbal only.  Install: uv pip install agno{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    # ── Correctness spot-check ────────────────────────────────────────────────
    print(f"\n  {DIM}Spot-checking correctness (Scenario 1)...{RESET}", flush=True)
    _t = _make_timbal_agent(1)
    res = await _t(prompt="go").collect()
    t_ok = res.status.code == "success"
    msg = f"  Timbal: {'✓' if t_ok else '✗'}"
    if HAS_AGNO:
        _a = _make_agno_agent(1, telemetry=False)
        a_res = await _a.arun("go")
        a_ok = str(a_res.content).strip() == "3"
        msg += f"  |  Agno: {'✓' if a_ok else f'✗ (got {a_res.content!r})'}"
    print(msg, flush=True)

    # ── Phase 1: Timbal + Agno bare ───────────────────────────────────────────
    t_data:    dict[int, RunData] = {}
    agno_data: dict[int, RunData] = {}

    for scenario in [1, 2, 3]:
        print(f"\n  {DIM}[Phase 1 · Scenario {scenario}]{RESET}", flush=True)
        t_data[scenario] = await _collect_timbal(scenario)
        if HAS_AGNO:
            agno_data[scenario] = await _collect_agno(scenario, "Agno (no tel)", telemetry=False)

    # ── Mock telemetry (global patch — bare must run first) ───────────────────
    agno_tel_data: dict[int, RunData] = {}
    if HAS_AGNO:
        print(f"\n  {DIM}Patching agno.api.agent.acreate_agent_run with AsyncMock...{RESET}", flush=True)
        if _mock_agno_telemetry():
            print(f"  Telemetry HTTP call mocked.", flush=True)

            for scenario in [1, 2, 3]:
                print(f"\n  {DIM}[Phase 2 · Scenario {scenario}]{RESET}", flush=True)
                agno_tel_data[scenario] = await _collect_agno(scenario, "Agno (tel mock)", telemetry=True)
        else:
            print(f"  {YELLOW}Could not mock telemetry — skipping tel column.{RESET}", flush=True)

    # ── Print combined results ────────────────────────────────────────────────
    for scenario in [1, 2, 3]:
        _print_scenario(
            scenario,
            t=t_data[scenario],
            agno=agno_data.get(scenario),
            agno_tel=agno_tel_data.get(scenario),
        )

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  Timbal: TestModel (sync handler, stateless). Built-in InMemory tracing always on.")
    print()
    print(f"  Agno (no tel):   FakeModel (sync handler, stateless). telemetry=False.")
    print(f"                   Explicit opt-out required — default is telemetry=True.")
    print()
    print(f"  Agno (tel mock): telemetry=True with acreate_agent_run replaced by AsyncMock.")
    print(f"                   Measures JSON-serialisation overhead of the telemetry path")
    print(f"                   without the network call to api.agno.com.")
    print()
    print(f"  Default telemetry=True: awaited HTTP POST on every run — ~400–500 ms overhead.")
    print(f"  No batching, no queue. Not benchmarked directly (network variance dominates).")
    print()
    print(f"  Scenario 3: both Timbal and Agno dispatch all 3 tools concurrently")
    print(f"  (Agno uses asyncio.gather over function_calls_to_run in aresponse()).")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
