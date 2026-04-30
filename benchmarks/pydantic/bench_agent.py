#!/usr/bin/env python3
"""
Timbal vs PydanticAI — full agent loop benchmark.

Three scenarios, each running the identical pipeline on both frameworks:
  1. Single tool call:   prompt → LLM → tool(add) → LLM → answer
  2. Multi-step (3 tools): prompt → LLM → add → LLM → mul → LLM → sub → LLM → answer
  3. Parallel tools:     prompt → LLM → [add, mul, neg] concurrent → LLM → answer
     (both Timbal and PydanticAI dispatch tool calls concurrently)

Timbal uses TestModel for fully offline, deterministic LLM simulation.
PydanticAI uses FunctionModel — a callable-based fake LLM, also offline.

Columns:
  Timbal           — built-in InMemory tracing (always on)
  PAI (bare)       — PydanticAI with no observability
  PAI + Logfire    — PydanticAI with Logfire instrumentation (send_to_logfire=False,
                     so span-creation overhead is measured without network noise)

Collection order:
  Phase 1 (bare):  Timbal + PAI bare  (all 3 scenarios)
  → activate Logfire (global, irreversible — must run bare first)
  Phase 2 (+obs):  PAI + Logfire      (all 3 scenarios)
  → print combined tables

Run:
    uv run python benchmarks/pydantic/bench_agent.py
    uv run python benchmarks/pydantic/bench_agent.py --quick
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
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
# PYDANTIC AI — FunctionModel agent
#
# FunctionModel lets us pass a plain Python callable as the LLM. It receives
# the full message history and returns a ModelResponse. By counting ToolReturnPart
# items in the history we can drive a deterministic tool-call sequence — no shared
# counter state, safe for concurrent async runs.
#
# Concurrency note:
#   PydanticAI dispatches multiple tool calls from a single ModelResponse
#   concurrently via asyncio. Scenario 3 (3 parallel tools) is truly parallel on
#   both Timbal and PydanticAI — no sequential penalty on either side.
# ═══════════════════════════════════════════════════════════════════════════════

HAS_PAI     = False
HAS_LOGFIRE = False

try:
    from pydantic_ai import Agent as PAIAgent  # noqa: E402
    from pydantic_ai.models.function import FunctionModel, AgentInfo  # noqa: E402
    from pydantic_ai.messages import (  # noqa: E402
        ModelMessage, ModelResponse, TextPart, ToolCallPart,
    )

    def _count_pai_tool_returns(messages: list) -> int:
        return sum(
            1 for msg in messages
            for part in msg.parts
            if hasattr(part, "part_kind") and part.part_kind == "tool-return"
        )

    def _make_pai_agent(scenario: int) -> PAIAgent:
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

        if scenario == 1:
            def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                step = _count_pai_tool_returns(messages)
                if step == 0:
                    return ModelResponse(parts=[ToolCallPart(tool_name="add", args='{"a": 1, "b": 2}', tool_call_id="c1")])
                return ModelResponse(parts=[TextPart("3")])
        elif scenario == 2:
            def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                step = _count_pai_tool_returns(messages)
                if step == 0:
                    return ModelResponse(parts=[ToolCallPart(tool_name="add", args='{"a": 1, "b": 2}', tool_call_id="c1")])
                elif step == 1:
                    return ModelResponse(parts=[ToolCallPart(tool_name="multiply", args='{"a": 3, "b": 4}', tool_call_id="c2")])
                elif step == 2:
                    return ModelResponse(parts=[ToolCallPart(tool_name="subtract", args='{"a": 12, "b": 3}', tool_call_id="c3")])
                return ModelResponse(parts=[TextPart("9")])
        else:
            def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                step = _count_pai_tool_returns(messages)
                if step == 0:
                    return ModelResponse(parts=[
                        ToolCallPart(tool_name="add", args='{"a": 1, "b": 2}', tool_call_id="c1"),
                        ToolCallPart(tool_name="multiply", args='{"a": 3, "b": 4}', tool_call_id="c2"),
                        ToolCallPart(tool_name="negate", args='{"x": 5}', tool_call_id="c3"),
                    ])
                return ModelResponse(parts=[TextPart("done")])

        return PAIAgent(FunctionModel(llm), tools=tools_map[scenario])

    HAS_PAI = True

except ImportError as e:
    print(f"PydanticAI not found: {e}")


def _try_init_logfire() -> bool:
    """Instrument Logfire lazily. Called after all bare measurements complete.

    send_to_logfire=False disables HTTP export — we measure span-creation overhead
    only, without network noise. Once instrument_pydantic_ai() is called it patches
    PydanticAI globally and cannot be undone; bare measurements must run first.
    """
    global HAS_LOGFIRE
    try:
        import logfire as _logfire
        _logfire.configure(send_to_logfire=False, console=False)
        _logfire.instrument_pydantic_ai()
        HAS_LOGFIRE = True
    except (ImportError, Exception):
        pass
    return HAS_LOGFIRE


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


async def _collect_pai(scenario: int, label: str) -> RunData:
    agent = _make_pai_agent(scenario)
    factory = lambda: agent.run("go")
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

COL_W = 14


def _print_scenario(
    scenario: int,
    t: RunData,
    pai: RunData | None,
    pai_lf: RunData | None,
) -> None:
    section(f"Scenario {scenario}: {SCENARIO_NAMES[scenario]}")

    cols: list[tuple[str, RunData]] = [("Timbal", t)]
    if pai:
        cols.append(("PAI (bare)", pai))
    if pai_lf:
        cols.append(("PAI+Logfire", pai_lf))

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
    print(f"{BOLD}  Timbal vs PydanticAI — agent loop benchmark{RESET}")
    print(f"  {N_ITERS} iters · burst {N_BURST} · {N_MEM} mem · {TP_OPS} throughput ops")
    print(f"  Timbal: TestModel (offline) · PydanticAI: FunctionModel (offline)")
    if HAS_PAI:
        print(f"  PydanticAI found — will measure bare + Logfire (+obs)")
        print(f"  {DIM}Both frameworks run fully async and dispatch parallel tool calls concurrently.{RESET}")
    else:
        print(f"  {YELLOW}PydanticAI not found — Timbal only.  Install: uv pip install pydantic-ai logfire{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    # ── Correctness spot-check ────────────────────────────────────────────────
    print(f"\n  {DIM}Spot-checking correctness (Scenario 1)...{RESET}", flush=True)
    _t = _make_timbal_agent(1)
    res = await _t(prompt="go").collect()
    t_ok = res.status.code == "success"
    msg = f"  Timbal: {'✓' if t_ok else '✗'}"
    if HAS_PAI:
        _p = _make_pai_agent(1)
        p_res = await _p.run("go")
        p_ok = str(p_res.output).strip() == "3"
        msg += f"  |  PAI: {'✓' if p_ok else f'✗ (got {p_res.output!r})'}"
    print(msg, flush=True)

    # ── Phase 1: Timbal + PAI bare ────────────────────────────────────────────
    t_data:   dict[int, RunData] = {}
    pai_data: dict[int, RunData] = {}

    for scenario in [1, 2, 3]:
        print(f"\n  {DIM}[Phase 1 · Scenario {scenario}]{RESET}", flush=True)
        t_data[scenario] = await _collect_timbal(scenario)
        if HAS_PAI:
            pai_data[scenario] = await _collect_pai(scenario, "PAI bare")

    # ── Activate Logfire (global, cannot uninstrument) ────────────────────────
    pai_lf_data: dict[int, RunData] = {}
    if HAS_PAI:
        print(f"\n  {DIM}Activating Logfire instrumentation (send_to_logfire=False)...{RESET}", flush=True)
        if _try_init_logfire():
            print(f"  Logfire instrumented.", flush=True)

            for scenario in [1, 2, 3]:
                print(f"\n  {DIM}[Phase 2 · Scenario {scenario}]{RESET}", flush=True)
                pai_lf_data[scenario] = await _collect_pai(scenario, "PAI+Logfire")
        else:
            print(f"  {YELLOW}Logfire not found — skipping obs column.  Install: uv pip install logfire{RESET}", flush=True)

    # ── Print combined results ────────────────────────────────────────────────
    for scenario in [1, 2, 3]:
        _print_scenario(
            scenario,
            t=t_data[scenario],
            pai=pai_data.get(scenario),
            pai_lf=pai_lf_data.get(scenario),
        )

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  Timbal: TestModel (sync handler, stateless). Built-in InMemory tracing always on.")
    print()
    print(f"  PAI bare:    PydanticAI with FunctionModel (sync handler, stateless).")
    print(f"               No observability instrumentation.")
    print()
    print(f"  PAI+Logfire: Logfire instrumented via logfire.instrument_pydantic_ai().")
    print(f"               send_to_logfire=False — span-creation overhead only, no HTTP calls.")
    print(f"               Logfire patches PydanticAI globally; bare pass must run first.")
    print()
    print(f"  Scenario 3: both Timbal and PydanticAI dispatch all 3 tools concurrently")
    print(f"  (PydanticAI uses asyncio.gather for multiple ToolCallParts in one response).")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
