#!/usr/bin/env python3
"""
Timbal vs CrewAI — full agent loop benchmark.

Three scenarios, each running the identical pipeline on both frameworks:
  1. Single tool call:   prompt → LLM → tool(add) → LLM → answer
  2. Multi-step (3 tools): prompt → LLM → add → LLM → mul → LLM → sub → LLM → answer
  3. Parallel tools:     prompt → LLM → [add, mul, neg] concurrent → LLM → answer
     (CrewAI dispatches tools sequentially via ReAct — benchmarked as-is)

Timbal uses TestModel for fully offline, deterministic LLM simulation.
CrewAI uses a patched OpenAI completion provider — also offline.

Columns:
  Timbal           — built-in InMemory tracing (always on)
  CA (sync)        — Crew.kickoff() via run_in_executor — what most teams do;
                     kickoff_async() is identical (it's just asyncio.to_thread(kickoff))
  CA (async)       — Crew.akickoff() — native async path added in late 2024;
                     requires CrewAI >= 0.80 or so
  CA sync+AO       — CA (sync) + AgentOps instrumentation (HTTP mocked)
  CA async+AO      — CA (async) + AgentOps instrumentation (HTTP mocked)

Collection order:
  Pass 1 (bare):    Timbal + CA sync + CA async  (all 3 scenarios)
  → activate AgentOps (global, irreversible — must run bare first)
  Pass 2 (+obs):    CA sync+AO + CA async+AO     (all 3 scenarios)
  → print combined tables

Concurrency note:
  CrewAI sync burst is capped at CA_BURST_CONC threads. Crew.kickoff() is not
  thread-safe for concurrent reuse on the same instance; even with fresh instances
  per call, GIL contention + internal asyncio calls from threads caused deadlocks
  above ~5 threads during development (see GitHub #2632, #1234). CA (async) burst
  runs fully concurrent — akickoff() is a proper coroutine.

Run:
    uv run python benchmarks/crewai/bench_agent.py
    uv run python benchmarks/crewai/bench_agent.py --quick
"""

from __future__ import annotations

import argparse
import contextvars
import dataclasses
import logging
import os
import threading
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-bench")
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

N_ITERS    = 20  if _args.quick else 100
N_WARMUP   = 3   if _args.quick else 10
N_MEM      = 20  if _args.quick else 100
N_BURST    = {1: 10, 2: 5, 3: 8}   if _args.quick else {1: 30, 2: 20, 3: 25}
TP_OPS     = 30  if _args.quick else 100
CONC_LEVELS = [1, 5, 20]
CA_BURST_CONC = 5   # cap for sync kickoff: not thread-safe above ~5 (see #2632)
WIDTH = 110

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
# CREWAI — patched fake LLM, tool definitions, crew factory
#
# Concurrency story (see module docstring for details):
#
#   kickoff()         — sync, blocks a thread, not safe for concurrent reuse on
#                       the same Crew instance (usage_metrics mutated per run)
#   kickoff_async()   — NOT truly async: asyncio.to_thread(self.kickoff, ...)
#                       same GIL / thread-safety issues as kickoff()
#   akickoff()        — genuinely async: awaits all the way to litellm.acompletion()
#                       added late 2024; still mutates self.usage_metrics so fresh
#                       instance per concurrent call is required
#
# Fake LLM strategy:
#   Sync path  → threading.local counter (each executor thread owns its own state)
#   Async path → contextvars.ContextVar counter (each asyncio Task owns its copy;
#                asyncio.gather creates separate Tasks so concurrent akickoff() calls
#                are fully isolated without any locking)
# ═══════════════════════════════════════════════════════════════════════════════

HAS_CREWAI   = False
HAS_AGENTOPS = False
HAS_AKICKOFF = False

try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew, LLM as CrewLLM  # noqa: E402
    from crewai.tools import BaseTool as CrewBaseTool  # noqa: E402
    from pydantic import BaseModel  # noqa: E402
    from crewai.llms.providers.openai.completion import OpenAICompletion as _OAICompletion  # noqa: E402

    # ── Sync fake LLM (threading.local — safe because kickoff() is single-threaded) ──
    _crew_state    = threading.local()
    _crew_scenario = 1  # module-level, set before each measurement pass

    def _fake_crew_llm(self, messages, tools=None, callbacks=None, **kwargs):
        if not hasattr(_crew_state, "n"):
            _crew_state.n = 0
        _crew_state.n += 1

        if _crew_scenario == 1:
            if _crew_state.n % 2 == 1:
                return 'Thought: add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            return "Thought: done.\nFinal Answer: 3"

        elif _crew_scenario == 2:
            step = (_crew_state.n - 1) % 4
            if step == 0:
                return 'Thought: add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            elif step == 1:
                return 'Thought: mul.\nAction: multiply\nAction Input: {"a": 3, "b": 4}'
            elif step == 2:
                return 'Thought: sub.\nAction: subtract\nAction Input: {"a": 12, "b": 3}'
            return "Thought: done.\nFinal Answer: 9"

        else:  # scenario 3
            step = (_crew_state.n - 1) % 4
            if step == 0:
                return 'Thought: add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            elif step == 1:
                return 'Thought: mul.\nAction: multiply\nAction Input: {"a": 3, "b": 4}'
            elif step == 2:
                return 'Thought: neg.\nAction: negate\nAction Input: {"x": 5}'
            return "Thought: done.\nFinal Answer: done"

    _OAICompletion.call = _fake_crew_llm
    _OAICompletion.supports_function_calling = lambda self: False

    # ── Async fake LLM (ContextVar — each asyncio Task gets its own copy) ──────
    # asyncio.gather() creates a separate Task per coroutine, copying the current
    # context at creation time. Each Task independently increments _async_crew_n
    # starting from 0, with no shared state between concurrent akickoff() calls.
    _async_crew_n: contextvars.ContextVar[int] = contextvars.ContextVar("_async_crew_n", default=0)

    async def _fake_crew_llm_async(self, messages, tools=None, callbacks=None, **kwargs):
        n = _async_crew_n.get() + 1
        _async_crew_n.set(n)

        if _crew_scenario == 1:
            if n % 2 == 1:
                return 'Thought: add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            return "Thought: done.\nFinal Answer: 3"

        elif _crew_scenario == 2:
            step = (n - 1) % 4
            if step == 0:
                return 'Thought: add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            elif step == 1:
                return 'Thought: mul.\nAction: multiply\nAction Input: {"a": 3, "b": 4}'
            elif step == 2:
                return 'Thought: sub.\nAction: subtract\nAction Input: {"a": 12, "b": 3}'
            return "Thought: done.\nFinal Answer: 9"

        else:  # scenario 3
            step = (n - 1) % 4
            if step == 0:
                return 'Thought: add.\nAction: add\nAction Input: {"a": 1, "b": 2}'
            elif step == 1:
                return 'Thought: mul.\nAction: multiply\nAction Input: {"a": 3, "b": 4}'
            elif step == 2:
                return 'Thought: neg.\nAction: negate\nAction Input: {"x": 5}'
            return "Thought: done.\nFinal Answer: done"

    # Patch the async call path used by akickoff() (litellm async → OpenAICompletion.acall)
    if hasattr(_OAICompletion, "acall"):
        _OAICompletion.acall = _fake_crew_llm_async

    class _AddInput(BaseModel):
        a: int
        b: int

    class _MulInput(BaseModel):
        a: int
        b: int

    class _SubInput(BaseModel):
        a: int
        b: int

    class _NegInput(BaseModel):
        x: int

    class _CrewAdd(CrewBaseTool):
        name: str = "add"
        description: str = "Add two integers."
        args_schema: type[BaseModel] = _AddInput
        def _run(self, **kwargs) -> str: return str(kwargs["a"] + kwargs["b"])

    class _CrewMul(CrewBaseTool):
        name: str = "multiply"
        description: str = "Multiply two integers."
        args_schema: type[BaseModel] = _MulInput
        def _run(self, **kwargs) -> str: return str(kwargs["a"] * kwargs["b"])

    class _CrewSub(CrewBaseTool):
        name: str = "subtract"
        description: str = "Subtract b from a."
        args_schema: type[BaseModel] = _SubInput
        def _run(self, **kwargs) -> str: return str(kwargs["a"] - kwargs["b"])

    class _CrewNeg(CrewBaseTool):
        name: str = "negate"
        description: str = "Negate a number."
        args_schema: type[BaseModel] = _NegInput
        def _run(self, **kwargs) -> str: return str(-kwargs["x"])

    def _make_crew(scenario: int) -> Crew:
        global _crew_scenario
        _crew_scenario = scenario
        if scenario == 1:
            tools, desc = [_CrewAdd()], "Run add(1,2)."
        elif scenario == 2:
            tools, desc = [_CrewAdd(), _CrewMul(), _CrewSub()], "Run add(1,2), multiply(3,4), subtract(12,3)."
        else:
            tools, desc = [_CrewAdd(), _CrewMul(), _CrewNeg()], "Run add(1,2), multiply(3,4), negate(5)."
        llm = CrewLLM(model="openai/gpt-4o-mini")
        agent = CrewAgent(role="Bench", goal="Execute.", backstory="bench", tools=tools, llm=llm, verbose=False)
        task = CrewTask(description=desc, expected_output="Result.", agent=agent)
        return Crew(agents=[agent], tasks=[task], verbose=False, output_log_file=False)

    HAS_CREWAI   = True
    HAS_AKICKOFF = hasattr(Crew, "akickoff")

except ImportError as e:
    print(f"CrewAI not found: {e}")


def _try_init_agentops() -> bool:
    """Instrument AgentOps lazily. Called after all bare measurements complete."""
    global HAS_AGENTOPS
    try:
        from opentelemetry.sdk.trace.export import SpanExportResult as _SpanResult
        from agentops.sdk.exporters import AuthenticatedOTLPExporter as _AOExporter
        import agentops as _agentops
        from agentops.instrumentation.agentic.crewai.instrumentation import CrewaiInstrumentor as _CI

        _AOExporter.export = lambda self, spans: _SpanResult.SUCCESS
        _agentops.init(api_key="fake-key-for-bench", skip_auto_end_session=True, auto_start_session=False)
        _CI().instrument()
        HAS_AGENTOPS = True
    except (ImportError, Exception):
        pass
    return HAS_AGENTOPS


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


def _latency_sync(factory, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        factory()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        factory()
        samples.append((time.perf_counter() - t0) * 1e6)
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


def _memory_sync(factory, n: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        factory()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        factory()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


async def _burst_async(factory, n: int) -> list[float]:
    """Fully concurrent burst for async factories (Timbal and CA async)."""
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


async def _burst_ca_sync(ca_factory, n: int) -> list[float]:
    """Semaphore-bounded burst for CA sync (Crew.kickoff via run_in_executor).

    Hard cap at CA_BURST_CONC threads. Above ~5 concurrent Crew instances,
    GIL contention + internal asyncio calls from threads caused deadlocks
    (see GitHub crewAIInc/crewAI #2632). This cap is not a CrewAI setting —
    it reflects the practical concurrency ceiling for sync kickoff.
    """
    sem = asyncio.Semaphore(CA_BURST_CONC)

    async def _bounded():
        async with sem:
            await ca_factory()

    for _ in range(min(3, n)):
        await _bounded()
    gc.collect()

    samples: list[float] = []

    async def _timed():
        t0 = time.perf_counter()
        await _bounded()
        samples.append((time.perf_counter() - t0) * 1e6)

    await asyncio.gather(*[_timed() for _ in range(n)])
    return sorted(samples)


async def _burst_memory_async(factory, n: int) -> tuple[float, float]:
    """Peak memory while n coroutines run concurrently (async/Timbal path)."""
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    await asyncio.gather(*[factory() for _ in range(n)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _clear_traces()
    return peak, peak / n


async def _burst_memory_ca_sync(ca_factory, n: int) -> tuple[float, float]:
    """Peak memory during semaphore-bounded sync burst (CA_BURST_CONC threads)."""
    sem = asyncio.Semaphore(CA_BURST_CONC)

    async def _bounded():
        async with sem:
            await ca_factory()

    for _ in range(min(3, n)):
        await _bounded()
    gc.collect()
    tracemalloc.start()
    await asyncio.gather(*[_bounded() for _ in range(n)])
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
    latency: list[float]        # µs, N_ITERS samples
    mem_peak: float             # bytes
    mem_per: float              # bytes per run
    burst: list[float]          # µs, N_BURST[scenario] samples
    burst_mem_peak: float       # bytes — peak during concurrent burst
    burst_mem_per: float        # bytes per concurrent run
    throughput: list[float]     # ops/s per CONC_LEVELS entry


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
    print(f"{mem_peak/1024:.0f} KB peak", flush=True)

    print(f"    Timbal burst ({n_burst} concurrent)...", end=" ", flush=True)
    burst = await _burst_async(factory, n_burst)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    print(f"    Timbal burst memory ({n_burst} concurrent)...", end=" ", flush=True)
    burst_mem_peak, burst_mem_per = await _burst_memory_async(factory, n_burst)
    print(f"{burst_mem_peak/1024:.0f} KB peak", flush=True)

    tp = []
    for conc in CONC_LEVELS:
        print(f"    Timbal throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput_async(factory, TP_OPS, conc)
        tp.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=lat, mem_peak=mem_peak, mem_per=mem_per, burst=burst,
                   burst_mem_peak=burst_mem_peak, burst_mem_per=burst_mem_per, throughput=tp)


async def _collect_ca_sync(scenario: int, label: str) -> RunData:
    """Measure using Crew.kickoff() via run_in_executor (sync path, thread-capped)."""
    global _crew_scenario
    _crew_scenario = scenario

    crew = _make_crew(scenario)
    n_burst = N_BURST[scenario]

    def _make_and_run():
        global _crew_scenario
        _crew_scenario = scenario
        _make_crew(scenario).kickoff()

    async def ca_sync_factory():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _make_and_run)

    print(f"    {label} latency...", end=" ", flush=True)
    lat = _latency_sync(crew.kickoff, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(lat, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = _memory_sync(crew.kickoff, N_MEM, N_WARMUP)
    print(f"{mem_peak/1024:.0f} KB peak", flush=True)

    print(f"    {label} burst ({n_burst} bounded, conc≤{CA_BURST_CONC})...", end=" ", flush=True)
    burst = await _burst_ca_sync(ca_sync_factory, n_burst)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    print(f"    {label} burst memory ({n_burst} bounded, conc≤{CA_BURST_CONC})...", end=" ", flush=True)
    burst_mem_peak, burst_mem_per = await _burst_memory_ca_sync(ca_sync_factory, n_burst)
    print(f"{burst_mem_peak/1024:.0f} KB peak", flush=True)

    tp = []
    for conc in CONC_LEVELS:
        ca_conc  = min(conc, CA_BURST_CONC)
        ca_total = min(TP_OPS, 20) if conc >= 10 else TP_OPS
        print(f"    {label} throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput_async(ca_sync_factory, ca_total, ca_conc)
        tp.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=lat, mem_peak=mem_peak, mem_per=mem_per, burst=burst,
                   burst_mem_peak=burst_mem_peak, burst_mem_per=burst_mem_per, throughput=tp)


async def _collect_ca_async(scenario: int, label: str) -> RunData:
    """Measure using Crew.akickoff() — native async path.

    Each call creates a fresh Crew instance (akickoff still mutates self.usage_metrics).
    ContextVar isolation means concurrent akickoff() calls each track their own
    LLM call count independently — no locking needed.
    """
    global _crew_scenario
    _crew_scenario = scenario
    n_burst = N_BURST[scenario]

    # For sequential latency/memory: reuse one crew, reset ContextVar each call.
    seq_crew = _make_crew(scenario)

    async def ca_async_seq():
        """Sequential factory: shared crew instance, reset counter each call."""
        global _crew_scenario
        _crew_scenario = scenario
        _async_crew_n.set(0)
        await seq_crew.akickoff()

    async def ca_async_fresh():
        """Concurrent factory: fresh Crew per call (usage_metrics isolation).
        asyncio.gather creates a separate Task per coroutine — each Task inherits
        a copy of the current context, so _async_crew_n is independently owned."""
        global _crew_scenario
        _crew_scenario = scenario
        c = _make_crew(scenario)
        _async_crew_n.set(0)
        await c.akickoff()

    print(f"    {label} latency...", end=" ", flush=True)
    lat = await _latency_async(ca_async_seq, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(lat, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory_async(ca_async_seq, N_MEM, N_WARMUP)
    print(f"{mem_peak/1024:.0f} KB peak", flush=True)

    print(f"    {label} burst ({n_burst} concurrent)...", end=" ", flush=True)
    burst = await _burst_async(ca_async_fresh, n_burst)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    print(f"    {label} burst memory ({n_burst} concurrent)...", end=" ", flush=True)
    burst_mem_peak, burst_mem_per = await _burst_memory_async(ca_async_fresh, n_burst)
    print(f"{burst_mem_peak/1024:.0f} KB peak", flush=True)

    tp = []
    for conc in CONC_LEVELS:
        print(f"    {label} throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput_async(ca_async_fresh, TP_OPS, conc)
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
    3: "Parallel tools  (Timbal: concurrent | CrewAI: sequential ReAct)",
}

COL_W = 14


def _print_scenario(
    scenario: int,
    t: RunData,
    ca_sync: RunData | None,
    ca_async: RunData | None,
    ca_sync_ao: RunData | None,
    ca_async_ao: RunData | None,
) -> None:
    section(f"Scenario {scenario}: {SCENARIO_NAMES[scenario]}")
    if scenario == 3:
        print(f"  {DIM}Timbal dispatches 3 tools concurrently; CrewAI does 3 sequential ReAct steps.{RESET}")

    cols: list[tuple[str, RunData]] = [("Timbal", t)]
    if ca_sync:
        cols.append(("CA (sync)", ca_sync))
    if ca_async:
        cols.append(("CA (async)", ca_async))
    if ca_sync_ao:
        cols.append(("CA sync+AO", ca_sync_ao))
    if ca_async_ao:
        cols.append(("CA async+AO", ca_async_ao))

    hdr = f"  {'':>12}" + "".join(f"  {c:>{COL_W}}" for c, _ in cols)
    sep = f"  {'─' * 12}" + f"  {'─' * COL_W}" * len(cols)

    # ── Latency ──────────────────────────────────────────────────────────────
    subsection(f"Latency  (×{N_ITERS}  Timbal async · CA sync: sequential · CA async: sequential)")
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
        print(f"  {name:<{fw}}  {d.mem_peak/1024:>10.1f} KB  {d.mem_per:>10.0f}  B")

    # ── Burst ────────────────────────────────────────────────────────────────
    n_burst = N_BURST[scenario]
    subsection(
        f"Burst  ({n_burst} concurrent — Timbal/CA async: unbounded · CA sync: bounded conc≤{CA_BURST_CONC})"
    )
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d.burst, p)) for _, d in cols]
        print(f"  {label:>12}" + "".join(f"  {v:>{COL_W}}" for v in vals))
    wall_parts = [f"{name}: {max(d.burst)/1e3:.1f} ms" for name, d in cols]
    print(f"\n  {DIM}wall: {' | '.join(wall_parts)}{RESET}")

    # ── Burst memory ─────────────────────────────────────────────────────────
    subsection(f"Burst memory  ({n_burst} concurrent — peak during full burst, no GC between)")
    fw = 24
    print(f"  {'framework':<{fw}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * fw}  {'─' * 12}  {'─' * 12}")
    for name, d in cols:
        print(f"  {name:<{fw}}  {d.burst_mem_peak/1024:>10.1f} KB  {d.burst_mem_per:>10.0f}  B")

    # ── Throughput ────────────────────────────────────────────────────────────
    subsection(f"Throughput  ({TP_OPS} loops  ·  CA sync capped at conc≤{CA_BURST_CONC})")
    print(hdr)
    print(sep)
    for i, conc in enumerate(CONC_LEVELS):
        vals = [f"{d.throughput[i]:>10.0f}/s" for _, d in cols]
        print(f"  {conc:>12}" + "".join(f"  {v:>{COL_W}}" for v in vals))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    global HAS_AKICKOFF
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal vs CrewAI — agent loop benchmark{RESET}")
    print(f"  {N_ITERS} iters · burst {N_BURST} · {N_MEM} mem · {TP_OPS} throughput ops")
    print(f"  Timbal: TestModel (offline) · CrewAI: patched OpenAI (offline)")
    if HAS_CREWAI:
        ao_str = "bare + AgentOps" if True else "bare only"
        async_str = " + akickoff() async" if HAS_AKICKOFF else " (akickoff not available)"
        print(f"  CrewAI found — will measure{async_str}")
        print(f"  {DIM}CA sync burst/throughput: capped at {CA_BURST_CONC} threads (kickoff not async-safe; see #2632).{RESET}")
        if HAS_AKICKOFF:
            print(f"  {DIM}CA async burst/throughput: fully concurrent (akickoff is native async).{RESET}")
    else:
        print(f"  {YELLOW}CrewAI not found — Timbal only.  Install: uv pip install crewai agentops{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    # ── Correctness spot-check ────────────────────────────────────────────────
    print(f"\n  {DIM}Spot-checking correctness (Scenario 1)...{RESET}", flush=True)
    _t = _make_timbal_agent(1)
    res = await _t(prompt="go").collect()
    t_ok = res.status.code == "success"
    msg = f"  Timbal: {'✓' if t_ok else '✗'}"
    if HAS_CREWAI:
        _c = _make_crew(1)
        c_result = _c.kickoff()
        c_ok = getattr(c_result, "raw", str(c_result)).strip() == "3"
        msg += f"  |  CA sync: {'✓' if c_ok else '✗'}"
    if HAS_AKICKOFF:
        try:
            _c2 = _make_crew(1)
            _async_crew_n.set(0)
            ca2_result = await _c2.akickoff()
            ca2_ok = getattr(ca2_result, "raw", str(ca2_result)).strip() == "3"
            msg += f"  |  CA async: {'✓' if ca2_ok else '✗ (disabling async path)'}"
            if not ca2_ok:
                HAS_AKICKOFF = False  # type: ignore[assignment]
        except Exception as exc:
            msg += f"  |  CA async: ✗ ({type(exc).__name__} — disabling async path)"
            HAS_AKICKOFF = False  # type: ignore[assignment]
    print(msg, flush=True)

    # ── Phase 1: Timbal + CA bare (sync + async) ──────────────────────────────
    t_data:        dict[int, RunData] = {}
    ca_sync_data:  dict[int, RunData] = {}
    ca_async_data: dict[int, RunData] = {}

    for scenario in [1, 2, 3]:
        print(f"\n  {DIM}[Phase 1 · Scenario {scenario}]{RESET}", flush=True)
        t_data[scenario] = await _collect_timbal(scenario)
        if HAS_CREWAI:
            ca_sync_data[scenario] = await _collect_ca_sync(scenario, "CA sync")
        if HAS_AKICKOFF:
            ca_async_data[scenario] = await _collect_ca_async(scenario, "CA async")

    # ── Activate AgentOps (global, cannot uninstrument) ───────────────────────
    ca_sync_ao_data:  dict[int, RunData] = {}
    ca_async_ao_data: dict[int, RunData] = {}
    if HAS_CREWAI:
        print(f"\n  {DIM}Activating AgentOps instrumentation...{RESET}", flush=True)
        if _try_init_agentops():
            print(f"  AgentOps instrumented.", flush=True)

            # ── Phase 2: CA+AgentOps (sync + async) ──────────────────────────
            for scenario in [1, 2, 3]:
                print(f"\n  {DIM}[Phase 2 · Scenario {scenario}]{RESET}", flush=True)
                ca_sync_ao_data[scenario] = await _collect_ca_sync(scenario, "CA sync+AO")
                if HAS_AKICKOFF:
                    ca_async_ao_data[scenario] = await _collect_ca_async(scenario, "CA async+AO")
        else:
            print(f"  {YELLOW}AgentOps not found — skipping obs columns.  Install: uv pip install agentops{RESET}", flush=True)

    # ── Print combined results ────────────────────────────────────────────────
    for scenario in [1, 2, 3]:
        _print_scenario(
            scenario,
            t=t_data[scenario],
            ca_sync=ca_sync_data.get(scenario),
            ca_async=ca_async_data.get(scenario),
            ca_sync_ao=ca_sync_ao_data.get(scenario),
            ca_async_ao=ca_async_ao_data.get(scenario),
        )

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  Timbal: TestModel (sync handler, stateless). Built-in InMemory tracing always on.")
    print()
    print(f"  CA sync:  Crew.kickoff() in run_in_executor — what most teams run in production.")
    print(f"            kickoff_async() is NOT truly async: it is asyncio.to_thread(self.kickoff)")
    print(f"            (crewAIInc/crewAI crew.py). Same GIL + thread-safety limits apply.")
    print(f"            Burst capped at {CA_BURST_CONC} threads: above that, instability from GIL contention")
    print(f"            and internal asyncio calls made from threads (GitHub #2632, #1234).")
    print()
    if HAS_AKICKOFF:
        print(f"  CA async: Crew.akickoff() — native async path (added late 2024). Truly awaits")
        print(f"            all the way to litellm.acompletion(). Still mutates self.usage_metrics")
        print(f"            per run, so fresh Crew per concurrent call is required. Burst fully")
        print(f"            concurrent (no semaphore cap). Global event bus singleton and shared")
        print(f"            SQLite path remain potential bottlenecks at high concurrency (#2632).")
        print()
    print(f"  AgentOps: CrewaiInstrumentor().instrument() patches CrewAI methods globally and")
    print(f"            cannot be undone. Bare pass runs first, then AgentOps is activated.")
    print(f"            HTTP export is mocked — only hot-path span overhead is measured.")
    print()
    print(f"  Scenario 3: Timbal dispatches all 3 tools in one LLM call (parallel).")
    print(f"  CrewAI uses ReAct (one tool per LLM call) — 3 sequential steps instead.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
