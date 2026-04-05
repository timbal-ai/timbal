#!/usr/bin/env python3
"""
Timbal vs LangGraph — full agent loop benchmark.

Three scenarios, each running the identical pipeline on both frameworks:
  1. Single tool call:   prompt → LLM → tool(add) → LLM → answer
  2. Multi-step (3 tools): prompt → LLM → add → LLM → mul → LLM → sub → LLM → answer
  3. Parallel tools:     prompt → LLM → [add, mul, neg] concurrent → LLM → answer

All LLMs are faked. Observability included: Timbal built-in tracing, LangSmith for LangGraph.
Agent/graph creation is excluded from timing — only execution is measured.
Metrics: latency, memory, burst, throughput.

LangGraph implementation note:
  Uses langgraph.prebuilt.create_react_agent — the recommended prebuilt agent in the
  LangGraph docs. This is intentional: it represents the standard path most teams take.
  A hand-rolled custom StateGraph agent could in principle be marginally faster by
  skipping the prebuilt abstraction layer, but the fundamental architectural point
  holds regardless: any LangGraph agent loop is a graph with a conditional back-edge,
  and the Pregel superstep machinery runs on every iteration whether you use the
  prebuilt or not. The gap vs Timbal is architectural, not a consequence of using
  the prebuilt.

Run:
    uv run python benchmarks/langchain/bench_agent.py
    uv run python benchmarks/langchain/bench_agent.py --quick
"""

from __future__ import annotations

import argparse
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
parser.add_argument("--timbal-only", action="store_true", help="Skip LangGraph, measure Timbal only")
_args, _ = parser.parse_known_args()

N_ITERS = 20 if _args.quick else 100
N_WARMUP = 3 if _args.quick else 10
N_BURST = {1: 20, 2: 10, 3: 15}  # Scaled by scenario weight
N_MEM = 20 if _args.quick else 100
THROUGHPUT_OPS = 30 if _args.quick else 200
CONCURRENCY_LEVELS = [1, 10, 50]
WIDTH = 76

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
# Timbal trace cleanup
# ═══════════════════════════════════════════════════════════════════════════════

from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


def _clear_traces():
    InMemoryTracingProvider._storage.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — fake LLM agent factory
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Agent, TestModel  # noqa: E402
from timbal.types.message import Message  # noqa: E402
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent  # noqa: E402


def _count_tool_results(messages) -> int:
    return sum(
        1 for m in messages
        for c in (m.content if hasattr(m, "content") and m.content else [])
        if isinstance(c, ToolResultContent)
    )


def _make_timbal_agent(scenario: int):
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
# LANGGRAPH — reusable compiled graph with resettable fake LLM
# ═══════════════════════════════════════════════════════════════════════════════

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

try:
    from unittest.mock import MagicMock
    from langchain_core.tracers.langchain import LangChainTracer

    def _make_ls_tracer() -> LangChainTracer:
        tracer = LangChainTracer()
        tracer._persist_run_single = MagicMock(return_value=None)
        return tracer

    HAS_LANGSMITH = True
except (ImportError, Exception):
    HAS_LANGSMITH = False


class _FakeLLM(FakeMessagesListChatModel):
    """Stateless fake LLM: inspects message history to determine the response step.

    Counts ToolMessages in the conversation to find the current step — no shared
    counter state, safe for concurrent async ainvoke() on a single shared graph.
    """

    def bind_tools(self, tools, **kw):
        return self

    def _step(self, messages) -> int:
        return sum(1 for m in messages if isinstance(m, ToolMessage))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self._step(messages) % len(self.responses)
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self._step(messages) % len(self.responses)
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])


def _lc_responses(scenario: int) -> list[AIMessage]:
    if scenario == 1:
        return [
            AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="3"),
        ]
    elif scenario == 2:
        return [
            AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"}]),
            AIMessage(content="", tool_calls=[{"name": "subtract", "args": {"a": 12, "b": 3}, "id": "c3", "type": "tool_call"}]),
            AIMessage(content="9"),
        ]
    else:
        return [
            AIMessage(content="", tool_calls=[
                {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"},
                {"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"},
                {"name": "negate", "args": {"x": 5}, "id": "c3", "type": "tool_call"},
            ]),
            AIMessage(content="done"),
        ]


_LC_TOOLS = None


def _get_lc_tools():
    global _LC_TOOLS
    if _LC_TOOLS is None:
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
        _LC_TOOLS = {
            "add": StructuredTool.from_function(add),
            "multiply": StructuredTool.from_function(multiply),
            "subtract": StructuredTool.from_function(subtract),
            "negate": StructuredTool.from_function(negate),
        }
    return _LC_TOOLS


def _make_lc_graph(scenario: int):
    """Compile a graph with a stateless fake LLM. One instance is safe for all concurrent runs."""
    td = _get_lc_tools()
    tools = {1: [td["add"]], 2: [td["add"], td["multiply"], td["subtract"]], 3: [td["add"], td["multiply"], td["negate"]]}[scenario]
    resps = _lc_responses(scenario)
    llm = _FakeLLM(responses=list(resps))
    return create_react_agent(llm, tools)


_LG_INPUT = {"messages": [HumanMessage(content="go")]}


# ═══════════════════════════════════════════════════════════════════════════════
# LANGCHAIN AgentExecutor — pre-LangGraph agent pattern
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent  # noqa: E402
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
    HAS_AGENT_EXECUTOR = True
except ImportError:
    # AgentExecutor was removed in LangChain 1.x. LangChain fully deprecated the
    # LCEL-based agent pattern (AgentExecutor + create_tool_calling_agent) and
    # committed to LangGraph as the sole agent runtime. If you need to benchmark
    # AgentExecutor, install langchain==0.3.x — but note it conflicts with
    # langchain-core>=1.x which langgraph requires.
    HAS_AGENT_EXECUTOR = False


def _make_agent_executor(scenario: int):
    """AgentExecutor + create_tool_calling_agent — the pre-LangGraph LCEL agent pattern.

    Uses the same _FakeLLM and tools as the LangGraph benchmark. AgentExecutor
    manages the tool loop internally (no graph, no Pregel supersteps).

    Note: removed in LangChain 1.x. Only available with langchain==0.3.x.
    """
    if not HAS_AGENT_EXECUTOR:
        return None
    td = _get_lc_tools()
    tools = {
        1: [td["add"]],
        2: [td["add"], td["multiply"], td["subtract"]],
        3: [td["add"], td["multiply"], td["negate"]],
    }[scenario]
    resps = _lc_responses(scenario)
    llm = _FakeLLM(responses=list(resps))
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


_AE_INPUT = {"input": "go"}


# ═══════════════════════════════════════════════════════════════════════════════
# AgentExecutor vs LangGraph — Scenario 2 head-to-head
# ═══════════════════════════════════════════════════════════════════════════════

async def bench_executor_comparison() -> None:
    """Head-to-head: Timbal vs LangGraph create_react_agent vs LangChain AgentExecutor.

    Runs Scenario 2 (3-tool multi-step chain) — the most complex scenario —
    across all three agent implementations. Shows whether the prebuilt LangGraph
    agent or the older AgentExecutor pattern is faster, and how both compare to Timbal.
    """
    if not HAS_AGENT_EXECUTOR:
        section("AgentExecutor vs create_react_agent vs Timbal  (Scenario 2: 3-tool chain)")
        print(f"\n  {DIM}AgentExecutor was removed in LangChain 1.x.{RESET}")
        print(f"  {DIM}LangChain fully deprecated the LCEL agent pattern (AgentExecutor +{RESET}")
        print(f"  {DIM}create_tool_calling_agent) and committed to LangGraph as the sole{RESET}")
        print(f"  {DIM}agent runtime. create_react_agent (langgraph.prebuilt) IS the{RESET}")
        print(f"  {DIM}current official agent API — the benchmark already covers it.{RESET}")
        print(f"\n  {DIM}To test AgentExecutor: install langchain==0.3.x (conflicts with{RESET}")
        print(f"  {DIM}langchain-core>=1.x which langgraph requires).{RESET}")
        return

    section("AgentExecutor vs create_react_agent vs Timbal  (Scenario 2: 3-tool chain)")
    print(f"  {DIM}LLM → add → LLM → multiply → LLM → subtract → LLM → answer{RESET}")
    print(f"  {DIM}Timbal: built-in tracing  |  LG / AE bare: no tracing  |  +Smith: LangChainTracer (HTTP mocked){RESET}")

    SCENARIO = 2
    t_agent  = _make_timbal_agent(SCENARIO)
    lg_graph = _make_lc_graph(SCENARIO)
    ae_exec  = _make_agent_executor(SCENARIO)
    ls_tracer = _make_ls_tracer() if HAS_LANGSMITH else None

    async def t_run():   await t_agent(prompt="go").collect()
    async def lg_run():  await lg_graph.ainvoke(_LG_INPUT)
    async def ae_run():  await ae_exec.ainvoke(_AE_INPUT)
    async def lg_run_traced(): await lg_graph.ainvoke(_LG_INPUT, config={"callbacks": [ls_tracer]})
    async def ae_run_traced(): await ae_exec.ainvoke(_AE_INPUT, config={"callbacks": [ls_tracer]})

    COL_W = 12
    has_smith = HAS_LANGSMITH
    cols     = ["Timbal", "LG react", "AE"] + (["LG+Smith", "AE+Smith"] if has_smith else [])
    runners  = [t_run, lg_run, ae_run]       + ([lg_run_traced, ae_run_traced] if has_smith else [])
    hdr = f"  {'':>{COL_W}}" + "".join(f"  {c:>{COL_W}}" for c in cols)
    sep = f"  {'─'*COL_W}" + f"  {'─'*COL_W}" * len(cols)

    # ── Latency ──────────────────────────────────────────────────────────────
    subsection(f"Latency  (×{N_ITERS} sequential runs)")

    all_samples = []
    for run_fn in runners:
        for _ in range(N_WARMUP):
            await run_fn()
        _clear_traces()
        gc.collect()
        s = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            await run_fn()
            s.append((time.perf_counter() - t0) * 1e6)
        _clear_traces()
        all_samples.append(s)

    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        fn = statistics.mean if p is None else lambda s: pct(s, p)
        vals = [fn(s) for s in all_samples]
        print(f"  {label:>{COL_W}}" + "".join(f"  {fmt_us(v):>{COL_W}}" for v in vals))

    # ── Memory ───────────────────────────────────────────────────────────────
    subsection(f"Memory per run  (×{N_MEM} runs, peak / N)")

    fw_w = 20
    print(f"  {'framework':<{fw_w}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─'*fw_w}  {'─'*12}  {'─'*12}")
    for name, run_fn in zip(cols, runners):
        for _ in range(N_WARMUP):
            await run_fn()
        _clear_traces()
        gc.collect()
        tracemalloc.start()
        for _ in range(N_MEM):
            await run_fn()
            _clear_traces()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {name:<{fw_w}}  {peak/1024:>10.1f} KB  {peak/N_MEM:>10.0f}  B")

    # ── Burst ─────────────────────────────────────────────────────────────────
    n_burst = N_BURST[SCENARIO]
    subsection(f"Burst p50/p95  ({n_burst} concurrent loops)")

    burst_results = []
    for run_fn in runners:
        await asyncio.gather(*[run_fn() for _ in range(5)])
        _clear_traces()
        gc.collect()
        samples: list[float] = []
        async def _timed(fn=run_fn):
            t0 = time.perf_counter()
            await fn()
            samples.append((time.perf_counter() - t0) * 1e6)
        await asyncio.gather(*[_timed() for _ in range(n_burst)])
        _clear_traces()
        burst_results.append(sorted(samples))

    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = [pct(s, p) for s in burst_results]
        print(f"  {label:>{COL_W}}" + "".join(f"  {fmt_us(v):>{COL_W}}" for v in vals))
    wall_parts = [f"{c}: {max(s)/1e3:.1f} ms" for c, s in zip(cols, burst_results)]
    print(f"\n  {DIM}wall: {' | '.join(wall_parts)}{RESET}")

    # ── Throughput ─────────────────────────────────────────────────────────────
    subsection(f"Throughput  ({THROUGHPUT_OPS} loops)")
    print(hdr)
    print(sep)
    for conc in CONCURRENCY_LEVELS:
        ops_row = []
        for run_fn in runners:
            sem = asyncio.Semaphore(conc)
            async def _bounded(fn=run_fn):
                async with sem:
                    await fn()
            gc.collect()
            t0 = time.perf_counter()
            await asyncio.gather(*[_bounded() for _ in range(THROUGHPUT_OPS)])
            ops_row.append(THROUGHPUT_OPS / (time.perf_counter() - t0))
            _clear_traces()
        print(f"  {conc:>{COL_W}}" + "".join(f"  {v:>10.0f}/s" for v in ops_row))


# ═══════════════════════════════════════════════════════════════════════════════
# Per-scenario benchmark
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_NAMES = {
    1: "Single tool call  (LLM → add → LLM → answer)",
    2: "Multi-step  (LLM → add → LLM → mul → LLM → sub → LLM → answer)",
    3: "Parallel tools  (LLM → [add, mul, neg] → LLM → answer)",
}


async def bench_scenario(scenario: int) -> None:
    section(f"Scenario {scenario}: {SCENARIO_NAMES[scenario]}")

    t_agent = _make_timbal_agent(scenario)
    if not _args.timbal_only:
        lg_graph = _make_lc_graph(scenario)
        # Fresh tracer per scenario to avoid order_map accumulation
        ls_tracer = _make_ls_tracer() if HAS_LANGSMITH else None
    else:
        lg_graph = ls_tracer = None

    # ── helpers ──────────────────────────────────────────────────────────────

    async def t_run():
        await t_agent(prompt="go").collect()

    async def lg_run():
        await lg_graph.ainvoke(_LG_INPUT)

    async def lg_run_traced():
        await lg_graph.ainvoke(_LG_INPUT, config={"callbacks": [ls_tracer]})

    # ── Latency (sequential — reuse agent/graph) ────────────────────────────
    subsection(f"Latency  (×{N_ITERS})")

    cols = ["Timbal"]
    if not _args.timbal_only:
        cols.append("LG (bare)")
        if HAS_LANGSMITH:
            cols.append("LG+Smith")
    print(f"  {'':>12}" + "".join(f"  {c:>12}" for c in cols))
    print(f"  {'─' * 12}" + "  ────────────" * len(cols))

    # Timbal
    for _ in range(N_WARMUP):
        await t_run()
    _clear_traces()
    gc.collect()
    t_samples = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        await t_run()
        t_samples.append((time.perf_counter() - t0) * 1e6)
    _clear_traces()

    lg_samples = []
    ls_samples = None
    if not _args.timbal_only:
        # LG bare (async, sequential)
        for _ in range(N_WARMUP):
            await lg_run()
        gc.collect()
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            await lg_run()
            lg_samples.append((time.perf_counter() - t0) * 1e6)

        # LG + LangSmith
        if HAS_LANGSMITH:
            for _ in range(N_WARMUP):
                await lg_run_traced()
            gc.collect()
            ls_samples = []
            for _ in range(N_ITERS):
                t0 = time.perf_counter()
                await lg_run_traced()
                ls_samples.append((time.perf_counter() - t0) * 1e6)

    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = [statistics.mean(t_samples) if p is None else pct(t_samples, p)]
        if lg_samples:
            vals.append(statistics.mean(lg_samples) if p is None else pct(lg_samples, p))
        if ls_samples:
            vals.append(statistics.mean(ls_samples) if p is None else pct(ls_samples, p))
        print(f"  {label:>12}" + "".join(f"  {fmt_us(v)}" for v in vals))

    # ── Memory ───────────────────────────────────────────────────────────────
    subsection(f"Memory  (×{N_MEM} runs)")

    # Timbal
    for _ in range(N_WARMUP):
        await t_run()
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(N_MEM):
        await t_run()
        _clear_traces()
    _, t_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  {'framework':<24}  {'peak':>12}  {'per run':>12}")
    print(f"  {'─' * 24}  {'─' * 12}  {'─' * 12}")
    print(f"  {'Timbal':<24}  {t_peak / 1024:>10.1f} KB  {t_peak / N_MEM:>10.0f}  B")

    if not _args.timbal_only:
        for _ in range(N_WARMUP):
            await lg_run()
        gc.collect()
        tracemalloc.start()
        for _ in range(N_MEM):
            await lg_run()
        _, lg_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {'LG (bare)':<24}  {lg_peak / 1024:>10.1f} KB  {lg_peak / N_MEM:>10.0f}  B")

        if HAS_LANGSMITH:
            for _ in range(N_WARMUP):
                await lg_run_traced()
            gc.collect()
            tracemalloc.start()
            for _ in range(N_MEM):
                await lg_run_traced()
            _, ls_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f"  {'LG + LangSmith':<24}  {ls_peak / 1024:>10.1f} KB  {ls_peak / N_MEM:>10.0f}  B")

    # ── Burst (concurrent) ─────────────────────────────────────────────────
    if HAS_LANGSMITH and not _args.timbal_only:
        ls_tracer = _make_ls_tracer()
    n_burst = N_BURST[scenario] if _args.quick else N_BURST.get(scenario, 50)
    subsection(f"Burst  ({n_burst} concurrent loops)")
    if not _args.timbal_only:
        print(f"  {DIM}Both use 1 shared instance. LangGraph uses native ainvoke().{RESET}")

    # Warmup
    if not _args.timbal_only:
        await asyncio.gather(*[lg_run() for _ in range(5)])
    await asyncio.gather(*[t_run() for _ in range(5)])
    _clear_traces()
    gc.collect()

    t_burst: list[float] = []
    _t_done = 0
    async def t_timed():
        nonlocal _t_done
        t0 = time.perf_counter()
        await t_run()
        elapsed = (time.perf_counter() - t0) * 1e6
        t_burst.append(elapsed)
        _t_done += 1
        print(f"\r  burst [{_t_done:>{len(str(n_burst))}}/{n_burst}]  last {fmt_us(elapsed)}", end="", flush=True)
    await asyncio.gather(*[t_timed() for _ in range(n_burst)])
    print()
    _clear_traces()
    t_burst.sort()

    lg_burst: list[float] = []
    ls_burst = None
    if not _args.timbal_only:
        # LG bare — single shared graph, concurrent ainvoke()
        await asyncio.gather(*[lg_run() for _ in range(5)])
        gc.collect()
        async def lg_timed():
            t0 = time.perf_counter()
            await lg_run()
            lg_burst.append((time.perf_counter() - t0) * 1e6)
        await asyncio.gather(*[lg_timed() for _ in range(n_burst)])
        lg_burst.sort()

        if HAS_LANGSMITH:
            await asyncio.gather(*[lg_run_traced() for _ in range(5)])
            gc.collect()
            ls_burst_list: list[float] = []
            async def ls_timed():
                t0 = time.perf_counter()
                await lg_run_traced()
                ls_burst_list.append((time.perf_counter() - t0) * 1e6)
            await asyncio.gather(*[ls_timed() for _ in range(n_burst)])
            ls_burst = sorted(ls_burst_list)

    cols_h = "  {'Timbal':>12}"
    hdr = f"  {'':>12}  {'Timbal':>12}"
    if lg_burst:
        hdr += f"  {'LG (bare)':>12}"
    if ls_burst:
        hdr += f"  {'LG+Smith':>12}"
    sep = f"  {'─' * 12}  {'─' * 12}"
    if lg_burst:
        sep += f"  {'─' * 12}"
    if ls_burst:
        sep += f"  {'─' * 12}"
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        line = f"  {label:>12}  {fmt_us(pct(t_burst, p))}"
        if lg_burst:
            line += f"  {fmt_us(pct(lg_burst, p))}"
        if ls_burst:
            line += f"  {fmt_us(pct(ls_burst, p))}"
        print(line)
    wall = f"\n  {DIM}wall: Timbal {max(t_burst)/1e3:.1f} ms"
    if lg_burst:
        wall += f"  |  LG bare {max(lg_burst)/1e3:.1f} ms"
    if ls_burst:
        wall += f"  |  LG+Smith {max(ls_burst)/1e3:.1f} ms"
    print(wall + RESET)

    # ── Throughput ───────────────────────────────────────────────────────────
    # Both Timbal and LG run fully concurrently on 1 shared instance.
    subsection(f"Throughput  ({THROUGHPUT_OPS} loops)")
    if not _args.timbal_only:
        print(f"  {DIM}Both use 1 shared instance, semaphore-bounded async.{RESET}")

    if _args.timbal_only:
        print(f"  {'concurrency':>12}  {'Timbal':>12}")
        print(f"  {'─' * 12}  {'─' * 12}")
    else:
        has_ls_tp = HAS_LANGSMITH
        hdr = f"  {'concurrency':>12}  {'Timbal':>12}  {'LG (bare)':>12}"
        sep = f"  {'─' * 12}  {'─' * 12}  {'─' * 12}"
        if has_ls_tp:
            hdr += f"  {'LG+Smith':>12}"
            sep += f"  {'─' * 12}"
        print(hdr)
        print(sep)

    for conc in CONCURRENCY_LEVELS:
        sem = asyncio.Semaphore(conc)

        async def t_bounded():
            async with sem:
                await t_run()

        gc.collect()
        t0 = time.perf_counter()
        await asyncio.gather(*[t_bounded() for _ in range(THROUGHPUT_OPS)])
        t_ops = THROUGHPUT_OPS / (time.perf_counter() - t0)
        _clear_traces()

        if _args.timbal_only:
            print(f"  {conc:>12}  {t_ops:>10.0f}/s")
        else:
            lg_sem = asyncio.Semaphore(conc)

            async def lg_bounded():
                async with lg_sem:
                    await lg_run()

            t0 = time.perf_counter()
            await asyncio.gather(*[lg_bounded() for _ in range(THROUGHPUT_OPS)])
            lg_ops = THROUGHPUT_OPS / (time.perf_counter() - t0)

            line = f"  {conc:>12}  {t_ops:>10.0f}/s  {lg_ops:>10.0f}/s"

            if has_ls_tp:
                ls_sem = asyncio.Semaphore(conc)

                async def ls_bounded():
                    async with ls_sem:
                        await lg_run_traced()

                t0 = time.perf_counter()
                await asyncio.gather(*[ls_bounded() for _ in range(THROUGHPUT_OPS)])
                ls_ops = THROUGHPUT_OPS / (time.perf_counter() - t0)
                line += f"  {ls_ops:>10.0f}/s"

            print(line)


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal vs LangGraph — agent loop benchmark{RESET}")
    print(f"  {N_ITERS} iters · burst {N_BURST} · {N_MEM} mem · {THROUGHPUT_OPS} throughput")
    mode = "LangGraph + LangSmith" if HAS_LANGSMITH else "LangGraph (bare only)"
    print(f"  {mode}")
    print(f"  {DIM}Agent/graph creation excluded. Traces cleared between batches.{RESET}")
    if not _args.timbal_only:
        print(f"  {DIM}Both frameworks: 1 shared instance, native async (ainvoke / .collect()).{RESET}")
        print(f"  {DIM}LLMs faked; step detected from message history — no shared counter state.{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    for scenario in [1, 2, 3]:
        await bench_scenario(scenario)

    await bench_executor_comparison()

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  All measurements reuse 1 pre-built agent/graph (creation excluded).")
    print(f"  LLMs faked via message inspection. Timbal traces cleared between batches.")
    print(f"  Both frameworks run natively async on a single shared instance.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
