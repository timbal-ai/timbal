#!/usr/bin/env python3
"""
Runtime calibration — LangGraph Python vs LangGraph.js (vs Timbal) agent loop.

Matching workloads on the Python and JavaScript implementations of LangGraph,
plus Timbal. Ratios include both implementation and runtime differences, and
must not be used as a pure language factor to normalize other frameworks.

Scenarios (identical to benchmarks/langchain and benchmarks/mastra):
  1. Single tool call   2. Multi-step (3 tools)   3. Parallel tools

All LLMs faked via message-history inspection. All columns are bare (no
observability) except Timbal, whose built-in tracing is always on — the
calibration ratio itself is computed LG.py-bare vs LG.js-bare, same framework,
same (lack of) tracing.

Run:
    uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_agent.py
    uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_agent.py --quick
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
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--timbal-only", action="store_true", help="Skip LangGraph on both runtimes")
parser.add_argument("--lgjs-json", type=Path, default=None, help="Reuse a previous LG.js JSON result instead of running Node")
_args, _ = parser.parse_known_args()

N_ITERS = 20 if _args.quick else 100
N_WARMUP = 3 if _args.quick else 10
N_BURST = {1: 20, 2: 10, 3: 15}
N_MEM = 20 if _args.quick else 100
THROUGHPUT_OPS = 30 if _args.quick else 200
CONCURRENCY_LEVELS = [1, 10, 50]
WIDTH = 76

BENCH_DIR = Path(__file__).parent

# ── Display helpers ──────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
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
# LANGGRAPH.JS — spawn the Node-side benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_lgjs_side(script: str) -> dict | None:
    if _args.timbal_only:
        return None
    if _args.lgjs_json:
        return json.loads(_args.lgjs_json.read_text())
    node = shutil.which("node")
    if node is None:
        print(f"{DIM}  node not found on PATH — skipping LangGraph.js.{RESET}")
        return None
    if not (BENCH_DIR / "node_modules").exists():
        print(f"{DIM}  benchmarks/langgraph_js/node_modules missing — run `npm install` there first.{RESET}")
        return None
    cmd = [node, "--expose-gc", script] + (["--quick"] if _args.quick else [])
    print(f"{DIM}  running LangGraph.js side: {' '.join(cmd[1:])}…{RESET}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=BENCH_DIR, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"{DIM}  LangGraph.js side failed (exit {proc.returncode}).{RESET}")
        return None
    return json.loads(proc.stdout)


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — fake LLM agent factory (identical to benchmarks/langchain)
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Agent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.state import get_run_context, set_run_context  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402


async def _fresh_collect(runnable, **kwargs):
    """Each sample is a root run; explicit parent_id still links session turns."""
    previous = get_run_context()
    set_run_context(None)
    try:
        return await runnable(**kwargs).collect()
    finally:
        set_run_context(previous)


from timbal.types.content import TextContent, ToolResultContent, ToolUseContent  # noqa: E402
from timbal.types.message import Message  # noqa: E402


def _clear_traces():
    InMemoryTracingProvider._storage.clear()


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
# LANGGRAPH PYTHON — same fake LLM pattern as benchmarks/langchain
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    HAS_LANGGRAPH_PY = True
except ImportError:
    HAS_LANGGRAPH_PY = False

if HAS_LANGGRAPH_PY:
    class _FakeLLM(FakeMessagesListChatModel):
        """Stateless fake LLM: counts ToolMessages to find the current step."""

        def bind_tools(self, tools, **kw):
            return self

        def _step(self, messages) -> int:
            return sum(1 for m in messages if isinstance(m, ToolMessage))

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult
            idx = self._step(messages) % len(self.responses)
            return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult
            idx = self._step(messages) % len(self.responses)
            return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

    def _lg_responses(scenario: int) -> list:
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
        return [
            AIMessage(content="", tool_calls=[
                {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"},
                {"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"},
                {"name": "negate", "args": {"x": 5}, "id": "c3", "type": "tool_call"},
            ]),
            AIMessage(content="done"),
        ]

    def _make_lg_graph(scenario: int):
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

        td = {n.__name__: StructuredTool.from_function(n) for n in (add, multiply, subtract, negate)}
        tools = {
            1: [td["add"]],
            2: [td["add"], td["multiply"], td["subtract"]],
            3: [td["add"], td["multiply"], td["negate"]],
        }[scenario]
        return create_react_agent(_FakeLLM(responses=_lg_responses(scenario)), tools)

    _LG_INPUT = {"messages": [HumanMessage(content="go")]}


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement (same procedure as the Node side)
# ═══════════════════════════════════════════════════════════════════════════════


async def measure_python(run, scenario: int, clear=None) -> dict:
    async def _noop():
        pass

    clear = clear or (lambda: None)

    for _ in range(N_WARMUP):
        await run()
    clear()
    gc.collect()
    lat = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        await run()
        lat.append((time.perf_counter() - t0) * 1e6)
    clear()

    for _ in range(N_WARMUP):
        await run()
    clear()
    gc.collect()
    tracemalloc.start()
    for _ in range(N_MEM):
        await run()
        clear()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_burst = N_BURST[scenario]
    await asyncio.gather(*[run() for _ in range(5)])
    clear()
    gc.collect()
    burst: list[float] = []

    async def timed():
        t0 = time.perf_counter()
        await run()
        burst.append((time.perf_counter() - t0) * 1e6)

    await asyncio.gather(*[timed() for _ in range(n_burst)])
    clear()

    tp = {}
    for conc in CONCURRENCY_LEVELS:
        sem = asyncio.Semaphore(conc)

        async def bounded():
            async with sem:
                await run()

        gc.collect()
        t0 = time.perf_counter()
        await asyncio.gather(*[bounded() for _ in range(THROUGHPUT_OPS)])
        tp[str(conc)] = THROUGHPUT_OPS / (time.perf_counter() - t0)
        clear()

    return {
        "latency_us": {
            "mean": statistics.mean(lat),
            "p50": pct(lat, 50), "p75": pct(lat, 75), "p95": pct(lat, 95), "p99": pct(lat, 99),
            "max": max(lat),
        },
        "mem_per_run_bytes": peak / N_MEM,
        "burst_us": {
            "p50": pct(burst, 50), "p75": pct(burst, 75), "p95": pct(burst, 95), "p99": pct(burst, 99),
            "max": max(burst),
        },
        "throughput_ops_s": tp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Combined output
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_NAMES = {
    "1": "Single tool call  (LLM → add → LLM → answer)",
    "2": "Multi-step  (LLM → add → LLM → mul → LLM → sub → LLM → answer)",
    "3": "Parallel tools  (LLM → [add, mul, neg] → LLM → answer)",
}

COL_W = 12


def print_scenario(key: str, timbal: dict, lgpy: dict | None, lgjs: dict | None) -> None:
    section(f"Scenario {key}: {SCENARIO_NAMES[key]}")

    cols = ["Timbal"]
    frames = [timbal]
    if lgpy:
        cols.append("LG.py")
        frames.append(lgpy)
    if lgjs:
        cols.append("LG.js")
        frames.append(lgjs)

    hdr = f"  {'':>{COL_W}}" + "".join(f"  {c:>{COL_W}}" for c in cols)
    sep = f"  {'─' * COL_W}" + f"  {'─' * COL_W}" * len(cols)

    subsection(f"Latency  (×{N_ITERS} sequential runs)")
    print(hdr)
    print(sep)
    for label in ["mean", "p50", "p95", "p99"]:
        vals = [f["latency_us"][label] for f in frames]
        print(f"  {label:>{COL_W}}" + "".join(f"  {fmt_us(v):>{COL_W}}" for v in vals))

    subsection(f"Memory per run  (×{N_MEM} runs) — different instruments, do not ratio across runtimes")
    print(f"  {'framework':<{COL_W + 12}}  {'per run':>14}  {'method':<24}")
    print(f"  {'─' * (COL_W + 12)}  {'─' * 14}  {'─' * 24}")
    print(f"  {'Timbal':<{COL_W + 12}}  {timbal['mem_per_run_bytes']:>12,.0f} B  {'tracemalloc peak/N':<24}")
    if lgpy:
        print(f"  {'LG.py':<{COL_W + 12}}  {lgpy['mem_per_run_bytes']:>12,.0f} B  {'tracemalloc peak/N':<24}")
    if lgjs:
        print(f"  {'LG.js':<{COL_W + 12}}  {lgjs['mem_per_run_bytes']:>12,.0f} B  {'V8 heap growth/N (≈)':<24}")

    n_burst = N_BURST[int(key)]
    subsection(f"Burst p50/p95  ({n_burst} concurrent runs)")
    print(hdr)
    print(sep)
    for label in ["p50", "p75", "p95", "p99", "max"]:
        vals = [f["burst_us"][label] for f in frames]
        print(f"  {label:>{COL_W}}" + "".join(f"  {fmt_us(v):>{COL_W}}" for v in vals))

    subsection(f"Throughput  ({THROUGHPUT_OPS} runs)")
    print(hdr)
    print(sep)
    for conc in CONCURRENCY_LEVELS:
        vals = [f["throughput_ops_s"][str(conc)] for f in frames]
        print(f"  {conc:>{COL_W}}" + "".join(f"  {v:>10.0f}/s" for v in vals))

    if lgpy and lgjs:
        subsection("Implementation/runtime comparison  (LG.py ÷ LG.js)")
        p50 = lgpy["latency_us"]["p50"] / lgjs["latency_us"]["p50"]
        mean = lgpy["latency_us"]["mean"] / lgjs["latency_us"]["mean"]
        tp10 = lgjs["throughput_ops_s"]["10"] / lgpy["throughput_ops_s"]["10"]
        print(f"  latency p50: {p50:.2f}x  ·  latency mean: {mean:.2f}x  ·  throughput c=10: {tp10:.2f}x")


async def _verify_agent_loops():
    """Reject shortcuts caused by leaked history before measuring anything."""
    for scenario, calls, answer in ((1, 2, "3"), (2, 4, "9"), (3, 2, "done")):
        agent = _make_timbal_agent(scenario)
        for _ in range(2):
            before = agent.model.call_count
            out = await _fresh_collect(agent, prompt="go")
            assert out.status.code == "success", out.status
            assert out.output.collect_text() == answer, out.output
            assert agent.model.call_count - before == calls, "Agent skipped tool steps"
        before = agent.model.call_count
        outputs = await asyncio.gather(*[_fresh_collect(agent, prompt="go") for _ in range(4)])
        assert agent.model.call_count - before == 4 * calls, "Concurrent runs shared history"
        assert all(out.status.code == "success" and out.output.collect_text() == answer for out in outputs)
    if not _args.timbal_only and HAS_LANGGRAPH_PY:
        for scenario, count, answer in ((1, 1, "3"), (2, 3, "9"), (3, 3, "done")):
            graph = _make_lg_graph(scenario)
            for _ in range(2):
                result = await graph.ainvoke(_LG_INPUT)
                messages = result["messages"]
                assert messages[-1].content == answer, messages[-1]
                assert sum(isinstance(m, ToolMessage) for m in messages) == count
    _clear_traces()


async def main() -> None:
    await _verify_agent_loops()
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Runtime calibration — LangGraph Python vs LangGraph.js (agent loop){RESET}")
    print(f"  {N_ITERS} iters · burst {N_BURST} · {N_MEM} mem · {THROUGHPUT_OPS} throughput")
    print(f"  {DIM}LG.py÷LG.js compares both implementations and runtimes; it is not a pure language factor.{RESET}")
    print(f"  {DIM}Timbal shown for reference (built-in tracing always on; LG columns are bare).{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print()

    if not HAS_LANGGRAPH_PY and not _args.timbal_only:
        print(f"{DIM}  langgraph (Python) not importable — run via:{RESET}")
        print(f"{DIM}  uv run --no-sync --with langgraph --with langchain-core python {Path(__file__).relative_to(Path.cwd())}{RESET}")

    lgjs = run_lgjs_side("bench_agent.mjs")
    if lgjs:
        m = lgjs["meta"]
        print(f"{DIM}  LG.js side: node {m['node']} · @langchain/langgraph {m['langgraph']} · @langchain/core {m['langchain_core']}{RESET}")

    ratios = []
    for key in ["1", "2", "3"]:
        scenario = int(key)
        t_agent = _make_timbal_agent(scenario)

        async def t_run(agent=t_agent):
            await _fresh_collect(agent, prompt="go")

        timbal = await measure_python(t_run, scenario, clear=_clear_traces)

        lgpy = None
        if HAS_LANGGRAPH_PY and not _args.timbal_only:
            graph = _make_lg_graph(scenario)

            async def lg_run(g=graph):
                await g.ainvoke(_LG_INPUT)

            lgpy = await measure_python(lg_run, scenario)

        lgjs_s = lgjs["scenarios"][key]["bare"] if lgjs else None
        print_scenario(key, timbal, lgpy, lgjs_s)
        if lgpy and lgjs_s:
            ratios.append(lgpy["latency_us"]["p50"] / lgjs_s["latency_us"]["p50"])

    print()
    print(f"{DIM}{'─' * WIDTH}")
    if ratios:
        print(f"  Comparison summary: LG.py ÷ LG.js latency p50 spans {min(ratios):.2f}-{max(ratios):.2f}x across")
        print("  matching agent loops (>1 = LG.js faster on this workload).")
        print("  Do not apply this ratio to unrelated frameworks.")
    print("  All measurements reuse pre-built agents/graphs (creation excluded).")
    print("  LLMs faked via message inspection on all sides — no network, no API keys.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
