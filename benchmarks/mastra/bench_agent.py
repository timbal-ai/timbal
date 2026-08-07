#!/usr/bin/env python3
"""
Timbal vs Mastra — full agent loop benchmark (cross-language).

Three scenarios, identical pipelines on both frameworks:
  1. Single tool call:   prompt → LLM → tool(add) → LLM → answer
  2. Multi-step (3 tools): prompt → LLM → add → LLM → mul → LLM → sub → LLM → answer
  3. Parallel tools:     prompt → LLM → [add, mul, neg] concurrent → LLM → answer

Mastra is a TypeScript framework, so each side runs in its native runtime:
Timbal on CPython/asyncio (in this process), Mastra on Node/V8 (spawned as a
subprocess running bench_agent.mjs with the same methodology). Numbers compare
the full framework+runtime stacks — the thing a user actually deploys — not
language-neutral algorithm quality. See README.md for what is and isn't
comparable; memory in particular uses different instruments per runtime
(tracemalloc vs V8 heap growth) and must not be ratio'd across languages.

All LLMs are faked via message-history inspection. Observability: Timbal
built-in tracing (always on); Mastra shown both bare and with
@mastra/observability enabled (export mocked).

Run:
    uv run python benchmarks/mastra/bench_agent.py
    uv run python benchmarks/mastra/bench_agent.py --quick
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
parser.add_argument("--timbal-only", action="store_true", help="Skip Mastra, measure Timbal only")
parser.add_argument("--mastra-json", type=Path, default=None, help="Reuse a previous Mastra JSON result instead of running Node")
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
# MASTRA — spawn the Node-side benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_mastra_side(script: str) -> dict | None:
    if _args.timbal_only:
        return None
    if _args.mastra_json:
        return json.loads(_args.mastra_json.read_text())
    node = shutil.which("node")
    if node is None:
        print(f"{DIM}  node not found on PATH — running Timbal only.{RESET}")
        return None
    if not (BENCH_DIR / "node_modules").exists():
        print(f"{DIM}  benchmarks/mastra/node_modules missing — run `npm install` in benchmarks/mastra first.{RESET}")
        return None
    cmd = [node, "--expose-gc", script] + (["--quick"] if _args.quick else [])
    print(f"{DIM}  running Mastra side: {' '.join(cmd[1:])}…{RESET}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=BENCH_DIR, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"{DIM}  Mastra side failed (exit {proc.returncode}) — running Timbal only.{RESET}")
        return None
    return json.loads(proc.stdout)


# ═══════════════════════════════════════════════════════════════════════════════
# TIMBAL — fake LLM agent factory (identical to benchmarks/langchain)
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Agent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402
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
# Timbal measurement (same procedure as the Node side)
# ═══════════════════════════════════════════════════════════════════════════════


async def measure_timbal(scenario: int) -> dict:
    agent = _make_timbal_agent(scenario)

    async def run():
        await agent(prompt="go").collect()

    # Latency
    for _ in range(N_WARMUP):
        await run()
    _clear_traces()
    gc.collect()
    lat = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        await run()
        lat.append((time.perf_counter() - t0) * 1e6)
    _clear_traces()

    # Memory
    for _ in range(N_WARMUP):
        await run()
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(N_MEM):
        await run()
        _clear_traces()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Burst
    n_burst = N_BURST[scenario]
    await asyncio.gather(*[run() for _ in range(5)])
    _clear_traces()
    gc.collect()
    burst: list[float] = []

    async def timed():
        t0 = time.perf_counter()
        await run()
        burst.append((time.perf_counter() - t0) * 1e6)

    await asyncio.gather(*[timed() for _ in range(n_burst)])
    _clear_traces()

    # Throughput
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
        _clear_traces()

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


def print_scenario(key: str, timbal: dict, mastra: dict | None) -> None:
    section(f"Scenario {key}: {SCENARIO_NAMES[key]}")

    cols = ["Timbal"]
    frames = [timbal]
    if mastra:
        cols += ["Mastra", "Mastra+obs"]
        frames += [mastra["bare"], mastra["obs"]]

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
    if mastra:
        for name, f in [("Mastra", mastra["bare"]), ("Mastra+obs", mastra["obs"])]:
            print(f"  {name:<{COL_W + 12}}  {f['mem_per_run_bytes']:>12,.0f} B  {'V8 heap growth/N (≈)':<24}")

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


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal vs Mastra — agent loop benchmark (cross-language){RESET}")
    print(f"  {N_ITERS} iters · burst {N_BURST} · {N_MEM} mem · {THROUGHPUT_OPS} throughput")
    print(f"  {DIM}Timbal: CPython {sys.version_info.major}.{sys.version_info.minor}/asyncio, built-in tracing (always on).{RESET}")
    print(f"  {DIM}Mastra: Node/V8 subprocess, AI SDK MockLanguageModelV4, bare + observability (export mocked).{RESET}")
    print(f"  {DIM}Same scenarios, same fake-LLM-by-message-inspection, same procedure per side.{RESET}")
    print(f"  {DIM}Cross-runtime comparison — see README.md for what is and isn't comparable.{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print()

    mastra = run_mastra_side("bench_agent.mjs")
    if mastra:
        m = mastra["meta"]
        print(f"{DIM}  Mastra side: node {m['node']} · @mastra/core {m['mastra_core']} · @mastra/observability {m['mastra_observability']} · ai {m['ai_sdk']}{RESET}")

    for key in ["1", "2", "3"]:
        timbal = await measure_timbal(int(key))
        print_scenario(key, timbal, mastra["scenarios"][key] if mastra else None)

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print("  All measurements reuse 1 pre-built agent (creation excluded).")
    print("  LLMs faked via message inspection on both sides — no network, no API keys.")
    print("  Each framework runs in its native runtime; numbers include runtime differences.")
    print("  Memory uses per-runtime instruments (tracemalloc vs V8 heap) — not cross-comparable.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
