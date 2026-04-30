#!/usr/bin/env python3
# ruff: noqa: T201
"""
Timbal vs Google ADK - full agent loop benchmark.

Three scenarios, each running the identical pipeline on both frameworks:
  1. Single tool call:   prompt -> LLM -> tool(add) -> LLM -> answer
  2. Multi-step:         prompt -> LLM -> add -> LLM -> mul -> LLM -> sub -> LLM -> answer
  3. Parallel tools:     prompt -> LLM -> [add, mul, neg] concurrent -> LLM -> answer

Timbal uses TestModel for fully offline deterministic LLM simulation.
Google ADK uses a custom BaseLlm implementation, also fully offline.

Run:
    uv run --with google-adk python benchmarks/google_adk/bench_agent.py
    uv run --with google-adk python benchmarks/google_adk/bench_agent.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import statistics
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import structlog  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--quick", action="store_true")
_args, _ = parser.parse_known_args()

N_ITERS = 20 if _args.quick else 100
N_WARMUP = 3 if _args.quick else 10
N_MEM = 20 if _args.quick else 100
N_BURST = {1: 20, 2: 10, 3: 15} if _args.quick else {1: 50, 2: 30, 3: 40}
TP_OPS = 30 if _args.quick else 200
CONC_LEVELS = [1, 10, 50]
WIDTH = 96

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
YELLOW = "\033[33m"


def section(title: str) -> None:
    print()
    print(f"{BOLD}{CYAN}{'-' * WIDTH}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'-' * WIDTH}{RESET}")


def subsection(title: str) -> None:
    print(f"\n  {BOLD}{title}{RESET}")


def fmt_us(us: float) -> str:
    if us >= 1_000:
        return f"{us / 1_000:>8.2f} ms"
    return f"{us:>8.1f} us"


def pct(samples: list[float], p: float) -> float:
    idx = min(int(len(samples) * p / 100), len(samples) - 1)
    return sorted(samples)[idx]


from timbal import Agent as TimbalAgent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent  # noqa: E402
from timbal.types.message import Message  # noqa: E402


def _clear_timbal_traces() -> None:
    InMemoryTracingProvider._storage.clear()


def _count_timbal_tool_results(messages) -> int:
    return sum(
        1
        for msg in messages
        for content in (msg.content if hasattr(msg, "content") and msg.content else [])
        if isinstance(content, ToolResultContent)
    )


def _make_timbal_agent(scenario: int) -> TimbalAgent:
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
            if _count_timbal_tool_results(messages) == 0:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2})],
                )
            return Message(role="assistant", content=[TextContent(type="text", text="3")], stop_reason="end_turn")

    elif scenario == 2:

        def handler(messages):
            step = _count_timbal_tool_results(messages)
            if step == 0:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2})],
                )
            if step == 1:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(type="tool_use", id="c2", name="multiply", input={"a": 3, "b": 4})],
                )
            if step == 2:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(type="tool_use", id="c3", name="subtract", input={"a": 12, "b": 3})],
                )
            return Message(role="assistant", content=[TextContent(type="text", text="9")], stop_reason="end_turn")

    else:

        def handler(messages):
            if _count_timbal_tool_results(messages) == 0:
                return Message(
                    role="assistant",
                    content=[
                        ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2}),
                        ToolUseContent(type="tool_use", id="c2", name="multiply", input={"a": 3, "b": 4}),
                        ToolUseContent(type="tool_use", id="c3", name="negate", input={"x": 5}),
                    ],
                )
            return Message(role="assistant", content=[TextContent(type="text", text="done")], stop_reason="end_turn")

    return TimbalAgent(name="bench_agent", model=TestModel(handler=handler), tools=tools)


HAS_ADK = False

try:
    from google.adk.agents import Agent as AdkAgent  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.models.base_llm import BaseLlm  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.models.llm_response import LlmResponse  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.runners import Runner  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.sessions import InMemorySessionService  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.genai import types as genai_types  # noqa: E402

    HAS_ADK = True

    class _FakeAdkLlm(BaseLlm):
        scenario: int

        def supported_models(self) -> list[str]:
            return ["fake-adk"]

        async def generate_content_async(self, llm_request, stream: bool = False):
            del stream
            step = _count_adk_function_responses(llm_request.contents)
            if self.scenario == 1:
                if step == 0:
                    yield _adk_response([_adk_call("add", {"a": 1, "b": 2})])
                else:
                    yield _adk_response([genai_types.Part.from_text(text="3")])
            elif self.scenario == 2:
                if step == 0:
                    yield _adk_response([_adk_call("add", {"a": 1, "b": 2})])
                elif step == 1:
                    yield _adk_response([_adk_call("multiply", {"a": 3, "b": 4})])
                elif step == 2:
                    yield _adk_response([_adk_call("subtract", {"a": 12, "b": 3})])
                else:
                    yield _adk_response([genai_types.Part.from_text(text="9")])
            else:
                if step == 0:
                    yield _adk_response(
                        [
                            _adk_call("add", {"a": 1, "b": 2}),
                            _adk_call("multiply", {"a": 3, "b": 4}),
                            _adk_call("negate", {"x": 5}),
                        ]
                    )
                else:
                    yield _adk_response([genai_types.Part.from_text(text="done")])

    def _adk_call(name: str, args: dict) -> object:
        return genai_types.Part.from_function_call(name=name, args=args)

    def _adk_response(parts: list) -> object:
        return LlmResponse(content=genai_types.Content(role="model", parts=parts))

    def _count_adk_function_responses(contents) -> int:
        return sum(
            1
            for content in (contents or [])
            for part in (content.parts or [])
            if getattr(part, "function_response", None)
        )

    def _adk_prompt() -> object:
        return genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="calculate")])

    def _make_adk_agent(scenario: int) -> object:
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
        return AdkAgent(
            name="bench_agent",
            model=_FakeAdkLlm(model="fake-adk", scenario=scenario),
            instruction="Use the available tools to calculate the answer.",
            tools=tools,
        )

    class _AdkBench:
        def __init__(self, scenario: int) -> None:
            self.app_name = f"google_adk_bench_{scenario}"
            self.user_id = "bench_user"
            self.session_service = InMemorySessionService()
            self.runner = Runner(
                app_name=self.app_name,
                agent=_make_adk_agent(scenario),
                session_service=self.session_service,
            )

        async def make_sessions(self, n: int) -> list[str]:
            sessions = []
            for _ in range(n):
                session = await self.session_service.create_session(app_name=self.app_name, user_id=self.user_id)
                sessions.append(session.id)
            return sessions

        async def run(self, session_id: str) -> None:
            async for _ in self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=_adk_prompt(),
            ):
                pass

except ImportError:
    pass


async def _run_timbal(agent: TimbalAgent) -> None:
    _clear_timbal_traces()
    await agent(prompt="calculate").collect()


async def _latency_timbal(agent: TimbalAgent) -> list[float]:
    for _ in range(N_WARMUP):
        await _run_timbal(agent)

    samples = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter_ns()
        await _run_timbal(agent)
        samples.append((time.perf_counter_ns() - t0) / 1_000)
    return samples


async def _latency_adk(bench) -> list[float]:
    sessions = await bench.make_sessions(N_WARMUP + N_ITERS)
    for session_id in sessions[:N_WARMUP]:
        await bench.run(session_id)

    samples = []
    for session_id in sessions[N_WARMUP:]:
        t0 = time.perf_counter_ns()
        await bench.run(session_id)
        samples.append((time.perf_counter_ns() - t0) / 1_000)
    return samples


async def _memory_timbal(agent: TimbalAgent) -> int:
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(N_MEM):
        await _run_timbal(agent)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    total = sum(stat.size_diff for stat in after.compare_to(before, "lineno"))
    return max(total, 0) // N_MEM


async def _memory_adk(bench) -> int:
    sessions = await bench.make_sessions(N_MEM)
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for session_id in sessions:
        await bench.run(session_id)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    total = sum(stat.size_diff for stat in after.compare_to(before, "lineno"))
    return max(total, 0) // N_MEM


async def _burst_timbal(agent: TimbalAgent, n: int) -> list[float]:
    async def one() -> float:
        t0 = time.perf_counter_ns()
        await _run_timbal(agent)
        return (time.perf_counter_ns() - t0) / 1_000

    return await asyncio.gather(*(one() for _ in range(n)))


async def _burst_adk(bench, n: int) -> list[float]:
    sessions = await bench.make_sessions(n)

    async def one(session_id: str) -> float:
        t0 = time.perf_counter_ns()
        await bench.run(session_id)
        return (time.perf_counter_ns() - t0) / 1_000

    return await asyncio.gather(*(one(session_id) for session_id in sessions))


async def _throughput_timbal(agent: TimbalAgent, concurrency: int) -> float:
    sem = asyncio.Semaphore(concurrency)

    async def one() -> None:
        async with sem:
            await _run_timbal(agent)

    t0 = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(TP_OPS)))
    return TP_OPS / (time.perf_counter() - t0)


async def _throughput_adk(bench, concurrency: int) -> float:
    sessions = await bench.make_sessions(TP_OPS)
    sem = asyncio.Semaphore(concurrency)

    async def one(session_id: str) -> None:
        async with sem:
            await bench.run(session_id)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(session_id) for session_id in sessions))
    return TP_OPS / (time.perf_counter() - t0)


def _print_latency(name: str, samples: list[float]) -> None:
    print(
        f"    {name:<10} mean {fmt_us(statistics.mean(samples))}  "
        f"p50 {fmt_us(pct(samples, 50))}  p95 {fmt_us(pct(samples, 95))}"
    )


async def run_scenario(scenario: int, title: str) -> None:
    section(f"Scenario {scenario}: {title}")
    timbal_agent = _make_timbal_agent(scenario)
    adk_bench = _AdkBench(scenario) if HAS_ADK else None

    subsection("Latency")
    timbal_lat = await _latency_timbal(timbal_agent)
    _print_latency("Timbal", timbal_lat)
    if adk_bench:
        adk_lat = await _latency_adk(adk_bench)
        _print_latency("ADK", adk_lat)
    else:
        print(f"    {YELLOW}ADK skipped: install with `uv run --with google-adk ...`{RESET}")

    subsection("Memory")
    timbal_mem = await _memory_timbal(timbal_agent)
    print(f"    {'Timbal':<10} {timbal_mem:>8,} B/run")
    if adk_bench:
        adk_mem = await _memory_adk(adk_bench)
        print(f"    {'ADK':<10} {adk_mem:>8,} B/run")

    subsection("Burst")
    burst_n = N_BURST[scenario]
    t0 = time.perf_counter()
    timbal_burst = await _burst_timbal(timbal_agent, burst_n)
    timbal_wall = (time.perf_counter() - t0) * 1_000
    print(f"    {'Timbal':<10} p50 {fmt_us(pct(timbal_burst, 50))}  wall {timbal_wall:>8.2f} ms")
    if adk_bench:
        t0 = time.perf_counter()
        adk_burst = await _burst_adk(adk_bench, burst_n)
        adk_wall = (time.perf_counter() - t0) * 1_000
        print(f"    {'ADK':<10} p50 {fmt_us(pct(adk_burst, 50))}  wall {adk_wall:>8.2f} ms")

    subsection("Throughput")
    for concurrency in CONC_LEVELS:
        timbal_tp = await _throughput_timbal(timbal_agent, concurrency)
        if adk_bench:
            adk_tp = await _throughput_adk(adk_bench, concurrency)
            print(f"    c={concurrency:<3} Timbal {timbal_tp:>8.0f}/s   ADK {adk_tp:>8.0f}/s")
        else:
            print(f"    c={concurrency:<3} Timbal {timbal_tp:>8.0f}/s")


async def main() -> None:
    print(f"{BOLD}Timbal vs Google ADK - Agent Loop Benchmark{RESET}")
    print(f"{DIM}mode={'quick' if _args.quick else 'full'}; no API calls; sessions pre-created outside timing{RESET}")

    await run_scenario(1, "single tool")
    await run_scenario(2, "3-step chain")
    await run_scenario(3, "parallel tools")


if __name__ == "__main__":
    asyncio.run(main())
