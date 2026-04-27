#!/usr/bin/env python3
# ruff: noqa: T201
"""
Timbal delegation vs Google ADK transfer benchmark.

This is not a DAG/workflow benchmark. Google ADK has agent transfer/sub-agent routing,
not a first-class workflow engine. The ADK side uses the real transfer_to_agent tool from
a root agent to a worker sub-agent. The Timbal side uses the closest equivalent
composition: a supervisor agent calls a worker agent as a tool, then returns the result.

Run:
    uv run --with google-adk python benchmarks/google_adk/bench_transfer.py
    uv run --with google-adk python benchmarks/google_adk/bench_transfer.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
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
N_BURST = 15 if _args.quick else 40
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


def needed_runs() -> int:
    return (N_WARMUP + N_ITERS) + (N_WARMUP + N_MEM) + min(5, N_BURST) + N_BURST + TP_OPS * len(CONC_LEVELS) + 5


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


def make_timbal_delegation() -> TimbalAgent:
    worker = TimbalAgent(
        name="worker",
        model=TestModel(responses=[Message(role="assistant", content=[TextContent(type="text", text="worker-done")])]),
        tools=[],
    )

    async def worker_tool(input: str) -> str:
        """Delegate work to the worker agent."""
        result = await worker(prompt=input).collect()
        return result.output.collect_text()

    def supervisor_handler(messages):
        if _count_timbal_tool_results(messages) == 0:
            return Message(
                role="assistant",
                content=[ToolUseContent(type="tool_use", id="c1", name="worker_tool", input={"input": "go"})],
            )
        return Message(
            role="assistant",
            content=[TextContent(type="text", text="worker-done")],
            stop_reason="end_turn",
        )

    return TimbalAgent(name="supervisor", model=TestModel(handler=supervisor_handler), tools=[worker_tool])


async def run_timbal(agent: TimbalAgent) -> str:
    _clear_timbal_traces()
    result = await agent(prompt="go").collect()
    return result.output.collect_text()


HAS_ADK = False

try:
    from google.adk.agents import Agent as AdkAgent  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.models.base_llm import BaseLlm  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.models.llm_response import LlmResponse  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.runners import Runner  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.sessions import InMemorySessionService  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.genai import types as genai_types  # noqa: E402

    HAS_ADK = True

    class _FakeTransferLlm(BaseLlm):
        role: str

        def supported_models(self) -> list[str]:
            return ["fake-adk"]

        async def generate_content_async(self, llm_request, stream: bool = False):
            del llm_request, stream
            if self.role == "root":
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part.from_function_call(
                                name="transfer_to_agent",
                                args={"agent_name": "worker"},
                            )
                        ],
                    )
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part.from_text(text="worker-done")],
                    )
                )

    class AdkTransferBench:
        def __init__(self) -> None:
            self.app_name = "google_adk_transfer_bench"
            self.user_id = "bench_user"
            self.session_service = InMemorySessionService()
            worker = AdkAgent(
                name="worker",
                description="Worker agent for delegated requests.",
                model=_FakeTransferLlm(model="fake-adk", role="worker"),
            )
            root = AdkAgent(
                name="root",
                model=_FakeTransferLlm(model="fake-adk", role="root"),
                instruction="Transfer the request to the worker.",
                sub_agents=[worker],
            )
            self.runner = Runner(app_name=self.app_name, agent=root, session_service=self.session_service)
            self._sessions: list[str] = []

        async def prepare(self, n: int) -> None:
            for _ in range(n):
                session = await self.session_service.create_session(app_name=self.app_name, user_id=self.user_id)
                self._sessions.append(session.id)

        async def run(self) -> str:
            session_id = self._sessions.pop()
            final_text = ""
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="go")]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text or ""
            return final_text

except ImportError as e:
    print(f"Google ADK not found or incompatible: {e}")


async def _latency_async(factory, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        await factory()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)
    return samples


async def _memory_async(factory, n: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        await factory()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await factory()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


async def _burst_async(factory, n: int) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    gc.collect()
    samples: list[float] = []

    async def _timed() -> None:
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    await asyncio.gather(*[_timed() for _ in range(n)])
    return sorted(samples)


async def _throughput_async(factory, total: int, conc: int) -> float:
    sem = asyncio.Semaphore(conc)

    async def _bounded() -> None:
        async with sem:
            await factory()

    gc.collect()
    t0 = time.perf_counter()
    await asyncio.gather(*[_bounded() for _ in range(total)])
    elapsed = time.perf_counter() - t0
    return total / elapsed


@dataclasses.dataclass
class RunData:
    latency: list[float]
    mem_peak: float
    mem_per: float
    burst: list[float]
    throughput: list[float]


async def collect(label: str, factory) -> RunData:
    print(f"    {label} latency...", end=" ", flush=True)
    lat = await _latency_async(factory, N_ITERS, N_WARMUP)
    print(f"p50={fmt_us(pct(lat, 50)).strip()}", flush=True)

    print(f"    {label} memory...", end=" ", flush=True)
    mem_peak, mem_per = await _memory_async(factory, N_MEM, N_WARMUP)
    print(f"{mem_peak / 1024:.0f} KB peak", flush=True)

    print(f"    {label} burst ({N_BURST} concurrent)...", end=" ", flush=True)
    burst = await _burst_async(factory, N_BURST)
    print(f"p50={fmt_us(pct(burst, 50)).strip()}", flush=True)

    tp = []
    for conc in CONC_LEVELS:
        print(f"    {label} throughput conc={conc}...", end=" ", flush=True)
        ops = await _throughput_async(factory, TP_OPS, conc)
        tp.append(ops)
        print(f"{ops:.0f}/s", flush=True)

    return RunData(latency=lat, mem_peak=mem_peak, mem_per=mem_per, burst=burst, throughput=tp)


def print_results(title: str, rows: list[tuple[str, RunData]]) -> None:
    section(title)
    col_w = 18
    hdr = f"  {'':>12}" + "".join(f"  {label:>{col_w}}" for label, _ in rows)
    sep = f"  {'-' * 12}" + f"  {'-' * col_w}" * len(rows)

    subsection(f"Latency  (x{N_ITERS} sequential runs)")
    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = []
        for _, d in rows:
            v = statistics.mean(d.latency) if p is None else pct(d.latency, p)
            vals.append(fmt_us(v))
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    subsection(f"Memory  (x{N_MEM} runs)")
    print(f"  {'framework':<24}  {'peak':>12}  {'per run':>12}")
    print(f"  {'-' * 24}  {'-' * 12}  {'-' * 12}")
    for label, d in rows:
        print(f"  {label:<24}  {d.mem_peak / 1024:>10.1f} KB  {d.mem_per:>10.0f}  B")

    subsection(f"Burst  ({N_BURST} concurrent)")
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d.burst, p)) for _, d in rows]
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    subsection(f"Throughput  ({TP_OPS} loops)")
    print(hdr)
    print(sep)
    for i, conc in enumerate(CONC_LEVELS):
        vals = [f"{d.throughput[i]:>10.0f}/s" for _, d in rows]
        print(f"  {conc:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))


async def main() -> None:
    print()
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal delegation vs Google ADK transfer benchmark{RESET}")
    print(f"  {N_ITERS} iters | {N_BURST} burst | {N_MEM} mem | {TP_OPS} throughput ops")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")

    t_agent = make_timbal_delegation()
    msg = f"  Timbal: {'OK' if await run_timbal(t_agent) == 'worker-done' else 'FAIL'}"

    adk_bench = None
    if HAS_ADK:
        adk_bench = AdkTransferBench()
        await adk_bench.prepare(needed_runs() + 1)
        msg += f"  |  ADK: {'OK' if await adk_bench.run() == 'worker-done' else 'FAIL'}"
    else:
        msg += f"  |  {YELLOW}ADK skipped{RESET}"
    print(msg, flush=True)

    rows = [("Timbal delegation", await collect("Timbal delegation", lambda: run_timbal(t_agent)))]
    if adk_bench:
        rows.append(("ADK transfer", await collect("ADK transfer", adk_bench.run)))

    print_results("Delegation / transfer", rows)
    print()
    print(f"{DIM}{'-' * WIDTH}")
    print("  Google ADK has a true sub-agent transfer primitive via transfer_to_agent.")
    print("  Timbal column is the closest agent-composition equivalent: supervisor -> worker tool -> final.")
    print("  This is not a workflow/DAG benchmark; ADK does not expose a comparable workflow engine.")
    print(f"{'-' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
