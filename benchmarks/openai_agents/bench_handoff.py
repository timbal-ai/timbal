#!/usr/bin/env python3
"""
Timbal delegation vs OpenAI Agents SDK handoff benchmark.

This is not a DAG/workflow benchmark. OpenAI Agents SDK has handoffs, not a first-class
workflow engine. The OpenAI side uses a real handoff from a triage agent to a worker
agent. The Timbal side uses the closest equivalent composition: a supervisor agent calls
a worker agent as a tool, then returns the worker result.

Run:
    uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_handoff.py
    uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_handoff.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import json
import logging
import os
import statistics
import time
import tracemalloc
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
warnings.filterwarnings("ignore")

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


def _make_timbal_delegation() -> TimbalAgent:
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
                content=[
                    ToolUseContent(type="tool_use", id="c1", name="worker_tool", input={"input": "go"}),
                ],
            )
        return Message(
            role="assistant",
            content=[TextContent(type="text", text="worker-done")],
            stop_reason="end_turn",
        )

    return TimbalAgent(
        name="supervisor",
        model=TestModel(handler=supervisor_handler),
        tools=[worker_tool],
    )


HAS_OPENAI_AGENTS = False

try:
    from agents import Agent as OAIAgent  # noqa: E402
    from agents import RunConfig, Runner  # noqa: E402
    from agents.items import ModelResponse  # noqa: E402
    from agents.models.interface import Model as OAIModel  # noqa: E402
    from agents.tracing import set_trace_processors  # noqa: E402
    from agents.tracing.processor_interface import TracingProcessor  # noqa: E402
    from agents.usage import Usage  # noqa: E402
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText  # noqa: E402

    class _InMemoryTraceProcessor(TracingProcessor):
        def __init__(self) -> None:
            self.traces = []
            self.spans = []

        def on_trace_start(self, trace) -> None:
            self.traces.append(("start", trace))

        def on_trace_end(self, trace) -> None:
            self.traces.append(("end", trace))

        def on_span_start(self, span) -> None:
            self.spans.append(("start", span))

        def on_span_end(self, span) -> None:
            self.spans.append(("end", span))

        def shutdown(self) -> None:
            self.clear()

        def force_flush(self) -> None:
            return None

        def clear(self) -> None:
            self.traces.clear()
            self.spans.clear()

    _TRACE_PROCESSOR = _InMemoryTraceProcessor()
    set_trace_processors([_TRACE_PROCESSOR])

    def _clear_oai_traces() -> None:
        _TRACE_PROCESSOR.clear()

    def _text(text: str) -> ResponseOutputMessage:
        return ResponseOutputMessage(
            id=f"msg_{text}",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )

    class _FakeHandoffModel(OAIModel):
        def __init__(self, role: str) -> None:
            self.role = role

        async def get_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            *,
            previous_response_id,
            conversation_id,
            prompt,
        ) -> ModelResponse:
            _ = (
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            if self.role == "triage":
                output = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id="fc_handoff",
                        call_id="c1",
                        name=handoffs[0].tool_name,
                        arguments=json.dumps({}),
                        status="completed",
                    )
                ]
            else:
                output = [_text("worker-done")]
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    def _make_oai_handoff() -> OAIAgent:
        worker = OAIAgent(
            name="worker",
            handoff_description="Worker agent for delegated requests.",
            model=_FakeHandoffModel("worker"),
        )
        return OAIAgent(name="triage", model=_FakeHandoffModel("triage"), handoffs=[worker])

    HAS_OPENAI_AGENTS = True
except ImportError as e:
    print(f"OpenAI Agents SDK not found or incompatible: {e}")

    def _clear_oai_traces() -> None:
        return None


def _clear_all_traces() -> None:
    _clear_timbal_traces()
    _clear_oai_traces()


async def _latency_async(factory, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        await factory()
    _clear_all_traces()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)
    _clear_all_traces()
    return samples


async def _memory_async(factory, n: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        await factory()
    _clear_all_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await factory()
        _clear_all_traces()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, peak / n


async def _burst_async(factory, n: int) -> list[float]:
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_all_traces()
    gc.collect()
    samples: list[float] = []

    async def _timed():
        t0 = time.perf_counter()
        await factory()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    await asyncio.gather(*[_timed() for _ in range(n)])
    _clear_all_traces()
    return sorted(samples)


async def _throughput_async(factory, total: int, conc: int) -> float:
    sem = asyncio.Semaphore(conc)

    async def _bounded():
        async with sem:
            await factory()

    gc.collect()
    t0 = time.perf_counter()
    await asyncio.gather(*[_bounded() for _ in range(total)])
    elapsed = time.perf_counter() - t0
    _clear_all_traces()
    return total / elapsed


@dataclasses.dataclass
class RunData:
    latency: list[float]
    mem_peak: float
    mem_per: float
    burst: list[float]
    throughput: list[float]


async def _collect(label: str, factory) -> RunData:
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


def _print_results(rows: list[tuple[str, RunData]]) -> None:
    section("Delegation / handoff")
    col_w = 16
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
    print(f"{BOLD}  Timbal delegation vs OpenAI Agents SDK handoff benchmark{RESET}")
    print(f"  {N_ITERS} iters | {N_BURST} burst | {N_MEM} mem | {TP_OPS} throughput ops")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")

    t_agent = _make_timbal_delegation()
    t_res = await t_agent(prompt="go").collect()
    msg = f"  Timbal: {'OK' if t_res.output.collect_text() == 'worker-done' else 'FAIL'}"
    if HAS_OPENAI_AGENTS:
        oai_res = await Runner.run(_make_oai_handoff(), "go", run_config=RunConfig(tracing_disabled=True))
        msg += f"  |  OAI: {'OK' if str(oai_res.final_output) == 'worker-done' else 'FAIL'}"
    print(msg, flush=True)

    rows = [("Timbal delegation", await _collect("Timbal delegation", lambda: t_agent(prompt="go").collect()))]
    if HAS_OPENAI_AGENTS:
        oai_agent = _make_oai_handoff()
        rows.append(
            (
                "OAI handoff",
                await _collect(
                    "OAI handoff",
                    lambda: Runner.run(oai_agent, "go", run_config=RunConfig(tracing_disabled=False), max_turns=5),
                ),
            )
        )

    _print_results(rows)
    print()
    print(f"{DIM}{'-' * WIDTH}")
    print("  OpenAI Agents SDK has a true handoff primitive.")
    print("  Timbal column is the closest agent-composition equivalent: supervisor -> worker tool -> final.")
    print(f"{'-' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
