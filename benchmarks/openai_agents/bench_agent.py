#!/usr/bin/env python3
"""
Timbal vs OpenAI Agents SDK - full agent loop benchmark.

Three scenarios, each running the identical pipeline on both frameworks:
  1. Single tool call:   prompt -> LLM -> tool(add) -> LLM -> answer
  2. Multi-step:         prompt -> LLM -> add -> LLM -> mul -> LLM -> sub -> LLM -> answer
  3. Parallel tools:     prompt -> LLM -> [add, mul, neg] concurrent -> LLM -> answer

Timbal uses TestModel for fully offline deterministic LLM simulation.
OpenAI Agents uses a custom Model implementation, also fully offline.

Columns:
  Timbal          - built-in InMemory tracing (always on)
  OAI (bare)      - OpenAI Agents with tracing_disabled=True
  OAI + tracing   - OpenAI Agents with a local in-memory tracing processor, no export

Run:
    uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent.py
    uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent.py --quick
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
from typing import Any

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
                    content=[
                        ToolUseContent(type="tool_use", id="c2", name="multiply", input={"a": 3, "b": 4})
                    ],
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


HAS_OPENAI_AGENTS = False

try:
    from agents import Agent as OAIAgent  # noqa: E402
    from agents import RunConfig, Runner, function_tool  # noqa: E402
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

    def _count_oai_tool_outputs(input_items: str | list[dict[str, Any]]) -> int:
        if not isinstance(input_items, list):
            return 0
        return sum(1 for item in input_items if item.get("type") == "function_call_output")

    def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> ResponseFunctionToolCall:
        return ResponseFunctionToolCall(
            type="function_call",
            id=f"fc_{call_id}",
            call_id=call_id,
            name=name,
            arguments=json.dumps(args),
            status="completed",
        )

    def _text(text: str) -> ResponseOutputMessage:
        return ResponseOutputMessage(
            id=f"msg_{text}",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )

    class _FakeOAIModel(OAIModel):
        def __init__(self, scenario: int) -> None:
            self.scenario = scenario

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
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            step = _count_oai_tool_outputs(input)
            if self.scenario == 1:
                output = [_tool_call("c1", "add", {"a": 1, "b": 2})] if step == 0 else [_text("3")]
            elif self.scenario == 2:
                if step == 0:
                    output = [_tool_call("c1", "add", {"a": 1, "b": 2})]
                elif step == 1:
                    output = [_tool_call("c2", "multiply", {"a": 3, "b": 4})]
                elif step == 2:
                    output = [_tool_call("c3", "subtract", {"a": 12, "b": 3})]
                else:
                    output = [_text("9")]
            else:
                output = (
                    [
                        _tool_call("c1", "add", {"a": 1, "b": 2}),
                        _tool_call("c2", "multiply", {"a": 3, "b": 4}),
                        _tool_call("c3", "negate", {"x": 5}),
                    ]
                    if step == 0
                    else [_text("done")]
                )
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    @function_tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @function_tool
    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    @function_tool
    async def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    @function_tool
    async def negate(x: int) -> int:
        """Negate a number."""
        return -x

    def _make_oai_agent(scenario: int) -> OAIAgent:
        tools = {1: [add], 2: [add, multiply, subtract], 3: [add, multiply, negate]}[scenario]
        return OAIAgent(name="bench_agent", model=_FakeOAIModel(scenario), tools=tools)

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


async def _burst_memory_async(factory, n: int) -> tuple[float, float]:
    await asyncio.gather(*[factory() for _ in range(min(5, n))])
    _clear_all_traces()
    gc.collect()
    tracemalloc.start()
    await asyncio.gather(*[factory() for _ in range(n)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _clear_all_traces()
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
    _clear_all_traces()
    return total / elapsed


@dataclasses.dataclass
class RunData:
    latency: list[float]
    mem_peak: float
    mem_per: float
    burst: list[float]
    burst_mem_peak: float
    burst_mem_per: float
    throughput: list[float]


async def _collect_timbal(scenario: int) -> RunData:
    agent = _make_timbal_agent(scenario)
    factory = lambda: agent(prompt="go").collect()
    return await _collect("Timbal", scenario, factory)


async def _collect_oai(scenario: int, label: str, tracing_disabled: bool) -> RunData:
    agent = _make_oai_agent(scenario)
    run_config = RunConfig(tracing_disabled=tracing_disabled)
    factory = lambda: Runner.run(agent, "go", run_config=run_config)
    return await _collect(label, scenario, factory)


async def _collect(label: str, scenario: int, factory) -> RunData:
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

    return RunData(
        latency=lat,
        mem_peak=mem_peak,
        mem_per=mem_per,
        burst=burst,
        burst_mem_peak=burst_mem_peak,
        burst_mem_per=burst_mem_per,
        throughput=tp,
    )


SCENARIO_NAMES = {
    1: "Single tool  (LLM -> add -> LLM -> answer)",
    2: "Multi-step  (LLM -> add -> LLM -> mul -> LLM -> sub -> LLM -> answer)",
    3: "Parallel tools  (LLM -> [add, mul, neg] concurrent -> LLM -> answer)",
}


def _print_scenario(scenario: int, t: RunData, oai_bare: RunData | None, oai_trace: RunData | None) -> None:
    section(f"Scenario {scenario}: {SCENARIO_NAMES[scenario]}")

    cols: list[tuple[str, RunData]] = [("Timbal", t)]
    if oai_bare:
        cols.append(("OAI (bare)", oai_bare))
    if oai_trace:
        cols.append(("OAI + tracing", oai_trace))

    col_w = 16
    hdr = f"  {'':>12}" + "".join(f"  {c:>{col_w}}" for c, _ in cols)
    sep = f"  {'-' * 12}" + f"  {'-' * col_w}" * len(cols)

    subsection(f"Latency  (x{N_ITERS} sequential runs)")
    print(hdr)
    print(sep)
    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        vals = []
        for _, d in cols:
            v = statistics.mean(d.latency) if p is None else pct(d.latency, p)
            vals.append(fmt_us(v))
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))

    subsection(f"Memory  (x{N_MEM} runs)")
    fw = 24
    print(f"  {'framework':<{fw}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'-' * fw}  {'-' * 12}  {'-' * 12}")
    for name, d in cols:
        print(f"  {name:<{fw}}  {d.mem_peak / 1024:>10.1f} KB  {d.mem_per:>10.0f}  B")

    n_burst = N_BURST[scenario]
    subsection(f"Burst  ({n_burst} concurrent)")
    print(hdr)
    print(sep)
    for label, p in [("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99), ("max", 100)]:
        vals = [fmt_us(pct(d.burst, p)) for _, d in cols]
        print(f"  {label:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))
    wall = " | ".join(f"{name}: {max(d.burst) / 1e3:.1f} ms" for name, d in cols)
    print(f"\n  {DIM}wall: {wall}{RESET}")

    subsection(f"Burst memory  ({n_burst} concurrent)")
    print(f"  {'framework':<{fw}}  {'peak':>12}  {'per run':>12}")
    print(f"  {'-' * fw}  {'-' * 12}  {'-' * 12}")
    for name, d in cols:
        print(f"  {name:<{fw}}  {d.burst_mem_peak / 1024:>10.1f} KB  {d.burst_mem_per:>10.0f}  B")

    subsection(f"Throughput  ({TP_OPS} loops)")
    print(hdr)
    print(sep)
    for i, conc in enumerate(CONC_LEVELS):
        vals = [f"{d.throughput[i]:>10.0f}/s" for _, d in cols]
        print(f"  {conc:>12}" + "".join(f"  {v:>{col_w}}" for v in vals))


async def main() -> None:
    print()
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal vs OpenAI Agents SDK - agent loop benchmark{RESET}")
    print(f"  {N_ITERS} iters | burst {N_BURST} | {N_MEM} mem | {TP_OPS} throughput ops")
    print("  Timbal: TestModel offline | OpenAI Agents: custom Model offline")
    if HAS_OPENAI_AGENTS:
        print("  OpenAI Agents SDK found - will measure tracing disabled + local tracing processor")
    else:
        print(f"  {YELLOW}OpenAI Agents SDK unavailable. Run with: uv run --with openai-agents --with 'openai>=2.26,<3'{RESET}")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")

    print(f"\n  {DIM}Spot-checking correctness (Scenario 1)...{RESET}", flush=True)
    t_res = await _make_timbal_agent(1)(prompt="go").collect()
    t_ok = t_res.status.code == "success"
    msg = f"  Timbal: {'OK' if t_ok else 'FAIL'}"
    if HAS_OPENAI_AGENTS:
        oai_res = await Runner.run(_make_oai_agent(1), "go", run_config=RunConfig(tracing_disabled=True))
        oai_ok = str(oai_res.final_output).strip() == "3"
        msg += f"  |  OAI: {'OK' if oai_ok else f'FAIL (got {oai_res.final_output!r})'}"
    print(msg, flush=True)

    t_data: dict[int, RunData] = {}
    oai_bare_data: dict[int, RunData] = {}
    oai_trace_data: dict[int, RunData] = {}

    for scenario in [1, 2, 3]:
        print(f"\n  {DIM}[Scenario {scenario}]{RESET}", flush=True)
        t_data[scenario] = await _collect_timbal(scenario)
        if HAS_OPENAI_AGENTS:
            oai_bare_data[scenario] = await _collect_oai(scenario, "OAI (bare)", tracing_disabled=True)
            oai_trace_data[scenario] = await _collect_oai(scenario, "OAI + tracing", tracing_disabled=False)

    for scenario in [1, 2, 3]:
        _print_scenario(
            scenario,
            t=t_data[scenario],
            oai_bare=oai_bare_data.get(scenario),
            oai_trace=oai_trace_data.get(scenario),
        )

    print()
    print(f"{DIM}{'-' * WIDTH}")
    print("  Timbal: TestModel, built-in InMemory tracing always on.")
    print("  OAI (bare): custom offline Model with RunConfig(tracing_disabled=True).")
    print("  OAI + tracing: same custom Model with local in-memory tracing processor, no export.")
    print("  Dependency note: OpenAI Agents SDK 0.14.6 requires openai>=2.26; use uv --with flags above.")
    print(f"{'-' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
