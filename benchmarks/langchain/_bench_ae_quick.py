#!/usr/bin/env python3
"""
Quick AgentExecutor benchmark — Scenario 2 only (3-tool multi-step chain).
Runs in an isolated environment with langchain==0.3.x.

    uv run --isolated --with langchain==0.3.19 python benchmarks/langchain/_bench_ae_quick.py
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import statistics
import time
import tracemalloc
import warnings

logging.disable(logging.WARNING)
os.environ.setdefault("OPENAI_API_KEY", "fake")
warnings.filterwarnings("ignore")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from unittest.mock import MagicMock

try:
    from langchain_core.tracers.langchain import LangChainTracer
    def _make_tracer():
        t = LangChainTracer()
        t._persist_run_single = MagicMock(return_value=None)
        return t
    HAS_SMITH = True
except Exception:
    HAS_SMITH = False

RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[36m"; DIM = "\033[2m"

N_ITERS  = 50
N_WARMUP = 5
N_MEM    = 30
N_BURST  = 10
CONC     = [1, 10, 50]


# ── Fake LLM ──────────────────────────────────────────────────────────────────

class _FakeLLM(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kw): return self
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


RESPONSES = [
    AIMessage(content="", tool_calls=[{"name": "add",      "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"}]),
    AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"}]),
    AIMessage(content="", tool_calls=[{"name": "subtract", "args": {"a": 12, "b": 3}, "id": "c3", "type": "tool_call"}]),
    AIMessage(content="9"),
]


def _make_executor():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
    tools = [
        StructuredTool.from_function(add),
        StructuredTool.from_function(multiply),
        StructuredTool.from_function(subtract),
    ]
    llm = _FakeLLM(responses=list(RESPONSES))
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def fmt(us: float) -> str:
    return f"{us/1000:>8.2f} ms" if us >= 1000 else f"{us:>8.1f} µs"


async def main():
    print()
    print(f"{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  AgentExecutor — Scenario 2 (3-tool chain){RESET}")
    print(f"  langchain 0.3.x  |  {N_ITERS} iters  |  {N_BURST} burst")
    print(f"  LLM → add → LLM → multiply → LLM → subtract → LLM → answer")
    print(f"{BOLD}{'═'*64}{RESET}")

    ae = _make_executor()
    tracer = _make_tracer() if HAS_SMITH else None
    inp = {"input": "go"}

    async def run():      return await ae.ainvoke(inp)
    async def run_ls():   return await ae.ainvoke(inp, config={"callbacks": [tracer]})

    cols = ["AE (bare)"] + (["AE+Smith"] if HAS_SMITH else [])
    runners = [run] + ([run_ls] if HAS_SMITH else [])

    # ── Correctness ─────────────────────────────────────────────────────────
    result = await run()
    print(f"\n  Correctness: output={result['output']!r}  (expected '9')")

    # ── Latency ─────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Latency  (×{N_ITERS}){RESET}")
    hdr = f"  {'':>8}" + "".join(f"  {c:>12}" for c in cols)
    sep = f"  {'─'*8}" + f"  {'─'*12}" * len(cols)
    print(hdr); print(sep)

    all_s = []
    for fn in runners:
        for _ in range(N_WARMUP): await fn()
        gc.collect()
        s = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            await fn()
            s.append((time.perf_counter() - t0) * 1e6)
        all_s.append(s)

    for label, p in [("mean", None), ("p50", 50), ("p95", 95), ("p99", 99)]:
        fn2 = statistics.mean if p is None else lambda s: sorted(s)[min(int(len(s)*p/100), len(s)-1)]
        vals = [fn2(s) for s in all_s]
        print(f"  {label:>8}" + "".join(f"  {fmt(v):>12}" for v in vals))

    # ── Memory ──────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Memory  (×{N_MEM} runs){RESET}")
    print(f"  {'framework':<16}  {'peak':>10}  {'per run':>10}")
    print(f"  {'─'*16}  {'─'*10}  {'─'*10}")
    for name, fn in zip(cols, runners):
        for _ in range(N_WARMUP): await fn()
        gc.collect()
        tracemalloc.start()
        for _ in range(N_MEM): await fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {name:<16}  {peak/1024:>8.1f} KB  {peak/N_MEM:>8.0f}  B")

    # ── Burst ────────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Burst  ({N_BURST} concurrent){RESET}")
    print(hdr); print(sep)
    burst_all = []
    for fn in runners:
        await asyncio.gather(*[fn() for _ in range(3)])
        gc.collect()
        samples: list[float] = []
        async def _t(f=fn):
            t0 = time.perf_counter()
            await f()
            samples.append((time.perf_counter() - t0) * 1e6)
        await asyncio.gather(*[_t() for _ in range(N_BURST)])
        burst_all.append(sorted(samples))
    for label, p in [("p50", 50), ("p95", 95)]:
        vals = [sorted(s)[min(int(len(s)*p/100), len(s)-1)] for s in burst_all]
        print(f"  {label:>8}" + "".join(f"  {fmt(v):>12}" for v in vals))
    wall = " | ".join(f"{c}: {max(s)/1e3:.1f} ms" for c, s in zip(cols, burst_all))
    print(f"  {DIM}wall: {wall}{RESET}")

    # ── Throughput ────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Throughput  (100 ops){RESET}")
    print(hdr); print(sep)
    for conc in CONC:
        row = []
        for fn in runners:
            sem = asyncio.Semaphore(conc)
            async def _b(f=fn):
                async with sem: await f()
            gc.collect()
            t0 = time.perf_counter()
            await asyncio.gather(*[_b() for _ in range(100)])
            row.append(100 / (time.perf_counter() - t0))
        print(f"  {conc:>8}" + "".join(f"  {v:>10.0f}/s" for v in row))

    print()
    print(f"{DIM}  For reference — LangGraph create_react_agent Scenario 2 (from main benchmark):")
    print(f"  LG bare p50: ~5.5 ms  |  LG+Smith p50: ~10 ms  |  Timbal p50: ~382 µs{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
