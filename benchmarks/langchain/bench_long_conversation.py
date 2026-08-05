#!/usr/bin/env python3
"""
Timbal vs LangGraph — long conversations / long messages benchmark.

Measures how framework overhead scales with conversation length and message
size. LLMs are faked on both sides (immediate text answer, no tools), so the
numbers isolate message validation, history processing, memory persistence,
and tracing — the per-turn cost a long-running chat assistant pays on every
request.

Scenarios:
  1. Long history, single turn (stateless): an N-message conversation is
     passed into one agent turn. N scales.
  2. Long messages, single turn: 20-message history, message size scales.
  3. Multi-turn session (stateful): T turns on one conversation. Timbal
     persists memory via parent_id chaining (in-memory tracing provider);
     LangGraph uses its MemorySaver checkpointer. Both rebuild/extend the
     conversation each turn, so per-turn latency grows with turn index.

Timbal runs with built-in tracing on (always-on by design). LangGraph runs
bare, per request — this is Timbal-with-observability vs LangGraph-without.

Run:
    uv run python benchmarks/langchain/bench_long_conversation.py
    uv run python benchmarks/langchain/bench_long_conversation.py --quick
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
_args, _ = parser.parse_known_args()

N_ITERS = 10 if _args.quick else 50
N_WARMUP = 3 if _args.quick else 8
N_MEM = 10 if _args.quick else 50
HISTORY_SIZES = [10, 50] if _args.quick else [10, 50, 200]
MESSAGE_SIZES = [200, 5_000] if _args.quick else [200, 5_000, 20_000]
SESSION_TURNS = 8 if _args.quick else 20
N_SESSIONS = 5 if _args.quick else 15
WIDTH = 76

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
# TIMBAL
# ═══════════════════════════════════════════════════════════════════════════════

from timbal import Agent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402
from timbal.types.message import Message  # noqa: E402


def _clear_traces():
    InMemoryTracingProvider._storage.clear()


def _make_timbal_agent() -> Agent:
    """Fake LLM: immediately answers with a short text (no tools)."""
    return Agent(
        name="chat_agent",
        model=TestModel(handler=lambda messages: f"ok ({len(messages)} msgs seen)"),
        tools=[],
    )


def _timbal_history(n_messages: int, text: str) -> list[Message]:
    """Alternating user/assistant history ending in a user message."""
    history = []
    for i in range(n_messages - 1):
        role = "user" if i % 2 == 0 else "assistant"
        history.append(Message.validate({"role": role, "content": f"[{i}] {text}"}))
    history.append(Message.validate({"role": "user", "content": f"[final] {text}"}))
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH
# ═══════════════════════════════════════════════════════════════════════════════

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402


class _FakeLLM(FakeMessagesListChatModel):
    """Stateless fake LLM: always answers with a short text (no tool calls)."""

    def bind_tools(self, tools, **kw):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"ok ({len(messages)} msgs seen)"))]
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"ok ({len(messages)} msgs seen)"))]
        )


def _make_lc_graph(checkpointer=None):
    llm = _FakeLLM(responses=[AIMessage(content="unused")])
    return create_react_agent(llm, tools=[], checkpointer=checkpointer)


def _lc_history(n_messages: int, text: str) -> list:
    history = []
    for i in range(n_messages - 1):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        history.append(cls(content=f"[{i}] {text}"))
    history.append(HumanMessage(content=f"[final] {text}"))
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════════


async def _latency(run_fn, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        await run_fn()
    _clear_traces()
    gc.collect()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await run_fn()
        samples.append((time.perf_counter() - t0) * 1e6)
        # Clear per iteration (outside the timed window) so accumulated
        # in-memory traces don't skew later samples — mirrors _memory.
        _clear_traces()
    return samples


async def _memory(run_fn, n: int, warmup: int) -> float:
    for _ in range(warmup):
        await run_fn()
    _clear_traces()
    gc.collect()
    tracemalloc.start()
    for _ in range(n):
        await run_fn()
        _clear_traces()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / n


def _print_scaling_table(row_label: str, rows: list, t_p50s: list, lg_p50s: list) -> None:
    print(f"  {row_label:>12}  {'Timbal p50':>12}  {'LG p50':>12}  {'ratio':>8}")
    print(f"  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 8}")
    for row, t, lg in zip(rows, t_p50s, lg_p50s):
        print(f"  {row:>12}  {fmt_us(t):>12}  {fmt_us(lg):>12}  {lg / t:>7.2f}x")


def _print_slope(unit: str, rows: list, t_p50s: list, lg_p50s: list, scale: float = 1.0) -> None:
    dn = (rows[-1] - rows[0]) * scale
    t_slope = (t_p50s[-1] - t_p50s[0]) / dn
    lg_slope = (lg_p50s[-1] - lg_p50s[0]) / dn
    subsection(f"Marginal cost per {unit}  (slope {rows[0]} → {rows[-1]})")
    print(f"  Timbal          +{t_slope:.2f} µs / {unit}")
    print(f"  LG (bare)       +{lg_slope:.2f} µs / {unit}")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1 — Long history, single turn (stateless)
# ═══════════════════════════════════════════════════════════════════════════════


async def bench_long_history() -> None:
    section(f"Scenario 1: Long history, single turn  (N messages in → 1 LLM turn, ×{N_ITERS})")
    print(f"  {DIM}History injected per call (stateless). ~60-char messages.{RESET}")

    t_agent = _make_timbal_agent()
    lg_graph = _make_lc_graph()
    text = "x" * 40

    t_p50s, lg_p50s = [], []
    for n in HISTORY_SIZES:
        t_history = _timbal_history(n, text)
        lg_history = _lc_history(n, text)

        async def t_run(h=t_history):
            await t_agent(messages=h).collect()

        async def lg_run(h=lg_history):
            await lg_graph.ainvoke({"messages": h})

        t_p50s.append(pct(await _latency(t_run, N_ITERS, N_WARMUP), 50))
        lg_p50s.append(pct(await _latency(lg_run, N_ITERS, N_WARMUP), 50))

    subsection("Latency p50 by history length")
    _print_scaling_table("N messages", HISTORY_SIZES, t_p50s, lg_p50s)
    _print_slope("message", HISTORY_SIZES, t_p50s, lg_p50s)

    # Memory at largest N
    n = HISTORY_SIZES[-1]
    t_history = _timbal_history(n, text)
    lg_history = _lc_history(n, text)

    async def t_run_mem():
        await t_agent(messages=t_history).collect()

    async def lg_run_mem():
        await lg_graph.ainvoke({"messages": lg_history})

    t_mem = await _memory(t_run_mem, N_MEM, N_WARMUP)
    lg_mem = await _memory(lg_run_mem, N_MEM, N_WARMUP)
    subsection(f"Memory per run  (N={n}, ×{N_MEM} runs)")
    print(f"  Timbal          {t_mem / 1024:>8.1f} KB")
    print(f"  LG (bare)       {lg_mem / 1024:>8.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2 — Long messages, single turn
# ═══════════════════════════════════════════════════════════════════════════════


async def bench_long_messages() -> None:
    n = 20
    section(f"Scenario 2: Long messages, single turn  ({n}-message history, size scales, ×{N_ITERS})")

    t_agent = _make_timbal_agent()
    lg_graph = _make_lc_graph()

    t_p50s, lg_p50s = [], []
    for size in MESSAGE_SIZES:
        text = "lorem ipsum " * max(1, size // 12)
        text = text[:size]
        t_history = _timbal_history(n, text)
        lg_history = _lc_history(n, text)

        async def t_run(h=t_history):
            await t_agent(messages=h).collect()

        async def lg_run(h=lg_history):
            await lg_graph.ainvoke({"messages": h})

        t_p50s.append(pct(await _latency(t_run, N_ITERS, N_WARMUP), 50))
        lg_p50s.append(pct(await _latency(lg_run, N_ITERS, N_WARMUP), 50))

    subsection("Latency p50 by message size")
    _print_scaling_table("msg bytes", MESSAGE_SIZES, t_p50s, lg_p50s)
    _print_slope("KB of message", MESSAGE_SIZES, t_p50s, lg_p50s, scale=1 / 1000)

    # Memory at largest size
    size = MESSAGE_SIZES[-1]
    text = ("lorem ipsum " * max(1, size // 12))[:size]
    t_history = _timbal_history(n, text)
    lg_history = _lc_history(n, text)

    async def t_run_mem():
        await t_agent(messages=t_history).collect()

    async def lg_run_mem():
        await lg_graph.ainvoke({"messages": lg_history})

    t_mem = await _memory(t_run_mem, N_MEM, N_WARMUP)
    lg_mem = await _memory(lg_run_mem, N_MEM, N_WARMUP)
    subsection(f"Memory per run  (msg={size} B, ×{N_MEM} runs)")
    print(f"  Timbal          {t_mem / 1024:>8.1f} KB")
    print(f"  LG (bare)       {lg_mem / 1024:>8.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Multi-turn session (stateful memory)
# ═══════════════════════════════════════════════════════════════════════════════


async def _timbal_session(agent: Agent, turns: int, turn_samples: dict[int, list[float]] | None = None) -> str:
    """Run one Timbal conversation: memory persists via parent_id chaining."""
    parent_id = None
    last_text = ""
    for t in range(1, turns + 1):
        t0 = time.perf_counter()
        kwargs = {"prompt": f"turn {t}"}
        if parent_id is not None:
            kwargs["parent_id"] = parent_id
        out = await agent(**kwargs).collect()
        elapsed = (time.perf_counter() - t0) * 1e6
        if turn_samples is not None and t in turn_samples:
            turn_samples[t].append(elapsed)
        parent_id = out.run_id
        last_text = out.output.collect_text()
    return last_text


async def _lc_session(graph, thread_id: str, turns: int, turn_samples: dict[int, list[float]] | None = None) -> str:
    """Run one LangGraph conversation: memory persists via MemorySaver checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    last_text = ""
    for t in range(1, turns + 1):
        t0 = time.perf_counter()
        result = await graph.ainvoke({"messages": [HumanMessage(content=f"turn {t}")]}, config)
        elapsed = (time.perf_counter() - t0) * 1e6
        if turn_samples is not None and t in turn_samples:
            turn_samples[t].append(elapsed)
        last_text = result["messages"][-1].content
    return last_text


async def bench_session() -> None:
    section(f"Scenario 3: Multi-turn session  ({SESSION_TURNS} turns, ×{N_SESSIONS} sessions)")
    print(f"  {DIM}Stateful memory: Timbal parent_id chaining (in-memory tracing provider){RESET}")
    print(f"  {DIM}vs LangGraph MemorySaver checkpointer. History grows every turn.{RESET}")

    t_agent = _make_timbal_agent()
    checkpoints = sorted({1, max(2, SESSION_TURNS // 4), SESSION_TURNS // 2, SESSION_TURNS})

    # Warmup one session each
    await _timbal_session(t_agent, 3)
    _clear_traces()
    warm_graph = _make_lc_graph(checkpointer=MemorySaver())
    await _lc_session(warm_graph, "warmup", 3)
    gc.collect()

    # Timed sessions
    t_turns: dict[int, list[float]] = {c: [] for c in checkpoints}
    t_walls: list[float] = []
    for _ in range(N_SESSIONS):
        t0 = time.perf_counter()
        await _timbal_session(t_agent, SESSION_TURNS, t_turns)
        t_walls.append((time.perf_counter() - t0) * 1e3)
        _clear_traces()

    lg_turns: dict[int, list[float]] = {c: [] for c in checkpoints}
    lg_walls: list[float] = []
    for s in range(N_SESSIONS):
        graph = _make_lc_graph(checkpointer=MemorySaver())  # fresh saver per session (≙ cleared traces)
        t0 = time.perf_counter()
        await _lc_session(graph, f"s{s}", SESSION_TURNS, lg_turns)
        lg_walls.append((time.perf_counter() - t0) * 1e3)

    subsection("Per-turn latency p50 (history grows with turn index)")
    print(f"  {'turn':>12}  {'Timbal':>12}  {'LG (bare)':>12}  {'ratio':>8}")
    print(f"  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 8}")
    for c in checkpoints:
        t = pct(t_turns[c], 50)
        lg = pct(lg_turns[c], 50)
        print(f"  {c:>12}  {fmt_us(t):>12}  {fmt_us(lg):>12}  {lg / t:>7.2f}x")

    subsection(f"Full-session wall time  ({SESSION_TURNS} turns)")
    print(f"  Timbal          {statistics.median(t_walls):>8.1f} ms")
    print(f"  LG (bare)       {statistics.median(lg_walls):>8.1f} ms")

    # Memory per session
    gc.collect()
    tracemalloc.start()
    for _ in range(3):
        await _timbal_session(t_agent, SESSION_TURNS)
        _clear_traces()
    _, t_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    tracemalloc.start()
    for s in range(3):
        graph = _make_lc_graph(checkpointer=MemorySaver())
        await _lc_session(graph, f"m{s}", SESSION_TURNS)
    _, lg_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    subsection(f"Memory per session  ({SESSION_TURNS} turns, peak / 3 sessions)")
    print(f"  Timbal          {t_peak / 3 / 1024:>8.1f} KB")
    print(f"  LG (bare)       {lg_peak / 3 / 1024:>8.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def _verify() -> None:
    print(f"\n  {DIM}Verifying correctness...{RESET}")

    # Stateless history: both fake LLMs must see the full injected history.
    n = 10
    t_agent = _make_timbal_agent()
    out = await (t_agent(messages=_timbal_history(n, "hello"))).collect()
    t_text = out.output.collect_text()
    _clear_traces()

    lg_graph = _make_lc_graph()
    lg_out = await lg_graph.ainvoke({"messages": _lc_history(n, "hello")})
    lg_text = lg_out["messages"][-1].content
    print(f"  stateless n={n}: Timbal='{t_text}'  LangGraph='{lg_text}'")
    assert "10 msgs seen" in t_text, t_text
    assert "10 msgs seen" in lg_text, lg_text

    # Stateful session: by turn 3 both must have accumulated 5 messages
    # (u1, a1, u2, a2, u3) when the fake LLM is called.
    t_text = await _timbal_session(_make_timbal_agent(), 3)
    _clear_traces()
    lg_text = await _lc_session(_make_lc_graph(checkpointer=MemorySaver()), "verify", 3)
    print(f"  session turn 3: Timbal='{t_text}'  LangGraph='{lg_text}'")
    assert "5 msgs seen" in t_text, t_text
    assert "5 msgs seen" in lg_text, lg_text
    print(f"  {DIM}ok — both frameworks accumulate identical histories{RESET}")


async def main() -> None:
    print()
    print(f"{BOLD}{'═' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal vs LangGraph — long conversations / long messages{RESET}")
    print(f"  {N_ITERS} iters · histories {HISTORY_SIZES} · msg sizes {MESSAGE_SIZES}")
    print(f"  sessions: {N_SESSIONS} × {SESSION_TURNS} turns")
    print(f"  {DIM}LLMs faked (instant answer, no tools) — pure framework overhead.{RESET}")
    print(f"  {DIM}Timbal: built-in tracing on. LangGraph: bare (no LangSmith).{RESET}")
    print(f"{BOLD}{'═' * WIDTH}{RESET}")

    await _verify()
    await bench_long_history()
    await bench_long_messages()
    await bench_session()

    print()
    print(f"{DIM}{'─' * WIDTH}")
    print(f"  Agent/graph creation excluded (except fresh per-session checkpointers,")
    print(f"  mirrored by Timbal's cleared traces). Histories prebuilt outside timing.")
    print(f"{'─' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
