# Timbal vs LangGraph — Benchmarks

> **PR #222 rerun (2026-10-24):** see [the comparison report](results/pr-222/README.md).
> Earlier raw results are historical and should not be used as current performance claims.
> The rerun fixes ambient run-context reuse (which could skip tool calls) and uses updated
> LangGraph dependencies. Python LangSmith results now disable both create/update uploads
> in a subclass so cloned tracers remain offline; they measure callback overhead only.
> Peak-memory divided by iteration count is not a per-request allocation measurement.


Pure framework overhead benchmarks. No LLM API calls — all handlers are trivial
synthetic functions. Workloads are deterministic; timings vary with machine load and runtime versions.

**Rerun environment:** Apple M1 Pro, CPython 3.11.11, asyncio event loop.
**Comparison outputs use full-mode runs** (200 iters on DAG/parallel, 100 iters on agents).
Raw outputs are stored in `results/`.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |
| `bench_workflow.py` | Small DAG topologies: sequential, fan-out/in, diamond |
| `bench_parallel.py` | Wide parallel fan-out: root → [N branches] → sink |
| `bench_double_fanout.py` | Double fan-out: root → [N×p1] → aggregator → [N×p2] → sink |
| `bench_long_conversation.py` | Large histories, long messages and chained conversation turns |
| `bench_revision.py` | Repeated JSON latency samples for base/PR and both LangGraph agent APIs |

## Setup

From the repo root, install Timbal and the benchmarked frameworks:

```bash
uv sync --dev
uv pip install -r benchmarks/langchain/requirements.txt
```

No API keys required — all LLM calls are faked.

## How to run

```bash
# Quick mode (~2–5 min each, fewer iterations)
uv run --no-sync python benchmarks/langchain/bench_agent.py --quick
uv run --no-sync python benchmarks/langchain/bench_workflow.py --quick
uv run --no-sync python benchmarks/langchain/bench_parallel.py --quick
uv run --no-sync python benchmarks/langchain/bench_double_fanout.py --quick

# Full mode (redirect stdout to save results)
uv run --no-sync python benchmarks/langchain/bench_agent.py
uv run --no-sync python benchmarks/langchain/bench_agent.py --agent-api current
uv run --no-sync python benchmarks/langchain/bench_workflow.py
uv run --no-sync python benchmarks/langchain/bench_parallel.py
uv run --no-sync python benchmarks/langchain/bench_double_fanout.py
```

---

## Observability configuration

Timbal uses its in-memory tracing provider. The comparison reports both bare
LangGraph and LangGraph with `LangChainTracer` callbacks. The offline tracer
subclass disables `_persist_run_single` and `_update_run_single`, including on
clones created by LangChain. This retains local callback/RunTree bookkeeping,
but excludes upload serialization, batching, network delivery and durable storage.
These are different observability configurations, not equivalent production
persistence guarantees. Compare both LangGraph columns rather than treating
only one as universally fair.

The older raw results used an instance-level mock and are retained only as
historical records. See the PR #222 rerun for corrected measurements.

---

## Results

The [2026-10-24 comparison report](results/pr-222/README.md) compares PR #222
against its merge base and updated LangGraph, with raw outputs and repeat samples.
The older `results/bench_*.txt` files are historical; their context isolation and
tracer mocking differ, so they must not be mixed with the corrected measurements.

The default agent benchmark retains `langgraph.prebuilt.create_react_agent` for
continuity. Pass `--agent-api current` to use `langchain.agents.create_agent`,
[the replacement recommended by LangChain](https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent).

The workloads measure orchestration overhead with mocked models/tools. They do
not predict end-to-end application speedups, and do not compare durability,
distributed execution or production tracing backends. The fake model adapters
are framework-specific. Both graph compilation and agent construction are excluded.
