# Timbal — Benchmarks

Framework overhead benchmarks. Each subdirectory targets a specific framework or
internal optimization, and contains scripts, raw results, and methodology notes.

## Philosophy

All benchmarks measure **pure framework overhead** — what the framework costs on top
of the actual work. Handlers are trivial (e.g. `return a + b`, `asyncio.sleep(0.001)`)
so results isolate the machinery: parameter resolution, context management, tracing,
event construction, serialization.

**Observability is always included on both sides.** A bare-framework comparison is
misleading — nobody ships without tracing in production. Competing frameworks are
benchmarked with their standard observability layer, with HTTP export mocked to isolate
the in-process cost. Timbal's tracing is built-in and always on.

## Setup

Timbal itself must be installed in the environment. From the repo root:

```bash
uv sync --dev
```

Each benchmark subdirectory also requires the framework it benchmarks. Install those
before running — see the subdirectory README for the exact command. For example:

```bash
# LangChain benchmarks
uv pip install langchain-core langsmith langgraph

# CrewAI benchmarks
uv pip install crewai agentops
```

No API keys required. All LLM calls are faked by inspecting message history.

## Running benchmarks

```bash
# Quick mode — fewer iterations, useful for sanity-checking
uv run python benchmarks/<dir>/bench_<name>.py --quick

# Full mode — more iterations, stable percentiles, results worth keeping
uv run python benchmarks/<dir>/bench_<name>.py
```

## What to look at

- **p50 latency** — steady-state single-call cost
- **burst p50/p95** — behavior under concurrent load
- **memory per run** — allocations per invocation (tracemalloc, not RSS)
- **throughput at c=10** — ops/s with a realistic concurrency level

p99 numbers tend to be noisy in short runs and are shown for completeness, not as
headline figures.

## Subdirectories

| Directory | What it compares |
|-----------|-----------------|
| `langchain/` | Timbal vs LangGraph + LangSmith — agents and DAG workflows |
| `crewai/` | Timbal vs CrewAI + AgentOps — multi-agent pipelines |

Raw benchmark outputs are stored in `<dir>/results/` after full-mode runs.
