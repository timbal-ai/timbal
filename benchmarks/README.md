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

## Codebase size

The core framework — `core`, `state`, `types`, `collectors`, and supporting modules — is **~10k lines** of Python (comments and blank lines excluded). That includes the agent engine, workflow DAG, tracing providers, event system, and all type definitions. It does not include code generation (~4k) or the evals harness (~3k), which are optional add-ons. The built-in tool library is excluded from all counts — tools are almost always a separate package for other frameworks, so including them would skew comparisons.

Total framework (excluding tools): **~17k lines**. For context, we'll be comparing this against LangChain/LangGraph, CrewAI, Agno, and PydanticAI. Spoiler: they're substantially larger for equivalent functionality.

### Test coverage

| Module       | Source | Tests  | Ratio  |
|--------------|-------:|-------:|-------:|
| core         |  ~3.6k | ~14.5k |  4.0×  |
| codegen      |  ~3.7k |  ~7.2k |  1.9×  |
| state        |  ~1.7k |  ~3.3k |  1.9×  |
| collectors   |  ~1.2k |  ~1.8k |  1.5×  |
| types        |  ~1.7k |  ~1.0k |  0.6×  |
| server       |  ~0.3k |  ~1.2k |  4.0×  |
| platform     |  ~0.4k |  ~0.7k |  1.8×  |
| utils        |  ~0.5k |  ~0.4k |  0.8×  |

Total: **~17k lines of source**, **~30k lines of tests**.

`core` and `server` are the most thoroughly tested (4× test-to-source). `types` is relatively under-tested because the types are simple data containers exercised indirectly through core tests.

## Subdirectories

| Directory | What it compares | Status |
|-----------|-----------------|--------|
| `langchain/` | Timbal vs LangGraph + LangSmith — agents and DAG workflows | stable |
| `crewai/` | Timbal vs CrewAI + AgentOps — multi-agent pipelines | stable |
| `agno/` | Timbal vs Agno — agents and multi-agent pipelines | WIP |
| `pydantic/` | Timbal vs PydanticAI + Logfire — agents | WIP |

**Coming next:** OpenAI Agents SDK and Google Agent Development Kit (ADK).

Have a framework you'd like to see benchmarked? Open an issue or send a PR — we're happy to add more.

Raw benchmark outputs are stored in `<dir>/results/` after full-mode runs.
