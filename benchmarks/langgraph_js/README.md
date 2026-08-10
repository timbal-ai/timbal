# Runtime Calibration - LangGraph Python vs LangGraph.js

This directory answers one question the cross-language benchmarks
(`benchmarks/mastra/`) cannot answer on their own: **how much of a
Python-vs-TypeScript performance gap is the language runtime, and how much is
the framework?**

LangGraph ships the same architecture in both languages — a prebuilt react
agent over a Pregel graph (`create_react_agent` / `createReactAgent`,
`StateGraph` in both). Running identical scenarios on both implementations
isolates the CPython-vs-V8 factor. The **LG.py ÷ LG.js ratio is the
calibration number**: apply it mentally when reading any Timbal (Python) vs
TypeScript-framework table in this repo.

Timbal appears in every table for reference, with its built-in tracing on as
always. The calibration ratio itself is computed bare-vs-bare (no
observability on either LangGraph side).

**Environment:** Apple Silicon M-series, CPython 3.11, Node 22,
`langgraph` (Python) latest, `@langchain/langgraph` 1.4.9. No API keys — all
LLMs faked via message-history inspection on all three sides, same procedure
per side (see `benchmarks/mastra/README.md` for the cross-runtime methodology;
this directory follows it exactly).

---

## Files

| File | What it does |
|------|--------------|
| `bench_agent.py` | Orchestrator: Timbal + LangGraph Python in-process, spawns `bench_agent.mjs`, prints combined tables + calibration ratios |
| `bench_agent.mjs` | LangGraph.js react-agent loop, JSON on stdout |
| `bench_workflow.py` | Orchestrator for StateGraph DAGs |
| `bench_workflow.mjs` | LangGraph.js StateGraph DAGs, JSON on stdout |
| `bench_lib.mjs` | Shared Node harness (copied from `benchmarks/mastra`) |

## Setup & running

```bash
cd benchmarks/langgraph_js && npm install && cd ../..

# langgraph (Python) is not in the repo lock — use an ephemeral overlay:
uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_agent.py --quick
uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_agent.py
uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_workflow.py --quick
uv run --no-sync --with langgraph --with langchain-core python benchmarks/langgraph_js/bench_workflow.py
```

Flags: `--timbal-only`, `--lgjs-json <file>` (replay a saved Node-side result).

---

## Results

Full-mode run, 2026-08-10. Raw outputs in `results/`.

### The calibration numbers

| Workload | LG.py ÷ LG.js (latency p50) | Interpretation |
|----------|:---------------------------:|----------------|
| Agent loop, single tool | 2.06x | V8 meaningfully faster |
| Agent loop, 3-step chain | 3.41x | V8 meaningfully faster |
| Agent loop, parallel tools | 3.25x | V8 meaningfully faster |
| DAG sequential | 1.01x | runtime is a wash |
| DAG fan-out/in | 1.06x | runtime is a wash |
| DAG diamond | 1.24x | runtime is a wash |

Two clean findings:

1. **Agent loops: the runtime factor is 2–3.4x.** Message-heavy agent loops
   (object churn, serialization, schema handling) benefit substantially from
   V8's JIT and allocator.
2. **Trivial DAG scheduling: the runtime factor is ~1x.** Pregel's superstep
   machinery costs about the same on both runtimes — scheduling overhead is
   dominated by the framework's own bookkeeping, not language speed.

### What this means for the Mastra benchmark

- Timbal's ~10x agent-loop advantage over Mastra (`benchmarks/mastra`) is
  measured *against* a 2–3.4x runtime headwind. Runtime-adjusted, the
  architectural gap is larger than the raw tables show, not smaller.
- Mastra's ~3x bare-workflow win over Timbal is **not** explained by V8: the
  same-framework DAG ratio is ~1x. That win is Mastra's lean workflow engine
  (and it disappears once its observability is enabled).

### Reference tables (agent loop, p50)

| Scenario | Timbal (traced) | LG.py (bare) | LG.js (bare) |
|----------|----------------:|-------------:|-------------:|
| Single tool | 280.4 µs | 2.84 ms | 1.38 ms |
| 3-step chain | 289.0 µs | 6.28 ms | 1.84 ms |
| Parallel tools | 285.5 µs | 3.39 ms | 1.04 ms |

Worth noting: Timbal with tracing on is ~4–6x faster than LangGraph.js bare —
a Python framework beating the same competitor's TypeScript implementation on
V8's home turf. Framework design dominates runtime choice at this scale.

Full latency/memory/burst/throughput tables for all six scenarios are in
`results/bench_agent.txt` and `results/bench_workflow.txt`. Memory columns use
per-runtime instruments (tracemalloc vs V8 heap growth) and must not be
ratio'd across languages.
