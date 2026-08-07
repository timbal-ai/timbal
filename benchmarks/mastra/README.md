# Timbal vs Mastra - Benchmarks

Pure framework overhead benchmarks. No real LLM/API calls. Timbal uses `TestModel`;
Mastra uses the AI SDK's `MockLanguageModelV4` test model.

**This is a cross-language comparison.** Timbal is Python, Mastra is TypeScript.
Read the methodology section before quoting any number.

**Environment:** Apple Silicon M-series. Timbal on CPython 3.11/asyncio;
Mastra on Node 22/V8. Raw output from full-mode runs is stored in `results/`.

---

## Methodology: what a cross-language comparison can and cannot say

Every other directory in `benchmarks/` compares two Python frameworks inside one
process. That is impossible here, so the design is:

- **Each framework runs in its native runtime.** Timbal runs on CPython/asyncio in
  the benchmark process; Mastra runs on Node/V8 in a subprocess (`bench_*.mjs`).
  No transpilation, no embedding, no artificial common denominator.
- **Everything else is held identical:** the same scenarios, the same
  fake-LLM-that-inspects-message-history trick, the same warmup counts, the same
  batch sizes, monotonic nanosecond timers on both sides (`time.perf_counter` /
  `process.hrtime.bigint`), forced GC between batches, one shared pre-built
  agent/workflow instance (creation excluded from timing).

### What the numbers mean

Latency, burst, and throughput compare **framework + language runtime stacks** —
the thing you actually deploy. If Mastra is faster at X, part of that is V8's JIT;
if Timbal is faster at Y, part of that is Timbal's architecture. The numbers do not
separate the two, and we don't pretend they do. They answer "what overhead do I ship
with", not "which codebase has better algorithms".

Concretely, expect two systematic runtime effects in Mastra's favor that have
nothing to do with either framework's design: V8 JIT-compiles hot loops (CPython
3.12 does not), and V8's allocation/GC path is faster for short-lived objects.
Timbal's wins happen *despite* that handicap; Mastra's wins should be read with
it in mind.

### What is explicitly NOT comparable

**Memory.** Python is measured with `tracemalloc` (exact peak of every traced
allocation); Node reports peak `heapUsed` growth over a batch (post-hoc, GC-timing
dependent, approximate). These are different instruments measuring different
things. The tables print both with their method labeled — never compute a
cross-language ratio from them. Within the Node columns (Mastra vs Mastra+obs)
the comparison is valid.

**p99 in short runs** — noisy on both sides, shown for completeness.

### Observability

Per the top-level benchmark philosophy, observability is included: nobody ships
without tracing. Timbal's tracing is built-in and always on. Mastra standalone
agents have no tracing, so Mastra gets two columns:

- **Mastra** — bare `new Agent(...)` / standalone workflow, no tracing.
- **Mastra+obs** — registered in a `Mastra` instance with `@mastra/observability`
  configured; spans are built and processed normally, and the exporter sink is a
  no-op subclass of `BaseExporter` (the analogue of LangSmith with HTTP mocked).
  Pending observability work is flushed (`Observability.flush()`) after warmup,
  between measured batches, and after every run inside the memory loop — the
  same points where the Timbal side clears its in-memory tracing storage — so
  no side carries deferred work into a timed phase or accumulates it into the
  memory measurement.

The honest apples-to-apples column against Timbal is **Mastra+obs**; the bare
column shows what Mastra costs with tracing off, which Timbal doesn't offer.

---

## Scenarios

Agent loop (`bench_agent.py` + `bench_agent.mjs`) — identical to every other
framework in `benchmarks/`:

- Single tool call: `LLM -> add -> LLM -> answer`
- Three-step chain: `LLM -> add -> LLM -> multiply -> LLM -> subtract -> LLM -> answer`
- Parallel tool calls: `LLM -> [add, multiply, negate] -> LLM -> answer`

Workflow DAG (`bench_workflow.py` + `bench_workflow.mjs`) — Mastra has a
first-class workflow engine (`createWorkflow` / `.then` / `.parallel`), so this is
a fair native-primitive comparison, same shapes as the LangGraph benchmark:

- Sequential: `A -> B -> C -> D`
- Fan-out/in: `A -> [B, C, D] -> E`
- Diamond: `A -> [B, C] -> D`

Mastra workflow runs include `createRun()` + `start()` — its normal programmatic
execution path. Timbal runs are `workflow(x=3).collect()`. Construction of the
workflow/agent object is excluded on both sides.

## Files

| File | What it does |
|------|--------------|
| `bench_agent.py` | Orchestrator: runs Timbal in-process, spawns `bench_agent.mjs`, prints combined tables |
| `bench_agent.mjs` | Mastra agent loop measurements, JSON on stdout |
| `bench_workflow.py` | Orchestrator for the DAG benchmark |
| `bench_workflow.mjs` | Mastra workflow measurements, JSON on stdout |
| `bench_lib.mjs` | Shared Node harness (percentiles, latency/memory/burst/throughput) |

## Setup

Timbal (repo root):

```bash
uv sync --dev
```

Mastra side (requires Node >= 20):

```bash
cd benchmarks/mastra && npm install
```

## Running

```bash
# Quick mode — sanity check
uv run python benchmarks/mastra/bench_agent.py --quick
uv run python benchmarks/mastra/bench_workflow.py --quick

# Full mode — results worth keeping
uv run python benchmarks/mastra/bench_agent.py
uv run python benchmarks/mastra/bench_workflow.py
```

No API keys required. Useful flags: `--timbal-only` skips the Node side;
`--mastra-json <file>` reuses a previous Node-side JSON result. To debug the
Mastra side alone: `node --expose-gc bench_agent.mjs --quick` (JSON on stdout,
progress on stderr).

---

## Results

Full-mode run, 2026-08-07: Apple Silicon M-series, CPython 3.11, Node v22.18,
`@mastra/core` 1.57.0, `@mastra/observability` 1.16.5, `ai` 7.0.56.
Raw outputs:

- `results/bench_agent.txt`
- `results/bench_workflow.txt`

### Agent loop

Full loop: prompt -> fake LLM -> tool call(s) -> fake LLM -> answer. Timbal has
tracing built in and always on; compare against Mastra+obs for parity.

| Scenario | Timbal p50 | Mastra p50 | Mastra+obs p50 |
|----------|-----------:|-----------:|---------------:|
| Single tool | 265.2 µs | 2.63 ms | 4.09 ms |
| 3-step chain | 274.8 µs | 3.74 ms | 6.64 ms |
| Parallel tools | 278.2 µs | 2.60 ms | 4.65 ms |

| Scenario | Timbal c=10 | Mastra c=10 | Mastra+obs c=10 |
|----------|------------:|------------:|----------------:|
| Single tool | 4,172/s | 512/s | 264/s |
| 3-step chain | 1,854/s | 284/s | 144/s |
| Parallel tools | 2,244/s | 374/s | 239/s |

Timbal's agent loop is roughly an order of magnitude faster per run and in
throughput — despite the V8-vs-CPython runtime handicap running the other way.
Mastra's `generate()` path does substantially more per call (message-list
management, schema conversion, processor pipeline), and enabling observability
roughly doubles its cost, while Timbal's numbers already include tracing.

### Workflow DAG

Trivial handlers, no LLM — pure scheduling overhead.

| Scenario | Timbal p50 | Mastra p50 | Mastra+obs p50 |
|----------|-----------:|-----------:|---------------:|
| Sequential | 315.1 µs | 95.8 µs | 289.4 µs |
| Fan-out/in | 467.6 µs | 130.7 µs | 337.9 µs |
| Diamond | 381.7 µs | 104.9 µs | 289.5 µs |

| Scenario | Timbal c=10 | Mastra c=10 | Mastra+obs c=10 |
|----------|------------:|------------:|----------------:|
| Sequential | 4,373/s | 13,091/s | 4,071/s |
| Fan-out/in | 3,131/s | 9,433/s | 3,440/s |
| Diamond | 4,039/s | 11,265/s | 4,014/s |

This one cuts the other way and we report it as-is: Mastra's bare workflow
engine on V8 is ~3x faster than Timbal on trivial DAGs. That advantage is a mix
of a lean run path and V8 JIT-compiling the hot loop — the benchmark cannot and
does not separate the two. With observability enabled (the production-parity
column), Mastra lands at rough parity with Timbal on latency and throughput,
with Timbal keeping tracing on for every number.

### Takeaway

For agent workloads — the workload both frameworks exist for — Timbal's loop is
~9-24x faster with tracing on both sides. For raw DAG scheduling of trivial
steps, Mastra without tracing is faster; with tracing it's a wash. As always,
framework overhead vanishes behind real LLM latency; these numbers measure the
floor each framework sets, not the ceiling of your app.
