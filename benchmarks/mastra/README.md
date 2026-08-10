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

**That runtime factor is now measured, not guessed:** `benchmarks/langgraph_js/`
runs LangGraph — the same framework, same architecture — on both CPython and
V8. Result: V8 makes identical agent loops **2.1–3.4x** faster and identical
DAG scheduling **~1.0–1.2x** faster (a wash). Use those factors when reading
the tables below.

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

The agent benchmark also includes an **AI SDK** column: raw Vercel AI SDK
`generateText` with the same tools and mock model — the substrate Mastra is
built on. It is a bare library call (no tracing, no memory, no agent
abstractions), included to decompose Mastra's cost into "the substrate" vs
"Mastra's layer on top", and as a best-case V8 baseline.

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

Full-mode runs, 2026-08-07 (workflow) and 2026-08-10 (agent, adding the AI SDK
column): Apple Silicon M-series, CPython 3.11, Node v22.18, `@mastra/core`
1.57.0, `@mastra/observability` 1.16.5, `ai` 7.0.56.
Raw outputs:

- `results/bench_agent.txt`
- `results/bench_workflow.txt`

### Agent loop

Full loop: prompt -> fake LLM -> tool call(s) -> fake LLM -> answer. Timbal has
tracing built in and always on; compare against Mastra+obs for parity.

| Scenario | Timbal p50 | AI SDK p50 | Mastra p50 | Mastra+obs p50 |
|----------|-----------:|-----------:|-----------:|---------------:|
| Single tool | 265.3 µs | 156.1 µs | 2.33 ms | 4.20 ms |
| 3-step chain | 296.3 µs | 340.6 µs | 3.99 ms | 6.92 ms |
| Parallel tools | 285.9 µs | 219.8 µs | 2.55 ms | 4.33 ms |

| Scenario | Timbal c=10 | AI SDK c=10 | Mastra c=10 | Mastra+obs c=10 |
|----------|------------:|------------:|------------:|----------------:|
| Single tool | 3,982/s | 7,996/s | 555/s | 278/s |
| 3-step chain | 1,916/s | 3,215/s | 288/s | 159/s |
| Parallel tools | 2,094/s | 5,497/s | 418/s | 233/s |

Timbal's agent loop is roughly an order of magnitude faster than Mastra per
run and in throughput — despite the V8-vs-CPython runtime handicap running the
other way (measured at 2.1–3.4x for agent loops in `benchmarks/langgraph_js/`).
Mastra's `generate()` path does substantially more per call (message-list
management, schema conversion, processor pipeline), and enabling observability
roughly doubles its cost, while Timbal's numbers already include tracing.

The AI SDK column decomposes Mastra's cost: raw `generateText` runs the
single-tool loop in ~156 µs, so **Mastra adds roughly 15x on top of its own
substrate**. It also shows the runtime effect honestly — a bare V8 library
call edges Timbal on single-round scenarios and loses on the 3-step chain,
while carrying none of Timbal's agent semantics (tracing, memory, structured
events, multi-turn state). A full traced agent runtime landing within ~2x of a
bare substrate call is the context for every other column.

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
engine on V8 is ~3x faster than Timbal on trivial DAGs. The calibration suite
(`benchmarks/langgraph_js/`) settles what we previously couldn't separate:
identical DAGs run at ~1.0–1.2x across CPython and V8, so this win is
**Mastra's lean engine, not the runtime**. Credit where due. With
observability enabled (the production-parity column), the advantage disappears
— rough parity with Timbal on latency and throughput, with Timbal keeping
tracing on for every number.

### Takeaway

For agent workloads — the workload both frameworks exist for — Timbal's loop is
~9-23x faster with tracing on both sides, and the runtime calibration says
that gap is architectural, not linguistic. For raw DAG scheduling of trivial
steps, Mastra without tracing is genuinely faster (engine design, not V8);
with tracing it's a wash. As always, framework overhead vanishes behind real
LLM latency; these numbers measure the floor each framework sets, not the
ceiling of your app.
