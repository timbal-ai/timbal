# Timbal vs CrewAI — Benchmarks

Pure framework overhead benchmarks. No real LLM API calls — all handlers are trivial
synthetic functions. Results are deterministic and reproducible.

**Environment:** Apple Silicon M-series, Python 3.12.8, asyncio event loop.
**Versions:** `timbal 1.3.3` · `crewai 1.13.0` · `agentops 0.4.21` · `openinference-instrumentation-crewai 1.1.1`
**Agent loop numbers from full-mode runs** (100 iters, 100 mem, 100 throughput ops).
**DAG/parallel numbers from full-mode runs** (200 iters, 200 burst).
Raw outputs are stored in `results/`.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: Timbal vs CrewAI sync, async (`akickoff`), sync+AgentOps, async+AgentOps |
| `bench_workflow.py` | Small DAG topologies: sequential, fan-out/in, diamond |
| `bench_parallel.py` | Wide parallel fan-out: root → [N branches] → sink |
| `bench_double_fanout.py` | Double fan-out: root → [N×p1] → aggregator → [N×p2] → sink |

## Setup

From the repo root, install Timbal and the benchmarked frameworks:

```bash
uv sync --dev
uv pip install "crewai==1.13.0" "agentops==0.4.21" "openinference-instrumentation-crewai==1.1.1"
```

No API keys required — all LLM calls are faked.

## How to run

```bash
# Quick mode (~2–5 min each, fewer iterations)
uv run python benchmarks/crewai/bench_agent.py --quick
uv run python benchmarks/crewai/bench_workflow.py --quick
uv run python benchmarks/crewai/bench_parallel.py --quick
uv run python benchmarks/crewai/bench_double_fanout.py --quick

# Full mode (~15–30 min each, results stored in results/)
uv run python benchmarks/crewai/bench_agent.py
uv run python benchmarks/crewai/bench_workflow.py
uv run python benchmarks/crewai/bench_parallel.py
uv run python benchmarks/crewai/bench_double_fanout.py
```

---

## CrewAI concurrency and throughput: the full picture

The ~30 ops/s throughput ceiling you'll see in the benchmarks is not a tuning problem
— it is structural. Understanding why requires looking at three things: what the
`kickoff` methods actually do, why concurrent execution is unstable, and what the
community has concluded after trying to work around it.

### The three `kickoff` methods

| Method | What it actually does |
|--------|-----------------------|
| `kickoff()` | Synchronous. Blocks the calling thread for the full duration. |
| `kickoff_async()` | **Not truly async.** A CrewAI maintainer confirmed: it is `asyncio.to_thread(self.kickoff, inputs)`. Same GIL limits and thread-safety issues as `kickoff()`. |
| `akickoff()` | Genuinely async — awaits all the way to `litellm.acompletion()`. Added late 2024. |

`kickoff_async()` is what most people reach for when they want "async CrewAI." It is a
thread wrapper. Everything below about sync limitations applies to it equally.

### Why concurrent execution is unstable

Three compounding problems — all confirmed by open GitHub issues:

**1. `Crew` instance state is mutable per run.**
`self.usage_metrics`, `self._kickoff_event_id`, and `self.stream` are all overwritten
on every call. The `kickoff_async()` source has a textbook race condition:
`self.stream = False` → `await ...` → `self.stream = True` in a `finally` block.
Two concurrent calls on the same instance corrupt each other.

**2. Global shared singletons.**
The `crewai_event_bus` is a module-level singleton. All concurrent crews emit into
the same bus. The default `KickoffTaskOutputsSQLiteStorage` writes to a single shared
file at `~/Library/Application Support/.../latest_kickoff_task_outputs.db` — with a
30-second SQLite lock timeout as mitigation, not a solution.

**3. Python GIL.**
`kickoff()` is CPU-bound Python. Multiple threads do not parallelize — they time-share.
Contention gets worse with more threads.

**From GitHub:**
- **[#2632](https://github.com/crewAIInc/crewAI/issues/2632):** Segfaults with concurrent crews. "When using multithreading or asyncio to make requests concurrently, we encountered segmentation fault errors." Reproducible with both threading and asyncio. The `FilteredStream` race was patched (PR #2818); underlying segfaults remained.
- **[#1234](https://github.com/crewAIInc/crewAI/issues/1234):** `kickoff_async()` calls run serially instead of concurrently. Sometimes the second crew never runs. Closed as "not planned."
- **[#4135](https://github.com/crewAIInc/crewAI/issues/4135):** Production failure cascade. `future.cancel()` returns `False` once a thread has started; `ThreadPoolExecutor` exits with `shutdown(wait=False)`; threads keep running, memory degrades, database connections leak, processes crash under OOM.
- **[Discussion #1538](https://github.com/crewAIInc/crewAI/discussions/1538):** "I cannot get multiple crews to execute concurrently... Things work fine when I force serial execution, but as soon as there is any parallelism things break down." Maintainer response: pointed at the `akickoff` docs. No acknowledgment of the mutable-instance race.

### The throughput ceiling is architectural

Even with `akickoff()` (truly async, no thread cap), throughput maxes out at ~28–31/s.
This benchmark uses fresh `Crew` instances per call — the correct pattern. The ceiling
exists because every call reconstructs the full `Crew` + `Agent` + `Task` + Pydantic
validator chain from scratch. There is no pooling, pre-warming, or instance reuse in
the framework or the community — the mutable instance state makes it structurally
impossible without reworking the core.

The official CrewAI "100x speed boost" blog post is about `uv install` being faster
than `pip install`. That is the level of runtime performance work happening at the
framework level.

### What the community does to work around it

There is **no official high-throughput guidance** from CrewAI. Community patterns:

- **Celery + Redis (most common):** Queue each `kickoff()` call as a background task.
  Each Celery worker handles one call at a time. Defang ships a starter kit for exactly
  this pattern. Throughput scales horizontally with worker count, not with concurrency.
- **Process-per-worker (Gunicorn/uvicorn):** True OS-level isolation per worker process.
  Each process has its own event bus singleton, SQLite connections, and GIL. The only
  approach that fully avoids the global singleton problem. Still bounded by per-call
  construction cost within each process.
- **`ThreadPoolExecutor` with `max_workers` cap:** What a CrewAI maintainer explicitly
  recommended for "running 1,000 crews simultaneously" — the official answer to
  "how do I scale" is "add more threads and hope." Under the GIL, this doesn't
  parallelize; it serialises more gracefully.
- **`akickoff()` with fresh instances:** The correct async path since late 2024. Removes
  the thread cap, improves burst scaling. Still hits the ~30/s ceiling. And as shown
  below, eliminates what little observability AgentOps provides.

The community verdict, after production deployments, is consistent:

> "By the time the crew functioned reliably, the use-case would have been implemented
> like 10 times if direct LLM API calls were used."
> — Ondřej Popelka, [CrewAI: Practical Lessons Learned](https://ondrej-popelka.medium.com/crewai-practical-lessons-learned-b696baa67242)

HackerNews consensus across multiple threads comparing agent frameworks: CrewAI is for
prototyping; LangGraph or PydanticAI for production throughput. Token overhead is also
real: a LangGraph vs CrewAI benchmark found CrewAI uses ~56% more tokens per request
due to unavoidable per-agent system prompts in the role-based model.

---

## Observability fairness

Timbal has built-in tracing on every run — no configuration required. A bare CrewAI
comparison is unfair: nobody ships without observability in production. But adding
observability to CrewAI reveals a deeper problem: **Crew and Flow are two separate
execution engines bolted into one package, and each has a different — and incomplete —
instrumentation story.**

### Crew agents: AgentOps, but only half of it works

The standard CrewAI observability tool is **AgentOps**. It patches `Crew.kickoff`,
`Agent.execute_task`, and related sync methods via `CrewaiInstrumentor().instrument()` —
wrapping every call in an OTel span and serialising run data for transmission.

**AgentOps does not instrument `akickoff()`.**
`CrewaiInstrumentor` only patches the synchronous execution paths. The native async
methods (`_aexecute_tasks`, `agent.aexecute_task`, etc.) are not touched. Running
`akickoff()` with AgentOps active produces exactly zero spans — the instrumentor fires
nothing. This was verified against `agentops 0.4.21` by reading
`agentops/instrumentation/agentic/crewai/instrumentation.py` directly.

This forces an impossible choice for every CrewAI team shipping to production:

| Path | Throughput | Observability |
|------|-----------|--------------|
| `kickoff()` / `kickoff_async()` (thread pool) | ~15/s at c=5 | AgentOps fires → observable |
| `akickoff()` (native async) | ~30/s | **AgentOps silent → zero observability** |

There is no configuration that gives you both. The benchmark therefore shows two fair
comparisons: **Timbal vs CA sync+AO** (both observable) and **Timbal vs CA async bare**
(both as fast as their respective models allow — and async CrewAI simply has no tracing
to account for).

In the benchmark, AgentOps HTTP transmission is mocked out — measuring the hot-path
instrumentation cost (patching overhead + span lifecycle + memory) with no network I/O.
`instrument()` patches globally and cannot be undone; bare measurements run first, then
AgentOps is activated.

### Flow DAGs: OpenInference, a different ecosystem entirely

AgentOps does not instrument Flow at all — `@start`/`@listen` dispatch runs through
`Flow._execute_method()`, which is completely separate from the Crew codepath AgentOps
patches.

The OTel standard for Flow is **OpenInference** (`openinference-instrumentation-crewai`
v1.1.1), which patches `_execute_method()` directly — producing one span per
`@start`/`@listen` invocation. This is a different package, a different instrumentation
surface, and a different activation/deactivation lifecycle from AgentOps.

So the two benchmark sets use two different observability stacks not by choice, but
because CrewAI ships two different execution engines that have never been unified under a
single tracing story:

- `bench_workflow.py` — fixed `@start`/`@listen` topology → **Flow+OI** column (4–5 spans per kickoff)
- `bench_parallel.py` / `bench_double_fanout.py` — dynamic N via `type()` → **Flow steps** column (N+3 / 2N+3 spans per kickoff, structurally equivalent to Timbal's per-branch model)

In all cases, HTTP export is mocked via `NullExporter` — measuring span creation +
attribute serialisation + processor pipeline with no network I/O. OpenInference supports
`uninstrument()`, so bare and instrumented measurements run cleanly isolated.

---

## Results

### 1. Agent loop (`bench_agent.py`)

Five columns:
- **Timbal** — `TestModel` + native async + built-in InMemory tracing (always on)
- **CA (sync)** — `Crew.kickoff()` via `run_in_executor`; what most teams run. Note: `kickoff_async()` is identical — it is `asyncio.to_thread(self.kickoff)`.
- **CA (async)** — `Crew.akickoff()` — native async path added late 2024
- **CA sync+AO** — CA sync + AgentOps (HTTP mocked)
- **CA async+AO** — CA async + AgentOps (AgentOps does not instrument `akickoff()` — cost is ~zero)

Burst concurrency: Timbal and CA async are fully concurrent (no cap). CA sync is capped
at 5 threads — above that, deadlocks occurred in development (#2632).

#### Scenario 1 — Single tool call: `LLM → add(1,2) → LLM → "3"`

**Latency** (×100 sequential runs)

|  | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| mean | **953.9 µs** | 3.32 ms | 3.63 ms | 4.74 ms | 3.09 ms |
| p50  | **978.2 µs** | 3.19 ms | 3.20 ms | 4.40 ms | 3.05 ms |
| p95  | **1.58 ms**  | 4.58 ms | 5.99 ms | 6.50 ms | 4.03 ms |
| p99  | **1.91 ms**  | 5.53 ms | 22.28 ms | 12.70 ms | 4.45 ms |

`akickoff()` is similar to sync kickoff at p50 (native async vs thread pool). Both 3–5×
slower than Timbal.

**Sequential memory** (×100 runs)

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| peak    | **58.6 KB**    | 986.4 KB   | 987.3 KB   | 20,272.3 KB | 987.5 KB |
| per run | **600 B**      | 10,101 B   | 10,110 B   | 207,588 B   | 10,112 B |

CA async+AO per-run (10,112 B) is within noise of CA async bare (10,110 B) — AgentOps
allocates zero extra on the async path. CA sync+AO: **208 KB per run** — 346× more than Timbal.

**Burst latency** (30 concurrent)

| | Timbal | CA sync (≤5 threads) | CA async (uncapped) | CA sync+AO | CA async+AO |
|--|--------|----------------------|---------------------|------------|-------------|
| p50  | **9.51 ms**  | 869.97 ms   | 498.83 ms  | 880.31 ms   | 502.52 ms |
| p75  | **9.64 ms**  | 1,046.63 ms | 696.11 ms  | 1,065.39 ms | 697.84 ms |
| p95  | **9.67 ms**  | 1,181.97 ms | 864.15 ms  | 1,326.15 ms | 870.23 ms |
| wall | **9.9 ms**   | 1,441.9 ms  | 893.1 ms   | 1,570.7 ms  | 899.4 ms  |

CA async burst wall (893 ms) is 1.6× better than CA sync (1,442 ms) — no thread
serialisation. Still 90× slower than Timbal.

**Burst memory** (30 concurrent — peak with all jobs in-flight, no GC)

| | Timbal | CA sync (≤5 threads) | CA async (uncapped) | CA sync+AO | CA async+AO |
|--|--------|----------------------|---------------------|------------|-------------|
| peak    | 1,121.3 KB | **811.6 KB**  | 1,724.1 KB | 1,168.7 KB | 1,726.1 KB |
| per run | 38,274 B   | **27,701 B**  | 58,848 B   | 39,893 B   | 58,919 B   |

CA sync burst memory is lower than CA async — because CA sync runs only 5 threads
concurrently (30 jobs in 6 serial batches), while CA async runs all 30 simultaneously.
CA async+AO burst memory matches CA async bare — confirms AgentOps is inert on async.

**Throughput** (100 loops)

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| c=1  | **1,692/s** | 31/s | 31/s | 30/s | 31/s |
| c=5  | **1,952/s** | 15/s | 31/s | 15/s | 31/s |
| c=20 | **1,935/s** | 17/s | 30/s | 20/s | 31/s |

CA async throughput is flat across concurrency levels (30–31/s) — coroutines don't queue.
CA sync degrades at c=5+ — thread cap means requests serialise. Both hit the same ~30/s
absolute ceiling regardless: per-call `Crew`/`Agent`/`Task` construction is the bottleneck,
not the concurrency model.

---

#### Scenario 2 — Multi-step: `LLM → add → LLM → mul → LLM → sub → LLM → answer`

**Latency** (×100 sequential runs)

|  | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| mean | **1.38 ms**  | 4.49 ms | 3.25 ms | 4.94 ms | 3.50 ms |
| p50  | **1.04 ms**  | 4.19 ms | 3.22 ms | 4.80 ms | 3.44 ms |
| p95  | **1.63 ms**  | 7.14 ms | 4.11 ms | 6.57 ms | 4.62 ms |
| p99  | **38.87 ms** | 8.87 ms | 4.89 ms | 7.82 ms | 6.09 ms |

**Sequential memory** (×100 runs)

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| peak    | **98.0 KB**  | 1,339.6 KB | 1,285.0 KB | 29,997.6 KB | 1,283.3 KB |
| per run | **1,003 B**  | 13,717 B   | 13,158 B   | 307,176 B   | 13,141 B   |

CA sync+AO: **307 KB per run** — 306× more than Timbal. CA async+AO within noise of CA async bare.

**Burst latency** (20 concurrent) and **burst memory**

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| burst p50        | **19.44 ms** | 761.79 ms  | 347.46 ms  | 769.41 ms  | 367.45 ms  |
| burst wall       | **19.6 ms**  | 1,222.7 ms | 599.8 ms   | 1,211.8 ms | 627.0 ms   |
| burst peak (mem) | 1,384.7 KB   | **929.6 KB** | 1,496.4 KB | 1,539.4 KB | 1,498.0 KB |
| burst per run    | 70,897 B     | **47,595 B** | 76,615 B   | 78,815 B   | 76,697 B   |

**Throughput** (100 loops)

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| c=1  | **383/s** | 31/s | 30/s | 27/s | 29/s |
| c=5  | **727/s** | 13/s | 32/s | 16/s | 30/s |
| c=20 | **788/s** | 16/s | 26/s | 18/s | 30/s |

---

#### Scenario 3 — Parallel tools: `LLM → [add, mul, neg] → LLM → answer`

Timbal dispatches all 3 tool calls concurrently from one LLM response.
CrewAI uses ReAct — one tool per LLM call, 3 sequential steps instead.

**Latency** (×100 sequential runs)

|  | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| mean | **1.45 ms**  | 3.86 ms | 3.36 ms | 5.14 ms | 3.41 ms |
| p50  | **964.0 µs** | 3.70 ms | 3.27 ms | 4.87 ms | 3.36 ms |
| p95  | **1.91 ms**  | 5.61 ms | 4.67 ms | 8.49 ms | 4.56 ms |
| p99  | **43.90 ms** | 6.13 ms | 5.35 ms | 12.54 ms | 5.34 ms |

**Sequential memory** (×100 runs)

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| peak    | **89.1 KB** | 1,303.2 KB | 1,250.8 KB | 28,966.3 KB | 1,250.0 KB |
| per run | **912 B**   | 13,345 B   | 12,808 B   | 296,615 B   | 12,800 B   |

CA async+AO per-run (12,800 B) is slightly *lower* than CA async bare (12,808 B) — noise.
AgentOps is fully inert on the async path. CA sync+AO: **297 KB/run** — 325× more than Timbal.

**Burst latency** (25 concurrent) and **burst memory**

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| burst p50        | **11.92 ms** | 694.84 ms  | 445.03 ms  | 844.18 ms  | 433.60 ms  |
| burst wall       | **13.0 ms**  | 1,891.1 ms | 792.3 ms   | 1,693.2 ms | 780.7 ms   |
| burst peak (mem) | 1,583.6 KB   | **843.5 KB** | 1,864.8 KB | 1,501.7 KB | 1,862.3 KB |
| burst per run    | 64,865 B     | **34,551 B** | 76,381 B   | 61,508 B   | 76,279 B   |

**Throughput** (100 loops)

| | Timbal | CA sync | CA async | CA sync+AO | CA async+AO |
|--|--------|---------|----------|------------|-------------|
| c=1  | **1,151/s** | 30/s | 28/s | 29/s | 28/s |
| c=5  | **1,206/s** | 16/s | 30/s | 17/s | 30/s |
| c=20 | **1,290/s** | 19/s | 31/s | 21/s | 30/s |

---

#### Agent loop summary

| Metric | Timbal vs CA sync+AO | Timbal vs CA async (no obs) |
|--------|----------------------|-----------------------------|
| Latency p50 — single tool    | **4.5× faster** (978 µs vs 4.40 ms)  | **3.3× faster** (978 µs vs 3.20 ms)  |
| Latency p50 — 3-step chain   | **4.6× faster** (1.04 ms vs 4.80 ms) | **3.1× faster** (1.04 ms vs 3.22 ms) |
| Latency p50 — parallel tools | **5.1× faster** (964 µs vs 4.87 ms)  | **3.4× faster** (964 µs vs 3.27 ms)  |
| Memory/run — single tool     | **346× less** (600 B vs 208 KB)       | **17× less** (600 B vs 10.1 KB)      |
| Memory/run — 3-step chain    | **306× less** (1.0 KB vs 307 KB)      | **13× less** (1.0 KB vs 13.2 KB)     |
| Throughput c=5 — single tool | **130× more ops/s** (1,952 vs 15)     | **63× more ops/s** (1,952 vs 31)     |
| Throughput c=5 — multi-step  | **45× more ops/s** (727 vs 16)        | **23× more ops/s** (727 vs 32)       |
| Burst wall — 30 concurrent   | **159× faster** (9.9 ms vs 1,571 ms)  | **90× faster** (9.9 ms vs 893 ms)    |

`akickoff()` improves burst and concurrency scaling — but per-call `Crew`/`Agent`/`Task`
construction caps throughput at ~30/s regardless, and AgentOps fires no spans on the
async path so there is no observability.

---

### 2. Small DAG topologies (`bench_workflow.py`)

Three columns — all measured with equivalent hot-path observability cost:
- **Timbal** — built-in InMemory tracing always on (span creation, storage, ContextVar propagation per step)
- **Flow bare** — no instrumentation; fastest baseline
- **Flow+OI** — OpenInference (`openinference-instrumentation-crewai` v1.1.1) patches `Flow._execute_method()` — one span per `@start`/`@listen` invocation (4–5 spans per kickoff for these topologies); `NullExporter` mocks HTTP export

Three fixed topologies with trivial async handlers. Both frameworks use native asyncio.

**Scenario 1 — Sequential (A → B → C → D)**

| | Timbal | Flow bare | Flow+OI |
|--|--------|-----------|---------|
| p50 latency       | **1.26 ms** | 1.47 ms | 1.91 ms |
| p95 latency       | **1.56 ms** | 1.99 ms | 2.19 ms |
| burst p50 (200×)  | **114 ms** | 142 ms | 271 ms |
| burst wall (200×) | **201 ms** | 230 ms | 359 ms |
| throughput c=10   | **1,203/s** | 776/s | 665/s |
| throughput c=200  | **947/s** | 827/s | 532/s |
| memory per run    | **324 B** | 1,709 B | 1,840 B |

**Scenario 2 — Fan-out/in (A → [B, C, D] → E)**

| | Timbal | Flow bare | Flow+OI |
|--|--------|-----------|---------|
| p50 latency       | 1.42 ms | **1.33 ms** | 1.87 ms |
| p95 latency       | **1.66 ms** | 1.59 ms | 2.17 ms |
| burst p50 (200×)  | 244 ms | **128 ms** | 379 ms |
| burst wall (200×) | 245 ms | **209 ms** | 452 ms |
| throughput c=10   | **911/s** | 557/s | 494/s |
| throughput c=200  | 718/s | **826/s** | 545/s |
| memory per run    | **431 B** | 1,885 B | 2,031 B |

**Scenario 3 — Diamond (A → [B, C] → D)**

| | Timbal | Flow bare | Flow+OI |
|--|--------|-----------|---------|
| p50 latency       | **1.23 ms** | 1.33 ms | 1.79 ms |
| p95 latency       | **1.41 ms** | 1.84 ms | 2.16 ms |
| burst p50 (200×)  | 209 ms | **115 ms** | 277 ms |
| burst wall (200×) | 210 ms | **190 ms** | 359 ms |
| throughput c=10   | **1,235/s** | 848/s | 721/s |
| throughput c=200  | **802/s** | 746/s | 633/s |
| memory per run    | **366 B** | 1,724 B | 1,880 B |

**Takeaway:** Timbal wins p50 latency in 2 of 3 scenarios and leads throughput across
the board. With comparable observability (Flow+OI), Timbal also wins burst latency in
Scenario 1; Flow bare retains a burst edge for fan-out shapes. Timbal uses **4.7–5.3×
less memory per run** vs Flow bare, and memory is the one dimension where adding OI
to Flow makes negligible difference (+130–146 B/run).

---

### 3. Wide parallel fan-out (`bench_parallel.py`)

Topology: `root → [N async branches] → sink`

Two columns — both with equivalent per-branch observability:
- **Timbal** — one asyncio.Task + Span per branch; N+3 spans per kickoff
- **Flow steps** — N individual `@listen` methods built dynamically via `type()`; OI instruments each → N+3 spans per kickoff; structurally equivalent to Timbal

**Key finding:** Flow steps (+133.3 µs/branch) is *slower* than Timbal (+105.8 µs/branch) at sequential p50 — CrewAI's `@listen` dispatch machinery is heavier than Timbal's per-step Task+Span path. Under high burst concurrency (N=64), Flow steps converges with or beats Timbal (both carry equivalent span pipeline cost; Flow steps' lighter per-coroutine state tips the balance at extreme N).

#### Scenario A — Trivial branches (pure scheduling overhead)

**Latency p50** (×200 sequential runs)

| N | Timbal | Flow steps |
|---|--------|------------|
|  4 | **803 µs** | 2.12 ms |
|  8 | **1.16 ms** | 2.61 ms |
| 32 | **3.45 ms** | 5.81 ms |
| 64 | **7.13 ms** | 10.0 ms |

Per-branch overhead: **Timbal +105.8 µs** → **Flow steps +133.3 µs** — @listen dispatch > Timbal's Task+Span path.

**Burst p50** (200 concurrent)

| N | Timbal | Flow steps |
|---|--------|------------|
|  4 | **113 ms** | 245 ms |
| 16 | **510 ms** | 567 ms |
| 32 | **940 ms** | 955 ms |
| 64 | 2,471 ms | **1,847 ms** |

At N=64 burst, Flow steps (1,847 ms) is ~25% faster than Timbal (2,471 ms) — both carry equivalent per-branch observability; Flow steps' lighter per-coroutine state (no full `RunContext` per branch) reduces contention at 200×64 concurrent tasks.

#### Scenario B — 1 ms async sleep per branch (parallelism verification)

**Latency p50** (×200 sequential runs)

| N | Timbal | Flow steps |
|---|--------|------------|
|  4 | **1.96 ms** | 3.22 ms |
|  8 | **2.14 ms** | 3.37 ms |
| 32 | **3.64 ms** | 5.39 ms |
| 64 | **6.82 ms** | 9.82 ms |

Per-branch overhead with real work: **Timbal +81.4 µs** → **Flow steps +111.8 µs**.

---

### 4. Double fan-out (`bench_double_fanout.py`)

Topology: `root → [N×p1] → aggregator → [N×p2] → sink` (two full fan-out cycles)

Two columns — same structure as Section 3. Flow steps: full `@listen` topology with OI → 2N+3 spans per kickoff.

**Per-branch overhead:** **Timbal +131.0 µs** → **Flow steps +209.7 µs**. Flow steps is ~1.6× more expensive per branch at sequential p50. Under burst, they converge — both carry 2N+3 spans per kickoff.

#### Scenario A — Trivial branches (latency p50)

| N | Steps | Timbal | Flow steps |
|---|-------|--------|------------|
|  16 |  35 |  **3.92 ms** | 6.09 ms |
|  32 |  67 |  **7.53 ms** | 11.8 ms |
|  64 | 131 | **14.4 ms** | 21.8 ms |
| 128 | 259 | **33.2 ms** | 52.9 ms |

**Burst p50** (100 concurrent)

| N | Steps | Timbal | Flow steps |
|---|-------|--------|------------|
|  16 |  35 |  **504 ms** | 704 ms |
|  32 |  67 |  1,014 ms | **1,021 ms** |
|  64 | 131 | 2,214 ms | **2,198 ms** |
| 128 | 259 | **4,838 ms** | 5,329 ms |

At N=32–64 burst, Timbal and Flow steps converge — equivalent observability cost dominates. At N=128, Timbal retains a modest edge.

#### Scenario B — 1 ms async sleep per branch (latency p50)

| N | Steps | Timbal | Flow steps |
|---|-------|--------|------------|
|  16 |  35 |  **5.09 ms** | 6.50 ms |
|  64 | 131 | **14.5 ms** | 21.2 ms |
| 128 | 259 | **33.1 ms** | 53.8 ms |

**Burst p50** (100 concurrent)

| N | Steps | Timbal | Flow steps |
|---|-------|--------|------------|
|  16 |  35 | 729 ms | **613 ms** |
|  32 |  67 | 1,255 ms | **1,163 ms** |
|  64 | 131 | **2,499 ms** | 2,549 ms |
| 128 | 259 | **5,531 ms** | 6,346 ms |

With 1 ms async work, Flow steps has a burst edge at N≤32 (lighter per-branch state under task load), while Timbal catches up at N=64+ where span overhead equalises.

#### Memory

Sequential peak (per run) and burst peak (100 concurrent, no GC):

| N | Steps | T (InMemory) | T (no tracing) | Flow steps |
|---|-------|-------------|----------------|------------|
| 16 seq | 35 | 294.8 KB | **283.4 KB** | 345.3 KB |
| 128 seq | 259 | 1.7 MB | **1.7 MB** | 694.2 KB |
| 16 burst | 35 | 23.2 MB | **23.2 MB** | 11.2 MB |
| 128 burst | 259 | 165.7 MB | **165.7 MB** | 58.8 MB |

Timbal InMemory ≈ Timbal no-tracing throughout — the tracing cost is CPU time, not heap.
Flow steps memory (58.8 MB at N=128 burst) is ~2.8× lighter than Timbal because `@listen` methods
share a single `self.state` Pydantic model rather than holding a full `RunContext` per branch.

---

## Framework ergonomics

### The design that causes the complexity

CrewAI launched in early 2023 around a specific metaphor: **AI as a crew of employees**.
Each agent is a persona — you give it a `role` ("Senior Researcher"), a `goal` ("Find
accurate information"), and a `backstory` ("You have 10 years of experience…"). Tasks are
assigned to agents like job tickets. The crew is assembled and "kicked off" like a project
kickoff meeting.

This made intuitive sense in early 2023, when models were weaker and persona scaffolding
genuinely helped coax useful output from them. It also made the framework immediately
legible to non-engineers: anyone who manages people could reason about "assigning tasks to
agents."

The cost is that this metaphor hardcoded assumptions that are now liabilities:

- **Every tool needs to be a class.** Because tools in 2023 needed heavy schema scaffolding
  for reliable parsing, every tool requires a Pydantic input schema and a `BaseTool`
  subclass. Plain Python functions are not accepted. This was a reasonable tradeoff when
  models were bad at JSON; in 2025 it is ceremony.

- **Every call reconstructs the full object graph.** `Crew` + `Agent` + `Task` + `Crew`
  were designed around a single-run "kickoff" model, not a server handling requests.
  Instance state is mutated per run (`self.usage_metrics`, `self.stream`, `self._kickoff_event_id`),
  making reuse across concurrent calls impossible without reworking the core. The
  ~30 ops/s ceiling is a direct consequence.

- **`akickoff()` is a retrofit.** Native async support was added in late 2024, years after
  the framework launched. The async path works but nothing was redesigned around it —
  AgentOps doesn't instrument it, the mutable-state problems remain, and the concurrency
  story was never resolved.

- **Flow inherits the same class-decorator pattern.** `@start`/`@listen` wire up at class
  definition time — a pattern from the Django/Flask era. To create N branches dynamically,
  you have to use `type()` Python metaprogramming to build a class at runtime. This is not
  a quirk; it is the only way the framework allows programmatic DAG construction.

The result is that things that feel natural in Python — passing a function, using a for
loop, building pipelines programmatically — require ceremony at every layer.

---

Lines of code below are non-blank framework-specific lines only (tool definitions, state
schema, step/node wiring). Benchmark harness, measurement helpers, and fake LLM response
logic are excluded.

### Agent loop (`bench_agent.py`)

**Lines of code** (tool definitions + agent assembly, Scenario 2: 3-tool chain)

| | Timbal | CrewAI |
|--|--------|--------|
| Lines | 13 | 44 |

Timbal: four plain Python functions (def + docstring + return = 3 lines each = 12 lines)
plus a single `Agent(model=..., tools=[...])` call (1 line). Total: 13 lines.

CrewAI: four Pydantic input-schema classes (11 lines), four `BaseTool` subclasses with
`name`/`description`/`args_schema`/`_run` fields (20 lines), plus
`CrewLLM + CrewAgent + CrewTask + Crew` assembly (13 lines). Total: 44 lines.
No plain functions accepted as tools — each requires a schema class and a `BaseTool` subclass.

### Small DAG topologies (`bench_workflow.py`)

**Lines of code** (non-blank framework-specific lines: handler functions + wiring)

| Topology | Timbal | CrewAI Flow |
|----------|--------|-------------|
| Sequential (A → B → C → D)   | 16 | 15 |
| Fan-out/in (A → [B,C,D] → E) | 24 | 23 |
| Diamond (A → [B,C] → D)      | 20 | 19 |

For fixed topologies the LOC is nearly identical — Flow's `@start`/`@listen` decorator
pattern maps cleanly to Timbal's `.step()` chain. The difference is in data flow:
Flow requires each step to write its output into `self.state["key"]` (explicit mutation),
while Timbal reads step outputs from immutable context (`get_run_context().step_span("name").output`).
For small fixed DAGs, both work; the distinction compounds at scale.

### Dynamic fan-out (`bench_parallel.py` / `bench_double_fanout.py`)

**Lines of code** (non-blank framework-specific lines: state schema + branch factory + wiring)

| | Timbal | CrewAI Flow |
|--|--------|-------------|
| Wide fan-out (N branches)    | 36 | 24 |
| Double fan-out (2×N branches) | 63 | 41 |

CrewAI is more compact per-file — `self.state.results.append(result)` inside each `@listen`
method is concise. But the compact line count obscures what is required to make it work.
The fundamental problem is how dynamic N is expressed:

```python
# Timbal — a for loop
for i in range(n):
    wf.step(branch_fn_i, depends_on=["root"], x=root_getter())
```

```python
# CrewAI Flow — Python metaprogramming
methods = {"root": start()(root_fn), "__module__": __name__, ...}
for i in range(n):
    methods[f"branch_{i}"] = listen(root_m)(branch_fn_i)
FlowClass = type(f"DynamicFlow_{n}", (Flow[StateModel],), methods)
```

Timbal uses `.step()` in a for loop — the same pattern you'd use to build any list.
CrewAI requires `type()` to dynamically construct a class at runtime, because `@listen`
decorators resolve their wiring at class definition time. On top of that, you need a
Pydantic state model (`class _WideState(BaseModel): results: list = Field(default_factory=list)`)
and a `Flow[StateModel]` base class — because there is no fan-in without shared mutable
state. For a reader unfamiliar with Python metaclasses, this code is opaque. For a
reader who is familiar, it is a sign that the framework was not designed for this use case.

---

## Summary

### Agent loop (the structural ceiling)

The ~30 ops/s throughput ceiling is not a tuning problem — it is the cost of
reconstructing the `Crew` + `Agent` + `Task` + validator graph on every call.
There is no pool, no pre-warming, no reuse pattern in the ecosystem. The community
workarounds (Celery queues, process-per-worker) scale horizontally, not vertically.

| Metric | Timbal vs CA sync+AO | Timbal vs CA async (no obs) |
|--------|----------------------|-----------------------------|
| Latency p50 — single tool    | **4.5× faster** (978 µs vs 4.40 ms)   | **3.3× faster** (978 µs vs 3.20 ms)  |
| Latency p50 — 3-step chain   | **4.6× faster** (1.04 ms vs 4.80 ms)  | **3.1× faster** (1.04 ms vs 3.22 ms) |
| Memory/run — single tool     | **346× less** (600 B vs 208 KB)        | **17× less** (600 B vs 10.1 KB)      |
| Throughput c=5 — single tool | **130× more ops/s** (1,952 vs 15)      | **63× more ops/s** (1,952 vs 31)     |
| Burst wall — 30 concurrent   | **159× faster** (9.9 ms vs 1,571 ms)  | **90× faster** (9.9 ms vs 893 ms)    |

### DAG scheduling (Timbal vs CrewAI Flow)

The fair comparison pairs frameworks with equivalent per-step observability: Timbal (built-in tracing) vs Flow+OI / Flow steps (OpenInference). Flow bare is shown in `bench_workflow.py` as a zero-observability lower bound.

| Benchmark | Timbal p50 | Flow (no obs) p50 | Flow (with obs) p50 | Note |
|-----------|-----------|-----------------|---------------------|------|
| Sequential (4 steps)              | **1.26 ms** | 1.47 ms (bare) | 1.91 ms (+OI) | Timbal fastest |
| Fan-out/in (5 steps)              | 1.42 ms | **1.33 ms** (bare) | 1.87 ms (+OI) | Flow bare 6% faster |
| Diamond (4 steps)                 | **1.23 ms** | 1.33 ms (bare) | 1.79 ms (+OI) | Timbal fastest |
| Wide fan-out (trivial, N=64)      | **7.13 ms** | — | 10.0 ms (steps) | Timbal 29% faster (seq); Flow steps faster at burst N=64 |
| Double fan-out (trivial, N=128)   | **33.2 ms** | — | 52.9 ms (steps) | Timbal 37% faster (seq) |
| Double fan-out (1 ms sleep, N=128)| **33.1 ms** | — | 53.8 ms (steps) | Timbal 38% faster (seq) |

### Why the numbers look the way they do

**On agent loops:** CrewAI reconstructs a full object graph per call. `akickoff()` removes
the thread cap and improves burst scaling, but doesn't reduce construction cost — both
paths cap at ~30/s. It also eliminates what little observability AgentOps provides:
`CrewaiInstrumentor` only patches the sync path. The choice is observable-but-slow (sync)
or fast-but-blind (async). Timbal has neither tradeoff.

**On wide fan-out:** "Flow steps" (N individual `@listen` methods + OI) is the
equivalent-observability comparison — same N+3 spans per kickoff as Timbal. At sequential
p50, Flow steps (+133 µs/branch) is *slower* than Timbal (+106 µs/branch): CrewAI's `@listen`
dispatch chain (`_execute_listeners` → `_execute_single_listener` → `_execute_method` +
OI wrapping + Pydantic state proxy) is heavier per step than Timbal's asyncio.Task + Span.
Under high burst concurrency (N=64, 200 concurrent), Flow steps (1,847 ms) is ~25% *faster*
than Timbal (2,471 ms) — both carry the same span count, but Flow steps' lighter per-branch
state (no full `RunContext` per coroutine) reduces pressure at extreme task counts.

**On small DAGs:** With comparable observability (Timbal vs Flow+OI — both carry 4–5 spans
per kickoff for these fixed topologies), Timbal wins p50 latency in 2/3 scenarios, wins
burst in Scenario 1, and leads throughput across the board. Timbal uses 4.7–5.3× less
memory per run. Flow bare retains a burst latency edge for fan-out shapes. Flow+OI is
consistently slowest on burst — the 4–5 spans add contention at c=200.

### What Timbal's observability buys you — included, no setup

- Full event streaming (`StartEvent`, `DeltaEvent`, `OutputEvent`)
- Per-step span recording with parent-child relationships
- ContextVar isolation for safe concurrent async execution
- Session persistence and run chaining across invocations
- Multi-provider LLM routing with a single API

To get equivalent observability in CrewAI on the sync path, you need AgentOps — at which
point CrewAI is 4–5× slower on agent loops and 208–346× more memory-intensive per run.
On the async path, AgentOps gives you nothing.
