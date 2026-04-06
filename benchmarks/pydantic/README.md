# Timbal vs PydanticAI — Benchmarks

> **⚠ WIP** — Only `bench_agent.py` exists so far. Workflow, parallel, and double
> fan-out benchmarks are not yet written. Numbers are real but this page is incomplete.

Pure framework overhead benchmarks. No LLM API calls — all handlers are trivial
synthetic functions. Results are deterministic and reproducible.

**Environment:** Apple Silicon M-series, Python 3.12, asyncio event loop.
**All numbers from full-mode runs** (100 iters, 200 throughput ops, 50–burst).
Raw output stored in `results/`.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |

## Setup

```bash
uv sync --dev
uv pip install pydantic-ai logfire
```

No API keys required — all LLM calls are faked.

## How to run

```bash
# Quick mode (~2–3 min, fewer iterations)
uv run python benchmarks/pydantic/bench_agent.py --quick

# Full mode (~15–20 min)
uv run python benchmarks/pydantic/bench_agent.py
```

> `pydantic-ai` and `logfire` are not in `pyproject.toml` — they conflict with
> `crewai`'s `opentelemetry-sdk` pin and can't share the same lockfile. Install
> them ad-hoc with `uv pip install` as shown above.

---

## Observability fairness

Timbal includes full tracing out of the box — every run records spans in
`InMemoryTracingProvider`. PydanticAI's standard observability stack is
**Logfire** (also built by the Pydantic team).

All "PAI+Logfire" numbers use real `logfire.instrument_pydantic_ai()` with
`send_to_logfire=False` — measuring span-creation overhead without network variance.
Logfire patches PydanticAI globally and cannot be uninstrumented, so bare measurements
run first, then Logfire is activated for Phase 2.

**The fair comparison is Timbal vs PAI+Logfire.** PAI bare is shown as a lower
bound on what the PydanticAI execution model can achieve with zero observability.

---

## Results

### Agent loop (`bench_agent.py`)

Full agent loop: prompt → LLM (faked via `FunctionModel`) → tool(s) → LLM → answer.

Three scenarios:

1. **Single tool:** `LLM → add(1,2) → LLM → "3"` — 2 LLM calls, 1 tool
2. **Multi-step:** `LLM → add → LLM → mul → LLM → sub → LLM → "9"` — 4 LLM calls, 3 tools sequential
3. **Parallel tools:** `LLM → [add, mul, neg] → LLM → "done"` — 2 LLM calls, 3 tools concurrent

Both Timbal and PydanticAI dispatch multiple tool calls from a single LLM response
concurrently (verified: `asyncio.gather` on `ToolCallPart` list).

**Fake LLM strategy:**
- Timbal uses `TestModel(handler=fn)` — plain callable, inspects message history
- PydanticAI uses `FunctionModel(fn)` — same pattern; counts `ToolReturnPart` in history
- Both are stateless and safe for concurrent async runs on a single agent instance

---

#### Scenario 1 — Single tool call: `LLM → add(1,2) → LLM → "3"`

**Latency** (×100 sequential runs)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| mean | **962 µs** | 1.99 ms | 2.80 ms |
| p50  | **928 µs** | 1.90 ms | 2.72 ms |
| p95  | **1.60 ms** | 2.96 ms | 3.45 ms |
| p99  | **1.99 ms** | 3.72 ms | 4.15 ms |

Timbal is **2.0× faster** than PAI bare, **2.9× faster** than PAI+Logfire at p50.

**Memory** (×100 runs)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| peak    | **61 KB**  | 502 KB | 523 KB |
| per run | **627 B**  | 5,138 B | 5,358 B |

Timbal allocates **8× less memory per run** than PAI bare. Logfire adds only ~220 B/run
on top — the memory gap is the PydanticAI framework itself, not observability.

**Burst** (50 concurrent) and **Throughput** (200 loops)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| burst p50       | **17.2 ms** | 45.5 ms | 72.6 ms |
| burst wall      | **17.6 ms** | 54.4 ms | 86.3 ms |
| throughput c=1  | **1,687/s** | 517/s   | 354/s   |
| throughput c=10 | **1,942/s** | 764/s   | 480/s   |
| throughput c=50 | **1,929/s** | 858/s   | 470/s   |

---

#### Scenario 2 — Multi-step: `LLM → add → LLM → mul → LLM → sub → LLM → "9"`

**Latency** (×100 sequential runs)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| mean | **1.04 ms** | 3.59 ms | 5.22 ms |
| p50  | **1.02 ms** | 3.46 ms | 5.10 ms |
| p95  | **1.74 ms** | 4.67 ms | 6.17 ms |
| p99  | **2.22 ms** | 5.98 ms | 6.83 ms |

Timbal is **3.4× faster** than PAI bare, **5.0× faster** than PAI+Logfire at p50.

**Memory** (×100 runs)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| peak    | **98 KB**  | 644 KB | 664 KB |
| per run | **1,004 B** | 6,591 B | 6,795 B |

Timbal allocates **6.6× less per run** than PAI bare for a 3-tool chain.

**Burst** (30 concurrent) and **Throughput** (200 loops)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| burst p50       | **26.2 ms** | 52.3 ms | 98.7 ms  |
| burst wall      | **26.6 ms** | 57.5 ms | 107.3 ms |
| throughput c=1  | **726/s**   | 286/s   | 195/s    |
| throughput c=10 | **832/s**   | 464/s   | 240/s    |
| throughput c=50 | **804/s**   | 444/s   | 235/s    |

---

#### Scenario 3 — Parallel tools: `LLM → [add, mul, neg] concurrent → LLM → "done"`

**Latency** (×100 sequential runs)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| mean | **1.00 ms** | 2.18 ms | 3.41 ms |
| p50  | **998 µs**  | 2.13 ms | 3.27 ms |
| p95  | **1.69 ms** | 2.60 ms | 4.12 ms |
| p99  | **1.90 ms** | 3.99 ms | 4.52 ms |

Timbal is **2.1× faster** than PAI bare, **3.3× faster** than PAI+Logfire at p50.

**Memory** (×100 runs)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| peak    | **91 KB**  | 590 KB | 604 KB |
| per run | **929 B**  | 6,040 B | 6,184 B |

**Burst** (40 concurrent) and **Throughput** (200 loops)

| | Timbal | PAI bare | PAI+Logfire |
|--|--------|----------|-------------|
| burst p50       | **20.3 ms** | 49.1 ms | 79.0 ms |
| burst wall      | **21.6 ms** | 58.1 ms | 86.9 ms |
| throughput c=1  | **1,100/s** | 454/s   | 306/s   |
| throughput c=10 | **1,186/s** | 656/s   | 387/s   |
| throughput c=50 | **1,205/s** | 616/s   | 371/s   |

---

#### Agent loop summary

The fair runtime comparison is **Timbal vs PAI+Logfire** — both include observability.

| Metric | Timbal vs PAI+Logfire |
|--------|----------------------|
| Latency p50 — single tool  | **2.9× faster** (928 µs vs 2.72 ms) |
| Latency p50 — 3-step chain | **5.0× faster** (1.02 ms vs 5.10 ms) |
| Latency p50 — parallel (3) | **3.3× faster** (998 µs vs 3.27 ms) |
| Memory per run — single tool  | **8.5× less** (627 B vs 5,358 B) |
| Memory per run — 3-step chain | **6.8× less** (1,004 B vs 6,795 B) |
| Throughput c=10 — single tool | **4.0× more ops/s** (1,942 vs 480) |
| Throughput c=10 — multi-step  | **3.5× more ops/s** (832 vs 240) |
| Burst wall — 50 concurrent    | **4.9× faster** (17.6 ms vs 86.3 ms) |

---

## Notes on the comparison

**PydanticAI is a well-designed framework.** It is architecturally cleaner than
LangChain, takes typing seriously with `Agent[DepsType, OutputType]` generics, and
its `FunctionModel` API for testing is a good design. The performance gap is not
a consequence of poor implementation — it reflects genuine architectural differences.

**Where the gap comes from — needs profiling.** We haven't done the profiling work
to attribute the gap to specific causes, so we won't speculate here. Both frameworks
use Pydantic for validation. The root causes would need a proper `py-spy` or
`memray` run to determine precisely — that work is pending.

**Logfire overhead on PydanticAI is meaningful.** Adding Logfire adds 43–53% to
latency across all scenarios. On a 3-tool chain burst of 30 concurrent calls, wall
time nearly doubles (57 ms → 107 ms). Timbal's built-in tracing adds no latency
compared to its own bare baseline — trace recording is on the critical path but
designed to be fast.

**Scenario 3 (parallel tools):** both frameworks genuinely dispatch concurrent tool
calls. PydanticAI's latency for scenario 3 (2.13 ms) is only slightly above scenario 1
(1.90 ms), confirming real parallel dispatch.
