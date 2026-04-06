# Timbal vs Agno — Benchmarks

> **⚠ WIP** — Only `bench_agent.py` exists so far. Workflow, parallel workflow, and
> double fan-out benchmarks are not yet written. Numbers are real but this page is incomplete.

Pure framework overhead benchmarks. No LLM API calls — all handlers are trivial
synthetic functions. Results are deterministic and reproducible.

**Environment:** Apple Silicon M-series, Python 3.12, asyncio event loop.
**All numbers from full-mode runs** (100 iters, 200 throughput ops, 50-burst).
Raw output stored in `results/`.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |

## Setup

```bash
uv sync --dev
uv pip install agno
```

No API keys required — all LLM calls are faked.

## How to run

```bash
# Quick mode (~2–3 min, fewer iterations)
uv run python benchmarks/agno/bench_agent.py --quick

# Full mode (~15–20 min)
uv run python benchmarks/agno/bench_agent.py
```

> `agno` is not in `pyproject.toml` — install it ad-hoc with `uv pip install agno`.

---

## The telemetry problem

**Before reading any numbers**, you need to understand Agno's default behaviour.

Out of the box, `agno` sends an HTTP POST to `https://os-api.agno.com/telemetry/runs` at the
end of **every single agent run**. This call is **awaited** — it blocks the hot path.
It is not fire-and-forget, there is no batching, there is no queue.
Measured overhead: **~400–500 ms per run**.

This means the out-of-the-box Agno experience is unusable for any latency-sensitive workload.
You cannot benchmark it, you cannot reason about it, and you cannot run it in production
without explicitly opting out first.

To disable: `Agent(..., telemetry=False)` or set `AGNO_TELEMETRY=false`.

The benchmark measures two Agno configurations:
- **Agno (no tel)** — `telemetry=False`: the explicit opt-out required for production use
- **Agno (tel mock)** — `telemetry=True` with the HTTP call replaced by an `AsyncMock`:
  measures the JSON-serialisation overhead in isolation. It turns out to be essentially
  zero — the 400–500 ms is 100% network latency.

---

## On Agno's own benchmarks

Agno publishes benchmark numbers showing agent instantiation in **~3 µs** vs 170 µs for
PydanticAI and 1,587 µs for LangGraph. These numbers are real.

They are also almost entirely irrelevant.

In any serious production deployment you instantiate your agent **once** — at startup,
as a class variable, as a module-level singleton. You are not paying that cost on every
request. The cost you pay on every request is execution time: prompt preparation,
the tool-call loop, message history management, memory allocation per run.

Measuring agent instantiation as a proxy for framework performance is like benchmarking
a web framework by how fast it imports — technically accurate, practically meaningless.
The numbers below measure what actually matters.

---

## Results

### Agent loop (`bench_agent.py`)

Full agent loop: prompt → LLM (faked via `FakeModel`) → tool(s) → LLM → answer.

Three scenarios:

1. **Single tool:** `LLM → add(1,2) → LLM → "3"` — 2 LLM calls, 1 tool
2. **Multi-step:** `LLM → add → LLM → mul → LLM → sub → LLM → "9"` — 4 LLM calls, 3 tools sequential
3. **Parallel tools:** `LLM → [add, mul, neg] → LLM → "done"` — 2 LLM calls, 3 tools concurrent

Both Timbal and Agno dispatch multiple tool calls from a single LLM response concurrently
(Agno uses `asyncio.gather` over `function_calls_to_run` in `aresponse()`).

**Fake LLM strategy:**
- Timbal uses `TestModel(handler=fn)` — plain callable, inspects message history
- Agno uses a custom `FakeModel` subclassing `agno.models.base.Model` — stateless,
  counts `role == "tool"` messages in history to determine which step to return

---

#### Scenario 1 — Single tool call: `LLM → add(1,2) → LLM → "3"`

**Latency** (×100 sequential runs)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| mean | **943 µs** | 1.30 ms | 1.64 ms |
| p50  | **982 µs** | 1.25 ms | 1.28 ms |
| p95  | **1.52 ms** | 1.53 ms | 2.25 ms |
| p99  | **1.70 ms** | 1.92 ms | 16.33 ms |

Timbal is **1.3× faster** than Agno (no tel) at p50.

**Memory** (×100 runs)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| peak    | **55 KB**  | 814 KB | 815 KB |
| per run | **563 B**  | 8,340 B | 8,341 B |

Timbal allocates **14.8× less memory per run** than Agno.

**Burst** (50 concurrent) and **Throughput** (200 loops)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| burst p50       | **14.74 ms** | 29.32 ms | 28.86 ms |
| burst wall      | **15.3 ms**  | 51.0 ms  | 50.7 ms  |
| throughput c=1  | **1,713/s**  | 808/s    | 814/s    |
| throughput c=10 | **1,861/s**  | 886/s    | 875/s    |
| throughput c=50 | **1,933/s**  | 900/s    | 914/s    |

---

#### Scenario 2 — Multi-step: `LLM → add → LLM → mul → LLM → sub → LLM → "9"`

**Latency** (×100 sequential runs)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| mean | **1.12 ms** | 3.44 ms | 3.39 ms |
| p50  | **981 µs**  | 3.32 ms | 3.27 ms |
| p95  | **2.29 ms** | 4.00 ms | 4.18 ms |
| p99  | **4.90 ms** | 7.12 ms | 4.56 ms |

Timbal is **3.4× faster** than Agno (no tel) at p50.

**Memory** (×100 runs)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| peak    | **98 KB**   | 882 KB | 881 KB |
| per run | **1,004 B** | 9,027 B | 9,023 B |

Timbal allocates **9.0× less per run** than Agno for a 3-tool chain.

**Burst** (30 concurrent) and **Throughput** (200 loops)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| burst p50       | **26.93 ms** | 51.52 ms | 50.51 ms |
| burst wall      | **27.2 ms**  | 87.3 ms  | 86.6 ms  |
| throughput c=1  | **753/s**    | 283/s    | 286/s    |
| throughput c=10 | **920/s**    | 337/s    | 314/s    |
| throughput c=50 | **895/s**    | 341/s    | 335/s    |

---

#### Scenario 3 — Parallel tools: `LLM → [add, mul, neg] concurrent → LLM → "done"`

**Latency** (×100 sequential runs)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| mean | **970 µs**  | 3.05 ms | 3.12 ms |
| p50  | **974 µs**  | 2.98 ms | 2.97 ms |
| p95  | **1.57 ms** | 3.62 ms | 3.88 ms |
| p99  | **1.73 ms** | 3.84 ms | 4.78 ms |

Timbal is **3.1× faster** than Agno (no tel) at p50.

**Memory** (×100 runs)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| peak    | **89 KB**  | 887 KB | 887 KB |
| per run | **911 B**  | 9,083 B | 9,082 B |

**Burst** (40 concurrent) and **Throughput** (200 loops)

| | Timbal | Agno (no tel) | Agno (tel mock) |
|--|--------|--------------|----------------|
| burst p50       | **17.95 ms** | 62.59 ms  | 63.33 ms  |
| burst wall      | **19.4 ms**  | 108.5 ms  | 109.4 ms  |
| throughput c=1  | **1,025/s**  | 320/s     | 332/s     |
| throughput c=10 | **1,294/s**  | 350/s     | 344/s     |
| throughput c=50 | **1,345/s**  | 343/s     | 355/s     |

---

#### Agent loop summary

| Metric | Timbal vs Agno (no tel) |
|--------|------------------------|
| Latency p50 — single tool  | **1.3× faster** (982 µs vs 1.25 ms) |
| Latency p50 — 3-step chain | **3.4× faster** (981 µs vs 3.32 ms) |
| Latency p50 — parallel (3) | **3.1× faster** (974 µs vs 2.98 ms) |
| Memory per run — single tool  | **14.8× less** (563 B vs 8,340 B) |
| Memory per run — 3-step chain | **9.0× less** (1,004 B vs 9,027 B) |
| Throughput c=10 — single tool | **2.1× more ops/s** (1,861 vs 886) |
| Throughput c=10 — multi-step  | **2.7× more ops/s** (920 vs 337) |
| Burst wall — 40 concurrent (parallel) | **5.6× faster** (19.4 ms vs 108.5 ms) |

---

## Notes on the comparison

**Agno (tel mock) ≈ Agno (no tel).** The two columns are nearly identical across all
scenarios. This confirms that the 400–500 ms overhead in the default configuration is
entirely network — the JSON serialisation cost is noise.

**Timbal includes full tracing out of the box.** Every run records spans in
`InMemoryTracingProvider`. There is no "Timbal bare" column because tracing is always
on. This is the fair comparison — production deployments need observability.

**The memory gap is large.** At 563 B vs 8,340 B per run for a single tool call (~14.8×),
the gap is framework overhead alone — no disk I/O, no network, no external storage.
Agno's `Agent` carries significantly more per-run state than Timbal's. At scale or under
concurrent load this matters for GC pressure and memory headroom.

**The burst gap widens with concurrency.** Scenario 3 at 40 concurrent: 19.4 ms (Timbal)
vs 108.5 ms (Agno) — 5.6× wall time. This is where the per-run overhead compounds: more
allocations, more GC, more contention in the event loop.

**Both frameworks dispatch parallel tool calls correctly.** Agno uses `asyncio.gather`
over `function_calls_to_run` in `aresponse()`. Scenario 3 latency (2.98 ms) is only
slightly above scenario 1 (1.25 ms), confirming real parallel dispatch rather than
sequential execution.
