# Timbal vs Agno — Benchmarks

Pure framework overhead benchmarks. No real LLM API calls. Agent model responses are
faked, and workflow steps are intentionally tiny so the numbers isolate framework cost.

**Environment:** Apple Silicon M-series, Python 3.12, asyncio event loop.  
**All numbers below are from full-mode runs.** Raw output is stored in `results/`.

---

## The Short Version

Timbal is very strong on **agent loops**. Against Agno with telemetry disabled, Timbal is
**2.2-4.3x faster at p50**, allocates **~9-14x less memory per run**, and handles
concurrent agent bursts much better.

Agno Workflow is the opposite story. With telemetry disabled, Agno's `Workflow` and
`Parallel` primitives are leaner than Timbal Workflow on tiny synthetic DAGs and large
fan-outs. That is real signal. Timbal's Workflow runtime gives richer DAG semantics and
built-in tracing, but it currently pays too much overhead for high-cardinality, tiny-step
graphs.

So the narrative is:

- **Agents:** Timbal wins clearly.
- **Workflow DAGs:** Agno is faster/lighter on tiny steps and wide `Parallel` workloads.
- **Telemetry:** Agno's default telemetry is still a serious footgun for agents: the
  network call is awaited on the hot path. Benchmarks use `telemetry=False`.
- **Next work:** keep the agent advantage, then optimize Timbal Workflow scheduling,
  branch fan-out, and concurrent burst behavior.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |
| `bench_workflow.py` | Small workflow shapes: sequential, fan-out/in, diamond |
| `bench_parallel.py` | Wide fan-out: root -> [N branches] -> sink |
| `bench_double_fanout.py` | Double fan-out: root -> [N phase 1] -> aggregate -> [N phase 2] -> sink |

## Setup

```bash
uv sync --dev
uv pip install agno
```

`agno` is not in `pyproject.toml`; install it ad-hoc for this benchmark.

## How To Run

```bash
# Quick mode
uv run python benchmarks/agno/bench_agent.py --quick
uv run python benchmarks/agno/bench_workflow.py --quick
uv run python benchmarks/agno/bench_parallel.py --quick
uv run python benchmarks/agno/bench_double_fanout.py --quick

# Full mode
uv run python benchmarks/agno/bench_agent.py
uv run python benchmarks/agno/bench_workflow.py
uv run python benchmarks/agno/bench_parallel.py
uv run python benchmarks/agno/bench_double_fanout.py
```

---

## Telemetry Fairness

Agno agents default to `telemetry=True`, which sends an awaited HTTP POST to
`https://os-api.agno.com/telemetry/runs` at the end of every run. In earlier measurement
this added roughly **400-500 ms per run**. That is network latency, not framework compute.

The agent benchmark reports:

- `Agno (no tel)`: `telemetry=False`, the production-sane baseline.
- `Agno (tel mock)`: `telemetry=True` with the HTTP call mocked, measuring local
  serialization overhead only.

Workflow benchmarks use `telemetry=False`. Timbal includes `InMemoryTracingProvider`
by default; there is no "Timbal bare" column.

Important caveat: same high-level observability does not mean identical granularity.
Timbal records each workflow step as a first-class span. Agno `Parallel` also records
parallel branch outputs, but its execution model is not identical to Timbal's DAG
scheduler.

---

## Agent Loop Results

Full loop: prompt -> fake LLM -> tool call(s) -> fake LLM -> answer.

| Scenario | Timbal p50 | Agno no-tel p50 | Timbal advantage |
|----------|-----------:|----------------:|-----------------:|
| Single tool | 786.6 us | 1.75 ms | 2.2x faster |
| 3-step chain | 946.9 us | 4.05 ms | 4.3x faster |
| Parallel tools | 749.8 us | 3.25 ms | 4.3x faster |

| Scenario | Timbal memory/run | Agno no-tel memory/run | Timbal advantage |
|----------|------------------:|-----------------------:|-----------------:|
| Single tool | 590 B | 8,363 B | 14.2x less |
| 3-step chain | 1,014 B | 9,042 B | 8.9x less |
| Parallel tools | 979 B | 9,077 B | 9.3x less |

| Scenario | Timbal throughput c=10 | Agno no-tel throughput c=10 | Timbal advantage |
|----------|-----------------------:|----------------------------:|-----------------:|
| Single tool | 1,634/s | 711/s | 2.3x higher |
| 3-step chain | 773/s | 302/s | 2.6x higher |
| Parallel tools | 1,167/s | 291/s | 4.0x higher |

Timbal's agent loop is the headline win. Agno carries much more per-run state and the
gap grows in multi-step and parallel-tool cases.

---

## Small Workflow Results

`bench_workflow.py` uses Agno's real `Workflow` and `Parallel` primitives.

| Scenario | Timbal p50 | Agno p50 | Notes |
|----------|-----------:|---------:|-------|
| Sequential A -> B -> C -> D | 1.00 ms | 440.2 us | Agno is 2.3x faster |
| Fan-out A -> [B,C,D] -> E | 1.10 ms | 620.1 us | Agno is 1.8x faster |
| Diamond A -> [B,C] -> D | 997.7 us | 539.5 us | Agno is 1.8x faster |

| Scenario | Timbal memory/run | Agno memory/run |
|----------|------------------:|----------------:|
| Sequential | 295 B | 502 B |
| Fan-out | 375 B | 518 B |
| Diamond | 336 B | 514 B |

Timbal allocates less in these small workflow runs, but Agno's runtime latency and
concurrent burst behavior are better. This points at scheduler/event overhead rather than
raw allocation volume.

---

## Wide Fan-Out Results

`bench_parallel.py` measures `root -> [N branches] -> sink`.

| Width | Timbal trivial p50 | Agno trivial p50 | Timbal async p50 | Agno async p50 |
|------:|-------------------:|-----------------:|-----------------:|---------------:|
| 4 | 1.03 ms | 578.6 us | 2.13 ms | 1.85 ms |
| 8 | 1.51 ms | 746.7 us | 2.55 ms | 1.99 ms |
| 16 | 2.49 ms | 1.01 ms | 3.29 ms | 2.05 ms |
| 32 | 4.24 ms | 1.26 ms | 4.74 ms | 2.38 ms |
| 64 | 7.98 ms | 2.44 ms | 8.79 ms | 3.25 ms |

| Width | Timbal burst p50 | Agno burst p50 | Timbal memory/run | Agno memory/run |
|------:|-----------------:|---------------:|------------------:|----------------:|
| 4 | 171.4 ms | 62.9 ms | 389 B | 525 B |
| 8 | 254.4 ms | 79.3 ms | 586 B | 583 B |
| 16 | 544.8 ms | 118.9 ms | 957 B | 613 B |
| 32 | 1,224.4 ms | 228.0 ms | 1,728 B | 842 B |
| 64 | 3,153.5 ms | 1,542.4 ms | 3,182 B | 1,142 B |

Agno scales better as branch count rises. Timbal's per-branch scheduling and tracing cost
is visible, especially under concurrent bursts.

---

## Double Fan-Out Results

`bench_double_fanout.py` measures two explicit fan-out phases:
`root -> [N phase 1] -> aggregate -> [N phase 2] -> sink`.

| Width per phase | Timbal trivial p50 | Agno trivial p50 | Timbal async p50 | Agno async p50 |
|----------------:|-------------------:|-----------------:|-----------------:|---------------:|
| 4 | 1.82 ms | 757.0 us | 4.16 ms | 3.29 ms |
| 8 | 2.69 ms | 1.02 ms | 5.26 ms | 3.47 ms |
| 16 | 4.99 ms | 1.17 ms | 5.89 ms | 3.61 ms |
| 32 | 7.49 ms | 1.68 ms | 8.06 ms | 3.99 ms |

| Width per phase | Timbal burst p50 | Agno burst p50 | Timbal memory/run | Agno memory/run |
|----------------:|-----------------:|---------------:|------------------:|----------------:|
| 4 | 121.5 ms | 53.2 ms | 1,033 B | 750 B |
| 8 | 301.4 ms | 64.3 ms | 1,661 B | 850 B |
| 16 | 524.4 ms | 101.5 ms | 2,845 B | 1,107 B |
| 32 | 1,119.0 ms | 209.1 ms | 5,244 B | 1,544 B |

This is the strongest workflow signal for Agno. It keeps latency relatively flat as the
number of parallel steps grows, while Timbal pays more for each first-class branch.

---

## Notes On The Comparison

**Agent loops are not workflow graphs.** Timbal is substantially better in the full
agent loop, even though Agno Workflow is leaner in standalone DAG microbenchmarks.

**Agno telemetry must be disabled for latency-sensitive agent use.** The mocked telemetry
column is close to no-telemetry, confirming the expensive part is the awaited network
call, not JSON serialization.

**Workflow is the Timbal improvement target.** The fan-out results show concrete areas to
optimize: task scheduling, branch result collection, event emission, and burst behavior
when thousands of tiny step tasks are alive at once.

**The right narrative is balanced.** Timbal is already excellent for agents. Agno's
Workflow runtime gives us a useful lower-overhead reference point for where Timbal
Workflow should go next.
