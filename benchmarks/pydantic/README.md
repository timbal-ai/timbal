# Timbal vs PydanticAI / Pydantic Graph — Benchmarks

Pure framework overhead benchmarks. No real LLM API calls — all model responses are
faked, and all handlers are intentionally tiny so the numbers isolate framework cost.

**Environment:** Apple Silicon M-series, Python 3.12, asyncio event loop.  
**All numbers below are from full-mode runs.** Raw output is stored in `results/`.

---

## The Short Version

Timbal is very strong where it matters most for this comparison: **full agent loops**.
Against PydanticAI with Logfire enabled, Timbal is **5.0–8.5× faster at p50** and
allocates **~6–7× less memory per run**.

Pydantic Graph is a different story. It is a very lean typed state-machine runner, not
a DAG scheduler. For graph/control-flow microbenchmarks, it is often faster because it
does less: fewer tasks, fewer events, fewer branch objects, and less per-step machinery.
That is real signal, not noise. Timbal has optimization work to do on small workflow
steps, skipped branches, and high-concurrency bursts.

That is the narrative:

- **Agents:** Timbal is already excellent.
- **DAG workflows:** Timbal gives richer semantics and per-step observability, but pays
  overhead for tiny synthetic steps.
- **Fan-out:** Pydantic Graph cannot natively express first-class DAG fan-out, so its
  fastest implementation uses manual `asyncio.gather` inside one node. That is not the
  same workload as Timbal's per-branch scheduling/tracing.
- **Control flow:** when we remove the `gather` escape hatch, Pydantic Graph is still
  leaner. Timbal's skipped-branch model is the biggest improvement target.

The goal from here is straightforward: keep the agent-loop advantage, then make Workflow
leaner until it is also best-in-class for branchy, tiny-step workloads.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |
| `bench_workflow.py` | Small workflow shapes: sequential, fan-out/in, diamond |
| `bench_parallel.py` | Wide fan-out: root → [N branches] → sink |
| `bench_double_fanout.py` | Double fan-out: root → [N×p1] → aggregator → [N×p2] → sink |
| `bench_control_flow.py` | Branch-heavy sequential control flow: repeated decision → branch → join rounds |
| `bench_linear_loop.py` | Linear sequential loop with no branches/skips/gather |

## Setup

```bash
uv sync --dev
uv pip install pydantic-ai logfire
```

`pydantic-ai` and `logfire` are not in `pyproject.toml`; they conflict with other
benchmark dependencies and should be installed ad-hoc.

## How To Run

```bash
# Quick mode
uv run python benchmarks/pydantic/bench_agent.py --quick
uv run python benchmarks/pydantic/bench_workflow.py --quick
uv run python benchmarks/pydantic/bench_parallel.py --quick
uv run python benchmarks/pydantic/bench_double_fanout.py --quick
uv run python benchmarks/pydantic/bench_control_flow.py --quick
uv run python benchmarks/pydantic/bench_linear_loop.py --quick

# Full mode
uv run python benchmarks/pydantic/bench_agent.py
uv run python benchmarks/pydantic/bench_workflow.py
uv run python benchmarks/pydantic/bench_parallel.py
uv run python benchmarks/pydantic/bench_double_fanout.py
uv run python benchmarks/pydantic/bench_control_flow.py
uv run python benchmarks/pydantic/bench_linear_loop.py
```

---

## Observability Fairness

Timbal includes tracing by default via `InMemoryTracingProvider`.

For PydanticAI agents, the fair comparison is **Timbal vs PAI+Logfire**:

- `PAI bare`: PydanticAI with no observability, shown only as a lower bound.
- `PAI+Logfire`: real `logfire.instrument_pydantic_ai()` with `send_to_logfire=False`.

For Pydantic Graph, the fair comparison is **Timbal vs PG+Logfire**:

- `PG bare`: `auto_instrument=False`, shown only as a lower bound.
- `PG+Logfire`: `auto_instrument=True` with `logfire.configure(send_to_logfire=False, console=False)`.

Network export is disabled on the Pydantic side, so these numbers measure local span
creation/instrumentation cost without HTTP variance.

Important caveat: **same observability provider does not always mean same observability
granularity.** In fan-out benchmarks, Timbal records each branch as a workflow step.
Pydantic Graph has no native DAG fan-out, so its fastest implementation records one
manual fan-out node and runs the branches inside that node.

---

## Agent Loop Results

This is the cleanest apples-to-apples benchmark: both frameworks run full agent loops
with fake LLMs and tool calls, and both include observability in the fair column.

| Scenario | Timbal p50 | PAI+Logfire p50 | Timbal advantage |
|----------|-----------:|----------------:|-----------------:|
| Single tool | 711.5 µs | 3.67 ms | 5.2× faster |
| 3-step chain | 783.7 µs | 6.70 ms | 8.5× faster |
| Parallel tools | 773.6 µs | 3.85 ms | 5.0× faster |

| Scenario | Timbal memory/run | PAI+Logfire memory/run | Timbal advantage |
|----------|------------------:|-----------------------:|-----------------:|
| Single tool | 598 B | 5,488 B | 9.2× less |
| 3-step chain | 1,045 B | 7,307 B | 7.0× less |
| Parallel tools | 933 B | 6,622 B | 7.1× less |

| Scenario | Timbal throughput c=10 | PAI+Logfire throughput c=10 | Timbal advantage |
|----------|-----------------------:|----------------------------:|-----------------:|
| Single tool | 1,649/s | 375/s | 4.4× higher |
| 3-step chain | 740/s | 245/s | 3.0× higher |
| Parallel tools | 1,108/s | 364/s | 3.0× higher |

This is the headline result. PydanticAI is well designed, but Timbal's agent runtime is
substantially faster and lighter once both sides have observability enabled.

---

## Small Workflow Results

`bench_workflow.py` compares small graph shapes. The sequential case is structurally
comparable. The fan-out and diamond cases are not: Timbal schedules/traces each branch,
while Pydantic Graph uses one manual branch node.

| Scenario | Timbal p50 | PG+Logfire p50 | Notes |
|----------|-----------:|---------------:|-------|
| Sequential `A → B → C → D` | 1.07 ms | 592.3 µs | Closest structural match |
| Fan-out/in | 1.18 ms | 465.2 µs | PG traces one manual branch node |
| Diamond | 1.02 ms | 484.9 µs | PG traces one manual branch node |

Takeaway: Pydantic Graph is thinner for small state-machine-style work. Timbal is still
around the low-millisecond range while preserving Workflow semantics and step-level
events/traces.

---

## Fan-Out Results

These benchmarks are useful, but they are not a direct DAG-scheduler comparison.
Pydantic Graph cannot express `N` first-class parallel graph branches. Its implementation
is one graph node containing `asyncio.gather(...)`.

### Wide Fan-Out

Async-work scenario: root → `N` branches, each sleeping 1 ms → sink.

| Width | Timbal p50 | PG+Logfire p50 | Timbal burst p50 | PG+Logfire burst p50 |
|------:|-----------:|---------------:|-----------------:|---------------------:|
| 4 | 2.19 ms | 2.01 ms | 166.4 ms | 56.8 ms |
| 8 | 2.49 ms | 2.11 ms | 245.4 ms | 62.5 ms |
| 16 | 2.80 ms | 2.13 ms | 517.2 ms | 70.3 ms |
| 32 | 4.43 ms | 2.05 ms | 1,078.5 ms | 126.8 ms |
| 64 | 8.09 ms | 2.22 ms | 2,255.7 ms | 328.7 ms |

### Double Fan-Out

Async-work scenario: root → `N` phase-1 branches → aggregator → `N` phase-2 branches → sink.

| Width | Timbal p50 | PG+Logfire p50 | Timbal burst p50 | PG+Logfire burst p50 |
|------:|-----------:|---------------:|-----------------:|---------------------:|
| 16 | 6.02 ms | 3.72 ms | 516.5 ms | 70.9 ms |
| 32 | 9.50 ms | 3.72 ms | 1,091.9 ms | 90.3 ms |
| 64 | 18.0 ms | 3.82 ms | 2,532.1 ms | 168.9 ms |
| 128 | 39.6 ms | 4.62 ms | 5,827.5 ms | 336.7 ms |

Takeaway: Timbal pays per branch because branches are real workflow steps. Pydantic Graph
pays for a small fixed number of graph nodes and leaves branch scheduling to user-owned
Python code. This is a valid implementation comparison, but not equivalent semantics.

---

## Control-Flow Results

These were added to remove the `asyncio.gather` escape hatch and understand where Timbal
can improve.

### Branchy Control Flow

`decision → left/right → join`, repeated for `N` rounds. No workload-level gather.
Timbal unrolls the bounded loop into explicit workflow steps; Pydantic Graph uses a
natural while-style graph loop.

| Rounds | Timbal p50 | PG+Logfire p50 | Timbal throughput c=10 | PG+Logfire throughput c=10 |
|-------:|-----------:|---------------:|-----------------------:|---------------------------:|
| 8 | 5.22 ms | 2.19 ms | 284/s | 455/s |
| 16 | 11.3 ms | 4.24 ms | 127/s | 243/s |
| 32 | 23.8 ms | 8.03 ms | 57/s | 123/s |
| 64 | 51.3 ms | 15.6 ms | 26/s | 59/s |

This is the clearest Workflow improvement target. The first branchy version used sync
handlers and was much worse; converting the Timbal side to async cut latency roughly in
half. Removing untaken branches cuts it further. That tells us the optimization path is
concrete: avoid threadpool paths for tiny callables and reduce skipped-step overhead.

### Linear Loop

No branch selection, no skipped steps, no gather. This isolates per-executed-step cost.

| Steps | Timbal p50 | PG+Logfire p50 | Timbal memory/run | PG+Logfire memory/run |
|------:|-----------:|---------------:|------------------:|----------------------:|
| 8 | 1.44 ms | 822.0 µs | 385 B | 779 B |
| 16 | 3.26 ms | 1.43 ms | 590 B | 778 B |
| 32 | 6.70 ms | 2.78 ms | 1,022 B | 765 B |
| 64 | 13.7 ms | 5.60 ms | 1,869 B | 761 B |

Once branches/skips are removed, the gap becomes much more reasonable: Timbal is roughly
2.4× slower than PG+Logfire on p50 latency for many tiny sequential steps, while memory
is competitive at small sizes and grows with stored per-step trace data.

---

## What We Should Improve

These benchmarks point to specific engineering work, not vague "performance tuning":

- **Fast path for tiny async workflow steps.** Avoid unnecessary event/dump/validation
  overhead when a step is a simple internal callable and full streaming semantics are not
  needed.
- **Skipped-branch overhead.** Current Workflow declares all steps upfront. Untaken
  branches still wait, evaluate `when`, mark skipped, and signal the queue. Branch-heavy
  workloads pay heavily for that.
- **Burst behavior.** High-concurrency fan-out creates many step tasks and queue events.
  We should profile task creation, queue signaling, context-var restoration, and trace
  persistence under burst.
- **Sync callable path.** Tiny sync functions pay threadpool/context-copy overhead. The
  benchmark now uses async handlers, but production users will write sync functions. We
  should either optimize this path or document that micro-step workflows should use async.
- **Graph-style control flow.** Pydantic Graph's while-loop model is genuinely lean. If
  Timbal wants to be best here too, Workflow needs a leaner control-flow primitive or a
  compiled execution plan for bounded loops/branches.

## Bottom Line

We are already very good where the product story is strongest: **agents with tools,
streaming events, structured traces, and observability always on**. Timbal beats
PydanticAI+Logfire by large margins there.

Pydantic Graph exposes real areas where we can get better. It is leaner for tiny
state-machine and branch-loop workloads, especially when Timbal pays for skipped
branches or large numbers of per-branch tasks. That is not a reason to soften the result.
It is a roadmap: keep the agent advantage, tighten Workflow's hot paths, and make Timbal
the best runtime across both agent loops and workflow/control-flow execution.
