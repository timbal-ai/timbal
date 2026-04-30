# Timbal vs OpenAI Agents SDK - Benchmarks

Pure framework overhead benchmarks. No real OpenAI API calls. Timbal uses `TestModel`;
OpenAI Agents SDK uses a custom offline `Model` implementation.

**Environment:** Apple Silicon M-series, Python 3.12, asyncio event loop.  
**All numbers below are from full-mode runs.** Raw output is stored in `results/`.

---

## The Short Version

Timbal is faster on steady-state agent-loop latency and uses less memory per run.
Against OpenAI Agents SDK with local tracing enabled, Timbal is **2.3-4.7x faster at p50**
across the three agent-loop scenarios.

OpenAI Agents SDK is relatively competitive under high concurrency, especially with
tracing disabled. In the parallel-tool throughput case at concurrency 10, it is basically
neck-and-neck with Timbal. But for the actual per-request agent loop cost, Timbal has the
clear advantage.

OpenAI Agents SDK does **not** expose a first-class workflow/DAG primitive comparable to
Timbal `Workflow`, LangGraph `StateGraph`, Agno `Workflow`, or Pydantic Graph. It has
agent orchestration primitives: `handoff(...)` and `Agent.as_tool(...)`. So the fair
follow-up benchmarks are orchestration benchmarks, not fake DAG workflow benchmarks.

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |
| `bench_handoff.py` | OpenAI true handoff vs closest Timbal delegation equivalent |
| `bench_agent_as_tool.py` | Supervisor -> worker-as-tool -> supervisor final answer |

## Dependency Note

OpenAI Agents SDK `0.14.6` requires `openai>=2.26,<3`. The repo's normal `uv run`
environment may resolve an older `openai` from the project lock, so run this benchmark
with explicit `uv --with` dependencies:

```bash
uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent.py
uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_handoff.py
uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent_as_tool.py
```

Quick mode:

```bash
uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent.py --quick
uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_handoff.py --quick
uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent_as_tool.py --quick
```

If you install `openai-agents` directly into the project environment and then see an
import error like `cannot import name 'ResponseToolSearchCall'`, the SDK is being loaded
with an incompatible `openai` package. Use the `uv run --with ...` command above so the
benchmark gets a compatible temporary environment without changing the repo's dependency
lock.

---

## Observability Fairness

The benchmark reports three columns:

- `Timbal`: built-in `InMemoryTracingProvider`, always on.
- `OAI (bare)`: OpenAI Agents SDK with `RunConfig(tracing_disabled=True)`.
- `OAI + tracing`: OpenAI Agents SDK with a local in-memory tracing processor, no export.

The fair comparison is `Timbal` vs `OAI + tracing`. `OAI (bare)` is included as a lower
bound for SDK runtime cost without observability.

---

## Agent Loop Results

Full loop: prompt -> fake LLM -> tool call(s) -> fake LLM -> answer.

### Latency

| Scenario | Timbal p50 | OAI bare p50 | OAI + tracing p50 | Timbal vs traced |
|----------|-----------:|-------------:|------------------:|-----------------:|
| Single tool | 747.8 us | 1.53 ms | 1.71 ms | 2.3x faster |
| 3-step chain | 716.0 us | 3.33 ms | 3.36 ms | 4.7x faster |
| Parallel tools | 771.2 us | 1.87 ms | 1.83 ms | 2.4x faster |

### Memory

| Scenario | Timbal memory/run | OAI bare memory/run | OAI + tracing memory/run | Timbal vs traced |
|----------|------------------:|--------------------:|-------------------------:|-----------------:|
| Single tool | 579 B | 1,204 B | 1,570 B | 2.7x less |
| 3-step chain | 1,009 B | 1,806 B | 2,137 B | 2.1x less |
| Parallel tools | 971 B | 1,528 B | 1,809 B | 1.9x less |

### Burst

| Scenario | Timbal burst p50 | OAI bare burst p50 | OAI + tracing burst p50 | Timbal vs traced |
|----------|-----------------:|-------------------:|-----------------------:|-----------------:|
| Single tool, 50 concurrent | 17.44 ms | 22.86 ms | 26.72 ms | 1.5x faster |
| 3-step chain, 30 concurrent | 29.09 ms | 32.72 ms | 34.04 ms | 1.2x faster |
| Parallel tools, 40 concurrent | 21.58 ms | 28.08 ms | 28.96 ms | 1.3x faster |

### Throughput

| Scenario | Timbal c=1 | OAI + tracing c=1 | Timbal c=10 | OAI + tracing c=10 | Timbal c=50 | OAI + tracing c=50 |
|----------|-----------:|------------------:|------------:|-------------------:|------------:|-------------------:|
| Single tool | 1,502/s | 573/s | 1,710/s | 1,357/s | 1,816/s | 1,650/s |
| 3-step chain | 647/s | 292/s | 760/s | 690/s | 812/s | 695/s |
| Parallel tools | 974/s | 515/s | 1,118/s | 1,078/s | 1,144/s | 932/s |

The concurrency story is more nuanced than the latency story. Timbal is clearly faster
at c=1, but OpenAI Agents narrows the gap at c=10/c=50, especially in the parallel-tools
scenario.

---

## Notes On The Comparison

**There is no OpenAI Agents workflow benchmark.** The SDK surface has handoffs,
agents-as-tools, guardrails, sessions/memory, tracing, realtime/voice, MCP/tools, and
sandboxing. It does not have a DAG/workflow runner. We should not invent one for the
benchmark suite.

**The SDK can be benchmarked offline.** A custom `Model` can return Responses API output
items directly, so this benchmark avoids network calls entirely.

**Tracing overhead is visible but not catastrophic.** OpenAI Agents local tracing adds
roughly 0.0-0.2 ms in these runs, plus some memory. Export is disabled.

**OpenAI Agents is strongest under concurrency.** Its c=10/c=50 throughput is close in the
parallel-tools scenario. Timbal still wins p50 latency and memory per run.

**The remaining useful OpenAI Agents work is limited.** We now cover agent loops, true
handoff, and agent-as-tool composition. More benchmarks should only be added for real SDK
primitives, not synthetic DAG shapes.

---

## Handoff Results

`bench_handoff.py` measures OpenAI Agents' real handoff primitive:
`triage agent -> handoff -> worker agent`.

Timbal does not have that exact primitive. The Timbal column is the closest composition
equivalent: `supervisor agent -> worker agent tool -> final answer`. This is not perfectly
structural parity, but it is the fair alternative to inventing a fake workflow benchmark.

| Metric | Timbal delegation | OpenAI handoff |
|--------|------------------:|---------------:|
| latency p50 | 730.6 us | 1.42 ms |
| memory/run | 749 B | 1,233 B |
| burst p50, 40 concurrent | 20.62 ms | 18.66 ms |
| throughput c=1 | 1,240/s | 741/s |
| throughput c=10 | 1,388/s | 1,340/s |
| throughput c=50 | 1,314/s | 1,596/s |

Interpretation: Timbal is faster for single-run latency and lighter per run. OpenAI's
true handoff primitive does well under high concurrency and wins c=50 throughput in this
run.

---

## Agent-As-Tool Results

`bench_agent_as_tool.py` measures the directly comparable composition:
`supervisor -> worker-as-tool -> supervisor final answer`.

| Metric | Timbal delegation | OpenAI `Agent.as_tool()` |
|--------|------------------:|-------------------------:|
| latency p50 | 726.3 us | 2.35 ms |
| memory/run | 749 B | 1,702 B |
| burst p50, 40 concurrent | 20.40 ms | 32.85 ms |
| throughput c=1 | 1,210/s | 440/s |
| throughput c=10 | 1,314/s | 964/s |
| throughput c=50 | 844/s | 870/s |

This is the cleaner apples-to-apples orchestration result. Timbal is **3.2x faster at p50**
and uses **2.3x less memory per run**. At very high concurrency, the throughput gap
mostly closes.
