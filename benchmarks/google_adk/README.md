# Timbal vs Google ADK - Benchmarks

Pure framework overhead benchmarks. No real Gemini/API calls. Timbal uses `TestModel`;
Google Agent Development Kit uses a custom offline `BaseLlm` implementation.

**Environment:** Apple Silicon M-series, Python 3.12, asyncio event loop.  
Raw output from full-mode runs should be stored in `results/`.

---

## The Short Version

Start with agents. ADK's public surface is agent-first: `Agent`, tools, `Runner`, sessions,
callbacks, sub-agents, and transfer controls. It does not expose a LangGraph-style DAG
runner comparable to Timbal `Workflow`, so we should not manufacture fake workflow
benchmarks just to fill the matrix.

The useful first benchmark is the same one used for LangChain, Agno, CrewAI, PydanticAI,
and OpenAI Agents:

- Single tool call: `LLM -> add -> LLM -> answer`
- Three-step chain: `LLM -> add -> LLM -> multiply -> LLM -> subtract -> LLM -> answer`
- Parallel tool calls: `LLM -> [add, multiply, negate] -> LLM -> answer`

After that, benchmark ADK-native orchestration:

- Agent transfer: `root agent -> transfer_to_agent(worker) -> worker final answer`
- Callback-enabled execution: no-op model/tool callbacks around a single-tool loop

---

## Files

| File | What it measures |
|------|-----------------|
| `bench_agent.py` | Full agent loop: fake LLM + tool calls, latency/memory/throughput |
| `bench_transfer.py` | ADK's real sub-agent transfer vs Timbal supervisor-worker delegation |
| `bench_callbacks.py` | No-op lifecycle callback overhead around model/tool execution |

## Dependency Note

Google ADK is not in the repo lock. Run with an ad-hoc dependency:

```bash
uv run --with google-adk python benchmarks/google_adk/bench_agent.py --quick
uv run --with google-adk python benchmarks/google_adk/bench_agent.py
uv run --with google-adk python benchmarks/google_adk/bench_transfer.py --quick
uv run --with google-adk python benchmarks/google_adk/bench_transfer.py
uv run --with google-adk python benchmarks/google_adk/bench_callbacks.py --quick
uv run --with google-adk python benchmarks/google_adk/bench_callbacks.py
```

No API keys required. The benchmark subclasses ADK's `BaseLlm` and returns deterministic
`LlmResponse` objects with `function_call` parts.

---

## Comparison Shape

The ADK column includes the real ADK `Runner` plus `InMemorySessionService`, because that
is the normal programmatic execution path. Sessions are pre-created outside the timed
region so the benchmark measures invocation/runtime overhead rather than session setup.

There is no separate "ADK tracing" column yet. ADK stores invocation events in the session
service and emits event objects from `Runner.run_async`; once we confirm the production
observability path teams are expected to use, add a second traced/exported column if it
has measurable hot-path cost.

There is no ADK workflow benchmark. Same rationale as OpenAI Agents SDK: ADK does not
expose a first-class DAG/workflow runner comparable to Timbal `Workflow`, LangGraph
`StateGraph`, Agno `Workflow`, or Pydantic Graph. A fake DAG built out of agents/tools
would benchmark a shape ADK is not designed around and would make the comparison less
honest, not more complete.

---

## Results

Raw outputs:

- `results/bench_agent.txt`
- `results/bench_transfer.txt`
- `results/bench_callbacks.txt`

### Agent Loop

Full loop: prompt -> fake LLM -> tool call(s) -> fake LLM -> answer.

| Scenario | Timbal p50 | ADK p50 | Timbal advantage |
|----------|-----------:|--------:|-----------------:|
| Single tool | 571.9 us | 1.02 ms | 1.8x faster |
| 3-step chain | 1.27 ms | 2.98 ms | 2.3x faster |
| Parallel tools | 806.4 us | 1.60 ms | 2.0x faster |

| Scenario | Timbal memory/run | ADK memory/run | Timbal advantage |
|----------|------------------:|---------------:|-----------------:|
| Single tool | 393 B | 17,681 B | 45.0x less |
| 3-step chain | 812 B | 37,253 B | 45.9x less |
| Parallel tools | 632 B | 23,814 B | 37.7x less |

| Scenario | Timbal throughput c=10 | ADK throughput c=10 | Timbal advantage |
|----------|-----------------------:|--------------------:|-----------------:|
| Single tool | 1,245/s | 1,052/s | 1.2x higher |
| 3-step chain | 667/s | 317/s | 2.1x higher |
| Parallel tools | 1,190/s | 529/s | 2.2x higher |

ADK is competitive under concurrency for the trivial single-tool case, but the gap widens
as tool-loop depth or parallel tool handling increases. The bigger signal is allocation:
ADK's runner/session/event path allocates tens of KB per run for these offline loops.

### Sub-Agent Transfer

`bench_transfer.py` measures ADK's real `transfer_to_agent` primitive:
`root agent -> transfer_to_agent(worker) -> worker final answer`.

Timbal does not have that exact primitive. The Timbal column is the closest composition
equivalent: `supervisor agent -> worker agent tool -> final answer`.

| Metric | Timbal delegation | ADK transfer |
|--------|------------------:|-------------:|
| latency p50 | 696.3 us | 977.1 us |
| memory/run | 781 B | 18,433 B |
| burst p50, 40 concurrent | 20.05 ms | 21.49 ms |
| throughput c=1 | 1,360/s | 968/s |
| throughput c=10 | 1,378/s | 1,028/s |
| throughput c=50 | 1,405/s | 1,037/s |

ADK's transfer primitive is reasonably close on burst latency, but still allocates ~24x
more per run and trails throughput by roughly 25-35%.

### Callback-Enabled Loop

`bench_callbacks.py` measures no-op lifecycle hooks/callbacks around the same single-tool
loop. Timbal uses Runnable `pre_hook`/`post_hook` on the agent and tool. ADK uses
before/after model callbacks and before/after tool callbacks.

| Metric | Timbal hooks | ADK callbacks |
|--------|-------------:|--------------:|
| latency p50 | 763.2 us | 1.01 ms |
| memory/run | 884 B | 18,245 B |
| burst p50, 40 concurrent | 22.53 ms | 22.85 ms |
| throughput c=1 | 1,247/s | 929/s |
| throughput c=10 | 1,521/s | 1,062/s |
| throughput c=50 | 1,541/s | 1,054/s |

Callback overhead itself is not catastrophic on either side. The latency gap is modest;
the memory gap is again the main difference.

---

## Next Useful ADK Benchmarks

- MCP/toolset benchmark only if ADK's toolset path adds meaningful runtime overhead beyond
  plain function tools.
- Observability/export benchmark once the expected production ADK tracing stack is clear.
