# Timbal vs LangGraph Python and JavaScript

The same synthetic agent and small-DAG workloads run through Timbal, LangGraph
Python and LangGraph.js. Models and tools are mocked; construction is excluded.
Timbal uses in-memory tracing, while both LangGraph columns are bare.

See the [2026-10-24 PR #222 report](../langchain/results/pr-222/README.md) for the
updated comparison, exact versions and raw outputs. Older `results/bench_*.txt`
files are historical and used a different context-isolation harness.

## Run

From the repository root:

```bash
uv sync --dev
uv pip install -r benchmarks/langchain/requirements.txt
npm ci --prefix benchmarks/langgraph_js
uv run --no-sync python benchmarks/langgraph_js/bench_agent.py
uv run --no-sync python benchmarks/langgraph_js/bench_workflow.py
```

Pass `--quick` for fewer samples, `--timbal-only` to skip competitors, or
`--lgjs-json <file>` to replay previously saved Node-side measurements.
Redirect stdout to save the combined tables. The `.mjs` entry points emit JSON.

`bench_agent.py` uses the legacy `create_react_agent` / `createReactAgent` APIs
for continuity. The Python benchmark in `../langchain/bench_agent.py` additionally
supports the current `langchain.agents.create_agent` via `--agent-api current`.

## Interpretation

Each Timbal sample starts with a fresh run context. Agent checks verify the
expected model calls, tool results and final answers before measurements begin.

Python/JavaScript ratios compare **two implementations on two runtimes**. They
do not isolate a pure language factor and should not be used to normalize
another framework's results. Memory instruments also differ (`tracemalloc`
versus V8 heap measurements); do not publish cross-language memory ratios.

These tests measure framework overhead, not end-to-end LLM application latency,
production persistence or feature equivalence.
