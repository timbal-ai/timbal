# PR #222 performance check — 2026-10-24

No material regression was detected in the tested existing execution paths. This is
an empirical result for the workloads below, not a guarantee for every deployment.
No production runtime code was changed during this benchmark work.

Compared implementation `4e3dc4a6cd1c7099a12b69fea4eb559dfb468c89` against merge base
`49af46e9724467d600c77ec466fa896e488ce895`, using the **same corrected harness and
Python environment** for both. Apple M1 Pro (10 logical CPUs, 16 GiB), macOS 26.5,
CPython 3.11.11 and Node 22.18.0. Full versions: [environment.json](environment.json).

Dependencies updated: LangGraph Python **1.2.17**, LangChain **1.4.7**,
langchain-core **1.7.0**, LangSmith **0.14.12**; LangGraph.js **1.5.2** and
@langchain/core **1.3.0**. Benchmark dependencies are pinned separately from Timbal's
runtime dependencies. The local development Python environment was not upgraded.

## Regression check

All five Python comparison scripts and both Python/JavaScript comparison scripts
completed in **full mode on both revisions**. Runs were serial, alternating which
revision ran first across scripts. See [execution log](full-run-log.txt).

An additional three fresh-process repeats per revision used 200 timed runs per
Timbal case after 10 warmups. The table shows the **median of the three process
medians**, not a confidence interval. Positive changes mean the PR was slower.
Raw samples and process ranges are in `base_repeat_*.json`, `pr_repeat_*.json` and
[summary.json](summary.json). Laptop scheduling and clock variation are visible;
large apparent improvements in a single group should not be attributed to this PR.

| Workload | Base p50 (ms) | PR p50 (ms) | Change |
|---|---:|---:|---:|
| Single-tool agent | 0.276 | 0.273 | -1.2% |
| Three-step agent | 0.591 | 0.593 | +0.4% |
| Parallel-tool agent | 0.572 | 0.542 | -5.3% |
| Sequential DAG | 0.471 | 0.368 | -22.0% |
| Fan-out/in DAG | 0.582 | 0.478 | -17.8% |
| Diamond DAG | 0.503 | 0.422 | -16.0% |
| Parallel branches: 4 | 0.564 | 0.533 | -5.3% |
| Parallel branches: 8 | 0.729 | 0.705 | -3.3% |
| Parallel branches: 16 | 1.142 | 1.053 | -7.8% |
| Parallel branches: 32 | 1.797 | 1.812 | +0.8% |
| Parallel branches: 64 | 3.261 | 3.375 | +3.5% |
| Double fan-out width: 16 | 1.928 | 1.936 | +0.4% |
| Double fan-out width: 32 | 3.379 | 3.358 | -0.6% |
| Double fan-out width: 64 | 6.836 | 6.761 | -1.1% |
| Double fan-out width: 128 | 13.846 | 14.502 | +4.7% |

The largest positive change in that sweep was +4.7% on the 259-step graph, with
overlapping process ranges. Four additional alternating repeats per revision
focused on noisy cases (200 runs each, no competitor runs between cases):

| Focused rerun | Base p50 (ms) | PR p50 (ms) | Change |
|---|---:|---:|---:|
| Parallel-tool agent | 0.498 | 0.509 | +2.11% |
| Sequential DAG | 0.358 | 0.359 | +0.31% |
| 64 parallel branches | 3.304 | 3.251 | -1.61% |
| 259-step double fan-out | 13.816 | 13.826 | +0.07% |

All focused base/PR process ranges overlap. The suspected wide-graph slowdown did
not reproduce consistently: the 259-step case is +0.07% in the focused rerun.
The earlier apparent ~22% sequential-DAG improvement also did not reproduce
(+0.31% here). Do not market this PR as a performance optimization.

## Current recommended agent API: the primary blog comparison

LangChain recommends [`create_agent` in place of `create_react_agent`](https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent).
Both APIs were checked for the expected tool-result count and final answer.
Timbal's checks also verify model-call counts across repeated and concurrent runs.
The following uses three PR-process repeats: 200 Timbal samples and 100 samples
for each LangGraph API per process, with 10 warmups. Values are median process p50s.
Timbal has in-memory tracing enabled; LangGraph has no tracer or checkpointer here.

| Mocked agent workload | Timbal p50 (ms) | Current `create_agent` (ms) | Current / Timbal | Legacy `create_react_agent` (ms) |
|---|---:|---:|---:|---:|
| Single-tool agent | 0.273 | 1.636 | 6.00× | 2.729 |
| Three-step agent | 0.593 | 3.511 | 5.92× | 6.424 |
| Parallel-tool agent | 0.542 | 2.632 | 4.86× | 3.301 |

A defensible headline is: **“On three mocked agent-loop workloads, Timbal had
about 4.9–6.0× lower median execution latency than LangChain's current
`create_agent` running on LangGraph.”** State the versions and machine with it.
This is framework/adapter overhead, **not** a 5–6× end-to-end LLM application speedup.

The existing full-mode graph comparison gives roughly 2.6–3.3× lower median
latency on small DAGs, 1.5–2.0× on 4–64 trivial parallel branches, and 1.7–2.1× on
35–259-step double-fan-out graphs versus bare LangGraph Python. Those are
single full-mode runs per revision, not the repeated-agent statistic above.

See [current API full output](pr_bench_agent_current.txt),
[legacy agent output](pr_bench_agent.txt), [small DAGs](pr_bench_workflow.txt),
[parallel branches](pr_bench_parallel.txt), [double fan-out](pr_bench_double_fanout.txt)
and [long conversations](pr_bench_long_conversation.txt).

## JavaScript comparison

The [agent](../../../langgraph_js/results/pr-222/pr_bench_agent.txt) and
[DAG](../../../langgraph_js/results/pr-222/pr_bench_workflow.txt) full-mode outputs
include updated LangGraph.js. In those single runs, Timbal's median agent latency
was 0.309/0.603/0.484 ms versus LangGraph.js 1.34/1.81/1.03 ms (about 2.1–4.3×).
These use legacy `createReactAgent`; they are not the current Python API comparison.
Ratios include implementation **and** runtime differences and cannot isolate a
pure Python-versus-JavaScript factor or normalize unrelated frameworks.

## Tracing and voice

The existing trace-persistence benchmark ran on both revisions with 8 model/tool
rounds, two 40 KiB tool results per round, zero mock model delay, and three runs
per durability mode. Synchronous median latency was **7.65 s base / 7.68 s PR**;
buffered latency was **0.98 s / 0.98 s**. Both revisions made 26 synchronous saves
(44.5 MB) or two buffered saves (4.3 MB), with no failures. This verifies the
simulated persistence path; it is not a live platform/network test.
[Base output](base_tracing.txt), [PR output](pr_tracing.txt).

The synthetic voice check used eight sessions for 20 seconds, explicit (`raw`)
turn commits every four seconds, half-second mock TTS replies, VAD and recording
off, and local STT/TTS/LLM mocks. Both revisions completed **40 measured turns**
with no errors and no late microphone frames. Base/PR loop-lag p99 was
**3.38 / 4.17 ms**, below the 20 ms budget; end-of-utterance to first-audio p95
was **55.6 / 35.6 ms**. Aggregate CPU was 0.035 / 0.042 cores. This short smoke
check is not precise enough to claim CPU equivalence, a speedup, production
capacity, or performance with VAD/real providers. [Base](base_voice_raw.json),
[PR](pr_voice_raw.json).

The first lexical/no-VAD configuration produced zero measured turns; those
`*_voice_idle.*` outputs are retained but **excluded** from the turn-performance
comparison. The explicit-commit rerun above exercised actual responses.

Two p95 increases in the broad sweep were also rerun in four fresh processes per
revision: the three-step agent's p95 was 0.758 / 0.711 ms, and the 32-branch graph's
was 2.201 / 2.211 ms. The focused 259-step p95 was 48.722 / 49.161 ms. No consistent
large tail-latency regression emerged. Full ranges are in `summary.json`.

## Harness corrections and limits

- Each independent Timbal sample now clears/restores the ambient run context.
  Previously repeated calls could inherit history and skip tool work. Explicit
  `parent_id` links remain in the conversation benchmark.
- The offline LangSmith tracer overrides both persistence methods **on a subclass**.
  LangChain clones tracers; the old instance-level mock could be lost, and it did
  not disable update uploads. The new column measures local callbacks/RunTree
  bookkeeping, excluding upload serialization, batching, network and durability.
  It is not an observability-equivalence comparison and is secondary to the bare
  LangGraph numbers used above.
- The agent/DAG output previously called maximum invocation latency “wall”. That
  label is corrected: max invocation latency is not total burst elapsed time.
  Throughput is timed around the complete batch. ANSI formatting was stripped
  from saved output; labels were clarified without changing recorded values.
- Older “memory per run” columns divide peak memory by iteration count and are
  **not per-request allocation figures**. Do not use them for blog memory ratios.
  The double-fan-out benchmark separately records actual per-run/burst peaks.
  Python and V8 memory instruments are not directly comparable.
- These are mocked workloads, with different framework-specific model adapters,
  reused agent/graph instances, synchronous trivial agent tools, and no network.
  Compilation/construction is excluded. Framework ordering within a script is
  fixed. The machine was a normal laptop, not a pinned-CPU benchmark host.
- The existing paths tested here do not exhaustively benchmark event storms,
  thousands of owned sessions, real provider streaming, or new continuation
  features under production load. No universal zero-overhead claim is justified.

## Reproduce

From the repo root, use CPython 3.11.11 and Node 22.18.0. The installed versions
used here are recorded in `environment.json`; use the **same environment** for
both source revisions. Disable platform credentials/tracing configuration so
Timbal selects the in-memory provider (the repeat harness asserts traces exist).
No API keys are needed. Use `--no-sync` after installing the benchmark-only packages.

```bash
uv sync --dev --python 3.11.11
uv pip install -r benchmarks/langchain/requirements.txt
npm ci --prefix benchmarks/langgraph_js

BENCH_REPO="$PWD"
BENCH_PYTHON="$PWD/.venv/bin/python"
mkdir -p /tmp/timbal-pr222-base /tmp/timbal-pr222-results
git archive 49af46e9724467d600c77ec466fa896e488ce895 python | tar -x -C /tmp/timbal-pr222-base
cd /tmp

# Repeat sweep: alternate source order; use the current corrected harness for both.
for round in 1 2 3; do
  order="base pr"
  if [ "$round" = 2 ]; then order="pr base"; fi
  for label in $order; do
    source_dir="$BENCH_REPO/python"
    if [ "$label" = base ]; then source_dir=/tmp/timbal-pr222-base/python; fi
    PYTHONPATH="$source_dir" PYTHONHASHSEED=0 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
      "$BENCH_PYTHON" "$BENCH_REPO/benchmarks/langchain/bench_revision.py" \
      > "/tmp/timbal-pr222-results/${label}_repeat_${round}.json"
  done
done
"$BENCH_PYTHON" "$BENCH_REPO/benchmarks/langchain/summarize_revision.py" /tmp/timbal-pr222-results
```

For the full original outputs, run each of `bench_agent.py`, `bench_workflow.py`,
`bench_parallel.py`, `bench_double_fanout.py`, `bench_long_conversation.py` in
`benchmarks/langchain/`, and `bench_agent.py` / `bench_workflow.py` in
`benchmarks/langgraph_js/`, without `--quick`. Use the same `PYTHONPATH` switch
above for each revision. Add `--agent-api current` to the Python agent benchmark
to reproduce the current recommended API's full table.

Focused repeats use `bench_revision.py --skip-competitors --case agent_3
--case dag_sequential --case parallel_64 --case double_128`, four alternating
repeats per source saved as `{base,pr}_targeted_{1,2,3,4}.json`. The p95 follow-up
uses `--case agent_2 --case parallel_32`, saved as `{base,pr}_tail_{1,2,3,4}.json`.
The summary script includes those optional groups when present.

Additional checks (again run with each source's `PYTHONPATH`):

```bash
"$BENCH_PYTHON" "$BENCH_REPO/benchmarks/trace_persistence/bench_trace_persistence.py" \
  --rounds 8 --payload-kb 40 --model-ms 0 --runs 3
"$BENCH_PYTHON" "$BENCH_REPO/benchmarks/voice/bench_cpu.py" \
  --sessions 8 --detector raw --no-vad --duration 20 --warmup 3 --settle 3 \
  --turn-period 4 --reply-secs 0.5 --json /tmp/voice-result.json
```

Keep older benchmark outputs separate: comparing them with these results would
mix different context isolation, tracer mocks, package versions and machines.
