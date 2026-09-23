# Foreground timeout overhead

Compare the same streaming `Tool` with `timeout=None` and `timeout=120`.
The deadline never intentionally expires: this measures the steady-state cost
of enforcing it, including the relay and timer setup/cancellation per event.

```bash
uv run python benchmarks/timeout/bench_streaming.py --quick
uv run python benchmarks/timeout/bench_streaming.py --json /tmp/timeout-streaming.json
```

The default scenarios emit 1, 100, and 1,000 `TextDelta` chunks with no network
calls or sleeps in the handler. Unlike the agent benchmarks' `TestModel`, these
exercise token-like streaming rather than whole-message responses.

Both modes use `InMemoryTracingProvider` and include event construction,
collection, and trace storage. Tool construction, warmup, trace cleanup, and
an event-loop tick between runs are excluded from timing. Each run starts with
a fresh run context. The event-loop tick allows cancelled timers to be reclaimed
instead of accumulating them across a synthetic uninterrupted benchmark loop.

Full mode uses seven rounds of 200 runs per mode and stream length, alternating
mode order each round. The summary reports the median of round p50 latencies,
their percentage difference, and the difference divided by the chunk count.
Per-chunk overhead includes fixed per-run work; use longer streams to assess
its scaling. JSON output also retains each round's p50/p95 and environment.
Use `--chunks`, `--rounds`, and `--iterations` to override these defaults.

This benchmark compares the enabled and disabled feature on one revision. To
check regressions in the default path, run the same script against both commits
using the same Python environment. Small differences need repeated measurements;
these are framework overhead numbers, not predictions of end-to-end LLM latency.

An unset timeout has negligible overhead: a per-call check remains, but the
stream relay and timers are bypassed. A positive timeout adds work per streamed
event. There is deliberately no timing assertion or CI performance threshold.
