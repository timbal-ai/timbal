# Platform HTTP pooling benchmark

This benchmark exercises `run_test → Agent → PlatformTracingProvider → _request`
against a local HTTPS server. It uses `TestModel`, a dummy token, and a temporary
trusted certificate; no provider calls or production trace writes are made.

The baseline disables only the runner HTTP scope. The pooled arm uses the scope.
Both arms synchronously persist both trace snapshots, and the benchmark asserts
that the final successful state has been saved before `run_test` returns. Each
run closes its client; connections are reused within a run, never across samples.
The two arms alternate order after one discarded warmup per arm.

## Recorded results

Raw samples and environment versions are in `results/`:

- Local TLS, 15 samples per arm: median **8.62 ms fresh → 6.18 ms pooled**;
  connections per run **2 → 1**, with **2 trace writes** in both arms.
- **Synthetic 50 ms delay per new connection**, 15 samples per arm:
  median **111.29 ms → 57.81 ms**; connections **2 → 1**, writes **2 → 2**.
- Server closes after every response, 5 samples per arm: median
  **8.89 ms → 8.31 ms**; both arms reconnect and use **2 connections / 2 writes**.

These establish connection reuse and persistence behavior, not production latency
savings. The injected delay represents connection setup cost only, not a full
network RTT model. Model execution, cold starts, and production infrastructure
are outside this benchmark.

## Reproduce

From the repository root, use a Python environment with the repository's Python
package and development dependencies installed, plus `openssl` on PATH:

```bash
PYTHONPATH=python python benchmarks/http_pooling/bench_trace_http.py --samples 15
PYTHONPATH=python python benchmarks/http_pooling/bench_trace_http.py --samples 15 --connection-delay-ms 50
PYTHONPATH=python python benchmarks/http_pooling/bench_trace_http.py --samples 5 --close-after-response
```

Add `--output /tmp/trace-http.json` to retain raw samples. Certificate verification
remains enabled; trust is configured only for the benchmark process.
