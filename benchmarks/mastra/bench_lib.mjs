/**
 * Shared measurement harness for the Mastra side of the Timbal vs Mastra
 * benchmarks. Mirrors the Python-side methodology:
 *
 *   - monotonic ns timer (process.hrtime.bigint ~ time.perf_counter)
 *   - warmup runs before every measured batch
 *   - forced GC between batches (requires --expose-gc)
 *   - latency: sequential runs, percentiles over per-run wall time
 *   - burst: N concurrent runs, per-run wall time percentiles
 *   - throughput: worker-pool bounded concurrency, ops/s
 *   - memory: peak V8 heapUsed growth over a batch / N runs
 *     (approximate — see NOTES.md; NOT comparable with Python tracemalloc)
 *
 * All progress goes to stderr; the caller prints JSON results to stdout.
 */

if (typeof globalThis.gc !== 'function') {
  console.error('error: run with node --expose-gc');
  process.exit(1);
}

const nowUs = () => process.hrtime.bigint();
const elapsedUs = (t0) => Number(process.hrtime.bigint() - t0) / 1e3;

export function pct(samples, p) {
  const sorted = [...samples].sort((a, b) => a - b);
  const idx = Math.min(Math.floor((sorted.length * p) / 100), sorted.length - 1);
  return sorted[idx];
}

export function summarize(samples) {
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  return {
    mean,
    p50: pct(samples, 50),
    p75: pct(samples, 75),
    p95: pct(samples, 95),
    p99: pct(samples, 99),
    max: Math.max(...samples),
  };
}

/**
 * `drain` is the Node analogue of the Python side's `_clear_traces()`: it is
 * awaited wherever the Timbal harness resets its in-memory tracing storage
 * (after warmup and between measured batches), so pending async observability
 * work (span export through the event bus) never bleeds into a timed phase.
 * Runners without observability pass no drain.
 */
const drainAndGc = async (drain) => {
  if (drain) await drain();
  globalThis.gc();
};

export async function latency(fn, iters, warmup, drain) {
  for (let i = 0; i < warmup; i++) await fn();
  await drainAndGc(drain);
  const samples = [];
  for (let i = 0; i < iters; i++) {
    const t0 = nowUs();
    await fn();
    samples.push(elapsedUs(t0));
  }
  await drainAndGc(drain);
  return samples;
}

export async function memory(fn, iters, warmup, drain) {
  for (let i = 0; i < warmup; i++) await fn();
  await drainAndGc(drain);
  const baseline = process.memoryUsage().heapUsed;
  let peak = baseline;
  for (let i = 0; i < iters; i++) {
    await fn();
    // Sample BEFORE draining: tracemalloc's peak is a continuous high-water
    // mark that includes this run's allocations up to the moment Timbal calls
    // _clear_traces(), so the heap must be read while buffered observability
    // work is still resident.
    const cur = process.memoryUsage().heapUsed;
    if (cur > peak) peak = cur;
    // Then drain, mirroring the per-run _clear_traces() on the Python side
    // (no GC — the Python side doesn't collect per run either).
    if (drain) await drain();
  }
  return { peak_growth_bytes: peak - baseline, per_run_bytes: (peak - baseline) / iters };
}

export async function burst(fn, n, warmup, drain) {
  await Promise.all(Array.from({ length: Math.min(warmup, n) }, () => fn()));
  await drainAndGc(drain);
  const samples = [];
  const t0 = nowUs();
  await Promise.all(
    Array.from({ length: n }, async () => {
      const s = nowUs();
      await fn();
      samples.push(elapsedUs(s));
    }),
  );
  const wallMs = elapsedUs(t0) / 1e3;
  await drainAndGc(drain);
  return { samples, wallMs };
}

export async function throughput(fn, ops, concurrency, drain) {
  await drainAndGc(drain);
  let next = 0;
  const t0 = nowUs();
  const workers = Array.from({ length: Math.min(concurrency, ops) }, async () => {
    while (next < ops) {
      next++;
      await fn();
    }
  });
  await Promise.all(workers);
  const elapsed = elapsedUs(t0);
  if (drain) await drain();
  return ops / (elapsed / 1e6);
}

/** Full measurement suite for one runner. Returns a JSON-friendly record. */
export async function measure(
  fn,
  { iters, warmup, memIters, burstN, burstWarmup = 5, throughputOps, concurrencyLevels, label, drain },
) {
  console.error(`    [${label}] latency ×${iters}…`);
  const lat = summarize(await latency(fn, iters, warmup, drain));

  console.error(`    [${label}] memory ×${memIters}…`);
  const mem = await memory(fn, memIters, warmup, drain);

  console.error(`    [${label}] burst ×${burstN}…`);
  const b = await burst(fn, burstN, burstWarmup, drain);

  const tp = {};
  for (const c of concurrencyLevels) {
    console.error(`    [${label}] throughput c=${c}…`);
    tp[c] = await throughput(fn, throughputOps, c, drain);
  }

  return {
    latency_us: lat,
    mem_per_run_bytes: mem.per_run_bytes,
    mem_peak_growth_bytes: mem.peak_growth_bytes,
    burst_us: summarize(b.samples),
    burst_wall_ms: b.wallMs,
    throughput_ops_s: tp,
  };
}

export function readParams() {
  const quick = process.argv.includes('--quick');
  return { quick };
}

export async function meta() {
  const { readFileSync } = await import('node:fs');
  const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8')).version;
  return {
    node: process.version,
    mastra_core: read('./node_modules/@mastra/core/package.json'),
    mastra_observability: read('./node_modules/@mastra/observability/package.json'),
    ai_sdk: read('./node_modules/ai/package.json'),
  };
}
