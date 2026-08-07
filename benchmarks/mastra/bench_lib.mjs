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

export async function latency(fn, iters, warmup) {
  for (let i = 0; i < warmup; i++) await fn();
  globalThis.gc();
  const samples = [];
  for (let i = 0; i < iters; i++) {
    const t0 = nowUs();
    await fn();
    samples.push(elapsedUs(t0));
  }
  return samples;
}

export async function memory(fn, iters, warmup) {
  for (let i = 0; i < warmup; i++) await fn();
  globalThis.gc();
  const baseline = process.memoryUsage().heapUsed;
  let peak = baseline;
  for (let i = 0; i < iters; i++) {
    await fn();
    const cur = process.memoryUsage().heapUsed;
    if (cur > peak) peak = cur;
  }
  return { peak_growth_bytes: peak - baseline, per_run_bytes: (peak - baseline) / iters };
}

export async function burst(fn, n) {
  await Promise.all(Array.from({ length: Math.min(5, n) }, () => fn()));
  globalThis.gc();
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
  return { samples, wallMs };
}

export async function throughput(fn, ops, concurrency) {
  globalThis.gc();
  let next = 0;
  const t0 = nowUs();
  const workers = Array.from({ length: Math.min(concurrency, ops) }, async () => {
    while (next < ops) {
      next++;
      await fn();
    }
  });
  await Promise.all(workers);
  return ops / (elapsedUs(t0) / 1e6);
}

/** Full measurement suite for one runner. Returns a JSON-friendly record. */
export async function measure(fn, { iters, warmup, memIters, burstN, throughputOps, concurrencyLevels, label }) {
  console.error(`    [${label}] latency ×${iters}…`);
  const lat = summarize(await latency(fn, iters, warmup));

  console.error(`    [${label}] memory ×${memIters}…`);
  const mem = await memory(fn, memIters, warmup);

  console.error(`    [${label}] burst ×${burstN}…`);
  const b = await burst(fn, burstN);

  const tp = {};
  for (const c of concurrencyLevels) {
    console.error(`    [${label}] throughput c=${c}…`);
    tp[c] = await throughput(fn, throughputOps, c);
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
