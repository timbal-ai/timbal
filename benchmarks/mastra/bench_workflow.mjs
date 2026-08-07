/**
 * Mastra side of the workflow/DAG benchmark. Run by bench_workflow.py — do not
 * run directly unless debugging (JSON goes to stdout, progress to stderr).
 *
 * Scenarios (identical shapes to benchmarks/langchain/bench_workflow.py):
 *   1. Sequential:  A → B → C → D
 *   2. Fan-out/in:  A → [B, C, D] → E
 *   3. Diamond:     A → [B, C] → D
 *
 * Trivial handlers, no LLM — pure DAG scheduling overhead. Workflows are
 * built once (construction excluded); each run is createRun() + start(),
 * Mastra's normal programmatic execution path.
 *
 * Variants: bare (standalone workflow) and obs (registered in a Mastra
 * instance with @mastra/observability enabled, exporter mocked).
 *
 * Usage: node --expose-gc bench_workflow.mjs [--quick]
 */

import { Mastra } from '@mastra/core';
import { createStep, createWorkflow } from '@mastra/core/workflows';
import { Observability, BaseExporter } from '@mastra/observability';
import { z } from 'zod';

import { measure, meta, readParams } from './bench_lib.mjs';

const { quick } = readParams();

const N_ITERS = quick ? 50 : 200;
const N_WARMUP = quick ? 5 : 20;
const N_MEM = quick ? 50 : 200;
const N_BURST = quick ? 100 : 500;
const THROUGHPUT_OPS = quick ? 200 : 1000;
const CONCURRENCY_LEVELS = [1, 10, 50, 200];

// ── Steps ────────────────────────────────────────────────────────────────────

const io = z.object({ x: z.number() });

const step = (id, fn) =>
  createStep({
    id,
    inputSchema: io,
    outputSchema: io,
    execute: async ({ inputData }) => ({ x: fn(inputData.x) }),
  });

// ── Workflow factories (identical DAG shapes to the Timbal side) ─────────────

function sequential() {
  // A → B → C → D
  return createWorkflow({ id: 'sequential', inputSchema: io, outputSchema: io })
    .then(step('step_a', (x) => x + 1))
    .then(step('step_b', (x) => x * 2))
    .then(step('step_c', (x) => x + 10))
    .then(step('step_d', (x) => x - 3))
    .commit();
}

function fanout() {
  // A → [B, C, D] → E
  const combine = createStep({
    id: 'step_e',
    inputSchema: z.object({
      branch_b: io,
      branch_c: io,
      branch_d: io,
    }),
    outputSchema: io,
    execute: async ({ inputData }) => ({
      x: inputData.branch_b.x + inputData.branch_c.x + inputData.branch_d.x,
    }),
  });
  return createWorkflow({ id: 'fanout', inputSchema: io, outputSchema: io })
    .then(step('step_a', (x) => x + 1))
    .parallel([step('branch_b', (x) => x * 2), step('branch_c', (x) => x * 3), step('branch_d', (x) => x * 4)])
    .then(combine)
    .commit();
}

function diamond() {
  // A → [B, C] → D
  const combine = createStep({
    id: 'combine',
    inputSchema: z.object({ path_b: io, path_c: io }),
    outputSchema: io,
    execute: async ({ inputData }) => ({ x: inputData.path_b.x + inputData.path_c.x }),
  });
  return createWorkflow({ id: 'diamond', inputSchema: io, outputSchema: io })
    .then(step('step_a', (x) => x + 1))
    .parallel([step('path_b', (x) => x + 10), step('path_c', (x) => x * 5)])
    .then(combine)
    .commit();
}

const FACTORIES = { sequential, fanout, diamond };

// ── Observability variant ────────────────────────────────────────────────────

class MockedExporter extends BaseExporter {
  name = 'mocked';
  async _exportTracingEvent(_event) {}
  async onLogEvent(_event) {}
  async onMetricEvent(_event) {}
}

function withObservability(factory, name) {
  const wf = factory();
  const observability = new Observability({
    configs: { bench: { serviceName: 'bench', exporters: [new MockedExporter()] } },
  });
  const mastra = new Mastra({
    workflows: { [name]: wf },
    observability,
    logger: false,
  });
  return { wf: mastra.getWorkflow(name), observability };
}

// ── Main ─────────────────────────────────────────────────────────────────────

const runWorkflow = (wf) => async () => {
  const run = await wf.createRun();
  const result = await run.start({ inputData: { x: 3 } });
  if (result.status !== 'success') throw new Error(`workflow run failed: ${result.status}`);
};

const results = {
  meta: await meta(),
  params: { quick, N_ITERS, N_WARMUP, N_MEM, N_BURST, THROUGHPUT_OPS, CONCURRENCY_LEVELS },
  scenarios: {},
};

const common = {
  iters: N_ITERS,
  warmup: N_WARMUP,
  memIters: N_MEM,
  burstN: N_BURST,
  // The Timbal side pre-runs 10 concurrent executions before burst timing
  // (bench_workflow.py); keep the concurrent warmup identical.
  burstWarmup: 10,
  throughputOps: THROUGHPUT_OPS,
  concurrencyLevels: CONCURRENCY_LEVELS,
};

for (const [name, factory] of Object.entries(FACTORIES)) {
  console.error(`  scenario ${name}`);
  const bare = factory();
  const { wf: obs, observability } = withObservability(factory, name);
  results.scenarios[name] = {
    bare: await measure(runWorkflow(bare), { ...common, label: 'mastra bare' }),
    obs: await measure(runWorkflow(obs), {
      ...common,
      label: 'mastra+obs',
      drain: () => observability.flush(),
    }),
  };
}

console.log(JSON.stringify(results));
process.exit(0);
