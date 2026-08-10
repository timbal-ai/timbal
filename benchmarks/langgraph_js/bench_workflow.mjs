/**
 * LangGraph.js side of the runtime-calibration DAG benchmark. Run by
 * bench_workflow.py — do not run directly unless debugging.
 *
 * StateGraph shapes identical to benchmarks/langchain/bench_workflow.py:
 *   1. Sequential:  A → B → C → D
 *   2. Fan-out/in:  A → [B, C, D] → E
 *   3. Diamond:     A → [B, C] → D
 *
 * Trivial handlers, no LLM — pure Pregel scheduling overhead on V8.
 *
 * Usage: node --expose-gc bench_workflow.mjs [--quick]
 */

import { Annotation, StateGraph, START, END } from '@langchain/langgraph';

import { measure, meta, readParams } from './bench_lib.mjs';

const { quick } = readParams();

const N_ITERS = quick ? 50 : 200;
const N_WARMUP = quick ? 5 : 20;
const N_MEM = quick ? 50 : 200;
const N_BURST = quick ? 100 : 500;
const THROUGHPUT_OPS = quick ? 200 : 1000;
const CONCURRENCY_LEVELS = [1, 10, 50, 200];

// ── Graph factories (identical shapes to benchmarks/langchain) ───────────────

function sequential() {
  const S = Annotation.Root({ x: Annotation(), a: Annotation(), b: Annotation(), c: Annotation(), d: Annotation() });
  return new StateGraph(S)
    .addNode('A', (s) => ({ a: s.x + 1 }))
    .addNode('B', (s) => ({ b: s.a * 2 }))
    .addNode('C', (s) => ({ c: s.b + 10 }))
    .addNode('D', (s) => ({ d: s.c - 3 }))
    .addEdge(START, 'A')
    .addEdge('A', 'B')
    .addEdge('B', 'C')
    .addEdge('C', 'D')
    .addEdge('D', END)
    .compile();
}

function fanout() {
  const S = Annotation.Root({
    x: Annotation(),
    a: Annotation(),
    bb: Annotation(),
    cc: Annotation(),
    dd: Annotation(),
    e: Annotation(),
  });
  return new StateGraph(S)
    .addNode('A', (s) => ({ a: s.x + 1 }))
    .addNode('B', (s) => ({ bb: s.a * 2 }))
    .addNode('C', (s) => ({ cc: s.a * 3 }))
    .addNode('D', (s) => ({ dd: s.a * 4 }))
    .addNode('E', (s) => ({ e: s.bb + s.cc + s.dd }))
    .addEdge(START, 'A')
    .addEdge('A', 'B')
    .addEdge('A', 'C')
    .addEdge('A', 'D')
    .addEdge('B', 'E')
    .addEdge('C', 'E')
    .addEdge('D', 'E')
    .addEdge('E', END)
    .compile();
}

function diamond() {
  const S = Annotation.Root({
    x: Annotation(),
    a: Annotation(),
    b: Annotation(),
    c: Annotation(),
    combined: Annotation(),
  });
  return new StateGraph(S)
    .addNode('A', (s) => ({ a: s.x + 1 }))
    .addNode('B', (s) => ({ b: s.a + 10 }))
    .addNode('C', (s) => ({ c: s.a * 5 }))
    .addNode('D', (s) => ({ combined: s.b + s.c }))
    .addEdge(START, 'A')
    .addEdge('A', 'B')
    .addEdge('A', 'C')
    .addEdge('B', 'D')
    .addEdge('C', 'D')
    .addEdge('D', END)
    .compile();
}

const FACTORIES = { sequential, fanout, diamond };

// ── Main ─────────────────────────────────────────────────────────────────────

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
  // Matches the 10 concurrent pre-runs on the Python side before burst timing.
  burstWarmup: 10,
  throughputOps: THROUGHPUT_OPS,
  concurrencyLevels: CONCURRENCY_LEVELS,
};

for (const [name, factory] of Object.entries(FACTORIES)) {
  console.error(`  scenario ${name}`);
  const graph = factory();
  results.scenarios[name] = {
    bare: await measure(() => graph.invoke({ x: 3 }), { ...common, label: 'lg.js' }),
  };
}

console.log(JSON.stringify(results));
process.exit(0);
