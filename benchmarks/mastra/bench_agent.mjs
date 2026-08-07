/**
 * Mastra side of the agent-loop benchmark. Run by bench_agent.py — do not run
 * directly unless debugging (JSON goes to stdout, progress to stderr).
 *
 * Scenarios (identical to the Timbal side and to every other framework in
 * benchmarks/):
 *   1. Single tool call:    LLM → add → LLM → answer
 *   2. Multi-step (3 tools): LLM → add → LLM → mul → LLM → sub → LLM → answer
 *   3. Parallel tools:      LLM → [add, mul, neg] concurrent → LLM → answer
 *
 * The LLM is a MockLanguageModelV4 (AI SDK test util) that decides its next
 * response by counting tool results in the message history — no shared counter
 * state, safe for concurrent runs on a single shared agent.
 *
 * Two variants per scenario:
 *   - bare: plain `new Agent(...)` — no tracing (Mastra's default standalone path)
 *   - obs:  agent registered in a `Mastra` instance with `@mastra/observability`
 *           enabled and the exporter mocked (spans built, export dropped) —
 *           the equivalent of LangGraph+LangSmith-with-HTTP-mocked
 *
 * Usage: node --expose-gc bench_agent.mjs [--quick]
 */

import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { Observability, BaseExporter } from '@mastra/observability';
import { MockLanguageModelV4 } from 'ai/test';
import { z } from 'zod';

import { measure, meta, readParams } from './bench_lib.mjs';

const { quick } = readParams();

const N_ITERS = quick ? 20 : 100;
const N_WARMUP = quick ? 3 : 10;
const N_MEM = quick ? 20 : 100;
const N_BURST = { 1: 20, 2: 10, 3: 15 };
const THROUGHPUT_OPS = quick ? 30 : 200;
const CONCURRENCY_LEVELS = [1, 10, 50];

// ── Tools ────────────────────────────────────────────────────────────────────

const num = z.number();
const tools = {
  add: createTool({
    id: 'add',
    description: 'Add two numbers.',
    inputSchema: z.object({ a: num, b: num }),
    execute: async ({ a, b }) => a + b,
  }),
  multiply: createTool({
    id: 'multiply',
    description: 'Multiply two numbers.',
    inputSchema: z.object({ a: num, b: num }),
    execute: async ({ a, b }) => a * b,
  }),
  subtract: createTool({
    id: 'subtract',
    description: 'Subtract b from a.',
    inputSchema: z.object({ a: num, b: num }),
    execute: async ({ a, b }) => a - b,
  }),
  negate: createTool({
    id: 'negate',
    description: 'Negate a number.',
    inputSchema: z.object({ x: num }),
    execute: async ({ x }) => -x,
  }),
};

const SCENARIO_TOOLS = {
  1: { add: tools.add },
  2: { add: tools.add, multiply: tools.multiply, subtract: tools.subtract },
  3: { add: tools.add, multiply: tools.multiply, negate: tools.negate },
};

// ── Fake LLM (stateless — step derived from message history) ─────────────────

function countToolResults(prompt) {
  let n = 0;
  for (const msg of prompt) {
    if (msg.role !== 'tool') continue;
    for (const part of Array.isArray(msg.content) ? msg.content : []) {
      if (part.type === 'tool-result') n++;
    }
  }
  return n;
}

const toolCall = (id, name, input) => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: name,
  input: JSON.stringify(input),
});

const usage = { inputTokens: 1, outputTokens: 1, totalTokens: 2 };
const respond = (content, finishReason) => ({ finishReason, usage, content, warnings: [] });
const answer = (text) => respond([{ type: 'text', text }], 'stop');

function makeModel(scenario) {
  const handlers = {
    1: (step) =>
      step === 0
        ? respond([toolCall('c1', 'add', { a: 1, b: 2 })], 'tool-calls')
        : answer('3'),
    2: (step) => {
      if (step === 0) return respond([toolCall('c1', 'add', { a: 1, b: 2 })], 'tool-calls');
      if (step === 1) return respond([toolCall('c2', 'multiply', { a: 3, b: 4 })], 'tool-calls');
      if (step === 2) return respond([toolCall('c3', 'subtract', { a: 12, b: 3 })], 'tool-calls');
      return answer('9');
    },
    3: (step) =>
      step === 0
        ? respond(
            [
              toolCall('c1', 'add', { a: 1, b: 2 }),
              toolCall('c2', 'multiply', { a: 3, b: 4 }),
              toolCall('c3', 'negate', { x: 5 }),
            ],
            'tool-calls',
          )
        : answer('done'),
  };
  const handler = handlers[scenario];
  return new MockLanguageModelV4({
    doGenerate: async (options) => handler(countToolResults(options.prompt)),
  });
}

// ── Agent factories ──────────────────────────────────────────────────────────

function makeBareAgent(scenario) {
  return new Agent({
    id: 'bench_agent',
    name: 'bench_agent',
    instructions: 'You are a calculator. Use the tools.',
    model: makeModel(scenario),
    tools: SCENARIO_TOOLS[scenario],
  });
}

/** Export mocked at the sink: spans are built and processed, then dropped. */
class MockedExporter extends BaseExporter {
  name = 'mocked';
  async _exportTracingEvent(_event) {}
  async onLogEvent(_event) {}
  async onMetricEvent(_event) {}
}

function makeObsAgent(scenario) {
  const agent = new Agent({
    id: 'bench_agent_obs',
    name: 'bench_agent_obs',
    instructions: 'You are a calculator. Use the tools.',
    model: makeModel(scenario),
    tools: SCENARIO_TOOLS[scenario],
  });
  const observability = new Observability({
    configs: { bench: { serviceName: 'bench', exporters: [new MockedExporter()] } },
  });
  const mastra = new Mastra({
    agents: { bench_agent_obs: agent },
    observability,
    logger: false,
  });
  return { agent: mastra.getAgent('bench_agent_obs'), observability };
}

// ── Main ─────────────────────────────────────────────────────────────────────

const results = { meta: await meta(), params: { quick, N_ITERS, N_WARMUP, N_MEM, N_BURST, THROUGHPUT_OPS, CONCURRENCY_LEVELS }, scenarios: {} };

for (const scenario of [1, 2, 3]) {
  console.error(`  scenario ${scenario}`);
  const common = {
    iters: N_ITERS,
    warmup: N_WARMUP,
    memIters: N_MEM,
    burstN: N_BURST[scenario],
    throughputOps: THROUGHPUT_OPS,
    concurrencyLevels: CONCURRENCY_LEVELS,
  };

  const bare = makeBareAgent(scenario);
  const { agent: obs, observability } = makeObsAgent(scenario);

  results.scenarios[scenario] = {
    bare: await measure(() => bare.generate('go'), { ...common, label: 'mastra bare' }),
    obs: await measure(() => obs.generate('go'), {
      ...common,
      label: 'mastra+obs',
      drain: () => observability.flush(),
    }),
  };
}

console.log(JSON.stringify(results));
process.exit(0);
