/**
 * LangGraph.js side of the runtime-calibration agent benchmark. Run by
 * bench_agent.py — do not run directly unless debugging (JSON on stdout,
 * progress on stderr).
 *
 * Purpose: LangGraph exists in both Python and TypeScript with the same
 * architecture (prebuilt react agent over a Pregel graph). Running the exact
 * scenarios of benchmarks/langchain/bench_agent.py on LangGraph.js isolates
 * the language-runtime factor (CPython vs V8) from framework design — the
 * variable no cross-framework, cross-language benchmark can separate on its
 * own.
 *
 * Scenarios (identical to every other framework in benchmarks/):
 *   1. Single tool call:    LLM → add → LLM → answer
 *   2. Multi-step (3 tools): LLM → add → LLM → mul → LLM → sub → LLM → answer
 *   3. Parallel tools:      LLM → [add, mul, neg] concurrent → LLM → answer
 *
 * The fake LLM subclasses BaseChatModel and picks its response by counting
 * ToolMessages in the history — stateless, mirroring the Python _FakeLLM.
 *
 * Usage: node --expose-gc bench_agent.mjs [--quick]
 */

import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { z } from 'zod';

import { measure, meta, readParams } from './bench_lib.mjs';

const { quick } = readParams();

const N_ITERS = quick ? 20 : 100;
const N_WARMUP = quick ? 3 : 10;
const N_MEM = quick ? 20 : 100;
const N_BURST = { 1: 20, 2: 10, 3: 15 };
const THROUGHPUT_OPS = quick ? 30 : 200;
const CONCURRENCY_LEVELS = [1, 10, 50];

// ── Fake tool-calling chat model (stateless — step from message history) ─────

class FakeToolLLM extends BaseChatModel {
  constructor(responses) {
    super({});
    this.responses = responses;
  }
  _llmType() {
    return 'fake-tool-llm';
  }
  bindTools() {
    return this;
  }
  async _generate(messages) {
    const step = messages.filter((m) => m instanceof ToolMessage || m.getType?.() === 'tool').length;
    return { generations: [{ message: this.responses[step % this.responses.length], text: '' }] };
  }
}

// ── Tools ────────────────────────────────────────────────────────────────────

const num = z.number();
const TOOLS = {
  add: tool(async ({ a, b }) => a + b, {
    name: 'add',
    description: 'Add two numbers.',
    schema: z.object({ a: num, b: num }),
  }),
  multiply: tool(async ({ a, b }) => a * b, {
    name: 'multiply',
    description: 'Multiply two numbers.',
    schema: z.object({ a: num, b: num }),
  }),
  subtract: tool(async ({ a, b }) => a - b, {
    name: 'subtract',
    description: 'Subtract b from a.',
    schema: z.object({ a: num, b: num }),
  }),
  negate: tool(async ({ x }) => -x, {
    name: 'negate',
    description: 'Negate a number.',
    schema: z.object({ x: num }),
  }),
};

const tc = (id, name, args) => ({ name, args, id, type: 'tool_call' });

function makeAgent(scenario) {
  const responses = {
    1: [
      new AIMessage({ content: '', tool_calls: [tc('c1', 'add', { a: 1, b: 2 })] }),
      new AIMessage({ content: '3' }),
    ],
    2: [
      new AIMessage({ content: '', tool_calls: [tc('c1', 'add', { a: 1, b: 2 })] }),
      new AIMessage({ content: '', tool_calls: [tc('c2', 'multiply', { a: 3, b: 4 })] }),
      new AIMessage({ content: '', tool_calls: [tc('c3', 'subtract', { a: 12, b: 3 })] }),
      new AIMessage({ content: '9' }),
    ],
    3: [
      new AIMessage({
        content: '',
        tool_calls: [tc('c1', 'add', { a: 1, b: 2 }), tc('c2', 'multiply', { a: 3, b: 4 }), tc('c3', 'negate', { x: 5 })],
      }),
      new AIMessage({ content: 'done' }),
    ],
  }[scenario];

  const tools = {
    1: [TOOLS.add],
    2: [TOOLS.add, TOOLS.multiply, TOOLS.subtract],
    3: [TOOLS.add, TOOLS.multiply, TOOLS.negate],
  }[scenario];

  return createReactAgent({ llm: new FakeToolLLM(responses), tools });
}

const INPUT = { messages: [{ role: 'user', content: 'go' }] };

// ── Main ─────────────────────────────────────────────────────────────────────

const results = {
  meta: await meta(),
  params: { quick, N_ITERS, N_WARMUP, N_MEM, N_BURST, THROUGHPUT_OPS, CONCURRENCY_LEVELS },
  scenarios: {},
};

for (const scenario of [1, 2, 3]) {
  console.error(`  scenario ${scenario}`);
  const agent = makeAgent(scenario);
  results.scenarios[scenario] = {
    bare: await measure(() => agent.invoke(INPUT), {
      iters: N_ITERS,
      warmup: N_WARMUP,
      memIters: N_MEM,
      burstN: N_BURST[scenario],
      throughputOps: THROUGHPUT_OPS,
      concurrencyLevels: CONCURRENCY_LEVELS,
      label: 'lg.js',
    }),
  };
}

console.log(JSON.stringify(results));
process.exit(0);
