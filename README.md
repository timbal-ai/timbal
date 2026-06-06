# Timbal

[![PyPI](https://img.shields.io/pypi/v/timbal)](https://pypi.org/project/timbal/)
[![Python](https://img.shields.io/pypi/pyversions/timbal)](https://pypi.org/project/timbal/)
[![License](https://img.shields.io/github/license/timbal-ai/timbal)](LICENSE)

Simple, performant, battle-tested framework for building reliable AI applications.

Timbal gives you **Agents** (autonomous reasoning) and **Workflows** (explicit pipelines) behind one interface. No hidden magic: async functions, Pydantic validation, and event-driven streaming. If you know `async`/`await`, you already know how it works.

**Documentation:** [docs.timbal.ai](https://docs.timbal.ai)

---

## Quickstart

```bash
pip install timbal
```

```python
import asyncio

from timbal import Agent
from timbal.tools import WebSearch

agent = Agent(
    name="assistant",
    model="anthropic/claude-sonnet-4-6",
    tools=[WebSearch()],
    max_tokens=1024,
)

async def main():
    result = await agent(prompt="What's new in AI this week?").collect()
    print(result.output)

asyncio.run(main())
```

Set `ANTHROPIC_API_KEY` (or the key for your chosen provider) in a `.env` file or your environment.

Workflows use the same interface:

```python
import asyncio
import httpx

from timbal import Workflow
from timbal.state import get_run_context

async def fetch(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        return (await client.get(url)).text

workflow = (
    Workflow(name="scraper")
    .step(fetch)
    .step(
        lambda content: len(content),
        content=lambda: get_run_context().step_span("fetch").output,
    )
)

async def main():
    result = await workflow.collect(url="https://timbal.ai")
    print(result.output)

asyncio.run(main())
```

See the [Quickstart](https://docs.timbal.ai/quickstart) for the full app flow (`timbal create` → `timbal start`).

---

## Why Timbal

**The most performant agent framework.** In overhead benchmarks against LangGraph, CrewAI, the OpenAI Agents SDK, PydanticAI, and Agno (observability on both sides, faked LLMs), Timbal runs agent loops several times faster with a fraction of the memory. See [Benchmarks](#benchmarks).

**Small and hackable.** The core framework is under 10k lines. Easy to read, modify, and fork. Other frameworks are bloated with legacy, indirection, and abstraction.

**One interface.** Agents, Workflows, and Tools share the same calling convention and event stream. Compose them freely.

**Human in the loop.** [Approval gates](https://docs.timbal.ai/human-in-the-loop/approval-gates) and [`suspend()`](https://docs.timbal.ai/human-in-the-loop/suspend) pause any run for a human, persist state, and resume across process restarts. Works on agents, workflow steps, and tools.

**Provider-agnostic.** Swap models by changing a string (`anthropic/claude-sonnet-4-6` → `openai/gpt-5.5`). Built-in `FallbackModel` chains providers for automatic failover.

---

## Features

| | |
|---|---|
| [Memory & compaction](https://docs.timbal.ai/agents/memory) | Persistent context with strategies to stay under the context window |
| [Tools & MCP](https://docs.timbal.ai/agents/tools) | Built-in tool library, your own functions, any MCP server |
| [Structured output](https://docs.timbal.ai/agents/structured-output) | Typed Pydantic models instead of raw text |
| [Skills](https://docs.timbal.ai/agents/skills) | Reusable tool packages the agent loads on demand |
| [Tracing](https://docs.timbal.ai/core-concepts/tracing) | Full span traces, exportable over OTLP |
| [Evals](https://docs.timbal.ai/evals) | Declarative YAML evaluation suite with built-in validators |
| [Deployment](https://docs.timbal.ai/deployment) | Run locally with `timbal start`, ship to the platform or self-host |

Install extras as needed:

```bash
pip install 'timbal[server]'      # HTTP serving
pip install 'timbal[documents]'   # PDF, Excel, Word
pip install 'timbal[evals]'       # evals CLI
pip install 'timbal[all]'         # everything
```

---

## Benchmarks

Pure framework-overhead benchmarks: trivial handlers, faked LLM calls, observability on both sides.

| Metric (single tool call) | Timbal | LangGraph + LangSmith | CrewAI |
|---|---|---|---|
| p50 latency | **1.1 ms** | 5.2 ms | 3.2 ms |
| memory / run | **2.2 KB** | 110 KB | 10 KB |
| throughput @ c=10 | **1716/s** | 224/s | 31/s |

Reproduce with [`benchmarks/README.md`](benchmarks/README.md). Full suite covers LangGraph, CrewAI, Agno, PydanticAI, OpenAI Agents SDK, and Google ADK.

---

## Full app

The CLI scaffolds and runs a complete application (UI + API + workforce of Python agents/workflows):

```bash
timbal create my-project
cd my-project
timbal start
```

Deploy by connecting the repo to the [Timbal Platform](https://app.timbal.ai), or self-host the components yourself. See [deployment docs](https://docs.timbal.ai/deployment).

---

## Development

```bash
git clone https://github.com/timbal-ai/timbal.git
cd timbal
uv sync --dev
uv run pytest
```

Contributor reference: [`CLAUDE.md`](CLAUDE.md), [`benchmarks/README.md`](benchmarks/README.md).

---

## Documentation

[docs.timbal.ai](https://docs.timbal.ai)

## Contributing

Pull requests and issues welcome.

## License

Apache 2.0. See [LICENSE](LICENSE).
