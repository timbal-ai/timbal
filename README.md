# Timbal

> **Timbal 2.0 is in beta.** The API is stable — we're finalizing docs and tooling before the full release.

Simple, performant, battle-tested framework for building reliable AI applications.

Full documentation: [docs.timbal.ai](https://docs.timbal.ai)

---

## Installation

```bash
pip install timbal
```

Timbal is modular. The bare install includes the agent/workflow engine, both Anthropic and OpenAI providers, MCP support, and tracing. Install extras only when you need them:

| Extra | What it adds | When to use |
|---|---|---|
| `timbal[server]` | FastAPI + uvicorn | Serving agents over HTTP |
| `timbal[documents]` | PyMuPDF + openpyxl + python-docx | Reading PDFs, Excel, Word files |
| `timbal[evals]` | rich | Running the evals CLI |
| `timbal[codegen]` | libcst + ruff | Using the code generation tools |
| `timbal[all]` | Everything above | |

```bash
pip install 'timbal[server]'
pip install 'timbal[documents,evals]'
pip install 'timbal[all]'
```

### From source

```bash
git clone https://github.com/timbal-ai/timbal.git
cd timbal
uv sync --dev
```

---

## Two patterns, one interface

### Agent — autonomous reasoning

The LLM decides what to do. You provide tools and a goal.

```python
from timbal import Agent
from timbal.tools import WebSearch
from datetime import datetime

def get_datetime() -> str:
    return datetime.now().isoformat()

agent = Agent(
    name="assistant",
    model="anthropic/claude-sonnet-4-6",
    tools=[get_datetime, WebSearch()],
    max_tokens=1024,
)

result = await agent.collect(prompt="What time is it in Tokyo right now?")
print(result.output)
```

### Workflow — explicit pipelines

You define the steps. The framework handles concurrency and dependency resolution.

```python
import httpx
from timbal import Workflow
from timbal.state import get_run_context
from timbal.tools import Write

async def fetch(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        return (await client.get(url)).text

workflow = (
    Workflow(name="scraper")
    .step(fetch)
    .step(
        Write(),
        path="./output.html",
        content=lambda: get_run_context().step_span("fetch").output,
    )
)

await workflow.collect(url="https://timbal.ai")
```

Independent steps run concurrently. Dependencies are inferred automatically from `step_span()` references — no manual wiring needed.

---

## Calling runnables

All `Agent`, `Workflow`, and `Tool` instances share the same interface.

```python
# Collect all events, return final OutputEvent
result = await agent.collect(prompt="Hello")
print(result.output)           # final result
print(result.status.code)      # "success" | "error" | "cancelled"
print(result.usage)            # {"anthropic/claude-sonnet-4-6:input_tokens": 42, ...}

# Or stream events
async for event in agent(prompt="Hello"):
    if isinstance(event, DeltaEvent) and isinstance(event.item, TextDelta):
        print(event.item.text_delta, end="", flush=True)
```

---

## Models

Any provider, one interface. Model strings follow `provider/model-name`:

```
anthropic/claude-sonnet-4-6       openai/gpt-4o
anthropic/claude-opus-4-7         openai/gpt-5.5
anthropic/claude-opus-4-6         openai/gpt-5.5-2026-04-23
anthropic/claude-haiku-4-5        openai/o3
google/gemini-2.5-flash           togetherai/deepseek-ai/DeepSeek-V4-Pro
google/gemini-2.5-pro-preview     togetherai/moonshotai/Kimi-K2.6
groq/llama-3.3-70b-versatile      xai/grok-4
cerebras/llama-3.1-8b             sambanova/Meta-Llama-3.3-70B-Instruct
```

Recent additions (see `python/timbal/models.yaml`): DeepSeek V4 via Together/Fireworks, GLM-5.1, MiniMax M2.7, Kimi K2.6, Qwen 3.6 Plus (Fireworks), and the `gpt-5.5-2026-04-23` snapshot.

Full list and context window sizes in `python/timbal/core/models.py`.

---

## Structured output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    summary: str

agent = Agent(model="openai/gpt-4o-mini", output_model=Analysis)
result = await agent.collect(prompt="Analyse: 'Timbal makes AI easy'")
print(result.output.sentiment)     # "positive"
print(result.output.confidence)    # 0.97
```

---

## Streaming

```python
from timbal.types.events import DeltaEvent
from timbal.types.events.delta import TextDelta

async for event in agent(prompt="Write a short poem"):
    if isinstance(event, DeltaEvent) and isinstance(event.item, TextDelta):
        print(event.item.text_delta, end="", flush=True)
```

---

## Memory compaction

Long conversations grow large. Timbal has built-in strategies to keep context under control.

```python
from timbal.core.memory_compaction import (
    compact_tool_results,
    keep_last_n_messages,
    keep_last_n_turns,
    summarize,
)

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    max_tokens=2048,
    memory_compaction=[
        compact_tool_results(keep_last_n=2),   # compress old tool outputs
        keep_last_n_turns(10),                 # keep last 10 user↔assistant turns
    ],
    memory_compaction_ratio=0.75,              # trigger at 75% context window usage
)
```

**Strategies:**
- `compact_tool_results(keep_last_n, threshold, replacement)` — strips old tool results, optionally replacing with a summary string
- `keep_last_n_messages(n)` — hard truncation, structure-aware (no orphaned tool pairs)
- `keep_last_n_turns(n)` — keep last N user+assistant pairs
- `summarize(threshold, model, keep_last_n, max_summary_tokens)` — async LLM-based summarization of old messages

Strategies are applied in order. Multiple strategies can be combined.

---

## Skills

Skills are reusable, self-documenting tool packages. They sit on disk and are loaded into the agent's context only when the LLM explicitly requests them via the auto-injected `read_skill` tool.

```
skills/
└── web_research/
    ├── SKILL.md          # frontmatter + docs shown to the LLM
    └── tools/
        ├── search.py
        └── scrape.py
```

```yaml
# SKILL.md
---
name: "web_research"
description: "Search the web and scrape pages"
---
Use `search(query)` to find pages, then `scrape(url)` to get the content.
```

```python
agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    skills_path="./skills",
    max_tokens=2048,
)
```

The agent sees skill names and descriptions at startup. It calls `read_skill("web_research")` to load the tools and documentation when needed — keeping context clean until the skill is actually required.

---

## MCP servers

Connect agents to any [Model Context Protocol](https://modelcontextprotocol.io) server.

```python
from timbal.core import MCPServer

# Local server via stdio
mcp = MCPServer(
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "."],
)

# Remote server via HTTP
mcp = MCPServer(
    transport="http",
    url="https://my-mcp-server.com",
    headers={"Authorization": "Bearer token"},
)

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    tools=[mcp],
    max_tokens=2048,
)
```

---

## Conditional workflows

```python
workflow = (
    Workflow(name="pipeline")
    .step(validate_input)
    .step(
        process,
        when=lambda: get_run_context().step_span("validate_input").output["valid"],
        data=lambda: get_run_context().step_span("validate_input").output["data"],
    )
    .step(
        notify_failure,
        when=lambda: not get_run_context().step_span("validate_input").output["valid"],
    )
)
```

Steps with `when=` are skipped (not failed) when the condition is False. Downstream steps that depend on a skipped step are also skipped automatically.

---

## Observability

Timbal has a layered tracing system. Every run produces a full span trace.

```python
from timbal.state.tracing.providers import JsonlTracingProvider
from timbal.state.tracing.exporters import OTelExporter
from pathlib import Path

provider = JsonlTracingProvider.configured(
    _path=Path("traces.jsonl"),
    _exporters=[
        OTelExporter(
            endpoint="http://localhost:4318",
            service_name="my-agent",
            headers={"x-honeycomb-team": "YOUR_KEY"},
        ),
    ],
)

agent = Agent(model="...", tracing_provider=provider)
```

`OTelExporter` is fire-and-forget — it never adds latency to your runs. Compatible with Jaeger, Honeycomb, Datadog, Grafana Tempo, and any OTLP backend. Custom exporters:

```python
from timbal.state.tracing.providers.base import Exporter

class MyExporter(Exporter):
    async def export(self, run_context) -> None:
        spans = list(run_context._trace.values())
        await my_backend.send(spans)
```

---

## Session chaining

Link runs so an agent can recall what happened in a previous session — even across process restarts.

```python
from timbal.state.tracing.providers import JsonlTracingProvider
from pathlib import Path

provider = JsonlTracingProvider.configured(_path=Path("sessions.jsonl"))
agent = Agent(model="...", tracing_provider=provider)

run1 = await agent.collect(prompt="My name is Alice.")
print(run1.run_id)   # "abc123"

# Next session — agent remembers the previous one
from timbal.state.context import RunContext
ctx = RunContext(parent_id="abc123", tracing_provider=provider)
run2 = await agent.collect(prompt="What's my name?", run_context=ctx)
```

---

## HTTP serving

Requires `pip install 'timbal[server]'`. Serve any agent or workflow over HTTP with one command.

```bash
python -m timbal.server.http \
  --import_spec path/to/agent.py::my_agent \
  --host 0.0.0.0 \
  --port 4444 \
  --workers 4
```

| Endpoint | Method | Description |
|---|---|---|
| `/healthcheck` | GET | Returns 204 |
| `/params_model_schema` | GET | JSON schema of inputs |
| `/return_model_schema` | GET | JSON schema of output |
| `/run` | POST | Execute and wait |
| `/stream` | POST | Stream events as SSE |
| `/cancel/{run_id}` | POST | Cancel a running execution |

```bash
curl -X POST http://localhost:4444/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

---

## Evals

Declarative evaluation suite with built-in validators.

```yaml
# evals.yaml
evals:
  - name: "greets_user"
    params:
      prompt: "Say hello to Alice"
    agent!:
      output!:
        contains_all!: ["Hello", "Alice"]
      duration!:
        lt!: 5
```

```python
from timbal.evals.runner import run_eval

result = await run_eval(eval_config, agent=my_agent)
print(result.passed)
print(result.validator_results)
```

**Validators:** `contains`, `contains_all`, `contains_any`, `starts_with`, `ends_with`, `pattern`, `length`, `min_length`, `max_length`, `eq`, `lt`, `gt`, `type`, `email`, `json`, `semantic` (LLM-based), `language`. All support `not_` negation.

---

## Testing without API calls

```python
from timbal.core.test_model import TestModel

model = TestModel(responses=["The answer is 42."])
agent = Agent(name="test", model=model, tools=[])

result = await agent.collect(prompt="What is the answer?")
assert result.output.collect_text() == "The answer is 42."
assert result.status.code == "success"
```

Responses cycle to the last item when exhausted. Pass `Message` objects to test tool-calling flows. No network calls.

---

## Hooks

Pre/post hooks run around every execution and have access to the full run context.

```python
def log_pre():
    ctx = get_run_context()
    print(f"Starting run {ctx.id}")

def log_post():
    span = get_run_context().current_span()
    print(f"Output: {span.output}")

tool = Tool(handler=my_fn, pre_hook=log_pre, post_hook=log_post)
```

Hooks are parameterless callables. Both sync and async are supported.

---

## Running tests

```bash
uv run pytest
```

```bash
uv run pytest python/tests/core/test_jsonl_tracing_provider.py
uv run pytest python/tests/core/test_otel_exporter.py::TestRetry
```

---

## Benchmarks

```bash
cd benchmarks/langchain
uv pip install langchain-core langsmith langgraph

# Quick mode (default)
uv run pytest bench_*.py -v

# Full mode
TIMBAL_BENCH_MODE=full uv run pytest bench_*.py -v
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology and how to read results.

---

## Repository structure

```
timbal/
├── python/
│   ├── timbal/
│   │   ├── core/             # Agent, Workflow, Tool, LLM router, Skills, MCP
│   │   ├── state/            # RunContext, tracing providers + exporters
│   │   ├── types/            # Message, File, Events
│   │   ├── collectors/       # Output processing
│   │   ├── evals/            # Evaluation framework
│   │   ├── server/           # HTTP serving
│   │   ├── platform/         # Timbal platform integration
│   │   └── tools/            # Built-in tool library
│   └── tests/core/
├── benchmarks/
│   ├── README.md
│   └── langchain/
├── CLAUDE.md                 # Codebase guide for AI agents
└── pyproject.toml
```

---

## Why Timbal

**Transparent by default.** No hidden magic. Under the hood it's async functions, Pydantic validation, and event-driven streaming — nothing you couldn't build yourself, just already built well.

**Production-shaped.** The core abstractions were refined through real production deployments before the framework was open-sourced. Fast failure, clear error messages, stable interfaces.

**One interface for everything.** Agents, workflows, and tools all share the same `__call__` / `.collect()` convention and the same event stream. Compose them freely.

**Provider-agnostic.** Anthropic, OpenAI, Google, Groq, xAI, Cerebras, SambaNova — same code, swap the model string.

---

## Documentation

[docs.timbal.ai](https://docs.timbal.ai)

## Contributing

Pull requests and issues welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE).
