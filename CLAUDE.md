# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing
- **Install dependencies**: `uv sync --dev` (from repo root — `pyproject.toml` is at root)
- **Run all tests**: `uv run pytest` (from repo root)
- **Run single test**: `uv run pytest python/tests/core/test_file.py::TestClass::test_method`
- **Linting**: `uv run ruff check`
- **Format**: `uv run ruff format`
- **Fix lint**: `uv run ruff check --fix`
- **Line length**: 120 chars (configured in `pyproject.toml`)

### Benchmarks
- **Langchain benchmarks**: `cd benchmarks/langchain && uv pip install langchain-core langsmith langgraph && uv run pytest bench_*.py`
- **Quick mode** (faster, fewer iterations): default
- **Full mode**: set env `TIMBAL_BENCH_MODE=full`

---

## Repository Layout

```
timbal/
├── python/
│   ├── timbal/               # Main package
│   │   ├── __init__.py       # Top-level exports: Agent, Workflow
│   │   ├── core/
│   │   │   ├── runnable.py   # Base class for all executable components
│   │   │   ├── agent.py      # Agent execution engine
│   │   │   ├── workflow.py   # DAG workflow engine
│   │   │   ├── tool.py       # Tool wrapper
│   │   │   ├── llm_router.py # Multi-provider LLM dispatch
│   │   │   ├── models.py     # Model strings + context window lookup
│   │   │   └── test_model.py # Offline TestModel for testing
│   │   ├── state/
│   │   │   ├── __init__.py   # get_run_context, get_call_id, etc.
│   │   │   ├── context.py    # RunContext definition
│   │   │   └── tracing/
│   │   │       ├── providers/
│   │   │       │   ├── base.py       # TracingProvider ABC + Exporter ABC
│   │   │       │   ├── in_memory.py  # Default in-memory provider
│   │   │       │   ├── jsonl.py      # JSONL file provider
│   │   │       │   └── platform.py   # Timbal platform provider
│   │   │       └── exporters/
│   │   │           └── otel.py       # OTelExporter (fire-and-forget OTLP/HTTP)
│   │   ├── types/
│   │   │   ├── message.py    # Message with role + content list
│   │   │   ├── file.py       # File type with auto content detection
│   │   │   ├── content/      # TextContent, ToolUseContent, FileContent, etc.
│   │   │   └── events/       # StartEvent, DeltaEvent, OutputEvent
│   │   ├── collectors/       # Output processing; TimbalCollector is default
│   │   └── tools/            # Built-in tool library (Bash, Slack, Gmail, etc.)
│   └── tests/
│       └── core/             # Test files mirroring package structure
├── benchmarks/
│   ├── README.md             # General benchmark guide
│   └── langchain/            # Timbal vs LangChain/LangGraph benchmarks
├── planning/                 # In-progress feature plans (gitignored)
└── pyproject.toml            # Root package config + dev deps
```

---

## Core Primitives

### Agent

Autonomous execution unit. An LLM with tools that runs until it decides to stop.

```python
from timbal import Agent

agent = Agent(
    name="my_agent",            # required — used as path in traces
    model="anthropic/claude-sonnet-4-6",  # see Models section
    tools=[my_fn, AnotherTool()],         # functions or Runnable instances
    system_prompt="You are...",           # str or sync/async callable -> str
    max_iter=10,                          # max LLM↔tool iterations (default: 10)
    max_tokens=1024,                      # required for Anthropic models
    output_model=MyPydanticModel,         # structured output via Pydantic
    temperature=0.7,
    model_params={"thinking": {"type": "enabled", "budget_tokens": 2000}},
    tracing_provider=MyProvider,          # see Tracing section
    memory_compaction=compact_tool_results(),
)
```

**Key constructor params:**
- `model` — provider-prefixed string or `TestModel` instance
- `tools` — list of functions, dicts `{"name", "description", "handler"}`, or `Runnable`
- `system_prompt` — str, or a callable (sync/async) that returns str at runtime
- `output_model` — Pydantic model for structured output
- `max_iter` — max LLM→tool→LLM loops before forced stop
- `max_tokens` — required for Anthropic; sets max completion tokens
- `memory_compaction` — strategy or list of strategies; triggers at `memory_compaction_ratio` (default 0.75) of context window
- `tracing_provider` — `TracingProvider` subclass, `None` to disable, or `TRACING_UNSET` (default, auto-detects)
- `default_params` — fixed or callable defaults applied before user kwargs
- `pre_hook` / `post_hook` — parameterless callables; can call `get_run_context()`

---

### Workflow

Explicit DAG execution. Steps run concurrently; dependencies are auto-inferred or explicit.

```python
from timbal import Workflow
from timbal.state import get_run_context

workflow = (
    Workflow(name="my_workflow")
    .step(fetch_data)                           # auto-named "fetch_data"
    .step(                                       # explicit wiring
        process_data,
        data=lambda: get_run_context().step_span("fetch_data").output,
    )
    .step(
        save_result,
        when=lambda: get_run_context().step_span("process_data").output["ok"],
    )
)

result = await workflow.collect(url="https://...")
```

**`.step(runnable, depends_on=None, when=None, **kwargs)`**
- `runnable` — function, dict, or `Runnable`
- `depends_on` — explicit list of step names to wait for
- `when` — parameterless callable returning bool; step is skipped if False
- `**kwargs` — param overrides; can be plain values or callables for runtime resolution
- Returns `self` for chaining

**Dependency resolution (automatic):**
The framework inspects `when` and `**kwargs` callables for `step_span()` calls and automatically adds those steps as dependencies. No need to specify `depends_on` when using `get_run_context().step_span()`.

**Concurrent execution:** Independent steps run in parallel via asyncio. DAG cycle detection runs after each `.step()`.

---

### Tool

Wraps any callable as a Timbal Runnable. Usually you don't instantiate directly — pass functions to `Agent.tools` and they're wrapped automatically.

```python
from timbal.core.tool import Tool

tool = Tool(
    name="add_numbers",
    handler=lambda x, y: x + y,
    description="Add two integers",
    default_params={"y": 0},
)
```

Schema is auto-generated from type hints and docstrings for LLM consumption.

---

## Calling Runnables

All `Agent`, `Workflow`, and `Tool` instances share the same calling convention.

### `__call__(**kwargs)` → async generator of Events

```python
async for event in agent(prompt="Hello"):
    if isinstance(event, DeltaEvent):
        print(event.item.text_delta, end="")   # streaming token
    elif isinstance(event, OutputEvent):
        print(event.output)                     # final result
```

### `.collect(**kwargs)` → `OutputEvent`

Consumes all events and returns the final `OutputEvent`. Subsequent calls return the cached result.

```python
result = await agent.collect(prompt="Hello")
print(result.output)          # final output (str, dict, Pydantic model, etc.)
print(result.status.code)     # "success" | "error" | "cancelled"
print(result.usage)           # {"anthropic/claude-sonnet-4-6:input_tokens": 42, ...}
print(result.t0, result.t1)   # Unix ms timestamps
```

**Input params for Agent:**
- `prompt` — str or `Message`; converted to a user message
- `messages` — full `list[Message]`; bypasses memory resolution when provided

**Input params for Workflow:** whatever the first steps' unbound params are — they become the workflow's inputs.

---

## Event System

All events inherit `BaseEvent`:
```python
class BaseEvent(BaseModel):
    type: str           # "START" | "DELTA" | "OUTPUT"
    run_id: str
    parent_run_id: str | None
    path: str           # "agent_name" or "workflow.step_name"
    call_id: str
    parent_call_id: str | None
```

### `StartEvent` — fires when a runnable begins
No additional fields.

### `DeltaEvent` — streaming content
```python
event.item  # one of:
    TextDelta(id, text_delta)          # incremental LLM text
    Text(id, text)                      # complete text block
    ToolUse(id, name, input)            # tool call (input accumulates)
    ToolUseDelta(id, input_delta)       # incremental tool input
    Thinking(id, thinking)              # reasoning (Anthropic extended thinking)
    ThinkingDelta(id, thinking_delta)
    ContentBlockStop(id)                # block finished
    Custom(id, data)                    # custom content
```

### `OutputEvent` — final result
```python
class OutputEvent(BaseEvent):
    input: Any
    status: RunStatus        # .code: "success" | "error" | "cancelled"
    output: Any              # final return value
    error: dict | None       # {type, message, traceback} on failure
    t0: int                  # start time, Unix ms
    t1: int                  # end time, Unix ms
    usage: dict[str, int]    # token counts keyed by "{model}:{token_type}"
    metadata: dict[str, Any]
```

---

## Models

### Model strings

Provider-prefixed strings. Examples:
```
anthropic/claude-opus-4-7
anthropic/claude-opus-4-6
anthropic/claude-sonnet-4-6
anthropic/claude-haiku-4-5
openai/gpt-5.5
openai/gpt-5.5-2026-04-23
openai/gpt-4o
openai/gpt-4o-mini
openai/o3
openai/o1-mini
google/gemini-2.5-flash
google/gemini-2.5-pro-preview
groq/llama-3.3-70b-versatile
xai/grok-4
cerebras/llama-3.1-8b
sambanova/Meta-Llama-3.3-70B-Instruct
```

Full list: `python/timbal/core/models.py`. Context window lookup:
```python
from timbal.core.models import get_context_window
tokens = get_context_window("anthropic/claude-sonnet-4-6")  # int | None
```

### TestModel — offline testing

```python
from timbal.core.test_model import TestModel

# Cycle through fixed responses
model = TestModel(responses=["Hello!", "Goodbye."])

# Dynamic handler
model = TestModel(handler=lambda messages: f"Echo: {messages[-1].collect_text()}")

agent = Agent(name="test", model=model, tools=[])
result = await agent.collect(prompt="Hi")
assert result.output.collect_text() == "Hello!"
print(model.call_count)  # 1
```

Responses can be strings or `Message` objects (for tool-calling flows). Cycles to the last response when exhausted. No network calls.

---

## Tracing

### Providers

Providers persist and retrieve traces. Passed as a **class** (not instance) to `Agent`.

```python
from timbal.state.tracing.providers import (
    TracingProvider,      # ABC
    InMemoryTracingProvider,   # default
    JsonlTracingProvider,      # append to .jsonl file
    PlatformTracingProvider,   # Timbal platform
)
```

**`configured(**kwargs)`** — creates an isolated subclass with class-level attributes set. Original class is never mutated.

```python
provider = JsonlTracingProvider.configured(_path=Path("traces.jsonl"))
agent = Agent(model="...", tracing_provider=provider)
```

**Session chaining** — pass `parent_id` via `RunContext` to retrieve the parent run's trace in `get()`. Used for multi-turn memory across process restarts.

### Exporters

Write-only sinks attached to any provider. Fire after `_store()` completes.

```python
from timbal.state.tracing.exporters import OTelExporter

provider = JsonlTracingProvider.configured(
    _path=Path("traces.jsonl"),
    _exporters=[
        OTelExporter(
            endpoint="http://localhost:4318",
            service_name="my-agent",
            headers={"x-honeycomb-team": "YOUR_KEY"},
            retry_delays=(1.0, 2.0, 4.0),
        ),
    ],
)
```

**`OTelExporter`** — fire-and-forget OTLP HTTP/JSON. `export()` returns immediately after scheduling a background task. `close()` drains all in-flight tasks. Works as async context manager.

**Custom exporter:**
```python
from timbal.state.tracing.providers.base import Exporter

class MyExporter(Exporter):
    async def export(self, run_context) -> None:
        # run_context._trace contains all spans
        # run_context.id is the run ID
        ...
```

### Implementing a custom provider

```python
class MyProvider(TracingProvider):
    endpoint: str = ""

    @classmethod
    async def get(cls, run_context) -> Trace | None:
        # return parent run's Trace, or None
        ...

    @classmethod
    async def _store(cls, run_context) -> None:
        # persist run_context._trace keyed by run_context.id
        ...
```

---

## RunContext & Context Access

`RunContext` carries all execution state for a single run.

```python
from timbal.state import get_run_context, get_call_id, get_parent_call_id

ctx = get_run_context()   # RunContext | None
ctx.id                    # run ID (UUID7)
ctx.parent_id             # parent run ID for session chaining
ctx.platform_config       # PlatformConfig | None
ctx._trace                # Trace — all spans for this run
```

All context vars are concurrency-safe via `contextvars.ContextVar` — isolated per async task.

### `step_span(name, default=...)` — access step outputs in workflows

```python
# In workflow default params or when conditions:
output = get_run_context().step_span("fetch_data").output
```

Returns the `Span` for the named step. Raises `SpanNotFound` if missing and no default provided.

### `update_usage(key, value)`

```python
get_run_context().update_usage("my_api:calls", 1)
```

Propagates up the call stack. Usage accumulates in `OutputEvent.usage`.

---

## Types

### Message

```python
from timbal.types.message import Message
from timbal.types.content import TextContent, ToolUseContent, ToolResultContent, FileContent

msg = Message(
    role="user",
    content=[TextContent(text="Hello"), FileContent(file=my_file)],
)
msg.collect_text()   # concatenate all TextContent.text
msg.to_anthropic_input()
msg.to_openai_chat_completions_input()
```

### File

```python
from timbal.types.file import File

f = File.from_path("/path/to/doc.pdf")
f = File.from_url("https://...")
f = await File.from_upload(upload_obj)
```

Auto-detects MIME type. Serializes to base64 for LLM APIs.

### Content types
`TextContent`, `ToolUseContent`, `ToolResultContent`, `FileContent`, `ThinkingContent`, `CustomContent` — all in `timbal.types.content`.

---

## Built-in Tools

Large tool library in `python/timbal/tools/`. Import selectively:

```python
from timbal.tools import WebSearch, Bash, Edit, Write
from timbal.tools.slack import send_message
from timbal.tools.gmail import send_email
from timbal.tools.tavily import search
```

Full list: `python/timbal/tools/__init__.py`.

---

## Collectors

Collectors process event streams. The default (`TimbalCollector`) is used transparently via `.collect()`. You only interact with the collector system when building custom integrations.

```python
from timbal.collectors import BaseCollector, get_collector_registry

class MyCollector(BaseCollector):
    @classmethod
    def can_handle(cls, event): ...
    def process(self, event): ...
    def result(self): ...
```

Collectors are lazily loaded to avoid importing provider SDKs at module init.

---

## Key Patterns

### Structured output
```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]

agent = Agent(model="...", output_model=Summary)
result = await agent.collect(prompt="Summarise this...")
summary: Summary = result.output
```

### Streaming tokens
```python
from timbal.types.events import DeltaEvent
from timbal.types.events.delta import TextDelta

async for event in agent(prompt="Write a poem"):
    if isinstance(event, DeltaEvent) and isinstance(event.item, TextDelta):
        print(event.item.text_delta, end="", flush=True)
```

### Multi-step workflow with conditional skip
```python
workflow = (
    Workflow(name="pipeline")
    .step(validate_input)
    .step(
        process,
        when=lambda: get_run_context().step_span("validate_input").output["valid"],
        data=lambda: get_run_context().step_span("validate_input").output["data"],
    )
)
```

### Testing without API calls
```python
from timbal.core.test_model import TestModel

agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
result = await agent.collect(prompt="test")
assert result.status.code == "success"
```

### OTel observability
```python
from timbal.state.tracing.exporters import OTelExporter
from timbal.state.tracing.providers import JsonlTracingProvider
from pathlib import Path

async with OTelExporter(endpoint="http://localhost:4318") as exporter:
    provider = JsonlTracingProvider.configured(
        _path=Path("traces.jsonl"),
        _exporters=[exporter],
    )
    agent = Agent(model="...", tracing_provider=provider)
    result = await agent.collect(prompt="Hello")
# exporter.close() awaits all in-flight exports before exiting
```

---

## Testing Strategy

- Tests live in `python/tests/core/` mirroring the package structure
- All async tests use `pytest-asyncio` (mode=AUTO — no `@pytest.mark.asyncio` needed if configured)
- Use `TestModel` to avoid API calls in unit tests
- `tmp_path` pytest fixture for file-based provider tests
- Test classes group related tests: `TestProviderName`, `TestFeature`
