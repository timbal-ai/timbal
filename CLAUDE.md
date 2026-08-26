# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing
- **Install dependencies**: `uv sync --dev` (from repo root — `pyproject.toml` is at root)
- **Run all tests**: `uv run pytest` (from repo root)
- **Run single test**: `uv run pytest python/tests/core/test_agent.py::TestClass::test_method`
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
│   │   │   ├── tool_set.py   # ToolSet ABC for runtime tool resolution
│   │   │   ├── mcp.py        # MCPServer — MCP servers as tool sources
│   │   │   ├── llm/          # Multi-provider LLM dispatch (router, registry, clients, retry + one module per API)
│   │   │   ├── models.py     # Model strings + context window lookup
│   │   │   └── test_model.py # Offline TestModel for testing
│   │   ├── state/
│   │   │   ├── __init__.py   # get_run_context, get_call_id, etc.
│   │   │   ├── context.py    # RunContext definition
│   │   │   ├── background.py # Session-scoped background-task store
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

result = await workflow(url="https://...").collect()
```

**`.step(runnable, depends_on=None, when=None, while_=None, **kwargs)`**
- `runnable` — function, dict, or `Runnable`
- `depends_on` — explicit list of step names to wait for
- `when` — parameterless callable returning bool; step is skipped if False
- `while_` — int (>= 1, run exactly N times) or parameterless callable evaluated after each iteration (do-while: always runs at least once). Each iteration gets its own span; `step_span()` returns the latest. Params resolve ONCE before iteration 1 — the step owns its cursor state. No built-in cap for callable conditions. Pausing mid-loop (approval/suspend) restarts the loop from iteration 1 on resume, and one approval resolution covers every iteration (approval id derives from the fixed input).
- `**kwargs` — param overrides; can be plain values or callables for runtime resolution
- Returns `self` for chaining

**Dependency resolution (automatic):**
The framework inspects `when`/`while_` and `**kwargs` callables for `step_span()` calls and automatically adds those steps as dependencies (a `while_` self-reference is ignored — not a cycle). No need to specify `depends_on` when using `get_run_context().step_span()`.

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

### MCPServer

Connects any MCP server as a tool source. Tools are resolved at runtime (it's a `ToolSet`) and exposed to the LLM with the server-declared JSON schemas.

```python
from timbal.core import MCPServer

agent = Agent(
    name="my_agent",
    model="...",
    tools=[
        MCPServer(transport="stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."]),
        MCPServer(name="timbal", transport="http", url="https://api.timbal.ai/mcp", headers={"Authorization": "Bearer ..."}),
    ],
)
```

- `transport` — `"stdio"` (spawns `command` + `args` with optional `env`) or `"http"` (streamable HTTP at `url` with optional `headers`)
- `name` — optional identifier; when set, tools are exposed as `{name}__{tool}` (bare name still used for `call_tool`). Required for codegen (`remove-tool --name`) and whenever multiple servers might share tool names
- Connections are lazy; the tool list is cached until `await server.close()`
- Results: text → str, images/audio/blobs → `File`s in a `Message`, `structuredContent` fallback, `isError` → raised so the LLM sees an error tool result
- Codegen: `python -m timbal.codegen add-mcp --name x --url ... --headers '{"Authorization": "Bearer $API_KEY"}'` ($VAR placeholders become `os.environ` lookups; also supports `--command/--args/--env` and `--from-json` with standard `mcpServers` configs)

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
anthropic/claude-fable-5
anthropic/claude-opus-5
anthropic/claude-sonnet-5
anthropic/claude-opus-4-8
anthropic/claude-opus-4-7
anthropic/claude-opus-4-6
anthropic/claude-sonnet-4-6
anthropic/claude-haiku-4-5
openai/gpt-5.5
openai/gpt-5.5-2026-04-23
openai/gpt-4o
openai/gpt-4o-mini
openai/o3
openai/gpt-5.4-nano
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

## Background Tasks

`background_mode="auto"|"always"` detaches a child (`python/timbal/state/background.py`). The spawn returns `{"task_id", "status": "running"}` immediately; the child's events go to an append-only log, not the parent's stream.

```python
from timbal.state import (
 cancel_background_task,
 get_background_task, # peek a bounded summary — does NOT drain
 list_background_tasks,
 read_background_transcript, # raw events from a cursor: (task_id, after=)
)

Tool(name="build", handler=..., background_mode="always",
 on_background_cancel=lambda record: remote.stop(record.metadata["run_id"]))

# A foreground async-generator tool with background_mode="auto" can be parked
# cooperatively while its collector keeps being consumed:
run = agent(prompt="start the build")
handoff = run.background()  # or run.background(call_id=...) if several auto children
async for event in run:
 ...
task = await handoff  # {"task_id": "...", "status": "running"}
```

- `collector.background()` parks one in-flight streaming runnable with `background_mode="auto"` at its next event. Foreground events are not copied into the ring. With one eligible child, omit the argument; with several, pass `call_id=` (from that child's `START`/`DELTA`) or it fails fast as ambiguous. Continue consuming the collector while the returned future is pending.
- Tasks live on a `BackgroundTaskStore` bound to the `RunContext`, inherited across sequential turns via `parent_id`. Concurrent runs of the same Agent get isolated stores (no shared `parent_id`), so a foreign `task_id` is `not_found`. Process-local — does not survive a restart.
- Once *this session* has a task, the agent auto-gains `get_background_task` / `list_background_tasks` / `cancel_background_task`. Opt in to `read_background_transcript` with `background_transcript_tool=True`.
- The log is peekable, not consume-once: the parent agent and a frontend can both watch one child. `record.log.subscribe(after=)` replays then streams live.
- `get_background_task` returns `{status, task_id, name, title, input, started_at, summary: {text, phase, pct, last_tool, tools_in_flight, event_count}, transcript_cursor}` (+ `result`/`error` when done). `summary.text` is capped — for briefing, not for dumping a build into context.
- `background_timeout` → store enforces wall-clock deadline; status `timed_out`. `background_stall_timeout` → no log events within window; status `stalled` (timer resets on every event).
- Cancel stops in-flight work: the Task is cancelled *and* the handler generator is closed (an async gen suspended at a yield would otherwise never run its `finally`), then `on_background_cancel` fires for work the loop can't reach.
- `wait_for_background(task_id, timeout=..., after=...)` blocks until terminal (no `after`) or until the log advances past `after` — app/frontends; does not ack completion notices.

---

## Tool Result Offloading & Memory Compaction

Two layers keep the context window bounded (`python/timbal/core/tool_result_offload.py` and `python/timbal/core/memory_compaction.py`):

### Production-time offload (size-triggered, per result)

```python
from timbal.core import LocalOffloadStore, Spill, ToolResultLimit, Truncate

agent = Agent(
 model="...",
 tool_result_limit=ToolResultLimit( # or an int shorthand for the threshold
 threshold=20_000, # chars of text content
 action=Spill(preview_chars=1_000), # or Truncate(strategy="head"|"tail"|"head_tail")
 store=LocalOffloadStore(), # default; keep-forever, opt-in cleanup_after=timedelta
 ),
 tools=[
 Tool(name="logs", handler=..., result_limit=ToolResultLimit(threshold=8_000, action=Truncate(strategy="tail"))),
 Tool(name="docs", handler=..., result_limit=None), # exempt
 ],
)
```

- Oversized results are reduced **once, when produced** — before entering memory/dumps — so history stays append-only (prompt-cache friendly) and the reduction persists into traces.
- `Spill` is lossless: payload goes to the store, a preview + handle stays inline, and a bounded `read_tool_result(handle, offset, limit, pattern)` tool is auto-registered for paged read-back. Falls back to `Truncate` when the store fails.
- Always exempt: error results, pinned tools (`pin_result=True`), `read_tool_result` itself. Precedence: `Tool.result_limit` > agent `tool_result_limit`.
- Offload events are recorded in `span.metadata["offload"]`; the handle lives on `ToolResultContent.offload_handle`.

### History compaction (utilization-triggered, whole memory)

`memory_compaction=` strategies fire at `memory_compaction_ratio` (default 0.75) of the context window: `compact_tool_results(keep_last_n, replacement, keep_offloaded=True)`, `keep_last_n_messages(n)`, `keep_last_n_turns(n)`, `summarize(...)`.

`summarize()` builds a sectioned summary message where everything except the LLM summary is mechanical: user messages carried **verbatim** (`preserve_user_messages=True`), a **canonical record** of each summarized region written to the offload store and readable via `read_tool_result` (`canonical_record=True`; store shared automatically from `tool_result_limit`), a conservative continuation note, and an optional `rehydrate=` callable re-run every pass. `compact_tool_results` keeps offloaded placeholders intact by default so their handles stay dereferenceable.

---

## Guardrails

Content policy at the edges of a run (`python/timbal/guardrails/`): `input` (before the first LLM call — a block spends zero tokens), `model_output` (final assistant message, stream-safe), `model_step` (opt-in: every assistant message, incl. intermediate tool-calling steps; per-stage override `on_step=`), `tool_args` (after Pydantic validation, before the approval gate), `tool_result` (before offload, so rails see full text).

```python
Agent(..., guardrails="default")  # DetectPII(redact) + RedactSecrets + PromptInjection(block)
Agent(..., guardrails=["pii:redact", "injection:block", "moderation:warn"])

from timbal.guardrails import DetectPII, LLMJudge, Verdict, guardrail
Agent(
 ...,
 guardrails=[
 DetectPII(on_input="redact", on_output="block", types=["email", "ssn"]),
 LLMJudge("Must not give medical advice", model="openai/gpt-5.4-nano", action="retry"),
 guardrail(lambda text: Verdict.block("competitor") if "acme" in text else True, stages=["model_output"]),
 ],
 guardrail_mode="shadow", # record verdicts, enforce nothing (rollout); default "enforce"
 max_guardrail_retries=2, # budget for retry (reask) verdicts per turn
)
Tool(handler=..., guardrails=[...]) # tool-local rails, work with or without an agent
```

- **Verdicts**: `allow | block | replace | retry | escalate | warn` (`Verdict.block/redact/retry/escalate/warn` helpers). Callable coercion: `True`/`None`→allow, `False`→block, `str`→replace.
- **Block** → `OutputEvent` with `status.code="blocked"`, `status.reason="guardrail:{rail}:{stage}"`, output = user-safe `blocked_message` as an assistant Message (also appended to memory). Blocked tool args feed `[Blocked by guardrail]` back to the LLM.
- **`escalate`** (tool_args) forces the existing HITL approval gate (`ApprovalEvent`, `kind="guardrail_escalation"`, resume flow unchanged).
- **Streaming**: redact-only deterministic rails scrub text AND thinking deltas in flight (per-content-block holdback window); block/retry-capable rails buffer-until-verdict — no chunk escapes. Thinking blocks on stored messages are scrubbed too.
- **Trace redaction**: `provider.configured(_trace_redactor=timbal.guardrails.trace_redactor(...))` scrubs every span's serialized surfaces (incl. the inner LLM span) on copies at store/export time — live run untouched; resumed sessions load the redacted history. Deterministic rails only.
- **Built-ins** (lazy exports from `timbal.guardrails`): `DetectPII`, `RedactSecrets`, `PromptInjection` (patterns + optional `model=` classifier), `KeywordGuard`, `MaxLength`, `Moderate` (OpenAI moderation / llama-guard-style), `TopicGuard`, `LLMJudge`.
- **Rubrics** (`timbal.guardrails.rubric`): `parse_rubric` (markdown bullets / list of str/dicts with weights) + `grade_rubric` (one isolated structured judge per criterion; verdicts pass/fail/unknown + reason; weighted score vs `pass_threshold`). Consumers: `LLMJudge(rubric=..., action="retry")` — grade → revise → re-grade loop with failing criteria as feedback, per-criterion results in verdict metadata — and the `rubric!` eval validator (`timbal/evals/validators/rubric.py`), whose failure message lists every failing criterion with the judge's reason. Write criteria around verifiable structure, not unverifiable facts.
- **Observability**: `GuardrailEvent` per triggered rail (stream + wire), report on `OutputEvent.metadata["guardrails"]`, usage keys `guardrails:triggered` / `guardrails:shadow_triggered`, `agent.explain_guardrails()`.
- **Testing**: `await check_guardrails(agent_or_spec, text, stage="input")` runs rails only (no LLM loop) and returns a per-rail report.
- Rail crashes fail **open** by default (recorded as `action="error"`); `strict=True` fails closed. Orchestrators never apply their own `guardrails` config as tool-local rails on themselves — only parent-injected rails apply to a sub-agent used as a tool.
- **Sampled monitoring**: `Guardrail(sample_rate=0.05, shadow=True)` grades ~5% of checks (online-evals pattern; sampled-out checks record nothing). Sampling an enforcing rail logs a warning — enforcement gaps + streaming still buffers every run.

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
# Optional metadata (not sent to providers). Runtime control messages use:
# metadata={"source": "runtime", "kind": "background_task_completed"}
# Filter with msg.is_runtime() when painting human transcripts.
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

- Tests live under `python/tests/` mirroring the package (`core/`, `core/llm/`, `state/tracing/`, `collectors/`, …)
- All async tests use `pytest-asyncio` (mode=AUTO — no `@pytest.mark.asyncio` needed if configured)
- Use `TestModel` to avoid API calls in unit tests
- `tmp_path` pytest fixture for file-based provider tests
- Test classes group related tests: `TestProviderName`, `TestFeature`
- `python/tests/guardrails/conftest.py` provides `StreamingTestModel`, which streams real
  `TextDelta`/`ThinkingDelta`/`ToolUse` items through the router — use it whenever a test
  depends on delta handling rather than just the final message
- `test_injection_corpus.py` is a regression fence for the injection pattern pack. Changes
  to the pack must keep it green; genuinely uncatchable attacks belong in `KNOWN_GAPS`
  (xfail) rather than being deleted
