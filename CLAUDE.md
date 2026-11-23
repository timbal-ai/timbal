# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing
- **Install dependencies**: `uv sync --dev` (from repo root — `pyproject.toml` is at root)
- **Run all tests**: `uv run pytest -n auto` (from repo root; `-n auto` = one pytest-xdist worker per core, serial without it). Coverage is opt-in: add `--cov=timbal --cov-report=term-missing`
- **Run single test**: `uv run pytest python/tests/core/test_agent.py::TestClass::test_method`
- **Linting**: `uv run ruff check`
- **Format**: `uv run ruff format`
- **Fix lint**: `uv run ruff check --fix`
- **Line length**: 120 chars (configured in `pyproject.toml`)

### Releases
- **Release branches**: `main` is developed on and heads to 3.0 (tagged `v3.0.0.dev0`: it builds as `3.0.0.devN`; nothing is published from it until 3.0). The 2.x line is released from `release/<major>.<minor>` (today `release/2.8`): fixes, model updates, new tools and docs land on `main` first and reach it only as backports: `scripts/backport.sh <pr> [<line>]` cherry-picks a merged PR and opens a PR into `release/<line>`. Never commit to a release branch directly. A new 2.x minor is cut from the previous release branch, not from `main`. Tag on the release branch, then `scripts/release.sh <version>` (GitHub release + CLI binaries, then PyPI via `scripts/publish-pypi.sh`; it asks first). Versions come from tags (`release-branch-semver`; a release branch builds as its next patch). Full flow: `RELEASING.md`.

### Benchmarks
- **Perf gate**: `uv run python benchmarks/instructions/gate.py --base origin/main` counts bytecode instructions per run (`sys.monitoring`, Python 3.12+) for agent, workflow and tool cases on both revisions and fails on any case over +1%. Counts repeat to ~0.01% and do not depend on machine load, unlike wall time (±5–8% here). CI runs it on every PR (`.github/workflows/perf.yml`). `count.py` alone prints this checkout's counts.
- **LangGraph benchmarks**: `uv pip install -r benchmarks/langchain/requirements.txt`, then `uv run --no-sync python benchmarks/langchain/bench_agent.py` (or the other standalone `bench_*.py` scripts). JavaScript dependencies: `npm ci --prefix benchmarks/langgraph_js`.
- **Quick mode**: pass `--quick`; full mode is the default. `bench_revision.py` emits repeated-comparison samples as JSON.
- **Current agent API**: `bench_agent.py --agent-api current` benchmarks `langchain.agents.create_agent`; the default retains legacy `create_react_agent` for historical comparisons. Keep the same environment/harness and switch only `PYTHONPATH` when comparing Timbal revisions.

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
    timeout=120,                          # wall-clock seconds per turn (default: None = no limit)
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
- `timeout` — wall-clock seconds for one foreground call (any `Runnable`; for an Agent, the whole turn). On expiry the handler is cancelled and closed; the call ends `status.code="timeout"` with a `RunTimeout` error (`timbal.errors.RunTimeout`, a `TimeoutError`) and keeps partial output. A parent agent sees a child's timeout as an error tool result; a workflow step fails. Pre-hook, handler, and post-hook share one deadline starting after validation + the approval gate; detached children use `background_timeout`. A handler's own `TimeoutError` stays `error`
- `stream_idle_timeout` — seconds a model stream may send nothing (default 300; `None` = forever). A stall before the first chunk is retried like a timeout (`max_retries`); after it the call fails with `timbal.errors.ModelStreamStalled` (a `TimeoutError`), since a retry would repeat sent output; `is_retryable_provider_error` (the transient-only `FallbackModel` predicate) counts a stall before the first chunk, never one after it
- `max_consecutive_denials` — denied (approval) / blocked (`tool_args` guardrail) tool calls in a row before the agent stops routing around them (default 3; `None` = off): a runtime notice (`kind="denial_limit"`) tells it to ask the user and the next model call gets no tools; root `metadata.denial_limit`. Any other tool result resets the count
- Empty answer at the output limit — a final answer (no tool calls) whose `stop_reason` is `max_tokens` / `max_output_tokens` / `length` and that has no visible text (a reasoning model spent the budget thinking) is dropped from memory and retried once with `max_tokens` ×4 (capped at 32,000; `metadata.max_tokens_retry = {from, to}`); still empty, or no `max_tokens` set to raise → `timbal.errors.ModelOutputTruncated` (status `error`; a parent agent sees an error tool result; `ChatSession` `turn_finished.error = {type, message}`, shown in the playground). Truncated answers that do have text are kept (`status.reason = "max_tokens"`)
- `tool_search` — `"auto"` (default) | `"client"` | `"dispatch"` | `"off"`: how the model reaches tools marked `Tool(defer_loading=True)` / `MCPServer(defer_loading=True)` without their schemas on every call. **Native** (`auto` on Claude 4.5+ and OpenAI gpt-5.4+ via the Responses API — `TIMBAL_OPENAI_API=chat_completions` falls back to client): every tool is sent, the deferred ones flagged `defer_loading: true`, and the provider searches within the same response. Anthropic: plus `tool_search_tool_bm25_20251119`; the `server_tool_use` + `tool_search_tool_result` blocks are kept verbatim in memory (`CustomContent`), an interrupted search gets a `tool_search_tool_result_error`. OpenAI: plus `{"type": "tool_search"}`; deferred tools sharing `Tool.search_group` (an `MCPServer` sets its `name`) go as one `namespace` — without it OpenAI still shows every deferred function's name + description and only the schemas are deferred (measured −7% vs −61%). The `tool_search_call` / `tool_search_output` items are stored as `CustomContent` and replayed as top-level input items (skipped by Anthropic and Chat Completions); a loaded function's `function_call` carries `namespace` (its group, or its own name), kept on `ToolUseContent.namespace` and replayed or the API 400s. SDK 2.24 has no types for these items and parses them as `ResponseOutputMessage`, so the collector checks `item.type` first. **Client** (other models, `FallbackModel`, `TestModel`, or `"client"`): the model gets the non-deferred tools, the deferred ones already loaded (appended in load order), and a read-only `search_tools(query, limit=5)` listing the rest; keyword scoring over name > param names > description, query words every candidate shares (an MCP `name__` prefix) don't count, matches under half the top score are dropped; after 2 searches in a run that find nothing it tells the model to stop, and under a tool policy its description says hidden tools can't be found (a read-only wake otherwise searches for the write tool over and over); it is renamed (`find_tools`, …, then numbered: `search_tools_2`, `_unused_name`) if an agent tool already has the name (an unprefixed MCP server's own `search_tools`), also one a tool policy hides from the run (the policy narrows what a run may call, not what a name means, so the added tools are named the same in a read-only wake as in any other turn; the names come from the step's one resolution, `_resolve_step_tools` → `_plan_tool_search(tools, model, names)`: a `ToolSet` is asked once per step of the loop, never a second time for names); loaded names persist in session data (`__loaded_tools[agent_path]`). A deferred tool called by exact name still runs (and loads). **Dispatch** (`"dispatch"`, opt-in; for non-native models, where client mode's loads change the tools array and so rewrite the whole cached prefix on Anthropic — measured on Haiku 5.5: the call after a load wrote 8,007 tokens and read none, against 563 written and 14,984 read with dispatch): the offered tools never change — the non-deferred ones, `search_tools` (fixed description listing every deferred tool; returns each match's `anthropic_schema` instead of loading) and `call_tool(name, arguments)` (renamed `call_found_tool`, … if taken, then numbered: `call_tool_2`; `is_call_tool_name` is exactly the names it can get, for what reads a stored call by name: `_dispatched`, a recovery notice). `Agent._dispatched` rewrites a `call_tool` call into the call itself in `_multiplex_tools` (same tool_use id; `arguments` may be a JSON string, and left out or `null` (also the JSON text `"null"`) it is `{}`; anything else that is not an object — a list, a number, an empty string — is a malformed call that stays with the dispatcher, whose validation answers with an error, and is never run as a call with no arguments; only in dispatch mode, and only for a call named like the dispatcher, so other modes pay nothing), so it runs exactly like a direct call (approval gate keyed on the real tool, also on resume; guardrails; its own span `agent.<tool>`); an unknown name runs `call_tool`'s own handler, an error result pointing at `search_tools`. Memory keeps the `call_tool` tool_use, and a durable session records it that way too (`in_flight.dispatched`); its recovery notice pairs each request with the span that recorded its tool call id (`metadata.tool_call_id`; by name only for a span without one), so one call is listed once, with its own outcome, also next to a dispatch that failed as `call_tool` (`chat_session._tool_calls`). `metadata.tool_search.mode="dispatch"`, `loaded` = names found. Nothing persists in session data. Tool policy (`read_only` wakes) applies before deferral; built-in tools are never deferred. Each run records `metadata.tool_search = {mode, loaded, searches: [{query, loaded}]}` (native: names from the provider's result), surfaced on `ChatSession`'s `turn_finished.tool_search`; the playground shows provider-side searches as 🔎 cards. **Cost, measured** (30 tools, ~7K tokens of schemas): one request — Haiku 5.5 −67% native, −68% client; gpt-6-luna −61% native (namespaces), −80% client. Over a 5-turn session loading a new tool every turn, billed input with warm caches (read ×0.1, write ×1.25): `off` ≈ native, client ≈ 2× — every client load changes the `tools` array ahead of the conversation, so the whole history is cache-written again; native injects found tools after the cached prefix. Deferral pays for big catalogs (context window, cold first request); for a few KB of stable tools `off` is as cheap once cached
- `max_tokens` — required for Anthropic; sets max completion tokens
- `memory_compaction` — strategy or list of strategies; triggers at `memory_compaction_ratio` (default 0.75) of context window
- `system_prompt` as sections — a dict `{name: text}` (or a callable returning one): a root agent pins them in session data (`__prompt_sections[agent_path] = {pinned, digests, key}`) and sends the same prompt every turn: each section as `<section name="…">`, then a fixed preamble saying the application may send `<system_update section="…" key="K">` blocks that replace a section (`removed="true"` drops it) and that only blocks with key `K` count (a fresh random key per pin, so a user who types the tag cannot rewrite instructions; OpenAI models ignore a user-role update without this preamble). A changed, new or removed section reaches the model as one runtime user message (`metadata.kind = "prompt_update"`, `PROMPT_UPDATE_KIND`; its metadata says what it announces: the pin's `key`, `sections = {name: digest | null}`, and `digest`, the digest of the note's own text) inserted before that turn's prompt; root `metadata.prompt_sections`, usage `prompt_sections:updated`. **What the model has been told is read from the history, not from a record** (`_sections_untold`: the pinned prompt as it is rendered, then the notes of that pin in the turn's memory, in order): the pin is right only together with the notes the model gets, so a history that lost one is told again and one that holds it is not. That covers a history the caller supplies (`messages=`), a turn another agent answered in between, a turn cut before its memory was stored, and a second turn continuing the same run. A note says what its text says, which is all a model gets of it. Its metadata holds the digest of the text it was written with (`digest`), and a note with another text (a caller's `messages=` shortened it, a trace redactor rewrote it) is no proof that the model reads what it records. That matters only where nothing else would be said: when a section still is what such a note recorded, `_sections_untold` returns `None` and the sections are pinned again. Where the section changed again, the note written now is the last word on it and the pin stays. The pin's key is the other half: outside the model's own words (assistant messages are not looked at) it belongs in the pin's notes only. A note stripped of its metadata or joined to another message, a block somebody typed, a note read back through a tool result, a rewritten note that speaks of a section it did not record: text the model may follow and nothing vouches for, so `None` again, and the fresh key voids all of it (file attachments are not looked at). The record keeps each pinned section's digest next to its text, and the digest says which text was pinned: where the application still gives that text, the prompt is rendered from it and not from the stored copy (`_pin_as_written`). So with a trace redactor (`_trace_redactor`) that rewrites a section's text in the store: while the section stays the same, the same prompt bytes and no note; when it changes, a note, with the prompt showing the store's redacted copy of the old text from then on (also when the pinned text comes back: a section the history's notes speak of is read from the notes, so one that toggles does not move the prompt each time; and if a compaction inside the turn then takes those notes, what the pin renders is that redacted copy, which is not the section, so the sections are pinned again; a compaction as the turn starts is read before the pin is rendered, and the prompt goes back to the application's text under the same key); if it then stays, the next turn pins again (the stored note no longer says it); if it changes every turn, a note a turn and the pin stays. Pinned on the first turn, and pinned again (a fresh key, so notes of the earlier pin left in the history stop counting) as soon as a compaction took an update the model still needs, as a turn starts or inside one (`_repin_if_untold`: after a mid-turn compaction the calls left in that turn get the current sections). A compaction that loses nothing moves nothing: the pinned bytes stay, also for a history compacted on every turn. A change is never put into a resumed tool batch (a user message between a `tool_use` and its result is refused by the providers): nothing records it as said, so the next turn says it. The session record is replaced at a pin, never written in place (a run's session data starts as a shallow copy of the run it continues). Skills become a `skills` section. Subagents send the sections plainly joined, and so does a `VoiceSession` for its greeting and idle prompts and for turn one's override after an opener (`_agent_system_prompt`). A `str` prompt (or callable) behaves exactly as before. Measured with a ~5.5k-token prompt whose date line changes every turn: plain string → Haiku 5.5 writes the whole prompt to cache every turn, gpt-6-luna caches nothing; sections → turns 2–3 read ~11k (Haiku) / ~3.7k (Luna) cached tokens, both models use the updated date.
- `delivery` — `"text"` (default) | `"tool"`: how a root agent's replies reach the user. `"tool"`: a `send_message(text, end_turn=False)` tool (`read_only`, and `interruptible=False`: a `now` message does not cut a send in flight, which could be between recording its message and the message being out; a fixed note `DEFAULT_DELIVERY_NOTE` naming it is appended to the system prompt) is the only way text reaches the user. Each message is screened by the agent's `model_output` rails, then delivered at once as a `Custom` delta `{kind: "message", text, index, end_turn}` and listed in the agent span's `metadata.messages_sent`. The turn's output is a synthetic `<agent>.delivery` OutputEvent whose assistant message joins what was sent (`metadata.delivery = {mode, messages, end_turn?, silent?, fallback?, before_pause?}`). Subagents ignore it (they answer with text), and so does an agent with an `output_model` (its turn ends on the model's first answer, before any tool call of it would run). Real models (Haiku 5.5, gpt-6-luna) send an early update with their first tool call and end the turn with `end_turn`.
  - **Screening**: a block/retry verdict returns an error result so the model rewrites the message; a replace rewrites it, and the result then says so and quotes what went out (`Sent, after a guardrail changed it. The user received: …`), since `Sent.` would be a receipt for text the user never got; a message a rail leaves nothing of is not sent (an error result, no empty `message`). The agent's other tool rails treat it as the call it is (the tool is built at run time and wired like the agent's own, `_wire_tool_guardrails`): `tool_result` rails see its result, and `tool_args` rails run on the call before anything is sent, so an application that gates calls at that stage gates what goes to the user too. A block there is the usual `[Blocked by guardrail]` result and no message; an escalation is an approval gate, and nothing is sent until a person lets it go (a send gated that way goes out when it is approved, which is after later sends of its batch that were not gated; the approval is for the one message, `approval_grantable=False`, since "don't ask again" for this tool would cover everything the agent says); a rewrite of the call's text, by a rail at that stage or by the person approving it (`override_input`), is said in the receipt like a rewrite at `model_output` (`_send_as_written`: the handler compares what it sends with the call the agent's memory holds). The gate has no way round it through text: a draft that is delivered stands for a send the model did not make, so it gets a send's checks (`_gate_draft`): the `tool_args` rails decide on it as on a call (`tool_name` the send tool's, `tool_args = {text, end_turn: True}`), and what they leave of it is screened by the `model_output` rails as in the send's handler. For a final answer that second screening happens only where the first stage changed the text: the `model_output` rails have seen a final answer once already, as the model wrote it, which is one difference with a send (whose text they see only after the `tool_args` rails are through with it, so a rail there that acts on the model's raw wording acts on a draft and not on the same text sent). `on_max_iter`'s reply, which nothing has seen, is always screened, and shown to those rails with what it holds besides text (its files, after the text), also when it has no text. Where the checks rewrote a draft, the message in memory and its stored entry are replaced by a copy that says what went out (`_as_delivered`, `_reworded`: the text replaced where it was, the rest of the message in place; the model call's own span and event keep what the model produced), and alone that message is the output. What they refuse is not delivered, and neither is what they would ask a person about, who cannot be asked for a call that was never made, nor a draft a rail answers with arguments the tool would not take; with nothing else sent the turn's output is then empty (`metadata.delivery = {mode, messages: 0, withheld: True}`: nothing of the refused message, not the file its text came with; after a pause the output is the `before_pause` one), and a rewrite is what is delivered and output. Limits: a rail at this stage is given the JSON projection of the call, as for any tool (`Runnable._apply_tool_args_guardrails`), so one that matches text meets escapes for non-ASCII characters, line breaks and quotes where one that reads `ctx.tool_args["text"]` gets the text: rails on what the text says belong at `model_output`, which screens every send and the final draft as plain text. A draft the gate withholds or rewrites records `metadata.guardrails` and usage but emits no `GuardrailEvent`, like a screened send. A turn cut while a rail decides (a timeout, an interrupt) ends with its last step's output like any cut turn, which here is the draft: a channel shows `message` events. And an approval goes by the call's path and input, so a send with the very same text later in the resumed run is not asked again. A rail there sees the silent reply sent as a message like any other send: one that escalates every send should let `SILENT_REPLY` through, or a turn with nothing to say waits on an approval. A slash command's answer does not pass this gate: the command's own call passes the rails as the tool it is. Not covered: the text is on the wire unscreened as the call's input (`ToolUse` deltas, the tool span's `input`), as any tool's arguments are, so a channel shows `message` events, not tool inputs; a screened send records span `metadata.guardrails` and usage but emits no `GuardrailEvent`; the final draft is still screened as the final answer, and a block there ends the turn `blocked` though its messages went out.
  - **Name**: `send_message`, else `send_to_user` / `reply_to_user`, else numbered (`send_message_2`, `_unused_name`). Picked at the turn's first step among every tool name the agent has there (`_resolve_step_tools`: a tool set's, so an MCP server's own `send_message`, and those a tool policy hides) and fixed for the turn; a tool set that brings the name later in the turn is not offered that step. The tool is added after the tool policy filter (no policy hides the reply; a step with tools withheld gets none), is never deferred, and in dispatch mode `call_tool(name="send_message")` runs it too. Among the tools a model calls, only its own result carries `ToolResultContent.delivered` (a delivered slash command's result carries it too).
  - **`end_turn=True`** ends the turn after that tool batch with no further model call, whatever the batch's other calls returned, unless something joined the turn's memory meanwhile (`steered`: a user message, an application event, the notice for a message a guardrail dropped): like a final answer in text, the turn does not end before answering it.
  - **Drafts**: other assistant text is a draft. It is delivered (one `message` event with `fallback: True`) when the turn sent nothing and it is not the silent reply, or when its step had no send tool: past `max_iter`, after a denial limit withheld the tools, and `on_max_iter`'s own assistant reply. After messages the turn sent it joins `messages_sent` and the output (`delivery.fallback`); alone it is the output already (`delivery.messages = 0`). What an `on_max_iter` hook's reply holds besides text (a file) is in the output either way, next to all the turn told the user, also when the reply has no text (`_deliver_reply`). The draft is the step's own answer (`last_output_stored`): an empty answer is not kept in memory, where the last message is then the user's prompt or a runtime notice, and neither is ever delivered.
  - **Slash commands**: no model runs, so the command's own result is delivered as a `fallback` message, and it is the run's output too (`_deliver_command`): the result's text, with its files in the output (a message event carries text only). The text is the tool result a model would read (a dict as its `str()`): as the tool's `tool_result` guardrails left it (a blocked one: the rail's `blocked_message`, not the notice written for the model, and no files, in the output or in the stored result, which would hand them to the next turn; one a rail leaves nothing of: no message, an empty output) and before a result limit reduces it (`command_run.answer`, taken in `_process_tool_event` between the two). Only a result: a denied or failed command delivers nothing, and its output is the tool's own event as in text mode. The tool's own output event, earlier on the stream, holds the raw output like any tool's. A list is several values, a line each (`_one_text`; glued, `[12, 34]` reads `1234`; an item that ends its own line gets no second break): the result's text items are made one text before anything reads them, so the result's guardrails check and rewrite what is delivered. Among the values a generator tool yielded, strings that follow one another are pieces of one stream and are joined as they are; any other value has its line. (An agent's or a workflow's handler is a generator too, and their output is not its chunks: a workflow whose last step returns a list delivers a line each.) A message's text blocks are one text, as a model's own answer is. In history the command is what the user was told (`_keep_command`). What the user receives is the answer's text and its files, and with either the stored result is marked `delivered` like a message sent (with the start of the text, `_COMMAND_MARK_CHARS`, empty for an answer that is only files: the result holds the text, or what a result limit left of it, and a mark as long would store it twice; `None` is "not delivered"), so `compact_tool_results` keeps the call and the result that a store keeping the history as calls hands the next turn. With a text delivered, the assistant message the command leaves in live memory is that text. With nothing delivered as text (no result: denied, failed, timed out; or no text in it: a guardrail removed it all, only files, no output) no assistant message stands for the command, and live memory gets the call and the result as such a store has them: the notice for a model, the files. It gets them too, before the assistant message, when an answer with text has files: they are part of what the user received, and a tool result is where a tool's files are kept for a model (no provider takes an image or a document in an assistant turn). The text is then in live memory twice, in the result and as the assistant message. A tool result is not taken everywhere either, for a command as for any tool that returns an image: Anthropic and the Responses API take it there, a Chat Completions tool message does not (OpenAI drops it, Gemini refuses the request; see `ToolResultContent.to_openai_chat_completions_input`). Outputs of calls nested in the command are not kept in live memory. Text mode is unchanged: live memory keeps `str()` of the raw output of the command and of every call nested in it, and `tool_result` guardrails read a list's items glued, as for any tool result (`tool_result_text`). Pre-existing and unchanged in both modes: after a delivered text, live memory and the stored history differ in shape (an assistant text, with the call and the result before it only when the answer has files / a call and its result, so for a blocked result the stored history has the notice for a model rather than the `blocked_message`); a blocked result of a tool the model called, and of a command in text mode, keeps its files (`replace_tool_result_text`), and there the `tool_result` guardrails are not run on a result without text, so one that is only files is not screened (a delivered command's is: its rails run on the empty text, and one that reads `ctx.payload` decides on the files). A command that paused resumes as a command on every store (`_reopen_command`), so its answer is delivered, screened and kept like one that never paused.
  - **Order**: the sends of one batch reach the stream in the order the model wrote them, whatever screening or trace save each takes (`_SendOrder`, only for a batch with two or more): after its own screening a send waits for the ones before it to be settled, then records and publishes in one step with no await. Settled means the earlier message has left the agent, not that its handler returned: a call's emitted event reaches the stream only after the call's span was stored, so the agent settles a send as its `message` event passes (`_in_send_order`, wrapped around the batch's events), or at the call's own output whatever became of it (refused before its handler ran, failed, cancelled). A send that publishes nothing (refused, the silent token) settles itself; a call the order does not know waits for none. Each `message` event stays inside its call on the stream: after its START, before its OUTPUT.
  - **Silence**: sending exactly the silent token ends the turn; `metadata.silent` only on a turn that told the user nothing, in this run or in the one it resumes (a channel hides a silent turn), and not once the turn goes on past that `end_turn` (a message steered in: what it answers next is delivered).
  - **Denials**: a message sent is not a call that went through. It neither resets `max_consecutive_denials` nor enters `max_denials_window`, or a model that says "trying again" before each refused call would never reach either. A send a `tool_args` rail blocks counts like any blocked call.
  - **Across a pause** (approval, `ask_user`): the run that resumes carries `metadata.sent_before_pause` and an `end_turn` asked for in the paused batch, with the `silent` it came with (`resolve_memory`; the turn then ends silent after the resumed calls, also when the resuming run has no silence token). A draft after the resume stays private, the carried `end_turn` ends the turn once the resumed calls ran, and the resumed run's output is its own messages (an empty one when it sent none; `delivery.before_pause`). A run that cannot use the carried flags (it answers in text, or no call of the paused batch is left) drops them.
  - **Per call**: an agent set to `"tool"` answers a call in text when the call passes `delivery="text"`. Only that value, only that way round, and only where it overrides something: there it is taken out of the call's inputs. Any other value, and any value (`"text"` too) passed to an agent that would not deliver by tool anyway (not set to `"tool"`, a subagent, one with an `output_model`), stays what it was: an input of the application's own that goes on to the model call's input with the rest of them. A `VoiceSession` passes it: a call reads the reply aloud as it streams, where a draft would be spoken and a sent message not.
  - With `store_before_acting` a send is not behind the barrier (`read_only`); a durable session records it like any call (`in_flight.dispatched`).
- `tracing_provider` — `TracingProvider` subclass, `None` to disable, or `TRACING_UNSET` (default, auto-detects)
- `default_params` — fixed or callable defaults applied before user kwargs
- `pre_hook` / `post_hook` — parameterless callables; can call `get_run_context()`
- `approval_reviewer` — decides approval gates of the agent's tool calls before a person is asked (`python/timbal/core/approval_reviewer.py`): an `ApprovalReviewer(model=..., instructions=, kinds=(None, "mcp_tool"), context_messages=6, timeout=30, max_tokens=400)`, a model string, or a function `(ApprovalRequest) -> "approve" | "deny" | "escalate" | bool | (verdict, reason) | ReviewVerdict`. Hooked in `Runnable._apply_approval_gate` after resume values, grants and cancels (so a person's answer and "don't ask again" win), via `_review_gate`: the nearest calling agent with a reviewer (subagents inherit), shown the tool, description, arguments (redacted view; `hidden_arguments` when redaction hides a value), what the person said and whether a person started the turn. Evidence is by provenance: `user_messages` are the `said_by_the_user` messages (`role="user"`, no `source`, not a summary from before summaries were marked) of the agent the person talks to — the root agent of the run nothing else started, found up the call chain and across runs a call's code started. `RunContext._invoker = (run, call_id)` is that link, set in `Runnable._stream` when a run's root span is minted while a call that is still running is ambient (`_LIVE_CALLS`: call id → run context, for every call whose stream has not ended): an agent called inside a tool or step body, forked from the running run (where `invoked_by` is stamped) or on a run context the handler built and set, or with none set; it is its own run's root. A detached call (a background child) is kept apart while its code runs (`_DETACHED_CALLS`, from the task being made to its end: its stream is over, its handler is not), so a context it builds is linked the same way; a fork is linked by itself while the turn is still running (as is a plain task a tool started), and by `current_background_task_id()` once it has ended. An ambient id that is none of these is one a stream abandoned mid-way left behind in its consumer's task, and links nothing. A run linked at the mint (a context the handler built, or none) also inherits the caller's tool policy when it has none of its own, like a forked one. When such a run ends, the code that started it gets its ambient call ids back (they were left `None`, so only the first run a handler started was linked, inherited its tool policy, or was seen as delegated). The run context set then may be one the handler swapped in, which those ids are not from: `RunContext.update_usage` records on the root of the context it is called on when no call of that context is ambient (none, or another context's). A subagent's, a workflow step's or such an agent's own prompt is `delegation`, never authorization, and with a workflow or a tool at the top there are no `user_messages`. `attended`: that top run's root `metadata.trigger.kind` is `user_message` / `resume` / absent (`ChatSession` stamps the trigger on wake, recovery and resume turns too). A message posted to a session as the user's (`ChatSession.send`, `post_event(source="user")`) by a call's code, or by a task that code left running, is a tool's or a model's text: `SessionEvent.sent_by = {run_id, call_id}` is taken where it is posted, from the ambient call id whether or not that call is still running (the turn that receives it runs in the session's own tasks, with the ambient call ids cleared; not from an id a stream left mid-way left behind in its consumer's task), kept in `to_dict()` / `restore_event` so a durable session restores it, and carried by the message it becomes (`metadata.sent_by`). A turn's batch never joins a person's messages with ones a call posted (`user_batch` takes the leading ones that share a sender; the rest wait for the next turn), so a person's latest words are always a message of their own. `said_by_the_user` is false for it; when it is the request the person-facing agent is acting on, the reviewer gets it as `delegation` with `attended=False`. A call that has ended also links what its code left running: its id is ambient there (a task a tool started that calls an agent, or posts to a session, after the tool or its turn is over, also when the tool's stream was cut meanwhile). The one ambient id that is nobody's code is what a stream left mid-way leaves in the task that was consuming it (`_ABANDONED_CALLS`: call id → that task, `RunContext._driver`, noted on `GeneratorExit`, bounded; `_left_behind` compares it with the current task): a person's next turn or message from that task is theirs. A model reviewer escalates without a call when the arguments exceed `ARGUMENTS_BUDGET` (8,000 chars), a value is hidden or only a delegation is available, and its answer counts only as one schema-valid JSON verdict (no prose around it, no repeated key). A reviewer's `approve`/`deny` claims the gate like a person's answer (`claim_approval`) when the run it continues left that approval waiting (`RunContext._left_waiting()`: the ids of the parent's spans still `approval_required`, read once when a gate asks, whatever the parent's root status and whether or not resume values were given; read with `get_strict`, since the built-in providers' `get` answers a failed read with `None`; a read that fails is tried again after 0.05 s and 0.2 s, and one that keeps failing raises `timbal.errors.ApprovalStateUnreadable`: the gate's call fails and is neither resolved, claimed nor asked again, for a person's answer too, since deciding it blind could claim under the wrong run or re-ask it under an identity later runs do not share; no read with a provider that keeps no claims). A call that waited nowhere is not claimed (two turns from one finished run both make it). A session grant (`scope="session"`) that approves a gate the run being continued left waiting claims it as well (a fresh call it approves does not). A waiting gate is one tool call: the lookup is by `(approval_id, tool_call_id)`, since the approval id alone is the call's path and arguments and a second, identical call (the model reads the same status again) is a gate of its own, asked and answered on its own. A claim (a person's, a grant's or a reviewer's) is made under the run the gate first waited in, not always the one being continued: a partial resume asks again for the gate it was not given an answer for, which then waits in both runs (`metadata.approval.pending_since`, set when a gate waits again in a run continuing one it waited in), and runs continuing either must be after the same claim. **The run that loses a claim stops** (a person's answer or a reviewer's): an agent no longer feeds `approval_already_claimed` back to its model as a tool result and goes on; it ends `cancelled` / `approval_already_claimed` with the tool_use left unresolved (`RunCancelled(reason=)`, `agent._STOP_REASONS`; a workflow stops the same way), like a gated tool called directly always did. It stops once its tool batch is over, not at the event: a call of that batch this run did claim is running, and cutting it would leave it claimed and never done (two continuations that each win one gate of a pair both finish theirs, then both stop). The other run is the one continuing the turn, so nothing that would follow is done twice. `approve`/`deny` become an `ApprovalResolution` (`approver_id="reviewer:<model>"`, `metadata.reviewed`; a denial reaches the model as `[Approval denied] Denied by the approval reviewer: …` and counts as a denial); `escalate` — also any error, timeout or unreadable answer — emits the `ApprovalEvent` as before with `metadata.review`. Span `metadata.approval.review = {verdict, reason, reviewer}`, usage `approvals:reviewer_approve|deny|escalate`. Guardrail escalations are not reviewed unless their kind is listed. The model prompt asks to check every argument value against the request and to escalate when unsure; arguments and excerpts are data. Measured (5 cases with a clear answer: requested refund, unrequested refund, unattended delete, injected email, 100× charge): Haiku 5.5, GPT-6 Luna, gpt-5.4-nano 5/5 twice each after that instruction — before it nano approved the 100× charge. `call_judge` (also behind `LLMJudge`) sends no `temperature` (Haiku 5.5 rejects it)
- `max_denials_window` — `(denied, calls)`, e.g. `(10, 50)`: the `max_consecutive_denials` stop (tools withheld, notice `DEFAULT_DENIAL_WINDOW_NOTICE`) once `denied` of the agent's last `calls` tool calls were denied or blocked, in a row or not; the outcomes live in session data (`__denial_window[agent_path]`), so it counts across turns and wakes; cleared when it fires; `metadata.denial_limit` gets `{denials, window}`. Default `None`

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

**Slash commands** (`command="name"` on a `Tool`, an `Agent` or a `Workflow` in an agent's `tools`; `docs/agents/commands.mdx`): a prompt `/name args` runs it without the model (`Agent.handler`, the `if commands:` branch). The agent stores a made-up `tool_use` before the command runs, marked `metadata.kind="command"` (`_COMMAND_CALL`), and the result after it; in text mode live memory gets `str(output)` of each finished step as assistant text instead. A command that pauses (approval, `suspend()`) has no result, so the paused turn's history ends with the marked call, in a store and in memory (`_leave_command_waiting` sets memory to what is stored, also when the caller stops reading at the approval card). The turn that continues it starts as any turn continuing a pending call (no completion notices or session events taken, no second input-guardrail screening), then takes the call back out (`_reopen_command`) and the command branch runs the command again from the user's message: no model call, the run's output is the command's, a denial ends the run `cancelled` / `approval_denied`, and a `suspend()` answer reaches its question (its id hashes the tool call id, which a command's run has not). The command runs from its start, so an agent's or a workflow's finished steps repeat, as for any resumed sub-agent or workflow. Run again by a resume, the command is given its stream to the end before the turn pauses or stops on a lost approval claim, as a batch of pending calls is (a call in flight may be one the user has just approved); a first run stops at the first pause (nothing is approved yet, and what finished there would take effect again on the resume); `Cancel` stops at once. If the step runs no command for the call taken out (the command was renamed, removed or given to another tool since the pause, a tool policy hides it, a compaction took the user's message), it goes back and runs as the pending call it was stored as, followed by a model call (`_run_as_pending`); so does the unmarked call of a run an earlier version paused. A command sets `_llm_memory_saved`: no model ran in its step. It is False there only after a `now` cut took a partial answer back; salvaged after a command that pauses, that answer would follow the command's stored call and the resume would not find the call. Tests: `python/tests/core/test_command_resume.py`.

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
        MCPServer(
            name="timbal", transport="http", url="https://api.timbal.ai/mcp", headers={"Authorization": "Bearer ..."},
            approval="destructive",                   # gate tools the server annotates as destructive
            sampling_model="anthropic/claude-haiku-4-5",  # answer sampling/createMessage with this model
        ),
    ],
)
```

- `transport` — `"stdio"` (spawns `command` + `args` with optional `env`) or `"http"` (streamable HTTP at `url` with optional `headers`)
- `name` — optional identifier; when set, tools are exposed as `{name}__{tool}` (bare name still used for `call_tool`). Required for codegen (`remove-tool --name`) and whenever multiple servers might share tool names
- Connections are lazy and each lives in its own owner task (`close()` works from any task, any order). Failure handling: a transport that dies mid-call fails the call immediately (`connection lost`) and reconnects on the next call; a stale HTTP session (`Session terminated` 404 after a server restart) is reset and the request retried once; connect failures surface as `ConnectionError("MCP server 'x' failed to connect: <real cause>")`. `timeout=` (seconds, default 600; `None` = forever) bounds `tools/list`/`tools/call` on a live session, `tool_timeouts={bare_name: seconds | None}` overrides it per tool; `connect_timeout=` bounds the handshake separately (stdio cold starts are slow; HTTP TCP connect is capped at 30s regardless). The HTTP transport's read timeout is unbounded — a bare `httpx.AsyncClient()` 5s default used to hang any tool slower than 5s because the SDK swallows the `ReadTimeout`. `tools/call` is never retried after a mid-flight loss; `tools/list` is. The tool list is cached until `await server.close()`. Tools are sorted by name (stable LLM tools prefix → prompt-cache friendly)
- Pinned `mcp>=1.26,<2`: SDK 2.x (2026-07-28 protocol) renames `FastMCP`→`MCPServer`, replaces the `ClientSession` layering with `Client`, and changes `streamable_http_client`'s yield — migration is a separate effort
- Results: text → str, images/audio/blobs → `File`s in a `Message`, `structuredContent` fallback, `isError` → raised so the LLM sees an error tool result
- `MCPTool` carries `input_schema`, `title`, `tool_annotations` (server `ToolAnnotations` as a dict)
- **`approval=`** — client-side policy routed through the normal approval gate (`ApprovalEvent`, `kind="mcp_tool"`, `ui={server, tool, title, annotations}`, `input_schema` = server schema). `"destructive"` gates tools whose annotations say destructive (`destructiveHint` not false and `readOnlyHint` not true; unannotated tools are *not* gated), `"all"` gates everything, a one-arg callable receives the `mcp.types.Tool` at resolve time, a two-arg callable `(tool, arguments)` is decided per call with the LLM's arguments (may be async; returns bool/dict/`ApprovalPolicyDecision`) — for one tool multiplexing subcommands. The gate runs client-side before the wire call, so it is unaffected by parallel tool calls and by rebuilding the `MCPServer` between pause and resume. Default `None`
- **"Don't ask again"** (any gated runnable, not MCP-specific) — resume with `{approval_id: {"approved": true, "scope": "session"}}`: approves the call and stores an `ApprovalGrant` in the run's session data (`RunContext.get_session()`, root-span `session`, so it persists with the trace on every provider) under the decision's `grant_key` (default: runnable path = "this tool"; `ApprovalPolicyDecision.grant_key` / `Runnable.approval_grant_key` narrow it, e.g. `codegen:delete-workforce`). Later gates in the same `parent_id` chain with that key are approved without an `ApprovalEvent` (`span.metadata.approval.granted`, resolution `scope="session"`, `metadata.granted_by`; usage `approvals:granted` / `approvals:remembered`). `ApprovalEvent.grant_key` / `.grantable` tell the client what the button remembers and whether to show it; `grantable=False` (`Runnable.approval_grantable` or the decision) approves such a resolution for the call only (`grant_refused`). Explicit resume values beat grants; declines are never remembered; `expires_at` bounds the grant
- **Elicitation** (`elicitation=True`, default) — the client advertises the capability and a server `elicit()` mid tool call becomes a `suspend()`: the run ends `input_required` with an `InteractionEvent` of `kind="mcp_elicitation"`, `payload={server, tool, mode, message, requested_schema | url + elicitation_id}`, `response_schema=requested_schema`. Resume with `{interaction_id: value}` where `value` is a content dict (→ accept), `True` (accept, empty content), `False`/`None` (decline), `{"action": "accept"|"decline"|"cancel", "content": {...}}`, or `Cancel(...)`. Content is checked against `requested_schema` (required keys, primitive types, enums) before it is sent. Constraints: the tool is re-invoked on resume, so the server's tool must be idempotent up to the `elicit()` point and its message deterministic for identical input (the suspension id hashes `(path, payload, tool_call_id)`). Two or more calls in flight on one session cannot be attributed when one elicits (the protocol carries no correlation), so the question is answered `cancel`: a call the server then fails gets an error telling the LLM to call the tool on its own; one the server answers normally succeeds with an auto-cancelled note appended to its text result (every in-flight call carries it). Prefer `approval=` for confirmations on tools likely to run in parallel. URL mode is best-effort: resume `True` once the out-of-band flow is done. `elicitation=False` hides the capability — servers should then fail closed on tools that need confirmation. **Stateless HTTP servers** (no `Mcp-Session-Id`) cannot receive our answers: detected at connect (warning), and a server-initiated request fails the tool call fast (`...server is stateless...`) instead of hanging; requests a server sends without `related_request_id` never leave it — only `timeout=` covers those
- **`tools/list_changed`** — marks the cached list stale; it is refetched at the next run boundary (different run id), never mid-run
- **Logging** — server `notifications/message` are emitted through structlog (`"MCP server log"`, with `server`, `level`, `logger`, `data`)
- **`defer_loading=True`** — marks every tool of the server deferred: found through `Agent.tool_search` instead of carrying all schemas on every call (big catalogs)
- **`sampling_model=`** — model string (or `TestModel`) used to serve `sampling/createMessage`; runs as an `mcp_sampling` LLM span under the in-flight tool call. The SDK answers server requests inline on the receive loop, so other traffic on that session waits while the LLM call runs
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
anthropic/claude-fable-5-2
anthropic/claude-fable-5
anthropic/claude-opus-5
anthropic/claude-sonnet-5
anthropic/claude-opus-4-8
anthropic/claude-opus-4-7
anthropic/claude-opus-4-6
anthropic/claude-sonnet-4-6
anthropic/claude-haiku-4-5
openai/gpt-6.2-astra
openai/gpt-6.2-sol
openai/gpt-6-astra
openai/gpt-5.5
openai/gpt-5.5-2026-04-23
openai/gpt-4o
openai/gpt-4o-mini
openai/o3
openai/gpt-5.4-nano
google/gemini-2.5-flash
google/gemini-2.5-pro-preview
groq/openai/gpt-oss-120b
xai/grok-4
cerebras/gpt-oss-120b
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

**`SqliteTracingProvider`** — one connection per file, WAL, `synchronous="NORMAL"` (survives a crash or kill, not a power cut; `"FULL"` opt-in); a forked child opens its own (`os.register_at_fork` clears `_DATABASES` and replaces `_DATABASES_LOCK`; the parent's connections are kept in `_INHERITED_AFTER_FORK`, never closed there: closing one would let go of the parent's file locks); a `spans` table upserted per span — a save writes the spans new or still running, every span once the root ended — and `runs.spans` still holds the whole trace for ended runs (older rows read too).

**Session chaining** — pass `parent_id` via `RunContext` to retrieve the parent run's trace in `get()`. Used for multi-turn memory across process restarts. `get()` should return `Trace(spans, parent_id=<the run that one continued>)`: a turn cut before its agent assembled memory stores none, and the next turn rebuilds it from the turn before plus the cut prompt (`Agent._previous_memory`, up to 8 such turns back); a run cut before its input was resolved (admission, session data) stores the input as it was given, so that prompt exists. A cut batch answered only in part (a `now` cut, a resumed batch) gets `INTERRUPTED_TOOL_RESULT` for its other calls on the next turn (`_synthesize_missing_tool_results`; one tool message per call, since Chat Completions sends one result per message), which would otherwise run them as pending. Live spans (in-memory) and serialized ones must agree, so `span.memory` stays unset until resolution completes. `get_strict()` is `get()` except that a failed read raises (JSONL, SQLite, platform: 404 → `None`); the default calls `get()`.

### Platform storage

- **Saves are buffered by default** (`PlatformTracingProvider.default_durability = "buffered"`; `TracingProvider.default_durability` is `"sync"` elsewhere): a save while the run goes is a PATCH round trip (~85 ms on dev, more with payload), and awaiting one per finished span added ~2 per tool step, seconds on a long turn. Mid-run saves are stored in the background and coalesced; the run's end and every other barrier still wait. A crash loses at most the lag (≤ `max_lag_s`, 30 s; past it saves are awaited again). Precedence: `configured(durability=)`, then `TIMBAL_TRACE_DURABILITY`, then the provider's default — so `sync` stays one setting away. The default adds no `tracing:*` usage keys (`metrics_on` follows explicit settings only).
- `PlatformTracingProvider` negotiates per host from response headers: `Accept-Encoding: gzip` → gzip PATCH bodies (`compress=` / `TIMBAL_TRACE_GZIP` force it); `Accept-Patch: application/vnd.timbal.trace-delta+json` → span deltas (`deltas=` forces): mid-run saves carry the finished spans the platform has not acknowledged plus the root — whole until a whole root carrying `memory` is acknowledged (`_DeltaRun.root_stored`; one copy per run, so a process dying mid-turn still leaves its input, history and session data stored; a memory-less root such as a workflow's keeps going whole, it is small), then as a header the platform merges into it, the root's end the full root and any span whose content changed since acknowledged; acknowledgement only on success, so a failed save's spans go with the next. Delta spans are `{"seq": <start index, root = 0>, "span": {...}}` (no other span takes 0, even one preceding the root). A delta counts as landed only if the response echoes `X-Timbal-Snapshot`; without it (today's API answers a delta with 204) the save is resent whole and deltas are off for the host. Bodies over 256 KiB are gzipped off the event loop. Every PATCH carries `X-Timbal-Snapshot` (also the delta body's `snapshot`): ns since the epoch, strictly increasing, raised to any `X-Timbal-Snapshot` a response returns (the server drops a save not newer than the last it applied; another process may continue the run). `X-Timbal-Snapshot-Applied: false` with a newer stored snapshot this process did not send (`_own_snapshots`) → resent once, numbered above it. Saves are single-flight per run (`_flights`): one PATCH on the wire, saves arriving meanwhile join one next round sent from the latest state; only the run's end goes past a PATCH on the wire for `_stall_s` (10 s), at most 2 on the wire per run; a success settles earlier rounds still on the wire; `_unsaved` holds runs with a save asked for that has not landed (at most 256; one pushed out is logged, `platform_trace_unsaved_evicted`). An ended run whose save failed on a transient error (429, 5xx, timeout, connection) is resent in the background from its latest state (`_retries`: waits `_retry_delays_s` 1→60 s, within `_retry_window_s` 600 s, then `platform_trace_retry_exhausted`); a shutdown flush cancels those and sends the runs itself. Resends (background and flush) go through the run's own provider class (`_Flight.provider`, `_owner(context)`), so `configured(compress=, deltas=)` holds; a background resend skips its turn while buffered uploads hold the run (if they hold it the whole window, it logs `platform_trace_retry_handed_to_uploads`, not exhaustion). Shutdown: `uploads.flush_all()` (server lifespan) ends with `PlatformTracingProvider.flush()` — ends first and without the stall wait, unlanded runs resent from their latest state, bounded by `TIMBAL_TRACE_FLUSH_TIMEOUT_SECS` (5 s), returns + logs (`platform_trace_flush_incomplete`) run ids that did not land; scripts get an `atexit` flush on a fresh loop (`asyncio.run` cancels pending saves). `413` is never a format fallback and never resent as is: the run is marked `_oversized` and from then on saved whole with long strings cut (`_shrink`: caps 64 KB → 4 KB → 512 chars until under 40 MB; root `metadata.trace_truncated = {max_chars, strings_cut}`; the local trace is untouched), so its end still lands instead of staying `running`; a second 413 is final. Reads come back zstd-compressed from hosts that compress responses: `httpx[zstd]` (the `zstandard` dependency) makes httpx advertise and decode zstd, no provider code involved (measured on dev: a 1 MB run read is 25.6 KB on the wire, same latency as uncompressed; without it httpx asks for gzip, which the server encoded ~3× slower). A GET's `Trace.parent_id` is the response's `parent_idempotency_key` (the parent's run id, monolith `01203842`) when present, else `parent_id` (the parent's row id, which a GET also resolves). Trace PATCHes and GETs share one connection pool per event loop (`_pooled`, the `platform_http_session` machinery: no response cookies, per loop and pid; a caller-scoped session wins; closed at `flush()`), so a save pays no TLS handshake (measured on dev: delta save p50 287 → 85 ms); cancelling a caller does not cancel its save (trace PATCHes get a 120 s read timeout when `TIMBAL_HTTP_READ_TIMEOUT` is unset — the client default is unbounded). `get()` waits for queued rounds too, and asks `?select=resume` once a host sent `X-Timbal-Select: resume`. Agent spans always serialize a `memory` key (`null` until assembled, `Agent._dumps_memory`), so the resume projection is "root + every span with a `memory` key". A 400/415/422 to either format retries the save whole and uncompressed; if that lands, the format is off for the host for good (`_HostCaps.*_rejected`: outlives later advertisements, overrides `compress=True` / `deltas=True` — today's server answers gzip with 400). A successful whole save clears the run's delta acknowledgements. Server side: `rfcs/platform-trace-storage.md` (measured: a 40-step run uploads 72.6 MB today, 1.5 MB gzipped, 0.07 MB as gzipped deltas; 82–92 % of a trace is LLM spans repeating their input — content-addressed inputs, RFC step B).
- **Keep-alive saves** (`PlatformTracingProvider.keepalive_s`; `None` → `TIMBAL_TRACE_KEEPALIVE_SECS`, default 300; 0 turns it off). Every `_store` re-arms a per-run timer (`_keepalives`). When it fires on a run that hasn't ended, the run saves again through the normal single-flight path: with deltas, that is the root header alone (~150 B).
  - **Why:** a long tool call or a slow model otherwise leaves the run silent, and the platform reads a running run as `timeout` once it has gone `RUN_TIMEOUT_MINUTES` (30) without a trace save: `COALESCE(last_saved_at, created_at)`, where `last_saved_at` is written only by trace saves, header-only deltas included (monolith#574). The span sweep (`trace_spans::sweep`) settles a run's span rows on the same rule. So a live run keeps reading `running` and keeps its rows.
  - **Progress:** keep-alive saves (contextvar `_KEEPALIVE`) don't count as the run's own progress. A run that hasn't saved for itself in `TIMBAL_TRACE_KEEPALIVE_MAX_SECS` (default 4 h, a hung tool) stops (`platform_trace_keepalive_stopped`) and is left to read `timeout`.
  - **No timer:** an ended run, `_closing`, or a run that never saved mid-run (`durability="end"`). `flush()` cancels this loop's timers. A failed keep-alive logs `platform_trace_keepalive_failed`; the timer was already re-armed.
  - **Hosts before monolith#574** derive `timeout` from `created_at` instead: there a run over 30 minutes reads `timeout` while alive, whatever it saves.
- `ChatSession` turns on a platform provider default to `durability="buffered"` with `store_before_acting=True` (also when `ChatSession(durability="buffered")`; a host that takes deltas no longer makes them `sync`), unless the provider, `TIMBAL_TRACE_DURABILITY` or `ChatSession(durability=)` chose otherwise. `_turn_provider` sets `durability` only when the provider's mode differs: the platform default is buffered already, and an explicit mode would also turn on the `tracing:*` usage keys. `TracingProvider.store_before_acting` (default False, so existing buffered providers are unchanged): in buffered mode the agent stores the run (a barrier, `_save_trace(barrier=True)`) before a model's tool batch that is not all `read_only` (unknown names count as acting; a `call_tool` dispatch is judged by the tool it runs, so a read-only target does not wait for a save, and one that resolves to nothing counts as acting), so a crash between a call and its effects never loses the call that durable-session recovery reads; one awaited save per acting batch instead of ~2 per step under `sync` (measured on dev, 20 steps, no model: 11.8 s before → 4.5 s pooled sync → 2.4 s buffered). Resolved once per turn (`stores_first`, `agent._stores_before_acting`). A barrier that fails is logged and counted like any save (`tracing:save_failed`) and the batch still runs. Not behind the barrier: a slash command's tool, and the calls a resume left pending (the paused run stored them when it ended).

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
- Once *this session* has a task, the agent auto-gains `get_background_task` / `list_background_tasks` / `cancel_background_task`, and `message_background_task` once one of its tasks takes messages (below). Opt in to `read_background_transcript` with `background_transcript_tool=True`.
- **Steering a background agent**: an `Agent` detached as a task (`steering` not `False`; tools and workflows never) reads messages sent to it after each step, as one user message (`metadata.steered`, `sources`; input rails screen it, `max_iter` budget restored; a final answer with one waiting calls the model again), like a root turn's `lane="next"`. Senders: the parent LLM via `message_background_task(task_id, message)` (auto-added once the session has such a task, `store.has_steerable`, sticky; not read-only, so a `read_only` tool policy hides it; listings show `takes_messages: true`), the app via `store.message_task(task_id, text, source="app")`, the user via `ChatSession.message_task` (`source="user"`; log records `task_message` and `task_message_settled`, not persisted). Returns a `SessionEvent` receipt: `delivered` (with the run id) once read, which is once it is in the task's memory (`take_messages`, then `mark_read` after screening and the append, which is also when `metadata.task_messages` and usage `background:messages_delivered` are recorded: a task cancelled while an input rail still screens a message it took never read it, and one a rail redacts down to nothing is `leave_unread`, `undelivered` at once), `undelivered` if the task finished first. The completion notice then lists it ("finished before reading your message", `unread_messages`), and so does a poll that stands in for the notice (`get_background_task` / `list_background_tasks` with `ack_completion`: `unread_messages` in the summary and the listing, `BackgroundTaskStore._ended`). Raises `LookupError` (unknown), `ValueError` (finished, which includes cancelled / timed out / stalled and still unwinding; not an agent; empty; > 4,000 chars). Thread-safe, with no hop onto the loop: a message is accepted under the task's own lock (`BackgroundTask._accept`), and the task's end closes the inbox under the same lock (`_drop_unread`), so an accepted message is read or reported and a later one is refused, whichever thread sent it. `unread_messages` is read by the parent's model, in a runtime notice its input guardrails do not screen, so it quotes a message only to a reader it is safe with (`BackgroundTask.unread_texts`): one a rail redacted down to nothing is named by its id, never quoted (`UNREAD_REMOVED_BY_GUARDRAIL`: the report would put back what the rail took out), and so is one a user wrote (`UNREAD_FROM_USER`: the task ended before taking it or was cancelled while screening it, so no guardrail cleared its text, and a user's words reach a model through that model's input guardrails); the parent's own message and the application's are quoted. The user's side learns what became of a message from its receipt, which `ChatSession.message_task` also writes to the session log (`task_message_settled`: `task_id`, `event_id`, `state`, `run_id`). `receipt.settled()` can be awaited from any loop (a sender on another thread): each waiter is woken on the loop it waits on (`call_soon_threadsafe`), and registering a waiter or an `on_settle` callback and settling meet under one lock (`_RECEIPT_LOCK`, re-entrant since an abandoned wait cleans up wherever its coroutine is collected, and replaced in a forked child). Only the task's own frame reads it (`record.call_id`, set with `record.steerable` where the task is spawned), not agents nested inside it, and the root turn's `now` cut never reaches a child. Whether a child steers is resolved once per turn (`Agent._steers` → `_task_record`); a root turn pays nothing for it. The message is in the task's memory, never in the root agent's: an approval reviewer gets it as the `delegation` (the task's own reviewer) or not at all (its caller's), never as `user_messages`, whatever its source. Child span `metadata.task_messages`, usage `background:messages_delivered`. In-process like the tasks themselves.
- The log is peekable, not consume-once: the parent agent and a frontend can both watch one child. `record.log.subscribe(after=)` replays then streams live; `numbered=True` yields `(cursor, event)` with each event's own position (the gap marker as `(cursor the replay starts after, BACKGROUND_LOG_GAPPED)`), from the subscription's snapshot, so appends and trims meanwhile cannot shift them.
- `get_background_task` returns `{status, task_id, name, title, input, started_at, summary: {text, phase, pct, last_tool, tools_in_flight, event_count}, transcript_cursor}` (+ `result`/`error` when done). `summary.text` is capped — for briefing, not for dumping a build into context.
- `background_timeout` → store enforces wall-clock deadline; status `timed_out`. `background_stall_timeout` → no log events within window; status `stalled` (timer resets on every event).
- Cancel stops in-flight work: the Task is cancelled *and* the handler generator is closed (an async gen suspended at a yield would otherwise never run its `finally`), then `on_background_cancel` fires for work the loop can't reach.
- `wait_for_background(task_id, timeout=..., after=...)` blocks until terminal (no `after`) or until the log advances past `after` — app/frontends; does not ack completion notices.
- **Session events** (`BackgroundTaskStore.post_event` / `timbal.state.post_session_event(run_id, text, key=, source="app"|"client", announce=, create=False)`): nonblocking and thread-safe (a post from another thread hops onto the store's loop), returns a `SessionEvent` receipt (`state`: `pending` → `delivered` with `run_id` | `superseded`; `await event.settled()`). Any run id of a session with a store reaches it (also a run an owned session continued from before it had one); no store → `LookupError` unless `create=True` (registers one under `run_id`; only the turn chained on it finds it, so pass the latest run). Same `key` supersedes a pending event only. ≤ 4,000 chars. Every root turn drains pending events (not user messages — see Chat Sessions & Steering) with completion notices into one runtime message before its prompt (`format_session_notice`; byte-identical to the old notice without events), records `metadata.session_events` + usage `session_events:delivered` (+ a client event's screening records in `metadata.guardrails`); resumes of a paused run don't drain. `<session_event id source key task_id?>` blocks; client-source text is labelled information-not-instructions and its closing tag is neutralized.
- **Announce policy** (`when_idle` | `silent` | `interrupt`, per producer): `Tool(..., background_announce=)` for a child's completion, `announce=` on `post_event` / `post_session_event` / `notify` / `report_progress`; `None` = the channel's default. Decides whether an item *starts* a turn; every turn still receives everything pending. Text driver: defaults `when_idle`; `interrupt` acts as `when_idle` (no mid-run cut). Voice: events `when_idle`, completions `silent` unless `announce_background_results`; `interrupt` skips the quiet-line wait, cuts the reply in flight via `interrupt()` (heard-prefix kept), still blocked by closed/unstarted/parked run/other channel's turn, records `trigger.announce="interrupt"`. Client events can't interrupt (`PermissionError` / `not_allowed`). Only-silent work pending/running → the driver stops `idle`.
- **Progress** (`timbal.state.report_progress(text, *, announce=None)`): from inside a background child only (`RuntimeError` elsewhere; spawn paths set `current_background_task_id()`). Event keyed `progress:{task_id}` with `task_id` → newer report replaces an unread one; the task's completion supersedes pending reports, and a report arriving after it is superseded on arrival. Rendered with `task_id=` and a "still running, never present as finished" note. LiveKit async tools' `ctx.update()` model.
- **`SessionScheduler`** (`python/timbal/core/session_scheduler.py`): one per owning channel (text driver, voice call); owns the session's `BackgroundTaskStore` by weakref (`bind` raises if another channel owns it). Policy resolution (`pending_turn` → `interrupt`/`when_idle`/None, `may_need_turn`), event-driven `wait_ready(admit)`, atomic `take()` → `Followup`, `turn_started`/`turn_finished`. Owner's runs carry `RunContext._continuation_owner` (copied by `ensure_background_store`; no store allocated for agents that never spawn). Every other root turn → `admit_foreign` in `Runnable._stream` (async, before param resolution / `pre_hook`): owner idle → preempts (driver `stop_reason="preempted"`, pending notices ride the turn); owner busy or another admitted turn running → `on_busy="reject"` (`SessionBusyError`) or `"enqueue"` (waits; re-asks the store only if another channel took the session over). `release()` is deferred while turns admitted or queued on the scheduler remain (`releasing(store)`; then forgets the store if idle), so they still go one at a time and a new owner waits behind them; another channel's own run admitted there takes the session at once; a call calls `take_over(store)`, which from a releasing owner makes it the successor (store + head pass to it, then `work_arrived()`; a new driver is refused while one is in line). `RunContext._foreign_turn_of` holds the admitting scheduler strongly until the turn's `finally`, so a dropped channel still completes its release. `continuation_for_run` skips a stopped driver still holding it. Session data loads after admission, so a queued turn reads what the turn it waited for saved. Admitted turns leave via `RunContext._foreign_turn_of` in `_stream`'s `finally`. A store records the parent it was created/adopted under when that run had no store (`_joined_from`, via `_register_run`); a root turn with no store whose parent is such a run of an *owned* store (`store_joined_from`, scans the `_OWNED_STORES` WeakSet) is admitted too and adopts the store on admission (unowned stores: nothing changes); `post_session_event`, `continuation_for_run`, a new driver's first run and a call's `_session_store` look there too. A new driver gives way only to another driver (or a call that ended and is releasing): on a live call's session its first run fails before admission. While one runs the owner starts nothing; a voice caller turn is busy from its commit (a cut is `holding()` until then) and waits only for *running* other turns (queued ones wait for it). `scheduler.head` = the session's latest root run (owner runs via `note_run`, admitted turns): an admitted turn with an older `parent_id` (queued, or a stale client) is re-chained onto it before it reads history/session data (`RunContext._continue_from`, root `metadata.requested_parent_id`; its START still shows the requested parent), and the call's next turn chains onto a newer head (turn one too: the seeded context is re-chained). Voice scheduler: `can_interrupt=True` — an `interrupt` item goes ahead of *queued* (never running) other turns, and `holding()` keeps them out from the cut until its turn is busy; `on_work` — the store calls `work_arrived()` on every new completion/event, so work posted between turns (`report_progress`, `post_session_event`) starts the call's watcher. `forget_idle_background_store(store)` drops its run-id map entries when the driver/call ends with nothing left.
- **Text + voice in one session** (`ChatSession` + `VoiceSession` on one conversation, either opened first): while the call is live it owns the session — events, completions and progress are spoken — and the chat answers only what is typed, as chat turns queued with the call's (its turns are admitted on the call's scheduler even with `on_busy="reject"`; the call's next turn sees them in history). Hung up: the session goes back to the chat with the call's head. **A question waiting** (a run that ended on an approval / `ask_user`) keeps the session with the channel that asked it: any other turn chained onto that run would be taken for its answer (prompt dropped, gated tools asked again). `SessionScheduler.asking` = `head_parked` and the head is the channel's own question (`_own_head`: its latest run, a question it took over, or, for a call, the run that answered its question over HTTP and asked another); `head_parked` is set for a channel's own runs too (`turn_finished(parked)`: the chat passes how its turn's run ended, the call its turn's root status; `note_parked` for a chat opened or restored on a question). A chat that is asking does not lend (`_maybe_lend`); a call hands the session back when a turn of the lent chat's ends on a question (`_return_to_asker`, from `foreign_finished`) and is lent it again once the answer is in. A turn that brings no `resume` values queues in `admit_foreign` (`_holds`): the chat's behind any parked head, whoever asked, as when the session is its own; a call's (any run of a channel that `borrows`, also a call with no store of its own that first meets the chat with that turn: it becomes the `_borrower` once admitted) behind the chat's own question, at most its turn deadline: past it the caller hears the timeout line and what they said is not kept. Other clients' turns are admitted as before, and a run another client left parked a call takes over as a call alone does. While lent the chat reads the session's head from the call (`session_head` / `session_parked`: `ChatSession.parked`, the worker's parked-head wait; the call's `_notify` wakes it) and waits for the call's turns before it takes what was typed (`other_turn_first` counts the call's turns: a turn that queued at admission has started and cannot put its work back). Hand-overs carry the question: a chat that closes passes the session to the call waiting for it (`release` makes the `_borrower` its successor) with `head_parked`, so the call's events wait (`run_awaiting_caller`) and its caller is asked again; a call that hangs up on its own question hands it to the chat (`_succeed` sets `_own_head`), which takes it over (`ChatSession._take_question`: `parked`, listed by `state()`, answered by `resume()`, dropped by `interrupt()` or by an `interrupt` message already waiting; log record `parked`) instead of holding everything typed behind a run only the phone could resume. A run whose stored trace holds no question (not parked, or nothing of it stored where the chat reads) is not waited for (`SessionScheduler.not_waiting`, log `chat_session_question_missing`): no turn of the chat can be taken for its answer, since the turn reads the same trace. It is read once more right before each turn the chat starts on that run (`_dropped_question_holds`), because a run's end can be stored late (a failed save sent again): found parked then, it is the chat's question after all (and the session comes back from a call that has it). A read that fails is neither "no question" nor "a question": the chat's turn does not start and the chat asks again (0.5 s doubling, ≤ 30 s, sooner when something arrives; log `chat_session_question_unreadable`), and nothing else changes: before the take-over the chat keeps waiting on the question, on a re-read a call that has the session keeps it and nothing is marked parked. Both channels are assumed to read one trace store: a call whose runs are stored elsewhere shares no history with the chat, and a question the chat dropped as not found may still be parked for the next call on that store. The re-read is the chat's and process-local (`_dropped_question` is not in the snapshot): if the run is stored late, a call lent the session meanwhile starts its event turn on it without looking, and so does a durable session restored in between. A chat opened on the call's own question leaves the session with the call and may answer it. One chat per session: a second one on a session lent to a call is refused (`join`, `RuntimeError`; HTTP 409), as on a session a chat owns, until the first is closed or reaped. The chat's own store is found by the run it continues from the moment the session object exists (`session_continues_from`; before, from its first turn), so a call (`VoiceSession._session_store` looks up a session continuing its last run too; one continuing an older run of the call is a branch, and is left alone) or a plain client continuing that run asks the chat first, also a chat restored on a question: plain `agent(parent_id=<that run>)` turns then chain into the session instead of branching. While lent the chat runs no wake (the application's turn, not something typed: it runs once the session is back), except a recovery turn, which goes first and right after a turn already ahead of it. A call that ends before it was lent the session stops waiting for it (`release` clears the chat's `_borrower`). An admitted turn leaves the session however it ends (`foreign_finished` in a `finally` of `Runnable._stream`; a last save cut by an interrupt is made once more, shielded, before the next turn is let in, since that turn reads it): one interrupted during its last save used to stay counted as running, and no turn of that session started again. Known gaps: a chat turn that had already started (the instant between taking its message and being admitted) when the call's turn ahead of it ended on a question stays held if the caller then hangs up, until that question is answered or the turn is interrupted; a call that continues a run parked before the call began, with no chat on it, still starts event turns on it (it does not know the run's status). Mechanics in `SessionScheduler`: `lends` (chat) / `borrows` (call) / `user_messages=False` (call: leaves `source="user"` events to the chat it shares the session with; with no chat there it answers them itself, as a call alone always did: `takes_user_messages`; a durable chat closed with `keep` marks the messages it leaves unanswered (`SessionEvent.kept_for` = its session id, process-local): they are in its snapshot and come back with it, so no other channel takes them meanwhile (the call, or another chat opened on the conversation, would answer them a second time), while what is sent afterwards is answered as usual (`pending_user_batch(store)` is the batch a turn starts with, for `take_followup` and for the chat's collect window alike: a kept message ahead in the store does not end the wait for the collected ones behind it); `from_store` clears the mark as it takes them back (`restore_event`), and a session only ever marks unmarked messages and lists, stores, withdraws and takes back its own (`ChatSession._mine`), so a second durable chat on the conversation does not take over the first one's; from that close on, only what was pending when it let go of the task store is the closed session's (`_left`): its last snapshot is written after that point, and what reaches the call meanwhile (a user message it answers, an application's event it speaks) is neither stored with the chat nor delivered again on restore; a restore that fails marks again only the messages it had taken back, not one sent to the call since (`_only_keep`). Known gap, pinned by a strict xfail in `test_chat_and_call.py`: an application event that was pending at that close is in the chat's snapshot, and if the call speaks it before the chat is restored, the restored chat queues it again (the closed session no longer follows what the call takes; the same holds for a keyed event superseded meanwhile, and for two durable chats on one conversation, which both store it). A call and a durable chat together are not durable, see below; a chat opened on a live call that already holds a user message starts its worker for it; `take_followup(users=, others=)`). The same ownership holds inside a turn (`SessionScheduler.takes`, asked by `Agent._inject_background_completions` and `_take_steering` through the run's `_continuation_owner`): a chat turn that runs while the call has the session takes no finished-task notice and no application event, at its start or between tool batches, only what the user wrote; they stay pending for the call to speak. And a call's turn that steers (`Agent(steering=True)`) takes the events posted for it, not what the user wrote while a chat shares the session, whether the call has the session or its turn was admitted by a chat that has it (`takes(event, store)` looks at the store's owner). A call's `take_over` of a chat-owned store sets `_borrower` and lends it (`_maybe_lend`) once the chat is idle (nothing running or queued there; until then the call's turns are admitted by the chat as before). The chat stays `lent` (`store` set, not owner: `pending_turn` counts only user messages, no completions/tasks) and sits in the call's `_successors`; the call's `work_arrived` wakes it. `release()` (also not deferred) hands the store back via `_hand_back` → `_succeed`. A chat opened on a live call's store `join`s it lent from the start (no `RuntimeError`); a chat closed while lent drops out of `_successors`. Other channels are unchanged: an `AgentContinuation` driver neither lends nor borrows. Not durable: a restart mid-call restores the chat at its own last head. Tests: `python/tests/voice/test_chat_and_call.py`.
- **Automatic follow-up turns** (opt-in, `python/timbal/core/continuation.py`): `agent.stream_until_idle(prompt=..., max_idle_secs=300, max_continuations=10)` / `AgentContinuation(agent, admit=..., on_busy="reject"|"enqueue", **agent_kwargs)` streams a turn, then starts a fresh run (own `run_id`, `parent_run_id` = previous) when the scheduler says a turn is due. `take_followup(store)` takes notices + events; the follow-up's `prompt` is the notice (runtime user message opening with an own-turn line, so it is in the stored `input`), root span `metadata.trigger = {kind: "background_completion"|"session_event", continuation_index, task_ids, event_ids?, announce?}`. Receipts settle at the run's START; `Followup.release()` hands work back if the run never starts. `admit` (sync/async, parameterless) gates each follow-up. `stop()`, `continuation_for_run(run_id)`, `stop_reason`, `last_status`. Never cancels children; process-local. Design: `rfcs/session-events.md`
- **Voice event turns** (`VoiceSession`, not the text driver — voice turns must run through `_run_turn`): `await session.notify(text, key=, source=, announce=)` → `SessionEvent`. A watcher (`scheduler.wait_ready`) starts an event turn (`_run_turn(followup=...)`: notice as prompt, no fillers, `TurnMetrics.event=True`, root `trigger`, client gets `EventTurnStarted{task_ids, event_ids, keys, urgent}` → `event_turn_started`) once the line is quiet (`_event_turn_blocker`: no other channel's turn, no turn, playback done, no greeting/idle prompt/held fragment, 0.6 s caller quiet, VAD silent, the run the next turn chains onto not parked — the call's in-flight run once it asked the caller (an `interrupt` re-checks after its cut), its last run, or a newer head from another channel via `scheduler.head_parked`, set from the admitted turn's root status in `foreign_finished`; a turn that asks its caller makes the call's store, so the HTTP resume is admitted and becomes the head). A call continuing a conversation owns and watches its existing store from `run()` on, with `scheduler.head` starting at the run it continues (a stale chat admitted before turn one re-chains onto it). A caller who speaks first gets pending events with their turn. `source="client"` needs `client_events=True` + passes the agent's input guardrails (block → `PermissionError`, log `voice_client_event_blocked`). Server: `VoiceConfig.events = {announce_background_results, client_notify, on_busy}` is server-config only (not in `CLIENT_SETTABLE_VOICE_FIELDS`); WebRTC/LiveKit `notify` frames (`announce`; legacy `policy: "when_idle"`) → `notify_ack` receipt or `rejected` with reason (`not_enabled`/`blocked`/`invalid`/`closed`/`not_allowed`); LiveKit frames with no participant (server `SendData`) are app events. `current_voice_session()` for tools and their tasks.

---

## Chat Sessions & Steering

A text conversation that stays open while its turns run (`python/timbal/core/chat_session.py`): messages arrive at any time, turns run one at a time, every event goes to one log.

```python
from timbal.core import ChatSession

async with ChatSession(agent, busy_mode="steer", collect_window=1.0) as session:
    session.send("refactor the parser")                   # idle: starts a turn
    session.send("also keep the old API", mode="steer")   # joins the running turn after its tool batch
    session.send("then update the docs", mode="followup") # its own turn afterwards
    async for item in session.events(after=0):            # every turn's events + {"type": "SESSION", "kind": ...}
        ...
```

- **Modes** while a turn runs (`busy_mode` when none is given): `steer` (lane `"next"`: the running turn takes it after its current tool batch, and does not end before answering it; messages a turn ends without taking start the next one together), `now` (steer without waiting — see below), `followup` (a turn of its own, one per message), `collect` (one later turn for all collected messages once none arrived for `collect_window` s), `interrupt` (cancels the running turn — it ends `cancelled`/`interrupted`, keeping its partial memory (in the stored dump too: `_salvage_interrupted_llm_output` appends to `_memory_dump`, and a call closed by `GeneratorExit` dumps its output; only a model call in flight is salvaged, `_llm_memory_saved` being False from its start until its output is in memory: a turn left between two steps, cancelled or failed while the next step's tools are resolved, keeps its history as it stands instead of getting the last model output a second time) — and goes ahead of everything queued). Idle: any mode starts a turn.
- **`now`** (same turn, unlike `interrupt`): `ChatSession` sets the turn's `RunContext._cut`. The steering root agent then (a) cuts the answer being streamed, at its next chunk or, when the model is silent (a stalled provider, a slow next token), without waiting for one: the model's stream is read by a task of its own, one event at a time (`_CuttableModelEvents`: the stream never runs ahead of the agent, and keeps its task and context between events), the agent waits for that read and for the cut at once, and a cut that comes first cancels the read where it waits, so the call ends interrupted with what it had produced on its span; a cancellation of the turn itself while it reads is passed on to the stream, as before, unless the read had already ended and its event was on its way to the agent: that cancellation never reached the stream, so it is re-raised (returning the event would take it away, and an interrupted turn would run the tools the model had just asked for); runs that cannot be cut (`_cut` unset) iterate the stream directly — its visible text is kept in memory as an assistant message (`stop_reason="interrupted"`, `metadata.interrupted="steer_now"`; dropped when output rails buffer, scrubbed when they redact, removed again if nothing was steered in so history never ends assistant-final) — or (b) cancels the tool calls in flight that allow it (`Tool.interruptible`; `None` follows `read_only`, so readers and read-only MCP tools are cut and anything that acts runs to the end). A cut call returns `[Interrupted] …` (status `cancelled`/`steered`, emitted on the tool's own span so clients close its card); a batch with a cut runs its tools as tasks (no single-tool fast path) and is cut once. It waits `NOW_COALESCE_SECS` (0.25) before draining, so a burst of `now` messages is one cut; draining clears the cut. Root `metadata.steer_now = [{cut: "answer", kept_chars} | {cut: "tool", tool}]` (`kept_chars` is what is in memory: 0 when the partial answer was removed again). Without steering (`Agent.steering=False`) `now` is just a queued message. Known gap: a model that streams nothing for a while (hidden reasoning) is only cut at its next chunk
- **`Agent.steering`** (`bool | None`, default `None` = on for runs a `ChatSession` starts via `RunContext._steering`, off otherwise; never for a turn whose channel sets `_steering=False` (a wake), even with `True`; root agents only, except that an agent detached as a background task reads the messages sent to that task the same way unless `steering=False`, see Background Tasks): at each step boundary — after a tool batch, and before a final answer would end the turn — the agent drains `lane="next"` events. Each user message joins as its own user message (`metadata.kind="user_message"`, `steered=True`; it restores the `max_iter` budget, and the tools a `max_consecutive_denials` stop withheld for that step; a step repeated without one — an answer cut by a message a guardrail dropped, an application event that joined — stays tool-less) after input guardrails screen it: a redact rewrites it, a block drops just that message for a runtime notice (`kind="steer_blocked"`) and the turn and its work go on. What counts is what joined memory, not what was taken (`steered` / `steered_user` in the loop): a message a rail redacts down to nothing joins nothing, so it neither keeps the turn open (no model call on a history that ends with the answer just given) nor restores the budget and the tools; an answer it cut is taken back and given again. App/client events join as a runtime notice. Root span `metadata.steered_events`. The turn-start drain (`_inject_background_completions`) never takes user messages.
- **Inbox**: user messages are `SessionEvent`s with `source="user"` and `mode=` (and `sent_by` when a running call's code posted it: not a person's words for an approval reviewer, see `approval_reviewer`) (`post_event(..., source="user", mode="steer"|"followup"|"collect")`; no `key`, no 4,000-char cap); app/client events take `lane="next"|"later"` (default `"later"`). `post_session_event(..., lane=, mode=)` too. `SessionEvent.on_settle(cb)` fires synchronously on delivery, so a channel records it before the run's next event. `take_followup` starts with `user_batch()` (interrupting messages first; else one `followup`, or the leading run of `steer`/`collect`) as a plain user prompt (`trigger.kind="user_message"`) and leaves other events for that turn's start.
- **Channel**: owns the session's `BackgroundTaskStore` through a `SessionScheduler(on_busy="enqueue", can_interrupt=True)`; each turn runs on its own task (`_pump`) with a `RunContext` it builds (store adopted, `mark_own`), so `interrupt()` cancels just the turn. Finished background tasks and session events start turns like `AgentContinuation`. Log records: `message`, `message_settled`, `turn_started` (with `trigger`), `turn_finished` (`status`, `parked`), `unparked`, `resume_dropped`, `turn_failed` (`error`, `attempt`), `restore_failed`, `interrupt`, `task_message` / `task_message_settled` (`ChatSession.message_task`: the message, and its receipt's outcome), `closed`. A turn that fails before it starts (no OUTPUT) puts its work back and is retried after 0.5 s, then 1 s; after 3 failures in a row only new work retries it (work the failed attempt did not have, compared by id: handing its own work back does not count). Needs a tracing provider; process-local unless durable (below).
- **Parked turns**: a turn that ends on an approval / `ask_user` parks the session: messages wait and `resume(values)` answers it (re-sending the parked turn's prompt, which memory resolution drops). Answers sent before the worker takes them are merged (an id answered differently → `ValueError`, HTTP 409); while a turn runs `resume()` raises (`RuntimeError`, 409). Another client resuming the parked run also answers it: `parked` holds only while the scheduler head is that run (log `unparked`), and answers still queued for it are dropped (`resume_dropped`). `interrupt()` (also a message sent as `interrupt`) resumes with `Cancel` for every pending id not answered yet — those tools never run, answers already given stand, the interrupting message goes next (log `interrupt.abandoned`). A turn that has ended on a question and is not settled yet (its trace is still being stored: `busy`, not `parked`) takes the stop as well: the `Cancel`s are queued and stored when it is taken (`_cancel_pending`, read from the turn's own context), and the worker resumes with them once the turn is settled, or the process that restores the session does; `interrupt()` is false for an ended turn that left no question. A stop on a running turn is stored with it (`in_flight.stopped`): if the session is parked once that turn is settled, the question is abandoned too — the cut reached the run in its last save, after it had ended on the question (no OUTPUT arrives then: `_pump` takes the outcome from the root span, `_ended_output`, logs it, and the save is made again), or it cut a resume before it used its answers; `_recover` does the same for a finished, parked run. A turn cancelled before its task's first step hands its work back (`_run_turn`).
- **Continuing a run** (`ChatSession(agent, parent_id=...)`): `await session.restore()` (`async with`, the worker's first step, `POST /chat/sessions`) reads the parent with `TracingProvider.get_strict` (`None` only for a run not stored; the agent's `TRACING_UNSET` is resolved first). Parked → the session starts parked (pending ids from the parent's trace, its prompt kept for the resume). A parked wake keeps its options from its prompt's metadata (`tool_policy`, `silent`, no steering); a callable policy is not stored, so each agent notes on its span the tool paths its latest resolution allowed (`metadata.tool_policy_allowed`), and the restore allows exactly those, per agent by its latest span (a permission taken back stays taken back; an agent with no resolution in the run gets none). A failed lookup leaves restore unresolved: `restore()` raises (`POST /chat/sessions` → 503), the worker retries with backoff (0.5 s doubling, ≤ 30 s) and admits no message until it knows; an `interrupt()` or `resume()` sent meanwhile is applied once it does (`resume_dropped` if the parent was not parked).
- **Wakes** (`session.wake(text, key=, tool_policy="read_only", silent_ok=True)` → wake id): a turn for the application (routine, heartbeat), from its own queue: it runs only when no user message or event turn is due, never folds into a user's turn (a wake turn does not steer: a message sent during it waits for its own turn, with full tools and no silence token), and a pending wake with the same `key` is replaced (`wake_superseded`). Prompt is a runtime user message (`kind="wake"`; trusted, so input guardrails skip it); `trigger.kind="wake"`. A resume of a parked wake keeps its options. The worker races `scheduler.wait_ready` against new work, so a wake is not held back by a running background task; one held behind another client's turn of the session starts when that turn ends (`_until_scheduler_or_work`: the scheduler's `changed()` future is taken in the step that read its state, so a change in between is not missed).
- **Tool policy** (`RunContext._tool_policy`, set per turn by the channel, or per run by `context.tool_policy` on `POST /run` / `/stream`: `"read_only"`, a list of names (`[]`: none) or `null`; else 400; an agent started inside a call of the run, also a background one after the run ended, inherits it (`Runnable._stream`'s fork); each agent records `metadata.tool_policy` / `tool_policy_allowed` on its span, the root span `metadata.tool_policy`, and under a callable an agent of a run started inside a call also notes its allowed paths on that outermost call's span (`nested_tool_policy_allowed`, `RunContext._tool_policy_origin`) so a parked wake restores them; agents also stamp `tool_policy_decided_at` (ns; nested records `{allowed, at}`) and the restore keeps each agent's latest *decision*, not its latest-started span; an HTTP resume (`resume` + `parent_id`) without `tool_policy` keeps the paused run's (`server/http.py` `_paused_run_policy`, `get_strict`: a failed read → `503` + `Retry-After` before any job); helpers in `core/tool_policy.py`; advertised by `GET /meta` `capabilities` and `X-Timbal-Capabilities` on `/healthcheck`, confirmed by `X-Timbal-Tool-Policy` on the response; `context.require_new: true` makes any id the job store still knows, finished included, a 409): `"read_only"` (tools with `Tool.read_only=True`, MCP tools whose server says `readOnlyHint`, the agent's own readers `read_skill` / `read_offloaded` / `get_|list_background_task(s)` / `read_background_transcript`), a collection of names, or a callable. `Agent._resolve_tools` filters what the model sees — every agent in the run, subagents included — and a hidden tool the model calls anyway gets the usual `tool_not_found` result.
- **Silent turns**: turns the runtime starts (events, completions, wakes with `silent_ok`; `allow_silence=True`) get a note asking for exactly `SILENT_REPLY` (`"NO_REPLY"`) when nothing needs the user, and `RunContext._silence_token`; a root agent whose final answer is exactly the token sets `metadata.silent`, and `turn_finished.silent` is true. User turns never get the token.
- **HTTP** (`python/timbal/server/chat.py`, Agent runnables only): `GET /chat/` (playground: timeline with steer/queued badges, tool cards, approval/interaction cards — an answer refused with 409 before the turn parks is re-sent when it does, other failures re-enable the card — raw event log), `GET /chat/meta`, `POST /chat/sessions` (`{parent_id?, busy_mode?, collect_window?}`), `GET|DELETE /chat/sessions/{id}`, `POST .../messages` (`{text, mode?}` → 202 receipt), `POST .../interrupt`, `POST .../resume` (`{resume: {...}}`), `POST .../wake` (`{text, key?, tool_policy?: "read_only" (default) | [names] | null, silent_ok?}`), `GET .../events?after=` (SSE, `id:` = the event's own cursor, `Last-Event-ID` honored, keepalive comments, `event: gapped`), `GET .../log?after=&wait_ms=` (JSON long-poll). Idle sessions are reaped after `TIMBAL_CHAT_SESSION_TTL_S` (default 3600).
- **Durable sessions** (`ChatSession(session_store=)`, `python/timbal/state/chat_store.py`): a `SessionSnapshot` per session — head, turns, last status, parked, the last prompt/options (`tool_policy`, `silence`, `steer`: a resume re-sends them, so a restored parked wake still takes no steered messages), `pending` events (`SessionEvent.to_dict`, ids and `posted_at` kept), wakes, unanswered `resume` values (`Cancel` as `{"$cancel": reason}`, an `ApprovalResolution` as the JSON object the approval gate accepts, every field kept; anything else must survive a JSON round trip unchanged or `resume()` raises `TypeError` before queuing it, HTTP 400 — `SessionSnapshot.to_json` has no `default=` and `allow_nan=False`, so a value JSON cannot carry, `NaN` and `Infinity` included, fails its save instead of being stored as its `str()`; a typed approval is checked on its own field values, since pydantic's JSON mode would write `NaN` as `null`; every check ahead of a save is `chat_store.storable(value)`, the encoder `to_json` writes with, so a check and the write cannot disagree — turn arguments it refuses are named in `kwargs_unstored`, a last prompt it refuses is kept as its text, and `allow_silence` / `collect_window` / a wake's `silent_ok` are stored as `bool` / `float`), `in_flight` (the running turn: `run_id`, `parent_id`, `trigger`, the events it took, its wake, its `tool_policy`, `stopped` when a stop reached it), `log_cursor` — saved ahead of every change (`_record` kinds `message`, `message_settled`, `wake*`, `turn_started`, `turn_finished`, `interrupt`, `closed`, plus `resume()`), one save at a time, bursts coalesced; `await session.flush()` waits (routes call it before answering, so a `202` survives a restart; a route waits at most `TIMBAL_CHAT_STORE_TIMEOUT_S`, default 10 s, `server/chat.py` `_store_timeout`, for a store that hangs instead of failing: past it `_stored` answers `503` — without `Retry-After` and with the change kept when `ChatSession.may_be_stored(mark)` says the write that is out holds it (`_saving`: it may still land, so a retry could run it twice), with `Retry-After` and the change taken back when no write holds it; `close(store_timeout=)` gives the store that long to settle and then closes without it (`close_timed_out`; `close_error` when not kept, so `DELETE` answers `503`; the save loop is cancelled), restores (`_restore`), reaping, startup restores and shutdown use it too, the last three waiting once and not once per session; a session that lost ownership closes without waiting for the store at all; `SqliteChatSessionStore` runs its calls on one daemon thread per file (`chat_store._StoreThread`, ended after 30 s idle, forgotten in a forked child), never on the loop's shared pool: callers that stop waiting for a hung store would leave a stuck thread each there, until nothing else got one, and hold the process open at exit; a call whose caller stopped waiting before it began is not run; `sqlite.database()` opens a file outside the registry lock every file goes through, so one that cannot be opened holds up only its own callers, and keeps the connection it opened unless another thread stored a live one meanwhile (`_Database.alive()` compares the file's device and inode with the one it opened, not just whether a file is at the path: one deleted and written again is another file); a lease renewal that gets no answer within a third of the lease, or within what is left of it, counts as one that failed and is not waited for any longer, so ownership is given up when the lease runs out, not when a renewal comes back) and raises `SessionNotStored` when the store failed (after one retry) or `SessionLeased` when ownership was lost; routes (`_stored`) answer `409`, or decide with `session.mark()` / `stored_since(mark)` (a change counter: every `_persist` counts, a good write records how many it covers): the route's own change is stored and a later one failed → the acknowledgement holds; the store holds nothing of it → the route takes it back (`ChatSession.withdraw(event_or_wake_id)` — a withdrawn wake puts back the one of its key it replaced — / `withdraw_answers(values)`, events settle `withdrawn`) and answers `503` with `Retry-After`; a turn has it → `503` without. Routes send with `send(..., defer_cut=True)` and call `cut_for(event, mode)` after the write, so a `now` / `interrupt` message cuts nothing before it is stored (and never the turn that took it); a restored parked session with a pending `interrupt` message abandons the question (`from_store`). A restore honours a stop stored with the turn: a resume cut before any of its calls ran comes back with the question abandoned, not with its answers replayed (`in_flight.stopped`; live, settling the cut turn does the same). A stopped turn that ran no tool call is not queued again either: its messages, read as the user's latest words behind the stop, would ask for the gated tool again, and its wake would redo what was stopped. They go to the next turn as a silent notice of what the user stopped (`_stopped_notice`; recovery wakes a restore creates are not affected). The stop counts as stored when only its `interrupt` message is: queued behind the dead turn, it was sent to stop it (`_interrupts_waiting`). A recovery turn is read-only *and* no wider than the dead turn's own policy (`read_only_within`, `core/tool_policy.py`: `ReadOnlyWithin`, stored as `{"read_only_within": [...], "by": "name"|"path"}`; a callable's restores from the dead run's trace, else nothing; a snapshot from before `in_flight.tool_policy` existed falls back to the turn's wake, then to `last_options`, never to plain read-only; narrowed or not, its prompt carries the read-only note). `session_store_for` refuses `TIMBAL_CHAT_STORE=sqlite://…` for an agent whose traces are process-local (`TracingProvider.durable`): the snapshot would outlive the runs it points at. That, like a value it does not understand, does not stop the server (`_choose_store` keeps it as `app.state.chat_store_error`, log `chat_store_misconfigured`): `/run` and the rest work, every session route answers `503` with the reason and no `Retry-After` (never served without a store instead), `/chat/meta` reports `store_error`; the lifespan restores inside its `try`, so a startup that fails there still runs its shutdown half. A session the reaper may let go is one that is neither busy nor `due` (`ChatSession.due`: queued answers, a wake, or `SessionScheduler.may_need_turn()` — a message (one still collecting included), an event or completion whose announce policy starts a turn, a background task still running whose result will. A silent event or silent task is not due: it starts nothing and rides with the next turn. A parked question is not due either, it waits for its user). `close()` sets `close_error` when it could neither store `closed` nor delete the row (`DELETE` → `503`). **Write-ahead of turns** (`_pump` → `_write_ahead(mark, wait=)`): a turn's input (`in_flight`: its events, wake, resume answers) is stored when the scheduler takes it, and the agent is not called until that write succeeded — a failed one raises `SessionNotStored` in the turn (`turn_failed`, `reason="not_stored"`; work handed back, answers to a parked turn included; `_store_failures` backs the worker off up to 30 s and retries without new work). `in_flight.dispatched` lists the tool calls the turn ran (`{id, name, input}`): every agent of the run awaits `RunContext._before_tool_dispatch(agent_path, [(tool_use_id, name, input)])` right before it runs tool calls (a model's batch, a slash command, the calls a resume left pending — `Agent`, three sites), and the session (`_note_dispatch`, root agent's calls only: a sub-agent's are covered by the call that runs it) stores them first, waiting for the store (`wait=True`). Told by the agent, not inferred from events, so it holds for any tool name and the task-per-call path. A message steered into the running turn is appended to `in_flight.events` (`steered: true`, `_settled`). Events queued through the task store (`store.post_event`: app, client, silent ones too) are persisted from `_work_arrived` (not task progress, not while `from_store` holds the worker). Turn arguments (`ChatSession(..., **kwargs)`) are stored in `config.kwargs`; those that are not JSON are named in `config.kwargs_unstored` and `from_store` raises `ValueError` unless they are passed again. `_renew_lease` keeps an absolute deadline: the lease counts from when the request that last confirmed it was sent (`_lease_from`: the creating save, `from_store`'s acquire, a renewal — the store counted from no earlier), renewals are due a third of a lease after that (at once after a restore that took longer) and again a tenth after a failed one, and the session is `lost` — turn cancelled, closed `keep=True`, nothing written — at `_LEASE_MARGIN` (a tenth) of a lease before its end, whether a renewal is still out or not; so the old owner has stopped by the time the store can give the session to another (a renewal that lands after a stall still counts from when it was sent; one refused, `SessionLeased`, loses the session at once). A turn is held open until its trace is stored (`_run_turn` → `_trace_stored`, on the worker and not on the turn's task: an interrupt cancels that task and must not end the wait — only a close does, and the turn then stays in flight; `RunContext._trace_unsaved` is set by a failed write-through save and cleared by the next good one; the session retries `_save_trace(barrier=True)` with a backoff). That wait is a durable session's only (`ChatSession._holds_turns_for_their_trace`): without a `session_store` a refused save is best effort as for any run — logged (`chat_session_trace_not_stored`, `waiting=False`), counted in `tracing:save_failed`, and the turn is settled, so a platform answering 403 for good does not leave the session busy with every later message queued. `busy` is true until `_run_turn` has settled the turn (status, parked), not only while its task runs: `state()` never shows an idle session without the outcome of the turn it just ran. Saves are compare-and-swap on `version` and, with `owner=` (the session passes `PROCESS_OWNER`), refuse while another owner holds the lease, expired or not (`SnapshotConflict`): an owner whose lease lapsed and was taken cannot write before the new one does; a lease (`process_owner()` = host:pid:uuid, resolved per pid at use time so forked workers do not share the parent's — `chat_store.PROCESS_OWNER` still reads it, but importing the name freezes it; `lease_ttl_s` 30, renewed every 10 s; a new row is inserted with its owner's lease in the same write, `save(..., expect_version=None, owner=)`) keeps one process per session (`SessionLeased`; a lease held by a dead process of this host is free at once, so a restart after `kill -9` does not wait it out); losing either closes the session `keep=True` without writing. Stores: `InMemoryChatSessionStore`, `SqliteChatSessionStore(path)` (table `chat_sessions` in the `SqliteTracingProvider` file, same connection via `sqlite.database()`); no platform store yet (needs `SessionRuntime`, monolith#553). `close()` deletes the row, conditionally (`delete(id, expect_version=, owner=)`: `SnapshotConflict` → `lost`, the other owner's row is left alone); `close(keep=True)` (server shutdown, idle reaping, lost ownership) keeps it and marks a turn it cuts `in_flight.cut="shutdown"` — only one still in flight: a turn that had ended when the worker was stopped, or that ends by itself all the same (`_ended_by_itself`: not `cancelled`/`interrupted`), is settled first (`_settle_turn`), and `_recover` adopts a run that ended by itself even if it is marked cut (a parked one keeps waiting for its answer). `ChatSession.from_store(agent, store, id)` (LookupError / ValueError for another agent's / SessionLeased; takes the lease *then* reads the row; any failure releases the lease and writes nothing (the release is waited for only while the lease lasts, `_answered_within`: a store that does not answer cannot hold a failed or cancelled restore for good); no worker starts until it returns; the dead turn is settled first, its events requeued ahead of the pending ones; its trace is read strictly — an unreadable run fails the restore instead of requeueing input that may have run tools): queued events back with their ids (restored in the process that closed it, the run's task store may still hold them: `from_store` takes them out with `drain_events(settle=False)` and queues them again from the snapshot, the events themselves via `restore_event(data, held=)` so their receipts settle, in the order another process would get: what a dead turn took ahead of what was queued behind it, a newer event with the same key standing; those the snapshot does not name go back after, those the dead run's memory shows are settled delivered; every one is followed by the restored session, and the closed one does nothing more: `_persist`, `_settled` and `_event_settled` return once `_close_done` is set, since a late write of it would carry the live session's lease owner), wakes, parked state from the head run's stored trace (a parked snapshot whose head run is not in the trace store, `chat_session_parked_run_missing`, is restored not parked: nothing can answer it, queued answers are dropped with `resume_dropped`; `state()` lists pending approvals; `resume()` works), the log restarted `_LOG_RESTART_GAP` past the saved cursor (an old cursor reads as a gap) with a `restored` record. The dead turn (`_recover`): root finished and not cut → adopted (head, status, parked), and the events its memory shows (`_shown_events`: `metadata.event_ids`) are not restored from the snapshot's `pending`; nothing `dispatched` (whatever is stored of it: a run cut during its model call by a shutdown stores a trace that need not hold its prompt) → its events (or wake) queued again, a resume's answers put back, head back on its parent; nothing stored but `dispatched` (a provider that saves at the end of a run, or buffered) → not run again: the recovery notice says what it was answering (`asked`: its events, steered ones too, or its wake), lists the recorded calls as unconfirmed, and that the rest was not recorded; if it was a resume, the parked question is abandoned (`_abandon_question`: a `Cancel` for what still waits, run by a resume turn before the recovery one) — never answered again, since claims protect a gated call but not the tool that asked a question nor what the model called next; with a trace, recorded calls it does not show are unconfirmed too (`_tool_calls(context, dispatched)`), a steered event its memory does not show (by `metadata.event_ids`, not text) is queued again, and a pending event it does show is not restored from an older snapshot; cut mid-run → head on it, and `on_dead_turn` = `"continue"` (default: a read-only recovery wake first in line, `kind="recovery"`, `trigger={kind: "recovery", dead_run_id, cut: "crash"|"shutdown"}`, silence allowed) / `"notify"` (the notice as a silent app event riding the next turn) / `"drop"`. The notice lists completed calls (an outcome stored: a `tool_result` in memory or a finished tool span) and unconfirmed ones (requested in a stored model call's output or memory, or a started span, with no outcome) and says to check before redoing — tools are never re-run automatically (at most once). A recovery wake runs before queued messages (the turn the user saw cut is settled first). A dead *resume* turn that stored nothing (`in_flight.resume` / `.approved`): see above — answers put back if it ran nothing, the question abandoned and a recovery turn naming the approved and recorded calls otherwise (the same with or without approval claims). `parked` clears when a resume turn starts. Several workers on one file: the worker not serving a session answers `409` (`SessionLeased`), never a second copy. The playground shows the `restored` record (tool cards of the cut turn → "cut by restart") and reconnects on its own. Tests: `python/tests/core/test_chat_session_crash_points.py` — the sweeps: 18 scenarios, each run once per write to the session store and the trace store with a fault at that write (the process dies; dies again during the recovery; shuts down in order; the session store or the trace store is away for 3 writes; the trace store goes away and the process dies or shuts down 1 or 3 writes into the outage; the user stops the session from inside that write; another process takes the session; the process stalls past its lease, `Clock` jumps `time.monotonic`), then restored as a new process, settled, and checked (no side-effect tool twice, no tool run after a stop of its question was acknowledged, nothing acknowledged lost, no event answered twice, settles idle, stored state matches, a second restore runs nothing, no task left). Add a scenario there for any new session feature; a bug found elsewhere should make a sweep fail first. `python/tests/core/test_chat_session_durable.py` (crash = provider stops writing + tasks cancelled, after a last flush: one point of that space), `server/test_chat_durable_http.py`, `server/test_chat_durable_e2e.py` (real server under `kill -9`, SIGTERM, 2 workers; `-m integration`: Haiku 5.5 / GPT-6 Luna recovering a cut charge). Callable tool policies are stored as `"custom"`: a parked turn resumes with the tool paths its agents recorded (`restore_tool_policy`), a queued wake with no tools (never `"read_only"`, which can be wider). Server: the store is `TIMBAL_CHAT_STORE` (`memory` off, `sqlite:///rel.db` / `sqlite:////abs.db`) else next to the agent's SQLite traces; an unknown id is restored on first use (one restore per id), and so is one this process handed off (`ChatSession.handed_off`: lost its lease, or closed with `keep`) — its closed object is dropped once `wait_closed()` returns; `DELETE` keeps the session in the map until its close is done (a request meanwhile is a `404`, not a restore of the row still there) and answers `409` when the session was `lost` to another process; a handed-off one is dropped once `wait_closed()` returns (never a `404`, never a second copy bound to the task store while the first still closes; idle reaping closes before it unmaps). `_restore_once` returns a session this process already serves (it can always renew its own lease: a second `from_store` would be a second copy) and its task (`_restore_into`) puts the session in the map itself, shielded from callers that go away; shutdown (`chat_closing`: no new restores) waits `_SHUTDOWN_RESTORE_S` for restores in flight, cancels the rest (lease let go, nothing written) and gives their cleanup the store's bound once (`_store_timeout()`; a restore still letting go of its lease then is cancelled again and left behind, `chat_session_restore_abandoned`, its lease to run out; `chat_restores_cut` keeps a restore that finishes after that from registering its session: `_restore_into` closes it `keep=True` instead), waits no longer than that bound for the cancelled discovery / retry tasks either, then closes every session; sessions with work (`has_work`) at startup in the lifespan (those skipped on a live-looking lease or a failed read are retried every third of a lease while they still have work, `_retry_restores`, cancelled at shutdown; the listing itself is best effort and bounded, `_STARTUP_DISCOVERY_S`: a store that cannot be read — the SQLite traces of a `/run`-only deployment on a mount that is away — never stops the server from starting, `chat_sessions_discovery_failed`, and `_discover_later` looks again after `_DISCOVERY_RETRY_S` until it answers; `SqliteChatSessionStore.list_ids` skips a row it cannot read, `chat_session_snapshot_unreadable`; `with_work=True` reads the `has_work` column every save writes, through the partial index `idx_chat_sessions_work` (`WHERE has_work IS NOT 0`), so idle rows are neither visited nor parsed, and parses only a row from before the column, where it is NULL; `_discover_later` and `_retry_restores` give each store call `_STARTUP_DISCOVERY_S`, and a session that gets no answer ends the round and goes last in the next), shutdown and reaping keep them; `/chat/meta.durable`, `state().durable`. Not durable: background children (process tasks), the event log itself (the turns are in the traces). On the platform, delta saves store a running root whole once (`_DeltaRun.root_stored`), so a turn cut mid-run there keeps its prompt.

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
 store=LocalOffloadStore(), # default root $TIMBAL_OFFLOAD_DIR or ~/.timbal/offload; keep-forever, opt-in cleanup_after=timedelta
 ),
 tools=[
 Tool(name="logs", handler=..., result_limit=ToolResultLimit(threshold=8_000, action=Truncate(strategy="tail"))),
 Tool(name="docs", handler=..., result_limit=None), # exempt
 ],
)
```

- Oversized results are reduced **once, when produced** — before entering memory/dumps — so history stays append-only (prompt-cache friendly) and the reduction persists into traces.
- `Spill` is lossless: payload goes to the store, a preview + handle stays inline, and a bounded `read_offloaded(handle, offset, limit, pattern)` tool is auto-registered for paged read-back. Falls back to `Truncate` when the store fails.
- Always exempt: error results, pinned tools (`pin_result=True`), `read_offloaded` itself. Precedence: `Tool.result_limit` > agent `tool_result_limit`.
- Offload events are recorded in `span.metadata["offload"]`; the handle lives on `ToolResultContent.offload_handle`.
- `attachment_limit=` (same `ToolResultLimit`/int config, default `None`) bounds file attachments that providers receive as pasted text (anything but images, PDFs, audio, `.eml`). Applied **per request**, not to memory (`python/timbal/core/attachment_limit.py`): memory and traces keep the `FileContent`, the model gets a deterministic stand-in (`Spill` saves the text once under a content hash in the shared store, readable via `read_offloaded`), and conversations that already hold an oversized file are fixed on their next call. Always on, even without it: the LLM router cuts a text attachment estimated to exceed the model's context window (> 4 chars per window token) to a 20,000-char preview.
- `read_offloaded` (earlier `read_tool_result`; calls to the old name still dispatch to it, unadvertised) reads every handle: tool results, attachments, compacted transcripts. It splits lines over 10,000 chars into numbered segments, so minified content is pageable.

### History compaction (utilization-triggered, whole memory)

`memory_compaction=` strategies fire at `memory_compaction_ratio` (default 0.75) of the context window: `compact_tool_results(keep_last_n, replacement, keep_offloaded=True)`, `keep_last_n_messages(n)`, `keep_last_n_turns(n)`, `summarize(...)`.

`summarize()` builds a sectioned summary message where everything except the LLM summary is mechanical: user messages carried **verbatim** (`preserve_user_messages=True`), a **canonical record** of each summarized region written to the offload store and readable via `read_offloaded` (`canonical_record=True`; store shared automatically from `tool_result_limit`), a conservative continuation note, and an optional `rehydrate=` callable re-run every pass. `compact_tool_results` keeps offloaded placeholders intact by default so their handles stay dereferenceable.

With `delivery="tool"`, what the user was told is conversation, not tool output. A successful `send_message` stores the text it delivered (after `model_output` redaction) on its result: tool span `metadata["delivered"]` becomes `ToolResultContent.delivered`, kept through `content_factory`. In both modes, `compact_tool_results` keeps such a call and its result as they are, like a pinned or an offloaded pair. Before this, the default drop mode deleted every message the agent had sent, and Haiku then told the user it had "never sent" results it had.
- Kept as a call, not turned into assistant text (what `feat/bots` did): the model's own text in those turns is a draft the user never read, and once both are plain text nothing tells what was said from what was not. Keeping the text next to a converted send showed the draft as part of the reply; dropping it lost an earlier message, which a second pass inside a turn finds folded into the same assistant message (`_join`). As calls, the standing note in the system prompt (only what `send_message` sends reaches the user) stays true of the compacted history.
- Only real deliveries carry the mark: the agent's own delivery tool, by name, when the call succeeded, and the result of a slash command whose text was delivered. Not a failed send, not another tool that sets `metadata["delivered"]` on its span, and not an app's own tool that happens to be called `send_message` (Slack, email).
- A result a guardrail rewrote says what went out (in its text and in `delivered`), next to the call that holds what the model wrote.
- Positional compactors are unchanged: they drop old turns the same way in either mode. `summarize()` sees the calls like any tool's; what the agent told the user is not carried verbatim the way user messages are.

The summarizer call sends no `temperature` (Claude Haiku 5.5 rejects it).

**Prepared in the background** (`Agent(memory_compaction_prepare_ratio=0.6)`, opt-in, must be in `(0, memory_compaction_ratio)`): so no turn waits for a summarizer call.
- **Mid-turn** (`_compaction_boundary`, at each step boundary): past the prepare ratio, the strategies run on a task over a copy of the history before the trailing assistant batch (`_Prepared`: the prefix messages themselves). A finished one is swapped in at the next boundary if those messages are still there by identity (`_join` folds an assistant seam), else discarded (`compaction:prepared_stale`); a swap ends that boundary (its measurement is stale). Reaching `memory_compaction_ratio` while one runs awaits it instead of compacting inline in parallel (`asyncio.wait`: one that failed or was cancelled is a preparation that came to nothing, and the turn compacts inline; only the turn's own cancellation ends the wait); with none running, inline as before. A pause or error cancels it. Strategies that are plain functions run on a worker thread when prepared in the background (`_on_worker_thread`), so they do not hold the loop for other turns. A thread cannot be cancelled: each such call is tracked until it returns (`_Prepared.workers`), and a prepared compaction dropped while one is still at work (a pause for approval, a cut, an error, a history that changed) stays registered under its run as a handoff that hands on nothing (`_drop_prepared`, `_Handoff.nothing`), so neither that turn nor the ones that resume or continue it start another background compaction next to it (a turn ending past the prepare ratio meanwhile keeps the hold for the next). No turn waits for it: one that starts at the hard ratio compacts inline at once, as it does at the hard ratio mid-turn.
- **Next turn**: a turn that ends past the prepare ratio (or with one still running) hands it on: a detached task (`run_context._live_background`, then `_end_background_work()` re-stores the ended run like a background child's late output) writes `session["__prepared_compaction"][agent_path] = {covered, fingerprint (sha256 of the stored dump of the covered prefix), memory (dumped), steps, utilization}`. `_PREPARING[(run_id, path)]` holds a `_Handoff` until the run is stored again with it: `record` resolves as soon as the result exists (what a turn in this process takes, whatever the store is doing), `stored` ends with the trace write (what a turn reading the trace finds). The next turn (`_take_prepared`, `resolve_memory`, non-resume, direct parent) takes a record that is there, waits for one still being made only when its history plus the incoming prompt does not fit (at `memory_compaction_ratio`, or unmeasurable), else reads the parent's session data; a turn that did not wait keeps the handoff (`RunContext._unwaited_handoffs`) and, if it reaches the hard ratio later, takes that one, waiting for it, instead of compacting the same history inline next to it (`_take_predecessors`; only while what the turn holds of the messages it loaded is still in place, first and in order — `resolve_memory` records what each loaded message became, since cleaning empty text blocks hands back another object or drops the message); while the turn before's is on its way, or held by a turn that did not wait for it (also once that run is stored and gone from `_PREPARING`), no other background compaction starts, and a turn ending past the prepare ratio takes its record on for the turn after (awaiting the record, not the store; also when it ends at the hard ratio, so the turn after waits for that record instead of compacting inline; a record that is already there and no longer matches this turn's history as stored is left, and the turn prepares its own); applies it when the fingerprint of the parent's stored memory prefix (dumped: in-memory providers hand back `Message`s) matches, drops the context measurement for that check, removes the key. Metadata `compaction = {background: true, prepared_by?}`, usage `compaction:background` / `compaction:prepared` / `compaction:prepared_stale`.
- Measured (Haiku 5.5, 8-turn chat, ratio 0.45% / prepare 0.25% of 1M): inline, the turn crossing the ratio waited 2.5 s for its first output (~0.6 s otherwise); prepared, every compaction landed in the background and no turn exceeded 0.8 s. Only helps when history grows gradually: one parallel tool batch that jumps past the ratio still compacts inline. Tests: `python/tests/core/test_background_compaction.py`.

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

### `emit(data)` — fire-and-forget custom DELTA events

```python
from timbal.state import emit  # no-op without a run context (unit tests)

get_run_context().emit({"kind": "ui-event", "event": "preview_ready", "url": "..."})
emit({"kind": "progress", "pct": 50})  # module-level convenience
```

Out-of-band event channel from anywhere inside a handler (including helpers deep in the call stack): wraps `data` in a `Custom` delta item (`item.type == "custom"`, `item.id` = current call id) and interleaves it with the current call's processed events. Foreground calls surface emissions on the parent event stream; detached background children surface them in their background log/transcript. Always ordered before that call's OUTPUT.

- Never passes through the collector — unlike a generator `yield`, it cannot alter the call's output or the persisted span.
- Thread-safe from `offload_blocking` / sync-generator executor threads (marshalled onto the owning loop).
- Fire-and-forget: no await, no backpressure, never raises; outside a run it's a silent no-op (`emit` module function) or log-and-drop (`RunContext.emit`).
- Zero cost when unused: the per-call sink is allocated on first `emit()`, not on every runnable invocation.
- Generator handlers: emits between yields surface at the next chunk boundary (or at completion, still before OUTPUT).
- Plain (non-generator) handlers: emits flush when the handler returns, still before OUTPUT. Not raced against the handler — Task-wrapping every tool/step would show up in the framework-overhead benches.

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

`dump()` caches each Message's serialized form (validated against `len(content)`). Reassigning a field (`msg.content = [...]`) drops the cache; editing a content block in place (`block.text = ...`) does not — call `invalidate_message_dump_caches(msg)` before the next dump, or serializing providers store the pre-edit text. Serializing providers store a span's `_memory_dump`, not `span.memory`: rewrite memory in place → re-dump it before saving.

### File

```python
from timbal.types.file import File

f = File.from_path("/path/to/doc.pdf")
f = File.from_url("https://...")
f = await File.from_upload(upload_obj)
```

Auto-detects MIME type. Serializes to base64 for LLM APIs.

Signed platform CDN URLs (`timbalusercontent.com` / `platform_config.cdn` with `Signature`/`Key-Pair-Id`) are stored verbatim by `persist()` and re-signed on read (`load()` and the sync lazy fetch) through `POST /orgs/{org_id}/content/sign` when the run has a platform subject. Per read: (1) a stored URL that may still serve (not past its `Expires` by more than 60 s, `_CLOCK_SKEW_GRACE_SECS`: the local clock may be ahead of the CDN's) is the first request, exactly as before, and any answer but a 403 (expired by the CDN's clock, revoked key) is final; (2) else a memoized mint, under the same test of its own `Expires`, dropped if it fails in any way (error status, or the fetch raises); (3) else one sign call and the minted URL; (4) a stored URL that step 1 skipped is fetched before the read fails. So a read never fails where a plain GET of the stored URL would have succeeded, and never waits on the sign endpoint while its URL can still serve. The sign call has a 10 s deadline (`total_timeout` on `_request`: the whole call, retries included, cancelled at the deadline; the sync path uses `_request_sync`, which runs `_request` on a helper thread and blocks the caller no longer than that; the helper thread ends with its request, and all helpers resolve DNS on one shared on-demand pool of at most 32 threads (`_SyncDnsPool`, `_SYNC_DNS_WORKERS`) which they never wait for: a stalled resolver can hold at most those threads, and none remain once it answers). A failed sign call raises the stored URL's error chained to the sign failure; a minted URL that fails to download raises its own error. Mints are memoized per (platform + auth, org, object path) only once they serve a download. Without a platform subject, or for unsigned URLs, reads are unchanged.

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
- A test must not be able to wait forever: every `await` on the network stack or another task gets a deadline (`test_voice_rtc.py` `_step`). `loop_watchdog` (`python/tests/conftest.py`, used by the durable-session suites) fails a test whose event loop ran no callback for `TIMBAL_TEST_LOOP_STALL_S` (10 s) with every thread's stack, and notes on the report when the whole process stood still instead (CI runners have stopped a test process for 70-260 s: a session waited on then looks stuck). Backstops: `faulthandler_timeout = 300` (`pyproject.toml`) prints every thread's stack for a test still running after 5 minutes, and the CI test job has `timeout-minutes: 60`
- No fixed sleeps or tick counts for "X has happened by now": wait for the state (an event, `until(...)`, `records(...)`), and install a fault before the step it must catch, not in a time window after it. `TIMBAL_TEST_SLOW_IO=0.08 uv run pytest <files>` (`python/tests/conftest.py`; `_JITTER=1`, `_SEED=`) delays every chat-session store read/write and SQLite/JSONL trace save, which is how a slow CI runner breaks such a test: run the session tests under it (also at 0.25 and with jitter) after changing them
- Tests run in parallel (pytest-xdist) in CI and with `-n auto` locally: each worker is its own process and every worker must collect the same test ids, so a `parametrize` id must not depend on the process (`ids=repr` of an `object()` or of a value holding a timestamp breaks collection; `test_chat_session_durable.py` `_stable_id`), and a test must not share files, ports or global state with another test except through its own `tmp_path`
- Clocks: before Python 3.13, Windows' `time.monotonic()` / `time.time()` and asyncio's timers tick every 15.6 ms, so two events can get the same timestamp. Never order events by comparing their timestamps strictly: stamp them so they cannot tie (`VoiceSession._event_stamp`, `agent._policy_decision_ns`) or use a counter. A test deadline is a hang guard with headroom (a scenario that takes 0.2 s here can take seconds on a busy Windows runner), never a timing check. To find this class of bug on any OS, run tests with the clock in coarse ticks (`test_end_call.py` `_session_clock` does it for the voice session)
- CI (`.github/workflows/ci.yml`, planned by `scripts/ci_plan.py`) has two lanes. **Full**: every OS × Python 3.11–3.13, in the merge queue, on manual runs (`gh workflow run ci.yml --ref <branch> [-f os=windows-latest]`) and on pull requests until the merge queue is on. **Fast**: Ubuntu + Python 3.11 on pull requests into main once the repo variable `CI_MERGE_QUEUE=true`. Pull requests into `release/**` always run the full lane: GitHub's merge queue takes no branch pattern, so the `release` ruleset requires `CI result` without one and nothing else would run every OS before a backport lands. A PR, or a merge queue group (judged by everything it would merge), that changes nothing the tests read runs no tests. `CI result` is the one check to require. Each job runs the tests in parallel (one worker per core minus one), then the `serial` ones (`pytestmark = pytest.mark.serial`; none today) one at a time. A failed test is retried once (pytest-rerunfailures); one that passed only on retry is listed in the job summary and as a warning (`scripts/ci_flaky.py`): a flake to fix, never a reason to raise `--reruns`
- Test classes group related tests: `TestProviderName`, `TestFeature`
- `python/tests/guardrails/conftest.py` provides `StreamingTestModel`, which streams real
  `TextDelta`/`ThinkingDelta`/`ToolUse` items through the router — use it whenever a test
  depends on delta handling rather than just the final message
- `test_injection_corpus.py` is a regression fence for the injection pattern pack. Changes
  to the pack must keep it green; genuinely uncatchable attacks belong in `KNOWN_GAPS`
  (xfail) rather than being deleted
