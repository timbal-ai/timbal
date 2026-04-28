# Timbal Codegen

CLI tool for programmatically modifying Timbal agent and workflow source files. Uses [libcst](https://github.com/Instagram/LibCST) for safe, formatting-preserving code transformations.

## Usage

```bash
python -m timbal.codegen [--path <workspace>] [--dry-run] <operation> [options]
```

### Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--path` | `.` | Workspace directory containing `timbal.yaml` |
| `--dry-run` | off | Print transformed code to stdout without writing to disk |

### Workspace

Every workspace must have a `timbal.yaml` with a fully-qualified entry point:

```yaml
fqn: "workflow.py::workflow"
```

The entry point can be an `Agent` or a `Workflow`. Each operation validates the entry point type and rejects mismatches (e.g. `add-step` on an Agent entry point).

---

## Operations

### `add-tool` — Add a tool to an Agent

```bash
# Framework tool
python -m timbal.codegen add-tool --type WebSearch

# Framework tool with custom name
python -m timbal.codegen add-tool --type WebSearch --name my_search

# Custom tool from a function definition
python -m timbal.codegen add-tool --type Custom \
  --definition "def search(query: str) -> str:\n    return results"

# Add a tool to a specific step in a Workflow
python -m timbal.codegen add-tool --type WebSearch --step agent_a
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--type` | yes | `Bash`, `CalaSearch`, `Edit`, `Read`, `WebSearch`, `Write`, or `Custom` |
| `--definition` | Custom only | Full function definition as a string |
| `--name` | no | Override the default tool name |
| `--step` | no | Target step name within a Workflow (adds tool to that step's tools list) |

**Requires**: Agent entry point, or Workflow entry point when using `--step`.

**What it does**:
- Adds the import (`from timbal.tools import WebSearch` or `from timbal.core import Tool`)
- Creates a variable assignment (`web_search = WebSearch()`)
- Adds the variable to the Agent's `tools=[...]` list
- Idempotent — re-running updates rather than duplicates

---

### `remove-tool` — Remove a tool from an Agent

```bash
python -m timbal.codegen remove-tool --name web_search

# Remove a tool from a specific step in a Workflow
python -m timbal.codegen remove-tool --name web_search --step agent_a
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--name` | yes | Name of the tool variable or runtime name to remove |
| `--step` | no | Target step name within a Workflow (removes tool from that step's tools list) |

**Requires**: Agent entry point, or Workflow entry point when using `--step`.

Removes the tool reference from the Agent's `tools=[...]` list. Unused variables, functions, and imports are cleaned up automatically.

---

### `set-config` — Configure an Agent, tool, or workflow step

This is the unified configuration operation. Behavior depends on the entry point type and whether a target name is provided.

#### Configure the Agent entry point

```bash
python -m timbal.codegen set-config \
  --config '{"model": "openai/gpt-4o", "system_prompt": "You are helpful.", "max_iter": 5}'
```

Valid Agent fields: `name`, `description`, `model`, `system_prompt`, `max_iter`, `max_tokens`, `temperature`, `base_url`, `api_key`, `model_params`, `skills_path`.

Set a field to `null` to remove it:

```bash
python -m timbal.codegen set-config --config '{"system_prompt": null}'
```

#### Configure a tool on an Agent

```bash
python -m timbal.codegen set-config --name web_search --config '{"timeout": 30}'
```

Config fields are validated against the tool's schema. Supported configurable tools: `WebSearch`, `CalaSearch`, `Tool` (custom).

#### Configure a workflow step's constructor

```bash
python -m timbal.codegen set-config --name agent_b --config '{"model": "openai/gpt-4o"}'
```

Updates the step's variable assignment (e.g. `agent_b = Agent(model="openai/gpt-4o")`).

| Argument | Required | Description |
|----------|----------|-------------|
| `--name` | no | Target tool or step name. Omit to configure the entry point itself. |
| `--config` | depends | JSON object with constructor kwargs |

---

### `add-step` — Add a step to a Workflow

```bash
# Agent step
python -m timbal.codegen add-step --type Agent \
  --config '{"name": "agent_b", "model": "openai/gpt-4o-mini"}'

# Framework tool as a step
python -m timbal.codegen add-step --type WebSearch

# Custom function step
python -m timbal.codegen add-step --type Custom \
  --definition "def process(x: str) -> str:\n    return x.upper()"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--type` | yes | `Agent`, `Custom`, or a framework tool (`Bash`, `CalaSearch`, `Edit`, `Read`, `WebSearch`, `Write`) |
| `--config` | Agent only | JSON with Agent constructor params (must include `name`) |
| `--definition` | Custom only | Full function definition |
| `--name` | no | Override the step name |
| `--x` | no | X canvas position (float). Auto-computed if omitted. |
| `--y` | no | Y canvas position (float). Auto-computed if omitted. |

**Requires**: Workflow entry point.

**What it does**:
- Adds necessary imports
- Creates the variable assignment or function definition with `metadata={"position": {"x": ..., "y": ...}}`
- Appends a `workflow.step(...)` call after the last existing step (or after the entry point)
- Idempotent — re-running with the same name updates the existing step

Agent config fields are validated against the same set as `set-config`.

**Auto-positioning**: When `--x` and `--y` are omitted, the new step is placed automatically based on the existing nodes. The position is computed by finding the rightmost column of nodes and offsetting one column to the right (360px), vertically centered on that column. Bare function steps (not wrapped in `Tool`/`Agent`) are treated as nodes at the default column (`x=100`), stacked vertically with 140px spacing. If no existing steps have positions, the first step starts at `(100, 100)`.

---

### `remove-step` — Remove a step from a Workflow

```bash
python -m timbal.codegen remove-step --name agent_b
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--name` | yes | Name of the step to remove |

**Requires**: Workflow entry point.

Removes the `workflow.step(...)` call for the named step. Unused variables, functions, and imports are cleaned up automatically. If the step doesn't exist, the operation is a no-op.

---

### `set-param` — Set a parameter on a workflow step

Sets a parameter on a workflow step. Two modes: **map** wires a param from another step's output, **value** sets a static value.

```bash
# Map a param from another step's output
python -m timbal.codegen set-param --target agent_b --name prompt \
  --type map --source agent_a

# Map with a dot-notation key path into the source output
python -m timbal.codegen set-param --target agent_b --name prompt \
  --type map --source agent_a --key output.cleaned

# Map with nested index and attribute access
python -m timbal.codegen set-param --target agent_b --name prompt \
  --type map --source agent_a --key output.0.items.name

# Set a static value
python -m timbal.codegen set-param --target agent_a --name prompt \
  --type value --value '"Hello world"'

# Remove a param (set value to null)
python -m timbal.codegen set-param --target agent_b --name prompt \
  --type value --value 'null'
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--target` | yes | Target step name |
| `--name` | yes | Parameter name to set |
| `--type` | yes | `map` (wire from another step's output) or `value` (static value) |
| `--source` | map only | Source step name |
| `--key` | no | Dot-notation path into the source step's output (e.g. `output.cleaned`, `output.0.items.name`) |
| `--value` | value only | JSON literal for the static value. Use `null` to remove the param. |

**Requires**: Workflow entry point.

**Key path syntax**: The `--key` uses dot-notation where numeric segments become index access and string segments become attribute access:

| Key path | Generated Python |
|----------|-----------------|
| `output.cleaned` | `.output.cleaned` |
| `output.0.items` | `.output[0].items` |
| `output.0.data.name.2` | `.output[0].data.name[2]` |

**What it does**:
- `type=map`: adds a lambda kwarg that reads from the source step's output via `get_run_context().step_span("source").output`
- `type=map` with `--key`: traverses into the output using the dot-notation path
- `type=value`: sets a static value on the step's `.step()` call
- `type=value` with `null`: removes the param
- Adds `from timbal.state import get_run_context` import when needed (map type)
- Idempotent — re-running updates rather than duplicates

---

### `set-position` — Set the canvas position for a node

Sets the `(x, y)` position for a node on the ReactFlow canvas. The position is stored inside the runnable's `metadata` dict and surfaced as a top-level `position` key in `get-flow` output.

```bash
# Set position on the Agent entry point
python -m timbal.codegen set-position --x 100 --y 200

# Set position on a workflow step
python -m timbal.codegen set-position --name agent_a --x 150 --y 250
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--x` | yes | X coordinate (float) |
| `--y` | yes | Y coordinate (float) |
| `--name` | no | Step name (required for Workflow, rejected for Agent) |

**What it does**:
- Upserts `metadata={"position": {"x": ..., "y": ...}}` on the constructor
- Preserves existing metadata keys — only the `"position"` key is touched
- Idempotent — re-running updates the position in place

---

### `add-edge` — Add an ordering or conditional edge between workflow steps

Adds an execution ordering dependency between two steps. For wiring data between steps, use `set-param` instead.

```bash
# Pure ordering (adds source to target's depends_on)
python -m timbal.codegen add-edge --source agent_a --target agent_b

# Conditional edge (ordering + condition)
python -m timbal.codegen add-edge --source agent_a --target agent_b \
  --when 'lambda: get_run_context().step_span("agent_a").output.content != ""'
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--source` | yes | Source step name |
| `--target` | yes | Target step name |
| `--when` | no | Python expression for a conditional edge |

**Requires**: Workflow entry point.

**What it does**:
- Merges the source into the target's `depends_on=[...]` list
- With `--when`: adds a `when=` kwarg to the target's `.step()` call
- Adds `from timbal.state import get_run_context` import when needed
- Idempotent — re-running updates rather than duplicates

---

### `remove-edge` — Remove an edge between two workflow steps

```bash
python -m timbal.codegen remove-edge --source agent_a --target agent_b
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--source` | yes | Source step name |
| `--target` | yes | Target step name |

**Requires**: Workflow entry point.

Removes all references to the source step from the target's `.step()` call:
- Removes the source from `depends_on=[...]`
- Removes param kwargs that reference the source via `step_span("source")`
- Removes the `when=` kwarg if it references the source

---

### `convert-to-workflow` — Convert an Agent into a Workflow

```bash
# Basic conversion
python -m timbal.codegen convert-to-workflow

# With a custom workflow name
python -m timbal.codegen convert-to-workflow --name my_pipeline
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | entry point variable name | The `name=` kwarg for the `Workflow()` constructor |

**Requires**: Agent entry point.

Converts an Agent entry point into a Workflow with the Agent as its single step. The entry point variable name stays the same (so `timbal.yaml` doesn't need updating). The Agent is moved to a new variable named after its `name` kwarg (or `"agent"` if none).

**Before:**
```python
from timbal import Agent

workflow = Agent(name="agent_a", model="openai/gpt-4o-mini")
```

**After:**
```python
from timbal import Agent, Workflow

agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

workflow = Workflow(name="workflow")
workflow.step(agent_a)
```

---

### `list-tools` — List available framework tool types

```bash
python -m timbal.codegen list-tools
python -m timbal.codegen list-tools --no-cache   # skip disk cache
```

Outputs all framework tools as a flat JSON array. **Note:** with 700+ tools this can be slow. Prefer `get-tools` for paginated, filtered access.

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-cache` | off | Skip the disk cache and force a full rediscovery |

```json
{
  "tools": [
    {"type": "Bash", "module": "timbal.tools", "name": "bash", "description": null, "provider": null, "provider_logo": null},
    ...
  ]
}
```

---

### `get-tools` — Browse and search tools (paginated)

Two-tier tool discovery with pagination to avoid loading all tools at once.

#### Default: list providers

```bash
python -m timbal.codegen get-tools
```

Returns provider summaries sorted by tool count:

```json
{
  "providers": [
    {"name": "zendesk", "logo": "https://content.timbal.ai/assets/zendesk_favicon.svg", "tool_count": 442},
    {"name": "slack", "logo": "https://content.timbal.ai/assets/slack_favicon.svg", "tool_count": 27},
    {"name": "system", "logo": null, "tool_count": 6}
  ]
}
```

Tools with no provider appear under `"system"`.

#### Filter by provider

```bash
python -m timbal.codegen get-tools --provider slack
python -m timbal.codegen get-tools --provider system
```

#### Search across all tools

```bash
python -m timbal.codegen get-tools --search "create ticket"
```

Case-insensitive substring match on tool name, type (class name), and description.

#### Combined filters

```bash
python -m timbal.codegen get-tools --provider stripe --search invoice
```

`--provider` and `--search` compose as AND.

#### Pagination

```bash
python -m timbal.codegen get-tools --provider zendesk --limit 10 --offset 20
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider` | none | Filter by provider name (`"system"` for tools with no provider) |
| `--search` | none | Case-insensitive substring search on name, type, description |
| `--limit` | 50 | Max tools to return |
| `--offset` | 0 | Number of tools to skip |

Response includes pagination metadata:

```json
{
  "tools": [...],
  "total": 442,
  "limit": 10,
  "offset": 20
}
```

---

### `get-models` — Browse and search LLM models (paginated)

Two-tier model discovery with pagination, following the same pattern as `get-tools`.

#### Default: list providers

```bash
python -m timbal.codegen get-models
```

Returns provider summaries sorted by model count:

```json
{
  "providers": [
    {"name": "togetherai", "logo": "https://content.timbal.ai/assets/togetherai_favicon.svg", "model_count": 30},
    {"name": "openai", "logo": "https://content.timbal.ai/assets/openai_favicon.svg", "model_count": 20},
    {"name": "anthropic", "logo": "https://content.timbal.ai/assets/anthropic_favicon.svg", "model_count": 10}
  ]
}
```

#### Filter by provider

```bash
python -m timbal.codegen get-models --provider anthropic
```

#### Search across all models

```bash
python -m timbal.codegen get-models --search "vision"
```

Case-insensitive substring match on model `id`, `display_name`, and `description`.

#### Combined filters

```bash
python -m timbal.codegen get-models --provider openai --search "gpt-4"
```

`--provider` and `--search` compose as AND.

#### Pagination

```bash
python -m timbal.codegen get-models --provider togetherai --limit 10 --offset 20
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider` | none | Filter by provider name |
| `--search` | none | Case-insensitive substring search on id, display_name, description |
| `--limit` | 50 | Max models to return |
| `--offset` | 0 | Number of models to skip |

When filtering, response includes pagination metadata and full model objects:

```json
{
  "models": [
    {
      "id": "anthropic/claude-opus-4-6",
      "provider": "anthropic",
      "display_name": "Claude Opus 4.6",
      "description": "Anthropic's most capable model, excelling at planning and debugging within large codebases.",
      "input_price": 5.0,
      "output_price": 25.0,
      "context_window": 200000,
      "capabilities": ["vision", "tools", "reasoning"]
    }
  ],
  "total": 10,
  "limit": 50,
  "offset": 0
}
```

**Model fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Full routing ID passed to `model=` (e.g. `"anthropic/claude-opus-4-6"`) |
| `provider` | string | Provider key (`anthropic`, `openai`, `google`, `togetherai`, `xai`, `groq`, `fireworks`, `byteplus`, `xiaomi`, `cerebras`, `sambanova`) |
| `display_name` | string | Human-readable name for UI |
| `description` | string | One-sentence description from provider docs |
| `input_price` | float \| null | USD per 1M input tokens (`null` if not publicly listed) |
| `output_price` | float \| null | USD per 1M output tokens |
| `context_window` | int \| null | Maximum context in tokens |
| `capabilities` | string[] | Subset of `vision`, `tools`, `reasoning`, `audio`, `video`, `image_generation` |

**Model registry (`models.yaml`):**

Model metadata lives in `python/timbal/models.yaml`. To add or update a model, edit the YAML — it is the single source of truth. The `model=` type annotation in `llm_router.py` is derived from it via `scripts/generate_models.py`:

```bash
uv run python scripts/generate_models.py
```

---

### `get-flow` — Print the execution graph

```bash
python -m timbal.codegen get-flow
```

Outputs a JSON representation of the entry point's execution graph with `nodes` and `edges`.

Each node has a top-level `position` key (`{"x": ..., "y": ...}`) for ReactFlow canvas placement, defaulting to `{"x": 0, "y": 0}` when not set. Use `set-position` to configure it.

Each node's `data.params.properties` includes OpenAPI-style schema fields (`type`, `title`, `description`, etc.) plus a `value` field describing how the param is set:

```json
{
  "prompt": {
    "title": "Prompt",
    "type": "string",
    "value": {"type": "map", "source": "agent_a", "key": "output.cleaned"}
  }
}
```

The `value` field has two forms:
- **Map**: `{"type": "map", "source": "<step_name>"}` with an optional `"key"` for dot-notation path
- **Static**: `{"type": "value", "value": <json_value>}`
- **Absent**: param has no default set

Config fields that reference the model registry use `"x-timbal-ref": "models"` instead of inlining the full enum. Call `get-models` to get the available options:

```json
{
  "model": {
    "type": "string",
    "x-timbal-ref": "models",
    "value": "anthropic/claude-haiku-4-5"
  }
}
```

---

### `test` — Run the entry point

```bash
# Basic run
python -m timbal.codegen test

# With input
python -m timbal.codegen test --input '{"query": "hello"}'

# With streaming
python -m timbal.codegen test --stream

# With run context
python -m timbal.codegen test --context '{"id": "my-run-id"}'
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | `{}` | Input parameters as JSON |
| `--context`, `-c` | none | RunContext fields as JSON |
| `--stream`, `-s` | off | Print every event instead of only the final output |

---

## Pipeline

Every code transformation follows this pipeline:

```
timbal.yaml → parse entry point FQN
    → load source file
    → parse to CST (libcst)
    → apply transformer (add/remove/set-config/set-param/set-position)
    → remove unused code (iterative dead code elimination)
    → format with ruff
    → write to file (or stdout with --dry-run)
```

The dead code elimination pass automatically cleans up variables, functions, and imports that become unused after a transformation (e.g. removing a step also removes its variable assignment and import).

## Idempotency

All operations are idempotent. Running the same operation twice produces the same result — existing definitions are updated in place rather than duplicated.

## Error Handling

Operations fail with a non-zero exit code and a message to stderr when:

- `timbal.yaml` is missing or has no `fqn` field
- The source file doesn't exist
- The entry point type doesn't match the operation (Agent vs Workflow)
- Required arguments are missing (`--definition` for Custom, `name` for Agent steps, `--source` for map params)
- Config fields are unknown or invalid
- JSON arguments are malformed
