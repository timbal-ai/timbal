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
python -m timbal.codegen remove-tool web_search

# Remove a tool from a specific step in a Workflow
python -m timbal.codegen remove-tool web_search --step agent_a
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<tool_name>` | yes | Name of the tool variable or runtime name to remove |
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
python -m timbal.codegen set-config web_search --config '{"timeout": 30}'
```

Config fields are validated against the tool's schema. Supported configurable tools: `WebSearch`, `CalaSearch`, `Tool` (custom).

#### Configure a workflow step's constructor

```bash
python -m timbal.codegen set-config agent_b --config '{"model": "openai/gpt-4o"}'
```

Updates the step's variable assignment (e.g. `agent_b = Agent(model="openai/gpt-4o")`).

#### Wire workflow step inputs (`--params`)

```bash
python -m timbal.codegen set-config agent_b \
  --params '{"prompt": {"step": "agent_a"}}'
```

This generates a lambda that reads from the source step's output:

```python
workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
```

To index into the output with a key:

```bash
python -m timbal.codegen set-config agent_b \
  --params '{"prompt": {"step": "agent_a", "key": "result"}}'
```

Generates:

```python
workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output["result"])
```

Multiple params can be set at once:

```bash
python -m timbal.codegen set-config agent_b \
  --params '{"prompt": {"step": "agent_a"}, "context": {"step": "input_handler", "key": "data"}}'
```

#### Set step dependencies (`--depends-on`)

```bash
python -m timbal.codegen set-config agent_b --depends-on agent_a
python -m timbal.codegen set-config agent_c --depends-on agent_a --depends-on agent_b
```

Generates:

```python
workflow.step(agent_c, depends_on=["agent_a", "agent_b"])
```

#### Combined: constructor config + params + dependencies

All flags can be used together:

```bash
python -m timbal.codegen set-config agent_b \
  --config '{"model": "openai/gpt-4o"}' \
  --params '{"prompt": {"step": "agent_a"}}' \
  --depends-on agent_a
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<name>` | no | Target tool or step name. Omit to configure the entry point itself. |
| `--config` | depends | JSON object with constructor kwargs |
| `--params` | no | JSON object mapping input params to source steps (Workflow only) |
| `--depends-on` | no | Step dependency name, repeatable (Workflow only) |

**Removing params or dependencies**: To remove a `--params` mapping or `--depends-on` entry, re-send the full set without the one you want to remove. The provided values replace the existing ones entirely — there is no separate "remove" operation.

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

**Requires**: Workflow entry point.

**What it does**:
- Adds necessary imports
- Creates the variable assignment or function definition
- Appends a `workflow.step(...)` call after the last existing step (or after the entry point)
- Idempotent — re-running with the same name updates the existing step

Agent config fields are validated against the same set as `set-config`.

---

### `remove-step` — Remove a step from a Workflow

```bash
python -m timbal.codegen remove-step agent_b
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<step_name>` | yes | Name of the step to remove |

**Requires**: Workflow entry point.

Removes the `workflow.step(...)` call for the named step. Unused variables, functions, and imports are cleaned up automatically. If the step doesn't exist, the operation is a no-op.

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

### `rename` — Rename a step or tool

```bash
# Rename a workflow step
python -m timbal.codegen rename agent_a --to agent_b

# Rename a tool
python -m timbal.codegen rename web_search --to my_search
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<old_name>` | yes | Current runtime name of the step or tool |
| `--to` | yes | New name |

**What it does**:
- Renames the variable and updates the `name=` kwarg in the constructor
- Updates all references: `tools=[...]` list entries, `.step()` call arguments
- Updates string references in `depends_on=["..."]` and `step_span("...")` calls
- Cannot rename the entry point variable (referenced by `timbal.yaml`)

---

### `list-tools` — List available framework tool types

```bash
python -m timbal.codegen list-tools
```

Outputs a JSON array of framework tools discovered from `timbal.tools`:

```json
[
  {"type": "Bash", "module": "timbal.tools", "name": "bash", "description": null},
  {"type": "CalaSearch", "module": "timbal.tools", "name": "cala_search", "description": "Search for verified knowledge..."},
  ...
]
```

---

### `get-flow` — Print the execution graph

```bash
python -m timbal.codegen get-flow
```

Outputs a JSON representation of the entry point's execution graph.

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
    → apply transformer (add/remove/set-config)
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
- Required arguments are missing (`--definition` for Custom, `name` for Agent steps)
- Config fields are unknown or invalid
- JSON arguments are malformed
