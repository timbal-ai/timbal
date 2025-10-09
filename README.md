# Timbal

> 🚀 Version 1.0 is now live! Start with the [quickstart guide](https://docs.timbal.ai/quickstart) or explore our [examples](https://docs.timbal.ai/examples).

Timbal is an open-source python framework for building reliable AI applications, battle-tested in production with simple, transparent architecture that eliminates complexity while delivering blazing fast performance, robust typing, and API stability in an ever-changing ecosystem.

## Overview

Timbal provides two main patterns for building AI applications:

### 1. Agents
Autonomous AI agents that orchestrate LLM interactions with tool calling.

Perfect for:
- Complex reasoning tasks requiring multiple steps
- Dynamic tool selection based on context
- Open-ended problem solving with autonomous decision making
- Multi-turn conversations with memory across iterations

### 2. Workflows
Explicit step-by-step execution with full control over data flow and automatic dependency management.

Perfect for:
- Complex data processing pipelines with multiple stages
- Predictable processes with defined execution paths
- Performance-critical applications requiring concurrent execution
- Workflows where explicit control over step dependencies is needed

## Installation

We recommend using `uv`:

```bash
uv add timbal
```

Or with pip:

```bash
pip install timbal
```

For the latest developments, you can clone or fork the repository:

```bash
git clone https://github.com/timbal-ai/timbal.git
cd timbal
uv sync --dev
```

## Quick Examples

### Agents

Agents are autonomous execution units that orchestrate LLM interactions with tool calling. Define tools as Python functions - the framework handles schema generation, parameter validation, and execution orchestration:

```python
from datetime import datetime

from timbal import Agent
from timbal.tools import WebSearch

def get_datetime() -> str:
    return datetime.now().isoformat()

agent = Agent(
    name="demo_agent",
    model="openai/gpt-5-mini",
    tools=[get_datetime, WebSearch()]
)

await agent(prompt="What time is it?").collect()
await agent(prompt="What's the weather like in Barcelona?").collect()
```

The framework performs automatic introspection of function signatures and docstrings for tool schema generation. Architecture features:

**Execution Engine:**
- Asynchronous concurrent tool execution via multiplexed event queues
- Conversation state management with automatic memory persistence across iterations
- Multi-provider LLM routing with unified interface abstraction

**Tool System:**
- Runtime tool discovery with automatic OpenAI/Anthropic schema generation
- Support for nested Runnable composition and hierarchical agent orchestration
- Dynamic parameter validation using Pydantic models

**Advanced Runtime:**
- Template-based system prompt composition with runtime callable injection
- Configurable iteration limits with autonomous termination detection
- Event-driven streaming architecture with real-time processing capabilities
- Pre/post execution hooks for cross-cutting concerns and runtime interception

### Workflows

Workflows provide explicit step-by-step execution with automatic dependency management. Define your pipeline as a series of steps that execute concurrently while respecting dependencies:

```python
import httpx
from timbal import Workflow
from timbal.state import get_run_context
from timbal.tools import Write

async def fetch_content(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        res = await client.get(url)
        res.raise_for_status()
        return res.text

workflow = (Workflow(name="demo_workflow")
    .step(fetch_content)
    .step(Write(), path="./output.html", content=get_run_context().step_span("fetch_content").output) 
)

await workflow(url="https://timbal.ai").collect()
```

**Execution Flow:**

The workflow automatically infers execution order from data dependencies:

1. **`fetch_content`** runs first (no dependencies, takes url from workflow input)
2. **`Write`** executes once `fetch_content` completes, using its output to save the HTML content

Execution pattern: `fetch_content` → `Write`

**Key Features:**

- **Automatic Dependency Linking**: Steps using `get_run_context().step_span()` create automatic step dependencies
- **Workflow Input Propagation**: Unspecified step parameters become workflow-level inputs  
- **Concurrent Execution**: Independent steps run in parallel for optimal performance
- **DAG Validation**: Prevents circular dependencies with built-in cycle detection
- **Error Resilience**: Failed steps skip their dependents gracefully

**Architecture:**
- Steps execute as a directed acyclic graph (DAG) with dependency coordination
- Event-driven multiplexing streams output from all steps as they complete
- Built-in state management tracks step completion and data flow
- Pydantic validation ensures type safety across step boundaries

## Why are we building this?

**Simplicity over complexity.** Unlike LangGraph, CrewAI, and other frameworks that abstract away what's really happening, Timbal keeps things transparent. Under the hood, it's just LLM calls and async function execution - no hidden magic, no opaque abstractions.

**Developer experience first.** We believe you shouldn't need to learn a new mental model to build AI applications. If you understand functions and async/await from any modern programming language, you understand Timbal. Most agents are built in 10-20 lines of code.

**Battle-tested architecture.** Our core abstractions have been refined through real-world production usage. The framework is designed around proven patterns: async generators, Pydantic validation, and event-driven processing.

**Robust interfaces.** Strong typing and validation make it nearly impossible to break things from the outside, while clean abstractions make internal modifications straightforward. The framework fails fast with clear error messages.

**Performance by design.** Built for production workloads with concurrent execution, efficient memory management, and minimal overhead. Every design decision prioritizes speed and scalability.

**Stability in a chaotic ecosystem.** Providers change APIs monthly. Timbal provides a stable abstraction that shields your applications in production.

## Documentation

Full documentation is available at [docs.timbal.ai](https://docs.timbal.ai).

## Contributing

We welcome contributions! Please submit a Pull Request or open an issue.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
