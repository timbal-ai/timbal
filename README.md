# Timbal

> 🚀 **Agents**: API nearly stable, publishing in coming weeks | ⚠️ **Workflows**: Early work, major advances coming soon

The framework for building and orchestrating agentic AI applications—fast, scalable, and enterprise-ready. Clean async execution with automatic state persistence that enable tool-using agents that think, plan, and act in dynamic environments.

## Overview

Timbal provides two main patterns for building AI applications:

### 1. Agents
Autonomous AI agents that orchestrate LLM interactions with tool calling. They implement an execution pattern where an LLM can:
- Receive a prompt and generate a response
- Decide to call available tools based on the context
- Process tool results and continue the conversation
- Repeat until no more tool calls are needed or max iterations reached

Perfect for:
- Complex reasoning tasks requiring multiple steps
- Dynamic tool selection based on context
- Open-ended problem solving with autonomous decision making
- Multi-turn conversations with memory across iterations

### 2. Workflows
Explicit step-by-step execution with full control over data flow. Ideal for:
- Predictable processes with defined execution paths
- Strict control requirements over tool usage
- Performance-critical applications needing optimization

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
from timbal.steps.docx import create_docx

def get_datetime() -> str:
    return datetime.now().isoformat()

agent = Agent(
    name="datetime_agent",
    model="openai/gpt-5-mini",
    tools=[get_datetime, create_docx]
)

await agent(prompt="What time is it?").collect()
await agent(prompt="Cool, write that down on a word file for me").collect()
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

Fine-grained control over your AI pipeline with explicit step-by-step execution:

*Example coming as soon as the API is in beta.*

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
