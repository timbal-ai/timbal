# Timbal

A strongly opinionated framework for building and orchestrating agentic AI applications.

> 🚧 **Early Preview**: This project is in early development and APIs are subject to change. 

## Overview

Timbal provides two main patterns for building AI applications:

### 1. Agentic (Agent-based)
LLMs autonomously decide execution paths and tool usage based on goals. Best for:
- Complex reasoning tasks
- Dynamic tool selection
- Open-ended problem solving

### 2. Workflow (Flow-based) 
You explicitly define the execution steps and tool usage. Ideal for:
- Predictable processes
- Strict control requirements
- Performance-critical applications

## Quick Examples

### Agents 

In the following example we can see how Timbal makes it easy to build agents. You can use pre-built tools like `search`, or define your own tools using regular Python functions. No need to worry about complex schemas or interfaces - just write normal functions. Timbal supports any LLM provider (OpenAI, Anthropic, Gemini, local models) - just specify the model you want to use:

```python
from datetime import datetime

from timbal import Agent
from timbal.state.savers import InMemorySaver
from timbal.steps.perplexity import search

def get_datetime():
    return datetime.now().isoformat()

agent = Agent(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    tools=[
        get_datetime,
        {
            "tool": search,
            "description": "Search the internet."
        }
    ],
    state_saver=InMemorySaver(),
)

response = await agent.complete(prompt="What time is it?")
```

The agent will automatically understand how to use these tools based on their signatures and any optional metadata you provide.

### Workflows

Workflows provide fine-grained control over your AI pipeline. Unlike agents that make autonomous decisions, flows let you explicitly define each step and how data moves between them. Here's a simple RAG pipeline:

```python
from timbal import Flow
from timbal.state.savers import InMemorySaver

def retriever(query: str):
    return "..."

flow = (Flow()
    .add_step(retriever)
    .add_llm(model="gpt-4o-mini")
    .set_data_map("llm.prompt", "retriever.return")
    .set_input("retriever.query", "query")
    .set_output("response", "llm.return")
).compile(state_saver=InMemorySaver())

query = "..."

response = await flow.complete(prompt=query)
```

This pattern is ideal for applications requiring predictable execution paths, strict control over tool usage, or performance-critical processing.

## Installation

```bash
pip install timbal
```

## Documentation

The full documentation can be found [here](https://timbal-ai.github.io/timbal/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
