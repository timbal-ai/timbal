# Timbal

A strongly opinionated framework for building and orchestrating agentic AI applications.

> ⚠️ **Early Preview**: This project is in very early development and APIs are subject to change. Use at your own risk.

## Overview

Timbal provides a flexible framework for building complex AI applications with:

- Flow-based orchestration of LLM calls and tools
- First-class support for agentic behaviors
- Built-in memory management and state persistence
- Streaming responses and real-time updates
- Support for multiple LLM providers (OpenAI, Anthropic, Gemini, TogetherAI)

## Quick Example
```python
from timbal import Flow

# Create a flow
flow = Flow()

# Add an agent that can use tools
flow.add_agent(
    model="gpt-4",
    tools=[
        {
            "tool": search_web,
            "description": "Search the web for information"
        },
        {
            "tool": calculate,
            "description": "Perform calculations" 
        }
    ],
    max_iter=3  # Allow up to 3 tool use iterations
)

# Run the flow
async for chunk in flow.run(prompt="What is 235 * 18 and what year was that number first used in history?"):
    print(chunk, end="", flush=True)
```

## Installation

```bash
pip install timbal
```

## Documentation

[Documentation coming soon]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
