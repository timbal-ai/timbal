---
title: Runnables
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Runnables

<h2 className="subtitle" style={{marginTop: '0px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Executable primitives that provide consistent interfaces for all Timbal components.
</h2>

---

## What is a Runnable?
A <span style={{color: 'var(--timbal-purple)'}}><strong>Runnable</strong></span> is an executable unit capable of processing inputs and producing outputs through an async generator interface. It works as a wrapper that turns any callable into a standarized, traceable, and composable execution unit.

All runnables provide a unified interface and execution pattern, enabling seamless composition regardless of their underlying implementation:

- **[Tools](../agents/tools)** - Function wrappers with automatic schema generation and explicit parameter control
- **[Agents](../agents/index.md)** - Autonomous execution units that orchestrate LLM interactions with tool calling
- **[Workflows](../workflows/index.md)** - Programmable execution pipelines that orchestrate step-by-step processing

All runnables must have a unique `name`. This name is used for tracing, debugging, and referencing the runnable in workflows, agents, and other components. Whether you're creating a Tool, an Agent, or a Workflow, the `name` parameter is required and must be unique within your application context.

Here's how to create a basic Tool:
<CodeBlock language="python" code={`from timbal import Tool

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

add_tool = Tool(
    name="add_tool",
    handler=add,
)`}/>

---

## Parameter Handling and Basic Execution

Runnables can be called as a regular python function. You can **pass parameters as keyword arguments**, and they'll be automatically mapped to the appropriate parameters in the underlying functions:

<CodeBlock language="python" code={`result = await add_tool(a=5, b=3).collect() # Returns 8`}/>

You can also set default parameter values when creating the runnable:

<CodeBlock language="python" highlight={"4"} code={`add_tool = Tool(
    name="add_tool",
    handler=add,
    default_params={"b": 3}
)

result = await add_tool(a=5).collect() # Returns 8`}/>

Default parameters can also be callables that are evaluated at runtime:

<CodeBlock language="python" highlight={"6"} code={`import random

add_tool = Tool(
    name="add_tool",
    handler=add,
    default_params={"b": lambda: random.randint(0, 10)}
)

result = await add_tool(a=5).collect() # Returns a number between 5 - 15`}/>

Note that runtime parameters always override default values, including callable defaults:

<CodeBlock language="python" code={`result = await add_tool(a=5, b=10).collect() # Returns 15`}/>

> In the above example, `b=10` is passed at runtime and overrides the callable default that would generate a random number.
