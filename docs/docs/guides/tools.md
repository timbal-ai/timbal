---
sidebar: 'docsSidebar'
---

# Tools

## What is a tool?

A tool in Timbal is a special type of `Link` that allows LLMs to perform specific actions. Think of tools as functions that your AI can call when needed!

:::tip[Key Concept]
Tools are only executed when explicitly called by a LLM, making them efficient and purpose-driven.
:::

## Creating Tools: Step by Step Guide

Let's create a simple tool that tells the current time! 

### 1. Define Your Tool Function

First, create a function that will serve as your tool:

```python
from datetime import datetime

def get_time(location: str):
    """Get the current time for a specific location"""
    return datetime.now()
```

### 2. Set Up Your Flow

Create a new flow and add an LLM that will use the tool:

```python
flow = Flow()
flow.add_llm("step_1")  # This LLM will use our tool
```

### 3. Add the Tool

Add your tool function as a step:

```python
flow.add_step(get_time)  # Add the time tool
```

### 4. Connect Tool to LLM

Link the tool to your LLM with the special `is_tool` flag:

```python
flow.add_link(
    "get_time",           # Source: our tool
    "step_1",            # Target: the LLM
    is_tool=True,        # Mark as tool
    description="Get the current time for a specified location"  # Help LLM understand the tool
)
```

:::note[Tool Description]
The `description` parameter helps the LLM understand when and how to use the tool. Make it clear and specific!
:::

### 5. Using Tool Results

Want to use the tool's output in another step? Here's how:

```python
# Add a new step
flow.add_step("step_2")

# Connect tool result to the new step
flow.add_link(
    "get_time",          # Source: our tool
    "step_2",           # Target: next step
    is_tool_result=True  # Mark as tool result
)
```

## Example: Complete Flow

Here's how everything comes together:

```python
from timbal import Flow
from datetime import datetime

# Define tool
def get_time(location: str):
    return datetime.now()

# Create and configure flow
flow = (
    Flow()
    .add_llm("step_1")                    # Add LLM
    .add_step(get_time)                   # Add tool
    .add_step("step_2")                   # Add result handler
    .add_link(                            # Connect tool to LLM
        "get_time", 
        "step_1", 
        is_tool=True,
        description="Get the time of a location"
    )
    .add_link(                            # Connect tool result
        "get_time", 
        "step_2", 
        is_tool_result=True
    )
)
```

## Tool Execution Behavior

:::info[Important]
Tools have a special execution pattern:
- They only run when explicitly called
- If not needed, they remain dormant
- This makes your flow efficient and focused
:::








We have a `flow = Flow()`

A tool will be a function so:

1. We want a LLM:

```python
flow.add_llm("step_1")
```

2. First we added as a step and add a llm

```python
flow.add_step(get_time)
```

3. We have to link the tools to the llm, with a special input: 
`is_tool = True`

```python
flow.add_link("get_time", "step1", is_tool=True, description="Get the time of London")
```

Then if we wanted to pass the result of the tool to another step:

4. We add the new step:

```python
flow.add_step("step2")
```

5. In order to say that the input it will receive comes from a llm, so it is a tool result we use another variable:
`is_tool_result = True`

```python
flow.add_link(get_time, "step2", is_tool_result=True)
```

When we use tools, as we are seting in the link as tool, it will be only executed when the tool is called, if not needed it will not be executed.
