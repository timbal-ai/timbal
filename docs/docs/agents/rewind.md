---
title: Rewind
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Rewind

The **rewind** feature in Timbal allows you to branch or "rewind" the memory of an agent or flow by changing the `parent_id` in the `RunContext`. This is useful when you want to continue a conversation from a previous point, ignoring any newer context that was added after that point.

---

## How Rewind Works

Normally, when you pass a `parent_id` to a new `RunContext`, the agent loads the memory up to that point. If you change the `parent_id` to an earlier run, the agent will only have access to the memory up to that run, and **not** any of the memory added after.

This allows you to:
- Branch the conversation from any previous point.
- Ignore or overwrite later context by starting from an earlier memory state.

## Example

Let's see a practical example:

<CodeBlock language="python" code ={`from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver

agent = Agent(
    model="gpt-4o-mini",
    state_saver=InMemorySaver(),
)

# Step 1: Start the conversation
run1 = await agent.complete(prompt="Hello")
# Memory: ["Hello"]

# Step 2: Add a name
run2 = await agent.complete(
    context=RunContext(parent_id=run1.run_id),
    prompt="My name is David"
)
# Memory: ["Hello", "My name is David"]

# Step 3: Ask for the name (normal forward memory)
run3 = await agent.complete(
    context=RunContext(parent_id=run2.run_id),
    prompt="What is my name?"
)
print(run3.output.content[0].text)
# Output: "Your name is David!"

# Step 4: REWIND - branch from the first message, ignoring the 'David' memory
run4 = await agent.complete(
    context=RunContext(parent_id=run1.run_id),
    prompt="What is my name?"
)
print(run4.output.content[0].text)
# The agent does NOT know your name, because it only sees the memory up to "Hello"`}/>

## What happens under the hood?

- When you set `parent_id` to an earlier run, the agent loads only the memory up to that run.
- Any context added after that run is ignored for this new branch.
- This is like "rewinding" the conversation and starting a new path from that point.

**Visual Diagram**

<CodeBlock language="bash" code ={`Start ──▶ "My name is David" ──▶ "What is my name?" (→ "Your name is David!")
      │
      └───▶ (rewind) "What is my name?" (→ Agent does NOT know your name)`}/>


## Summary

- **Rewind** lets you branch or restart the conversation from any previous point by changing the `parent_id`.
- The agent will only see the memory up to the specified `parent_id`.
- This is useful for "what if" scenarios, testing, or branching conversations.