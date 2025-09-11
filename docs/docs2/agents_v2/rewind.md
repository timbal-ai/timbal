---
title: Rewind
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Rewind in Agent

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Branch conversations and rewind Agent memory to any previous point using the enhanced tracing and context system.
</h2>

---

The <span style={{color: 'var(--timbal-purple)'}}><strong>rewind</strong></span> feature allows you to go back to any point in a conversation and explore different paths. This creates branching conversations where you can test alternative responses or recover from errors.

## How It Works

Every agent interaction creates a **RunContext** with a unique `run_id`. The framework automatically traces each step, allowing you to reference any previous point in the conversation.

### Creating Branches

You can create new branches by setting `RunContext(parent_id=...)` to rewind to any previous point. The agent will only remember the conversation up to that specific point, ignoring everything that happened after.

#### Example

<CodeBlock language="python" code={`from timbal import Agent
from timbal.state import RunContext, set_run_context

agent = Agent(name="example", model="openai/gpt-4o-mini")

# Main conversation path
print("=== Main Path ===")
step1 = await agent(prompt="Hello").collect()
step2 = await agent(prompt="My name is David").collect() 
step3 = await agent(prompt="What's my name?").collect()
print(f"Normal: {step3.output.content[0].text}")  # "Your name is David"

# Branch from step 1 (skip the name introduction)
print("\\n=== Branch 1: Rewind to Hello ===")
context = RunContext(parent_id=step1.run_id)
set_run_context(context)

branch1 = await agent(prompt="What's my name?").collect()
print(f"Branch 1: {branch1.output.content[0].text}")  # "I don't know your name"`}/>

This creates a branching structure:

<CodeBlock language="bash" code={`Root: "Hello"
├── Main Path: "My name is David" → "What's my name?" → "David"
└── Branch 1: "What's my name?" → "I don't know"  

Each branch maintains independent memory from the rewind point.`}/>

## Use Cases

- **Testing**: Try different conversation paths
- **Error Recovery**: Go back before an error occurred  
- **Exploration**: Explore "what if" scenarios
- **A/B Testing**: Compare different response strategies
- **Debugging**: Isolate specific conversation states

Rewind gives you full control over conversation flow and memory management.