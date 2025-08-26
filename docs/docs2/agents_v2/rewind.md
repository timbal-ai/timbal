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

The <span style={{color: 'var(--timbal-purple)'}}><strong>rewind</strong></span> feature in Agent V2 builds upon the robust tracing system to allow sophisticated conversation branching and memory manipulation. With the enhanced architecture, you can rewind to any point in nested agent conversations and create complex branching scenarios.

## How Rewind Works

Agent's rewind functionality leverages the enhanced tracing system and automatic memory resolution:

1. **Tracing-Based Memory**: Every interaction is traced with detailed path information
2. **Nested Context Support**: Rewind works across nested agent calls and tool executions
3. **Automatic Resolution**: Memory is automatically resolved based on the specified parent context
4. **Enhanced Branching**: Create complex conversation trees with multiple branches

## Basic Rewind Example

<CodeBlock language="python" code ={`from timbal.core_v2 import Agent
from timbal.state import RunContext, set_run_context
from timbal.types.message import Message

# Initialize agent and state saver
agent = Agent(
    name="conversation_agent",
    model="openai/gpt-4"
)

# Step 1: Start the conversation
context1 = RunContext()
set_run_context(context1)

result1 = await agent(
    prompt="Hello, I'm a software engineer"
).collect()

# Step 2: Add more information
result2 = await agent(
    prompt="I work with Python and Django"
).collect()

# Step 3: Continue the conversation
result3 = await agent(
    prompt="What technologies do I work with?"
).collect()

print("Normal flow result:", result3.output.content[0].text)
# Expected: "You work with Python and Django"

# Step 4: REWIND - Branch from step 1, ignoring the Python/Django information
context_rewind = RunContext(parent_id=result1.run_id)
set_run_context(context_rewind)

result_rewind = await agent(
    prompt="What technologies do I work with?"
).collect()

print("Rewound result:", result_rewind.output.content[0].text)
# Expected: "I don't have information about specific technologies you work with"`}/>

## Rewind Visualization

<CodeBlock language="bash" code ={`Conversation Tree Visualization:

Root: "Hello, I'm a software engineer"
├──▶ Branch A: "I work with Python and Django"
│    └── Sub-A1: "What technologies do I work with?"
│        └── Response: "You work with Python and Django"
└──▶ Branch B: (Rewind to Root) "What technologies do I work with?"
       └── Response: "I don't have information about specific technologies you work with"

Each branch maintains memory only up to its parent node.`}/>

## Summary

Agent's rewind capabilities provide:

- **Enhanced Tracing**: Detailed path-based tracing for precise rewind points
- **Nested Support**: Rewind works across nested agent calls and tool executions
- **Flexible Branching**: Create complex conversation trees with multiple paths
- **Context Integration**: System context works seamlessly with rewind
- **Advanced Patterns**: Conditional rewind, content filtering, and smart memory management

The architecture makes rewind more powerful and reliable while providing advanced customization options for sophisticated conversation management.
