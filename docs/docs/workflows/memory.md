---
title: Using Memory
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Using Memory in Flows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Understand LLM memory sharing and flow state persistence for building conversational and stateful workflows.
</h2>

---

When we talk about "memory" in Timbal, there are two distinct concepts:


## LLM Memory

Each LLM step in your flow can have its own memory, controlled by the `memory_id` parameter in `.add_llm()`.

- **Shared memory**: If you use the same memory_id for multiple LLM steps, they will share the same conversational context and history.

- **Isolated memory**: If you use different memory_ids, each LLM step will have its own separate memory.

<CodeBlock language="python" code={`from timbal import Flow
flow = (
    Flow()
    .add_llm("llm1", model="gpt-4o-mini", memory_id="session1")
    .add_llm("llm2", model="gpt-4o-mini", memory_id="session1")  # Shares memory with llm1
    .add_llm("llm3", model="gpt-4o-mini", memory_id="session2")  # Has its own memory
)`}/>

Here, llm1 and llm2 share memory, while llm3 is independent.

### When to use:
- Share memory for multi-turn conversations or when you want LLMs to “remember” previous steps.

- Use separate memory for independent tasks.

## Flow State Persistence

You can persist the entire flow’s state (including all step outputs, memory, and context) by attaching a state saver with `.compile(state_saver=...`)`.

- InMemorySaver: Keeps state in RAM (good for testing).
- JSONLSaver: Saves state to a file (good for local persistence).
- TimbalPlatformSaver: Saves state to the Timbal platform (for production/cloud).

This allows you to:
- Resume or rewind flows.
- Track user sessions.
- Debug or audit flow runs.


<CodeBlock language="python" code={`from timbal import Flow
from timbal.state.savers import JSONLSaver

flow = (
    Flow()
    .add_llm("chat", model="gpt-4o-mini", memory_id="user_session")
    .set_input("chat.prompt", "prompt")
    .set_output("chat.return", "response")
    .compile(state_saver=JSONLSaver("state.jsonl"))
)`}/>

Every time you run the flow, the full state (inputs, outputs, memory, etc.) is saved to `state.jsonl`.

## Summary

LLM memory (memory_id) controls what the LLM “remembers” during a session.

State savers (state_saver) control whether the entire flow (including LLM memory) is saved and can be restored or analyzed later.

---

For more, see the [Flows Overview](/workflows) and [Advanced Flow Concepts](/workflows/advanced).