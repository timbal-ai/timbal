---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# State Management

Timbal provides a powerful state management system that enables your AI applications to maintain context, track conversations, and persist data across interactions.

---

## What are Snapshots?

A **Snapshot** is a complete record of a flow's execution at a specific point in time. Each time you run a flow, Timbal automatically creates snapshots that capture:

- Input parameters
- Output results
- Execution time
- Error states
- Step-by-step execution data
- Memory state
- Resource usage

<CodeBlock language="python" code={`from timbal import Flow
from timbal.state.savers import InMemorySaver

flow = (
    Flow()
    .add_llm("chat", model="gpt-4.1-nano", memory_id="chat_memory")
    .compile(state_saver=InMemorySaver())
)

# Each run creates a new snapshot
result = await flow.complete(prompt="Hello!")`}/>

## State Savers

State Savers determine where and how snapshots are stored. Timbal provides several built-in options:

### 1. InMemorySaver
<div style={{display: 'flex', alignItems: 'flex-start', gap: '1rem', margin: '1rem 0'}}>
  <div style={{flex: 1}}>
    <ul>
      <li>Stores snapshots in RAM</li>
      <li>Perfect for development and testing</li>
      <li>Data is lost when the program terminates</li>
    </ul>
  </div>
  <div style={{flex: 1}}>
    <CodeBlock language="python" code={`from timbal.state.savers import InMemorySaver
state_saver = InMemorySaver()`}/>
  </div>
</div>

### 2. JSONLSaver
<div style={{display: 'flex', alignItems: 'flex-start', gap: '1rem', margin: '1rem 0'}}>
  <div style={{flex: 1}}>
    <ul>
      <li>Stores snapshots in a JSONL file</li>
      <li>Good for local development and debugging</li>
      <li>Data persists between program runs</li>
    </ul>
  </div>
  <div style={{flex: 1}}>
    <CodeBlock language="python" code={`from timbal.state.savers import JSONLSaver
state_saver = JSONLSaver("snapshots.jsonl")`}/>
  </div>
</div>

### 3. TimbalPlatformSaver
<div style={{display: 'flex', alignItems: 'flex-start', gap: '1rem', margin: '1rem 0'}}>
  <div style={{flex: 1}}>
    <ul>
      <li>Stores snapshots in the Timbal Platform</li>
      <li>Ideal for production environments</li>
      <li>Provides centralized storage and monitoring</li>
    </ul>
  </div>
  <div style={{flex: 1}}>
    <CodeBlock language="python" code={`from timbal.state.savers import TimbalPlatformSaver
state_saver = TimbalPlatformSaver()`}/>
  </div>
</div>

## Creating Custom State Savers

You can create your own state saver by implementing the `BaseSaver` abstract class. This allows you to store snapshots in any backend of your choice (databases, cloud storage, etc.).

<CodeBlock language="python" code={`from abc import ABC, abstractmethod
from timbal.state.savers.base import BaseSaver
from timbal.state import RunContext, Snapshot

class CustomSaver(BaseSaver):
    @abstractmethod
    def get_last(self, path: str, context: RunContext) -> Snapshot | None:
        """Retrieve the last snapshot for a given path and context."""
        pass

    @abstractmethod
    def put(self, snapshot: Snapshot, context: RunContext) -> None:
        """Store a new snapshot."""
        pass`}/>

### State Savers Methods

#### get_last

The method **get_last** is used to retrieve the most recent snapshot for a given path and context.

It looks up the latest snapshot that matches the provided path (which identifies the flow or agent) and the given RunContext (which may include a parent_id to specify a chain of runs).

If a matching snapshot is found, it returns a Snapshot object.

#### put

Every time a flow or agent completes a run, Timbal creates a new Snapshot object representing the state of that run (including input, output, memory, etc.).

The put method is then called to save this snapshot using the configured state saver (e.g., in memory, in a file, or in a database).

Without put, no snapshots would be saved, and you wouldn’t be able to retrieve past state or maintain memory across runs.

## How Snapshots Work

1. **Creation**: Each time you run a flow, Timbal creates a snapshot that captures the complete state.

2. **Storage**: The snapshot is passed to your configured state saver, which stores it according to its implementation.

3. **Retrieval**: When running a flow with a parent context, Timbal automatically retrieves the last relevant snapshot to restore state.

<CodeBlock language="python" code={`from timbal import Flow
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver

flow = (
    Flow()
    .add_llm("chat", model="gpt-4.1-nano", memory_id="chat_memory")
    .compile(state_saver=InMemorySaver())
)

# First run creates a snapshot
result = await flow.complete(prompt="My name is David")

# Second run retrieves the previous snapshot
result = await flow.complete(
    context=RunContext(parent_id=result.run_id),
    prompt="What's my name?"
)`}/>

## Working with Snapshots

Here are some practical examples of how to work with state savers and snapshots:


<CodeBlock language="python" code={`from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver

# Create an agent with memory
agent = Agent(
    model="gpt-4o-mini",
    state_saver=InMemorySaver()
)

async def main():
    # First interaction
    result = await agent.complete(prompt="My name is David")
    # Second interaction with memory
    result = await agent.complete(
        context=RunContext(parent_id=result.run_id),
        prompt="What's my name?"
    )
    # Check the snapshots
    print(f"Number of snapshots: {len(agent.state_saver.snapshots)}")
    for snapshot in agent.state_saver.snapshots:
        print(f"Run ID: {snapshot.id}")
        print(f"Memory size: {len(snapshot.data['memory'].resolve())}")
`}/>

The output shows how the agent's memory grows and how snapshots are created and linked.

<CodeBlock language="bash" code={`Number of snapshots: 2
Memory size: 2
Memory size: 4`}/>

Each interaction (each call to complete) is independent and gets its own snapshot.

The first snapshot records the first message and response.

The second snapshot records the second message, the agent’s answer, and the updated memory (which now includes both turns).



For more details on implementing custom state savers, see the [Github documentation](https://github.com/timbal-ai/timbal/blob/main/python/timbal/state/snapshot.py).