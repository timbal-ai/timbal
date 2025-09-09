---
title: Events
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Events

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Monitor execution in real-time, handle streaming results, and debug your Runnables with comprehensive event tracking.
</h2>

---

## What are Events?
Events are the communication mechanism that Runnables use to stream information throughout their execution lifecycle. Every Runnable execution produces a sequence of events that can be consumed in real-time or collected for later processing.

### Event Properties
All events share common properties:
- `run_id`: Unique identifier for the execution run
- `parent_run_id`: ID of the parent run (for nested executions)
- `path`: Hierarchical path of the Runnable in the execution context
- `call_id`: Unique identifier for this specific call
- `parent_call_id`: ID of the parent call (for nested calls)

### Event Types

#### Start Event
Signals the beginning of a Runnable execution. Example:
<CodeBlock language="python" code={`StartEvent(
    run_id="run_123",
    parent_run_id="parent_456", 
    path="workflow.step1",
    call_id="call_789",
    parent_call_id="parent_call_101"
)`}/>

#### Chunk Event
Contains streaming or intermediate results during execution (generators, async generators, streaming responses).
<CodeBlock language="python" code={`ChunkEvent(
    run_id="run_123",
    parent_run_id="parent_456",
    path="workflow.step1", 
    call_id="call_789",
    parent_call_id="parent_call_101",
    chunk="Processing item 1..."
)`}/>


#### Output Event 
Contains the final result and execution metadata.

<CodeBlock language="python" code={`OutputEvent(
    run_id="run_123",
    parent_run_id="parent_456",
    path="workflow.step1",
    call_id="call_789", 
    parent_call_id="parent_call_101",
    input={"name": "Alice"},
    output="Hello, Alice!",
    error=None,
    t0=1640995200000,  # Start timestamp
    t1=1640995201000,  # End timestamp
    usage={"tokens": 150}
)`}/>




## Practical Examples
#### Collect Final Result
<CodeBlock language="python" code={`# Get just the final result
result = await runnable(**params).collect()
print(f"Final result: {result}")`}/>

#### Handle Errors
<CodeBlock language="python" code={`async for event in runnable(**params):
    if isinstance(event, OutputEvent) and event.error:
        print(f"Error: {event.error['message']}")
        print(f"Traceback: {event.error['traceback']}")`}/>


