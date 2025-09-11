---
title: Tracing & Observability
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Tracing & Observability

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Comprehensive execution tracing with input/output/error/timing capture for complete observability
</h2>

---

Timbal provides comprehensive execution tracing that captures every aspect of runnable execution, from input parameters to final outputs, including timing data, error states, and usage metrics. This tracing system enables complete observability into your application's behavior and performance.


## Traces

A <span style={{color: 'var(--timbal-purple)'}}><strong>Trace</strong></span> is the core data structure that captures execution information for every runnable execution, providing a complete audit trail of what happened.

### Events vs. Traces:
Output Events and Traces have a similar data, but serve different purposes. Events are in fact the source data that gets processed into persistent Trace records.

- <span style={{color: 'var(--timbal-purple)'}}><strong>Events</strong></span> are real-time notifications that stream during execution. They provide immediate feedback about what's happening (start, progress chunks, completion) but are not stored permanently. Events are consumed as they're generated and are ideal for real-time monitoring, progress tracking, and streaming responses.

- <span style={{color: 'var(--timbal-purple)'}}><strong>Traces</strong></span> are persistent records that capture the complete execution context. **They contain detailed execution metadata, timing information, resource usage, and complete input/output snapshots**.

**Memory Management and Persistence**: Traces persistence enable memory management by tracking resource usage patterns, maintaining execution history for debugging, and supporting long-term performance analysis and audit trails.

<!-- ## API
#### Execution Identification

- **`path`**: The unique path identifier of the runnable being executed
- **`call_id`**: Unique identifier for this specific execution instance
- **`parent_call_id`**: Links to the parent execution in hierarchical workflows (None for root executions)

#### Timing Information

- **`t0`**: Start timestamp (Unix timestamp in milliseconds)
- **`t1`**: End timestamp (None if execution hasn't completed)

#### Input/Output Capture

- **`input`**: The input parameters passed to the runnable
- **`output`**: The result returned by the runnable (None if not completed or error occurred)
- **`error`**: Any exception or error that occurred during execution (None if successful)

#### Usage and Metadata

- **`usage`**: Resource usage metrics (e.g., token counts, API calls, memory usage)
- **`metadata`**: Flexible storage for custom metrics, tags, or execution-specific data -->


<!-- ---

## Trace Lifecycle

### 1. Execution Start
When a runnable begins execution, a Trace is created with:
- `path`, `call_id`, `parent_call_id` set
- `t0` timestamp recorded
- `input` captured (may be None if input gathering fails)

### 2. During Execution
The trace remains in an "in progress" state with:
- `t1`, `output`, `error` all set to None
- `usage` and `metadata` can be updated as execution progresses

### 3. Execution Completion
When execution finishes (successfully or with error):
- `t1` timestamp recorded
- `output` set to the result (if successful)
- `error` set to the exception (if failed)
- `usage` finalized with final metrics -->

---

## Accessing Traces

After each runnable execution completes, a trace is automatically generated and stored within the RunContext. You can access the current trace at any point during execution using `get_run_context().tracing`, which provides direct access to the trace's metadata, usage metrics, and execution details.

<CodeBlock language="python" code={`from timbal.state import get_run_context

# Access the current trace
trace = get_run_context().tracing`}/>

---

## Handling Errors with Traces

<CodeBlock language="python" code={`from timbal import Tool 
from timbal.state import get_run_context


def validate_data(data: str) -> str:  
    # Data validation here
    pass

# The trace automatically captures any errors
tool = Tool(
    name="validator",
    handler=validate_data
)

try:
    result = await tool(data="hi").collect()
except Exception:
    # Access trace to see error details
    trace = get_run_context().tracing
    print(f"Error: {trace.error}")`}/>
