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

## The Trace Model

The <span style={{color: 'var(--timbal-purple)'}}><strong>Trace</strong></span> model is the core data structure that captures comprehensive execution information for every runnable execution. It provides a complete audit trail of what happened during execution.

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
- **`metadata`**: Flexible storage for custom metrics, tags, or execution-specific data


---

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
- `usage` finalized with final metrics

---

## Usage Examples

### Basic Trace Access

<CodeBlock language="python" code={`from timbal import Tool, get_run_context

def analyze_data(data: str) -> dict:
    """Analyze some data and return results."""
    context = get_run_context()
    
    # Access the current trace
    trace = context.trace
    
    # Add custom metadata
    trace.metadata["data_size"] = len(data)
    trace.metadata["analysis_type"] = "basic"
    
    return {"status": "analyzed", "data": data}

tool = Tool(handler=analyze_data, name="analyzer")
result = await tool(data="sample data").collect()`}/>

### Hierarchical Tracing

<CodeBlock language="python" code={`def parent_function(data: str) -> str:
    """Parent function that calls child functions."""
    context = get_run_context()
    parent_trace = context.trace
    
    # Parent trace metadata
    parent_trace.metadata["parent_operation"] = "data_processing"
    
    # Call child function - creates new trace with parent_call_id
    child_result = await child_function(data).collect()
    
    return f"Parent processed: {child_result}"

def child_function(data: str) -> str:
    """Child function with its own trace."""
    context = get_run_context()
    child_trace = context.trace
    
    # Child trace will have parent_call_id linking to parent
    child_trace.metadata["child_operation"] = "data_transformation"
    
    return f"Child processed: {data}"`}/>

### Error Tracing

<CodeBlock language="python" code={`def risky_operation(data: str) -> str:
    """Function that might fail."""
    try:
        # Some operation that might fail
        if len(data) < 5:
            raise ValueError("Data too short")
        return f"Processed: {data}"
    except Exception as e:
        # The trace will automatically capture the error
        context = get_run_context()
        context.trace.metadata["error_context"] = "validation_failed"
        raise  # Re-raise to let the trace system handle it

tool = Tool(handler=risky_operation, name="risky_tool")

try:
    result = await tool(data="hi").collect()
except Exception:
    # Access the trace to see what went wrong
    context = get_run_context()
    trace = context.trace
    print(f"Error in {trace.path}: {trace.error}")
    print(f"Error metadata: {trace.metadata}")`}/>

---

## Advanced Trace Features

### Custom Usage Tracking

<CodeBlock language="python" code={`def api_call_with_tracking(url: str) -> dict:
    """Make an API call and track usage."""
    import requests
    
    context = get_run_context()
    trace = context.trace
    
    # Track API call metrics
    response = requests.get(url)
    
    # Update usage metrics
    trace.usage["api_calls"] = trace.usage.get("api_calls", 0) + 1
    trace.usage["response_size"] = len(response.content)
    trace.usage["status_code"] = response.status_code
    
    # Add custom metadata
    trace.metadata["endpoint"] = url
    trace.metadata["response_time_ms"] = response.elapsed.total_seconds() * 1000
    
    return response.json()`}/>

### Trace Serialization for Storage

<CodeBlock language="python" code={`def store_trace_for_analysis(trace: Trace) -> None:
    """Store trace data for later analysis."""
    # Serialize the trace (handles complex objects via _input_dump/_output_dump)
    trace_data = trace.model_dump()
    
    # Store in database, file, or send to monitoring system
    print(f"Stored trace {trace.call_id}:")
    print(f"  Path: {trace_data['path']}")
    print(f"  Duration: {trace_data['t1'] - trace_data['t0']}ms")
    print(f"  Success: {trace_data['error'] is None}")
    print(f"  Usage: {trace_data['usage']}")
    print(f"  Metadata: {trace_data['metadata']}")`}/>

---

## Integration with Monitoring Systems

The Trace model is designed to integrate seamlessly with monitoring and observability platforms:

- **OpenTelemetry**: Export traces to OpenTelemetry-compatible systems
- **Custom Analytics**: Use trace data for custom performance analysis
- **Debugging**: Leverage complete execution history for troubleshooting
- **Performance Monitoring**: Track timing and resource usage across executions

---