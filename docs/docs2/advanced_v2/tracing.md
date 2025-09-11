---
title: Tracing and Events
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Tracing and Events

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Timbal's tracing system provides complete visibility into execution flows, capturing detailed information for monitoring, debugging, and analysis.
</h2>

---

The tracing system consists of three main components:
- **Events**: Generated during execution to capture state changes
- **Collectors**: Process events and manage state accumulation  
- **Traces**: Store complete execution information including timing, usage, and results

## Events

Events are emitted during execution to capture key moments and data:

### Core Event Types

- **StartEvent**: Emitted when execution begins
- **ChunkEvent**: Contains streaming response chunks
- **OutputEvent**: Final execution result with metadata

These events are found when making async calls to agents or workflows.

## Collectors

Collectors process events and accumulate results based on event types.

### Built-in Collectors

**TimbalCollector** - Automatically selected for Timbal events. Extracts final output from OutputEvent.

**DefaultCollector** - Fallback for any event type. Collects all events in a list when no specific collector matches.

### Custom Collectors

<CodeBlock language="python" code={`
from timbal.collectors.base import EventCollector
from timbal.collectors import register_collector

@register_collector
class CustomCollector(EventCollector):
    def __init__(self, run_context):
        super().__init__(run_context)
        self._result = None
    
    @classmethod
    def can_handle(cls, event):
        return hasattr(event, 'custom_field')
    
    def process(self, event):
        # Process event and update internal state
        self._result = event.custom_field
        return event
    
    def collect(self):
        return self._result
`}/>

## Tracing

Traces store complete execution information organized by call hierarchy.

### RunContext

The `RunContext` manages execution state and tracing:

<CodeBlock language="python" code={`from timbal.state import get_run_context

# Access current execution context
context = get_run_context()
if context:
    # Update usage metrics
    context.update_usage("tokens", 150)
    
    # Access trace data
    traces = context.tracing
`}/>

### Data Sharing

<CodeBlock language="python" code={`# Share data between execution steps
context.set_data("user_id", "12345")
user_id = context.get_data("user_id")

# Reference data from other steps
parent_data = context.get_data("..output")  # Parent output
sibling_data = context.get_data("step_name.shared.key")  # Sibling data
`}/>

## Tracing Providers

Providers handle trace storage and retrieval.

### Built-in Providers

**InMemoryTracingProvider** - Default in-memory storage

**PlatformTracingProvider** - Stores traces on Timbal platform

Configuration is automatic based on environment variables:
- `TIMBAL_API_HOST`, `TIMBAL_API_KEY` - Platform configuration
- `TIMBAL_ORG_ID`, `TIMBAL_APP_ID` - Organization context