---
title: Tracing and Events
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Tracing and Events

Timbal's tracing system provides complete visibility into agent and workflow execution, enabling monitoring, debugging, and analysis of AI system behavior.

## Events

Events are the foundation of the tracing system. Each execution generates a sequence of events that capture all relevant information:

### Event Types

- **StartEvent**: Marks the beginning of an execution
- **ChunkEvent**: Captures response fragments (especially useful for streaming)
- **OutputEvent**: Contains the final result and execution metadata

<CodeBlock language="python" code={`
from timbal.types.events import StartEvent, ChunkEvent, OutputEvent

# Events are automatically generated during execution
async for event in agent(prompt="Hello"):
    if isinstance(event, StartEvent):
        print(f"Starting execution: {event.run_id}")
    elif isinstance(event, ChunkEvent):
        print(f"Chunk: {event.content}")
    elif isinstance(event, OutputEvent):
`}/>

## Collectors

Collectors process and transform events according to specific needs. Timbal includes several predefined collectors:

### TimbalCollector

Processes native Timbal events and extracts the final result:

<CodeBlock language="python" code={`
from timbal.collectors.impl.timbal import TimbalCollector

# Automatically used for Timbal events
collector = TimbalCollector(run_context)
result = collector.collect()  # Returns the final result
`}/>

### OpenAICollector

Manages OpenAI streaming events, accumulating content and tool calls:

<CodeBlock language="python" code={`
from timbal.collectors.impl.openai import OpenAICollector

collector = OpenAICollector(run_context)
# Processes OpenAI streaming events
# Accumulates content and tool calls
`}/>

### DefaultCollector

Fallback collector that handles any event type:

<CodeBlock language="python" code={`
from timbal.collectors.impl.default import DefaultCollector

collector = DefaultCollector(run_context)
# Captures all events in an array
events = collector.collect()
`}/>

### Custom Collectors

You can create custom collectors by inheriting from `EventCollector`:

<CodeBlock language="python" code={`
from timbal.collectors.base import EventCollector
from timbal.collectors import register_collector

@register_collector
class CustomCollector(EventCollector):
    def __init__(self, run_context):
        super().__init__(run_context)
        self._processed_events = []
    
    @classmethod
    def can_handle(cls, event):
        return hasattr(event, 'custom_field')
    
    def process(self, event):
        # Process the event and update internal state
        self._processed_events.append(event)
        return event
    
    def collect(self):
        return self._processed_events
`}/>

## Tracing Savers

Tracing savers manage the storage and retrieval of execution traces:

### InMemoryTracingProvider

Default provider that stores traces in memory:

<CodeBlock language="python" code={`
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider

# Automatically used when no specific configuration is provided
provider = InMemoryTracingProvider()
`}/>

### Custom Providers

You can implement custom providers by inheriting from `TracingProvider`:

<CodeBlock language="python" code={`
from timbal.state.tracing.providers.base import TracingProvider

class DatabaseTracingProvider(TracingProvider):
    @classmethod
    async def get(cls, run_id: str):
        # Retrieve traces from database
        pass
    
    @classmethod
    async def put(cls, run_context):
        # Store traces in database
`}/>

## Run Context and Tracing

The `RunContext` manages the global execution state and traces:

<CodeBlock language="python" code={`
from timbal.state import RunContext, get_run_context

# Create an execution context
context = RunContext()
context.tracing  # Contains all execution traces

# Access the current context
current_context = get_run_context()
if current_context:
    # Update usage statistics
    current_context.update_usage("tokens", 150)
    
    # Save traces
    await current_context.save_tracing()
`}/>