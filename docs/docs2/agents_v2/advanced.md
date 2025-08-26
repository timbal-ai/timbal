---
title: Advanced
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Advanced Agent Concepts

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master advanced patterns for Agent including memory management, nested execution, custom schemas, and performance optimization.
</h2>

---

## Runnable Architecture

Agent is built on the powerful Runnable base class, which provides a unified interface for all executable components in the Timbal framework.

### Understanding Runnables

<CodeBlock language="python" code ={`from timbal.core import Runnable, Agent, Tool

# All of these are Runnables:
agent = Agent(name="my_agent", model="gpt-4")
tool = Tool(handler=my_function)

# They all share the same execution interface:
async for event in agent(**params):
    print(event)

# And they all support .collect() for final results:
result = await agent(**params).collect()`}/>

### Execution Characteristics

Runnables automatically detect and adapt to different execution patterns:

<CodeBlock language="python" code ={`# Synchronous function
def sync_function(x: int) -> int:
    return x * 2

# Asynchronous function  
async def async_function(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * 2

# Generator function
def gen_function(count: int):
    for i in range(count):
        yield i

# Async generator function
async def async_gen_function(count: int):
    for i in range(count):
        await asyncio.sleep(0.1)
        yield i

# All are automatically handled by Tool
tools = [
    Tool(handler=sync_function),      # Executed in thread pool
    Tool(handler=async_function),     # Executed directly in async context
    Tool(handler=gen_function),       # Converted to async generator
    Tool(handler=async_gen_function)  # Executed as async generator
]`}/>

## Advanced Parameter Management

### Custom Parameter Models

You can create custom parameter models for complex validation:

<CodeBlock language="python" code ={`from pydantic import BaseModel, Field
from typing import Literal

class WeatherParams(BaseModel):
    location: str = Field(description="City or location name")
    unit: Literal["celsius", "fahrenheit"] = Field(default="celsius")
    include_forecast: bool = Field(default=False)
    days: int = Field(default=1, ge=1, le=7, description="Number of forecast days")

def get_weather_advanced(**kwargs) -> str:
    # Function receives validated parameters
    params = WeatherParams.model_validate(kwargs)
    return f"Weather for {params.location} in {params.unit}"

# Tool automatically uses the parameter annotations
tool = Tool(handler=get_weather_advanced)`}/>

### Dynamic Parameter Filtering

Control parameter visibility dynamically:

<CodeBlock language="python" code ={`class ConditionalTool(Tool):
    def format_params_model_schema(self) -> dict:
        schema = super().format_params_model_schema()
        
        # Conditionally exclude parameters based on context
        if self.get_run_context().data.get("simple_mode"):
            schema["properties"] = {
                k: v for k, v in schema["properties"].items()
                if k in ["location"]  # Only show essential params
            }
        
        return schema`}/>

## Memory and Context Management

### Agent Memory Resolution

Agent automatically resolves memory from parent contexts:

<CodeBlock language="python" code ={`# Parent agent
parent = Agent(
    name="coordinator",
    model="gpt-4",
    tools=[child_agent]  # Child agent as tool
)

# When child_agent is called, it automatically:
# 1. Retrieves conversation history from parent
# 2. Maintains context across nested calls
# 3. Preserves tool call chains

result = await parent(
    prompt=Message(role="user", content="Use the child agent to analyze data")
).collect()`}/>

### Custom Memory Patterns

Implement custom memory resolution:

<CodeBlock language="python" code ={`class MemoryAwareAgent(Agent):
    async def _resolve_memory(self) -> list[Message]:
        memory = await super()._resolve_memory()
        
        # Add custom memory logic
        if self.should_include_system_memory():
            system_memory = await self.load_system_memory()
            memory = system_memory + memory
            
        return memory
    
    def should_include_system_memory(self) -> bool:
        # Custom logic for when to include system memory
        return self.get_run_context().data.get("include_system_context", False)`}/>

## System Prompt Templates

### Advanced Template Patterns

<CodeBlock language="python" code ={`agent = Agent(
    name="template_agent",
    model="claude-3-sonnet",
    system_prompt="You are a data analyst specialized in financial markets.",
)

# Use the agent with the configured system prompt
result = await agent(
    prompt=Message(role="user", content="Help me with analysis")
).collect()`}/>

### Template Factories

Create reusable template patterns:

<CodeBlock language="python" code ={`class TemplateFactory:
    @staticmethod
    def create_analytical_context(instructions: str, domain: str) -> str:
        return f"""
You are an expert analyst specializing in {domain}.

Core Instructions: {instructions}

Analysis Framework:
1. Identify key patterns and trends
2. Provide data-driven insights
3. Suggest actionable recommendations
4. Highlight potential risks or limitations

Always structure your response with clear sections and supporting evidence.
"""

    @staticmethod
    def create_creative_context(instructions: str, style: str) -> str:
        return f"""
You are a creative professional with expertise in {style}.

Core Instructions: {instructions}

Creative Process:
1. Understand the creative brief
2. Generate innovative concepts
3. Provide detailed explanations
4. Suggest variations and alternatives

Focus on originality while maintaining practical feasibility.
"""

# Usage
agent = Agent(
    name="versatile_agent",
    model="gpt-4",
    system_prompt="Help users with their requests"
)

# Use the agent
result = await agent(
    prompt=Message(role="user", content="Analyze market trends")
).collect()`}/>

## Concurrent Tool Execution

### Understanding Tool Multiplexing

Agent executes multiple tool calls concurrently:

<CodeBlock language="python" code ={`# When LLM decides to call multiple tools:
# [
#   {"name": "get_weather", "input": {"location": "NYC"}},
#   {"name": "get_news", "input": {"topic": "weather"}},
#   {"name": "send_alert", "input": {"message": "Check conditions"}}
# ]

# All three tools execute simultaneously, not sequentially
# Total execution time ≈ max(individual_times), not sum(individual_times)`}/>

### Custom Tool Orchestration

For advanced control over tool execution:

<CodeBlock language="python" code ={`class OrchestrationAgent(Agent):
    async def _multiplex_tools(self, _parent_call_id, tool_calls):
        # Group tools by priority or dependency
        high_priority = [tc for tc in tool_calls if tc.name.startswith("urgent_")]
        normal_priority = [tc for tc in tool_calls if tc not in high_priority]
        
        # Execute high priority tools first
        if high_priority:
            async for tool_call, event in super()._multiplex_tools(_parent_call_id, high_priority):
                yield tool_call, event
        
        # Then execute normal priority tools
        if normal_priority:
            async for tool_call, event in super()._multiplex_tools(_parent_call_id, normal_priority):
                yield tool_call, event`}/>

## Custom Event Handling

### Event-Driven Architectures

<CodeBlock language="python" code ={`from timbal.types.events import StartEvent, ChunkEvent, OutputEvent

class EventProcessorAgent(Agent):
    async def handler(self, **kwargs):
        # Custom event processing
        async for event in super().handler(**kwargs):
            # Intercept and process events
            if isinstance(event, ChunkEvent):
                # Transform streaming content
                event.chunk = self.process_chunk(event.chunk)
            elif isinstance(event, OutputEvent):
                # Add metadata to final output
                event.output.metadata = {"processed_by": self.name}
            
            yield event
    
    def process_chunk(self, chunk):
        # Custom chunk processing logic
        return chunk.upper() if isinstance(chunk, str) else chunk`}/>

## Performance Optimization

### Execution Profiling

<CodeBlock language="python" code ={`import time
from timbal.state import get_run_context

class ProfilingAgent(Agent):
    async def handler(self, **kwargs):
        start_time = time.time()
        run_context = get_run_context()
        
        # Add profiling data to context
        run_context.data["profiling"] = {
            "start_time": start_time,
            "tool_calls": 0,
            "llm_calls": 0
        }
        
        async for event in super().handler(**kwargs):
            # Track metrics
            if isinstance(event, OutputEvent):
                if "llm" in event.path:
                    run_context.data["profiling"]["llm_calls"] += 1
                else:
                    run_context.data["profiling"]["tool_calls"] += 1
            
            yield event
        
        # Log performance metrics
        total_time = time.time() - start_time
        print(f"Execution completed in {total_time:.2f}s")
        print(f"Metrics: {run_context.data['profiling']}")`}/>

### Resource Management

<CodeBlock language="python" code ={`class ResourceManagedAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrent_tools = kwargs.get("max_concurrent_tools", 5)
        self.tool_semaphore = asyncio.Semaphore(self.max_concurrent_tools)
    
    async def _enqueue_tool_events(self, _parent_call_id, tool_call, queue):
        # Limit concurrent tool execution
        async with self.tool_semaphore:
            await super()._enqueue_tool_events(_parent_call_id, tool_call, queue)`}/>

## Integration Patterns

### Custom LLM Routers

<CodeBlock language="python" code ={`from timbal.core.handlers.llm_router import llm_router

class CustomAgent(Agent):
    def model_post_init(self, __context):
        super().model_post_init(__context)
        
        # Customize LLM routing
        self._llm = Tool(
            name="custom_llm",
            handler=self.custom_llm_handler
        )
        self._llm.nest(self._path)
    
    async def custom_llm_handler(self, **kwargs):
        # Custom LLM logic before delegating to router
        if kwargs.get("model", "").startswith("custom-"):
            # Handle custom model routing
            return await self.handle_custom_model(**kwargs)
        else:
            # Delegate to standard router
            return await llm_router(**kwargs)`}/>

### Agent Composition

<CodeBlock language="python" code ={`class SpecializedAgent(Agent):
    def __init__(self, specialist_type: str, **kwargs):
        super().__init__(**kwargs)
        self.specialist_type = specialist_type
    
    def model_post_init(self, __context):
        super().model_post_init(__context)
        # Add specialized tools based on type
        if self.specialist_type == "data_analyst":
            self.tools.extend([
                Tool(handler=analyze_data),
                Tool(handler=create_visualization),
                Tool(handler=generate_report)
            ])
        elif self.specialist_type == "content_creator":
            self.tools.extend([
                Tool(handler=generate_text),
                Tool(handler=create_image),
                Tool(handler=optimize_content)
            ])

# Create specialized agents
data_agent = SpecializedAgent(
    name="data_specialist",
    model="gpt-4",
    specialist_type="data_analyst"
)

content_agent = SpecializedAgent(
    name="content_specialist", 
    model="claude-3-sonnet",
    specialist_type="content_creator"
)`}/>

## Error Handling and Recovery

### Robust Error Patterns

<CodeBlock language="python" code ={`class ResilientAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)
    
    async def handler(self, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                async for event in super().handler(**kwargs):
                    yield event
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == self.max_retries:
                    # Final attempt failed, re-raise
                    raise
                else:
                    # Wait before retry
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    print(f"Retrying agent execution (attempt {attempt + 2}/{self.max_retries + 1})")`}/>

## Testing Patterns

### Agent Testing

<CodeBlock language="python" code ={`import pytest
from timbal.core import Agent, Tool
from timbal.types.message import Message

@pytest.mark.asyncio
async def test_agent_execution():
    def mock_tool(query: str) -> str:
        return f"Mock result for: {query}"
    
    agent = Agent(
        name="test_agent",
        model="gpt-4",
        tools=[Tool(handler=mock_tool)]
    )
    
    result = await agent(
        prompt=Message(role="user", content="Test query")
    ).collect()
    
    assert isinstance(result.output, Message)
    assert result.output.role == "assistant"

@pytest.mark.asyncio 
async def test_tool_validation():
    def typed_tool(count: int, name: str) -> str:
        return f"Processed {count} items for {name}"
    
    tool = Tool(handler=typed_tool)
    
    # Test valid input
    result = await tool(count=5, name="test").collect()
    assert result.output == "Processed 5 items for test"
    
    # Test invalid input
    with pytest.raises(ValidationError):
        await tool(count="invalid", name="test").collect()`}/>

## Summary

Agent's advanced features enable:

- **Flexible Architecture**: Built on the extensible Runnable base class
- **Sophisticated Parameter Management**: Custom validation and dynamic filtering
- **Memory Management**: Automatic context resolution and custom patterns
- **Performance Optimization**: Concurrent execution and resource management
- **Robust Error Handling**: Retry patterns and graceful degradation
- **Extensibility**: Custom components and integration patterns

These advanced patterns allow you to build production-ready AI systems with reliability, performance, and maintainability in mind.
