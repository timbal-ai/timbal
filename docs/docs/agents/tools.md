---
title: Tools
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Understanding Tools

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Give your agents enhanced superpowers with automatic schema generation, improved parameter handling, and robust execution patterns.
</h2>

---

Tools are the enhanced way to give your Agents superpowers: they allow your agent to interact with the outside world with improved reliability, automatic schema generation, better parameter validation, and concurrent execution when multiple tools are called simultaneously.

## What is a Tool?

A **Tool** is a `Runnable` that wraps a callable function or method with automatic introspection and enhanced execution capabilities. Tools provide:

- **Automatic Schema Generation**: Function signatures are automatically converted to JSON schemas
- **Parameter Validation**: Input validation using Pydantic models
- **Execution Flexibility**: Support for sync, async, generator, and async generator functions
- **Enhanced Configuration**: Fine-grained control over parameter exposure and behavior

## How to Create a Tool

### Step 1: Define Your Functions

First, create the functions that will become your tools:

<CodeBlock language="python" code ={`def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a specific location."""
    return f"The weather in {location} is 22°{unit[0].upper()}"

async def search_web(query: str, max_results: int = 10) -> list[str]:
    """Search the web and return results."""
    # Simulate web search
    return [f"Result {i}: {query}" for i in range(max_results)]`}/>

### Step 2: Use Functions as Tools

Once you have your functions, you can use them in three ways:

#### Option A: Automatic Tool Creation (Simplest)

Functions are automatically wrapped as Tool instances:

<CodeBlock language="python" code ={`from timbal.core_v2 import Agent

agent = Agent(
    name="helper_agent",
    model="anthropic/claude-3-sonnet",
    tools=[
        get_weather,    # Automatically becomes a Tool
        search_web     # Async functions work seamlessly
    ]
)`}/>

#### Option B: Explicit Tool Creation (More Control)

For more control over tool behavior, create Tool instances explicitly:

<CodeBlock language="python" code ={`from timbal.core_v2 import Tool

weather_tool = Tool(
    name="get_weather",  # Optional: auto-generated from function name if not provided
    handler=get_weather,
    description="Get weather information for any location",
    params_mode="required",  # Only show required parameters
    exclude_params=["unit"]  # Hide specific parameters
)

agent = Agent(
    name="helper_agent",
    model="anthropic/claude-3-sonnet",
    tools=[weather_tool]
)`}/>

#### Option C: Dictionary Configuration

You can also use dictionaries for tool configuration:

<CodeBlock language="python" code ={`weather_tool = {
    "name": "get_weather",  # Optional: auto-generated from function name if not provided
    "handler": get_weather,
    "description": "Get weather information for any location",
    "params_mode": "all",
    "include_params": ["location", "unit"],
    "fixed_params": {"unit": "fahrenheit"}  # Always use fahrenheit
}

agent = Agent(
    name="helper_agent",
    model="anthropic/claude-3-sonnet",
    tools=[weather_tool]
)`}/>

You can combine multiple tools with different configurations in a single agent!

## Function Type Support

Tool automatically handles all Python function types - synchronous, asynchronous, generators, and async generators. This means you can use any existing function as a tool without modification, and the framework will handle the execution details automatically.

<CodeBlock language="python" code ={`# All these work seamlessly as tools
def sync_function(): return "result"
async def async_function(): return "result"  
def generator_function(): yield "item"
async def async_generator(): yield "item"

# Tool V2 handles the execution automatically
tools = [sync_function, async_function, generator_function, async_generator]`}/>

## Customizing Tool Parameters

You can control which parameters are visible to the LLM and how they are described.

### Adding Descriptions

There are two ways to add descriptions to help the LLM understand your tools:

#### 1. Parameter-level descriptions
Use `Field` to describe individual parameters and provide choices:

<CodeBlock language="python" code ={`from timbal.types.field import Field

def get_weather(
    location: str = Field(description="The city or location to get weather for"),
    unit: str = Field(choices=["celsius", "fahrenheit"], description="Temperature unit preference")
) -> str:
    ...`}/>

The `choices` parameter restricts the LLM to only use the specified values, making your tools more reliable and predictable.

#### 2. Tool-level descriptions
Add a description to the tool itself:

<CodeBlock language="python" code ={`Tool(
    handler=get_weather,
    description="Get current weather information for any location"
)`}/>

Both types of descriptions help the LLM understand when and how to use your tools effectively.

### Controlling Tool Parameters

- **schema_params_mode**: `"all"` (default) exposes all params, `"required"` exposes only required ones.
- **schema_include_params**: List of param names to always include.
- **schema_exclude_params**: List of param names to exclude.

#### Only show required params

<CodeBlock language="python" code ={`Tool(
    handler=get_weather,
    description="Get the weather for a location",
    schema_params_mode="required"
)`}/>

#### Include extra params

<CodeBlock language="python" code ={`Tool(
    handler=get_weather,
    description="Get the weather for a location",
    schema_params_mode="required",
    # Even if not required, 'unit' will be shown
    schema_include_params=["unit"]  
)`}/>

#### Exclude params

<CodeBlock language="python" code ={`Tool(
    handler=get_weather,
    description="Get the weather for a location",
    # 'unit' will not be shown to the LLM
    schema_exclude_params=["unit"]  
)`}/>

### Execution Hooks

Add pre and post execution hooks to customize tool behavior:

<CodeBlock language="python" code ={`async def pre_hook(input_data: dict[str, Any]) -> None:
    # Log input, validate, or modify parameters
    print(f"Calling tool with: {input_data}")
    input_data["location"] = input_data["location"].title()

async def post_hook(output: Any) -> None:
    # Log output, transform, or trigger events
    print(f"Tool returned: {output}")

tool = Tool(
    handler=get_weather,
    pre_hook=pre_hook,
    post_hook=post_hook
)`}/>

## Schema Generation

Tool automatically generates schemas for LLM integration. Your function signatures are converted to JSON schemas that work with different LLM providers:

<CodeBlock language="python" code ={`tool = Tool(handler=get_weather)

# Get schemas for different providers
openai_schema = tool.openai_schema    # OpenAI format
anthropic_schema = tool.anthropic_schema  # Anthropic format`}/>

## Error Handling

Tool automatically captures and handles errors from your functions, making your agents more robust:

<CodeBlock language="python" code ={`def risky_operation(value: int) -> str:
    if value < 0:
        raise ValueError("Value must be positive")
    return f"Success: {value}"

# Errors are automatically captured and reported to the agent
tool = Tool(handler=risky_operation)`}/>

## Tool Nesting and Paths

Tools automatically manage hierarchical paths when nested within agents. This enables proper tracing and debugging by creating clear execution hierarchies:

<CodeBlock language="python" code ={`# Tools automatically inherit paths from parent agents
agent = Agent(
    name="parent_agent",
    model="anthropic/claude-3-sonnet",
    tools=[
        Tool(name="child_tool", handler=some_function)
    ]
)

# Tool path becomes: "parent_agent.child_tool"
# This creates clear execution traces for debugging`}/>

## Summary

Tool provides significant improvements over the original tools:

- **Automatic Introspection**: Function signatures become tool schemas automatically
- **Enhanced Validation**: Pydantic-based parameter validation
- **Execution Flexibility**: Support for all Python callable types
- **Better Configuration**: Fine-grained parameter control
- **Performance**: Concurrent execution and optimized patterns
- **Robustness**: Improved error handling and tracing

For more advanced patterns, see the [Integrations](/integrations) and explore the built-in tools in the Timbal library!
