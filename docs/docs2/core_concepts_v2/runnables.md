---
title: Runnables
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Runnables

<h2 className="subtitle" style={{marginTop: '0px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Executable primitives that provide consistent interfaces for all Timbal components.
</h2>

---

## What is a Runnable?
A <span style={{color: 'var(--timbal-purple)'}}><strong>Runnable</strong></span> is an executable unit capable of processing inputs and producing outputs through an async generator interface. It works as a wrapper that turns any callable into a standarized, traceable, and composable execution unit. It supports sync, async, sync generators, and async generators execution patterns.

All runnables provide a unified interface and execution pattern, enabling seamless composition regardless of their underlying implementation:

- **[Tools](../agents_v2/tools)** - Enhanced function wrappers with schema generation and parameter control
- **[Agents](../agents_v2/index.md)** - Autonomous execution units that orchestrate LLM interactions with tool calling
- **[Workflows](../workflows_v2/index.md)** - Programmable execution pipelines that orchestrate step-by-step processing


### Runnable identifier

**All runnables must have a unique `name` defined, as it serves as the identifier for the runnable within Timbal's execution engine**. This name is used for tracing, debugging, and referencing the runnable in workflows, agents, and other components. Whether you're creating a Tool object, an Agent, or a Workflow, the name parameter is required and must be unique within your application context.

Here's how to create a basic Runnable component:
<CodeBlock language="python" code={`from timbal import Tool


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

tool = Tool(
    name="addition_tool",
    handler=add_numbers,
)`}/>

---

## Parameter Handling and Execution

Runnables provide a unified parameter handling system that works consistently across all types of runnables (Tools, Agents, and Workflows). **All follow the same format for passing inputs**: you can pass parameters directly as keyword arguments when calling the runnable, and these parameters are automatically mapped to the underlying handler function's parameters.

<CodeBlock language="python" code={`# Runnable call
result = await addition_tool(a=5, b=3).collect()  # Returns 8`}/>


### Default Parameters
The system also allows for parameter through default values defined in the Runnable.
Injected parameters can be overridden by explicitly passing them during execution:

<CodeBlock language="python" highlight={"7"} code={`from timbal import Tool


tool = Tool(
    name="addition_tool",
    handler=add_numbers,
    default_params={"b": 2}
)

result = await addition_tool(a=5).collect()  # Returns 8`}/>

<!-- <CodeBlock language="python" code={`def add_with_config(a: float, b: float, multiplier: int = 1) -> float:
    """Add two numbers with optional multiplier."""
    return (a + b) * multiplier

tool = Tool(
    name="configurable_adder",
    handler=add_with_config,
    default_params={"multiplier": 2}
)

# Use default multiplier
result1 = await tool(a=5, b=3).collect()  # Returns (5+3)*2 = 16

# Override the default multiplier
result2 = await tool(a=5, b=3, multiplier=3).collect()  # Returns (5+3)*3 = 24`}/> -->



<!-- ## API

#### Core Configuration
- `name` (str) - Required unique identifier for the runnable
- `description` (str | None = None) - Optional description used for LLM tool schemas
- `metadata` (dict | None = None) - Optional metadata for additional information

#### Schema Generation & Tool Calling
- `schema_params_mode` (Literal["all", "required"] = "all") - Controls parameter visibility in generated schemas
- `schema_include_params` (list[str] | None = None) - Explicitly include specific parameters in schema
- `schema_exclude_params` (list[str] | None = None) - Explicitly exclude specific parameters from schema

#### Runtime Parameter Injection
- `default_params` (dict | None = None) - Inject parameters at runtime (supports both static values and callable functions)

#### Execution Hooks
- `pre_hook` (Callable | None = None) - Hook function executed before the main handler
- `post_hook` (Callable | None = None) - Hook function executed after the main handler completes -->


<!-- --- -->
<!-- ## Execution Hooks

Runnables support pre and post-execution hooks for implementing cross-cutting concerns like logging, monitoring, or data transformation.

#### Pre-Hook
Executes before the main handler runs, useful for setup, validation, or data preparation:

<CodeBlock language="python" code={`def log_start():
    """Log the start of execution."""
    print(f"Starting execution...")

tool = Tool(
    handler=my_handler,
    pre_hook=log_start
)`}/>

#### Post-Hook
Executes after the main handler completes, useful for cleanup, result processing, or notifications: -->

<!-- <CodeBlock language="python" code={`def cleanup_resources():
    """Clean up resources after execution."""
    # Cleanup logic here
    pass

tool = Tool(
    handler=my_handler,
    post_hook=cleanup_resources
)`}/> -->

<!-- ## Parameter Injection and Transformation

Runnables provide powerful parameter injection capabilities that allow you to automatically inject values into your handlers without explicitly passing them. When you call a Runnable, any parameters defined in `default_params` are automatically merged with the arguments you provide, ensuring all required parameters are available.

- **Static Parameter Injection**: Inject fixed values that are always included in every execution
- **Dynamic Parameter Injection**: Inject values computed at runtime using callable functions. This is perfect for context-dependent values, timestamps, or configuration that changes between executions.


<CodeBlock language="python" code={`import time

def get_timestamp():
    """Get current timestamp for this execution."""
    return int(time.time())

def process_with_context(data: str, user: str, timestamp: int, config: dict) -> str:
    """Process data with runtime-injected context."""
    return f"User {user} processed {data} at {timestamp}"

tool = Tool(
    handler=process_with_context,
    default_params={
        "user": "user_123",             # Static
        "timestamp": get_timestamp,     # Dynamic
    }
)

# All parameters are automatically resolved and injected
result = await tool(data="test").collect()`}/> -->




---
## Nesting Runnables

Runnables can be combined in various ways to create powerful execution patterns. 

#### Example: Agent with Tools

<CodeBlock language="python" code={`from timbal import Tool, Agent, Workflow


def add_numbers(a: float, b: float) -> float:
  """Add two numbers together."""
  return a + b

def calculate_product(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Wrap functions as tools
addition_tool = Tool(name="addition_tool", handler=calcuadd_numberslate_sum)
product_tool = Tool(name="product_tool", handler=calculate_product)

# Create an agent that can use these tools
math_agent = Agent(
    name="math_assistant",
    model="openai/gpt-4o-mini",
    tools=[addition_tool, product_tool],
    system_prompt="You are a math assistant. Use the available tools to perform calculations."
)`}/>
