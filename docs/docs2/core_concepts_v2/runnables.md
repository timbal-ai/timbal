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

## API

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
- `post_hook` (Callable | None = None) - Hook function executed after the main handler completes


<!-- #### Functions

#### Tool Objects

<CodeBlock language="python" code={`from timbal import Tool

def calculate_tax(amount: float, rate: float = 0.1) -> float:
    """Calculate tax on an amount."""
    return amount * rate

tax_tool = Tool(
    handler=calculate_tax,
    description="Calculate tax on a monetary amount",
    schema_params_mode="required"  # Only show required params to LLM
)`}/>

#### Workflow
<CodeBlock language="python" code={`from timbal import Workflow

# Create a Workflow
analysis_agent = Workflow(

)

# Use agent as runnable in workflow
workflow = (
    Workflow(name="analysis_pipeline")
    .step(analysis_agent)  # Agent as runnable
)`}/>

#### Agents
<CodeBlock language="python" code={`from timbal import Agent

# Create an agent
analysis_agent = Agent(
    name="data_analyzer",
    model="openai/gpt-4o-mini",
    system_prompt="Analyze the provided data and return insights."
)

# Use agent as runnable in workflow
workflow = (
    Workflow(name="analysis_pipeline")
    .step(analysis_agent)  # Agent as runnable
)`}/> -->







<!-- ## Core Features

### Base Execution Unit with Parameter Validation and Schema Generation

Runnables serve as the fundamental execution units in Timbal, providing robust parameter validation and automatic schema generation capabilities.

#### Parameter Validation
Every Runnable uses Pydantic models for input validation, ensuring type safety and data integrity:

<CodeBlock language="python" code={`from timbal import Tool
from pydantic import BaseModel

class UserInput(BaseModel):
    name: str
    age: int
    email: str | None = None

def process_user(user: UserInput) -> dict:
    return {"processed": True, "user": user.model_dump()}

# The Tool automatically validates inputs against the UserInput model
user_tool = Tool(
    handler=process_user,
    name="process_user",
    description="Process user information"
)

# This will validate the input and raise ValidationError if invalid
result = await user_tool(name="John", age=25, email="john@example.com").collect()`}/>

#### Schema Generation
Runnables automatically generate JSON schemas for LLM tool calling, supporting both OpenAI and Anthropic formats:

<CodeBlock language="python" code={`# OpenAI-compatible schema
openai_schema = user_tool.openai_schema
# Returns: {"type": "function", "function": {"name": "process_user", ...}}

# Anthropic-compatible schema  
anthropic_schema = user_tool.anthropic_schema
# Returns: {"name": "process_user", "input_schema": {...}}

# Formatted schema with parameter filtering
filtered_schema = user_tool.format_params_model_schema()`}/> -->

---
## Execution Hooks

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
Executes after the main handler completes, useful for cleanup, result processing, or notifications:

<CodeBlock language="python" code={`def cleanup_resources():
    """Clean up resources after execution."""
    # Cleanup logic here
    pass

tool = Tool(
    handler=my_handler,
    post_hook=cleanup_resources
)`}/>

---
## Parameter Injection and Transformation

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
result = await tool(data="test").collect()`}/>


### Parameter Override Behavior

Injected parameters can be overridden by explicitly passing them during execution:

<CodeBlock language="python" code={`tool = Tool(
    handler=my_handler,
    default_params={"timeout": 30, "retries": 3}
)

# Override the default timeout but keep retries
result = await tool(data="test", timeout=60).collect()
# Final call: my_handler(data="test", timeout=60, retries=3)`}/>

---
## Nested Execution Patterns for Complex Workflows

Runnables support hierarchical execution patterns where runnables can call other runnables, creating complex execution graphs.

#### Example: Nested Tool Execution

<CodeBlock language="python" code={`def data_processor(data_path: str) -> str:
    """Process data"""
    result = await data_loader(data_path).collect()
    # Data processing logic here
    return f"Processed: {result}"

def data_loader(data_path: str) -> str:
    """Loads data"""
    # Data loading logic here
    return f"Further processed: {processed_data}"


# Create tools
loader_tool = Tool(handler=data_loader, name="loader")
processor_tool = Tool(handler=data_processor, name="processor")

# The execution trace will show the nested call hierarchy
result = await processor_tool(data_path="test.csv").collect()`}/>

<!-- #### Workflow Orchestration
Workflows can orchestrate multiple runnables in complex patterns:

<CodeBlock language="python" code={`from timbal import Workflow

def step1(data: str) -> str:
    return f"Step 1: {data}"

def step2(data: str) -> str:
    return f"Step 2: {data}"

def step3(data: str) -> str:
    return f"Step 3: {data}"

# Create a workflow with nested execution
workflow = (
    Workflow(name="nested_workflow")
    .step(step1)           # First step
    .step(step2)           # Second step  
    .step(step3)           # Third step
)

# Each step can access the full execution context
result = await workflow(data="input").collect()`}/>

#### Context Propagation
Context automatically flows through nested executions:

<CodeBlock language="python" code={`def parent_handler(data: str) -> str:
    """Parent handler that sets context data."""
    context = get_run_context()
    context.data["parent_data"] = "from_parent"
    
    # Call child handler - context is preserved
    child_result = await child_handler(data=data).collect()
    
    return f"Parent + {child_result}"

def child_handler(data: str) -> str:
    """Child handler that accesses parent context."""
    context = get_run_context()
    parent_data = context.data.get("parent_data", "not_found")
    
    return f"Child: {data} (parent: {parent_data})"

# Context flows from parent to child automatically
parent_tool = Tool(handler=parent_handler)
result = await parent_tool(data="test").collect()`}/> -->
