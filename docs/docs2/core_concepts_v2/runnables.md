---
title: Runnables
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Runnables

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
The fundamental building blocks that power tools, agents and workflows in Timbal.
</h2>

---

## What are Runnables?
A <span style={{color: 'var(--timbal-purple)'}}><strong>Runnable</strong></span> is an executable unit capable of processing inputs and producing outputs through an async generator interface. They work as a wrapper that turns any callable into a standarized, traceable, and composable execution unit. It supports both sync and async execution patterns.


### Class Parameters
- `name`: Required unique identifier.
- `description`: Optional description for LLM tool schemas.

#### Schema Control and Runtime Configuration
- `schema_params_mode`: Controls parameter visibility ("all" vs "required")
- `schema_include_params`: Explicitly include specific parameters
- `schema_exclude_params`: Exclude specific parameters from schemas
- `default_params`: Runtime parameter injection (both static and callable values)

#### Execution Hooks
- `pre_hook`: Pre-execution hook that runs before the main handler.
- `post_hook`: Post-execution hook that runs after the main handler.


### Types of Runnables
Runnables provide a unified interface regardless of the underlying implementation. Whether it is a simple function, a complex AI agent, or an entire workflow, they all follow the same execution pattern and can be composed together seamlessly.

#### Functions


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

#### Workflow as Runnables
#### Agents as Runnables
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
)`}/>




### Key points
- Any callable can be used as a runnable




<!-- TODO: explain in Events page -->
## The `Collect()` Method
The `collect()` method is a key feature of the Runnable system that allows you to easily extract the final result from an async generator without manually iterating through all the events.

<CodeBlock language="python" code={`# Instead of manually iterating through events
async for event in runnable(**params):
    if isinstance(event, OutputEvent):
        result = event.output
        break

# You can simply use collect()
result = await runnable(**params).collect()`}/>






