---
title: Parallel Execution
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows often need to run multiple operations at the same time. These examples demonstrate how to execute steps or workflows concurrently and merge their results using Timbal's automatic parallel execution capabilities.

## Parallel execution using steps

In this example, the workflow runs `step1` and `step2` in parallel. Each step receives the same input and runs independently. Their outputs are automatically namespaced by step name and passed together to `step3`, which combines the results and returns the final value.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool

# Define step 1: passes value from input to output
def step1(value: int) -> dict:
    """First step that processes the input value."""
    return {"value": value}

# Define step 2: passes value from input to output  
def step2(value: int) -> dict:
    """Second step that processes the input value."""
    return {"value": value}

# Define step 3: sums values from step1 and step2
def step3(step1_output: dict, step2_output: dict) -> dict:
    """Combines results from parallel steps."""
    return {
        "value": step1_output["value"] + step2_output["value"]
    }

# Create the parallel workflow
parallel_workflow = (
    Workflow(name="parallel-workflow")
    .step(step1)
    .step(step2)
    .step(step3, step1_output=lambda: get_run_context().get_data("step1.output"), 
          step2_output=lambda: get_run_context().get_data("step2.output"))
    # No explicit linking needed - steps run in parallel by default
)

# Alternative: Using Tool class for more complex steps
step1_tool = Tool(
    name="step1",
    description="passes value from input to output",
    handler=step1
)

step2_tool = Tool(
    name="step2", 
    description="passes value from input to output",
    handler=step2
)

step3_tool = Tool(
    name="step3",
    description="sums values from step1 and step2",
    handler=step3
)

tool_parallel_workflow = (
    Workflow(name="tool-parallel")
    .step(step1_tool)
    .step(step2_tool)
    .step(step3_tool, step1_output=lambda: get_run_context().get_data("step1.output"), 
          step2_output=lambda: get_run_context().get_data("step2.output"))
)
`}/>

## Parallel execution using workflows

In this example, the workflow uses nested workflows to run `workflow1` and `workflow2` in parallel. Each workflow contains a single step that returns the input value. Their outputs are automatically namespaced and passed to `step3`.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool
from timbal.state import get_run_context

# Define individual workflow steps
def workflow_step(value: int) -> dict:
    """Simple step that passes value from input to output."""
    return {"value": value}

# Create individual workflows
workflow1 = (
    Workflow(name="workflow-1")
    .step(workflow_step)
)

workflow2 = (
    Workflow(name="workflow-2")
    .step(workflow_step)
)

# Define the combining step
def combine_workflows(workflow1_output: dict, workflow2_output: dict) -> dict:
    """Combines results from parallel workflows."""
    return {
        "value": workflow1_output["value"] + workflow2_output["value"]
    }

# Create the main parallel workflow
parallel_workflows = (
    Workflow(name="parallel-workflows")
    .step(workflow1)
    .step(workflow2)
    .step(combine_workflows, 
          workflow1_output=lambda: get_run_context().get_data("workflow1.output"),
          workflow2_output=lambda: get_run_context().get_data("workflow2.output"))
)
`}/>

## How parallel execution works in Timbal

1. **Automatic Parallelism**: Steps without explicit dependencies run in parallel by default
2. **Data Namespacing**: Each step's output is automatically namespaced by step name
3. **Context Access**: Use `get_run_context().get_data("step_name.output")` to access parallel step outputs
4. **Result Combination**: Final steps can combine results from parallel executions
5. **No Explicit Linking**: Parallel steps don't need `.link()` calls

## Example usage

<CodeBlock language="python" code={`import asyncio
from timbal.state import get_run_context

async def main():
    # Run the parallel workflow
    result = await parallel_workflow(value=10).collect()
    
    print("Parallel workflow execution completed!")
    print(f"Input value: 10")
    print(f"Step1 output: {get_run_context().get_data('step1.output')}")
    print(f"Step2 output: {get_run_context().get_data('step2.output')}")
    print(f"Combined result: {result.output}")
    
    # The workflow executed step1 and step2 in parallel
    # Then step3 combined their results

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Key differences from Mastra

1. **Parallel Declaration**:
   - **Mastra**: Explicit `.parallel([step1, step2])` method
   - **Timbal**: Automatic parallel execution for independent steps

2. **Data Access**:
   - **Mastra**: Complex input schema with namespaced objects
   - **Timbal**: Simple context access with `get_run_context().get_data()`

3. **Workflow Structure**:
   - **Mastra**: `.parallel([...]).then(step3).commit()`
   - **Timbal**: Natural step ordering with automatic dependency resolution

4. **Schema Definition**:
   - **Mastra**: Zod schemas for input/output validation
   - **Timbal**: Python type hints and function signatures

5. **Execution Model**:
   - **Mastra**: Complex workflow engine with explicit parallel control
   - **Timbal**: Automatic parallel execution based on step dependencies

The Timbal approach is much more intuitive - steps run in parallel automatically when they don't depend on each other, and you can easily access their results using the context system.
