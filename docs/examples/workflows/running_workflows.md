---
title: Running Workflows
sidebar: 'examples'
draft: True
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows can be run from different environments. These examples demonstrate how to execute a workflow using a command line script or by calling the workflow directly from a client-side component.

## From the command line

In this example, a run script has been added to the `src` directory. The workflow processes input data and executes sequential steps.

<CodeBlock language="python" title="test_run_workflow.py" code={`import asyncio
from timbal.core import Workflow

# Define workflow steps
def step_1(value: int) -> dict:
    """First step that processes the input value."""
    return {"value": value, "step": 1}

def step_2(value: int) -> dict:
    """Second step that processes the input value."""
    return {"value": value, "step": 2}

# Create the workflow
sequential_workflow = (
    Workflow(name="sequentialSteps")
    .step(step_1)
    .step(step_2)
    .link("step_1", "step_2")
)

async def main():
    # Run the workflow with input data
    result = await sequential_workflow(value=10).collect()
    
    print("Workflow execution completed!")
    print(f"Final result: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())`}/>

### Run the script

Run the workflow using the following command:

<CodeBlock language="bash" code={`python test_run_workflow.py`}/>

Or if you prefer using `uv`:

<CodeBlock language="bash" code={`uv run test_run_workflow.py`}/>

### Command line output

The output from this workflow run will look similar to the below:


## Alternative: Simple workflow execution

For simpler use cases, you can run workflows directly without complex setup:

<CodeBlock language="python" code={`# Simple workflow execution
from timbal.core import Workflow

# Define a simple workflow
simple_workflow = (
    Workflow(name="simple")
    .step(lambda x: x * 2)
    .step(lambda x: x + 1)
)

async def run_simple():
    # Execute with input value 5
    result = await simple_workflow(x=5).collect()
    print(f"Input: 5, Output: {result.output}")

# Run the simple workflow
asyncio.run(run_simple())`}/>