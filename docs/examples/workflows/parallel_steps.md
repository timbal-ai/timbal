---
title: Parallel Execution
sidebar: 'examples'
draft: True
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows often need to run multiple operations at the same time. These examples demonstrate how to execute steps or workflows concurrently and merge their results using Timbal's automatic parallel execution capabilities.

## Parallel execution using steps

In this example, the workflow runs `step1` and `step2` in parallel. Each step receives the same input and runs independently. Their outputs are automatically namespaced by step name and passed together to `step3`, which combines the results and returns the final value.

<CodeBlock language="python" code={`from timbal.core import Workflow
from timbal.state import get_run_context

# Define step 1: passes value from input to output
def step1(value: int) -> int:
    """First step that processes the input value."""
    return value

# Define step 2: passes value from input to output  
def step2(value: int) -> int:
    """Second step that processes the input value."""
    return value

# Define step 3: sums values from step1 and step2
def step3(step1_output: int, step2_output: int) -> int:
    """Combines results from parallel steps."""
    return step1_output + step2_output

# Create the parallel workflow
parallel_workflow = (
    Workflow(name="parallel-workflow")
    .step(step1)
    .step(step2)
    .step(step3, step1_output=lambda: get_run_context().get_data("step1.output"), 
          step2_output=lambda: get_run_context().get_data("step2.output"))
    # No explicit linking needed - steps run in parallel by default
)`}/>

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Run the parallel workflow
    result = await parallel_workflow(value=10).collect()
    print(result.output) # 20
    
    # The workflow executed step1 and step2 in parallel
    # Then step3 combined their results

if __name__ == "__main__":
    asyncio.run(main())`}/>