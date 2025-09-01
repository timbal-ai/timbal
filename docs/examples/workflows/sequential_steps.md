---
title: Sequential Execution
sidebar: 'examples'
draft: True
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Many workflows involve executing steps one after another in a defined order. This example demonstrates how to use `.link()` to build a simple sequential workflow where the output of one step becomes the input of the next.

## Sequential execution using steps

In this example, the workflow runs `step1` and `step2` in sequence, passing the input through each step and returning the final result from `step2`.

<CodeBlock language="python" code={`from timbal.core import Workflow
from timbal.state import get_run_context

# Define step 1: passes value from input to output
def step1(x1: int) -> dict:
    """First step that processes the input value."""
    return x1

# Define step 2: passes value from input to output  
def step2(x2: int) -> dict:
    """Second step that processes the input value."""
    return x2

# Create the sequential workflow
sequential_workflow = (
    Workflow(name="sequential-workflow")
    .step(step1)
    .step(step2, x2=lambda: get_run_context().get_data("step1.output"))
)`}/>

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Run the sequential workflow
    result = await sequential_workflow(value=42).collect()
    print(f"Final output: {result.output}")
    
    # The workflow executed step1 -> step2 in sequence
    # Each step received the value and passed it through

if __name__ == "__main__":
    asyncio.run(main())`}/>