---
title: Tool as a Step
sidebar: 'examples'
draft: True
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows can include tools as steps. This example shows how to define a tool as a step in your Timbal workflow.

## Creating a tool

Create a simple tool that takes a string input and returns the reversed version.

<CodeBlock language="python" code={`from timbal.core import Tool

def reverse_string(input_text: str) -> dict:
    """Reverse the input string."""
    reversed_text = input_text[::-1]  # Python slice notation for reversal
    
    return reversed_text

# Create the Tool instance
reverse_tool = Tool(
    name="reverse-tool",
    description="Reverse the input string",
    handler=reverse_string
)

# Alternative: Using lambda function for simple operations
reverse_lambda_tool = Tool(
    name="reverse-lambda-tool",
    description="Reverse the input string using lambda",
    handler=lambda input_text: {"output": input_text[::-1]}
)`}/>

## Tool as step

Use a tool as a step by passing it directly to the `.step()` method. Use lambda functions to map workflow input to the tool's expected parameters.

<CodeBlock language="python" code={`from timbal.core import Workflow

# Create the workflow with tool as a step
tool_as_step_workflow = (
    Workflow(name="tool-step-workflow")
    .step(reverse_tool)
)`}/>


## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    result = await tool_as_step_workflow(input_text="hello").collect()
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())`}/>
