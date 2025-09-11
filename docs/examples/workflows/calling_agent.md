---
title: Calling an Agent Inside a Step
sidebar: 'examples'
draft: True
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows can call agents to generate dynamic responses from within a step. This example shows how to define an agent and invoke it directly from within a workflow step. The workflow takes a city name as input and returns a fact about the corresponding city.

## Creating an agent

Create a simple agent that returns facts about a city.

<CodeBlock language="python" code={`from timbal.core import Agent

city_agent = Agent(
    name="city-agent",
    description="Create facts for a city",
    system_prompt="Return an interesting fact based on the city provided",
    model="openai/gpt-4o"
)`}/>

## Calling an agent

Call the agent directly from within a workflow step using the agent instance.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool

def city_facts_tool(city: str) -> dict:
    """Tool that calls the city agent to get facts."""
    response = await city_agent(prompt=f"Create an interesting fact about {city}").collect()
    
    return facts = response.output.content[0].text


city_facts_tool_instance = Tool(
    name="city_facts",
    description="Gets facts about a city using the city agent",
    handler=city_facts_tool
)

tool_agent_workflow = (
    Workflow(name="tool-agent-workflow")
    .step(city_facts_tool_instance)
)`}/>

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    result = await call_agent_workflow(city="Paris").collect()
    print(result1.output)

if __name__ == "__main__":
    asyncio.run(main())`}/>