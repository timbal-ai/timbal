---
title: Calling an Agent Inside a Step
sidebar: 'examples'
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
from timbal.types.message import Message

def step1(city: str) -> dict:
    """Passes city name to agent and returns facts."""
    # Create a message for the agent
    prompt = Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
    })
    
    # Call the agent directly
    response = await city_agent(prompt=prompt).collect()
    
    # Extract the facts from the response
    facts = response.output.content[0].text
    
    return {
        "facts": facts
    }

# Create the workflow
call_agent_workflow = (
    Workflow(name="agent-workflow")
    .step(step1)
)

# Alternative: Using Tool class for more complex agent interactions
def city_facts_tool(city: str) -> dict:
    """Tool that calls the city agent to get facts."""
    prompt = Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
    })
    
    response = await city_agent(prompt=prompt).collect()
    facts = response.output.content[0].text
    
    return {
        "facts": facts,
        "city": city
    }

city_facts_tool_instance = Tool(
    name="city_facts",
    description="Gets facts about a city using the city agent",
    handler=city_facts_tool
)

tool_agent_workflow = (
    Workflow(name="tool-agent-workflow")
    .step(city_facts_tool_instance)
)
`}/>

## How agent calling works in Timbal

1. **Direct Agent Access**: Agents are used directly without needing a central registry
2. **Message Creation**: Use `Message.validate()` to create properly formatted prompts
3. **Agent Invocation**: Call agents with `agent(prompt=prompt).collect()`
4. **Response Extraction**: Access the response text from the output structure
5. **Workflow Integration**: Seamlessly integrate agent calls into workflow steps

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Test with different cities
    
    # Test with "Paris"
    print("=== Testing with city: Paris ===")
    result1 = await call_agent_workflow(city="Paris").collect()
    print(f"City: Paris")
    print(f"Facts: {result1.output['facts']}")
    
    # Test with "Tokyo"
    print("\n=== Testing with city: Tokyo ===")
    result2 = await call_agent_workflow(city="Tokyo").collect()
    print(f"City: Tokyo")
    print(f"Facts: {result2.output['facts']}")
    
    # Test with tool version
    print("\n=== Testing tool version with city: New York ===")
    result3 = await tool_agent_workflow(city="New York").collect()
    print(f"City: {result3.output['city']}")
    print(f"Facts: {result3.output['facts']}")

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Advanced agent calling patterns

<CodeBlock language="python" code={`# Agent calling with multiple prompts
def multi_prompt_agent(city: str) -> dict:
    """Call agent with multiple prompts for comprehensive information."""
    prompts = [
        f"Create an interesting fact about {city}",
        f"What is {city} known for?",
        f"Share a historical fact about {city}"
    ]
    
    responses = []
    for prompt_text in prompts:
        prompt = Message.validate({
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        })
        
        response = await city_agent(prompt=prompt).collect()
        responses.append(response.output.content[0].text)
    
    return {
        "city": city,
        "facts": responses[0],
        "known_for": responses[1],
        "history": responses[2]
    }

# Agent calling with error handling
def safe_agent_call(city: str) -> dict:
    """Call agent with error handling."""
    try:
        prompt = Message.validate({
            "role": "user",
            "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
        })
        
        response = await city_agent(prompt=prompt).collect()
        facts = response.output.content[0].text
        
        return {
            "success": True,
            "city": city,
            "facts": facts
        }
    except Exception as e:
        return {
            "success": False,
            "city": city,
            "error": str(e)
        }

# Agent calling with context
def contextual_agent_call(city: str, context: str = "") -> dict:
    """Call agent with additional context."""
    prompt_text = f"Create an interesting fact about {city}"
    if context:
        prompt_text += f" in the context of {context}"
    
    prompt = Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}]
    })
    
    response = await city_agent(prompt=prompt).collect()
    facts = response.output.content[0].text
    
    return {
        "city": city,
        "context": context,
        "facts": facts
    }
`}/>

## Key differences from Mastra

1. **Agent Registration**:
   - **Mastra**: Required `mastra.getAgent("cityAgent")` calls
   - **Timbal**: Direct agent usage without central registry

2. **Agent Calling**:
   - **Mastra**: `agent.generate("prompt")` with simple string
   - **Timbal**: `agent(prompt=prompt).collect()` with structured Message

3. **Workflow Structure**:
   - **Mastra**: `.then(step1).commit()` with complex step definition
   - **Timbal**: Simple `.step(step1)` with direct function calls

4. **Data Flow**:
   - **Mastra**: Complex schema-based data passing between steps
   - **Timbal**: Direct function calls and return values

5. **Error Handling**:
   - **Mastra**: Limited error handling in the workflow framework
   - **Timbal**: Full Python error handling capabilities

The Timbal approach makes agent calling much more straightforward:
- **No Registration**: Use agents directly in your workflow steps
- **Simple Invocation**: Call agents with natural Python syntax
- **Flexible Integration**: Easily combine agents with other workflow logic
- **Error Handling**: Use Python's native exception handling
- **Context Support**: Pass additional context to agents as needed

This approach gives you the power to create dynamic, AI-powered workflows without the complexity of centralized agent management.
