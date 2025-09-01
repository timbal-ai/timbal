---
title: Agent as a Step
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows can include agents as steps. This example shows how to define an agent as a step in your Timbal workflow.

## Creating an agent

Create a simple agent that returns facts about a city.

<CodeBlock language="python" code={`from timbal.core import Agent

city_agent = Agent(
    name="city-agent",
    description="Create facts for a city",
    system_prompt="Return an interesting fact based on the city provided",
    model="openai/gpt-4o"
)`}/>

## Agent input/output structure

Timbal agents expect a `Message` object as input and return a `Message` object as output. The input message contains the prompt, and the output message contains the generated text.

```python
# Input: Message with prompt
input_message = Message.validate({
    "role": "user",
    "content": [{"type": "text", "text": "Create an interesting fact about Paris"}]
})

# Output: Message with generated text
output_message = Message.validate({
    "role": "assistant",
    "content": [{"type": "text", "text": "Paris is known as the City of Light..."}]
})
```

## Agent as step

To use an agent as a step, pass it directly to the `.step()` method. Use lambda functions to transform the workflow input into the shape the agent expects.

<CodeBlock language="python" code={`from timbal.core import Workflow
from timbal.types.message import Message

# Create the workflow with agent as a step
agent_as_step_workflow = (
    Workflow(name="agent-step-workflow")
    .step(city_agent, prompt=lambda city: Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
    }))
)

# Alternative: Using a function to handle the transformation
def create_city_prompt(city: str) -> Message:
    """Create a prompt message for the city agent."""
    return Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
    })

function_agent_workflow = (
    Workflow(name="function-agent-workflow")
    .step(city_agent, prompt=lambda city: create_city_prompt(city))
)

# Alternative: Using Tool class for more complex transformations
from timbal.core import Tool

def city_prompt_tool(city: str) -> Message:
    """Tool that creates a prompt for the city agent."""
    return Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
    })

city_prompt_tool_instance = Tool(
    name="city_prompt",
    description="Creates a prompt for the city agent",
    handler=city_prompt_tool
)

tool_agent_workflow = (
    Workflow(name="tool-agent-workflow")
    .step(city_agent, prompt=lambda city: city_prompt_tool_instance(city=city))
)
`}/>

## How agent as step works in Timbal

1. **Direct Integration**: Agents can be used directly as workflow steps
2. **Input Transformation**: Use lambda functions to transform workflow input into agent-compatible format
3. **Message Handling**: Agents automatically handle Message objects for input and output
4. **Seamless Flow**: Agent output flows directly to the next step or workflow output
5. **Flexible Mapping**: Multiple ways to transform input data for the agent

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Test with different cities
    
    # Test with "Paris"
    print("=== Testing with city: Paris ===")
    result1 = await agent_as_step_workflow(city="Paris").collect()
    print(f"City: Paris")
    print(f"Facts: {result1.output.content[0].text}")
    
    # Test with "Tokyo"
    print("\n=== Testing with city: Tokyo ===")
    result2 = await agent_as_step_workflow(city="Tokyo").collect()
    print(f"City: Tokyo")
    print(f"Facts: {result2.output.content[0].text}")
    
    # Test with function version
    print("\n=== Testing function version with city: New York ===")
    result3 = await function_agent_workflow(city="New York").collect()
    print(f"City: New York")
    print(f"Facts: {result3.output.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Advanced agent step patterns

<CodeBlock language="python" code={`# Agent step with multiple inputs
def create_comprehensive_prompt(city: str, context: str = "") -> Message:
    """Create a comprehensive prompt for the city agent."""
    prompt_text = f"Create an interesting fact about {city}"
    if context:
        prompt_text += f" in the context of {context}"
    
    return Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}]
    })

comprehensive_agent_workflow = (
    Workflow(name="comprehensive-agent-workflow")
    .step(city_agent, prompt=lambda city, context: create_comprehensive_prompt(city, context))
)

# Agent step with conditional prompts
def conditional_city_prompt(city: str, language: str = "en") -> Message:
    """Create a language-specific prompt for the city agent."""
    if language == "es":
        prompt_text = f"Crea un hecho interesante sobre {city}"
    elif language == "fr":
        prompt_text = f"Créez un fait intéressant sur {city}"
    else:
        prompt_text = f"Create an interesting fact about {city}"
    
    return Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}]
    })

multilingual_agent_workflow = (
    Workflow(name="multilingual-agent-workflow")
    .step(city_agent, prompt=lambda city, language: conditional_city_prompt(city, language))
)

# Agent step with error handling
def safe_city_prompt(city: str) -> Message:
    """Create a safe prompt for the city agent with validation."""
    if not city or not isinstance(city, str):
        raise ValueError("City must be a non-empty string")
    
    return Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": f"Create an interesting fact about {city}"}]
    })

safe_agent_workflow = (
    Workflow(name="safe-agent-workflow")
    .step(city_agent, prompt=lambda city: safe_city_prompt(city))
)
`}/>

## Key differences from Mastra

1. **Step Creation**:
   - **Mastra**: `createStep(cityAgent)` with complex workflow orchestration
   - **Timbal**: Direct `.step(city_agent, ...)` with simple parameter mapping

2. **Input Mapping**:
   - **Mastra**: `.map(async ({ inputData }) => {...})` with complex async functions
   - **Timbal**: Lambda functions for simple transformations

3. **Workflow Structure**:
   - **Mastra**: `.map(...).then(step1).commit()` with complex chaining
   - **Timbal**: Simple `.step(agent, prompt=lambda: ...)` with direct parameter mapping

4. **Data Flow**:
   - **Mastra**: Complex schema-based data transformation between steps
   - **Timbal**: Direct parameter passing with lambda functions

5. **Output Handling**:
   - **Mastra**: Complex output schema validation and mapping
   - **Timbal**: Direct agent output flow without additional mapping

The Timbal approach makes using agents as steps much more straightforward:
- **Direct Integration**: Agents become workflow steps with simple parameter mapping
- **Flexible Input**: Use lambda functions to transform any input format
- **Natural Flow**: Agent output flows directly to the next step
- **Multiple Patterns**: Choose the approach that best fits your use case
- **Python Native**: Use familiar Python syntax and patterns

This approach gives you the power to seamlessly integrate AI agents into your workflows while maintaining the simplicity and flexibility of the Timbal framework.

## Tool as a Step

Workflows can include tools as steps. This example shows how to define a tool as a step in your Timbal workflow.

## Creating a tool

Create a simple tool that takes a string input and returns the reversed version.

<CodeBlock language="python" code={`from timbal.core import Tool

def reverse_string(input_text: str) -> dict:
    """Reverse the input string."""
    reversed_text = input_text[::-1]  # Python slice notation for reversal
    
    return {
        "output": reversed_text
    }

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
)
`}/>

## Tool as step

Use a tool as a step by passing it directly to the `.step()` method. Use lambda functions to map workflow input to the tool's expected parameters.

<CodeBlock language="python" code={`from timbal.core import Workflow

# Create the workflow with tool as a step
tool_as_step_workflow = (
    Workflow(name="tool-step-workflow")
    .step(reverse_tool, input_text=lambda word: word)
)

# Alternative: Using the lambda tool directly
lambda_tool_workflow = (
    Workflow(name="lambda-tool-workflow")
    .step(reverse_lambda_tool, input_text=lambda word: word)
)

# Alternative: Using a function to handle the mapping
def map_word_to_input(word: str) -> str:
    """Map the workflow word input to the tool's input_text parameter."""
    return word

function_tool_workflow = (
    Workflow(name="function-tool-workflow")
    .step(reverse_tool, input_text=lambda word: map_word_to_input(word))
)

# Alternative: Using Tool class for more complex transformations
def reverse_with_context(input_text: str, context: str = "") -> dict:
    """Reverse the input string with optional context."""
    reversed_text = input_text[::-1]
    
    result = {
        "output": reversed_text
    }
    
    if context:
        result["context"] = context
    
    return result

context_reverse_tool = Tool(
    name="context-reverse-tool",
    description="Reverse the input string with context",
    handler=reverse_with_context
)

context_tool_workflow = (
    Workflow(name="context-tool-workflow")
    .step(context_reverse_tool, 
          input_text=lambda word: word,
          context=lambda word: f"Reversed from '{word}'")
)
`}/>

## How tool as step works in Timbal

1. **Direct Integration**: Tools can be used directly as workflow steps
2. **Parameter Mapping**: Use lambda functions to map workflow input to tool parameters
3. **Flexible Input**: Tools can accept multiple parameters with different mapping strategies
4. **Seamless Flow**: Tool output flows directly to the next step or workflow output
5. **Multiple Patterns**: Choose the approach that best fits your use case

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Test with different words
    
    # Test with "hello"
    print("=== Testing with word: hello ===")
    result1 = await tool_as_step_workflow(word="hello").collect()
    print(f"Input: hello")
    print(f"Reversed: {result1.output['output']}")
    
    # Test with "python"
    print("\n=== Testing with word: python ===")
    result2 = await tool_as_step_workflow(word="python").collect()
    print(f"Input: python")
    print(f"Reversed: {result2.output['output']}")
    
    # Test with lambda tool
    print("\n=== Testing lambda tool with word: workflow ===")
    result3 = await lambda_tool_workflow(word="workflow").collect()
    print(f"Input: workflow")
    print(f"Reversed: {result3.output['output']}")
    
    # Test with context tool
    print("\n=== Testing context tool with word: timbal ===")
    result4 = await context_tool_workflow(word="timbal").collect()
    print(f"Input: timbal")
    print(f"Reversed: {result4.output['output']}")
    print(f"Context: {result4.output.get('context', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Advanced tool step patterns

<CodeBlock language="python" code={`# Tool step with multiple inputs
def reverse_with_options(input_text: str, preserve_case: bool = True, add_prefix: str = "") -> dict:
    """Reverse the input string with various options."""
    reversed_text = input_text[::-1]
    
    if not preserve_case:
        reversed_text = reversed_text.swapcase()
    
    if add_prefix:
        reversed_text = f"{add_prefix}{reversed_text}"
    
    return {
        "output": reversed_text,
        "original": input_text,
        "options_used": {
            "preserve_case": preserve_case,
            "add_prefix": add_prefix
        }
    }

advanced_reverse_tool = Tool(
    name="advanced-reverse-tool",
    description="Reverse string with advanced options",
    handler=reverse_with_options
)

advanced_tool_workflow = (
    Workflow(name="advanced-tool-workflow")
    .step(advanced_reverse_tool,
          input_text=lambda word: word,
          preserve_case=lambda word: len(word) > 5,  # Preserve case for long words
          add_prefix=lambda word: "REV:" if word.startswith('a') else "")
)

# Tool step with conditional logic
def smart_reverse(input_text: str, language: str = "en") -> dict:
    """Smart string reversal with language-specific handling."""
    if language == "es" and input_text.endswith("a"):
        # Spanish feminine words get special treatment
        reversed_text = f"¡{input_text[::-1]}!"
    elif language == "fr" and input_text.endswith("e"):
        # French words ending in 'e' get special treatment
        reversed_text = f"Voilà: {input_text[::-1]}"
    else:
        reversed_text = input_text[::-1]
    
    return {
        "output": reversed_text,
        "language": language,
        "special_handling": reversed_text != input_text[::-1]
    }

smart_reverse_tool = Tool(
    name="smart-reverse-tool",
    description="Smart string reversal with language support",
    handler=smart_reverse
)

smart_tool_workflow = (
    Workflow(name="smart-tool-workflow")
    .step(smart_reverse_tool,
          input_text=lambda word: word,
          language=lambda word: "es" if word.endswith("a") else "en")
)
`}/>

## Key differences from Mastra

1. **Step Creation**:
   - **Mastra**: `createStep(reverseTool)` with complex workflow orchestration
   - **Timbal**: Direct `.step(reverse_tool, ...)` with simple parameter mapping

2. **Input Mapping**:
   - **Mastra**: `.map(async ({ inputData }) => {...})` with complex async functions
   - **Timbal**: Lambda functions for simple parameter mapping

3. **Workflow Structure**:
   - **Mastra**: `.map(...).then(step1).commit()` with complex chaining
   - **Timbal**: Simple `.step(tool, param=lambda: ...)` with direct parameter mapping

4. **Data Flow**:
   - **Mastra**: Complex schema-based data transformation between steps
   - **Timbal**: Direct parameter passing with lambda functions

5. **Output Handling**:
   - **Mastra**: Complex output schema validation and mapping
   - **Timbal**: Direct tool output flow without additional mapping

The Timbal approach makes using tools as steps much more straightforward:
- **Direct Integration**: Tools become workflow steps with simple parameter mapping
- **Flexible Parameters**: Map any workflow input to tool parameters using lambda functions
- **Natural Flow**: Tool output flows directly to the next step
- **Multiple Patterns**: Choose the approach that best fits your use case
- **Python Native**: Use familiar Python syntax and patterns

This approach gives you the power to seamlessly integrate tools into your workflows while maintaining the simplicity and flexibility of the Timbal framework.
