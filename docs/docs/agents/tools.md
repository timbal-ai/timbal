---
title: Tools
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Understanding Tools

---

Tools are the way to give your Agents superpowers: they allow your agent to interact with the outside world, call APIs, perform calculations, and much more.

## What is a Tool?

A **Tool** is a function (or callable) that your agent can invoke. You can use:
- Your own Python functions
- Pre-built tools provided by Timbal
- Tools defined as objects or dictionaries

## How to Create a Tool

### 1. Define a Python Function

You can start with a simple function:

<CodeBlock language="python" code ={`def get_weather(location: str) -> str:
    return "The weather is sunny!"`}/>

### 2. Wrap it as a Tool

You can wrap your function as a Tool object, or just pass the function directly:

<CodeBlock language="python" code ={`from timbal.core.agent.types.tool import Tool

weather_tool = Tool(
    runnable=get_weather,
    description="Get the weather for a location",
)`}/>

Or, you can use a dictionary:

<CodeBlock language="python" code ={`weather_tool = {
    "runnable": get_weather,
    "description": "Get the weather for a location",
    # Only required params are exposed to the LLM
    "params_mode": "required",
}`}/>

### 3. Add the Tool to Your Agent

<CodeBlock language="python" code ={`from timbal import Agent

agent = Agent(
    tools=[weather_tool]
)`}/>

## Using Built-in Tools

Timbal comes with many built-in tools, such as `search` using Perplexity or `send_message` using Slack.  
You can add them directly:

<CodeBlock language="python" code ={`from timbal.steps.perplexity import search

agent = Agent(
    tools=[search]
)`}/>

Find more in the [Integrations](/integrations) section.

## Customizing Tool Parameters

You can control which parameters are visible to the LLM and how they are described.

### Field Descriptions and Choices

Use `Field` to add descriptions and choices to your function parameters:

<CodeBlock language="python" code ={`from timbal.types import Field

def get_weather(
    location: str = Field(description="The location to get the weather for"),
    unit: str = Field(choices=["celsius", "fahrenheit"], description="Temperature unit")
) -> str:
    ...`}/>

### Controlling Parameter Visibility

- **params_mode**: `"all"` (default) exposes all params, `"required"` exposes only required ones.
- **include_params**: List of param names to always include.
- **exclude_params**: List of param names to exclude.

#### Only show required params

<CodeBlock language="python" code ={`Tool(
    runnable=get_weather,
    description="Get the weather for a location",
    params_mode="required"
)`}/>

#### Include extra params

<CodeBlock language="python" code ={`Tool(
    runnable=get_weather,
    description="Get the weather for a location",
    params_mode="required",
    # Even if not required, 'unit' will be shown
    include_params=["unit"]  
)`}/>

#### Exclude params

<CodeBlock language="python" code ={`Tool(
    runnable=get_weather,
    description="Get the weather for a location",
    # 'unit' will not be shown to the LLM
    exclude_params=["unit"]  
)`}/>

#### Dictionary style

<CodeBlock language="python" code ={`{
    "runnable": get_weather,
    "description": "Get the weather for a location",
    "params_mode": "required",
    # You can combine include/exclude
    "include_params": ["unit"],
    "exclude_params": ["location"],
}`}/>

## Adding Descriptions

Always add a `description` to your tool! This helps the LLM understand when and how to use it.

<CodeBlock language="python" code ={`Tool(
    runnable=get_weather,
    description="Get the weather for a location"
)`}/>

## Using Tools in Agents

You can combine multiple tools, both custom and built-in:

<CodeBlock language="python" code ={`from timbal.steps.perplexity import search

agent = Agent(
    tools=[
        search,
        Tool(
            runnable=get_weather,
            description="Get the weather of a location",
            exclude_params=["query"]
        ),
        {
            "runnable": get_time,
            "description": "Get the time of a location",
            "params_mode": "required",
            "include_params": ["model"]
        }
    ]
)`}/>


## Summary

- Tools let your agent interact with the world.
- You can use your own functions, built-in tools, or dictionaries.
- Customize which parameters are visible and how they are described.
- Add clear descriptions for best results.

For more, see the [Integrations](/integrations) and [Advanced Tools](/agents/tools) docs!

