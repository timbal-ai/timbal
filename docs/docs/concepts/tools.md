---
title: Tools
sidebar: 'docsSidebar'
---

# Understanding Tools

Welcome to the world of Tools! These are like the Swiss Army knives for your Agents - they give them superpowers to interact with the world!

## What are Tools?

Tools are special functions that your Agents can use to:
- Search the interne
- Get weather information
- Check the time
- And much more!

Think of them as the hands and eyes of your Agent - they help it interact with the world beyond just thinking!

## How Tools Work

Let's break down how Tools make your Agents more powerful:

1. **Agent Requests a Tool**
   - Your Agent decides it needs information
   - It chooses the right tool for the job

2. **Tool Gets to Work**
   - The tool runs its specific function
   - It gathers the needed information

3. **Results Come Back**
   - The tool returns its findings
   - Your Agent uses this information to help you

## Creating Your Own Tools 

Ready to create your first tool? Let's do it step by step!

```python
# 1️⃣ Define your tool function
def get_weather(location: str) -> str:
    # Your weather-fetching code here
    return f"The weather in {location} is sunny!"

# 2️⃣ Create a Tool object
weather_tool = Tool(
    runnable=get_weather,
    description="Get the weather for a location",
    exclude_params=["query"]  # Parameters to exclude from the Agent's view
)

# 3️⃣ Add it to your Agent
agent = Agent(
    tools=[weather_tool]
)
```

## Different Types of Tools 

You can create tools in several ways:

1. **Simple Function Tools**
```python
Tool(
    runnable=simple_function,
    description="A simple tool"
)
```

2. **Dictionary Tools**
```python
{
    "runnable": your_function,
    "description": "A tool defined as a dictionary",
    "params_mode": "required"
}
```

3. **Pre-built Tools** 🏗️
```python
search_internet  # A ready-to-use tool for web searches
```

## Congratulations! 

You've just learned how to create and use Tools! Here's what you can do next:

- Create your own custom tools
- Combine multiple tools in an Agent
- Build something amazing!

Remember: The more tools you create, the more powerful your Agents become!

## Want to Learn More?

Check out these related concepts:
- Agents: Learn how to use tools with Agents
- Flows: Discover how to integrate tools into workflows
- Advanced Tools: Take your tools to the next level