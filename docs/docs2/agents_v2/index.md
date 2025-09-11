---
title: Overview
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Understanding Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master proven strategies for designing advanced, specialized AI agents using an architecture that work together seamlessly to tackle complex challenges.
</h2>

---

## What are Agents?

<span style={{color: 'var(--timbal-purple)'}}><strong>Agents</strong></span> are autonomous execution units that **orchestrate LLM interactions with tool calling**. 

Without tools, an agent functions as a basic LLM. The simplest agent requires just a name and model:

<CodeBlock language="python" code ={`from timbal import Agent

agent = Agent(
    name="my_agent",
    model="openai/gpt-5"
)  # That's it! You've created your first agent!`}/>

You can specify any model using the "provider/model" format. See all supported models in [Model Capabilities](/getting-started/model_capabilities).

Some models require specific parameters (like `max_tokens` for Claude). Use `model_params` to pass any additional model configuration: 

<CodeBlock language="python" code ={`agent = Agent(
    name="claude_agent",
    model="anthropic/claude-sonnet-4-latest",
    model_params={
        "max_tokens": 1024
    }
)`}/>

**Note:** Make sure to define all required environment variables—such as the API key model that you need—in your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key`}/>

Define tools as Python functions - the framework handles schema generation, parameter validation, and execution orchestration.

### Quick Example

<CodeBlock language="python" code ={`from timbal import Agent

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert temperature from Celsius to Fahrenheit"""
    fahrenheit = (celsius * 9/5) + 32
    return f"{celsius}°C = {fahrenheit}°F"

agent = Agent(
    name="demo_agent",
    model="openai/gpt-5-mini",
    tools=[celsius_to_fahrenheit],
    system_prompt="You are a helpful temperature conversion assistant."
)

await agent(prompt="What is 25 degrees Celsius in Fahrenheit?").collect()`}/>

The framework performs automatic introspection of function signatures and docstrings for tool schema generation.

**What happens behind the scenes:**

<div className="log-step-static">
  start_event ... path=demo_agent ...
</div>

<div className="log-step-static">
  start_event ... path=demo_agent.llm ...
</div>

<details className="log-step-collapsible">
<summary>
  output_event ... path=demo_agent.llm ...
</summary>
<CodeBlock language="bash" code={`output_event ...
    path=demo_agent.llm 
    input={
      'model': 'openai/gpt-5-mini', 
      'messages': [
        {'role': 'user', 'content': [
          {'type': 'text', 
          'text': 'How many fahrenheit are 25 celsius'
          }]
        }
      ],
      'system_prompt': 'You are a helpful temperature conversion assistant.',
      'tools': [
        {'name': 'celsius_to_fahrenheit',
        'description': '', 
        'input_schema': 
          {'properties': 
            {'celsius': {'title': 'Celsius', 'type': 'number'}},
          'required': ['celsius'],
          'title': 'CelsiusToFahrenheitParams',
          'type': 'object'
          }
        }
      ]
    }
    output={
      'role': 'assistant',
      'content': [
        {'type': 'tool_use',
        'id': '068b03c99cbe760c80003d233f0cc50f',
        'name': 'celsius_to_fahrenheit',
        'input': {'celsius': 25}
        }
      ]
    }
    usage={'gpt-5-mini-2025-08-07:input_text_tokens': 141, 'gpt-5-mini-2025-08-07:output_text_tokens': 91}
    ...`}/>
</details>

<div className="log-step-static">
  start_event ... path=demo_agent.celsius_to_fahrenheit ...
</div>

<details className="log-step-collapsible">
<summary>
  output_event ... path=demo_agent.celsius_to_fahrenheit ...
</summary>
<CodeBlock language="bash" code={`output_event ...
    path=demo_agent.celsius_to_fahrenheit
    input={'celsius': 25}
    output='25.0°C = 77.0°F'
    ...`}/>
</details>

<div className="log-step-static">
  start_event ... path=demo_agent.llm ...
</div>

<details className="log-step-collapsible">
<summary>
  output_event ..., path=demo_agent.llm ...
</summary>
<CodeBlock language="bash" code={`output_event ...
    path=demo_agent.llm, 
    input={
      'model': 'openai/gpt-5-mini',
      'messages': [
        {'role': 'user',
        'content': [
          {'type': 'text',
          'text': 'How many fahrenheit are 25 celsius?'}
        ]},
        {'role': 'assistant',
        'content': [
          {'type': 'tool_use',
          'id': '068b03c99cbe760c80003d233f0cc50f',
          'name': 'celsius_to_fahrenheit',
          'input': {'celsius': 25}
          }
        ]},
        {'role': 'tool',
        'content': [
          {'type': 'tool_result',
          'id': '068b03c99cbe760c80003d233f0cc50f',
          'content': [
            {'type': 'text', 'text': '25.0°C = 77.0°F'}
          ]}
        ]}
      ],
      'system_prompt': 'You are a helpful temperature conversion assistant.',
      'tools': [
        {'name': 'celsius_to_fahrenheit',
        'description': '',
        'input_schema': {'properties': {'celsius': {'title': 'Celsius', 'type': 'number'}},
        'required': ['celsius'],
        'title': 'CelsiusToFahrenheitParams',
        'type': 'object'}}
      ]
    } 
    output={
    'role': 'assistant',
    'content': [
      {'type': 'text',
      'text': '25°C is 77°F.'
      }]
    }
    usage={'gpt-5-mini-2025-08-07:input_text_tokens': 186, 'gpt-5-mini-2025-08-07:output_text_tokens': 10}
    ...`}/>
</details>

<details className="log-step-collapsible">
<summary>
  output_event ... path=demo_agent ...
</summary>
<CodeBlock language="bash" code={`output_event ...
    path=demo_agent
    input={'prompt': 'How many fahrenheit are 25 celsius?'}
    output={
      'role': 'assistant',
      'content': [
        {'type': 'text', 'text': '25°C is 77°F.'}
    ]}
    usage={'gpt-5-mini-2025-08-07:input_text_tokens': 327, 'gpt-5-mini-2025-08-07:output_text_tokens': 101}
    ...
`}/>
</details>

<div style={{marginTop: '2rem'}}>
</div>



## Architecture features

- <span style={{color: 'var(--timbal-purple)'}}><strong>Execution Engine</strong></span>:
    - **Asynchronous concurrent tool execution** via multiplexed event queues
    - Conversation state management with **automatic memory persistence across iterations**
    - **Multi-provider LLM routing** with unified interface abstraction
  
- <span style={{color: 'var(--timbal-purple)'}}><strong>Tool System</strong></span>:
  - Runtime **tool** discovery with automatic OpenAI/Anthropic **schema generation**
  - Support for nested Runnable composition and **hierarchical agent orchestration**
  - **Dynamic parameter validation** using Pydantic models

- <span style={{color: 'var(--timbal-purple)'}}><strong>Advanced Runtime</strong></span>:
  - Template-based **system prompt composition** with runtime callable injection
  - Configurable **iteration limits** with autonomous termination detection
  - Event-driven **streaming** architecture with real-time processing capabilities
  - Pre/post execution hooks for cross-cutting concerns and runtime interception

## Running an Agent

Agent provides a streamlined execution interface:

### Get a Complete Answer

<CodeBlock language="python" code ={`result = await agent(
    prompt="What is 25 degrees Celsius in Fahrenheit?"
).collect()`}/>

### Real-Time (Streaming) Output

<CodeBlock language="python" code ={`async for event in agent(
    prompt="What is 25 degrees Celsius in Fahrenheit?"
):
    print(event)`}/>

Below are the events that get printed when you run this example:

<div className="log-step-static">
  type='START' ... path='demo_agent' ...
</div>

<details className="log-step-collapsible">
<summary>
  type='OUTPUT' ... path='demo_agent' ...
</summary>
<CodeBlock language="bash" code={`type='OUTPUT'
path='demo_agent'
input={'prompt': 'What is 25 degrees Celsius in Fahrenheit?'}
output=Message(role=assistant, content=[TextContent(type='text', text='25°C is 77°F.')])
...
usage={'gpt-5-mini-2025-08-07:input_text_tokens': 325, 'gpt-5-mini-2025-08-07:output_text_tokens': 37}`}/>
</details>


## Next Steps

- Try creating your own Agent with different tools
- See examples in [Examples](/examples)
