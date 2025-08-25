---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Understanding Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master proven strategies for designing advanced, specialized AI agents that work together seamlessly to tackle complex challenges.
</h2>

---

## What is an Agent?

An <span style={{color: 'var(--timbal-purple)'}}><strong>Agent</strong></span> is like having your own AI-powered teammate—one that can understand your goals, reason about the best way to achieve them, and take actions on your behalf. Powered by advanced Large Language Models (LLMs), Agents go far beyond simple chatbots or assistants.

Think of an Agent as:

- A digital coworker that can read, write, and analyze information.
- A problem-solver that can break down complex tasks into steps and execute them.
- A connector that can use tools, access APIs, search the web, or interact with other software to get things done.

<CodeBlock language="python" code ={`Agent()  # That's it! You've created your first agent!`}/>

**Note:** Make sure to define all required environment variables—such as your OpenAI API key, your Gemini API key, the API key model that you need—in your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## Quick Example

Here's an example of an agent that uses a tool to get weather information:

<CodeBlock language="python" code ={`from timbal import Agent, Tool

# Define a weather tool
def get_weather(location: str) -> str:
    # This is a simplified example - in practice, you'd use a real weather API
    return "The weather is sunny"

# Create an agent with the weather tool
agent = Agent(
    model="gemini-2.5-pro-preview-03-25",
    tools=[
        Tool(
            runnable=get_weather,
            description="Get the weather for a specific location",
        )
    ]
)

# Use the agent to get weather information
response = await agent.complete(
    prompt="What's the weather like in New York?"
)`}/>

 **Agent Execution Logs**:

<div className="log-step-static">
  StartEvent(..., path='agent, ...)
</div>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='agent.llm-0', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='agent.llm-0', 
    input={
      'messages': [
        Message(
          role=user,
          content=[TextContent(
            type='text', 
            text="What's the weather like in New York?"
          )]
        )
      ], 
      'tools': [{
        'type': 'function', 
        'function': {
          'name': 'get_weather',
          'description': 'Get the weather for a specific location',
          'parameters': {
            'properties': {'location': {'title': 'Location', 'type': 'string'}}, 
            'required': ['location'],
            ...
          }
        }
      }], 
      'model': 'gpt-4',
      ...
    },
    output=Message(
      role=assistant,
      content=[ToolUseContent(
        type='tool_use', 
        id='...', 
        name='get_weather', 
        input={'location': 'New York'}
      )]
    ), ...)
`}/>
</details>

<div className="log-step-static">
  StartEvent(..., path='agent.get_weather-call_...', ...)
</div>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='agent.get_weather-...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='agent.get_weather-...',
    input={'location': 'New York'},
    output=Message(
      role=user,
      content=[TextContent(type='text', text='The weather is sunny')]
    ), ...)`}/>
</details>

<div className="log-step-static">
  StartEvent(..., path='agent.llm-1', ...)
</div>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='agent.llm-1', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='agent.llm-1', 
    input={
      'messages': [
        Message(
          role=user, 
          content=[TextContent(
            type='text',
            text="What's the weather like in New York?"
          )]
        ), 
        Message(
          role=assistant, 
          content=[ToolUseContent(
            type='tool_use',
            id='...',
            name='get_weather',
            input={'location': 'New York'}
          )]
        ),
        Message(
          role=user, 
          content=[ToolResultContent(
            type='tool_result', 
            id='call_...', 
            content=[TextContent(type='text', text='The weather is sunny')]
          )]
        )
      ],
      'tools': [{
        'type': 'function', 
        'function': {
          'name': 'get_weather',
          'description': 'Get the weather for a specific location',
          'parameters': {
            'properties': {'location': {'title': 'Location', 'type': 'string'}},
            'required': ['location'],
            ...
          }
        }
      }], 
      'model': 'gpt-4',
      ...
    },
    output=Message(
      role=assistant, 
      content=[TextContent(
        type='text',
        text='The weather in New York is sunny.'
      )]
    ),...)`}/>
</details>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='agent', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='agent',
    input={
      'prompt': {
        'role': 'user', 
        'content': [{
          'type': 'text',
          'text': "What's the weather like in New York?"
        }]
      }
    },
    output=Message(
      role=assistant, 
      content=[TextContent(
        type='text',
        text='The weather in New York is sunny.'
      )]
    ), ...)
`}/>
</details>

<div style={{marginTop: '2rem'}}>
This example shows how to create an agent with a custom tool. The agent can now use the weather tool to fetch real-time weather data when needed. You can add multiple tools to make your agent even more powerful!
</div>

## Key Capabilities of an Agent

Let's break down how an Agent thinks and works:

<div className="timeline">
<div className="timeline-item">
<div className="timeline-content">

<h4>Autonomous Reasoning</h4>
Agents can decide what to do next based on your instructions and the current situation.

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Tool Use</h4>
They can use built-in or custom tools (like searching the internet, fetching data, or running code) to accomplish tasks.

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Memory</h4>
Agents can remember previous interactions, decisions, or data, allowing them to work on long-term or multi-step projects.

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Adaptability</h4>
They can handle a wide range of tasks, from answering questions to automating workflows.

</div>
</div>
</div>

<style>{`
.timeline {
  display: flex;
  align-items: center;
  margin: 1rem 0;
  overflow-x: auto;
  padding: 0.5rem;
}

.timeline-item {
  width: 180px;
  text-align: left;
  padding: 0.5rem;
  background: var(--ifm-background-color);
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.timeline-connector {
  color: var(--ifm-color-primary);
  font-size: 1.2rem;
  padding: 0 0.5rem;
}

.timeline-content {
  padding: 0.5rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.timeline-content h3 {
  margin-bottom: 0.25rem;
  font-size: 1rem;
}

.timeline-content ul {
  list-style: disc;
  padding-left: 1.2em;
  margin: 0;
}

.timeline-content li {
  margin: 0.15rem 0;
  font-size: 0.85rem;
}

.timeline-content h4 {
  color: var(--ifm-color-primary);
  font-weight: bold;
  margin-bottom: 0.5em;
  margin-top: 0;
}
`}</style>


## Running an Agent

To execute an Agent, there are 2 possibilities depending on the synchronisation.

### Get a Complete Answer

For when the agent returns a complete response after processing. We will use the `complete()` function:

<CodeBlock language="python" code ={`response = await agent.complete(prompt="What time is it?")`}/>

### Real-Time (Streaming) Output

Otherwise, when we want to know specific information on each event we can find the response asynchrounsly by running `run()`:

<CodeBlock language="python" code ={`response = async for event in agent.run(prompt="What time is it?"):
    print(event) `}/>

Events tell you what's happening in your agent. Here's what you can do with them:

<CodeBlock language="python" code ={`async for event in agent.run(prompt="What time is it?"):
    if event.type == "START":
        print(f"Starting Agent: {event.step_id}")`}/>

<CodeBlock language="python" code ={`async for event in agent.run(prompt="What time is it?"):
    if event.type == "OUTPUT":
        print(f"Agent finished in {event.elapsed_time}ms")
        print(f"Outputs: {event.outputs}")`}/>


## Next Steps

- Try creating your own Agent with different tools
- Experiment with different configurations
- See an example agent in [Examples](/examples)

Remember: The more you practice, the better you'll become at creating powerful Agents!
