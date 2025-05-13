---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Understanding Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.2rem', fontWeight: 'normal'}}>
Master proven strategies for designing advanced, specialized AI agents that work together seamlessly to tackle complex challenges.
</h2>

## What is an Agent?

An Agent is like having a super-smart assistant powered by an LLM (Large Language Model). Think of it as your AI teammate that can think, reason, and take actions!

<CodeBlock language="python" code ={`Agent()  # That's it! You've created your first agent!`}/>

**Note:** Make sure to define all required environment variables—such as your OpenAI API key—in your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## How Agents Work

Let's break down how an Agent thinks and works:

<div className="timeline">
<div className="timeline-item">
<div className="timeline-content">

<h4>Receives Your Request</h4>
- You give the **Agent** a **task** or **question**.
- It carefully considers **what needs to be done**.

</div>
</div>

<div className="timeline-connector">→</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Makes Smart Decisions</h4>
* Decides if it can **answer directly**
* Or if it needs to **use special tools** to help

</div>
</div>

<div className="timeline-connector">→</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Uses Tools When Needed</h4>
- If tools are needed, it **knows** exactly which ones to use
- Gets the information it needs

</div>
</div>

<div className="timeline-connector">→</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Gives You the Perfect Answer</h4>
- Combines all the information
- Gives you a complete, thoughtful **response**

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

### Synchronous Output Mode

For when the agent returns a complete response after processing. We will use the `complete()` function:

<CodeBlock language="python" code ={`response = await agent.complete(prompt="What time is it?")`}/>

### Streaming Response

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


## Using Tools

Great! Now that you understand how Agents work, let's make yours even more powerful by adding some tools:

<CodeBlock language="python" code ={`Agent(
    tools = [
      search_internet,
      Tool(
        runnable = get_weather,
        description = "Get the weather of a location",
        exclude_params=["query"]
      ),
      {
        "runnable": get_time,
        "description": "Get the time of a location",
        "params_mode": "required",
        "invalid_key": "invalid_value",
        "include_params": "model"
      }
    ]
)`}/>

Agents can be equipped with tools—custom functions that expand their abilities beyond text generation. With tools, agents can perform tasks like calculations, interact with external systems, and process data. 

To learn more about creating and configurating tools, check out the [Tools documentation](/agents/tools).


## Using Memory in Agents

If we want our Agent to have memory we have to ensure this 2 steps:

First, we have to add an `state_saver`.

<CodeBlock language="python" code ={`Agent(
        model="gpt-4o-mini",
        state_saver=InMemorySaver(),
    )`}/>

Second, we have to add the context. To do it we have to first initialize a `RunContext`and when a response is generated updated with the parent context to ensure the tracebelity of it.

<CodeBlock language="python" code ={`from timbal.state import RunContext
run_context = RunContext()
flow_output_event = await agent.complete(
        context=run_context,
        prompt="What's my name?"
)
run_context = RunContext(parent_id=flow_output_event.run_id)`}/>

In order to know more about state savers and the ones done take a look at [State documentation](/state)

## Next Steps

- Try creating your own Agent with different tools
- Experiment with different configurations
- See an example agent in [Examples](/examples)

Remember: The more you practice, the better you'll become at creating powerful Agents!