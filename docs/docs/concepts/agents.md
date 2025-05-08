---
title: Agents
sidebar: 'docsSidebar'
---

# Understanding Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.2rem', fontWeight: 'normal'}}>
Master proven strategies for designing advanced, specialized AI agents that work together seamlessly to tackle complex challenges.
</h2>

## What is an Agent?

An Agent is like having a super-smart assistant powered by an LLM (Large Language Model). Think of it as your AI teammate that can think, reason, and take actions!

```python
Agent()  # That's it! You've created your first agent!
```

Note: Make sure to define all required environment variables—such as your OpenAI API key—in your .env file.

```bash title=".env"
OPENAI_API_KEY=your_api_key_here
```

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

## Let's Add Some Tools! 🛠️

Great! Now that you understand how Agents work, let's make yours even more powerful by adding some tools:

```python
Agent(
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
)
```

Agents can be equipped with tools—custom functions that expand their abilities beyond text generation. With tools, agents can perform tasks like calculations, interact with external systems, and process data. 

To learn more about creating and configurating tools, check out the [Tools documentation](/docs/concepts/tools.md).


## Using Memory in Agents


## Next Steps

- Try creating your own Agent with different tools
- Experiment with different configurations
- See an example agent in [Examples](/docs/examples)

Remember: The more you practice, the better you'll become at creating powerful Agents!