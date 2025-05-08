---
sidebar_position: 2
sidebar: 'docsSidebar'
---

# Quickstart

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.2rem', fontWeight: 'normal'}}>
Build your first flow with Timbal with 5 lines of code.
</h2>

<br />

We'll start implementing an **agent**. It will be a **simple chatbot** and gradually enhance it with advanced features. Let's dive in!

Before moving forward, ensure you've completed the installation of Timbal. If you haven't set it up yet, follow the **[installation guide](./installation)** to get started.

## Part 1: Build a Simple Chatbot

🛠 Let's create a simple chatbot that can respond to user messages.

**1. Import the class `Flow` from the `timbal` package.**

```python title="flow.py"
from timbal import Agent
```

**2. Initialize a `Flow` object.**

```python title="flow.py"
flow = Agent()
```

**3. Set your environment variables**
Before running your flow, make sure you have the keys needed set as environment variables in your `.env` file:

👀 It will depend on the LLM you're using, in this case, the default model is OpenAI.
```bash title=".env"
OPENAI_API_KEY=your_api_key_here
```

Only with the `Agent` class we have a flow that represents a llm that receives a `prompt` and returns a `response`.

Now let's run the chatbot!

```python title="flow.py"
response = flow.complete(prompt="What is the capital of Germany?")
print(response.content[0].text)
```

You will see an output like this:

```
The capital of Germany is Berlin.
```

:::tip[Congratulations!]
You've just created your first Timbal flow!
:::

This is the simplest flow you can create.

You can modify it as you want. For example, you can add tools to the agent.

## Part 2: Enhancing the Chatbot with Tools

When the chatbot encounters questions it can't answer from memory, we'll equip it with a tool. This allows the bot to fetch relevant information in real time, improving its responses.

For this example, we will use a tool capable of returning the current weather.

**1. We have a `search` tool defined.**
```python title="flow.py"
from timbal.steps.perplexity import search
```

**2. Add an agent node to the flow with the tool.**
We need to add the tool to the llm as a new node.
```python title="flow.py"
flow = Agent(tools=[search])
```

Let's visualize the graph we've built.

:::tip[Visualize the flow]
You can visualize the flow by calling `flow.plot()`.
:::

<div align="center">
<img src="/timbal/img/dag_tools.png" style={{ width: '50%'}}/>
</div>


But it does not retain the conversation:

```
user: My name is David
assistant: Hello David, how can I help you today?
user: What is my name?
assistant: I don't know but you can tell me.
```

Let's add memory to the chatbot.

## Part 3: Enhancing the Chatbot with Memory

Let's add memory to the chatbot.

It is very simple, we just need to set a context.

```python title="flow.py"
from timbal.state.savers import InMemorySaver
state_saver = InMemorySaver()

flow = Agent(
    state_saver=state_saver
)
```

And when we run the flow we have to add the `RunContext()` parameter. In order to have the tracebility of that run and the previous.

```python title="flow.py"
run_context = RunContext()
flow_output_event = await agent.complete(context=run_context, prompt=prompt)
run_context = RunContext(parent_id=flow_output_event.run_id)
```

And the previous conversation will look like:

```
user: My name is David
assistant: Hello David, how can I help you today?
user: What is my name?
assistant: Your name is David. Do you need anything else?
```

**That's it!** With 5 lines of code we've created a chatbot that has memory and can search internet.