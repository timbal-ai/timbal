---
sidebar_position: 2
sidebar: 'docsSidebar'
---

# Quickstart

Build your first flow with Timbal with 5 lines of code.

We'll start with a **simple chatbot** and gradually enhance it with advanced features. Let's dive in! 🌟

Before moving forward, ensure you've completed the installation of Timbal. If you haven't set it up yet, follow the **[installation guide](./installation)** to get started.

## Part 1: Build a Simple Chatbot

🛠 Let's create a simple chatbot that can respond to user messages.

**1. Import the class `Flow` from the `timbal` package.**

```python title="flow.py"
from timbal import Flow
```

**1. Initialize a `Flow` object.**

```python title="flow.py"
flow = Flow()
```

**3. Add an LLM node to the flow.**
 Nodes represent units of work. They are typically regular python functions.
The `add_llm()` function in our Flow will call an LLM and return the response.

```python title="flow.py"
flow.add_llm(model="gpt-4o-mini")
```

**4. Map the input `prompt` to the LLM's prompt.**
With this line, we're instructing the flow to use the provided `prompt` as the input for the LLM. 

This allows us to dynamically pass prompts into the flow, ensuring flexible and customizable interactions.

```python title="flow.py"
flow.set_data_map("llm.prompt", "prompt")
```

**5. Map the LLM's response to the output key `response`.**
With this line, we're directing the flow to set the LLM's response as the output. The result will be a dictionary where the key response holds the generated output, making it easy to access and use.

```python title="flow.py"
flow.set_output("response", "llm.return")
```

**6. Set your environment variables**
Before running your flow, make sure you have the keys needed set as environment variables in your `.env` file:

👀 It will depend on the LLM you're using, in this case we're using OpenAI.
```
OPENAI_API_KEY=...
```

Now let's run the chatbot!

```python title="flow.py"
from timbal import Flow
from timbal.state.savers import InMemorySaver

flow = (
    Flow()
    .add_llm(model="gpt-4o-mini") 
    .set_data_map("llm.prompt", "prompt")         
    .set_output("response", "llm.return")          
)
```

:::tip[Congratulations!]
You've just created your first Timbal flow!
:::

This is the simplest flow you can create. But it does not retain the conversation:

```
user: My name is David
assistant: Hello David, how can I help you today?
user: What is my name?
assistant: I don't know but you can tell me.
```

Let's add memory to the chatbot.

## Part 2: Enhancing the Chatbot with Memory

🧠 Let's add memory to the chatbot.

The code is the same as before but we're adding a `memory_id` to the LLM node.

```python title="flow.py"
flow.add_llm(model="gpt-4o-mini", memory_id="llm")
```

So the code will look like this:

```python {6} title="flow.py"
from timbal import Flow
from timbal.state.savers import InMemorySaver

flow = (
    Flow()
    .add_llm(model="gpt-4o-mini", memory_id="llm")
    .set_data_map("llm.prompt", "prompt")
    .set_output("response", "llm.return")
)
```

## Part 3: Enhancing the Chatbot with Tools

When the chatbot encounters questions it can’t answer from memory, we’ll equip it with a tool. This allows the bot to fetch relevant information in real time, improving its responses. 🚀

For this example, we will use a tool capable of returning the current weather.

**1. Define the `get_weather` tool.**
```python title="flow.py"
def get_weather() -> str:
    ...
```

**2. Add an agent node to the flow with the tool.**
We need to add the tool to the llm as a new node.
```python title="flow.py"
flow.add_agent(model="gpt-4o-mini", memory_id="agent", tools=[get_weather])
```

**3. Set the data map of the agent to the prompt.**
We need to pass the prompt to the agent.
```python title="flow.py"
flow.set_data_map("agent.prompt", "prompt")
```

**4. Return the response as we want**
We can map the response to the output of the flow.
```python title="flow.py"
flow.set_output("response", "agent.return")
```

Here's the full code:

```python {6} title="flow.py"
from timbal import Flow
from timbal.state.savers import InMemorySaver

flow = (
    Flow()
    .add_agent(model="gpt-4o-mini", memory_id="agent", tools=[get_weather])
    .set_data_map("agent.prompt", "prompt")
    .set_output("response", "agent.return")
)
```

Let's visualize the graph we've built.

:::tip[Visualize the flow]
You can visualize the flow by calling `flow.plot()`.
:::

<div align="center">
<img src="/img/dag_tools.png" alt="Alt text" style={{ width: '50%'}}/>
</div>

**That's it!** 💥 With 5 lines of code we've created a chatbot that indeed can answer questions about the current time.
