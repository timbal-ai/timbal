---
sidebar: 'docsSidebar'
---

# Memory in LLMs

Timbal makes it easy to manage conversation memory in your graph. These how-to guides show how to implement different strategies for that. 
Memory in Timbal allows LLMs to maintain context across interactions.

## Memory Management

We have seen how to add an LLM.

```python
flow = (
    Flow()
    .add_llm(id="chat_llm")
)
```

This will be a possible result of conversation:

```python
# First message

user: "Hi! My name is David"

assistant: "Hi David! How can I help you?"

# Second message

user: "What is my name?"

assistant: "I'm sorry I don't have access to know what is your name."

```

Why it doesn't know the name if I provided it before? This is because this LLM does not have memory... Let's provide it a memory:

Only adding in `memory_id`the name of the `id`, you are setting it as to have memory.

```python
flow = (
    Flow()
    .add_llm(id="chat_llm", memory_id="chat_llm")
)
```

Now the same conversation will look like this:

```python
# First message

user: "Hi! My name is David"

assistant: "Hi David! How can I help you?"

# Second message

user: "What is my name?"

assistant: "Your name is David. How can I assist you today?"

```

:::warning[ALERT]
In this version the `memory_id` has to have the `id` name of a llm!
:::


### Memory Sharing

Let's see what happens

LLMs can share memory by using the same `memory_id`. This is useful when you want multiple LLMs to have access to the same conversation history:

```python
flow = (
    Flow()
    # Both LLMs share the same memory
    .add_llm(id="llm1", memory_id="shared_memory")
    .add_llm(id="llm2", memory_id="shared_memory")
    )
```


Memory sharing is particularly useful in agent scenarios where you want the agent to maintain context across different interactions.

But if we hae a lot of interaction, the memory will be huge! 

No problem! We can limit the memory window size.

### Memory Window Size

Let's see an example:

```python
flow = (
    Flow()
    .add_llm(id="chat_llm", memory_id="chat_llm", memory_window_size=5)
)
```

So in this case, the memory will be limited to the last 5 messages.

And if we set a window size of 0? The memory will be limited to the last 0 messages. So... no memory!


## Memory and State Saving

When using a state saver, memory persists between flow executions:

```python
flow = (
Flow()
.add_llm(memory_id="persistent_memory")
.compile(state_saver=InMemorySaver())
)
```


This allows conversations to continue across multiple flow runs while respecting the configured window sizes.

See more about state savers in the [Saver](/docs/guides/saver) guide.
