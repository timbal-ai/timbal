---
title: Streaming
sidebar: 'docsSidebar'
---

# Streaming

Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.


There are two ways I could stream the output of a flow:

If I want to see the output as it's being generated. 

```python
Hi my name is... 
```


1. Using the `run()` method, which returns an async iterator that yields events as the flow executes.

```python
async for event in flow.run():
    print(event)
```

Otherwise I want to see the final result of the flow.

```python
Hi my name is David. How are you?
```

2. Using the `complete()` method, which returns the final result of the flow.

```python
result = await flow.complete()
```

:::warning[Important ⚠️]
Remember: These are coroutines! You need to:
- Use `await`
- Run them in an async function
:::

### Working with Events

Events tell you what's happening in your flow. Here's what you can do with them:

If you want to know when a step starts...
```python
async for event in flow.run():
    if event.type == "STEP_START":
        print(f"Starting step: {event.step_id}")
```

If you want to see output as it's generated...
```python
async for event in flow.run():
    if event.type == "STEP_CHUNK":
        print(event.step_chunk, end="")  # Print each piece as it arrives
```

If you want to know step results and timing...
```python
async for event in flow.run():
    if event.type == "STEP_OUTPUT":
        print(f"Step completed in {event.elapsed_time}ms")
        print(f"Result: {event.step_result}")
```

If you want the final flow results...
```python
async for event in flow.run():
    if event.type == "FLOW_OUTPUT":
        print(f"Flow finished in {event.elapsed_time}ms")
        print(f"Outputs: {event.outputs}")
```