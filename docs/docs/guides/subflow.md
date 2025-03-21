---
title: Subflows
sidebar: 'docsSidebar'
---

# Subflows

Subflows allow you to create modular, reusable flows that can be embedded within larger flows. Think of them as building blocks that you can use to create more complex applications!


## Creating a Simple Subflow

Let's start with a basic example. Here's a flow that uses an LLM:

```python
subflow = (
    Flow()
    .add_llm(id="llm")
    .set_data_map("llm.prompt", "prompt")
    .set_output("response", "llm.return")
)
```

:::note[Understanding the Subflow]
This flow:
1. Has an LLM step
2. Takes a `prompt` as input
3. Returns the LLM's response as output
:::

Add this flow to your main flow like this:


```python
flow = (
    Flow()
    .add_step("subflow", subflow)
    .set_data_value("subflow.prompt", "Tell me something interesting")
    .set_output("response", "subflow.return")
)
```

:::tip[Pro Tip ⭐]
Subflows are treated just like any other step in your main flow. You can:
- Set their inputs
- Use their outputs
- Connect them to other steps
:::

You can even chain multiple subflows together! Here's an example where one subflow validates the output of another:

```python
flow = (
    Flow()
    # First subflow: Generate content
    .add_step("subflow", subflow)
    .set_data_value("subflow.prompt", "Tell me something interesting")
    # Second subflow: Validate content
    .add_step("subflow2", subflow)
    .set_data_value("subflow2.prompt", "Is this information correct? {{subflow.return.response}}")
    .set_output("response", "subflow2.return")
)
```

:::note[Subgraphs]
Subgraphs are a way to create reusable flows within a larger flow.
:::