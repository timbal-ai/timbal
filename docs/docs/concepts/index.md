---
title: Overview
sidebar: 'docsSidebar'
---
import DocCardList from '@theme/DocCardList';
import { useCurrentSidebarCategory } from '@docusaurus/theme-common';


# Building with Timbal

## Core Concept: Flows as DAGs

All the applications made with **Timbal** are execution **flows** that are represented as **directed acyclic graphs (DAGs)**. 🎯

## What is a DAG?

A DAG is a directed graph that has no cycles. 

:::note[What is a Graph?]
A graph is a way to show how things are connected. 
It has dots (called **nodes**) and lines (called **edges**) that link them. Some lines have directions, and some don’t. 
If the line has an arrow, it means the direction of the connection. It can be a one-way or two-way connection.
The edges can show relationships, paths, or connections between the nodes. 
:::

In a DAG, each **node** represents a `step` in the flow, and each **edge** represents a `link` between two steps.
Nodes execute and edges mark the order of execution.

<div align="center">
<img src="/img/dag_example.png" alt="Alt text" style={{ width: '70%'}}/>
</div>

## Creating a Flow

First, you need to import the `Flow` class:

```python
from timbal import Flow
```

Create a new `Flow` object with an optional identifier:

```python
flow_1 = Flow(id="my-flow")  # Custom ID
flow_2 = Flow()              # Default ID: "flow"
```

:::tip[Why Use IDs? 🏷️]
IDs are particularly useful when creating multiple flows that you want to compose together!
:::

### Builder Pattern

Timbal uses the builder pattern for creating clear and readable flows. Here are three equivalent approaches:

```python
# Method 1: Traditional method calls
flow = Flow()
flow.add_step()
flow.add_step()

# Method 2: Reassignment
flow = Flow()
flow = flow.add_step()
flow = flow.add_step()

# Method 3: Method chaining (recommended) ⭐
flow = (
    Flow()
    .add_step()
    .add_step()
)
```

:::note[Why Method Chaining?]
The method chaining approach:
- Makes flow structure more visible 👀
- Reduces repetition ♻️
- Keeps flow configuration in one cohesive block 📦
:::

Choose whichever style matches your preferences and coding standards!😊 

Now that we have initialized our flow, we can start by adding methods.

## Adding Steps to Flow

Let's explore how to add steps to your flow! 🚀

### Basic Step Addition

```python
flow.add_step(handler)
```

:::success[Simple, right? ✨]
You've just added your first step!
:::

### Multiple Steps Challenge

But what about adding more steps?

If you try this:

```python
flow.add_step(handler)
flow.add_step(handler) # ⚠️ This will raise an error!
```

:::warning[Why Did This Fail? 🤔]
This will raise an error because the step `handler` already exists.
:::

But if we want two nodes to have the same function? 

:::tip[Solution: Named Steps]
We can add a name to the step like this:

```python
flow.add_step("step_1", handler) # ✅ Works!
flow.add_step("step_2", handler) # ✅ Works too!
```
:::

Now we have two steps with the same function.

Fantastic! You know how to put a handler in the step. Let's go a little deeper.

**What can be in a step?** 🔍
A step can be any function! We can set up the functions like this:

```python
def handler(x: int = Field(description="The input to the handler")):
    print(x)
```

See more about types of steps in the [steps](docs/advanced/index.md) section.

Now that you have seen how to set up the function. Imagine if you have 10 inputs and you want to pass them to the step.
Tired of setting data maps for multiple inputs? We've got you covered! 

:::tip[Easy Input Setting 🌟]
You can set the inputs of the step as kwargs.

```python
flow.add_step("step_1", handler, x=1, y=2)  # 🎯 All inputs in one line!
```
:::

Now that you can add as many steps as you want, let's try to connect them.

## Adding Links to Flow

### Basic Linking

Connect two steps like this:

```python
flow = (
    Flow()
    .add_step("step_1", handler)
    .add_step("step_2", handler)
    .add_link("step_1", "step_2") # Creates connection
)
```

You did it! Now we have a flow with two steps and a link between them.

<div align="center">
<img src="/img/dag_link.png" alt="Alt text" style={{ width: '50%'}}/>
</div>

*See more properties in the advanced [links](docs/advanced/link.md) section.*

But what about the data?... How we can set the input for each step? 

That's easy! Let's take a look at it:

## Setting Data

Our function *handler* has an input *x* that is an integer. We can set the input of the step like this:

```python {4}
flow = (
    Flow()
    .add_step("step_1", handler)
    .set_data_value("step_1.x", 1) # Sets input x=1
)
```

And the outputs?

Each output will be the `return` of the step. But we can map the output to the specific name we want to use.

```python
flow.set_output("result", "step_1.return")
```

And if we have two steps, how can we set the input of the second step as the output of the first step?

We only have to apply the 2 things explained before:

```python {6}
flow = (
    Flow()
    .add_step("step_1", handler, x=1)
    .add_step("step_2", handler)
    .add_link("step_1", "step_2") # Link the steps
    .set_data_map("step_2.x", "step_1.return") # Set input of step 2 as the output of step 1
)
```

And if we want each step return to be the input of the next step? We have to link all the steps...
It is redudant!

:::tip[Smart Linking! 🌟]
`set_data_map` and `set_data_value` are smart! They automatically create links between steps when needed. This means you can write:

```python
flow = (
    Flow()
    .add_step("step_1", handler, x=1)
    .add_step("step_2", handler)
    .set_data_map("step_2.x", "step_1.return")  # Link created automatically!
)
```
:::

## Savers

Save your flow's state. From Timbal, we provide you a simple saver that saves the flow's state in memory: `InMemorySaver`.

```python
from timbal.state.savers import InMemorySaver

flow.compile(state_saver=InMemorySaver())
```

You only need to set the `state_saver` when you compile the flow.
*See more about it in the [saver](docs/guides/saver.md) section.*

But we don't forget the most important part: how can we obtain the result of the flow?

## Running the Flow

To run the flow, we can use the `run()` method.

Get detailed execution events:

```python
async for event in flow.run():
    print(event) 
```

:::note[Running the Flow]
The `run()` method returns an async iterator that yields events as the flow executes.

:::

Or if it does not matter the specific event. Get the final result directly:

```python
result = await flow.complete()
```

Okay! Now you now the basics of Timbal.