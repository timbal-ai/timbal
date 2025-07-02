---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Orchestrating AI Workflows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Design, connect, and control multi-step AI pipelines using Flows—Timbal's flexible workflow engine.
</h2>

---

## What Are Flows?

A **Flow** is a programmable pipeline that lets you chain together steps—functions, LLMs, or even other flows—while controlling how data moves between them. Flows enable you to build complex, intelligent workflows with clear logic, memory, and branching.

<CodeBlock language="python" code={`from timbal import Flow

flow = Flow(id="my_flow")`}/>

---

## Building Blocks of a Flow

<div className="cards-container">
  <div className="card">
    <div className="card-content">
      <h3>Steps</h3>
      <p>
        <strong>Steps</strong> are the core units of work.

        Each step can be:
        - a function
        - a BaseStep
        - another flow. 
        
      Steps process data, perform actions, and pass results onward.
      </p>
    </div>
  </div>
  <div className="card">
    <div className="card-content">
      <h3>Links</h3>
      <p>
        <strong>Links</strong> define the order and dependencies between steps. 
        
        They control how data and execution flow from one step to another, and can be used for tool calls, tool results, and conditional branching.
      </p>
    </div>
  </div>
</div>

<CodeBlock language="python" code={`flow = (
    Flow()
    .add_step("step_1", handler_1)
    .add_step("step_2", handler_2)
    .add_link("step_1", "step_2")
)`}/>


<div style={{ textAlign: 'center' }}>
  <img src="/img/dag_link.png" style={{ width: '20rem' }} />
</div>
---

## Controlling Step Inputs

When building flows, you often need to control how each step receives its inputs. Timbal provides two powerful methods for this:

- **Data Maps** (`set_data_map`): Dynamically connect a step's input to the output of another step or a flow input.
- **Data Values** (`set_data_value`): Set a static value or template for a step's input.

### Data Maps
**Purpose:**
Connect a step's input parameter to the output of another step, or to a flow input.

**Syntax:**
<CodeBlock language="python" code={`.set_data_map("step_name.parameter", "source")`}/>

- `step_name.parameter`: The input parameter of a step (e.g., `check.fahrenheit`).
- `source`: The data key to use as the value. This can be:
  - The output of another step (e.g., `to_fahrenheit.return`)
  - A flow input (e.g., `input_x`)

**Example:**
<CodeBlock language="python" code={`.set_data_map("to_fahrenheit.celsius", "get_temp.return")
.set_data_map("check.fahrenheit", "to_fahrenheit.return")`}/>

This means:
- The `celsius` parameter of the `to_fahrenheit` step receives the output of `get_temp`.
- The `fahrenheit` parameter of the `check` step receives the output of `to_fahrenheit`.

### Data Values
**Purpose:**
Set a static value or template for a step's input.

**Syntax:**
<CodeBlock language="python" code={`.set_data_value("step_name.parameter", value)`}/>

- `step_name.parameter`: The input parameter of a step (e.g., `check.threshold`).
- `value`: A constant (e.g., `86`), or a template string (e.g., `"{{step_1.return}} and {{step_2.return}}"`).

**Example:**
<CodeBlock language="python" code={`.set_data_value("check.threshold", 86)`}/>

This means:
- The `threshold` parameter of the `check` step will always be set to `86`.

### Inputs and Outputs

**Inputs** and **outputs** in a flow are special cases of data mapping:

- **Inputs**: Use `.set_input("step.parameter", "input_name")` to specify that a step should receive its value from a flow input.
- **Outputs**: Use `.set_output("step.return", "result_name")` to specify which step's output is returned by the flow.

:::note
The output key will always be "**.return**" (e.g., "to_fahrenheit.return"), since it refers to the return value of the step.
:::

**Example:**

<CodeBlock language="python" code={`flow = (
    Flow()
    .add_step("to_fahrenheit", celsius_to_fahrenheit)
    .set_input("to_fahrenheit.celsius", "input_celsius")  # input_celsius is a flow input
    .set_output("to_fahrenheit.return", "fahrenheit")   # expose flow output
)`}/>

This means:
- The flow expects an input called `input_celsius`.
- The output of `to_fahrenheit` will be available as `fahrenheit` in the flow's result.

---

### Example: Temperature Alert Flow

<CodeBlock language="python" code={`from timbal import Flow

def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32

def check_threshold(fahrenheit, threshold):
    if fahrenheit > threshold:
        return "Alert: Temperature is too high!"
    else:
        return "Temperature is normal."

flow = (
    Flow()
    .add_step("to_fahrenheit", celsius_to_fahrenheit)
    .add_step("check", check_threshold)
    # Map Celsius input parameter to the function
    .set_input("to_fahrenheit.celsius", "input_celsius")
    # Map Fahrenheit output to threshold checker
    .set_data_map("check.fahrenheit", "to_fahrenheit.return")
    # Set a static threshold value
    .set_data_value("check.threshold", 86)
    .set_output("check.return", "status")
)

async def main():
    result = await flow.complete(input_celsius=35)
    print(result.output["status"])
`}/>

---

## Dynamic Data with String Interpolation

Template strings let you combine and transform outputs from multiple steps.  
This is especially useful for LLM prompts or merging results.

<CodeBlock language="python" code={`from timbal import Flow

def get_first_name():
    return "Alice"

def get_last_name():
    return "Smith"

def check_full_name(full_name):
    if full_name == "Alice Smith":
        return "Welcome, Alice Smith!"
    else:
        return f"User {full_name} not recognized."

flow = (
    Flow()
    .add_step("first_name", get_first_name)
    .add_step("last_name", get_last_name)
    .add_step("validate", check_full_name)
    # Interpolate the outputs of first_name and last_name into a full name string
    .set_data_value("validate.full_name", "{{first_name.return}} {{last_name.return}}")
    .set_output("validate.return", "message")
)

async def main():
    result = await flow.complete()
    print(result.output["message"])`}/>

---

## Integrating LLMs

You can add LLMs (Large Language Models) as steps in your flow using .add_llm().
LLMs can use memory, call tools, and be chained with other steps for advanced reasoning.
- **Memory**: Use the memory_id parameter to enable persistent context across runs.
- **Tool Use**: Connect LLMs to tools or functions using .add_link(..., is_tool=True) and .add_link(..., is_tool_result=True) for advanced workflows.
- **Prompt Construction**: Use string interpolation to dynamically build prompts from previous step outputs.

Suppose you want to fetch an email, then have an LLM summarize it:

<CodeBlock language="python" code={`from timbal import Flow

def get_email():
    return "Hi team, let's meet tomorrow at 10am to discuss the project. Best, Alice"

flow = (
    Flow()
    .add_step("fetch_email", get_email)
    .add_llm("llm", model="gpt-4o-mini", memory_id="persistent_memory")
    # Use string interpolation to build the prompt from the previous step
    .set_data_value("llm.prompt", "Summarize this email: {{fetch_email.return}}")
    .set_output("llm.return", "summary")
)

async def main():
    result = await flow.complete()
    print(result.output["summary"].content[0].text)`}/>

**What’s happening here?***
- `fetch_email` retrieves the email text.
- The LLM step receives a prompt that includes the email content.
- The LLM generates a summary, which is returned as the flow output.

:::tip
You can chain multiple steps, use memory for context, and connect LLMs to external tools for even more powerful workflows.
:::

For more, see the [Advanced documentation](/flows/advanced)

---

## Enabling Memory and Finalizing Your Flow

To enable advanced features like persistent memory, you need to finalize your flow using the .compile() method.

Compiling your flow validates its structure and (optionally) attaches a state saver for memory.

### Why compile?

Compiling ensures your flow is ready for production, with all steps, data maps, and memory configured correctly.

### How to enable memory?

Pass a state saver (like InMemorySaver) to .compile() to persist context across runs.

See [State Savers](/state) for more information.

<CodeBlock language="python" code={`from timbal.state.savers import InMemorySaver

flow = (
    Flow()
    .add_llm(memory_id="persistent_memory")
    .compile(state_saver=InMemorySaver())
)`}/>

---

## How to Run Your Flow

Once your flow is defined and compiled, you can execute it in two main ways:

**Get the final output:**
<CodeBlock language="python" code={`result = await flow.complete(input_x=123)
print(result.output["result"])`}/>

**Stream events as they happen:**
<CodeBlock language="python" code={`async for event in flow.run(input_x=123):
    print(event)`}/>

---

## Quick Reference

- **Steps**: Units of work (functions, LLMs, or flows)
- **Links**: Control execution and data flow
- **Data Maps/Values**: Connect and set step inputs
- **String Interpolation**: Combine outputs flexibly
- **LLMs**: Add language models as steps
- **Inputs/Outputs**: Define what goes in and out of your flow
- **compile()**: Finalize and enable memory/state

For more, see the [Flows documentation](/flows), [Advanced Flow Concepts](/flows/advanced), and [Examples](/examples).

<style>{`
.cards-container {
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
  flex-wrap: wrap;
}

.card {
  flex: 1;
  min-width: 300px;
  background: var(--ifm-background-color);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  overflow: hidden;
}

.card-content {
  padding: 1.5rem;
}

.card-content h3 {
  color: var(--ifm-color-primary);
  margin-top: 0;
  margin-bottom: 1rem;
}

[data-theme='dark'] .card-content h3 {
  color: #9d7cff;
}

.card-content ul {
  list-style: disc;
  padding-left: 1.2em;
  margin: 0;
}

.card-content li {
  margin: 0.5rem 0;
}

.capabilities {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin: 2rem 0;
}

.capability {
  display: flex;
  align-items: flex-start;
  gap: 1.5rem;
  padding: 1.5rem;
  background: var(--ifm-background-color);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.capability-icon {
  font-size: 2rem;
  line-height: 1;
  flex-shrink: 0;
}

.capability-content {
  flex: 1;
}

.capability-content h3 {
  color: var(--ifm-font-color-base);
  margin-top: 0;
  margin-bottom: 0.5rem;
}

.capability-content p {
  margin: 0;
  line-height: 1.5;
}
`}</style>
