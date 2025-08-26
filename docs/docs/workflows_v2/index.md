---
title: Overview
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Orchestrating AI Workflows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Design, connect, and control multi-step AI pipelines using Workflows—Timbal's flexible workflow engine.
</h2>

---

## What Are Workflows?

:::warning[Work in Progress]
The Workflow class is currently under development and not fully implemented. The examples in this documentation may not work as expected.
:::

A **Workflow** is a programmable pipeline that lets you chain together steps—functions, LLMs, or even other workflows—while controlling how data moves between them. Workflows enable you to build complex, intelligent pipelines with clear logic, memory, and branching.

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow

workflow = Workflow(name="my_workflow")`}/>

---

## Building Blocks of a Workflow

<div className="cards-container">
  <div className="card">
    <div className="card-content">
      <h3>Steps</h3>
      <p>
        <strong>Steps</strong> are the core units of work.

        Each step can be:
        - a function
        - a Tool
        - another workflow
        
      Steps process data, perform actions, and pass results onward.
      </p>
    </div>
  </div>
  <div className="card">
    <div className="card-content">
      <h3>Execution Order</h3>
      <p>
        <strong>Execution order</strong> is determined by the sequence in which steps are added to the workflow.
        
        Steps execute in the order they are added, with data flowing from one step to the next through parameter mapping.
      </p>
    </div>
  </div>
</div>

<CodeBlock language="python" code={`workflow = (
    Workflow(name="my_workflow")
    .add_step(step_1)
    .add_step(step_2)
)`}/>

<div style={{ textAlign: 'center', display: 'flex', justifyContent: 'center', alignItems: 'center', margin: '0rem 0' }}>
  <img src="/img/dag_link.png" style={{ width: '20rem', maxWidth: '100%' }} />
</div>
---

## Adding Steps to Your Workflow

Workflows use the `.add_step()` method to add steps. Each step can be:

- **Functions**: Direct function references
- **Tools**: Tool objects with handlers
- **Dictionaries**: Tool configurations
- **Other Workflows**: Nested workflow components

### Adding Function Steps

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow

def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32

def check_threshold(fahrenheit: float, threshold: float = 86) -> str:
    if fahrenheit > threshold:
        return "Alert: Temperature is too high!"
    else:
        return "Temperature is normal."

workflow = (
    Workflow(name="temperature_alert")
    .add_step(celsius_to_fahrenheit)
    .add_step(check_threshold)
)`}/>

### Adding Tool Steps

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow, Tool

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 25°C."

weather_tool = Tool(handler=get_weather)

workflow = (
    Workflow(name="weather_workflow")
    .add_step(weather_tool)
)`}/>

### Adding Steps with Parameters

You can pass fixed parameters to steps using keyword arguments:

<CodeBlock language="python" code={`workflow = (
    Workflow(name="temperature_alert")
    .add_step(celsius_to_fahrenheit, celsius=35)
    .add_step(check_threshold, threshold=90)
)`}/>

---

## Parameter Mapping and Data Flow

Workflows automatically handle parameter mapping between steps. The output of one step becomes the input of the next step based on function signatures.

### Automatic Parameter Mapping

<CodeBlock language="python" code={`def get_temperature() -> float:
    return 35.0

def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32

def check_threshold(fahrenheit: float, threshold: float = 86) -> str:
    if fahrenheit > threshold:
        return "Alert: Temperature is too high!"
    else:
        return "Temperature is normal."

workflow = (
    Workflow(name="temperature_alert")
    .add_step(get_temperature)
    .add_step(celsius_to_fahrenheit)  # Receives output from get_temperature
    .add_step(check_threshold)        # Receives output from celsius_to_fahrenheit
)`}/>

### Fixed Parameters

You can override automatic data flow with fixed parameters:

<CodeBlock language="python" code={`workflow = (
    Workflow(name="temperature_alert")
    .add_step(get_temperature)
    .add_step(celsius_to_fahrenheit)
    .add_step(check_threshold, threshold=90)  # Fixed threshold value
)`}/>

---

## Integrating LLMs

You can add LLMs as steps in your workflow using the LLM handlers:

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.core_v2.handlers import llm_router

def get_email() -> str:
    return "Hi team, let's meet tomorrow at 10am to discuss the project. Best, Alice"

workflow = (
    Workflow(name="email_summarizer")
    .add_step(get_email)
    .add_step(openai_llm, model="gpt-4o-mini", prompt="Summarize this email: {{get_email.return}}")
)`}/>

---

## Running Your Workflow

Once your workflow is defined, you can execute it in two main ways:

**Get the final output:**
<CodeBlock language="python" code={`result = await workflow.complete()
print(result.output)`}/>

**Stream events as they happen:**
<CodeBlock language="python" code={`async for event in workflow.run():
    print(event)`}/>

---

## Example: Complete Temperature Alert Workflow

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow

def get_temperature() -> float:
    return 35.0

def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32

def check_threshold(fahrenheit: float, threshold: float = 86) -> str:
    if fahrenheit > threshold:
        return "Alert: Temperature is too high!"
    else:
        return "Temperature is normal."

workflow = (
    Workflow(name="temperature_alert")
    .add_step(get_temperature)
    .add_step(celsius_to_fahrenheit)
    .add_step(check_threshold, threshold=90)
)

async def main():
    result = await workflow.complete()
    print(result.output)  # Will contain the final result from check_threshold
`}/>

---

For more, see the [Workflows documentation](/workflows), [Advanced Workflow Concepts](/workflows/advanced), and [Examples](/examples).

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
