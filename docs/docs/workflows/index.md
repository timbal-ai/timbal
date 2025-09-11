---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Orchestrating AI Workflows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Design, connect, and control multi-step AI pipelines using Workflows.
</h2>

---

## What Are Workflows?

<span style={{color: 'var(--timbal-purple)'}}><strong>Workflows</strong></span> are programmable execution pipelines that **orchestrate step-by-step processing with explicit control flow**.

<CodeBlock language="python" code={`from timbal.core import Workflow

workflow = Workflow(name="my_workflow")`}/>

This is the first step to create a workflow. Next, add components as building blocks.

---

## Building Blocks of a Workflow: Steps
<strong>Steps</strong> are the core units of work, which can process data, perform actions and pass results onward.

<!-- ### DAG-Based Execution
Workflows form a **Directed Acyclic Graph (DAG)** where:
- Steps can run in parallel when dependencies allow
- No circular dependencies (prevents infinite loops)
- Automatic dependency resolution and execution ordering

<div style={{ textAlign: 'center', display: 'flex', justifyContent: 'center', alignItems: 'center', margin: '0rem 0' }}>
  <img src="/img/dag_link.png" style={{ width: '20rem', maxWidth: '100%' }} />
</div> -->




### Adding Steps to the Workflow

Workflows use `.step()` method to add steps. **Any Runnable is valid to create a step**. This includes sync/async functions, Tool objects, Agents, and other Workflows. You can pass fixed parameters to steps using keyword arguments:

<!-- - **Functions**: Direct function references
- **Tools**: Tool objects with handlers
- **Dictionaries**: Tool configurations
- **Other Workflows**: Nested workflow components

<CodeBlock language="python" code={`workflow = (Workflow(name="my_workflow")
    .step(step_1)
    .step(step_2)
)`}/> -->


<!-- ### Adding Steps with Parameters -->



<CodeBlock language="python" code={`def celsius_to_fahrenheit(celsius: float) -> float:
    return (celsius * 9/5) + 32

def check_threshold(temperature: float, threshold: float) -> str:
    return "Alert!" if temperature > threshold else "Normal"
    
workflow = (Workflow(name="temperature_alert")
    .step(celsius_to_fahrenheit, celsius=35)
    .step(check_threshold, temperature=80, threshold=lambda: 85)
)`}/>


### Reusing Functions

**Each step in a workflow must have a unique name**. If you need to use the same function multiple times in a workflow, wrap it in a new Tool each time with distinct names.

<CodeBlock language="python" highlight="10" code={`# Create a Tool to reuse the function
threshold_checker_tool = Tool(
    name="threshold_checker",
    handler=check_threshold
)

workflow = (Workflow(name="temperature_monitoring", temperature=80)
    .step(celsius_to_fahrenheit, celsius=35)
    .step(check_threshold, threshold=lambda: 85)
    .step(threshold_checker_tool, threshold=lambda: 100)
)`}/>



<!-- ### Connecting Steps

Use `get_run_context().get_data("step_name.output")` to access outputs from neighbour steps:

<CodeBlock language="python" code={`workflow = (Workflow(name="temperature_alert")
    .step(step1, celsius=35)
    .step(step2, temperature=lambda: get_run_context().get_data("step1.output"))
)`}/>

The framework automatically handles the dependency of one step on another step's data.

In the above example, you don't need `.link()` because step2 uses step1's output. When a step depends on another step's data, they run sequentially (implicit linking).

To force sequential execution, use `.link()`:

<CodeBlock language="python" code={`def fetch_data():    # Takes 2 seconds
    time.sleep(2)
    return "data"

def process_data():  # Takes 3 seconds
    time.sleep(3)
    return "processed"

workflow = (Workflow(name="sequential_flow")
    .step(fetch_data)      # Starts at 0s, finishes at 2s
    .step(process_data)    # Starts at 2s, finishes at 5s
    .link("fetch_data", "process_data")
)
# Total time: 5 seconds (2 + 3)`}/>

Without `.link()`, steps run in parallel:

<CodeBlock language="python" code={`def send_email():       # Takes 2 seconds
    time.sleep(2)
    return "email sent"

def update_database():  # Takes 3 seconds
    time.sleep(3)
    return "db updated"

workflow = (Workflow(name="parallel_flow")
    .step(send_email)      # Starts at 0s, finishes at 2s
    .step(update_database) # Starts at 0s, finishes at 3s
)
# Total time: 3 seconds (max of 2 and 3)`}/> -->

---

## Integrating LLMs

You can add LLMs as steps. Timbal provides `llm_router` function that set as a Tool can work as an step.

<CodeBlock language="python" code={`from timbal.core import Tool
from timbal.core.workflow import Workflow
from timbal.core.llm_router import llm_router
from timbal.state import get_run_context
from timbal.types.message import Message

def get_email() -> str:
    return "Hi team, let's meet tomorrow at 10am to discuss the project. Best, Alice"

openai_llm = Tool(
  name="openai_llm",
  handler=llm_router,
  default_params={
    "model": "openai/gpt-4o-mini",
    "system_prompt": "You are a helpful assistant that summarizes emails concisely.",
  }
)

workflow = (
    Workflow(name="email_summarizer")
    .step(get_email)
    .step(openai_llm, messages=lambda: [Message.validate(f"Summarize this email: {get_run_context().get_data('get_email.output')}")])
    .link("get_email", "openai_llm")
)`}/>

## Default Parameters

Look in the above example that the parameter of the function `openai_llm` comes from `default_params` ('model' and 'system_prompt') and runtime parameters (messages).

Default parameters in Timbal allow you to set predefined values that are automatically injected into your runnable components, providing flexibility and reducing boilerplate code.

Default parameters are defined when creating a runnable and are merged with runtime parameters. Runtime parameters always override default parameters.

---

## Running the Workflow

Once your Workflow is defined, you can execute it in two main ways:

**Get the final output:**
<CodeBlock language="python" code={`result = await workflow().collect()
print(result.output)`}/>

**Stream events as they happen:**
<CodeBlock language="python" code={`async for event in workflow():
    print(event)`}/>

If a function in your flow needs a value per run (e.g., x), pass it when you call the workflow:

- Inputs are routed only to steps that declare those parameters
- Runtime inputs override step defaults
- The same input name can feed multiple steps unless a step overrides it

<CodeBlock language="python" code={`from timbal.core import Workflow

def multiply(x: int) -> int:
    return x * 2

workflow = Workflow(name="simple_flow").step(multiply)

# Run with a per-run input
result = await workflow(x=1).collect()   # x is routed to multiply
print(result.output)  # 2

# Or stream events while using the same input
async for event in workflow(x=3):
    print(event)  # Final output will be 6`}/>


---

## Key Features
- **Parallel Execution**: Independent steps run concurrently
- **Error Isolation**: Failed steps skip dependents, others continue
- **Type Safety**: Automatic parameter validation via Pydantic
- **Composition**: Workflows can contain other workflows

Check [Context](/workflows_v2/context.md) for data sharing between steps, and [Control Flow](/workflows_v2/control_flow.md) for different step execution behaviors.



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