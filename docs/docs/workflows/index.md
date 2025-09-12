---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Orchestrating AI Workflows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Build multi-step AI pipelines using Workflows.
</h2>

---

## What Are Workflows?

<span style={{color: 'var(--timbal-purple)'}}><strong>Workflows</strong></span> are programmable execution pipelines that **orchestrate step-by-step processing with explicit control flow**.

<CodeBlock language="python" code={`from timbal.core import Workflow

workflow = Workflow(name="my_workflow")`}/>

This creates a basic workflow. Next, add steps to build your pipeline.

---

## Building Blocks of a Workflow: Steps
<strong>Steps</strong> are the core units of work, which can process data, perform actions and pass results onward.


### Adding Steps to the Workflow

Use `.step()` to add steps to workflows. **Any Runnable is valid to create a step**.

<CodeBlock language="python" code={`def celsius_to_fahrenheit(celsius: float) -> float:
    return (celsius * 9/5) + 32

def check_threshold(temperature: float, threshold: float) -> str:
    return "Alert!" if temperature > threshold else "Normal"
    
workflow = (Workflow(name="temperature_alert")
    .step(celsius_to_fahrenheit, celsius=35)
    .step(check_threshold, temperature=80, threshold=lambda: 85)
)`}/>


### Step Names and Reusing Functions

**Each step in a workflow must have a unique name**. Like all Runnables, steps are identified by their names, which must be distinct within the workflow.

To use the same function multiple times in a workflow, wrap it in a new Tool with a distinct name for each usage:

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

---

## Running the Workflow

You can run your Workflow in two different ways:

**Get the final output:**
<CodeBlock language="python" code={`result = await workflow().collect()
print(result.output)`}/>

**Stream events as they happen:**
<CodeBlock language="python" code={`async for event in workflow():
    print(event)`}/>


---

## Key Features
- **Composition**: Workflows can contain other workflows
- **Parallel Execution**: Independent steps run concurrently
- **Type Safety**: Automatic parameter validation via Pydantic
- **Error Isolation**: Failed steps skip dependents, others continue

Check [Context](/workflows/context.md) for data sharing between steps, and [Control Flow](/workflows/control_flow.md) for different step execution behaviors.



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