---
title: Integrating LLMs
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Integrating LLMs into Workflows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Integrate Large Language Models into your workflows for intelligent processing, analysis, and decision-making.
</h2>

---


## Integration Patterns

There are two primary ways to integrate LLMs into workflows.

### 1. Using `llm_router` as a Tool

The `llm_router` function is the most flexible way to integrate LLMs into workflows. It allows you to create reusable LLM tools with predefined configurations.

<CodeBlock language="python" code={`from timbal.core import Tool, Workflow
from timbal.core.llm_router import llm_router
from timbal.state import get_run_context
from timbal.types.message import Message

# Create an LLM tool with default parameters
summarizer_llm = Tool(
    name="summarizer",
    handler=llm_router,
    default_params={
        "model": "openai/gpt-4o-mini",
        "system_prompt": "You are a helpful assistant that summarizes text concisely.",
    }
)

# Use in workflow
workflow = (
    Workflow(name="text_processor")
    .step(summarizer_llm, messages=lambda: [
        Message.validate(f"Summarize: {get_run_context().get_data('.input')}")
    ])
)`}/>

### 2. Direct Agent Integration

Agents can be used directly as workflow steps.
<CodeBlock language="python" code={`from timbal.core import Agent, Workflow

# Create an agent
analysis_agent = Agent(
    name="data_analyzer",
    model="openai/gpt-4o-mini",
    system_prompt="Analyze the provided data and return structured insights."
)

# Use agent as workflow step
workflow = (
    Workflow(name="data_analysis")
    .step(analysis_agent)
)`}/>




---

## Advanced Patterns



### Conditional LLM Processing

Use LLMs to make decisions about workflow execution:

<CodeBlock language="python" code={`from timbal.core import Tool, Workflow
from timbal.core.llm_router import llm_router
from timbal.state import get_run_context
from timbal.types.message import Message

# Decision-making LLM
decision_llm = Tool(
    name="decision_maker",
    handler=llm_router,
    default_params={
        "model": "openai/gpt-4o-mini",
        "system_prompt": "Respond with 'urgent' or 'normal' based on the content."
    }
)

# Processing LLMs
urgent_processor = Tool(
    name="urgent_processor",
    handler=llm_router,
    default_params={
        "model": "openai/gpt-4o",
        "system_prompt": "Process urgent requests with high priority."
    }
)

normal_processor = Tool(
    name="normal_processor",
    handler=llm_router,
    default_params={
        "model": "openai/gpt-4o-mini",
        "system_prompt": "Process normal requests efficiently."
    }
)

# Conditional workflow
conditional_workflow = (
    Workflow(name="conditional_processing")
    .step(decision_llm, messages=lambda: [
        Message.validate(f"Classify urgency: {get_run_context().get_data('.input')}")
    ])
    .step(urgent_processor, 
          messages=lambda: [Message.validate(f"Process urgently: {get_run_context().get_data('.input')}")],
          when=lambda: "urgent" in get_run_context().get_data("decision_maker.output").lower())
    .step(normal_processor,
          messages=lambda: [Message.validate(f"Process normally: {get_run_context().get_data('.input')}")],
          when=lambda: "normal" in get_run_context().get_data("decision_maker.output").lower())
)`}/>

