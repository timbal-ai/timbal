---
title: Supervisor Agent
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';


When building complex AI applications, you often need multiple specialized agents to collaborate on different aspects of a task. A supervisor agent enables one agent to act as a supervisor, coordinating the work of other agents, each focused on their own area of expertise. This structure allows agents to delegate, collaborate, and produce more advanced outputs than any single agent alone.

In this example, this system consists of three agents:

- A **Copywriter agent** that writes the initial content.
- An **Editor agent** that refines the content.
- A **Publisher agent** that supervises and coordinates the other agents.

## Prerequisites

This example uses the `openai` model. Make sure to add `OPENAI_API_KEY` to your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## Copywriter agent

This `copywriter_agent` is responsible for writing the initial blog post content based on a given topic.

<CodeBlock language="python" code={`from timbal.core import Agent

copywriter_agent = Agent(
    name="copywriter-agent",
    system_prompt="You are a copywriter agent that writes blog post copy.",
    model="openai/gpt-4o"
)`}/>

## Copywriter tool

The `copywriter_tool` provides an interface to call the `copywriter_agent` and passes in the topic.

<CodeBlock language="python" code={`from timbal.core import Tool

async def copywriter_tool(topic: str) -> dict[str, str]:
    """Calls the copywriter agent to write blog post copy."""
    # Call the copywriter agent
    result = await copywriter_agent(prompt=f"Create a blog post about {topic}").collect()
    
    return {
        "copy": result.output.content[0].text
    }

# Create the Tool instance
copywriter_tool_instance = Tool(
    name="copywriter_tool",
    description="Calls the copywriter agent to write blog post copy.",
    handler=copywriter_tool
)`}/>

## Editor agent

This `editor_agent` takes the initial copy and refines it to improve quality and readability.

<CodeBlock language="python" code={`from timbal.core import Agent

editor_agent = Agent(
    name="editor-agent",
    system_prompt="You are an editor agent that edits blog post copy.",
    model="openai/gpt-4o-mini"
)`}/>

## Editor tool

The `editor_tool` provides an interface to call the `editor_agent` and passes in the copy.

<CodeBlock language="python" code={`from timbal.core import Tool

async def editor_tool(copy: str) -> dict[str, str]:
    """Calls the editor agent to edit blog post copy."""
    # Call the editor agent
    result = await editor_agent(prompt=f"Edit the following blog post only returning the edited copy: {copy}").collect()
    
    return {
        "copy": result.output.content[0].text
    }

# Create the Tool instance
editor_tool_instance = Tool(
    name="editor_tool",
    description="Calls the editor agent to edit blog post copy.",
    handler=editor_tool
)`}/>

## Publisher agent

This `publisher_agent` coordinates the entire process by calling the `copywriter_tool` first, then the `editor_tool`.

<CodeBlock language="python" code={`from timbal.core import Agent
from timbal.tools.example_copywriter_tool import copywriter_tool_instance
from timbal.tools.example_editor_tool import editor_tool_instance

publisher_agent = Agent(
    name="publisher-agent",
    system_prompt="""You are a publisher agent that first calls the copywriter agent to write blog post copy about a specific topic and then calls the editor agent to edit the copy. Just return the final edited copy.""",
    model="openai/gpt-4.1-mini",
    tools=[copywriter_tool_instance, editor_tool_instance]
)`}/>

## Registering the agents

All three agents are created and available for use in the workflow. In Timbal, agents are used directly without needing a central registry.

<CodeBlock language="python" code={`# All agents are already created and available
# No central registration needed - use them directly

# Available agents:
# - copywriter_agent
# - editor_agent  
# - publisher_agent

# Available tools:
# - copywriter_tool_instance
# - editor_tool_instance

# The publisher_agent already has access to both tools
# and can coordinate the entire workflow`}/>

## Example usage

Use the publisher agent directly by calling it with a prompt message.

<CodeBlock language="python" code={`import asyncio

async def main():    
    response = await publisher_agent(prompt="Write a blog post about React JavaScript frameworks. Only return the final edited copy.").collect()
    
    response_text = response.output.content[0].text
    print(response_text)

if __name__ == "__main__":
    asyncio.run(main())
`}/>


<div>
  <Link className={styles.card} href="https://github.com/your-repo/design-tools" target="_blank" style={{display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '1.2rem', flexWrap: 'nowrap'}}>
    <span className={styles.icon}><svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></span>
    <span style={{flexShrink: 0}}>Supervisor Agent</span>
    <span style={{flexShrink: 0, marginLeft: 'auto', fontSize: '1.5rem'}}>↗</span>
  </Link>
</div> 