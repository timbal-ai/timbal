---
title: Adding a workflow
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

When building AI agents, it can be useful to combine them with workflows that perform multi-step tasks or fetch structured data. Timbal lets you pass workflows to an agent using the `workflows` parameter. Workflows provide a way for agents to trigger predefined sequences of steps, giving them access to more complex operations than a single tool can provide.

## Prerequisites

This example uses the `openai` model. Make sure to add `OPENAI_API_KEY` to your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## Creating a workflow

This workflow retrieves English Premier League fixtures for a given date. Clear input and output schemas keep the data predictable and easy for the agent to use.

<CodeBlock language="python" code={`import asyncio
import json
import urllib.request
from typing import Any

from timbal.core import Workflow, Tool

async def get_fixtures(date: str) -> dict[str, Any]:
    """Fetch match fixtures for English Premier League matches."""
    url = f"https://www.thesportsdb.com/api/v1/json/123/eventsday.php?d={date}&l=English_Premier_League"
    
    # Use run_in_executor to run the blocking urllib.request.urlopen in a thread
    response = await asyncio.get_event_loop().run_in_executor(None, lambda: urllib.request.urlopen(url))
    data = await asyncio.get_event_loop().run_in_executor(None, lambda: response.read())
    response_data = json.loads(data.decode())
    
    # Debug: print the full API response
    print(f"API Response: {response_data}")
    
    events = response_data.get("events", [])
    
    return {
        "fixtures": events,
        "raw_response": response_data,
        "url": url
    }

# Create the Tool for fetching fixtures
get_fixtures_tool = Tool(
    name="get_fixtures",
    description="Fetch match fixtures for English Premier League matches",
    handler=get_fixtures
)

# Create the workflow
soccer_workflow = (
    Workflow(name="soccer_workflow")
    .step(get_fixtures_tool)
)`}/>

## Adding a workflow to an agent

This agent uses `soccer_workflow` to answer fixture questions. The instructions tell it to compute the date, pass it in YYYY-MM-DD format, and return team names, match times, and dates.

<CodeBlock language="python" code={`from datetime import datetime
from timbal.core import Agent

soccer_agent = Agent(
    name="soccer-agent",
    description="A premier league soccer specialist",
    system_prompt=f"""You are a premier league soccer specialist. Use the soccerWorkflow to fetch match data.

    Calculate dates from {datetime.now().strftime("%Y-%m-%d")} and pass to workflow in YYYY-MM-DD format.

    Only show team names, match times, and dates.""",
    model="openai/gpt-4.1",
    workflows=[soccer_workflow]
)`}/>

## Example usage

Use the agent directly by calling it with a prompt message.

<CodeBlock language="python" code={`import asyncio

async def main():
    response = await soccer_agent(prompt="What matches are being played this weekend?").collect()
    response_text = response.output.content[0].text
    print(response_text)

if __name__ == "__main__":
    asyncio.run(main())`}/>

<div>
  <Link className={styles.card} href="https://github.com/your-repo/design-tools" target="_blank" style={{display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '1.2rem', flexWrap: 'nowrap'}}>
    <span className={styles.icon} style={{flexShrink: 0}}><svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></span>
    <span style={{flexShrink: 0}}>Adding a workflow</span>
    <span style={{flexShrink: 0, marginLeft: 'auto', fontSize: '1.5rem'}}>↗</span>
  </Link>
</div> 