---
title: Adding a tool
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

When building AI agents, you often need to extend their capabilities with external data or functionality. Timbal lets you pass tools to an agent using the `tools` parameter. Tools give agents a way to call specific functions, such as fetching data or performing calculations, to help answer a user's query.

## Prerequisites

This example uses the `openai` model. Make sure to add `OPENAI_API_KEY` to your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## Creating a tool

This tool provides historical weather data for London, returning arrays of daily temperature, precipitation, wind speed, snowfall, and weather conditions from January 1st of the current year up to today. This structure makes it easy for agents to access and reason about recent weather trends.

<CodeBlock language="python" code ={`import asyncio
import json
import urllib.request
from datetime import datetime
from typing import Any

from timbal.core import Tool


async def london_weather_tool() -> dict[str, Any]:
  """Returns year-to-date historical weather data for London."""
  start_date = f"{datetime.now().year}-01-01"
  end_date = datetime.now().strftime("%Y-%m-%d")
  
  url = f"https://archive-api.open-meteo.com/v1/archive?latitude=51.5072&longitude=-0.1276&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,snowfall_sum&timezone=auto"
  
  response = await asyncio.get_event_loop().run_in_executor(None, lambda: urllib.request.urlopen(url))
  data = await asyncio.get_event_loop().run_in_executor(None, lambda: response.read())
  weather_data = json.loads(data.decode())
  
  daily = weather_data["daily"]
  
  return {
      "date": daily["time"],
      "temp_max": daily["temperature_2m_max"],
      "temp_min": daily["temperature_2m_min"],
      "rainfall": daily["precipitation_sum"],
      "windspeed": daily["windspeed_10m_max"],
      "snowfall": daily["snowfall_sum"]
  }


# Create the Tool instance
london_weather_tool_instance = Tool(
  name="london_weather_tool",
  description="Returns year-to-date historical weather data for London",
  handler=london_weather_tool
)`}/>

## Adding a tool to an agent

This agent uses the `london_weather_tool_instance` to answer questions about historical weather in London. It has clear instructions that guide it to use the tool for every query and limit its responses to data available for the current calendar year.

<CodeBlock language="python" code={`from timbal.core import Agent

london_weather_agent = Agent(
    name="london-weather-agent",
    system_prompt="""You are a helpful assistant with access to historical weather data for London.
    - The data is limited to the current calendar year, from January 1st up to today's date.
    - Use the provided tool (londonWeatherTool) to retrieve relevant data.
    - Answer the user's question using that data.
    - Keep responses concise, factual, and informative.
    - If the question cannot be answered with available data, say so clearly.""",
    model="openai/gpt-4o",
    tools=[london_weather_tool_instance]
)`}/>

## Example usage

Use the agent directly by calling it with a prompt message.

<CodeBlock language="python" code={`async def main():
    # Call the agent directly
    response = await london_weather_agent(prompt="How many times has it rained this year?").collect()
    
    # Extract the text response
    response_text = response.output.content[0].text
    print(response_text)

if __name__ == "__main__":
    asyncio.run(main())
`}/>


<div>
  <Link className={styles.card} href="https://github.com/your-repo/design-tools" target="_blank" style={{display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '1.2rem', flexWrap: 'nowrap'}}>
    <span className={styles.icon}><svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></span>
    <span style={{flexShrink: 0}}>Adding a tool</span>
    <span style={{flexShrink: 0, marginLeft: 'auto', fontSize: '1.5rem'}}>↗</span>
  </Link>
</div> 