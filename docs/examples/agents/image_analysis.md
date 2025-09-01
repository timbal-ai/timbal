---
title: Image Analysis Agent
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

When building AI applications, you often need agents that can analyze and understand visual content. Timbal lets you create image analysis agents that can identify objects, describe scenes, and answer questions about visual content using the `tools` parameter.

## Prerequisites

This example uses the `openai` model. Make sure to add `OPENAI_API_KEY` to your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## Creating an agent

Create a simple agent that analyzes images to identify objects, describe scenes, and answer questions about visual content.

<CodeBlock language="python" code={`from timbal.core import Agent

image_analysis_agent = Agent(
    name="image-analysis",
    description="Analyzes images to identify objects and describe scenes",
    system_prompt="""You can view an image and identify objects, describe scenes, and answer questions about the content.
    You can also determine species of animals and describe locations in the image.""",
    model="openai/gpt-4o"
)`}/>

## Creating a function

This function provides a sample image URL for testing the agent's image analysis capabilities.

<CodeBlock language="python" code={`import random

def get_sample_image() -> str:
    """Get a sample image URL for testing image analysis."""
    sample_images = [
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",  # Forest
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",  # Mountains
        "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=800",  # Bird
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800",    # Cat
    ]

    return random.choice(sample_images)`}/>

## Example usage

Use the agent directly by calling it with a prompt message that includes an image.

<CodeBlock language="python" code={`from timbal.types.file import File

async def main():
    image_url = get_sample_image()
    
    # Create a message with image and text for the agent
    prompt = [File.validate(image_url), "Analyze this image and identify the main objects or subjects. If there are animals, provide their common name and scientific name. Also describe the location or setting in one or two short sentences."]
    
    # Call the image analysis agent directly
    response = await image_analysis_agent(prompt=prompt).collect()
    
    # Extract the text response
    response_text = response.output.content[0].text
    print(response_text)

if __name__ == "__main__":
    asyncio.run(main())`}/>

<div>
  <Link className={styles.card} href="https://github.com/your-repo/design-tools" target="_blank" style={{display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '1.2rem', flexWrap: 'nowrap'}}>
    <span className={styles.icon}><svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></span>
    <span style={{flexShrink: 0}}>Image Analysis</span>
    <span style={{flexShrink: 0, marginLeft: 'auto', fontSize: '1.5rem'}}>↗</span>
  </Link>
</div> 