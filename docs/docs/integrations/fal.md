---
title: Fal
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# Fal Integration

Timbal integrates with Fal.ai to enable AI-powered image generation. This integration allows you to generate high-quality images from text prompts using various models.

## Prerequisites

Before using the Fal integration, you'll need:

1. A [Fal.ai account](https://fal.ai)
2. An API key from your Fal.ai dashboard
3. Set the `FAL_KEY` environment variable with your API key

## Authentication Setup

Set your Fal.ai API key as an environment variable: `FAL_KEY="your-api-key-here"`

## Installation

Install the requirements by doing:
<CodeBlock language="bash" code ={`uv add timbal[steps-fal]`}/>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Generate Images</strong></span>

### Description
The **gen_images** step allows you to generate images from text prompts using Fal.ai's models.

### Example
<CodeBlock language="python" code ={`from timbal.steps.fal.text_to_image import gen_images

# Generate a single image
images = await gen_images(
    prompt="a green ferrari",
    model="fal-ai/flux-pro/v1.1-ultra"
)

# Returns a File object with the image URL
image = images[0]`}/>

### Parameters

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>prompt</code></td>
      <td><code>str</code></td>
      <td>The text prompt to generate an image from</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>model</code></td>
      <td><code>str</code></td>
      <td>The model to use for image generation. Default: "fal-ai/flux-pro/v1.1-ultra"</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## Agent Integration Example

<CodeBlock language="python" code ={`from timbal.steps.fal.text_to_image import gen_images
from timbal import Agent

agent = Agent(
    tools=[gen_images]
)

response = await agent.complete(
    prompt={
        "prompt": "a futuristic cityscape at night",
        "model": "fal-ai/flux-pro/v1.1-ultra"
    }
)`}/>

## Notes
- Make sure your Fal.ai API key is properly set in the environment variables
- The default model (fal-ai/flux-pro/v1.1-ultra) costs approximately $0.06 per image
- The function returns a list of File objects containing the generated image URLs
- Images are generated asynchronously and the function will wait for completion
- For more advanced usage and available models, see the [Fal.ai documentation](https://docs.fal.ai)