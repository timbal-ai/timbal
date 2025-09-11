---
title: Perplexity
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';

# Perplexity Integration

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Perform web searches and leverage Perplexity's advanced LLM models for real-time information retrieval.
</h2>

---

Timbal integrates with Perplexity to provide advanced search and LLM capabilities. 

This integration allows you to perform web searches and leverage Perplexity's models directly within your Timbal workflows.

## Prerequisites

Before using the Perplexity integration, you'll need:

1. A Perplexity API key – [Get your API key here](https://www.perplexity.ai/account/api/keys)
2. Store your obtained API key in an environment variable named `PERPLEXITY_API_KEY` to facilitate its use by the tools.


## <span style={{color: 'var(--timbal-purple)'}}><strong>Search</strong></span>

### Description
The **search** step allows you to perform a web search or interact with Perplexity's LLMs using a variety of parameters for customization.

### Example
<CodeBlock language="python" code ={`from timbal.steps.perplexity import search

result = await search(query="What is the capital of France?")`}/>

### Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `query` | `str` | Query to search for. | Yes |
| `model` | `str` | Model to use. Default: `sonar`. | No |
| `system_prompt` | `str` | System prompt to guide the LLM's behavior and role. | No |
| `temperature` | `float` | Amount of randomness in the response. Ranges from 0 to 2. Default: 0.2 | No |
| `top_p` | `float` | Nucleus sampling threshold. Ranges from 0 to 1. Default: 0.9 | No |
| `search_domain_filter` | `list/None` | Limit citations to specific domains. Only available in certain tiers. | No |
| `return_images` | `bool` | Whether to return images in the response. Only available in certain tiers. | No |
| `return_related_questions` | `bool` | Whether to return related questions. Only available in certain tiers. | No |
| `search_recency_filter` | `Literal['month', 'week', 'day', 'hour']` | Restrict search results to a recent time interval. | No |
| `top_k` | `int` | Number of tokens to keep for top-k filtering. 0 disables. | No |
| `stream` | `bool` | Whether to stream the response. | No |
| `presence_penalty` | `float` | Penalize new tokens based on presence. Ranges from -2.0 to 2.0. | No |
| `frequency_penalty` | `float` | Penalize new tokens based on frequency. >1.0 discourages repetition. | No |
| `response_format` | `dict/None` | Enable structured outputs with a JSON or Regex schema. | No |

### Agent Integration Example

<CodeBlock language="python" code ={`from timbal.steps.perplexity import search
from timbal import Agent

agent = Agent(
    tools=[search]
)

response = await agent.complete(prompt="What are the latest AI trends?")`}/>

## Notes
- For more advanced usage, see the [Perplexity API documentation](https://docs.perplexity.ai/api-reference/chat-completions).