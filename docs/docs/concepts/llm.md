---
title: LLMs
sidebar: 'docsSidebar'
---

# LLMs

A comprehensive guide to configuring and using Large Language Models (LLMs) in your Timbal projects.

## What are LLMs?

LLMs are the core of Timbal. They are the ones that will be used to create agents with tools. They enable agents to understand context, make decisions, and generate human-like responses. Here's what you need to know:

As you have seen in [Flows](./flow/index.md), LLMs behaves as steps that there function is to call an LLM provider.

So the way to use an LLM is to create a flow with an LLM step.

```python
from timbal import Flow

flow = (Flow()
    .add_llm()
)
```

### Setting Up

For Timbal to use an LLM, you need to set up the API key of the provider do you want the model to be used from. (e.g. OpenAI, Anthropic, TogetherAI or Gemini)

:::warning
Never commit API keys to version control. Use environment files (.env) or your system's secret management.
:::

### Attributes

:::tip
You don't have to worry about the options of the LLM. Timbal will take care of it for you.
:::

By specifying the `model` parameter, the kwargs parameters will be in function of the LLM provider.

| Attribute | Parameter | Type | Description | Provider Support |
| :-------- | :-------- | :--- | :---------- | :--------------- |
| **Prompt** | `prompt` | `str` | The first input to send to the LLM. | All providers |
| **System Prompt** | `system_prompt` | `str` | System prompt to guide the LLM's behavior and role. | All providers |
| **Model** | `model` | `str` | Name of the LLM model to use. | All providers |
| **Tools** | `tools` | `list[Tool \| dict]` | List of tools/functions the LLM can call. | All providers |
| **Tool Choice** | `tool_choice` | `dict[str, Any] \| str` | How the model should use the provided tools. | All providers |
| **Max Tokens** | `max_tokens` | `int` | The maximum number of tokens in the response. | All providers |
| **Temperature** | `temperature` | `float` | Sampling temperature (0-2 except for Anthropic which is 0-1). | All providers |
| **Frequency Penalty** | `frequency_penalty` | `float` | Penalty for token frequency. | OpenAI, TogetherAI |
| **Presence Penalty** | `presence_penalty` | `float` | Penalty for token presence. | OpenAI |
| **Top P** | `top_p` | `float` | Nucleus sampling parameter. | OpenAI, TogetherAI, Gemini|
| **Top K** | `top_k` | `int` | Only sample from the top K options for each token. | Anthropic|
| **Logprobs** | `logprobs` | `bool` | Whether to return logprobs with the returned text. | OpenAI |
| **Top Logprobs** | `top_logprobs` | `int` | Return log probabilities of the top N tokens. | OpenAI, TogetherAI |
| **Seed** | `seed` | `int` | Deterministic sampling parameter. | OpenAI, TogetherAI, Gemini |
| **Stop** | `stop` | `str \| list[str]` | Up to 4 sequences where the model will stop generating. | All providers |
| **Parallel Tool Calls** | `parallel_tool_calls` | `bool` | Whether to execute tool calls in parallel. | OpenAI, TogetherAI |
| **JSON Schema** | `json_schema` | `dict` | JSON schema for structured output. | All providers |

:::info
These are some of the models that could be used:
- OpenAI: gpt-4o, gpt-4o-mini, o1, o3-mini, o1-mini
- Anthropic: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
- TogetherAI: deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.3-70B-Instruct-Turbo, meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo, meta-llama/Llama-3.2-3B-Instruct-Turbo,Qwen/Qwen2.5-Coder-32B-Instruct, Qwen/Qwen2-VL-72B-Instruct, mistralai/Mistral-Small-24B-Instruct-2501, mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mixtral-8x22B-Instruct-v0.1, meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo
- Gemini: gemini-2.0-flash-lite-preview-02-05, gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-flash-8b, text-embedding-004
:::