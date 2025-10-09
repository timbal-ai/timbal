"""
INTERNAL USE ONLY

This module is intended for internal use and will be subject to frequent changes
as LLM providers constantly update their APIs and add new features. The external
APIs (Runnables, Agents, Workflows) will remain stable, but this module will
evolve to match provider changes.

Do not rely on this module's interface in external code.
"""
import os
from typing import Any, Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import Field

from ..errors import APIKeyNotFoundError
from ..types.message import Message
from ..utils import resolve_default
from .runnable import Runnable

OPENAI_API = os.getenv("TIMBAL_OPENAI_API", "responses")

# Model type with provider prefixes
Model = Literal[
    # OpenAI models
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/o1",
    "openai/o1-pro",
    "openai/o3",
    "openai/o3-mini",
    "openai/o3-pro",
    "openai/o3-deep-research",
    "openai/o4-mini",
    "openai/o4-mini-deep-research",
    "openai/gpt-4o-audio-preview",
    "openai/gpt-4o-mini-audio-preview",
    # Anthropic models
    "anthropic/claude-opus-4-1",
    "anthropic/claude-opus-4-0",
    "anthropic/claude-sonnet-4-0",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-3-7-sonnet-latest",
    "anthropic/claude-3-5-haiku-latest",
    # TogetherAI models
    "togetherai/mistralai/Mistral-Small-24B-Instruct-2501",
    "togetherai/mistralai/Mistral-7B-Instruct-v0.3",
    "togetherai/Qwen/Qwen2.5-VL-72B-Instruct",
    "togetherai/Qwen/Qwen2.5-Coder-32B-Instruct",
    "togetherai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "togetherai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "togetherai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "togetherai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "togetherai/deepseek-ai/DeepSeek-V3.1",
    "togetherai/deepseek-ai/DeepSeek-R1",
    # Gemini models
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-preview-native-audio-dialog",
    "google/gemini-2.5-flash-exp-native-audio-thinking-dialog",
    "google/gemini-2.5-flash-image-preview",
    "google/gemini-2.5-flash-preview-tts",
    "google/gemini-2.5-pro-preview-tts",
    "google/gemini-2.0-flash-preview-image-generation",
    "google/gemini-2.0-flash",
    "google/gemini-2.0-flash-lite",
]


# TODO Add more parameters
async def _llm_router(
    model: Model | str = Field(
        ...,
        description="Provider/Name of the LLM model to use.",
    ),
    system_prompt: str | None = Field(
        None,
        description="System prompt to guide the LLM's behavior and role.",
    ),
    messages: list[Message] = Field(
        default_factory=list,
        description="Chat history containing user and LLM messages.",
    ),
    tools: list[Runnable] | None = Field(
        default_factory=list,
        description="List of tools/functions the LLM can call.",
    ),
    max_tokens: int | None = Field(
        None,
        description="Maximum number of tokens to generate.",
    ),
    thinking: Any = Field(
        None,
        description=(
            "Thinking configuration for the LLM. Provider-specific. "
            "For OpenAI models, this should be a dictionary with 'effort' and 'summary' keys. "
            "For Anthropic models, this should be a dictionary with 'budget_tokens' key."
        ),
    ),
) -> Message: # type: ignore
    """
    Internal LLM router function.

    WARNING: This function is for internal use only and may change frequently
    as LLM providers update their APIs. Use the stable Agent/Workflow APIs instead.
    """
    model = resolve_default("model", model)
    system_prompt = resolve_default("system_prompt", system_prompt)
    messages = resolve_default("messages", messages)
    tools = resolve_default("tools", tools)
    max_tokens = resolve_default("max_tokens", max_tokens)
    thinking = resolve_default("thinking", thinking)

    if "/" not in model:
        raise ValueError("Model must be in format 'provider/model_name'")
    
    provider, model_name = model.split("/", 1)
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("OPENAI_API_KEY not found.")
        client = AsyncOpenAI(api_key=api_key)

    elif provider == "anthropic":
        if not max_tokens:
            raise ValueError("'max_tokens' is required for claude models.")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("ANTHROPIC_API_KEY not found.")
        client = AsyncAnthropic(api_key=api_key)

    elif provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("GEMINI_API_KEY not found.")
        client = AsyncOpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    elif provider == "togetherai":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("TOGETHER_API_KEY not found.")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    if provider == "anthropic":
        anthropic_messages = []
        for message in messages:
            anthropic_message = message.to_anthropic_input()
            anthropic_messages.append(anthropic_message)

        anthropic_kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if system_prompt:
            anthropic_kwargs["system"] = system_prompt

        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(tool.anthropic_schema)
        if anthropic_tools:
            anthropic_kwargs["tools"] = anthropic_tools

        if thinking:
            # {"type": "enabled", "budget_tokens": int}
            # budget_tokens must be >= 1024
            anthropic_kwargs["thinking"] = thinking

        res = await client.messages.create(**anthropic_kwargs)

        async for res_chunk in res:
            yield res_chunk

    elif provider == "openai" and OPENAI_API == "responses":
        responses_kwargs = {
            "model": model_name,
            "stream": True,
            "store": False,
            "include": ["web_search_call.action.sources"]
        }

        if system_prompt:
            responses_kwargs["instructions"] = system_prompt

        responses_kwargs["input"] = sum([message.to_openai_responses_input() for message in messages], [])

        responses_tools = [tool.openai_responses_schema for tool in tools]
        if responses_tools:
            responses_kwargs["tools"] = responses_tools
            responses_kwargs["parallel_tool_calls"] = True

        if max_tokens:
            responses_kwargs["max_output_tokens"] = max_tokens

        if thinking:
            # {"effort": "", "summary": ""}
            responses_kwargs["reasoning"] = thinking

        res = await client.responses.create(**responses_kwargs)

        async for res_chunk in res:
            yield res_chunk

    # Try with OpenAI Chat Completions compatible providers
    else:
        chat_completions_messages = []
        if system_prompt:
            chat_completions_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            chat_completions_message = message.to_openai_chat_completions_input()
            chat_completions_messages.append(chat_completions_message)

        chat_completions_kwargs = {
            "model": model_name,
            "messages": chat_completions_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        chat_completions_tools = []
        for tool in tools:
            chat_completions_tools.append(tool.openai_chat_completions_schema)
        if chat_completions_tools:
            chat_completions_kwargs["tools"] = chat_completions_tools

        if max_tokens:
            chat_completions_kwargs["max_completion_tokens"] = max_tokens

        res = await client.chat.completions.create(**chat_completions_kwargs)

        async for res_chunk in res:
            yield res_chunk
