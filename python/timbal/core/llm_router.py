import os
from typing import Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import Field

from ..errors import APIKeyNotFoundError
from ..types.message import Message
from ..utils import resolve_default
from .runnable import Runnable

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
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash-preview-native-audio-dialog",
    "gemini/gemini-2.5-flash-exp-native-audio-thinking-dialog",
    "gemini/gemini-2.5-flash-image-preview",
    "gemini/gemini-2.5-flash-preview-tts",
    "gemini/gemini-2.5-pro-preview-tts",
    "gemini/gemini-2.0-flash-preview-image-generation",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
]


# TODO Add more parameters
async def llm_router(
    model: Model = Field(
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
    tools: list[Runnable] = Field(
        default_factory=list,
        description="List of tools/functions the LLM can call.",
    ),
    max_tokens: int | None = Field(
        None,
        description="Maximum number of tokens to generate.",
    ),
) -> Message: # type: ignore
    """Route LLM requests to appropriate providers based on model prefix.

    Handles automatic provider detection and client initialization for OpenAI, Anthropic,
    Gemini, and TogetherAI models. Converts messages and tools to provider-specific formats
    and streams responses back as async chunks.

    Args:
        model: Provider-prefixed model name (e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet')
        system_prompt: Optional system instructions to guide model behavior
        messages: Conversation history as Message objects
        tools: Available Runnable tools for function calling
        max_tokens: Response length limit (required for Anthropic models)

    Returns:
        Message: Streaming response chunks from the selected provider

    Raises:
        ValueError: If model format is invalid or required parameters are missing
        APIKeyNotFoundError: If required API key environment variable is not set
    """
    model = resolve_default("model", model)
    system_prompt = resolve_default("system_prompt", system_prompt)
    messages = resolve_default("messages", messages)
    tools = resolve_default("tools", tools)
    max_tokens = resolve_default("max_tokens", max_tokens)

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

    elif provider == "gemini":
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

    # TODO Probably we'll move to google ai sdk
    if provider in ["openai", "gemini", "togetherai"]:
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            openai_message = message.to_openai_input()
            openai_messages.append(openai_message)

        openai_kwargs = {
            "model": model_name,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        openai_tools = []
        for tool in tools:
            openai_tools.append(tool.openai_schema)
        if openai_tools:
            openai_kwargs["tools"] = openai_tools

        if max_tokens:
            openai_kwargs["max_completion_tokens"] = max_tokens

        res = await client.chat.completions.create(**openai_kwargs)

        async for res_chunk in res:
            yield res_chunk

    elif provider == "anthropic":
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

        res = await client.messages.create(**anthropic_kwargs)

        async for res_chunk in res:
            yield res_chunk
