import os
from typing import Any, Literal

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ...errors import APIKeyNotFoundError
from ...types.field import Field, resolve_default
from ...types.message import Message
from ..runnable import Runnable

# Model type with provider prefixes
Model = Literal[
    # OpenAI models
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini", 
    "openai/o1",
    "openai/o1-mini",
    "openai/o3-mini",
    "openai/gpt-5",
    # Anthropic models
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-opus-4-1-20250805",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku-20241022", 
    "anthropic/claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-haiku-20240307",
    # TogetherAI models
    "togetherai/deepseek-ai/DeepSeek-R1",
    "togetherai/deepseek-ai/DeepSeek-V3",
    "togetherai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "togetherai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "togetherai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "togetherai/meta-llama/Meta-Llama-3.2-3B-Instruct-Turbo",
    "togetherai/Qwen/Qwen2.5-Coder-32B-Instruct",
    "togetherai/Qwen/Qwen2-VL-72B-Instruct",
    "togetherai/mistralai/Mistral-Small-24B-Instruct-2501",
    "togetherai/mistralai/Mistral-7B-Instruct-v0.3",
    "togetherai/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "togetherai/meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    # Gemini models
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-live-2.5-flash-preview",
    "gemini/gemini-2.5-flash-preview-native-audio-dialog",
    "gemini/gemini-2.5-flash-exp-native-audio-thinking-dialog",
    "gemini/gemini-2.5-flash-preview-tts",
    "gemini/gemini-2.5-pro-preview-tts",
    "gemini/gemini-2.0-flash-preview-image-generation",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite-preview-02-05",
]


async def handler(
    model: str | Model = Field(
        default="openai/gpt-4.1-mini", 
        description="Provider/Name of the LLM model to use.",
    ),
    system_prompt: str | None = Field(
        default=None, 
        description="System prompt to guide the LLM's behavior and role.",
    ),
    messages: list[Message] = Field(
        description="Chat history containing user and LLM messages.",
    ),
    tools: list[Runnable] = Field(
        default=[], 
        description="List of tools/functions the LLM can call.",
    ),
    tool_choice: dict[str, Any] | str = Field(
        default={"type": "auto"},
        description="How the model should use the provided tools"
    ),
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate.",
    ),
    temperature: float = Field(
        default=1,
        description=(
            "Sampling temperature (0-2 except for Anthropic which is 0-1). "
            "Higher values increase randomness, lower values increase determinism."
        )
    ),
    frequency_penalty: float = Field(
        default=0,
        description=(
            "Positive values penalize token frequency to reduce repetition. "
            "Ranges from -2.0 to 2.0."
        )
    ),
    presence_penalty: float = Field(
        default=0,
        description=(
            "Positive values penalize tokens based on presence to encourage new topics. "
            "Ranges from -2.0 to 2.0."
        )
    ),
    top_p: float = Field(
        default=1,
        description=(
            "Nucleus sampling parameter. Only tokens with cumulative probability "
            "mass up to top_p are considered."
        )
    ),
    top_k: int = Field(
        default=None,
        description="Only sample from the top K options for each subsequent token."
    ),
    logprobs: bool = Field(
        default=False,
        description="Whether to return logprobs with the returned text."
    ),
    top_logprobs: int = Field(
        default=None,
        description=(
            "Return log probabilities of the top N tokens (0-20). "
            "Requires logprobs=true."
        )
    ),
    seed: int = Field(
        default=None,
        description=(
            "Beta feature for deterministic sampling. Same seed and parameters "
            "should return same result."
        )
    ),
    stop: str | list[str] = Field(
        default=None,
        description="Where the model will stop generating."
    ),
    parallel_tool_calls: bool = Field(
        default=True,
        description="Whether to execute tool calls in parallel or sequentially."
    ),
    json_schema: dict = Field(
        default=None, 
        description="The JSON schema that the model MUST adhere to.",
    ),
) -> Message: # type: ignore
    """Route requests to appropriate LLM providers based on model name prefix.

    This gateway function handles routing to different LLM providers (OpenAI, Anthropic,
    Gemini, TogetherAI) based on the model name prefix.

    Args:
        model: Name of the LLM model to use.
        system_prompt: Instructions for the LLM to follow.
        messages: Chat history containing user and LLM messages.
        tools: List of available tool functions.
        tool_choice: Specification for tool selection.
        max_tokens: Maximum number of tokens in the response.
        temperature: Sampling temperature.
        frequency_penalty: Penalty for token frequency.
        presence_penalty: Penalty for token presence.
        top_p: Nucleus sampling parameter.
        top_k: Only sample from the top K options for each subsequent token.
        logprobs: Whether to return logprobs with the returned text.
        top_logprobs: Return log probabilities of the top N tokens.
        seed: Deterministic sampling parameter.
        stop: Up to 4 sequences where the model will stop generating.
        parallel_tool_calls: Whether to execute tool calls in parallel.
        json_schema: JSON schema for structured output.

    Yields:
        Any: Response chunks from the LLM provider
    """
    model = resolve_default("model", model)
    system_prompt = resolve_default("system_prompt", system_prompt)
    messages = resolve_default("messages", messages)
    tools = resolve_default("tools", tools)
    tool_choice = resolve_default("tool_choice", tool_choice)
    max_tokens = resolve_default("max_tokens", max_tokens)
    temperature = resolve_default("temperature", temperature)
    frequency_penalty = resolve_default("frequency_penalty", frequency_penalty)
    presence_penalty = resolve_default("presence_penalty", presence_penalty)
    top_p = resolve_default("top_p", top_p)
    top_k = resolve_default("top_k", top_k)
    logprobs = resolve_default("logprobs", logprobs)
    top_logprobs = resolve_default("top_logprobs", top_logprobs)
    seed = resolve_default("seed", seed)
    stop = resolve_default("stop", stop)
    parallel_tool_calls = resolve_default("parallel_tool_calls", parallel_tool_calls)
    json_schema = resolve_default("json_schema", json_schema)

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

    if provider in ["openai", "gemini", "togetherai"]:
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            openai_message = await message.to_openai_input(model=model_name)
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

        if json_schema:
            openai_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        res = await client.chat.completions.create(
            **openai_kwargs,
        )

        async for res_chunk in res:
            yield res_chunk

    elif provider == "anthropic":
        anthropic_messages = []
        for message in messages:
            anthropic_message = await message.to_anthropic_input(model=model_name)
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
            anthropic_tools.append(tool.to_anthropic_tool())
        if anthropic_tools:
            anthropic_kwargs["tools"] = anthropic_tools

        if json_schema:
            # TODO Anthropic doesn't have a direct json schema param... we could implement this with tool use.
            raise NotImplementedError("JSON schema validation is not supported for claude models.")

        res = await client.messages.create(
            **anthropic_kwargs,
        )

        async for res_chunk in res:
            yield res_chunk
