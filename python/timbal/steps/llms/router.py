import os

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ...errors import APIKeyNotFoundError
from ...core.base import BaseStep
from ...types import Field, Message


async def llm_router(
    model: str = Field(
        default="gpt-4o-mini", 
        description="Name of the LLM model to use.",
    ),
    # TODO Enable forcing a specific provider.
    system_prompt: str | None = Field(
        default=None, 
        description="System prompt to guide the LLM's behavior and role.",
    ),
    messages: list[Message] = Field(
        description="Chat history containing user and LLM messages.",
    ),
    tools: list[BaseStep] = Field(
        default=[], 
        description="List of tools/functions the LLM can call.",
    ),
    stream: bool = Field(
        default=False, 
        description="Whether to stream the response from the LLM.",
    ),
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate.",
    ),
    # TODO Add all the rest of parameters.
) -> Message: # type: ignore
    
    # Enable calling this router without pydantic model_validate()
    model = model.default if hasattr(model, "default") else model
    system_prompt = system_prompt.default if hasattr(system_prompt, "default") else system_prompt
    messages = messages.default if hasattr(messages, "default") else messages
    tools = tools.default if hasattr(tools, "default") else tools
    stream = stream.default if hasattr(stream, "default") else stream
    max_tokens = max_tokens.default if hasattr(max_tokens, "default") else max_tokens

    if model.startswith("claude"):

        if not max_tokens:
            raise ValueError("'max_tokens' is required for claude models.")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("ANTHROPIC_API_KEY not found.")
        client = AsyncAnthropic(api_key=api_key)

        anthropic_messages = []
        for message in messages:
            anthropic_message = await message.to_anthropic_input(model=model)
            anthropic_messages.append(anthropic_message)

        anthropic_kwargs = {}

        if system_prompt:
            anthropic_kwargs["system"] = system_prompt

        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(tool.to_anthropic_tool())
        if anthropic_tools:
            anthropic_kwargs["tools"] = anthropic_tools

        if stream:
            anthropic_kwargs["stream"] = True

        res = await client.messages.create(
            model=model,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            **anthropic_kwargs,
        )

        if stream:
            return (res_chunk async for res_chunk in res)
        else:
            return res

    # Grouppped openai sdk compatible (provider, model).
    else:

        if model.startswith(("gpt", "o1", "o3")):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise APIKeyNotFoundError("OPENAI_API_KEY not found.")
            client = AsyncOpenAI(api_key=api_key)
        elif model.startswith("gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise APIKeyNotFoundError("GEMINI_API_KEY not found.")
            client = AsyncOpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            # TODO Multiprovider. 
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise APIKeyNotFoundError("TOGETHER_API_KEY not found.")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
            )

        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            openai_message = await message.to_openai_input(model=model)
            openai_messages.append(openai_message)

        # Collect all non required params as kwargs.
        openai_kwargs = {}

        openai_tools = []
        for tool in tools:
            openai_tools.append(tool.to_openai_tool())
        if openai_tools:
            openai_kwargs["tools"] = openai_tools

        if stream:
            openai_kwargs["stream"] = True
            openai_kwargs["stream_options"] = {"include_usage": True}
        else: 
            openai_kwargs["stream"] = False

        if max_tokens:
            openai_kwargs["max_completion_tokens"] = max_tokens

        res = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            **openai_kwargs,
        )

        if stream:
            return (res_chunk async for res_chunk in res)
        else:
            return res

