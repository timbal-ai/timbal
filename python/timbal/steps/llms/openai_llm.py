import os
from typing import Any, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ...types import Field, Message, Tool

load_dotenv()

async def handler(
    # This is not actually used inside the function. It's a hack to have it as an argument
    # so we can use it when mapping data in the flow.
    prompt: Message = Field(default=None, description="Message to send to the LLM."), # noqa: ARG001
    system_prompt: str = Field(default=None, description="System prompt to guide the LLM's behavior and role."),
    memory: list[Message] = Field(description="A list containing the conversation history between the user and the LLM"),
    model: Literal["gpt-4o", "gpt-4o-mini", "o1", "o3-mini", "o1-mini"] | str = Field(default="gpt-4o", description="Name of the LLM model to use"),
    tools: list[Tool | dict] = Field(default=None, description="List of tools/functions that the LLM can call"),
    tool_choice: dict[str, Any] | str = Field(default = {"type": "auto"}, description="How the model should use the provided tools"),
    max_tokens: int = Field(default=None, description="The maximum number of tokens in the response"),
    frequency_penalty: float = Field(default=0, description="Penalizes token repetition on a scale of -2.0 to 2.0. Higher values reduce repetition"),
    logprobs: bool = Field(default=False, description="Whether to return logprobs with the returned text"),
    top_logprobs: int = Field(default=None, description="Number of most likely tokens (0-20) to return with probabilities. Requires logprobs=True"),
    presence_penalty: float = Field(default=0, description="Penalizes token presence on a scale of -2.0 to 2.0. Higher values encourage new topics"),
    seed: int = Field(default=None, description="Seed for deterministic sampling (Beta feature)"),
    stop: str | list[str] = Field(default=None, description="Up to 4 sequences where the model will stop generating further tokens"),
    temperature: float = Field(default=1, description="Amount of randomness injected into the response. Ranges from 0.0 to 2.0"),
    top_p: float = Field(default=1, description="Nucleus sampling parameter. Ranges from 0.0 to 1.0"),
    parallel_tool_calls: bool = Field(default=True, description="Whether to enable parallel function calling during tool use"),
    json_schema: dict = Field(default=None, description="The JSON input schema that the model must output"),
) -> Any:
    """This handler manages interactions with OpenAI's API, processing requests with 
    appropriate parameters and returning streaming responses.

    Args:
        prompt: Hack to pass an input to the LLM.
        memory: Chat history containing user and LLM messages.
        system_prompt: Instructions for the LLM to follow.
        model: Name of the OpenAI model to use.
        tools: List of tools/functions the LLM can call.
        tool_choice: How the model should use the provided tools.
        max_tokens: Maximum number of tokens in the response. Defaults to 1024.
        frequency_penalty: Value between -2.0 and 2.0 to penalize token frequency. Defaults to 0.
        logprobs: Whether to return log probabilities. Defaults to False.
        top_logprobs: Number of most likely tokens to return (0-20). Requires logprobs=True.
        presence_penalty: Value between -2.0 and 2.0 to penalize token presence. Defaults to 0.
        seed: Seed for deterministic sampling (Beta feature).
        stop: Sequences where token generation should stop.
        temperature: Sampling temperature between 0 and 2. Defaults to 1.
        top_p: Nucleus sampling probability threshold. Defaults to 1.
        parallel_tool_calls: Whether to execute tool calls in parallel. Defaults to True.
        json_schema: JSON schema for structured output (Anthropic models only).

    Returns:
        Any: The streaming response from the OpenAI API.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get default values for parameters
    system_prompt = system_prompt.default if hasattr(system_prompt, 'default') else system_prompt
    model = model.default if hasattr(model, 'default') else model
    tools = tools.default if hasattr(tools, 'default') else tools
    tool_choice = tool_choice.default if hasattr(tool_choice, 'default') else tool_choice
    max_tokens = max_tokens.default if hasattr(max_tokens, 'default') else max_tokens
    frequency_penalty = frequency_penalty.default if hasattr(frequency_penalty, 'default') else frequency_penalty
    logprobs = logprobs.default if hasattr(logprobs, 'default') else logprobs
    top_logprobs = top_logprobs.default if hasattr(top_logprobs, 'default') else top_logprobs
    presence_penalty = presence_penalty.default if hasattr(presence_penalty, 'default') else presence_penalty
    seed = seed.default if hasattr(seed, 'default') else seed
    stop = stop.default if hasattr(stop, 'default') else stop
    temperature = temperature.default if hasattr(temperature, 'default') else temperature
    top_p = top_p.default if hasattr(top_p, 'default') else top_p
    parallel_tool_calls = parallel_tool_calls.default if hasattr(parallel_tool_calls, 'default') else parallel_tool_calls
    json_schema = json_schema.default if hasattr(json_schema, 'default') else json_schema

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in memory:
        openai_msg = await msg.to_openai_input(model=model)
        messages.append(openai_msg)

    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
        "presence_penalty": presence_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "stream_options": {"include_usage": True},
    }

    # Optional parameters
    if stop:
        kwargs["stop"] = stop
    if frequency_penalty:
        kwargs["frequency_penalty"] = frequency_penalty
    if seed:
        kwargs["seed"] = seed
    if logprobs:
        kwargs["logprobs"] = logprobs
    if top_logprobs:
        kwargs["top_logprobs"] = top_logprobs
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    # Format parameters to match OpenAI's API requirements
    if tools:
        tools_openai = []
        for tool in tools:
            if isinstance(tool, Tool):
                tools_openai.append(tool.to_openai())
            else:
                tools_openai.append(tool)
        kwargs["tools"] = tools_openai
        if isinstance(tool_choice, dict):
            if tool_choice["type"] != "function":
                tool_choice = tool_choice["type"]
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls

    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "print_json",
                "schema": {
                    "type": "object",
                    "properties": json_schema
                }
            }
        } 

    response = await client.chat.completions.create(**kwargs)

    return response