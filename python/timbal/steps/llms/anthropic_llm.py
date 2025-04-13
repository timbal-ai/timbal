import os
from typing import Any, Literal

import structlog
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from ...types import Field, Message, Tool

logger = structlog.get_logger("timbal.steps.llms.anthropic_llm")

load_dotenv()

async def handler(
    # This is not actually used inside the function. It's a hack to have it as an argument
    # so we can use it when mapping data in the flow.
    prompt: Message = Field(default=None, description="Message to send to the LLM."), # noqa: ARG001
    system_prompt: str = Field(default=None, description="System prompt to guide the LLM's behavior and role."),
    memory: list[Message] = Field(description="A list containing the conversation history between the user and the LLM"),
    model: Literal["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"] | str = Field(default="claude-3-haiku-20240307", description="Name of the LLM model to use"),
    tools: list[Tool | dict] = Field(default=None, description="List of tools/functions that the LLM can call"),
    tool_choice: dict[str, Any] | str = Field(default = {"type": "auto"}, description="How the model should use the provided tools"),
    max_tokens: int = Field(default=1024, description="The maximum number of tokens in the response"),
    stop: str | list[str] = Field(default=None, description="Where the model will stop generating further tokens"),
    temperature: float = Field(default=1, description="Amount of randomness injected into the response. Ranges from 0.0 to 1.0"),
    top_k: int = Field(default=None, description="Only sample from the top K options for each subsequent token. Must be greater than 0"),
    top_p: float = Field(default=None, description="Nucleus sampling parameter. Ranges from 0.0 to 1.0"),
    json_schema: dict = Field(default=None, description="The JSON input schema that the model must output"),
) -> Any:
    """This handler manages interactions with Anthropic's API, processing requests with 
    appropriate parameters and returning streaming responses.

    Args:
        prompt: Hack to pass an input to the LLM.
        memory: Chat history containing user and LLM messages.
        system_prompt: Instructions for the LLM to follow.
        model: Name of the Anthropic model to use.
        tools: List of tools/functions the LLM can call.
        tool_choice: How the model should use the provided tools
        max_tokens: Maximum number of tokens in the response. Defaults to 1024.
        stop: Sequences where token generation should stop.
        temperature: Sampling temperature between 0 and 1. Defaults to 1.
        top_k: Only sample from the top K options for each subsequent token.
        top_p: Nucleus sampling parameter. Ranges from 0.0 to 1.0.
        json_schema: JSON input schema that the model must output.

    Returns:
        Any: The streaming response from the Anthropic API.
    """
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Get default values for parameters
    system_prompt = system_prompt.default if hasattr(system_prompt, 'default') else system_prompt
    model = model.default if hasattr(model, 'default') else model
    tools = tools.default if hasattr(tools, 'default') else tools
    tool_choice = tool_choice.default if hasattr(tool_choice, 'default') else tool_choice
    max_tokens = max_tokens.default if hasattr(max_tokens, 'default') else max_tokens
    stop = stop.default if hasattr(stop, 'default') else stop
    temperature = temperature.default if hasattr(temperature, 'default') else temperature
    top_k = top_k.default if hasattr(top_k, 'default') else top_k
    top_p = top_p.default if hasattr(top_p, 'default') else top_p
    json_schema = json_schema.default if hasattr(json_schema, 'default') else json_schema

    messages = []
    for msg in memory:
        anthropic_msg = await msg.to_anthropic_input(model=model)
        messages.append(anthropic_msg)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }

    # Optional parameters
    if system_prompt:
        kwargs["system"] = system_prompt
    if stop:
        kwargs["stop_sequences"] = stop
    if top_k:
        kwargs["top_k"] = top_k
    if top_p:
        kwargs["top_p"] = top_p

    # Change parameters to Anthropic format
    if isinstance(tool_choice, str):
        tool_choice = {"type": tool_choice}

    if isinstance(stop, list):
        kwargs["stop_sequences"] = stop[0]

    if tools:
        if "haiku" in model:
            logger.warning("This model does not support multiple tool calls. Only one tool can be called at a time.", model=model)
        
        for tool in tools:
            if isinstance(tool, Tool):
                tool = tool.to_anthropic()
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    if json_schema:
        kwargs["tools"] = [
            {
                "name": "print_json",
                "description": "Print in a JSON format",
                "input_schema": {
                    "type": "object",
                    "properties": json_schema
                }
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": "print_json"}

    response = await client.messages.create(**kwargs)

    return response