import os
from typing import Any, Literal

import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ...types import Field, Message, Tool

logger = structlog.get_logger("timbal.steps.llms.gemini_llm")

load_dotenv()

async def handler(
    # This is not actually used inside the function. It's a hack to have it as an argument
    # so we can use it when mapping data in the flow.
    prompt: Message = Field(default=None, description="Message to send to the LLM."), # noqa: ARG001
    system_prompt: str = Field(default=None, description="System prompt to guide the LLM's behavior and role."),
    memory: list[Message] = Field(description="A list containing the conversation history between the user and the LLM"),
    model: Literal["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "text-embedding-004"] | str = Field(default="gemini-2.0-flash-lite-preview-02-05", description="Name of the LLM model to use"),
    tools: list[Tool | dict] = Field(default=None, description="List of tools/functions that the LLM can call"),
    tool_choice: dict[str, Any] | str = Field(default = {"type": "auto"}, description="How the model should use the provided tools"),
    max_tokens: int = Field(default=None, description="The maximum number of tokens in the response"),
    presence_penalty: float = Field(default=0, description="Penalizes token presence on a scale of -2.0 to 2.0. Higher values encourage new topics"),
    seed: int = Field(default=None, description="Seed for deterministic sampling (Beta feature)"),
    stop: str | list[str] = Field(default=None, description="Up to 4 sequences where the model will stop generating further tokens."),
    temperature: float = Field(default=1, description="Amount of randomness injected into the response. Ranges from 0.0 to 2.0."),
    top_p: float = Field(default=1, description="Nucleus sampling parameter. Ranges from 0.0 to 1.0."),
    json_schema: dict = Field(default=None, description="The JSON input schema that the model must output"),
) -> Any:
    """This handler manages interactions with Gemini's API, processing requests with
    appropriate parameters and returning streaming responses.

    Args:
        prompt: Hack to pass an input to the LLM.
        memory: Chat history containing user and LLM messages.
        system_prompt: Instructions for the LLM to follow.
        model: Name of the Gemini model to use.
        tools: List of tools/functions the LLM can call.
        tool_choice: How the model should use the provided tools.
        max_tokens: Maximum number of tokens in the response. Defaults to 1024.
        presence_penalty: Value between -2.0 and 2.0 to penalize token presence. Defaults to 0.
        seed: Seed for deterministic sampling (Beta feature).
        stop: Sequences where token generation should stop.
        temperature: Sampling temperature between 0 and 2. Defaults to 1.
        top_p: Nucleus sampling probability threshold. Defaults to 1.
        json_schema: JSON schema for structured output.

    Returns:
        Any: The streaming response from the Gemini API.
    """
    client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Get default values for parameters
    system_prompt = system_prompt.default if hasattr(system_prompt, 'default') else system_prompt
    model = model.default if hasattr(model, 'default') else model
    tools = tools.default if hasattr(tools, 'default') else tools
    tool_choice = tool_choice.default if hasattr(tool_choice, 'default') else tool_choice
    max_tokens = max_tokens.default if hasattr(max_tokens, 'default') else max_tokens
    presence_penalty = presence_penalty.default if hasattr(presence_penalty, 'default') else presence_penalty
    seed = seed.default if hasattr(seed, 'default') else seed
    stop = stop.default if hasattr(stop, 'default') else stop
    temperature = temperature.default if hasattr(temperature, 'default') else temperature
    top_p = top_p.default if hasattr(top_p, 'default') else top_p
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
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    # Optional parameters
    if stop:
        kwargs["stop"] = stop
    if seed:
        kwargs["seed"] = seed
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    if presence_penalty:
        kwargs["presence_penalty"] = presence_penalty

    # Change parameters if provided to Gemini format
    if tools:
        logger.warning("This model does not support multiple tool calls. Only one tool can be called at a time.", model=model)
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