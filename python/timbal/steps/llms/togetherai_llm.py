import os
from typing import Any, Literal

import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ...types import Field, Message, Tool

logger = structlog.getLogger("timbal.steps.llms.togetherai_llm")

load_dotenv()

async def handler(
    # This is not actually used inside the function. It's a hack to have it as an argument
    # so we can use it when mapping data in the flow.
    prompt: Message = Field(default=None, description="Message to send to the LLM."), # noqa: ARG001
    system_prompt: str = Field(default=None, description="System prompt to guide the LLM's behavior and role."),
    memory: list[Message] = Field(description="A list containing the conversation history between the user and the LLM."),
    model: Literal[
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    ] | str = Field(default="meta-llama/Llama-3.2-3B-Instruct-Turbo", description="Name of the LLM model to use"),
    tools: list[Tool | dict] = Field(default=None, description="List of tools/functions that the LLM can call"),
    tool_choice: dict[str, Any] | str = Field(default = {"type": "auto"}, description="How the model should use the provided tools"),
    max_tokens: int = Field(default=None, description="Maximum length of response in tokens"),
    frequency_penalty: float = Field(default=0, description="Penalizes token repetition on a scale of -2.0 to 2.0. Higher values reduce repetition"),
    top_logprobs: int = Field(default=None, description="Number of most likely tokens (0-20) to return with probabilities"),
    presence_penalty: float = Field(default=0, description="Penalizes token presence on a scale of -2.0 to 2.0. Higher values encourage new topics"),
    seed: int = Field(default=None, description="Seed for deterministic sampling (Beta feature)"),
    stop: str | list[str] = Field(default=None, description="Up to 4 sequences where the model will stop generating further tokens."),
    temperature: float = Field(default=1, description="Amount of randomness injected into the response. Ranges from 0.0 to 2.0."),
    top_p: float = Field(default=1, description="Nucleus sampling parameter. Ranges from 0.0 to 1.0."),
    parallel_tool_calls: bool = Field(default=True, description="Whether to enable parallel function calling during tool use"),
    json_schema: dict = Field(default=None, description="The JSON input schema that the model must output."),
) -> Any:
    """This handler manages interactions with TogetherAI's API, processing requests with
    appropriate parameters and returning streaming responses.

    Args:
        prompt: Hack to pass an input to the LLM.
        memory: Chat history containing user and LLM messages.
        system_prompt: Instructions for the LLM to follow.
        model: Name of the TogetherAI model to use.
        tools: List of tools/functions the LLM can call.
        tool_choice: How the model should use the provided tools.
        max_tokens: Maximum number of tokens in the response.
        frequency_penalty: Value between -2.0 and 2.0 to penalize token frequency.
        top_logprobs: Number of most likely tokens to return (0-20). Requires logprobs=True.
        presence_penalty: Value between -2.0 and 2.0 to penalize token presence.
        seed: Seed for deterministic sampling (Beta feature).
        stop: Sequences where token generation should stop.
        temperature: Sampling temperature between 0 and 2.
        top_p: Nucleus sampling probability threshold.
        parallel_tool_calls: Whether to call multiple tools in parallel or not.
        json_schema: JSON schema for structured output.

    Returns:
        Any: The streaming response from the TogetherAI API.
    """
    client = AsyncOpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
    
    # Get default values for parameters
    system_prompt = system_prompt.default if hasattr(system_prompt, 'default') else system_prompt
    model = model.default if hasattr(model, 'default') else model
    tools = tools.default if hasattr(tools, 'default') else tools
    tool_choice = tool_choice.default if hasattr(tool_choice, 'default') else tool_choice
    max_tokens = max_tokens.default if hasattr(max_tokens, 'default') else max_tokens
    frequency_penalty = frequency_penalty.default if hasattr(frequency_penalty, 'default') else frequency_penalty
    top_logprobs = top_logprobs.default if hasattr(top_logprobs, 'default') else top_logprobs
    presence_penalty = presence_penalty.default if hasattr(presence_penalty, 'default') else presence_penalty
    seed = seed.default if hasattr(seed, 'default') else seed
    stop = stop.default if hasattr(stop, 'default') else stop
    temperature = temperature.default if hasattr(temperature, 'default') else temperature
    top_p = top_p.default if hasattr(top_p, 'default') else top_p
    parallel_tool_calls = parallel_tool_calls.default if hasattr(parallel_tool_calls, 'default') else parallel_tool_calls
    json_schema = json_schema.default if hasattr(json_schema, 'default') else json_schema

    if model in ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
                 "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "meta-llama/Llama-3.2-3B-Instruct-Turbo", 
                 "Qwen/Qwen2.5-Coder-32B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501", "mistralai/Mistral-7B-Instruct-v0.3",
                  "mistralai/Mixtral-8x22B-Instruct-v0.1"]:
        logger.warning("Not multimodal LLM. This model does not support image type inputs.", model=model)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in memory:
        openai_msg = await msg.to_openai_input(model=model)
        # Extract content from the message
        if isinstance(openai_msg["content"], list):
            is_text_only = all(
                (isinstance(item, dict) and "text" in item) or isinstance(item, str)
                for item in openai_msg["content"]
            )
            if is_text_only:
                content_text = []
                for item in openai_msg["content"]:
                    if isinstance(item, dict) and "text" in item:
                        content_text.append(item["text"])
                    elif isinstance(item, str):
                        content_text.append(item)
                openai_msg["content"] = " ".join(content_text)
            messages.append(openai_msg)

    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
        "logprobs": top_logprobs,
        "presence_penalty": presence_penalty,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Optional parameters
    if frequency_penalty:
        kwargs["frequency_penalty"] = frequency_penalty
    if stop:
        kwargs["stop"] = stop
    if seed:
        kwargs["seed"] = seed
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    # Change parameters if provided to TogetherAI format
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