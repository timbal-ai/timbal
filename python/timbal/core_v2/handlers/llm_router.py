import os

from openai import AsyncOpenAI
from pydantic import Field

from ...errors import APIKeyNotFoundError
from ...types.message import Message
from ..runnable import Runnable


async def handler(
    model: str = Field(
        default="gpt-4o-mini", 
        description="Name of the LLM model to use.",
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
    # TODO Add all the rest of parameters.
) -> Message: # type: ignore

    if not model.startswith("gpt"):
        raise NotImplementedError("Only OpenAI models are supported at the mometn.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("OPENAI_API_KEY not found.")

    client = AsyncOpenAI(api_key=api_key)

    openai_messages = []
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})
    for message in messages:
        openai_message = await message.to_openai_input(model=model)
        openai_messages.append(openai_message)

    # Collect all non required params as kwargs.
    openai_kwargs = {
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    openai_tools = []
    for tool in tools:
        openai_tools.append(tool.openai_schema)
    if openai_tools:
        openai_kwargs["tools"] = openai_tools

    res = await client.chat.completions.create(
        model=model,
        messages=openai_messages,
        **openai_kwargs,
    )

    async for res_chunk in res:
        yield res_chunk
