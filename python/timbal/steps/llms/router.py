import os

from openai import AsyncOpenAI

from ...graph.base import BaseStep
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
    # TODO Add all the rest of parameters.
) -> Message: # type: ignore
    
    # Enable calling this router without pydantic model_validate()
    model = model.default if hasattr(model, "default") else model
    system_prompt = system_prompt.default if hasattr(system_prompt, "default") else system_prompt
    messages = messages.default if hasattr(messages, "default") else messages
    tools = tools.default if hasattr(tools, "default") else tools

    if model in ["gpt-4o-mini", "gpt-4o"]:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            openai_message = message.to_openai_input()
            openai_messages.append(openai_message)

        # Collect all non required params as kwargs.
        openai_kwargs = {}

        openai_tools = []
        for tool in tools:
            openai_tools.append(tool.to_openai_tool())
        if openai_tools:
            openai_kwargs["tools"] = openai_tools

        res = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            stream=True,
            stream_options={"include_usage": True},
            **openai_kwargs,
        )

        async for res_chunk in res:
            yield res_chunk
    else:
        raise NotImplementedError(f"Model '{model}' is not implemented yet.")
