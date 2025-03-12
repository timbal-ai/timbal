from typing import Any

from ...types import Field, Message, Tool
from .anthropic_llm import handler as anthropic_llm
from .gemini_llm import handler as gemini_llm
from .openai_llm import handler as openai_llm
from .togetherai_llm import handler as togetherai_llm


async def handler(
    # This is not actually used inside the function. It's a hack to have it as an argument
    # so we can use it when mapping data in the flow.
    prompt: Message = Field(default=None, description="Message to send to the LLM."), # noqa: ARG001
    system_prompt: str = Field(default=None, description="System prompt to guide the LLM's behavior and role."),
    memory: list[Message] = Field(description="Chat history containing user and LLM messages."),
    model: str = Field(default="gpt-4o", description="Name of the LLM model to use."),
    tools: list[Tool | dict] = Field(default=None, description=" List of tools/functions the LLM can call."),
    tool_choice: dict[str, Any] | str = Field(
        default={"type": "auto"},
        description="How the model should use the provided tools"
    ),
    max_tokens: int = Field(
        default=1024,
        description="The maximum number of tokens in the response"
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
        description="JSON schema for structured output."
    ),
) -> Message: # type: ignore
    """Route requests to appropriate LLM providers based on model name prefix.

    This gateway function handles routing to different LLM providers (OpenAI, Anthropic,
    Gemini, TogetherAI) based on the model name prefix.

    Args:
        prompt: The input text to send to the LLM.
        memory: Chat history containing user and LLM messages.
        system_prompt: Instructions for the LLM to follow.
        model: Name of the LLM model to use.
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
    top_k = top_k.default if hasattr(top_k, 'default') else top_k
    parallel_tool_calls = parallel_tool_calls.default if hasattr(parallel_tool_calls, 'default') else parallel_tool_calls
    json_schema = json_schema.default if hasattr(json_schema, 'default') else json_schema

    # Route to appropriate provider based on model prefix
    if model.startswith("claude"):
        response = await anthropic_llm(
            prompt, system_prompt, memory, model, tools, tool_choice,
            max_tokens, stop, temperature, top_k, top_p, json_schema
        )
    elif model.startswith("gpt"):
        response = await openai_llm(
            prompt, system_prompt, memory, model, tools, tool_choice,
            max_tokens, frequency_penalty, logprobs, top_logprobs, presence_penalty,
            seed, stop, temperature, top_p, parallel_tool_calls, json_schema
        )
    elif model.startswith("gemini"):
        response = await gemini_llm(
            prompt, system_prompt, memory, model, tools, tool_choice,
            max_tokens, presence_penalty, seed, stop, temperature, top_p, json_schema
        )
    else:
        response = await togetherai_llm(
            prompt, system_prompt, memory, model, tools, tool_choice,
            max_tokens, frequency_penalty, top_logprobs, presence_penalty,
            seed, stop, temperature, top_p, parallel_tool_calls, json_schema
        )

    async for chunk in response:
        yield chunk
