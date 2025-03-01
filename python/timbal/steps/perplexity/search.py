import os
from typing import Any, Literal

import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ...types import Field

load_dotenv()

logger = structlog.get_logger("timbal.steps.perplexity.search")

async def search(
    query: str = Field(description="Query to search for."),
    model: str = Field(default="sonar", description="Model to use."),
    system_prompt: str = Field(default=None, description="System prompt to guide the LLM's behavior and role."),
    # max_tokens: int | None = Field(default= None,description="Maximum number of tokens to generate."),
    temperature: float = Field(default=0.2, description="Amount of randomness in the response. Ranges from 0 to 2."),
    top_p: float = Field(default=0.9, description="Nucleus sampling threshold. Ranges from 0 to 1."),
    search_domain_filter: Any = Field(default=None, description="Given a list of domains, limit the citations used by the online model to URLs from the specified domains. Currently limited to only 3 domains for whitelisting and blacklisting. For blacklisting add a - to the beginning of the domain string. Only available in certain tiers"),
    return_images: bool = Field(default=False, description="Whether to return images in the response. Only available in certain tiers."),
    return_related_questions: bool = Field(default=False, description="Whether to return related questions in the response. Only available in certain tiers."),
    search_recency_filter: Literal['month', 'week', 'day', 'hour'] = Field(default=None, description="Returns search results within the specified time interval - does not apply to images."),
    top_k: int = Field(default=0, description="The number of tokens to keep for highest top-k filtering. Ranges from 0 to 2048. If set to 0, top-k filtering is disabled. We recommend either altering top_k or top_p, but not both."),
    stream: bool = Field(default=False, description="Whether or not to incrementally stream the response with server-sent events with content-type: text/event-stream."),
    presence_penalty: float = Field(default=0, description="Presence penalty to use. Ranges from -2.0 to 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Incompatible with frequency_penalty."),
    frequency_penalty: float = Field(default=1, description="A multiplicative penalty greater than 0. Values greater than 1.0 penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Incompatible with presence_penalty."),
    response_format: dict = Field(default=None, description="Enable structured outputs with a JSON or Regex schema. Only available in certain tiers. e.g. { type: 'json_schema', json_schema: {'schema': object} }, { type: 'regex', regex: {'regex': str} }"),
) -> Any:
    """This handler manages interactions with Perplexity's API, processing requests with 
    appropriate parameters and returning streaming responses.

    Args:
        query: Query to search for.
        model: Model to use. Defaults to "sonar".
        system_prompt: System prompt to guide the LLM's behavior and role.
        max_tokens: Maximum number of tokens to generate.
        temperature: Amount of randomness in the response. Ranges from 0 to 2.
        top_p: Nucleus sampling threshold. Ranges from 0 to 1.
        search_domain_filter: Given a list of domains, limit the citations used by the online model to URLs from the specified domains. Currently limited to only 3 domains for whitelisting and blacklisting. For blacklisting add a - to the beginning of the domain string. Only available in certain tiers
        return_images: Whether to return images in the response. Only available in certain tiers.
        return_related_questions: Whether to return related questions in the response. Only available in certain tiers.
        search_recency_filter: Returns search results within the specified time interval - does not apply to images.
        top_k: The number of tokens to keep for highest top-k filtering. Ranges from 0 to 2048. If set to 0, top-k filtering is disabled. We recommend either altering top_k or top_p, but not both.
        stream: Whether or not to incrementally stream the response with server-sent events with content-type: text/event-stream.
        presence_penalty: Presence penalty to use. Ranges from -2.0 to 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Incompatible with frequency_penalty.
        frequency_penalty: A multiplicative penalty greater than 0. Values greater than 1.0 penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Incompatible with presence_penalty.
        response_format: Enable structured outputs with a JSON or Regex schema. Only available in certain tiers. e.g. { type: 'json_schema', json_schema: {'schema': object} }, { type: 'regex', regex: {'regex': str} }

    Returns:
        Any: The streaming response from the Perplexity API.
    """
    client = AsyncOpenAI(
        api_key=os.getenv('PERPLEXITY_API_KEY'), 
        base_url="https://api.perplexity.ai",
    )

    # Get the actual values from Field objects
    model = model.default if hasattr(model, 'default') else model
    system_prompt = system_prompt.default if hasattr(system_prompt, 'default') else system_prompt
    temperature = temperature.default if hasattr(temperature, 'default') else temperature
    # max_tokens = max_tokens.default if hasattr(max_tokens, 'default') else max_tokens
    top_p = top_p.default if hasattr(top_p, 'default') else top_p
    search_domain_filter = search_domain_filter.default if hasattr(search_domain_filter, 'default') else search_domain_filter
    return_images = return_images.default if hasattr(return_images, 'default') else return_images
    return_related_questions = return_related_questions.default if hasattr(return_related_questions, 'default') else return_related_questions
    search_recency_filter = search_recency_filter.default if hasattr(search_recency_filter, 'default') else search_recency_filter
    top_k = top_k.default if hasattr(top_k, 'default') else top_k
    stream = stream.default if hasattr(stream, 'default') else stream
    presence_penalty = presence_penalty.default if hasattr(presence_penalty, 'default') else presence_penalty
    frequency_penalty = frequency_penalty.default if hasattr(frequency_penalty, 'default') else frequency_penalty
    response_format = response_format.default if hasattr(response_format, 'default') else response_format

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    
    response = await client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
        )

    async for chunk in response:
        yield chunk
