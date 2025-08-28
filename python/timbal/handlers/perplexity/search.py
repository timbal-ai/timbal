import os
from typing import Literal

import httpx
import structlog
from dotenv import load_dotenv
from pydantic import Field

from ...errors import APIKeyNotFoundError
from ...utils import resolve_default

load_dotenv()

logger = structlog.get_logger("timbal.steps.perplexity.search")

async def search(
    query: str = Field(
        ...,
        description="Query to search for."
    ),
    model: Literal[
        "sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-reasoning", 
        "sonar-deep-research", "r1-1776"
    ] = Field(
        "sonar",
        description="Model to use."
    ),
    system_prompt: str | None = Field(
        None,
        description="System prompt to guide the LLM's behavior and role."
    ),
    search_mode: Literal["academic", "web"] = Field(
        "web",
        description="Search mode to use."
    ),
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        "medium",
        description="Controls how much computational effort the AI dedicates to each query for deep research models."
    ),
    search_domain_filter: list[str] | None = Field(
        None,
        description="A list of domains to limit search results to. Currently limited to 10 domains for Allowlisting and Denylisting. For Denylisting, add a - at the beginning of the domain string."
    ),
    search_recency_filter: str | None = Field(
        None,
        description="Filters search results based on time"
    ),
    search_after_date_filter: str | None = Field(
        None,
        description="Filters search results to only include content published after this date. Format can be flexible (e.g., '3/1/2025', 'March 1, 2025')."
    ),
    search_before_date_filter: str | None = Field(
        None,
        description="Filters search results to only include content published before this date. Format can be flexible (e.g., '3/1/2025', 'March 1, 2025')."
    ),
    web_search_options: dict | None = Field(
        None,
        description="Configuration for web search including search_context_size (low/medium/high) and user_location for geographic refinement (latitude, longitude, country)."
    ),
) -> str:
    """This handler manages interactions with Perplexity's API, processing requests with 
    appropriate parameters and returning a formatted string with the answer and citations.

    Args:
        query: Query to search for.
        model: Model to use. Defaults to "sonar".
        system_prompt: System prompt to guide the LLM's behavior and role.
        search_mode: Search mode to use. Defaults to "web".
        reasoning_effort: Controls how much computational effort the AI dedicates to each query for deep research models. Defaults to "medium".
        search_domain_filter: A list of domains to limit search results to. Currently limited to 10 domains for Allowlisting and Denylisting. For Denylisting, add a - at the beginning of the domain string.
        search_recency_filter: Filters search results based on time
        search_after_date_filter: Filters search results to only include content published after this date. Format can be flexible (e.g., '3/1/2025', 'March 1, 2025').
        search_before_date_filter: Filters search results to only include content published before this date. Format can be flexible (e.g., '3/1/2025', 'March 1, 2025').
        web_search_options: Configuration for web search including search_context_size (low/medium/high) and user_location for geographic refinement (latitude, longitude, country).
    Returns:
        str: A formatted string containing the answer with properly formatted citations as markdown links.
    """
    model = resolve_default("model", model)
    system_prompt = resolve_default("system_prompt", system_prompt)
    search_mode = resolve_default("search_mode", search_mode)
    reasoning_effort = resolve_default("reasoning_effort", reasoning_effort)
    search_domain_filter = resolve_default("search_domain_filter", search_domain_filter)
    search_recency_filter = resolve_default("search_recency_filter", search_recency_filter)
    search_after_date_filter = resolve_default("search_after_date_filter", search_after_date_filter)
    search_before_date_filter = resolve_default("search_before_date_filter", search_before_date_filter)
    web_search_options = resolve_default("web_search_options", web_search_options)

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("PERPLEXITY_API_KEY not found")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    
    # Build the request payload, only including non-None values
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "search_mode": search_mode,
        "reasoning_effort": reasoning_effort,
    }
    
    # Only add optional parameters if they have values
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    if search_after_date_filter:
        payload["search_after_date_filter"] = search_after_date_filter
    if search_before_date_filter:
        payload["search_before_date_filter"] = search_before_date_filter
    if web_search_options:
        payload["web_search_options"] = web_search_options
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0)) as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
        )
    
    response.raise_for_status()
    
    response_data = response.json()
    
    answer_content = response_data["choices"][0]["message"]["content"]
    
    # TODO: There are missing citations in the response, we need to fix this
    citations = response_data.get("citations", [])
    # Replace citation markers like [1], [2], etc. with markdown links
    formatted_content = answer_content
    for i, citation_url in enumerate(citations, 1):
        citation_marker = f"[{i}]"
        if citation_marker in formatted_content:
            formatted_content = formatted_content.replace(
                citation_marker, 
                f"[{i}]({citation_url})"
            )
    
    return formatted_content