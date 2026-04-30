"""
WebSearch — unified web-search tool.

When ``provider`` is *not* set (the default), WebSearch behaves as a
specification-only tool: it emits native search schemas for Anthropic
(``web_search_20250305``) and OpenAI/xAI Responses API (``{"type":
"web_search"}``), but is **not executable**.  This is the original behaviour.

When ``provider`` is set to ``"tavily"``, ``"scraperapi"``, ``"cala"``,
``"firecrawl"``, or ``"google"``,
WebSearch becomes a fully executable function-calling tool that works with
**every** LLM provider (Anthropic, OpenAI, Gemini, Groq, Cerebras, …).

**Credentials (executable mode):** resolution order is explicit ``api_key`` on the
tool (and ``cx`` for Google), then ``Integration("web_search")`` when the org
stores a bundled web-search credential (Google: expect ``api_key`` + ``cx`` in
that payload), then the provider-specific environment variable (``TAVILY_API_KEY``,
``SCRAPERAPI_KEY``, ``CALA_API_KEY``, ``FIRECRAWL_API_KEY``,
``GOOGLE_CUSTOM_SEARCH_API_KEY`` + ``GOOGLE_CSE_CX``).

For ``provider="google"`` (Custom Search JSON API), at most 10 results are
returned per request; pass ``start`` to paginate for more.
"""

import os
from functools import cached_property
from typing import Annotated, Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from pydantic import Field, SecretStr, computed_field

from ..core.tool import Tool
from ..platform.integrations import Integration
from ..tools.cala import _BASE_URL as _CALA_BASE_URL
from ..tools.cala import _resolve_api_key as _resolve_cala_key
from ..tools.firecrawl import _BASE_URL as _FIRECRAWL_BASE_URL
from ..tools.firecrawl import _resolve_api_key as _resolve_firecrawl_key
from ..tools.scraperapi import _STRUCTURED_URL as _SCRAPERAPI_STRUCTURED_URL
from ..tools.scraperapi import _resolve_api_key as _resolve_scraperapi_key
from ..tools.tavily import _BASE_URL as _TAVILY_BASE_URL
from ..tools.tavily import _resolve_api_key as _resolve_tavily_key


def _get_logger():
    import structlog

    return structlog.get_logger("timbal.tools.web_search")


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------


def _make_tavily_handler(
    *,
    integration=None,
    api_key=None,
    allowed_domains=None,
    blocked_domains=None,
    fixed_max_results=None,
):
    """Return an async handler that searches via Tavily."""

    async def web_search(
        query: str = Field(..., description="Search query"),
        max_results: int = Field(5, description="Maximum number of results (1-20)"),
        search_depth: str = Field("basic", description='"basic" or "advanced"'),
        topic: str = Field("general", description='"general", "news", or "finance"'),
        time_range: str | None = Field(None, description='Time filter: "day", "week", "month", or "year"'),
    ) -> dict:
        resolved_key = await _resolve_tavily_key(integration=integration, api_key=api_key)
        import httpx

        payload: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": fixed_max_results if fixed_max_results is not None else max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if time_range:
            payload["time_range"] = time_range
        if allowed_domains:
            payload["include_domains"] = allowed_domains
        if blocked_domains:
            payload["exclude_domains"] = blocked_domains

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_TAVILY_BASE_URL}/search",
                headers={"Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(10.0, read=None),
            )
            response.raise_for_status()
            return response.json()

    return web_search


def _make_scraperapi_handler(*, integration=None, api_key=None, user_location=None):
    """Return an async handler that searches via ScraperAPI Google Search."""

    # Map user_location.country → country_code if available.
    fixed_country = None
    if user_location and isinstance(user_location, dict):
        fixed_country = user_location.get("country")

    async def web_search(
        query: str = Field(..., description="Search query"),
        country_code: str | None = Field(None, description="ISO country code for localized results (e.g. us, gb)"),
        hl: str | None = Field(None, description="Host language code (e.g. en, es, fr)"),
        start: int | None = Field(None, description="Pagination offset (0-based, increments of 10)"),
    ) -> dict:
        resolved_key = await _resolve_scraperapi_key(integration=integration, api_key=api_key)
        import httpx

        params: dict[str, Any] = {"api_key": resolved_key, "query": query}
        cc = fixed_country or country_code
        if cc:
            params["country_code"] = cc
        if hl:
            params["hl"] = hl
        if start is not None:
            params["start"] = start

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{_SCRAPERAPI_STRUCTURED_URL}/google/search",
                params=params,
                timeout=httpx.Timeout(30.0, read=None),
            )
            response.raise_for_status()
            return response.json()

    return web_search


def _make_cala_handler(*, integration=None, api_key=None):
    """Return an async handler that searches via Cala."""

    async def web_search(
        query: str = Field(..., description="Natural language search query"),
    ) -> dict:
        resolved_key = await _resolve_cala_key(integration=integration, api_key=api_key)
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_CALA_BASE_URL}/knowledge/search",
                headers={"x-api-key": resolved_key, "Content-Type": "application/json"},
                json={"input": query},
                timeout=httpx.Timeout(10.0, read=None),
            )
            response.raise_for_status()
            return response.json()

    return web_search


def _make_firecrawl_handler(*, integration=None, api_key=None, max_results=None):
    """Return an async handler that searches via Firecrawl."""

    async def web_search(
        query: str = Field(..., description="Search query"),
        limit: int = Field(5, description="Number of results to return (max 20)"),
        tbs: str | None = Field(
            None,
            description='Time filter: "qdr:h" (hour), "qdr:d" (day), "qdr:w" (week), "qdr:m" (month), "qdr:y" (year)',
        ),
        location: str | None = Field(None, description="Geo-targeted location (e.g. 'Germany', 'San Francisco')"),
        country: str | None = Field(None, description="ISO country code for localized results (e.g. 'US', 'DE')"),
    ) -> dict:
        resolved_key = await _resolve_firecrawl_key(integration=integration, api_key=api_key)
        import httpx

        payload: dict[str, Any] = {
            "query": query,
            "limit": max_results if max_results is not None else limit,
        }
        if tbs:
            payload["tbs"] = tbs
        if location:
            payload["location"] = location
        if country:
            payload["country"] = country

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_FIRECRAWL_BASE_URL}/search",
                headers={"Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=httpx.Timeout(90.0, read=None),
            )
            response.raise_for_status()
            return response.json()

    return web_search


_GOOGLE_BASE_URL = "https://www.googleapis.com/customsearch/v1"


async def _resolve_google_credentials(
    *,
    integration: Any = None,
    api_key: SecretStr | None = None,
    cx: str | None = None,
) -> tuple[str, str]:
    """Resolve Google Custom Search ``(api_key, cx)`` from args, integration, or env."""
    resolved_key: str | None = None
    resolved_cx = cx
    if api_key is not None:
        resolved_key = api_key.get_secret_value()
    if isinstance(integration, Integration) and (resolved_key is None or resolved_cx is None):
        creds = await integration.resolve()
        if resolved_key is None:
            k = creds.get("api_key")
            if k is not None:
                resolved_key = str(k)
        if resolved_cx is None:
            c = creds.get("cx")
            if c is not None:
                resolved_cx = str(c)
    if resolved_key is None:
        resolved_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    if resolved_cx is None:
        resolved_cx = os.getenv("GOOGLE_CSE_CX")
    if not resolved_key or not resolved_cx:
        raise ValueError(
            "Google Custom Search requires both an API key and a Programmable Search Engine ID (cx). "
            "Set GOOGLE_CUSTOM_SEARCH_API_KEY and GOOGLE_CSE_CX, pass api_key/cx on the tool, "
            "or configure Integration('web_search') with api_key and cx."
        )
    return resolved_key, resolved_cx


def _make_google_handler(
    *,
    integration=None,
    api_key=None,
    cx=None,
    fixed_max_results=None,
):
    """Return an async handler that searches via Google Custom Search JSON API."""

    async def web_search(
        query: str = Field(..., description="Search query"),
        num: int = Field(10, description="Number of results to return per request (1-10)"),
        start: int | None = Field(None, description="Pagination offset for the first result (Custom Search API)."),
        gl: str | None = Field(None, description="Geolocation hint (ISO 3166-1 alpha-2, e.g. 'us', 'de')"),
        lr: str | None = Field(None, description="Language restrict (e.g. 'lang_en')"),
        date_restrict: str | None = Field(None, description='Time filter for results: "d1", "w1", "m1", "y1"'),
    ) -> dict:
        resolved_key, resolved_cx = await _resolve_google_credentials(integration=integration, api_key=api_key, cx=cx)
        import httpx

        n = fixed_max_results if fixed_max_results is not None else num
        n = max(1, min(int(n), 10))

        params: dict[str, Any] = {
            "key": resolved_key,
            "cx": resolved_cx,
            "q": query,
            "num": n,
        }
        if start is not None:
            params["start"] = start
        if gl:
            params["gl"] = gl
        if lr:
            params["lr"] = lr
        if date_restrict:
            params["dateRestrict"] = date_restrict

        async with httpx.AsyncClient() as client:
            response = await client.get(
                _GOOGLE_BASE_URL,
                params=params,
                timeout=httpx.Timeout(10.0, read=None),
            )
            response.raise_for_status()
            return response.json()

    return web_search


# ---------------------------------------------------------------------------
# WebSearch tool
# ---------------------------------------------------------------------------


class WebSearch(Tool):
    """See module docstring for native vs provider-backed behaviour and credential order."""

    name: str = "web_search"
    description: str | None = None

    # --- provider selection ---
    provider: Literal["tavily", "scraperapi", "cala", "firecrawl", "google"] | None = None

    # --- credential fields (used when provider is set) ---
    integration: Annotated[str, Integration("web_search")] | None = None
    api_key: SecretStr | None = None
    cx: str | None = Field(None, description="Google Programmable Search Engine ID (only for provider=google).")

    # --- native-mode fields ---
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, Any] | None = None
    max_results: int | None = None
    enable_image_understanding: bool = False

    def __init__(self, **kwargs: Any) -> None:
        provider = kwargs.get("provider")

        # Default description when a provider is set (LLMs need it for function-calling).
        if provider is not None and "description" not in kwargs:
            kwargs["description"] = "Search the web for information."

        if "handler" not in kwargs:
            if provider == "tavily":
                kwargs["handler"] = _make_tavily_handler(
                    integration=kwargs.get("integration"),
                    api_key=kwargs.get("api_key"),
                    allowed_domains=kwargs.get("allowed_domains"),
                    blocked_domains=kwargs.get("blocked_domains"),
                    fixed_max_results=kwargs.get("max_results"),
                )
            elif provider == "scraperapi":
                kwargs["handler"] = _make_scraperapi_handler(
                    integration=kwargs.get("integration"),
                    api_key=kwargs.get("api_key"),
                    user_location=kwargs.get("user_location"),
                )
            elif provider == "cala":
                kwargs["handler"] = _make_cala_handler(
                    integration=kwargs.get("integration"),
                    api_key=kwargs.get("api_key"),
                )
            elif provider == "firecrawl":
                kwargs["handler"] = _make_firecrawl_handler(
                    integration=kwargs.get("integration"),
                    api_key=kwargs.get("api_key"),
                    max_results=kwargs.get("max_results"),
                )
            elif provider == "google":
                kwargs["handler"] = _make_google_handler(
                    integration=kwargs.get("integration"),
                    api_key=kwargs.get("api_key"),
                    cx=kwargs.get("cx"),
                    fixed_max_results=kwargs.get("max_results"),
                )
            else:

                def _unreachable():
                    raise NotImplementedError("This is a specification-only tool")

                kwargs["handler"] = _unreachable

        super().__init__(**kwargs)

    @override
    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "provider": self.provider,
                    "allowed_domains": self.allowed_domains,
                    "blocked_domains": self.blocked_domains,
                    "user_location": self.user_location,
                    "integration": self.integration,
                    "api_key": self.api_key,
                    "cx": self.cx,
                }
            ),
        }

    # ------------------------------------------------------------------
    # Schema overrides — branch on provider
    # ------------------------------------------------------------------

    @override
    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """See base class."""
        if self.provider is not None:
            # Function-calling schema (works with any provider).
            return {
                "type": "function",
                "name": self.name,
                "description": self.description or "",
                "parameters": self.format_params_model_schema(),
            }

        # Native schema for OpenAI / xAI Responses API.
        schema: dict[str, Any] = {
            "type": "web_search",
        }
        filters: dict[str, Any] = {}
        if self.allowed_domains:
            filters["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            filters["excluded_domains"] = self.blocked_domains
        if filters:
            schema["filters"] = filters
        if self.user_location:
            schema["user_location"] = self.user_location
        if self.enable_image_understanding:
            schema["enable_image_understanding"] = True
        return schema

    @override
    @computed_field(repr=False)
    @cached_property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """See base class."""
        if self.provider is not None:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description or "",
                    "parameters": self.format_params_model_schema(),
                },
            }
        raise ValueError("WebSearch is not compatible with OpenAI's chat completions API.")

    @override
    @computed_field
    @cached_property
    def anthropic_schema(self) -> dict[str, Any]:
        """See base class."""
        if self.provider is not None:
            return {
                "name": self.name,
                "description": self.description or "",
                "input_schema": self.format_params_model_schema(),
            }

        # Native Anthropic web search schema.
        anthropic_schema: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
        }
        if self.allowed_domains:
            anthropic_schema["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            anthropic_schema["blocked_domains"] = self.blocked_domains
        if self.user_location:
            anthropic_schema["user_location"] = self.user_location
        return anthropic_schema
