import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.tavily.com"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Tavily API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("TAVILY_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Tavily API key not found. Set TAVILY_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class TavilySearch(Tool):
    name: str = "tavily_search"
    description: str | None = "Search the web using Tavily. Returns relevant results with titles, URLs, and content snippets."
    integration: Annotated[str, Integration("tavily")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _tavily_search(
            query: str = Field(..., description="Search query"),
            search_depth: str = Field("basic", description='"basic", "advanced", "fast", or "ultra-fast"'),
            topic: str = Field("general", description='"general", "news", or "finance"'),
            max_results: int = Field(5, description="Maximum number of results (1-20)"),
            time_range: str | None = Field(None, description='Time filter: "day", "week", "month", or "year"'),
            include_domains: list[str] | None = Field(None, description="Only include results from these domains"),
            exclude_domains: list[str] | None = Field(None, description="Exclude results from these domains"),
            include_answer: bool = Field(False, description="Include an AI-generated answer"),
            include_raw_content: bool = Field(False, description="Include cleaned page content in results"),
            include_images: bool = Field(False, description="Include image search results"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "query": query,
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
            }
            if time_range:
                payload["time_range"] = time_range
            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/search",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_tavily_search, **kwargs)


class TavilyExtract(Tool):
    name: str = "tavily_extract"
    description: str | None = "Extract content from web pages. Provide URLs to get their cleaned text or markdown content."
    integration: Annotated[str, Integration("tavily")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _tavily_extract(
            urls: str | list[str] = Field(..., description="URL or list of URLs to extract content from (max 20)"),
            extract_depth: str = Field("basic", description='"basic" or "advanced"'),
            include_images: bool = Field(False, description="Include images extracted from pages"),
            format: str = Field("markdown", description='"markdown" or "text"'),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "urls": urls,
                "extract_depth": extract_depth,
                "include_images": include_images,
                "format": format,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/extract",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_tavily_extract, **kwargs)


class TavilyCrawl(Tool):
    name: str = "tavily_crawl"
    description: str | None = (
        "Crawl a website and extract content from its pages. "
        "Follows links from a starting URL up to a specified depth."
    )
    integration: Annotated[str, Integration("tavily")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _tavily_crawl(
            url: str = Field(..., description="Starting URL to crawl"),
            max_depth: int = Field(1, description="How many links deep to crawl (1-5)"),
            max_breadth: int = Field(20, description="Links to follow per page level (1-500)"),
            limit: int = Field(50, description="Total pages to process"),
            instructions: str | None = Field(None, description="Natural language instructions to filter/focus the crawl"),
            select_paths: list[str] | None = Field(None, description="Regex patterns for paths to include"),
            exclude_paths: list[str] | None = Field(None, description="Regex patterns for paths to exclude"),
            extract_depth: str = Field("basic", description='"basic" or "advanced"'),
            format: str = Field("markdown", description='"markdown" or "text"'),
            allow_external: bool = Field(True, description="Follow links to external domains"),
            include_images: bool = Field(False, description="Extract images from pages"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "extract_depth": extract_depth,
                "format": format,
                "allow_external": allow_external,
                "include_images": include_images,
            }
            if instructions:
                payload["instructions"] = instructions
            if select_paths:
                payload["select_paths"] = select_paths
            if exclude_paths:
                payload["exclude_paths"] = exclude_paths

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/crawl",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(150.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_tavily_crawl, **kwargs)


class TavilyMap(Tool):
    name: str = "tavily_map"
    description: str | None = (
        "Discover and list all URLs on a website without extracting content. "
        "Use this to explore a site's structure before crawling or extracting."
    )
    integration: Annotated[str, Integration("tavily")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _tavily_map(
            url: str = Field(..., description="Root URL to map"),
            max_depth: int = Field(1, description="How many links deep to explore (1-5)"),
            max_breadth: int = Field(20, description="Links to follow per page level (1-500)"),
            limit: int = Field(50, description="Total links to discover"),
            instructions: str | None = Field(None, description="Natural language instructions to filter discovered URLs"),
            select_paths: list[str] | None = Field(None, description="Regex patterns for paths to include"),
            exclude_paths: list[str] | None = Field(None, description="Regex patterns for paths to exclude"),
            allow_external: bool = Field(True, description="Include links to external domains"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "allow_external": allow_external,
            }
            if instructions:
                payload["instructions"] = instructions
            if select_paths:
                payload["select_paths"] = select_paths
            if exclude_paths:
                payload["exclude_paths"] = exclude_paths

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/map",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(150.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_tavily_map, **kwargs)
