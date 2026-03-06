import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.firecrawl.dev/v2"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Firecrawl API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("FIRECRAWL_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Firecrawl API key not found. Set FIRECRAWL_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class FirecrawlScrape(Tool):
    name: str = "firecrawl_scrape"
    description: str | None = (
        "Scrape a single web page and extract its content as markdown, HTML, or structured data. "
        "Handles JavaScript rendering, anti-bot measures, and dynamic content automatically."
    )
    integration: Annotated[str, Integration("firecrawl")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _firecrawl_scrape(
            url: str = Field(..., description="The URL of the web page to scrape"),
            formats: list[str] = Field(
                ["markdown"],
                description='Output formats: "markdown", "html", "rawHtml", "links", "screenshot", "json"',
            ),
            only_main_content: bool = Field(
                True, description="Extract only the main content, excluding navigation, headers, and footers"
            ),
            include_tags: list[str] | None = Field(
                None, description="HTML tags, IDs, or classes to include exclusively"
            ),
            exclude_tags: list[str] | None = Field(
                None, description="HTML tags, IDs, or classes to exclude from the output"
            ),
            wait_for: int | None = Field(
                None, description="Milliseconds to wait for the page to load before scraping"
            ),
            timeout: int = Field(30000, description="Request timeout in milliseconds"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "url": url,
                "formats": formats,
                "onlyMainContent": only_main_content,
                "timeout": timeout,
            }
            if include_tags:
                payload["includeTags"] = include_tags
            if exclude_tags:
                payload["excludeTags"] = exclude_tags
            if wait_for is not None:
                payload["waitFor"] = wait_for

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/scrape",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_firecrawl_scrape, **kwargs)


class FirecrawlSearch(Tool):
    name: str = "firecrawl_search"
    description: str | None = (
        "Search the web and optionally scrape full page content from each result in one call. "
        "Supports web, news, and image sources with time-based filtering."
    )
    integration: Annotated[str, Integration("firecrawl")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _firecrawl_search(
            query: str = Field(..., description="The search query"),
            limit: int = Field(5, description="Number of results to return (max 20)"),
            sources: list[str] = Field(
                ["web"], description='Content sources: "web", "news", "images"'
            ),
            tbs: str | None = Field(
                None, description='Time filter: "qdr:h" (hour), "qdr:d" (day), "qdr:w" (week), "qdr:m" (month), "qdr:y" (year)'
            ),
            location: str | None = Field(None, description="ISO 3166-1 alpha-2 country code for localized results"),
            scrape_formats: list[str] | None = Field(
                None,
                description='Scrape each result in these formats: "markdown", "html", "links". Omit to skip scraping.',
            ),
            timeout: int = Field(30000, description="Request timeout in milliseconds"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "query": query,
                "limit": limit,
                "sources": sources,
                "timeout": timeout,
            }
            if tbs:
                payload["tbs"] = tbs
            if location:
                payload["location"] = {"country": location}
            if scrape_formats:
                payload["scrapeOptions"] = {"formats": scrape_formats}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/search",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_firecrawl_search, **kwargs)


class FirecrawlCrawl(Tool):
    name: str = "firecrawl_crawl"
    description: str | None = (
        "Crawl an entire website starting from a URL. Follows links, respects sitemaps, "
        "and returns content from all discovered pages. Runs asynchronously and polls until complete."
    )
    integration: Annotated[str, Integration("firecrawl")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _firecrawl_crawl(
            url: str = Field(..., description="The starting URL to crawl"),
            limit: int = Field(50, description="Maximum number of pages to crawl"),
            max_depth: int | None = Field(None, description="Maximum link depth to traverse from the starting URL"),
            include_paths: list[str] | None = Field(
                None, description="URL path patterns to include (e.g. /blog/*, /docs/*)"
            ),
            exclude_paths: list[str] | None = Field(
                None, description="URL path patterns to exclude (e.g. /admin/*, /private/*)"
            ),
            allow_external_links: bool = Field(False, description="Follow links to external domains"),
            allow_subdomains: bool = Field(False, description="Include subdomains when crawling"),
            formats: list[str] = Field(
                ["markdown"], description='Output formats per page: "markdown", "html", "rawHtml", "links", "screenshot"'
            ),
            only_main_content: bool = Field(
                True, description="Extract only main content, excluding navigation, headers, and footers"
            ),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import asyncio

            import httpx

            payload: dict[str, Any] = {
                "url": url,
                "limit": limit,
                "allowExternalLinks": allow_external_links,
                "allowSubdomains": allow_subdomains,
                "scrapeOptions": {
                    "formats": formats,
                    "onlyMainContent": only_main_content,
                },
            }
            if max_depth is not None:
                payload["maxDepth"] = max_depth
            if include_paths:
                payload["includePaths"] = include_paths
            if exclude_paths:
                payload["excludePaths"] = exclude_paths

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            async with httpx.AsyncClient() as client:
                # Start the crawl job.
                response = await client.post(
                    f"{_BASE_URL}/crawl",
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                job = response.json()
                job_id = job.get("id") or job.get("jobId")
                if not job_id:
                    return job

                # Poll for completion.
                while True:
                    await asyncio.sleep(2)
                    status_response = await client.get(
                        f"{_BASE_URL}/crawl/{job_id}",
                        headers=headers,
                        timeout=httpx.Timeout(30.0, read=None),
                    )
                    status_response.raise_for_status()
                    result = status_response.json()
                    status = result.get("status", "")
                    if status in ("completed", "failed", "cancelled"):
                        return result

        super().__init__(handler=_firecrawl_crawl, **kwargs)


class FirecrawlMap(Tool):
    name: str = "firecrawl_map"
    description: str | None = (
        "Discover all URLs on a website without scraping their content. "
        "Uses sitemaps and link discovery to build a complete URL map of a site."
    )
    integration: Annotated[str, Integration("firecrawl")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _firecrawl_map(
            url: str = Field(..., description="The root URL of the website to map"),
            limit: int | None = Field(None, description="Maximum number of URLs to return"),
            search: str | None = Field(
                None, description="Filter and rank URLs by relevance to this search term"
            ),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {"url": url}
            if limit is not None:
                payload["limit"] = limit
            if search:
                payload["search"] = search

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/map",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_firecrawl_map, **kwargs)


class FirecrawlExtract(Tool):
    name: str = "firecrawl_extract"
    description: str | None = (
        "Extract structured data from web pages using AI. Provide URLs and a natural language prompt "
        "or JSON schema to define the data you want. Supports wildcards (/*) for domain-wide extraction."
    )
    integration: Annotated[str, Integration("firecrawl")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _firecrawl_extract(
            urls: list[str] = Field(
                ...,
                description='URLs to extract from. Supports wildcards like "https://example.com/*" for domain-wide extraction.',
            ),
            prompt: str | None = Field(
                None, description="Natural language description of the data to extract"
            ),
            schema: dict | None = Field(
                None, description="JSON Schema defining the structure of data to extract"
            ),
            enable_web_search: bool = Field(
                False, description="Search related pages beyond the specified URLs for additional information"
            ),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            if not prompt and not schema:
                raise ValueError("At least one of 'prompt' or 'schema' is required for extraction.")

            payload: dict[str, Any] = {
                "urls": urls,
                "enableWebSearch": enable_web_search,
            }
            if prompt:
                payload["prompt"] = prompt
            if schema:
                payload["schema"] = schema

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/extract",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(120.0, read=None),
                )
                response.raise_for_status()
                result = response.json()

                # If async job, poll for completion.
                job_id = result.get("id") or result.get("jobId")
                if job_id and result.get("status") not in ("completed", "failed", "cancelled"):
                    import asyncio

                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    while True:
                        await asyncio.sleep(2)
                        status_response = await client.get(
                            f"{_BASE_URL}/extract/{job_id}",
                            headers=headers,
                            timeout=httpx.Timeout(30.0, read=None),
                        )
                        status_response.raise_for_status()
                        result = status_response.json()
                        if result.get("status") in ("completed", "failed", "cancelled"):
                            return result
                return result

        super().__init__(handler=_firecrawl_extract, **kwargs)
