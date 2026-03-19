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
            headers: dict[str, str] | None = Field(
                None, description="Custom HTTP headers to send with the request (e.g. cookies, user-agent)"
            ),
            wait_for: int | None = Field(
                None, description="Milliseconds to wait for the page to load before scraping"
            ),
            mobile: bool = Field(False, description="Emulate a mobile device when scraping"),
            skip_tls_verification: bool = Field(True, description="Skip TLS certificate verification"),
            location_country: str | None = Field(
                None, description="ISO 3166-1 alpha-2 country code for geo-located scraping (e.g. 'US', 'DE')"
            ),
            location_languages: list[str] | None = Field(
                None, description="Preferred languages/locales in priority order (e.g. ['en-US', 'de-DE'])"
            ),
            remove_base64_images: bool = Field(
                True, description="Remove base64-encoded images from the markdown output"
            ),
            block_ads: bool = Field(True, description="Block ads and cookie popups during scraping"),
            actions: list[dict] | None = Field(
                None,
                description=(
                    "Browser actions to perform before scraping. Each action is a dict with a 'type' key. "
                    'Types: "wait" (milliseconds or selector), "click" (selector), "write" (text), '
                    '"press" (key), "scroll" (direction), "screenshot", "scrape", "executeJavascript" (script).'
                ),
            ),
            timeout: int = Field(30000, description="Request timeout in milliseconds (1000-300000)"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "url": url,
                "formats": formats,
                "onlyMainContent": only_main_content,
                "mobile": mobile,
                "skipTlsVerification": skip_tls_verification,
                "removeBase64Images": remove_base64_images,
                "blockAds": block_ads,
                "timeout": timeout,
            }
            if include_tags:
                payload["includeTags"] = include_tags
            if exclude_tags:
                payload["excludeTags"] = exclude_tags
            if headers:
                payload["headers"] = headers
            if wait_for is not None:
                payload["waitFor"] = wait_for
            if location_country or location_languages:
                loc: dict[str, Any] = {}
                if location_country:
                    loc["country"] = location_country
                if location_languages:
                    loc["languages"] = location_languages
                payload["location"] = loc
            if actions:
                payload["actions"] = actions

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
            categories: list[str] | None = Field(
                None, description='Filter by category: "github", "research", "pdf"'
            ),
            tbs: str | None = Field(
                None, description='Time filter: "qdr:h" (hour), "qdr:d" (day), "qdr:w" (week), "qdr:m" (month), "qdr:y" (year)'
            ),
            location: str | None = Field(None, description="Geo-targeted location (e.g. 'Germany', 'San Francisco')"),
            country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code for localized results (e.g. 'US', 'DE')"),
            ignore_invalid_urls: bool = Field(False, description="Exclude invalid URLs from search results"),
            scrape_formats: list[str] | None = Field(
                None,
                description='Scrape each result in these formats: "markdown", "html", "rawHtml", "links". Omit to skip scraping.',
            ),
            scrape_only_main_content: bool = Field(
                True, description="When scraping results, extract only the main content"
            ),
            scrape_include_tags: list[str] | None = Field(
                None, description="When scraping results, HTML tags/IDs/classes to include exclusively"
            ),
            scrape_exclude_tags: list[str] | None = Field(
                None, description="When scraping results, HTML tags/IDs/classes to exclude"
            ),
            timeout: int = Field(60000, description="Request timeout in milliseconds"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "query": query,
                "limit": limit,
                "sources": sources,
                "timeout": timeout,
                "ignoreInvalidURLs": ignore_invalid_urls,
            }
            if categories:
                payload["categories"] = categories
            if tbs:
                payload["tbs"] = tbs
            if location:
                payload["location"] = location
            if country:
                payload["country"] = country
            if scrape_formats:
                scrape_opts: dict[str, Any] = {
                    "formats": scrape_formats,
                    "onlyMainContent": scrape_only_main_content,
                }
                if scrape_include_tags:
                    scrape_opts["includeTags"] = scrape_include_tags
                if scrape_exclude_tags:
                    scrape_opts["excludeTags"] = scrape_exclude_tags
                payload["scrapeOptions"] = scrape_opts

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/search",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(90.0, read=None),
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
            remove_base64_images: bool = Field(
                True, description="Remove base64-encoded images from the markdown output"
            ),
            block_ads: bool = Field(True, description="Block ads and cookie popups during scraping"),
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
                    "removeBase64Images": remove_base64_images,
                    "blockAds": block_ads,
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
