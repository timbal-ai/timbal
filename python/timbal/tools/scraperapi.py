import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "http://api.scraperapi.com"
_STRUCTURED_URL = "https://api.scraperapi.com/structured"
_ASYNC_URL = "https://async.scraperapi.com"


async def _resolve_api_key(*, integration: Any = None, api_key: SecretStr | None = None) -> str:
    """Resolve ScraperAPI key from integration, explicit field, or env var."""
    if isinstance(integration, Integration):
        credentials = await integration.resolve()
        return credentials["api_key"]
    if api_key is not None:
        return api_key.get_secret_value()
    env_key = os.getenv("SCRAPERAPI_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "ScraperAPI key not found. Set SCRAPERAPI_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class ScraperAPIScrape(Tool):
    name: str = "scraperapi_scrape"
    description: str | None = (
        "Scrape any web page and return its HTML, plain text, or markdown. "
        "Handles proxies, CAPTCHAs, and optionally JavaScript rendering."
    )
    integration: Annotated[str, Integration("scraperapi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _scraperapi_scrape(
            url: str = Field(..., description="The URL of the web page to scrape"),
            render: bool = Field(False, description="Enable JavaScript rendering (costs 10 credits)"),
            output_format: str | None = Field(
                None, description='Output format: "text" or "markdown". Omit to return raw HTML.'
            ),
            country_code: str | None = Field(
                None, description="ISO 3166-1 alpha-2 country code for geo-targeted proxies (e.g. us, gb, de)"
            ),
            device_type: str | None = Field(
                None, description='Set user-agent type: "desktop" or "mobile"'
            ),
            premium: bool = Field(False, description="Use premium residential proxies (costs 10 credits, 25 with render)"),
            autoparse: bool = Field(
                False, description="Auto-parse supported pages (e.g. Amazon, Google) and return structured JSON"
            ),
            wait_for_selector: str | None = Field(
                None, description="CSS selector to wait for before returning (requires render=True)"
            ),
            session_number: int | None = Field(
                None, description="Reuse the same proxy IP across requests by passing a consistent session number"
            ),
            keep_headers: bool = Field(False, description="Forward your custom request headers to the target"),
            retry_404: bool = Field(False, description="Retry requests that return a 404 status"),
        ) -> str:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            params: dict[str, Any] = {"api_key": api_key, "url": url}
            if render:
                params["render"] = "true"
            if output_format:
                params["output_format"] = output_format
            if country_code:
                params["country_code"] = country_code
            if device_type:
                params["device_type"] = device_type
            if premium:
                params["premium"] = "true"
            if autoparse:
                params["autoparse"] = "true"
            if wait_for_selector:
                params["wait_for_selector"] = wait_for_selector
            if session_number is not None:
                params["session_number"] = session_number
            if keep_headers:
                params["keep_headers"] = "true"
            if retry_404:
                params["retry_404"] = "true"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _BASE_URL,
                    params=params,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.text

        super().__init__(handler=_scraperapi_scrape, **kwargs)


class ScraperAPIAsyncScrape(Tool):
    name: str = "scraperapi_async_scrape"
    description: str | None = (
        "Submit a URL scraping job asynchronously and poll until it completes. "
        "Useful for long-running or JavaScript-rendered pages."
    )
    integration: Annotated[str, Integration("scraperapi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _scraperapi_async_scrape(
            url: str = Field(..., description="The URL of the web page to scrape"),
            render: bool = Field(False, description="Enable JavaScript rendering (costs 10 credits)"),
            output_format: str | None = Field(
                None, description='Output format: "text" or "markdown". Omit to return raw HTML.'
            ),
            country_code: str | None = Field(
                None, description="ISO 3166-1 alpha-2 country code for geo-targeted proxies"
            ),
            device_type: str | None = Field(
                None, description='Set user-agent type: "desktop" or "mobile"'
            ),
            premium: bool = Field(False, description="Use premium residential proxies"),
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import asyncio

            import httpx

            payload: dict[str, Any] = {"apiKey": api_key, "url": url}
            if render:
                payload["render"] = "true"
            if output_format:
                payload["output_format"] = output_format
            if country_code:
                payload["country_code"] = country_code
            if device_type:
                payload["device_type"] = device_type
            if premium:
                payload["premium"] = "true"

            async with httpx.AsyncClient() as client:
                # Submit the job.
                response = await client.post(
                    f"{_ASYNC_URL}/jobs",
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                job = response.json()
                job_id = job.get("id")
                status_url = job.get("statusUrl") or f"{_ASYNC_URL}/jobs/{job_id}"
                if not job_id:
                    return job

                # Poll for completion.
                while True:
                    await asyncio.sleep(2)
                    status_response = await client.get(
                        status_url,
                        timeout=httpx.Timeout(30.0, read=None),
                    )
                    status_response.raise_for_status()
                    result = status_response.json()
                    status = result.get("status", "")
                    if status in ("finished", "failed"):
                        return result

        super().__init__(handler=_scraperapi_async_scrape, **kwargs)


class ScraperAPIGoogleSearch(Tool):
    name: str = "scraperapi_google_search"
    description: str | None = (
        "Retrieve structured Google Search results including organic links, titles, snippets, "
        "and ads. Returns clean JSON without needing to parse HTML."
    )
    integration: Annotated[str, Integration("scraperapi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _scraperapi_google_search(
            query: str = Field(..., description="The search query"),
            country_code: str | None = Field(
                None, description="ISO 3166-1 alpha-2 country code for localized results (e.g. us, gb)"
            ),
            hl: str | None = Field(None, description="Host language code (e.g. en, es, fr)"),
            gl: str | None = Field(None, description="Geolocation country code for results"),
            start: int | None = Field(None, description="Pagination offset (0-based, increments of 10)"),
            tbs: str | None = Field(
                None,
                description='Time filter for results, e.g. "qdr:d" (past day), "qdr:w" (past week)',
            ),
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            params: dict[str, Any] = {"api_key": api_key, "query": query}
            if country_code:
                params["country_code"] = country_code
            if hl:
                params["hl"] = hl
            if gl:
                params["gl"] = gl
            if start is not None:
                params["start"] = start
            if tbs:
                params["tbs"] = tbs

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRUCTURED_URL}/google/search",
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_scraperapi_google_search, **kwargs)


class ScraperAPIAmazonProduct(Tool):
    name: str = "scraperapi_amazon_product"
    description: str | None = (
        "Retrieve structured Amazon product details by ASIN, including title, price, rating, "
        "description, and more. Returns clean JSON."
    )
    integration: Annotated[str, Integration("scraperapi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _scraperapi_amazon_product(
            asin: str = Field(..., description="The Amazon Standard Identification Number (ASIN) of the product"),
            country_code: str | None = Field(
                None, description="Amazon marketplace country code (e.g. us, gb, de)"
            ),
            tld: str | None = Field(
                None, description="Amazon TLD for the target marketplace (e.g. com, co.uk, de)"
            ),
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            params: dict[str, Any] = {"api_key": api_key, "asin": asin}
            if country_code:
                params["country_code"] = country_code
            if tld:
                params["tld"] = tld

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRUCTURED_URL}/amazon/product",
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_scraperapi_amazon_product, **kwargs)


class ScraperAPIAmazonSearch(Tool):
    name: str = "scraperapi_amazon_search"
    description: str | None = (
        "Search Amazon and retrieve structured product listings including titles, prices, ratings, "
        "and ASINs. Returns clean JSON."
    )
    integration: Annotated[str, Integration("scraperapi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _scraperapi_amazon_search(
            query: str = Field(..., description="Search query for Amazon products"),
            country_code: str | None = Field(
                None, description="Amazon marketplace country code (e.g. us, gb, de)"
            ),
            tld: str | None = Field(
                None, description="Amazon TLD for the target marketplace (e.g. com, co.uk, de)"
            ),
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            params: dict[str, Any] = {"api_key": api_key, "query": query}
            if country_code:
                params["country_code"] = country_code
            if tld:
                params["tld"] = tld

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRUCTURED_URL}/amazon/search",
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_scraperapi_amazon_search, **kwargs)
