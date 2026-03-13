import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_CF_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Return (api_token, account_id) from integration, explicit fields, or env vars."""
    api_token: str | None = None
    account_id: str | None = None

    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        print(credentials)
        api_token = credentials["api_token"]
        account_id = credentials["account_id"]
    else:
        if getattr(tool, "api_token", None) is not None:
            api_token = tool.api_token.get_secret_value()
        if getattr(tool, "account_id", None) is not None:
            account_id = tool.account_id

    if not api_token:
        api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")

    if not api_token:
        raise ValueError(
            "Cloudflare API token not found. Set CLOUDFLARE_API_TOKEN environment variable, "
            "pass api_token in config, or configure an integration."
        )
    if not account_id:
        raise ValueError(
            "Cloudflare account ID not found. Set CLOUDFLARE_ACCOUNT_ID environment variable, "
            "pass account_id in config, or configure an integration."
        )

    return api_token, account_id


def _auth_headers(api_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }


class CloudflareCrawlStart(Tool):
    name: str = "cloudflare_crawl_start"
    description: str | None = (
        "Initiate a Cloudflare Browser Rendering crawl job. "
        "Returns a job_id that can be polled with cloudflare_crawl_get."
    )
    integration: Annotated[str, Integration("cloudflare")] | None = None
    api_token: SecretStr | None = None
    account_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_token": self.api_token,
                    "account_id": self.account_id,
                },
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _crawl_start(
            url: str = Field(..., description="Starting URL for the crawl."),
            limit: int | None = Field(
                None,
                description="Maximum number of pages to crawl (default 10, max 100,000).",
            ),
            depth: int | None = Field(
                None,
                description="Maximum link depth from the starting URL (default 100,000).",
            ),
            formats: list[str] | None = Field(
                None,
                description="Response formats: 'html', 'markdown', or 'json'. Defaults to html.",
            ),
            render: bool | None = Field(
                None,
                description="If false, skips JavaScript execution for a faster fetch (default true).",
            ),
            source: str | None = Field(
                None,
                description="URL discovery source: 'all', 'sitemaps', or 'links' (default 'all').",
            ),
            include_external_links: bool | None = Field(
                None, description="Follow links to external domains (default false)."
            ),
            include_subdomains: bool | None = Field(
                None, description="Follow links to subdomains of the starting URL (default false)."
            ),
            include_patterns: list[str] | None = Field(
                None,
                description="Only visit URLs matching these wildcard patterns.",
            ),
            exclude_patterns: list[str] | None = Field(
                None,
                description="Skip URLs matching these wildcard patterns.",
            ),
            max_age: int | None = Field(
                None,
                description="Max cache age in seconds for crawled resources (default 86400, max 604800).",
            ),
            modified_since: int | None = Field(
                None,
                description="Unix timestamp (seconds). Only crawl pages modified after this time.",
            ),
        ) -> Any:
            api_token, account_id = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {"url": url}

            if limit is not None:
                body["limit"] = limit
            if depth is not None:
                body["depth"] = depth
            if formats is not None:
                body["formats"] = formats
            if render is not None:
                body["render"] = render
            if source is not None:
                body["source"] = source
            if max_age is not None:
                body["maxAge"] = max_age
            if modified_since is not None:
                body["modifiedSince"] = modified_since

            options: dict[str, Any] = {}
            if include_external_links is not None:
                options["includeExternalLinks"] = include_external_links
            if include_subdomains is not None:
                options["includeSubdomains"] = include_subdomains
            if include_patterns is not None:
                options["includePatterns"] = include_patterns
            if exclude_patterns is not None:
                options["excludePatterns"] = exclude_patterns
            if options:
                body["options"] = options

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_CF_API_BASE}/{account_id}/browser-rendering/crawl",
                    headers=_auth_headers(api_token),
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
                return {"job_id": data["result"], "success": data["success"]}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "CloudflareCrawl/Start"

        super().__init__(handler=_crawl_start, metadata=metadata, **kwargs)


class CloudflareCrawlGet(Tool):
    name: str = "cloudflare_crawl_get"
    description: str | None = (
        "Get the status or results of a Cloudflare crawl job. "
        "Poll with limit=1 while status is 'running', then fetch full results when completed."
    )
    integration: Annotated[str, Integration("cloudflare")] | None = None
    api_token: SecretStr | None = None
    account_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_token": self.api_token,
                    "account_id": self.account_id,
                },
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _crawl_get(
            job_id: str = Field(..., description="Crawl job ID returned by cloudflare_crawl_start."),
            limit: int | None = Field(
                None,
                description="Max records to return. Pass 1 to cheaply check status while running.",
            ),
            cursor: str | None = Field(
                None,
                description="Pagination cursor returned in the previous response when results exceed 10 MB.",
            ),
            status_filter: str | None = Field(
                None,
                description="Filter records by URL status: 'queued', 'completed', 'disallowed', 'skipped', 'errored', or 'cancelled'.",
            ),
        ) -> Any:
            api_token, account_id = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if limit is not None:
                params["limit"] = limit
            if cursor is not None:
                params["cursor"] = cursor
            if status_filter is not None:
                params["status"] = status_filter

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_CF_API_BASE}/{account_id}/browser-rendering/crawl/{job_id}",
                    headers=_auth_headers(api_token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "CloudflareCrawl/Get"

        super().__init__(handler=_crawl_get, metadata=metadata, **kwargs)


class CloudflareCrawlCancel(Tool):
    name: str = "cloudflare_crawl_cancel"
    description: str | None = "Cancel a running Cloudflare crawl job."
    integration: Annotated[str, Integration("cloudflare")] | None = None
    api_token: SecretStr | None = None
    account_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_token": self.api_token,
                    "account_id": self.account_id,
                },
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _crawl_cancel(
            job_id: str = Field(..., description="Crawl job ID to cancel."),
        ) -> Any:
            api_token, account_id = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_CF_API_BASE}/{account_id}/browser-rendering/crawl/{job_id}",
                    headers=_auth_headers(api_token),
                )
                response.raise_for_status()
                return response.json() if response.content else {"success": True}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "CloudflareCrawl/Cancel"

        super().__init__(handler=_crawl_cancel, metadata=metadata, **kwargs)
