import os
from typing import Annotated, Any
from urllib.parse import quote

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_WEBMASTERS_BASE = "https://searchconsole.googleapis.com/webmasters/v3"
_INSPECTION_BASE = "https://searchconsole.googleapis.com/v1"

def _resolve_site_url(tool: Any, site_url: str | None) -> str:
    resolved = site_url or tool.default_site_url or os.getenv("GOOGLE_SEARCH_CONSOLE_SITE_URL")
    if not resolved:
        raise ValueError(
            "Search Console site_url is required. Pass site_url, set default_site_url on the tool, "
            "or set GOOGLE_SEARCH_CONSOLE_SITE_URL."
        )
    return resolved


def _encoded_site_url(site_url: str) -> str:
    return quote(site_url, safe="")


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError(
        "Google Search Console credentials not found. Configure an integration or pass token."
    )


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

class GoogleSearchConsoleListSites(Tool):
    name: str = "google_search_console_list_sites"
    description: str | None = "List sites verified in Google Search Console."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config({"integration": self.integration, "token": self.token})}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sites() -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_WEBMASTERS_BASE}/sites",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_sites, **kwargs)


class GoogleSearchConsoleGetSite(Tool):
    name: str = "google_search_console_get_site"
    description: str | None = "Get Search Console site metadata and permission level."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_site(
            site_url: str | None = Field(
                None, description="Site URL, e.g. https://example.com/ or sc-domain:example.com"
            ),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_site, **kwargs)


class GoogleSearchConsoleAddSite(Tool):
    name: str = "google_search_console_add_site"
    description: str | None = "Add a site to Google Search Console."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config({"integration": self.integration, "token": self.token})}

    def __init__(self, **kwargs: Any) -> None:
        async def _add_site(
            site_url: str = Field(..., description="Site URL to add"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site_url)}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_add_site, **kwargs)


class GoogleSearchConsoleDeleteSite(Tool):
    name: str = "google_search_console_delete_site"
    description: str | None = "Remove a site from Google Search Console."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_site(
            site_url: str | None = Field(None, description="Site URL to remove"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return {"deleted": True, "siteUrl": site}

        super().__init__(handler=_delete_site, **kwargs)


class GoogleSearchConsoleSearchAnalytics(Tool):
    name: str = "google_search_console_search_analytics"
    description: str | None = "Query Search Console performance data with flexible dimensions and filters."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_analytics(
            site_url: str | None = Field(None, description="Site URL"),
            start_date: str = Field(..., description="Start date (YYYY-MM-DD)"),
            end_date: str = Field(..., description="End date (YYYY-MM-DD)"),
            dimensions: list[str] = Field(default_factory=list, description="Dimensions, e.g. query, page, country"),
            search_type: str = Field("web", description="Search type: web, image, video, news, discover, googleNews"),
            row_limit: int = Field(1000, description="Maximum rows to return"),
            start_row: int = Field(0, description="Zero-based row offset"),
            dimension_filter_groups: list[dict] | None = Field(None, description="Dimension filter groups"),
            aggregation_type: str | None = Field(None, description="auto, byPage, byProperty"),
            data_state: str | None = Field(None, description="final or all"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            body: dict[str, Any] = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": dimensions,
                "type": search_type,
                "rowLimit": row_limit,
                "startRow": start_row,
            }
            if dimension_filter_groups:
                body["dimensionFilterGroups"] = dimension_filter_groups
            if aggregation_type:
                body["aggregationType"] = aggregation_type
            if data_state:
                body["dataState"] = data_state

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}/searchAnalytics/query",
                    headers=_auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_analytics, **kwargs)


class GoogleSearchConsoleListSitemaps(Tool):
    name: str = "google_search_console_list_sitemaps"
    description: str | None = "List sitemaps submitted for a Search Console site."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sitemaps(
            site_url: str | None = Field(None, description="Site URL"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}/sitemaps",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_sitemaps, **kwargs)


class GoogleSearchConsoleGetSitemap(Tool):
    name: str = "google_search_console_get_sitemap"
    description: str | None = "Get details for a submitted sitemap."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sitemap(
            feedpath: str = Field(..., description="Sitemap URL or path"),
            site_url: str | None = Field(None, description="Site URL"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}/sitemaps/{_encoded_site_url(feedpath)}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_sitemap, **kwargs)


class GoogleSearchConsoleSubmitSitemap(Tool):
    name: str = "google_search_console_submit_sitemap"
    description: str | None = "Submit a sitemap URL to Google Search Console."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _submit_sitemap(
            feedpath: str = Field(..., description="Sitemap URL to submit"),
            site_url: str | None = Field(None, description="Site URL"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}/sitemaps/{_encoded_site_url(feedpath)}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_submit_sitemap, **kwargs)


class GoogleSearchConsoleDeleteSitemap(Tool):
    name: str = "google_search_console_delete_sitemap"
    description: str | None = "Delete a sitemap from Google Search Console."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_sitemap(
            feedpath: str = Field(..., description="Sitemap URL or path to delete"),
            site_url: str | None = Field(None, description="Site URL"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}/sitemaps/{_encoded_site_url(feedpath)}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return {"deleted": True, "feedpath": feedpath, "siteUrl": site}

        super().__init__(handler=_delete_sitemap, **kwargs)


class GoogleSearchConsoleInspectUrl(Tool):
    name: str = "google_search_console_inspect_url"
    description: str | None = "Inspect a URL index status via the Search Console URL Inspection API."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _inspect_url(
            inspection_url: str = Field(..., description="Fully qualified URL to inspect"),
            site_url: str | None = Field(None, description="Site URL property in Search Console"),
            language_code: str = Field("en-US", description="Language code for translated messages"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            body = {
                "inspectionUrl": inspection_url,
                "siteUrl": site,
                "languageCode": language_code,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_INSPECTION_BASE}/urlInspection/index:inspect",
                    headers=_auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_inspect_url, **kwargs)


class GoogleSearchConsoleSearchAnalyticsByPage(Tool):
    name: str = "google_search_console_search_analytics_by_page"
    description: str | None = "Query Search Console performance grouped by page."
    integration: Annotated[str, Integration("google_search_console")] | None = None
    token: SecretStr | None = None
    default_site_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "default_site_url": self.default_site_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_analytics_by_page(
            site_url: str | None = Field(None, description="Site URL"),
            start_date: str = Field(..., description="Start date (YYYY-MM-DD)"),
            end_date: str = Field(..., description="End date (YYYY-MM-DD)"),
            row_limit: int = Field(1000, description="Maximum rows to return"),
            start_row: int = Field(0, description="Zero-based row offset"),
            search_type: str = Field("web", description="Search type"),
        ) -> Any:
            token = await _resolve_token(self)
            site = _resolve_site_url(self, site_url)
            import httpx

            body = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": ["page"],
                "type": search_type,
                "rowLimit": row_limit,
                "startRow": start_row,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_WEBMASTERS_BASE}/sites/{_encoded_site_url(site)}/searchAnalytics/query",
                    headers=_auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_analytics_by_page, **kwargs)
