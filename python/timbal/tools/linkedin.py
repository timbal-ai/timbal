import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_LINKEDIN_API_BASE = "https://api.linkedin.com/v2"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve LinkedIn API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("LINKEDIN_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "LinkedIn API key not found. Set LINKEDIN_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class SearchPeople(Tool):
    name: str = "linkedin_search_people"
    description: str | None = (
        "Search for people on LinkedIn. Supports keyword query with optional filters "
        "by country, industry, seniority, company, and more."
    )
    integration: Annotated[str, Integration("linkedin")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_people(
                query: str = Field(..., description="Keywords to search for (name, title, skills, etc.)"),
                top_k: int = Field(10, description="Number of results to return (max 49 per LinkedIn API limits)"),
                country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code, e.g. 'US', 'GB', 'ES'"),
                industries: list[str] | None = Field(None, description="List of LinkedIn industry URNs or names, e.g. ['Software Development', 'Finance']"),
                seniority_levels: list[str] | None = Field(None, description="List of levels, e.g. ['DIRECTOR', 'VP', 'CXO', 'SENIOR', 'ENTRY', 'MANAGER']"),
                current_companies: list[str] | None = Field(None, description="List of company names or URNs to filter by current employer"),
                past_companies: list[str] | None = Field(None, description="List of company names or URNs to filter by past employer"),
                schools: list[str] | None = Field(None, description="List of school names to filter by education"),
                network_depths: list[str] | None = Field(None, description="List of connection degrees, e.g. ['F'] (1st), ['S'] (2nd), ['O'] (3rd+)"),
                title: str | None = Field(None, description="Filter by job title keyword"),
                start: int = Field(0, description="Pagination offset"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {
                "q": "people",
                "keywords": query,
                "count": min(top_k, 49),
                "start": start,
            }
            if country:
                params["facetGeoRegion"] = country
            if title:
                params["title"] = title
            if industries:
                params["facetIndustry"] = industries
            if seniority_levels:
                params["facetSeniority"] = seniority_levels
            if current_companies:
                params["facetCurrentEmployer"] = current_companies
            if past_companies:
                params["facetPastEmployer"] = past_companies
            if schools:
                params["facetSchool"] = schools
            if network_depths:
                params["facetNetwork"] = network_depths

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_LINKEDIN_API_BASE}/search",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LinkedIn/SearchPeople"
        super().__init__(handler=_search_people, metadata=metadata, **kwargs)


class SearchCompanies(Tool):
    name: str = "linkedin_search_companies"
    description: str | None = (
        "Search for companies on LinkedIn with optional filters by country, industry, and size."
    )
    integration: Annotated[str, Integration("linkedin")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_companies(
            query: str = Field(..., description="Company name or keyword"),
            top_k: int = Field(10, description="Number of results to return (max 49 per LinkedIn API limits)"),
            country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code, e.g. 'US', 'DE'"),
            industries: list[str] | None = Field(None, description="List of industry names to filter by"),
            company_sizes: list[str] | None = Field(None, description="List of size codes, e.g. ['B'] (2–10), ['C'] (11–50), ['D'] (51–200), ['E'] (201–500), ['F'] (501–1000), ['G'] (1001–5000), ['H'] (5001–10000), ['I'] (10001+)"),
            start: int = Field(0, description="Pagination offset"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {
                "q": "search",
                "query.keywords": query,
                "count": min(top_k, 49),
                "start": start,
            }
            if country:
                params["query.location.country"] = country
            if industries:
                params["facetIndustry"] = industries
            if company_sizes:
                params["facetCompanySize"] = company_sizes

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_LINKEDIN_API_BASE}/organizationsLookup",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LinkedIn/SearchCompanies"
        super().__init__(handler=_search_companies, metadata=metadata, **kwargs)


class SearchJobs(Tool):
    name: str = "linkedin_search_jobs"
    description: str | None = (
        "Search for job postings on LinkedIn with optional filters by country, "
        "industry, seniority, and job type."
    )
    integration: Annotated[str, Integration("linkedin")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_jobs(
            query: str = Field(..., description="Job title or keywords"),
            top_k: int = Field(10, description="Number of results to return (max 49 per LinkedIn API limits)"),
            country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code"),
            location: str | None = Field(None, description="City or region name, e.g. 'San Francisco Bay Area'"),
            industries: list[str] | None = Field(None, description="List of industry names to filter by"),
            seniority_levels: list[str] | None = Field(None, description="List of seniority levels, e.g. ['ENTRY_LEVEL', 'MID_SENIOR_LEVEL', 'DIRECTOR', 'EXECUTIVE']"),
            job_types: list[str] | None = Field(None, description="List of job types, e.g. ['FULL_TIME', 'PART_TIME', 'CONTRACT', 'INTERNSHIP', 'TEMPORARY']"),
            remote_filter: str | None = Field(None, description="Remote filter: 'REMOTE', 'ON_SITE', 'HYBRID'"),
            posted_at_range: str | None = Field(None, description="Time range: '24h', '7d', '30d', '90d'"),
            start: int = Field(0, description="Pagination offset"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {
                "keywords": query,
                "count": min(top_k, 49),
                "start": start,
            }
            if country:
                params["facetCountry"] = country
            if location:
                params["location"] = location
            if industries:
                params["facetIndustry"] = industries
            if seniority_levels:
                params["facetExperienceLevel"] = seniority_levels
            if job_types:
                params["facetJobType"] = job_types
            if remote_filter:
                params["facetRemoteType"] = remote_filter
            if posted_at_range:
                params["f_TPR"] = posted_at_range

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_LINKEDIN_API_BASE}/jobSearch",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LinkedIn/SearchJobs"
        super().__init__(handler=_search_jobs, metadata=metadata, **kwargs)


class Search(Tool):
    name: str = "linkedin_search"
    description: str | None = (
        "Unified LinkedIn search across people, companies, and jobs. "
        "Supports keyword query with optional type, country, and multi-select filters."
    )
    integration: Annotated[str, Integration("linkedin")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search(
            query: str = Field(..., description="Keyword(s) to search for"),
            type: str = Field("people", description="Entity type to search — 'people', 'companies', or 'jobs'"),
            top_k: int = Field(10, description="Max number of results (capped at 49 by LinkedIn API)"),
            country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code, e.g. 'US', 'GB', 'ES', 'DE'"),
            filters: list[str] | None = Field(None, description="List of filter strings in 'key:value' format. Supported keys: People filters: - 'industry:<name>'          e.g. 'industry:Software Development' - 'seniority:<level>'        e.g. 'seniority:DIRECTOR' - 'current_company:<name>'   e.g. 'current_company:Google' - 'past_company:<name>' - 'school:<name>' - 'network:<depth>'          e.g. 'network:F' (1st), 'network:S' (2nd) - 'title:<keyword>'          e.g. 'title:CTO' Company filters: - 'industry:<name>' - 'size:<code>'              e.g. 'size:G' (1001–5000 employees) Job filters: - 'job_type:<type>'          e.g. 'job_type:FULL_TIME' - 'seniority:<level>'        e.g. 'seniority:MID_SENIOR_LEVEL' - 'remote:<filter>'          e.g. 'remote:REMOTE' - 'posted:<range>'           e.g. 'posted:r86400' (last 24h) - 'location:<city/region>'"),
            start: int = Field(0, description="Pagination offset"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            parsed: dict[str, list[str]] = {}
            for f in (filters or []):
                if ":" in f:
                    k, v = f.split(":", 1)
                    parsed.setdefault(k.strip(), []).append(v.strip())

            if type == "people":
                params: dict[str, Any] = {
                    "q": "people",
                    "keywords": query,
                    "count": min(top_k, 49),
                    "start": start,
                }
                if country:
                    params["facetGeoRegion"] = country
                if "industry" in parsed:
                    params["facetIndustry"] = parsed["industry"]
                if "seniority" in parsed:
                    params["facetSeniority"] = parsed["seniority"]
                if "current_company" in parsed:
                    params["facetCurrentEmployer"] = parsed["current_company"]
                if "past_company" in parsed:
                    params["facetPastEmployer"] = parsed["past_company"]
                if "school" in parsed:
                    params["facetSchool"] = parsed["school"]
                if "network" in parsed:
                    params["facetNetwork"] = parsed["network"]
                if "title" in parsed:
                    params["title"] = parsed["title"][0]
                endpoint = f"{_LINKEDIN_API_BASE}/search"

            elif type == "companies":
                params = {
                    "q": "search",
                    "query.keywords": query,
                    "count": min(top_k, 49),
                    "start": start,
                }
                if country:
                    params["query.location.country"] = country
                if "industry" in parsed:
                    params["facetIndustry"] = parsed["industry"]
                if "size" in parsed:
                    params["facetCompanySize"] = parsed["size"]
                endpoint = f"{_LINKEDIN_API_BASE}/organizationsLookup"

            else:  # jobs
                params = {
                    "keywords": query,
                    "count": min(top_k, 49),
                    "start": start,
                }
                if country:
                    params["facetCountry"] = country
                if "location" in parsed:
                    params["location"] = parsed["location"][0]
                if "industry" in parsed:
                    params["facetIndustry"] = parsed["industry"]
                if "seniority" in parsed:
                    params["facetExperienceLevel"] = parsed["seniority"]
                if "job_type" in parsed:
                    params["facetJobType"] = parsed["job_type"]
                if "remote" in parsed:
                    params["facetRemoteType"] = parsed["remote"][0]
                if "posted" in parsed:
                    params["f_TPR"] = parsed["posted"][0]
                endpoint = f"{_LINKEDIN_API_BASE}/jobSearch"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LinkedIn/Search"
        super().__init__(handler=_search, metadata=metadata, **kwargs)
