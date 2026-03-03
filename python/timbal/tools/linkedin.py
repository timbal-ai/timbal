from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_LINKEDIN_API_BASE = "https://api.linkedin.com/v2"


class SearchPeople(Tool):
    name: str = "linkedin_search_people"
    description: str | None = (
        "Search for people on LinkedIn. Supports keyword query with optional filters "
        "by country, industry, seniority, company, and more."
    )
    integration: Annotated[str, Integration("linkedin")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_people(
            query: str,
            top_k: int = 10,
            country: str | None = None,
            industries: list[str] | None = None,
            seniority_levels: list[str] | None = None,
            current_companies: list[str] | None = None,
            past_companies: list[str] | None = None,
            schools: list[str] | None = None,
            network_depths: list[str] | None = None,
            title: str | None = None,
            start: int = 0,
        ) -> Any:
            """
            query: keywords to search for (name, title, skills, etc.).
            top_k: number of results to return (max 49 per LinkedIn API limits).
            country: ISO 3166-1 alpha-2 country code, e.g. "US", "GB", "ES".
            industries: list of LinkedIn industry URNs or names, e.g. ["Software Development", "Finance"].
            seniority_levels: list of levels, e.g. ["DIRECTOR", "VP", "CXO", "SENIOR", "ENTRY", "MANAGER"].
            current_companies: list of company names or URNs to filter by current employer.
            past_companies: list of company names or URNs to filter by past employer.
            schools: list of school names to filter by education.
            network_depths: list of connection degrees, e.g. ["F"] (1st), ["S"] (2nd), ["O"] (3rd+).
            title: filter by job title keyword.
            start: pagination offset.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                        "Authorization": f"Bearer {token}",
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
    integration: Annotated[str, Integration("linkedin")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_companies(
            query: str,
            top_k: int = 10,
            country: str | None = None,
            industries: list[str] | None = None,
            company_sizes: list[str] | None = None,
            start: int = 0,
        ) -> Any:
            """
            query: company name or keyword.
            country: ISO 3166-1 alpha-2 country code, e.g. "US", "DE".
            industries: list of industry names to filter by.
            company_sizes: list of size codes, e.g. ["B"] (2–10), ["C"] (11–50),
                           ["D"] (51–200), ["E"] (201–500), ["F"] (501–1000),
                           ["G"] (1001–5000), ["H"] (5001–10000), ["I"] (10001+).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                        "Authorization": f"Bearer {token}",
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
    integration: Annotated[str, Integration("linkedin")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_jobs(
            query: str,
            top_k: int = 10,
            country: str | None = None,
            location: str | None = None,
            industries: list[str] | None = None,
            seniority_levels: list[str] | None = None,
            job_types: list[str] | None = None,
            remote_filter: str | None = None,
            posted_at_range: str | None = None,
            start: int = 0,
        ) -> Any:
            """
            query: job title or keywords, e.g. "machine learning engineer".
            country: ISO 3166-1 alpha-2 country code.
            location: city or region name, e.g. "San Francisco Bay Area".
            seniority_levels: e.g. ["ENTRY_LEVEL", "MID_SENIOR_LEVEL", "DIRECTOR", "EXECUTIVE"].
            job_types: e.g. ["FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP", "TEMPORARY"].
            remote_filter: "REMOTE", "ON_SITE", or "HYBRID".
            posted_at_range: filter by posting age, e.g. "r86400" (24h), "r604800" (7 days),
                             "r2592000" (30 days).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                        "Authorization": f"Bearer {token}",
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
    integration: Annotated[str, Integration("linkedin")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search(
            query: str,
            type: str = "people",
            top_k: int = 10,
            country: str | None = None,
            filters: list[str] | None = None,
            start: int = 0,
        ) -> Any:
            """
            query: keyword(s) to search for.
            type: entity type to search — "people", "companies", or "jobs".
            top_k: max number of results (capped at 49 by LinkedIn API).
            country: ISO 3166-1 alpha-2 country code, e.g. "US", "GB", "ES", "DE".
            filters: list of filter strings in "key:value" format. Supported keys:
              People filters:
                - "industry:<name>"          e.g. "industry:Software Development"
                - "seniority:<level>"        e.g. "seniority:DIRECTOR"
                - "current_company:<name>"   e.g. "current_company:Google"
                - "past_company:<name>"
                - "school:<name>"
                - "network:<depth>"          e.g. "network:F" (1st), "network:S" (2nd)
                - "title:<keyword>"          e.g. "title:CTO"
              Company filters:
                - "industry:<name>"
                - "size:<code>"              e.g. "size:G" (1001–5000 employees)
              Job filters:
                - "job_type:<type>"          e.g. "job_type:FULL_TIME"
                - "seniority:<level>"        e.g. "seniority:MID_SENIOR_LEVEL"
                - "remote:<filter>"          e.g. "remote:REMOTE"
                - "posted:<range>"           e.g. "posted:r86400" (last 24h)
                - "location:<city/region>"
            start: pagination offset.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                        "Authorization": f"Bearer {token}",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LinkedIn/Search"
        super().__init__(handler=_search, metadata=metadata, **kwargs)
