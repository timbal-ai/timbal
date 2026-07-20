"""Bizneo HR REST API tools.

Auth uses API key ID + secret against a tenant base URL (e.g. https://{tenant}.bizneohr.com).
Exact path prefixes are placeholders until Bizneo OpenAPI is available; use ``BizneoHRRequest``
for arbitrary endpoints once credentials are configured.

Integration credentials (type: credentials):
- base_url: Tenant API base (e.g. https://acme.bizneohr.com)
- api_key: API key ID
- api_secret: API key secret
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_PREFIX = "/api/v1"
_DEFAULT_API_KEY_HEADER = "X-API-KEY"
_DEFAULT_API_SECRET_HEADER = "X-API-SECRET"


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _join_path(base_url: str, path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    if normalized_path.startswith(_API_PREFIX):
        return f"{base_url}{normalized_path}"
    return f"{base_url}{_API_PREFIX}{normalized_path}"


async def _resolve_credentials(tool: Any) -> tuple[str, str, str]:
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    base_url = creds.get("base_url") or getattr(tool, "base_url", None) or os.getenv("BIZNEO_BASE_URL")
    raw_key = creds.get("api_key") or getattr(tool, "api_key", None) or os.getenv("BIZNEO_API_KEY")
    api_key = raw_key.get_secret_value() if isinstance(raw_key, SecretStr) else raw_key
    api_secret = creds.get("api_secret") or (
        tool.api_secret.get_secret_value()
        if getattr(tool, "api_secret", None) and tool.api_secret
        else None
    ) or os.getenv("BIZNEO_API_SECRET")

    if not base_url or not api_key or not api_secret:
        raise ValueError(
            "Bizneo HR credentials not found. Configure integration with "
            "base_url, api_key, api_secret, or set BIZNEO_BASE_URL, BIZNEO_API_KEY, BIZNEO_API_SECRET."
        )
    return _normalize_base_url(str(base_url)), str(api_key), str(api_secret)


def _auth_headers(api_key: str, api_secret: str) -> dict[str, str]:
    key_header = os.getenv("BIZNEO_API_KEY_HEADER", _DEFAULT_API_KEY_HEADER)
    secret_header = os.getenv("BIZNEO_API_SECRET_HEADER", _DEFAULT_API_SECRET_HEADER)
    return {
        key_header: api_key,
        secret_header: api_secret,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _bizneo_request(
    tool: Any,
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    import httpx

    base_url, api_key, api_secret = await _resolve_credentials(tool)
    url = _join_path(base_url, path)
    headers = _auth_headers(api_key, api_secret)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.request(
            method.upper(),
            url,
            headers=headers,
            params=params or None,
            json=json_body,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _bizneo_config_fields(tool: Any) -> dict[str, Any]:
    return {
        "integration": tool.integration,
        "base_url": tool.base_url,
        "api_key": tool.api_key,
        "api_secret": tool.api_secret,
    }


class BizneoHRRequest(Tool):
    """Call any Bizneo HR REST endpoint (escape hatch until full OpenAPI coverage)."""

    name: str = "bizneo_hr_request"
    description: str | None = (
        "Call any Bizneo HR REST endpoint with method, path, optional query params and JSON body. "
        "Use when no dedicated tool exists yet."
    )
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _request(
            method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = Field(
                "GET",
                description="HTTP method.",
            ),
            path: str = Field(
                ...,
                description="API path relative to /api/v1 (e.g. 'jobs' or '/jobs/123').",
            ),
            query_params: dict[str, Any] | None = Field(
                None,
                description="Optional query string parameters.",
            ),
            body: dict[str, Any] | None = Field(None, description="Optional JSON request body."),
        ) -> Any:
            return await _bizneo_request(
                self,
                method=method,
                path=path,
                params=query_params,
                json_body=body,
            )

        super().__init__(handler=_request, **kwargs)


class BizneoHRListJobs(Tool):
    """List job openings / vacancies in Bizneo HR ATS."""

    name: str = "bizneo_hr_list_jobs"
    description: str | None = "List job openings / vacancies in Bizneo HR ATS."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_jobs(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(50, description="Results per page."),
        ) -> Any:
            return await _bizneo_request(
                self,
                method="GET",
                path="/jobs",
                params={"page": page, "per_page": per_page},
            )

        super().__init__(handler=_list_jobs, **kwargs)


class BizneoHRGetJob(Tool):
    """Get a single job opening / vacancy by ID."""

    name: str = "bizneo_hr_get_job"
    description: str | None = "Get a single job opening / vacancy by ID."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_job(
            job_id: str = Field(..., description="Job / vacancy ID."),
        ) -> Any:
            return await _bizneo_request(self, method="GET", path=f"/jobs/{job_id}")

        super().__init__(handler=_get_job, **kwargs)


class BizneoHRListDepartments(Tool):
    """List departments in Bizneo HR."""

    name: str = "bizneo_hr_list_departments"
    description: str | None = "List departments in Bizneo HR."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_departments(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(50, description="Results per page."),
        ) -> Any:
            return await _bizneo_request(
                self,
                method="GET",
                path="/departments",
                params={"page": page, "per_page": per_page},
            )

        super().__init__(handler=_list_departments, **kwargs)


class BizneoHRListLocations(Tool):
    """List office locations in Bizneo HR."""

    name: str = "bizneo_hr_list_locations"
    description: str | None = "List office locations in Bizneo HR."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_locations(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(50, description="Results per page."),
        ) -> Any:
            return await _bizneo_request(
                self,
                method="GET",
                path="/locations",
                params={"page": page, "per_page": per_page},
            )

        super().__init__(handler=_list_locations, **kwargs)


class BizneoHRListCandidates(Tool):
    """List candidates in Bizneo HR talent pool."""

    name: str = "bizneo_hr_list_candidates"
    description: str | None = "List candidates in Bizneo HR talent pool."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_candidates(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(50, description="Results per page."),
        ) -> Any:
            return await _bizneo_request(
                self,
                method="GET",
                path="/candidates",
                params={"page": page, "per_page": per_page},
            )

        super().__init__(handler=_list_candidates, **kwargs)


class BizneoHRGetCandidate(Tool):
    """Get a single candidate by ID."""

    name: str = "bizneo_hr_get_candidate"
    description: str | None = "Get a single candidate by ID."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_candidate(
            candidate_id: str = Field(..., description="Candidate ID."),
        ) -> Any:
            return await _bizneo_request(self, method="GET", path=f"/candidates/{candidate_id}")

        super().__init__(handler=_get_candidate, **kwargs)


class BizneoHRListEmployees(Tool):
    """List employees in Bizneo HR core HR."""

    name: str = "bizneo_hr_list_employees"
    description: str | None = "List employees in Bizneo HR core HR."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_employees(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(50, description="Results per page."),
        ) -> Any:
            return await _bizneo_request(
                self,
                method="GET",
                path="/employees",
                params={"page": page, "per_page": per_page},
            )

        super().__init__(handler=_list_employees, **kwargs)


class BizneoHRGetEmployee(Tool):
    """Get a single employee by ID."""

    name: str = "bizneo_hr_get_employee"
    description: str | None = "Get a single employee by ID."
    integration: Annotated[str, Integration("bizneo_hr")] | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_bizneo_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_employee(
            employee_id: str = Field(..., description="Employee ID."),
        ) -> Any:
            return await _bizneo_request(self, method="GET", path=f"/employees/{employee_id}")

        super().__init__(handler=_get_employee, **kwargs)
