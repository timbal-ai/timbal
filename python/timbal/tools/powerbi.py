"""Power BI REST API tools.

Auth (first match wins):

- **User delegated (OAuth)**: ``token`` / ``access_token`` from
  ``Integration("powerbi")``, tool field ``token``, or env ``POWERBI_ACCESS_TOKEN``.
- **App delegated (service principal)**: ``api_key`` as a Bearer token (existing),
  or Azure AD client credentials (``tenant_id`` + ``client_id`` + ``client_secret``)
  with scope ``https://analysis.windows.net/powerbi/api/.default``.
  Also accepts ``POWERBI_API_KEY`` or ``POWERBI_TENANT_ID`` / ``POWERBI_CLIENT_ID`` /
  ``POWERBI_CLIENT_SECRET``.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration

_BASE_URL = "https://api.powerbi.com/v1.0/myorg"
_POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    text = str(value).strip()
    return text or None


def _datasets_url(workspace_id: str | None, suffix: str = "") -> str:
    if workspace_id:
        return f"{_BASE_URL}/groups/{workspace_id}/datasets{suffix}"
    return f"{_BASE_URL}/datasets{suffix}"


def _reports_url(workspace_id: str | None, suffix: str = "") -> str:
    if workspace_id:
        return f"{_BASE_URL}/groups/{workspace_id}/reports{suffix}"
    return f"{_BASE_URL}/reports{suffix}"


async def _get_token_from_client_credentials(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Obtain an app-only token via the Azure AD client-credentials flow."""
    import httpx

    token_url = _TOKEN_URL.format(tenant_id=tenant_id)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": _POWERBI_SCOPE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError("Power BI service principal token response did not include access_token.")
        return str(token)


def _service_principal_parts(tool: Any, creds: dict[str, Any]) -> tuple[str, str, str] | None:
    tenant_id = (
        _secret_value(creds.get("tenant_id"))
        or _secret_value(getattr(tool, "tenant_id", None))
        or _secret_value(os.getenv("POWERBI_TENANT_ID"))
    )
    client_id = (
        _secret_value(creds.get("client_id"))
        or _secret_value(getattr(tool, "client_id", None))
        or _secret_value(os.getenv("POWERBI_CLIENT_ID"))
    )
    client_secret = (
        _secret_value(creds.get("client_secret"))
        or _secret_value(getattr(tool, "client_secret", None))
        or _secret_value(os.getenv("POWERBI_CLIENT_SECRET"))
    )
    if tenant_id and client_id and client_secret:
        return tenant_id, client_id, client_secret
    return None


async def _resolve_token(tool: Any) -> str:
    """Return a Power BI REST Bearer token (user-delegated OAuth or app-delegated)."""
    explicit_key = _secret_value(getattr(tool, "api_key", None))
    if explicit_key:
        return explicit_key
    explicit_token = _secret_value(getattr(tool, "token", None))
    if explicit_token:
        return explicit_token

    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    oauth = _secret_value(creds.get("token")) or _secret_value(creds.get("access_token"))
    if oauth:
        return oauth

    api_key = _secret_value(creds.get("api_key"))
    if api_key:
        return api_key

    sp = _service_principal_parts(tool, creds)
    if sp:
        return await _get_token_from_client_credentials(*sp)

    env_key = _secret_value(os.getenv("POWERBI_API_KEY"))
    if env_key:
        return env_key
    env_oauth = _secret_value(os.getenv("POWERBI_ACCESS_TOKEN"))
    if env_oauth:
        return env_oauth

    raise CredentialNotAvailable(
        "PowerBI",
        missing=["token", "api_key"],
        env_vars=[
            "POWERBI_ACCESS_TOKEN",
            "POWERBI_API_KEY",
            "POWERBI_TENANT_ID",
            "POWERBI_CLIENT_ID",
            "POWERBI_CLIENT_SECRET",
        ],
    )


def _raise_for_execute_queries(response: Any) -> None:
    """Map the documented SP + RLS 401 to a message the agent can act on."""
    if getattr(response, "status_code", None) != 401:
        response.raise_for_status()
        return
    snippet = (getattr(response, "text", None) or "")[:200]
    hint = (
        "Power BI executeQueries returned 401. Service principals cannot query "
        "semantic models with RLS (PowerBINotAuthorizedException); impersonatedUserName "
        "does not bypass that. Use a user-delegated OAuth integration "
        "(Dataset.Read.All + Workspace.Read.All)."
    )
    raise ValueError(f"{hint} Upstream: {snippet}" if snippet else hint)


class _PowerBITool(Tool):
    """Shared auth fields for Power BI tools (OAuth user-delegated + service principal)."""

    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None
    token: SecretStr | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_key": self.api_key,
                    "token": self.token,
                    "tenant_id": self.tenant_id,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            ),
        }


class PowerBIListWorkspaces(_PowerBITool):
    name: str = "powerbi_list_workspaces"
    description: str | None = "List all Power BI workspaces (groups) the authenticated user has access to."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspaces(
            top: int = Field(100, description="Number of workspaces to return (max 5000)."),
            skip: int = Field(0, description="Number of workspaces to skip for pagination."),
            filter: str | None = Field(None, description="OData $filter expression, e.g. 'type eq 'Workspace''."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"$top": top, "$skip": skip}
            if filter:
                params["$filter"] = filter

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_BASE_URL}/groups",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_workspaces, **kwargs)


class PowerBIListDatasets(_PowerBITool):
    name: str = "powerbi_list_datasets"
    description: str | None = "List Power BI datasets in a workspace or in My Workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_datasets(
            workspace_id: str | None = Field(
                None, description="Power BI workspace (group) ID. If omitted, lists datasets in My Workspace."
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _datasets_url(workspace_id),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_datasets, **kwargs)


class PowerBIGetDataset(_PowerBITool):
    name: str = "powerbi_get_dataset"
    description: str | None = "Get metadata for a specific Power BI dataset."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_dataset(
            dataset_id: str = Field(..., description="Power BI dataset ID"),
            workspace_id: str | None = Field(
                None, description="Power BI workspace (group) ID. If omitted, lists datasets in My Workspace."
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _datasets_url(workspace_id, f"/{dataset_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_dataset, **kwargs)


class PowerBIQueryDataset(_PowerBITool):
    name: str = "powerbi_query_dataset"
    description: str | None = (
        "Execute a DAX query against a Power BI dataset and return the results. "
        "Datasets with RLS require user-delegated OAuth; service principal always 401s."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _query_dataset(
            dataset_id: str = Field(..., description="Power BI dataset ID"),
            query: str = Field(
                ...,
                description="DAX query string, e.g. EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"Total\",[Total Sales])",
            ),
            workspace_id: str | None = Field(
                None, description="Power BI workspace (group) ID. If omitted, uses My Workspace."
            ),
            impersonated_user_name: str | None = Field(
                None,
                description=(
                    "UPN to impersonate. Ignored on models without RLS. Does not work with a "
                    "service principal on RLS models (API returns 401); use user-delegated OAuth instead."
                ),
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {"queries": [{"query": query}]}
            if impersonated_user_name:
                body["impersonatedUserName"] = impersonated_user_name

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    _datasets_url(workspace_id, f"/{dataset_id}/executeQueries"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                _raise_for_execute_queries(response)
                return response.json()

        super().__init__(handler=_query_dataset, **kwargs)


class PowerBIListReports(_PowerBITool):
    name: str = "powerbi_list_reports"
    description: str | None = "List Power BI reports in a workspace or in My Workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_reports(
            workspace_id: str | None = Field(
                None, description="Power BI workspace (group) ID. If omitted, lists reports in My Workspace."
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _reports_url(workspace_id),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_reports, **kwargs)


class PowerBIGetReport(_PowerBITool):
    name: str = "powerbi_get_report"
    description: str | None = "Get metadata for a specific Power BI report, including its embed URL."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_report(
            report_id: str = Field(..., description="Power BI report ID"),
            workspace_id: str | None = Field(
                None, description="Power BI workspace (group) ID. If omitted, uses My Workspace."
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _reports_url(workspace_id, f"/{report_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_report, **kwargs)
