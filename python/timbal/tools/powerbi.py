from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_BASE_URL = "https://api.powerbi.com/v1.0/myorg"


def _datasets_url(workspace_id: str | None, suffix: str = "") -> str:
    if workspace_id:
        return f"{_BASE_URL}/groups/{workspace_id}/datasets{suffix}"
    return f"{_BASE_URL}/datasets{suffix}"


def _reports_url(workspace_id: str | None, suffix: str = "") -> str:
    if workspace_id:
        return f"{_BASE_URL}/groups/{workspace_id}/reports{suffix}"
    return f"{_BASE_URL}/reports{suffix}"


class PowerBIListWorkspaces(Tool):
    name: str = "powerbi_list_workspaces"
    description: str | None = "List all Power BI workspaces (groups) the authenticated user has access to."
    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspaces(
            top: int = Field(100, description="Number of workspaces to return (max 5000)."),
            skip: int = Field(0, description="Number of workspaces to skip for pagination."),
            filter: str | None = Field(None, description="OData $filter expression, e.g. 'type eq 'Workspace''."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="PowerBI", env_var="POWERBI_API_KEY")
            import httpx

            params: dict[str, Any] = {"$top": top, "$skip": skip}
            if filter:
                params["$filter"] = filter

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_BASE_URL}/groups",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_workspaces, **kwargs)


class PowerBIListDatasets(Tool):
    name: str = "powerbi_list_datasets"
    description: str | None = "List Power BI datasets in a workspace or in My Workspace."
    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_datasets(
            workspace_id: str | None = Field(None, description="Power BI workspace (group) ID. If omitted, lists datasets in My Workspace.")
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="PowerBI", env_var="POWERBI_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _datasets_url(workspace_id),
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_datasets, **kwargs)


class PowerBIGetDataset(Tool):
    name: str = "powerbi_get_dataset"
    description: str | None = "Get metadata for a specific Power BI dataset."
    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_dataset(
            dataset_id: str = Field(..., description="Power BI dataset ID"),
            workspace_id: str | None = Field(None, description="Power BI workspace (group) ID. If omitted, lists datasets in My Workspace."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="PowerBI", env_var="POWERBI_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _datasets_url(workspace_id, f"/{dataset_id}"),
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_dataset, **kwargs)


class PowerBIQueryDataset(Tool):
    name: str = "powerbi_query_dataset"
    description: str | None = "Execute a DAX query against a Power BI dataset and return the results."
    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query_dataset(
            dataset_id: str = Field(..., description="Power BI dataset ID"),
            query: str = Field(..., description="DAX query string, e.g. EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"Total\",[Total Sales])"),
            workspace_id: str | None = Field(None, description="Power BI workspace (group) ID. If omitted, uses My Workspace."),
            impersonated_user_name: str | None = Field(None, description="UPN of a user to impersonate for row-level security (RLS)."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="PowerBI", env_var="POWERBI_API_KEY")
            import httpx

            body: dict[str, Any] = {"queries": [{"query": query}]}
            if impersonated_user_name:
                body["impersonatedUserName"] = impersonated_user_name

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    _datasets_url(workspace_id, f"/{dataset_id}/executeQueries"),
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_query_dataset, **kwargs)


class PowerBIListReports(Tool):
    name: str = "powerbi_list_reports"
    description: str | None = "List Power BI reports in a workspace or in My Workspace."
    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_reports(workspace_id: str | None = Field(None, description="Power BI workspace (group) ID. If omitted, lists reports in My Workspace.")) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="PowerBI", env_var="POWERBI_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _reports_url(workspace_id),
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_reports, **kwargs)


class PowerBIGetReport(Tool):
    name: str = "powerbi_get_report"
    description: str | None = "Get metadata for a specific Power BI report, including its embed URL."
    integration: Annotated[str, Integration("powerbi")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_report(
            report_id: str = Field(..., description="Power BI report ID"),
            workspace_id: str | None = Field(None, description="Power BI workspace (group) ID. If omitted, uses My Workspace."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="PowerBI", env_var="POWERBI_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    _reports_url(workspace_id, f"/{report_id}"),
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_report, **kwargs)
