from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_POWERBI_API_BASE = "https://api.powerbi.com/v1.0/myorg"


def _datasets_url(workspace_id: str | None, suffix: str = "") -> str:
    if workspace_id:
        return f"{_POWERBI_API_BASE}/groups/{workspace_id}/datasets{suffix}"
    return f"{_POWERBI_API_BASE}/datasets{suffix}"


def _reports_url(workspace_id: str | None, suffix: str = "") -> str:
    if workspace_id:
        return f"{_POWERBI_API_BASE}/groups/{workspace_id}/reports{suffix}"
    return f"{_POWERBI_API_BASE}/reports{suffix}"


class ListWorkspaces(Tool):
    name: str = "powerbi_list_workspaces"
    description: str | None = "List all Power BI workspaces (groups) the authenticated user has access to."
    integration: Annotated[str, Integration("powerbi")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspaces(
            top: int = 100,
            skip: int = 0,
            filter: str | None = None,
        ) -> Any:
            """
            top: number of workspaces to return (max 5000).
            skip: number of workspaces to skip for pagination.
            filter: OData $filter expression, e.g. "type eq 'Workspace'".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"$top": top, "$skip": skip}
            if filter:
                params["$filter"] = filter

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_POWERBI_API_BASE}/groups",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PowerBI/ListWorkspaces"

        super().__init__(handler=_list_workspaces, metadata=metadata, **kwargs)


class ListDatasets(Tool):
    name: str = "powerbi_list_datasets"
    description: str | None = "List Power BI datasets in a workspace or in My Workspace."
    integration: Annotated[str, Integration("powerbi")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_datasets(workspace_id: str | None = None) -> Any:
            """
            workspace_id: Power BI workspace (group) ID.
                          If omitted, lists datasets in My Workspace.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _datasets_url(workspace_id),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PowerBI/ListDatasets"

        super().__init__(handler=_list_datasets, metadata=metadata, **kwargs)


class GetDataset(Tool):
    name: str = "powerbi_get_dataset"
    description: str | None = "Get metadata for a specific Power BI dataset."
    integration: Annotated[str, Integration("powerbi")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_dataset(
            dataset_id: str,
            workspace_id: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _datasets_url(workspace_id, f"/{dataset_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PowerBI/GetDataset"

        super().__init__(handler=_get_dataset, metadata=metadata, **kwargs)


class QueryDataset(Tool):
    name: str = "powerbi_query_dataset"
    description: str | None = "Execute a DAX query against a Power BI dataset and return the results."
    integration: Annotated[str, Integration("powerbi")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query_dataset(
            dataset_id: str,
            query: str,
            workspace_id: str | None = None,
            impersonated_user_name: str | None = None,
        ) -> Any:
            """
            query: DAX query string to execute against the dataset.
              Example: "EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \\"Total\\",[Total Sales])"
            impersonated_user_name: UPN of a user to impersonate for row-level security (RLS).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"queries": [{"query": query}]}
            if impersonated_user_name:
                body["impersonatedUserName"] = impersonated_user_name

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _datasets_url(workspace_id, f"/{dataset_id}/executeQueries"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PowerBI/QueryDataset"

        super().__init__(handler=_query_dataset, metadata=metadata, **kwargs)


class ListReports(Tool):
    name: str = "powerbi_list_reports"
    description: str | None = "List Power BI reports in a workspace or in My Workspace."
    integration: Annotated[str, Integration("powerbi")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_reports(workspace_id: str | None = None) -> Any:
            """
            workspace_id: Power BI workspace (group) ID.
                          If omitted, lists reports in My Workspace.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _reports_url(workspace_id),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PowerBI/ListReports"

        super().__init__(handler=_list_reports, metadata=metadata, **kwargs)


class GetReport(Tool):
    name: str = "powerbi_get_report"
    description: str | None = "Get metadata for a specific Power BI report, including its embed URL."
    integration: Annotated[str, Integration("powerbi")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_report(
            report_id: str,
            workspace_id: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _reports_url(workspace_id, f"/{report_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PowerBI/GetReport"

        super().__init__(handler=_get_report, metadata=metadata, **kwargs)
