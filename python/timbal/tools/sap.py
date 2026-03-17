import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_api_key(tool: Any) -> str:
    """Resolve SAP API key/token from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        if isinstance(credentials, dict):
            key = credentials.get("api_key") or credentials.get("token")
        else:
            key = getattr(credentials, "api_key", None) or getattr(credentials, "token", None)
        if key:
            return str(key)
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("SAP_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "SAP API key not found. Set SAP_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


# ---------------------------------------------------------------------------
# Get Entities (list)
# ---------------------------------------------------------------------------


class SAPGetEntities(Tool):
    name: str = "sap_get_entities"
    description: str | None = (
        "Retrieve a list of entities from an SAP entity set. "
        "Supports Financial Item, Financial Planning Context, Team Project, Work Package, and other OData entity sets."
    )
    integration: Annotated[str, Integration("sap")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _sap_get_entities(
            base_url: str = Field(
                ...,
                description="Base URL of your SAP OData/REST API, e.g. https://your-system.s4hana.cloud.sap",
            ),
            entity_set: str = Field(
                ...,
                description="OData entity set name, e.g. 'FinancialItemConsumer', 'FinancialPlanningContext', 'TeamProjectConsumer', 'WorkPackageConsumer'",
            ),
            select: str | None = Field(
                None,
                description="Comma-separated properties to return, e.g. 'Id,Name,Status'. Omit for all properties.",
            ),
            filter_query: str | None = Field(
                None,
                description="OData $filter expression, e.g. \"Status eq 'Active'\" or \"Amount gt 100\"",
            ),
            order_by: str | None = Field(
                None,
                description="OData $orderby expression, e.g. 'CreatedAt desc' or 'Name asc'",
            ),
            top: int = Field(100, description="Maximum number of entities to return (OData $top)"),
            skip: int = Field(0, description="Number of entities to skip for pagination (OData $skip)"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            url = f"{base_url.rstrip('/')}/{entity_set}"
            params: dict[str, Any] = {"$top": top, "$skip": skip}
            if select:
                params["$select"] = select
            if filter_query:
                params["$filter"] = filter_query
            if order_by:
                params["$orderby"] = order_by

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_sap_get_entities, **kwargs)


# ---------------------------------------------------------------------------
# Get Entity (single by ID)
# ---------------------------------------------------------------------------


class SAPGetEntity(Tool):
    name: str = "sap_get_entity"
    description: str | None = (
        "Retrieve a single entity from an SAP entity set by its ID. "
        "Supports Financial Item, Financial Planning Context, Team Project, Work Package, and other OData entity sets."
    )
    integration: Annotated[str, Integration("sap")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _sap_get_entity(
            base_url: str = Field(
                ...,
                description="Base URL of your SAP OData/REST API",
            ),
            entity_set: str = Field(
                ...,
                description="OData entity set name, e.g. 'FinancialItemConsumer', 'TeamProjectConsumer'",
            ),
            entity_id: str = Field(
                ...,
                description="The unique identifier (ID) of the entity to retrieve",
            ),
            select: str | None = Field(
                None,
                description="Comma-separated properties to return. Omit for all properties.",
            ),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            url = f"{base_url.rstrip('/')}/{entity_set}('{entity_id}')"
            params: dict[str, Any] = {}
            if select:
                params["$select"] = select

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_sap_get_entity, **kwargs)


# ---------------------------------------------------------------------------
# Search in SAP
# ---------------------------------------------------------------------------


class SAPSearch(Tool):
    name: str = "sap_search"
    description: str | None = (
        "Search across SAP entities. Use a free-text query to find relevant entities "
        "in Financial Items, Financial Planning, Team Projects, Work Packages, and other SAP data."
    )
    integration: Annotated[str, Integration("sap")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _sap_search(
            base_url: str = Field(
                ...,
                description="Base URL of your SAP search API endpoint",
            ),
            query: str = Field(
                ...,
                description="Search query to find entities in SAP",
            ),
            entity_types: list[str] | None = Field(
                None,
                description="Optional list of entity types to search, e.g. ['FinancialItem', 'TeamProject']",
            ),
            limit: int = Field(20, description="Maximum number of results to return"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx
            
            url = base_url.rstrip("/")
            payload: dict[str, Any] = {"query": query, "limit": limit}
            if entity_types:
                payload["entity_types"] = entity_types

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_sap_search, **kwargs)
