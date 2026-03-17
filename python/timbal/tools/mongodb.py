import os
from typing import Annotated, Any

import httpx
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


def _atlas_url(app_id: str, action: str) -> str:
    return f"https://data.mongodb-api.com/app/{app_id}/endpoint/data/v1/action/{action}"


def _base_body(data_source: str, database: str, collection: str) -> dict[str, Any]:
    return {"dataSource": data_source, "database": database, "collection": collection}


async def _resolve_api_key(tool: Any) -> str:
    """Resolve MongoDB Data API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("MONGODB_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "MongoDB API key not found. Set MONGODB_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class CreateDocument(Tool):
    name: str = "mongodb_create_document"
    description: str | None = "Insert a new document into a MongoDB collection."
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_document(
            app_id: str = Field(..., description="Atlas App Services App ID (found in Atlas Data API settings)"),
            data_source: str = Field(..., description="Atlas cluster name, e.g. Cluster0"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            document: dict[str, Any] = Field(..., description="Document to insert as a JSON object"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body = {**_base_body(data_source, database, collection), "document": document}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "insertOne"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_document, **kwargs)


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------


class FindDocumentById(Tool):
    name: str = "mongodb_find_document_by_id"
    description: str | None = "Retrieve a single MongoDB document by its _id."
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_document_by_id(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            document_id: str = Field(..., description="String value of the document's _id field"),
            projection: dict[str, Any] | None = Field(
                None, description="Optional fields to include/exclude, e.g. {'name': 1, 'email': 1, '_id': 0}"
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": {"_id": {"$oid": document_id}},
            }
            if projection:
                body["projection"] = projection
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "findOne"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_find_document_by_id, **kwargs)


class FindDocument(Tool):
    name: str = "mongodb_find_document"
    description: str | None = "Find the first document in a MongoDB collection matching a query filter."
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_document(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            query_filter: dict[str, Any] = Field(
                ..., description="MongoDB query filter, e.g. {'status': 'active'} or {'age': {'$gt': 18}}"
            ),
            projection: dict[str, Any] | None = Field(None, description="Optional fields to include/exclude"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": query_filter,
            }
            if projection:
                body["projection"] = projection
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "findOne"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_find_document, **kwargs)


class SearchDocuments(Tool):
    name: str = "mongodb_search_documents"
    description: str | None = (
        "Search for documents in a MongoDB collection. "
        "Pass an empty filter {} to return all documents."
    )
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_documents(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            query_filter: dict[str, Any] | None = Field(
                None,
                description="MongoDB query filter. Pass {} or omit to return all documents.",
            ),
            projection: dict[str, Any] | None = Field(None, description="Fields to include/exclude"),
            sort: dict[str, Any] | None = Field(
                None,
                description="Sort order, e.g. {'createdAt': -1} (descending), {'name': 1} (ascending)",
            ),
            limit: int = Field(100, description="Max number of documents to return"),
            skip: int = Field(0, description="Number of documents to skip for pagination"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": query_filter or {},
                "limit": limit,
                "skip": skip,
            }
            if projection:
                body["projection"] = projection
            if sort:
                body["sort"] = sort
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "find"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_documents, **kwargs)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class UpdateDocument(Tool):
    name: str = "mongodb_update_document"
    description: str | None = "Update a single MongoDB document by its _id."
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_document(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            document_id: str = Field(..., description="String value of the document's _id field"),
            update: dict[str, Any] = Field(
                ...,
                description="MongoDB update operators, e.g. {'$set': {'status': 'done'}}",
            ),
            upsert: bool = Field(False, description="If True, creates the document if it does not exist"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": {"_id": {"$oid": document_id}},
                "update": update,
                "upsert": upsert,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "updateOne"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_document, **kwargs)


class UpdateDocuments(Tool):
    name: str = "mongodb_update_documents"
    description: str | None = "Update all MongoDB documents matching a query filter."
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_documents(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            query_filter: dict[str, Any] = Field(
                ...,
                description="MongoDB query filter to select documents to update",
            ),
            update: dict[str, Any] = Field(
                ...,
                description="MongoDB update operators, e.g. {'$set': {'status': 'processed'}}",
            ),
            upsert: bool = Field(False, description="If True, creates a document if none matched the filter"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": query_filter,
                "update": update,
                "upsert": upsert,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "updateMany"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_documents, **kwargs)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class DeleteDocument(Tool):
    name: str = "mongodb_delete_document"
    description: str | None = "Delete a single MongoDB document by its _id."
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_document(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            document_id: str = Field(..., description="String value of the document's _id field to delete"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": {"_id": {"$oid": document_id}},
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "deleteOne"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_document, **kwargs)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class ExecuteAggregation(Tool):
    name: str = "mongodb_execute_aggregation"
    description: str | None = (
        "Execute an aggregation pipeline on a MongoDB collection. "
        "Supports $match, $group, $sort, $project, $lookup, $unwind, $limit, $skip, and more."
    )
    integration: Annotated[str, Integration("mongodb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _execute_aggregation(
            app_id: str = Field(..., description="Atlas App Services App ID"),
            data_source: str = Field(..., description="Atlas cluster name"),
            database: str = Field(..., description="Database name"),
            collection: str = Field(..., description="Collection name"),
            pipeline: list[dict[str, Any]] = Field(
                ...,
                description="List of aggregation stage documents, e.g. [{'$group': {'_id': '$status', 'count': {'$sum': 1}}}]",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "pipeline": pipeline,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "aggregate"),
                    headers={"api-key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_execute_aggregation, **kwargs)
