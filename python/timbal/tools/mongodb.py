from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration


def _atlas_url(app_id: str, action: str) -> str:
    return f"https://data.mongodb-api.com/app/{app_id}/endpoint/data/v1/action/{action}"


def _base_body(data_source: str, database: str, collection: str) -> dict[str, Any]:
    return {"dataSource": data_source, "database": database, "collection": collection}


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class CreateDocument(Tool):
    name: str = "mongodb_create_document"
    description: str | None = "Insert a new document into a MongoDB collection."
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_document(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            document: dict[str, Any],
        ) -> Any:
            """
            app_id: Atlas App Services App ID (found in the Atlas Data API settings).
            data_source: the Atlas cluster name, e.g. "Cluster0".
            database: the database name.
            collection: the collection name.
            document: the document to insert as a JSON object.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body = {**_base_body(data_source, database, collection), "document": document}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "insertOne"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/CreateDocument"
        super().__init__(handler=_create_document, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------


class FindDocumentById(Tool):
    name: str = "mongodb_find_document_by_id"
    description: str | None = "Retrieve a single MongoDB document by its _id."
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_document_by_id(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            document_id: str,
            projection: dict[str, Any] | None = None,
        ) -> Any:
            """
            document_id: the string value of the document's _id field.
            projection: optional fields to include/exclude, e.g. {"name": 1, "email": 1, "_id": 0}.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": {"_id": {"$oid": document_id}},
            }
            if projection:
                body["projection"] = projection

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "findOne"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/FindDocumentById"
        super().__init__(handler=_find_document_by_id, metadata=metadata, **kwargs)


class FindDocument(Tool):
    name: str = "mongodb_find_document"
    description: str | None = "Find the first document in a MongoDB collection matching a query filter."
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_document(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            query_filter: dict[str, Any],
            projection: dict[str, Any] | None = None,
        ) -> Any:
            """
            query_filter: MongoDB query filter, e.g. {"status": "active"} or
                          {"age": {"$gt": 18}, "country": "US"}.
            projection: optional fields to include/exclude.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": query_filter,
            }
            if projection:
                body["projection"] = projection

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "findOne"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/FindDocument"
        super().__init__(handler=_find_document, metadata=metadata, **kwargs)


class SearchDocuments(Tool):
    name: str = "mongodb_search_documents"
    description: str | None = (
        "Search for documents in a MongoDB collection. "
        "Pass an empty filter {} to return all documents."
    )
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_documents(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            query_filter: dict[str, Any] | None = None,
            projection: dict[str, Any] | None = None,
            sort: dict[str, Any] | None = None,
            limit: int = 100,
            skip: int = 0,
        ) -> Any:
            """
            query_filter: MongoDB query filter. Pass {} or omit to return all documents.
                          Examples: {"status": "active"}, {"price": {"$lt": 50}},
                          {"$or": [{"city": "NYC"}, {"city": "LA"}]}.
            projection: fields to include/exclude, e.g. {"name": 1, "_id": 0}.
            sort: sort order, e.g. {"createdAt": -1} (descending), {"name": 1} (ascending).
            limit: max number of documents to return (default 100).
            skip: number of documents to skip for pagination.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/SearchDocuments"
        super().__init__(handler=_search_documents, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class UpdateDocument(Tool):
    name: str = "mongodb_update_document"
    description: str | None = "Update a single MongoDB document by its _id."
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_document(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            document_id: str,
            update: dict[str, Any],
            upsert: bool = False,
        ) -> Any:
            """
            document_id: the string value of the document's _id field.
            update: MongoDB update operators, e.g. {"$set": {"status": "done"}} or
                    {"$inc": {"views": 1}, "$set": {"updatedAt": "2024-01-01"}}.
            upsert: if True, creates the document if it does not exist.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": {"_id": {"$oid": document_id}},
                "update": update,
                "upsert": upsert,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "updateOne"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/UpdateDocument"
        super().__init__(handler=_update_document, metadata=metadata, **kwargs)


class UpdateDocuments(Tool):
    name: str = "mongodb_update_documents"
    description: str | None = "Update all MongoDB documents matching a query filter."
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_documents(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            query_filter: dict[str, Any],
            update: dict[str, Any],
            upsert: bool = False,
        ) -> Any:
            """
            query_filter: MongoDB query filter to select documents to update,
                          e.g. {"status": "pending"}.
            update: MongoDB update operators, e.g. {"$set": {"status": "processed"}}.
            upsert: if True, creates a document if none matched the filter.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": query_filter,
                "update": update,
                "upsert": upsert,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "updateMany"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/UpdateDocuments"
        super().__init__(handler=_update_documents, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class DeleteDocument(Tool):
    name: str = "mongodb_delete_document"
    description: str | None = "Delete a single MongoDB document by its _id."
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_document(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            document_id: str,
        ) -> Any:
            """
            document_id: the string value of the document's _id field to delete.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "filter": {"_id": {"$oid": document_id}},
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "deleteOne"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/DeleteDocument"
        super().__init__(handler=_delete_document, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class ExecuteAggregation(Tool):
    name: str = "mongodb_execute_aggregation"
    description: str | None = (
        "Execute an aggregation pipeline on a MongoDB collection. "
        "Supports $match, $group, $sort, $project, $lookup, $unwind, $limit, $skip, and more."
    )
    integration: Annotated[str, Integration("mongodb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _execute_aggregation(
            app_id: str,
            data_source: str,
            database: str,
            collection: str,
            pipeline: list[dict[str, Any]],
        ) -> Any:
            """
            pipeline: list of aggregation stage documents. Examples:
              Count by status:
                [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
              Filter then project:
                [{"$match": {"active": true}}, {"$project": {"name": 1, "email": 1}}]
              Join collections:
                [{"$lookup": {"from": "orders", "localField": "_id",
                              "foreignField": "userId", "as": "orders"}}]
              Top 5 by revenue:
                [{"$sort": {"revenue": -1}}, {"$limit": 5}]
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                **_base_body(data_source, database, collection),
                "pipeline": pipeline,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _atlas_url(app_id, "aggregate"),
                    headers={"api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "MongoDB/ExecuteAggregation"
        super().__init__(handler=_execute_aggregation, metadata=metadata, **kwargs)
