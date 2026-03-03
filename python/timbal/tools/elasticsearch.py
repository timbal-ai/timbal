from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration


class IngestAttachment(Tool):
    name: str = "elasticsearch_ingest_attachment"
    description: str | None = (
        "Ingest a base64-encoded file attachment into an Elasticsearch index "
        "using the ingest-attachment pipeline to extract text content."
    )
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _ingest_attachment(
            host: str,
            index: str,
            document_id: str,
            data: str,
            pipeline: str = "attachment",
            fields: dict[str, Any] | None = None,
        ) -> Any:
            """
            host: Elasticsearch base URL, e.g. "https://my-cluster.es.io:9200"
            data: base64-encoded content of the file to ingest.
            pipeline: ingest pipeline name (default "attachment" requires the ingest-attachment plugin).
            fields: additional document fields to store alongside the attachment.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"data": data}
            if fields:
                body.update(fields)

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{host}/{index}/_doc/{document_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"pipeline": pipeline},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/IngestAttachment"

        super().__init__(handler=_ingest_attachment, metadata=metadata, **kwargs)


class BulkOperations(Tool):
    name: str = "elasticsearch_bulk_operations"
    description: str | None = (
        "Execute multiple index, create, update, or delete operations in a single request."
    )
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_operations(
            host: str,
            operations: list[dict[str, Any]],
            index: str | None = None,
            refresh: str = "false",
        ) -> Any:
            """
            operations: list of action/source pairs. Each pair is two dicts:
              - action: {"index"|"create"|"update"|"delete": {"_index": ..., "_id": ...}}
              - source (omitted for delete): the document body or update payload
            Example:
              [
                {"index": {"_index": "my-index", "_id": "1"}},
                {"field": "value"},
                {"delete": {"_index": "my-index", "_id": "2"}}
              ]
            index: default index for operations that don't specify one.
            refresh: "true", "false", or "wait_for".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            ndjson = "\n".join(
                __import__("json").dumps(op) for op in operations
            ) + "\n"

            url = f"{host}/_bulk" if not index else f"{host}/{index}/_bulk"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/x-ndjson",
                    },
                    content=ndjson.encode(),
                    params={"refresh": refresh},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/BulkOperations"

        super().__init__(handler=_bulk_operations, metadata=metadata, **kwargs)


class DeleteDocument(Tool):
    name: str = "elasticsearch_delete_document"
    description: str | None = "Delete a document from an Elasticsearch index by ID."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_document(
            host: str,
            index: str,
            document_id: str,
            refresh: str = "false",
        ) -> Any:
            """
            refresh: "true", "false", or "wait_for".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{host}/{index}/_doc/{document_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"refresh": refresh},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/DeleteDocument"

        super().__init__(handler=_delete_document, metadata=metadata, **kwargs)


class GetDocument(Tool):
    name: str = "elasticsearch_get_document"
    description: str | None = "Retrieve a document from an Elasticsearch index by ID."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_document(
            host: str,
            index: str,
            document_id: str,
            source_includes: list[str] | None = None,
            source_excludes: list[str] | None = None,
        ) -> Any:
            """
            source_includes: list of fields to include in _source.
            source_excludes: list of fields to exclude from _source.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {}
            if source_includes:
                params["_source_includes"] = ",".join(source_includes)
            if source_excludes:
                params["_source_excludes"] = ",".join(source_excludes)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{host}/{index}/_doc/{document_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/GetDocument"

        super().__init__(handler=_get_document, metadata=metadata, **kwargs)


class IndexDocument(Tool):
    name: str = "elasticsearch_index_document"
    description: str | None = "Index (create or replace) a document in an Elasticsearch index."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _index_document(
            host: str,
            index: str,
            document: dict[str, Any],
            document_id: str | None = None,
            refresh: str = "false",
        ) -> Any:
            """
            document_id: if provided, uses PUT (upsert by ID); if omitted, uses POST (auto-generate ID).
            refresh: "true", "false", or "wait_for".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                if document_id:
                    response = await client.put(
                        f"{host}/{index}/_doc/{document_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"refresh": refresh},
                        json=document,
                    )
                else:
                    response = await client.post(
                        f"{host}/{index}/_doc",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"refresh": refresh},
                        json=document,
                    )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/IndexDocument"

        super().__init__(handler=_index_document, metadata=metadata, **kwargs)


class UpdateDocument(Tool):
    name: str = "elasticsearch_update_document"
    description: str | None = "Partially update a document in an Elasticsearch index."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_document(
            host: str,
            index: str,
            document_id: str,
            doc: dict[str, Any] | None = None,
            script: dict[str, Any] | None = None,
            upsert: dict[str, Any] | None = None,
            refresh: str = "false",
        ) -> Any:
            """
            doc: partial document fields to merge into the existing document.
            script: Painless script to run, e.g. {"source": "ctx._source.counter += params.count", "params": {"count": 1}}
            upsert: document to create if the ID does not exist (used with script).
            Provide either doc or script, not both.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {}
            if doc is not None:
                body["doc"] = doc
            if script is not None:
                body["script"] = script
            if upsert is not None:
                body["upsert"] = upsert

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/{index}/_update/{document_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"refresh": refresh},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/UpdateDocument"

        super().__init__(handler=_update_document, metadata=metadata, **kwargs)


class CreateIndex(Tool):
    name: str = "elasticsearch_create_index"
    description: str | None = "Create a new Elasticsearch index with optional mappings and settings."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_index(
            host: str,
            index: str,
            mappings: dict[str, Any] | None = None,
            settings: dict[str, Any] | None = None,
            aliases: dict[str, Any] | None = None,
        ) -> Any:
            """
            mappings: field type definitions, e.g. {"properties": {"title": {"type": "text"}}}.
            settings: index settings, e.g. {"number_of_shards": 1, "number_of_replicas": 1}.
            aliases: index aliases to create alongside the index.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {}
            if mappings:
                body["mappings"] = mappings
            if settings:
                body["settings"] = settings
            if aliases:
                body["aliases"] = aliases

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{host}/{index}",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/CreateIndex"

        super().__init__(handler=_create_index, metadata=metadata, **kwargs)


class UpdateIndexSettings(Tool):
    name: str = "elasticsearch_update_index_settings"
    description: str | None = "Update dynamic settings on an existing Elasticsearch index."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_index_settings(
            host: str,
            index: str,
            settings: dict[str, Any],
        ) -> Any:
            """
            settings: dynamic index settings to update.
            Example: {"number_of_replicas": 2, "refresh_interval": "30s"}
            Note: static settings (e.g. number_of_shards) cannot be changed after index creation.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{host}/{index}/_settings",
                    headers={"Authorization": f"Bearer {token}"},
                    json=settings,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/UpdateIndexSettings"

        super().__init__(handler=_update_index_settings, metadata=metadata, **kwargs)


class SearchDocuments(Tool):
    name: str = "elasticsearch_search_documents"
    description: str | None = "Search documents in one or more Elasticsearch indices using a query DSL."
    integration: Annotated[str, Integration("elasticsearch")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_documents(
            host: str,
            index: str,
            query: dict[str, Any] | None = None,
            size: int = 10,
            from_: int = 0,
            sort: list[dict[str, Any]] | None = None,
            source: list[str] | bool | None = None,
            aggs: dict[str, Any] | None = None,
            highlight: dict[str, Any] | None = None,
            knn: dict[str, Any] | None = None,
        ) -> Any:
            """
            index: index name or comma-separated list, e.g. "my-index" or "logs-*".
            query: Elasticsearch Query DSL, e.g.:
              {"match": {"title": "search term"}}
              {"bool": {"must": [{"match": {"status": "active"}}]}}
              {"match_all": {}}
            sort: sort criteria, e.g. [{"date": "desc"}, "_score"].
            source: list of fields to return, True/False to include/exclude all.
            aggs: aggregation definitions.
            highlight: highlight configuration for matched fields.
            knn: k-nearest neighbor search config for vector similarity search.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"size": size, "from": from_}
            if query:
                body["query"] = query
            if sort:
                body["sort"] = sort
            if source is not None:
                body["_source"] = source
            if aggs:
                body["aggs"] = aggs
            if highlight:
                body["highlight"] = highlight
            if knn:
                body["knn"] = knn

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/{index}/_search",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Elasticsearch/SearchDocuments"

        super().__init__(handler=_search_documents, metadata=metadata, **kwargs)
