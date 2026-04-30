import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_credentials(tool: Any) -> tuple[str, str | None]:
    """Resolve (api_key, endpoint) from integration, explicit fields, or env vars."""
    endpoint: str | None = None
    api_key: str
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        api_key = credentials["api_key"]
        endpoint = credentials["endpoint"]
    elif tool.api_key is not None:
        api_key = tool.api_key.get_secret_value()
    else:
        env_key = os.getenv("ELASTICSEARCH_API_KEY")
        if env_key:
            api_key = env_key
        else:
            raise ValueError(
                "Elasticsearch API key not found. Set ELASTICSEARCH_API_KEY, pass api_key, or configure an integration."
            )
    if endpoint is None:
        endpoint = os.getenv("ELASTICSEARCH_HOST")
    return api_key, endpoint


class ElasticsearchIngestAttachment(Tool):
    name: str = "elasticsearch_ingest_attachment"
    description: str | None = (
        "Ingest a base64-encoded file attachment into an Elasticsearch index "
        "using the ingest-attachment pipeline to extract text content."
    )
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _ingest_attachment(
            index: str = Field(..., description="Target index name."),
            document_id: str = Field(..., description="Document ID to use for the attachment."),
            data: str = Field(..., description="Base64-encoded content of the file to ingest."),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            pipeline: str = Field("attachment", description="Ingest pipeline name (default 'attachment' requires the ingest-attachment plugin)."),
            fields: dict[str, Any] | None = Field(None, description="Additional document fields to store alongside the attachment."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            body: dict[str, Any] = {"data": data}
            if fields:
                body.update(fields)

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base}/{index}/_doc/{document_id}",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    params={"pipeline": pipeline},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_ingest_attachment, **kwargs)


class ElasticsearchBulkOperations(Tool):
    name: str = "elasticsearch_bulk_operations"
    description: str | None = (
        "Execute multiple index, create, update, or delete operations in a single request."
    )
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_operations(
            operations: list[dict[str, Any]] = Field(..., description="List of action/source pairs. Each pair is two dicts: - action: {'index'|'create'|'update'|'delete': {'_index': ..., '_id': ...}} - source (omitted for delete): the document body or update payload. Example:[{'index': {'_index': 'my-index', '_id': '1'}}, {'field': 'value'}, {'delete': {'_index': 'my-index', '_id': '2'}}]"),
            index: str | None = Field(None, description="Default index for operations that don't specify one."),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            refresh: str = Field("false", description="Refresh behavior: 'false'|'true'|'wait_for'."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            ndjson = "\n".join(
                __import__("json").dumps(op) for op in operations
            ) + "\n"

            url = f"{base}/_bulk" if not index else f"{base}/{index}/_bulk"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"ApiKey {api_key}",
                        "Content-Type": "application/x-ndjson",
                    },
                    content=ndjson.encode(),
                    params={"refresh": refresh},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_operations, **kwargs)


class ElasticsearchDeleteDocument(Tool):
    name: str = "elasticsearch_delete_document"
    description: str | None = "Delete a document from an Elasticsearch index by ID."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_document(
            index: str = Field(..., description="Index name"),
            document_id: str = Field(..., description="Document ID to delete"),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            refresh: str = Field("false", description="Refresh behavior: 'false'|'true'|'wait_for'."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base}/{index}/_doc/{document_id}",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    params={"refresh": refresh},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_document, **kwargs)


class ElasticsearchGetDocument(Tool):
    name: str = "elasticsearch_get_document"
    description: str | None = "Retrieve a document from an Elasticsearch index by ID."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_document(
            index: str = Field(..., description="Index name"),
            document_id: str = Field(..., description="Document ID to retrieve"),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            source_includes: list[str] | None = Field(None, description="List of fields to include in _source"),
            source_excludes: list[str] | None = Field(None, description="List of fields to exclude from _source"),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            params: dict[str, Any] = {}
            if source_includes:
                params["_source_includes"] = ",".join(source_includes)
            if source_excludes:
                params["_source_excludes"] = ",".join(source_excludes)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base}/{index}/_doc/{document_id}",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_document, **kwargs)


class ElasticsearchIndexDocument(Tool):
    name: str = "elasticsearch_index_document"
    description: str | None = "Index (create or replace) a document in an Elasticsearch index."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _index_document(
            index: str = Field(..., description="Index name"),
            document: dict[str, Any] = Field(..., description="Document content to index"),
            document_id: str | None = Field(None, description="Document ID (optional, auto-generated if omitted). If provided, uses PUT (upsert by ID); if omitted, uses POST (auto-generate ID)."),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            refresh: str = Field("false", description="Refresh behavior: 'false'|'true'|'wait_for'."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            async with httpx.AsyncClient() as client:
                if document_id:
                    response = await client.put(
                        f"{base}/{index}/_doc/{document_id}",
                        headers={"Authorization": f"ApiKey {api_key}"},
                        params={"refresh": refresh},
                        json=document,
                    )
                else:
                    response = await client.post(
                        f"{base}/{index}/_doc",
                        headers={"Authorization": f"ApiKey {api_key}"},
                        params={"refresh": refresh},
                        json=document,
                    )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_index_document, **kwargs)


class ElasticsearchUpdateDocument(Tool):
    name: str = "elasticsearch_update_document"
    description: str | None = "Partially update a document in an Elasticsearch index."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_document(
            index: str = Field(..., description="Index name"),
            document_id: str = Field(..., description="Document ID to update"),
            doc: dict[str, Any] | None = Field(None, description="Partial document fields to merge into the existing document"),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            script: dict[str, Any] | None = Field(None, description="Painless script to run, e.g. {'source': 'ctx._source.counter += params.count', 'params': {'count': 1}}"),
            upsert: dict[str, Any] | None = Field(None, description="Document to create if the ID does not exist (used with script)"),
            refresh: str = Field("false", description="Refresh behavior: 'false'|'true'|'wait_for'."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            body: dict[str, Any] = {}
            if doc is not None:
                body["doc"] = doc
            if script is not None:
                body["script"] = script
            if upsert is not None:
                body["upsert"] = upsert

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base}/{index}/_update/{document_id}",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    params={"refresh": refresh},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_document, **kwargs)


class ElasticsearchCreateIndex(Tool):
    name: str = "elasticsearch_create_index"
    description: str | None = "Create a new Elasticsearch index with optional mappings and settings."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_index(
            index: str = Field(..., description="Index name"),
            mappings: dict[str, Any] | None = Field(None, description="Field type definitions, e.g. {'properties': {'title': {'type': 'text'}}}"),
            settings: dict[str, Any] | None = Field(None, description="Index settings, e.g. {'number_of_shards': 1, 'number_of_replicas': 1}"),
            aliases: dict[str, Any] | None = Field(None, description="Index aliases to create alongside the index"),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            body: dict[str, Any] = {}
            if mappings:
                body["mappings"] = mappings
            if settings:
                body["settings"] = settings
            if aliases:
                body["aliases"] = aliases

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base}/{index}",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_index, **kwargs)


class ElasticsearchUpdateIndexSettings(Tool):
    name: str = "elasticsearch_update_index_settings"
    description: str | None = "Update dynamic settings on an existing Elasticsearch index."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_index_settings(
            index: str = Field(..., description="Index name"),
            settings: dict[str, Any] = Field(..., description="Dynamic index settings to update, e.g. {'number_of_replicas': 2, 'refresh_interval': '30s'}. Static settings (e.g. number_of_shards) cannot be changed after index creation."),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base}/{index}/_settings",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    json=settings,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_index_settings, **kwargs)


class ElasticsearchListIndices(Tool):
    name: str = "elasticsearch_list_indices"
    description: str | None = "List all indices in the Elasticsearch cluster with health, document count, and size."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_indices(
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            index_pattern: str | None = Field(None, description="Optional index pattern to filter, e.g. 'test*' or 'logs-*'. Omit to list all."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            path = f"/_cat/indices/{index_pattern}" if index_pattern else "/_cat/indices"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base.rstrip('/')}{path}",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    params={"format": "json", "bytes": "b"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_indices, **kwargs)


class ElasticsearchListDocuments(Tool):
    name: str = "elasticsearch_list_documents"
    description: str | None = "List documents in an index. Returns document IDs and content with optional pagination."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_documents(
            index: str = Field(..., description="Index name to list documents from."),
            size: int = Field(10, description="Number of documents to return."),
            from_: int = Field(0, description="Starting offset for pagination."),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            source: list[str] | bool | None = Field(True, description="Fields to return. True for all, False for none, or list of field names."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

            body: dict[str, Any] = {"size": size, "from": from_, "query": {"match_all": {}}}
            if source is not None:
                body["_source"] = source

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base}/{index}/_search",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
            hits = data.get("hits", {})
            total = hits.get("total", {})
            if isinstance(total, dict):
                total = total.get("value", 0)
            return {
                "total": total,
                "documents": [
                    {"_id": h["_id"], **h.get("_source", {})}
                    for h in hits.get("hits", [])
                ],
            }

        super().__init__(handler=_list_documents, **kwargs)


class ElasticsearchSearchDocuments(Tool):
    name: str = "elasticsearch_search_documents"
    description: str | None = "Search documents in one or more Elasticsearch indices using a query DSL."
    integration: Annotated[str, Integration("elasticsearch")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_documents(
            index: str = Field(..., description="Index name or comma-separated list, e.g. 'my-index' or 'logs-*'"),
            query: dict[str, Any] | None = Field(None, description="Elasticsearch Query DSL, e.g. {'match': {'title': 'search term'}}, {'bool': {'must': [{'match': {'status': 'active'}}]}}, {'match_all': {}}"),
            size: int = Field(10, description="Number of hits to return."),
            from_: int = Field(0, description="Starting offset for pagination."),
            host: str | None = Field(None, description="Elasticsearch base URL. Defaults to integration endpoint or ELASTICSEARCH_HOST."),
            sort: list[dict[str, Any]] | None = Field(None, description="Sort criteria, e.g. [{'date': 'desc'}, '_score']."),
            source: list[str] | bool | None = Field(None, description="List of fields to return, True/False to include/exclude all."),
            aggs: dict[str, Any] | None = Field(None, description="Aggregation definitions."),
            highlight: dict[str, Any] | None = Field(None, description="Highlight configuration for matched fields."),
            knn: dict[str, Any] | None = Field(None, description="K-nearest neighbor search config for vector similarity search."),
        ) -> Any:
            api_key, default_host = await _resolve_credentials(self)
            base = host or default_host
            if not base:
                raise ValueError("host required. Set ELASTICSEARCH_HOST or pass host.")
            import httpx

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
                    f"{base}/{index}/_search",
                    headers={"Authorization": f"ApiKey {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_documents, **kwargs)
