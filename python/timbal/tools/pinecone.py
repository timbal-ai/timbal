import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.pinecone.io"

_INDEX_HOST_DESC = 'Index host, e.g. "my-index-xyz.svc.pinecone.io"'


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Pinecone API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("PINECONE_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Pinecone API key not found. Set PINECONE_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class PineconeListIndexes(Tool):
    name: str = "pinecone_list_indexes"
    description: str | None = "List all indexes in a Pinecone project."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_indexes() -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/indexes",
                    headers={"Api-Key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_indexes, **kwargs)


class PineconeCreateIndex(Tool):
    name: str = "pinecone_create_index"
    description: str | None = "Create a new Pinecone index."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_index(
            name: str,
            dimension: int,
            metric: str = Field("cosine", description='"cosine", "euclidean", or "dotproduct"'),
            cloud: str = Field("aws", description='"aws", "gcp", or "azure"'),
            region: str = Field("us-east-1", description="Cloud region for the index"),
            deletion_protection: str = Field("disabled", description='"enabled" or "disabled"'),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "spec": {"serverless": {"cloud": cloud, "region": region}},
                "deletion_protection": deletion_protection,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/indexes",
                    headers={"Api-Key": api_key, "Content-Type": "application/json"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_index, **kwargs)


class PineconeIndexStats(Tool):
    name: str = "pinecone_index_stats"
    description: str | None = "Get statistics for a Pinecone index."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _index_stats(
            index_host: str = Field(..., description=_INDEX_HOST_DESC),
            metadata_filter: dict[str, Any] | None = Field(None, description="Filter to count only matching vectors"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {}
            if metadata_filter:
                body["filter"] = metadata_filter

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://{index_host}/describe_index_stats",
                    headers={"Api-Key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_index_stats, **kwargs)


class PineconeUpsertVectors(Tool):
    name: str = "pinecone_upsert_vectors"
    description: str | None = "Upsert vectors into a Pinecone index."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert_vectors(
            index_host: str = Field(..., description=_INDEX_HOST_DESC),
            vectors: list[dict[str, Any]] = Field(..., description="List of vectors with id, values, and optional metadata"),
            namespace: str = Field("", description="Index partition"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://{index_host}/vectors/upsert",
                    headers={"Api-Key": api_key},
                    json={"vectors": vectors, "namespace": namespace},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_upsert_vectors, **kwargs)


class PineconeQuery(Tool):
    name: str = "pinecone_query"
    description: str | None = "Query a Pinecone index for nearest neighbors."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query(
            index_host: str = Field(..., description=_INDEX_HOST_DESC),
            vector: list[float] = Field(..., description="Query embedding"),
            top_k: int = Field(10, description="Number of nearest neighbors to return"),
            namespace: str = Field("", description="Index partition to search"),
            metadata_filter: dict[str, Any] | None = Field(None, description='Metadata filter, e.g. {"genre": {"$eq": "documentary"}}'),
            include_values: bool = Field(False, description="Return vector values in results"),
            include_metadata: bool = Field(True, description="Return metadata in results"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {
                "vector": vector,
                "topK": top_k,
                "namespace": namespace,
                "includeValues": include_values,
                "includeMetadata": include_metadata,
            }
            if metadata_filter:
                body["filter"] = metadata_filter

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://{index_host}/query",
                    headers={"Api-Key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_query, **kwargs)


class PineconeFetchVectors(Tool):
    name: str = "pinecone_fetch_vectors"
    description: str | None = "Fetch vectors by ID from a Pinecone index."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _fetch_vectors(
            index_host: str = Field(..., description=_INDEX_HOST_DESC),
            ids: list[str] = Field(..., description="Vector IDs to fetch"),
            namespace: str = Field("", description="Index partition to fetch from"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: list[tuple[str, str]] = [("ids", id_) for id_ in ids]
            if namespace:
                params.append(("namespace", namespace))

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://{index_host}/vectors/fetch",
                    headers={"Api-Key": api_key},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_fetch_vectors, **kwargs)


class PineconeDeleteVectors(Tool):
    name: str = "pinecone_delete_vectors"
    description: str | None = "Delete vectors from a Pinecone index."
    integration: Annotated[str, Integration("pinecone")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_vectors(
            index_host: str = Field(..., description=_INDEX_HOST_DESC),
            ids: list[str] | None = Field(None, description="Vector IDs to delete"),
            delete_all: bool = Field(False, description="Delete all vectors in the namespace"),
            namespace: str = Field("", description="Index partition to delete from"),
            metadata_filter: dict[str, Any] | None = Field(None, description="Metadata filter to select vectors for deletion"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {"namespace": namespace}
            if ids:
                body["ids"] = ids
            if delete_all:
                body["deleteAll"] = True
            if metadata_filter:
                body["filter"] = metadata_filter

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://{index_host}/vectors/delete",
                    headers={"Api-Key": api_key},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_vectors, **kwargs)
