import os
from typing import Annotated, Any

from pydantic import SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.pinecone.io"


async def _resolve_api_key(tool: Tool) -> str:
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
            metric: str = "cosine",
            cloud: str = "aws",
            region: str = "us-east-1",
            deletion_protection: str = "disabled",
        ) -> Any:
            """
            metric: "cosine", "euclidean", or "dotproduct".
            cloud: "aws", "gcp", or "azure".
            deletion_protection: "enabled" or "disabled".
            """
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
            index_host: str,
            metadata_filter: dict[str, Any] | None = None,
        ) -> Any:
            """
            index_host: the host for the index, e.g. "my-index-xyz.svc.pinecone.io"
            metadata_filter: optional filter to count only matching vectors.
            """
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
            index_host: str,
            vectors: list[dict[str, Any]],
            namespace: str = "",
        ) -> Any:
            """
            index_host: the host for the index, e.g. "my-index-xyz.svc.pinecone.io"
            vectors: list of vector objects, each with:
              - "id": unique string ID
              - "values": list of floats (the embedding)
              - "metadata": optional dict of key-value pairs for filtering
              - "sparse_values": optional sparse vector {"indices": [...], "values": [...]}
            namespace: partition within the index (default "" = default namespace).
            """
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
            index_host: str,
            vector: list[float],
            top_k: int = 10,
            namespace: str = "",
            metadata_filter: dict[str, Any] | None = None,
            include_values: bool = False,
            include_metadata: bool = True,
        ) -> Any:
            """
            index_host: the host for the index, e.g. "my-index-xyz.svc.pinecone.io"
            vector: query embedding as a list of floats.
            top_k: number of nearest neighbors to return.
            namespace: partition to search within (default "" = default namespace).
            metadata_filter: metadata filter, e.g. {"genre": {"$eq": "documentary"}}
            include_values: whether to return vector values in results.
            include_metadata: whether to return metadata in results.
            """
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
            index_host: str,
            ids: list[str],
            namespace: str = "",
        ) -> Any:
            """
            index_host: the host for the index, e.g. "my-index-xyz.svc.pinecone.io"
            ids: list of vector IDs to fetch.
            namespace: partition to fetch from (default "" = default namespace).
            """
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
            index_host: str,
            ids: list[str] | None = None,
            delete_all: bool = False,
            namespace: str = "",
            metadata_filter: dict[str, Any] | None = None,
        ) -> Any:
            """
            index_host: the host for the index, e.g. "my-index-xyz.svc.pinecone.io"
            ids: list of vector IDs to delete.
            delete_all: if True, deletes all vectors in the namespace.
            namespace: partition to delete from (default "" = default namespace).
            metadata_filter: metadata filter to select vectors for deletion.
            """
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
