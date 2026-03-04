from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_PINECONE_CONTROL_PLANE = "https://api.pinecone.io"


class ListIndexes(Tool):
    name: str = "pinecone_list_indexes"
    description: str | None = "List all Pinecone indexes in the project."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_indexes() -> Any:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_PINECONE_CONTROL_PLANE}/indexes",
                    headers={"Api-Key": api_key},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/ListIndexes"

        super().__init__(handler=_list_indexes, metadata=metadata, **kwargs)


class CreateIndex(Tool):
    name: str = "pinecone_create_index"
    description: str | None = "Create a new Pinecone index."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
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
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

            body: dict[str, Any] = {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "spec": {"serverless": {"cloud": cloud, "region": region}},
                "deletion_protection": deletion_protection,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_PINECONE_CONTROL_PLANE}/indexes",
                    headers={"Api-Key": api_key, "Content-Type": "application/json"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/CreateIndex"

        super().__init__(handler=_create_index, metadata=metadata, **kwargs)


class DescribeIndexStats(Tool):
    name: str = "pinecone_describe_index_stats"
    description: str | None = "Get statistics about a Pinecone index: vector count, dimension, and namespace breakdown."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _describe_index_stats(
            index_host: str,
            metadata_filter: dict[str, Any] | None = None,
        ) -> Any:
            """
            index_host: the host name for the index (without protocol), e.g. "my-index-xyz.svc.pinecone.io"
            metadata_filter: optional metadata filter to count only matching vectors.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/DescribeIndexStats"

        super().__init__(handler=_describe_index_stats, metadata=metadata, **kwargs)


class UpsertVectors(Tool):
    name: str = "pinecone_upsert_vectors"
    description: str | None = "Upsert vectors into a Pinecone index namespace."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert_vectors(
            index_host: str,
            vectors: list[dict[str, Any]],
            namespace: str = "",
        ) -> Any:
            """
            index_host: the host name for the index (without protocol), e.g. "my-index-xyz.svc.pinecone.io"
            vectors: list of vector objects, each with:
              - "id": unique string ID
              - "values": list of floats (the embedding)
              - "metadata": optional dict of key-value pairs for filtering
              - "sparse_values": optional sparse vector {"indices": [...], "values": [...]}
            namespace: partition within the index (default "" = default namespace).
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://{index_host}/vectors/upsert",
                    headers={"Api-Key": api_key},
                    json={"vectors": vectors, "namespace": namespace},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/UpsertVectors"

        super().__init__(handler=_upsert_vectors, metadata=metadata, **kwargs)


class VectorSearch(Tool):
    name: str = "pinecone_vector_search"
    description: str | None = "Search a Pinecone index for the nearest neighbors to a query vector."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _vector_search(
            index_host: str,
            vector: list[float],
            top_k: int = 10,
            namespace: str = "",
            metadata_filter: dict[str, Any] | None = None,
            include_values: bool = False,
            include_metadata: bool = True,
        ) -> Any:
            """
            index_host: the host name for the index (without protocol), e.g. "my-index-xyz.svc.pinecone.io"
            vector: query embedding as a list of floats.
            top_k: number of nearest neighbors to return.
            namespace: partition to search within (default "" = default namespace).
            metadata_filter: metadata filter, e.g. {"genre": {"$eq": "documentary"}}
            include_values: whether to return the vector values in results.
            include_metadata: whether to return metadata in results.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/VectorSearch"

        super().__init__(handler=_vector_search, metadata=metadata, **kwargs)


class FetchVectors(Tool):
    name: str = "pinecone_fetch_vectors"
    description: str | None = "Fetch vectors by ID from a Pinecone index."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _fetch_vectors(
            index_host: str,
            ids: list[str],
            namespace: str = "",
        ) -> Any:
            """
            index_host: the host name for the index (without protocol), e.g. "my-index-xyz.svc.pinecone.io"
            ids: list of vector IDs to fetch.
            namespace: partition to fetch from (default "" = default namespace).
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/FetchVectors"

        super().__init__(handler=_fetch_vectors, metadata=metadata, **kwargs)


class DeleteVectors(Tool):
    name: str = "pinecone_delete_vectors"
    description: str | None = "Delete vectors from a Pinecone index by ID or metadata filter."
    integration: Annotated[str, Integration("pinecone")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
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
            index_host: the host name for the index (without protocol), e.g. "my-index-xyz.svc.pinecone.io"
            ids: list of vector IDs to delete. Provide either ids, metadata_filter, or delete_all=True.
            delete_all: if True, deletes all vectors in the namespace.
            namespace: partition to delete from (default "" = default namespace).
            metadata_filter: metadata filter to select vectors for deletion.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "api_key" in credentials
            api_key = credentials["api_key"]

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Pinecone/DeleteVectors"

        super().__init__(handler=_delete_vectors, metadata=metadata, **kwargs)
