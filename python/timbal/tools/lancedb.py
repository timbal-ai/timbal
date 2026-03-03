from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration


class ListTables(Tool):
    name: str = "lancedb_list_tables"
    description: str | None = "List all tables in a LanceDB Cloud database."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tables(
            host: str,
            limit: int = 10,
            page_token: str | None = None,
        ) -> Any:
            """
            host: LanceDB Cloud project URL, e.g. "https://your-project.api.lancedb.com"
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if page_token:
                params["page_token"] = page_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{host}/v1/table/",
                    headers={"x-api-key": token},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/ListTables"

        super().__init__(handler=_list_tables, metadata=metadata, **kwargs)


class CreateTable(Tool):
    name: str = "lancedb_create_table"
    description: str | None = "Create a new table in LanceDB Cloud."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_table(
            host: str,
            table_name: str,
            schema: dict[str, Any] | None = None,
            data: list[dict[str, Any]] | None = None,
        ) -> Any:
            """
            host: LanceDB Cloud project URL, e.g. "https://your-project.api.lancedb.com"
            schema: Arrow schema as a JSON-serializable dict (optional if data is provided).
            data: initial rows to insert on creation (optional).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {}
            if schema:
                body["schema"] = schema
            if data:
                body["data"] = data

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/v1/table/{table_name}/create/",
                    headers={"x-api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/CreateTable"

        super().__init__(handler=_create_table, metadata=metadata, **kwargs)


class DropTable(Tool):
    name: str = "lancedb_drop_table"
    description: str | None = "Drop (delete) a table from LanceDB Cloud."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _drop_table(host: str, table_name: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{host}/v1/table/{table_name}/",
                    headers={"x-api-key": token},
                )
                response.raise_for_status()
                return {"dropped": True, "table": table_name}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DropTable"

        super().__init__(handler=_drop_table, metadata=metadata, **kwargs)


class InsertRecords(Tool):
    name: str = "lancedb_insert_records"
    description: str | None = "Insert or upsert records into a LanceDB Cloud table."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_records(
            host: str,
            table_name: str,
            data: list[dict[str, Any]],
            mode: str = "append",
        ) -> Any:
            """
            data: list of row dicts. Each row with a vector column should include a "vector" key
                  containing a list of floats, e.g. [{"vector": [0.1, 0.2, ...], "text": "..."}]
            mode: "append" to add rows, "overwrite" to replace the entire table.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/v1/table/{table_name}/insert/",
                    headers={"x-api-key": token},
                    json={"data": data, "mode": mode},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/InsertRecords"

        super().__init__(handler=_insert_records, metadata=metadata, **kwargs)


class VectorSearch(Tool):
    name: str = "lancedb_vector_search"
    description: str | None = "Search a LanceDB Cloud table by vector similarity."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _vector_search(
            host: str,
            table_name: str,
            vector: list[float],
            limit: int = 10,
            metric: str = "cosine",
            vector_column: str = "vector",
            columns: list[str] | None = None,
            where: str | None = None,
            nprobes: int | None = None,
            refine_factor: int | None = None,
        ) -> Any:
            """
            host: LanceDB Cloud project URL.
            vector: query embedding as a list of floats.
            metric: "cosine", "l2", or "dot".
            vector_column: name of the column containing embeddings (default "vector").
            columns: list of columns to return (default all).
            where: SQL WHERE clause to pre-filter results, e.g. "category = 'news'".
            nprobes: number of IVF partitions to probe (higher = better recall, slower).
            refine_factor: re-rank top-k results using exact distance (higher = better recall).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                "vector": vector,
                "limit": limit,
                "metric": metric,
                "vector_column": vector_column,
            }
            if columns:
                body["columns"] = columns
            if where:
                body["where"] = where
            if nprobes is not None:
                body["nprobes"] = nprobes
            if refine_factor is not None:
                body["refine_factor"] = refine_factor

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/v1/table/{table_name}/query/",
                    headers={"x-api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/VectorSearch"

        super().__init__(handler=_vector_search, metadata=metadata, **kwargs)


class FullTextSearch(Tool):
    name: str = "lancedb_full_text_search"
    description: str | None = "Search a LanceDB Cloud table using full-text search (BM25)."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _full_text_search(
            host: str,
            table_name: str,
            query: str,
            columns: list[str] | None = None,
            limit: int = 10,
            where: str | None = None,
        ) -> Any:
            """
            query: keyword query string for BM25 full-text search.
            columns: list of columns to return (default all).
            where: SQL WHERE clause to pre-filter results.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                "full_text_search": {"query": query},
                "limit": limit,
            }
            if columns:
                body["columns"] = columns
            if where:
                body["where"] = where

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/v1/table/{table_name}/query/",
                    headers={"x-api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/FullTextSearch"

        super().__init__(handler=_full_text_search, metadata=metadata, **kwargs)


class HybridSearch(Tool):
    name: str = "lancedb_hybrid_search"
    description: str | None = "Search a LanceDB Cloud table combining vector similarity and full-text search (hybrid)."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _hybrid_search(
            host: str,
            table_name: str,
            vector: list[float],
            query: str,
            limit: int = 10,
            vector_column: str = "vector",
            metric: str = "cosine",
            columns: list[str] | None = None,
            where: str | None = None,
        ) -> Any:
            """
            vector: query embedding for the vector similarity component.
            query: keyword query for the full-text search component.
            Results are reranked by combining both scores (RRF by default).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                "vector": vector,
                "full_text_search": {"query": query},
                "limit": limit,
                "vector_column": vector_column,
                "metric": metric,
            }
            if columns:
                body["columns"] = columns
            if where:
                body["where"] = where

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/v1/table/{table_name}/query/",
                    headers={"x-api-key": token},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/HybridSearch"

        super().__init__(handler=_hybrid_search, metadata=metadata, **kwargs)


class DeleteRecords(Tool):
    name: str = "lancedb_delete_records"
    description: str | None = "Delete records from a LanceDB Cloud table matching a SQL WHERE predicate."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_records(
            host: str,
            table_name: str,
            where: str,
        ) -> Any:
            """
            where: SQL WHERE predicate to select rows for deletion, e.g. "id = '123'" or "date < '2024-01-01'".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host}/v1/table/{table_name}/delete/",
                    headers={"x-api-key": token},
                    json={"where": where},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DeleteRecords"

        super().__init__(handler=_delete_records, metadata=metadata, **kwargs)


class DescribeTable(Tool):
    name: str = "lancedb_describe_table"
    description: str | None = "Get metadata and schema information for a LanceDB Cloud table."
    integration: Annotated[str, Integration("lancedb")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _describe_table(host: str, table_name: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{host}/v1/table/{table_name}/describe/",
                    headers={"x-api-key": token},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DescribeTable"

        super().__init__(handler=_describe_table, metadata=metadata, **kwargs)
