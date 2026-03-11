import os
from typing import Annotated, Any

import lancedb
import pyarrow as pa
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_api_key(tool: Any) -> str:
    """Resolve LanceDB API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("LANCEDB_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "LanceDB API key not found. Set LANCEDB_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class ListTables(Tool):
    name: str = "lancedb_list_tables"
    description: str | None = "List all tables in a LanceDB Cloud database. If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _list_tables(
            db_uri: str | None = Field(None, description="LanceDB Cloud project URL, e.g. 'db://default-p73jfr'. If not provided, will use integration's db_uri."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            
            table_names = db.table_names()
            
            return {
                "tables": [{"name": name} for name in table_names],
                "count": len(table_names)
            }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/ListTables"

        super().__init__(handler=_list_tables, metadata=metadata, **kwargs)


class CreateTable(Tool):
    name: str = "lancedb_create_table"
    description: str | None = "Create a new table in LanceDB Cloud."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _create_table(
            table_name: str = Field(..., description="Name of the table to create"),
            schema: dict[str, Any] | list[dict[str, str]] = Field(..., description="Arrow schema definition (dict or list format)"),
            db_uri: str | None = Field(None, description="LanceDB Cloud project URL (optional, uses integration default)"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            
            # Convert list format to dict if needed
            if isinstance(schema, list):
                schema_dict = {}
                for field in schema:
                    if isinstance(field, dict) and 'name' in field and 'type' in field:
                        schema_dict[field['name']] = field['type']
                schema = schema_dict
            
            # Convert dict schema to PyArrow Schema
            if isinstance(schema, dict):
                arrow_fields = []
                for field_name, field_type in schema.items():
                    if field_type == "string":
                        arrow_fields.append(pa.field(field_name, pa.string()))
                    elif field_type == "int" or field_type == "integer":
                        arrow_fields.append(pa.field(field_name, pa.int64()))
                    elif field_type == "float":
                        arrow_fields.append(pa.field(field_name, pa.float32()))
                    elif field_type.startswith("float[") and field_type.endswith("]"):
                        # Parse vector dimension: float[1536] -> 1536
                        try:
                            dim = int(field_type[6:-1])
                            arrow_fields.append(pa.field(field_name, pa.list_(pa.float32(), dim)))
                        except ValueError:
                            arrow_fields.append(pa.field(field_name, pa.float32()))
                    else:
                        # Default to string for unknown types
                        arrow_fields.append(pa.field(field_name, pa.string()))
                schema = pa.schema(arrow_fields)
            
            # Create table with schema
            table = db.create_table(table_name, schema=schema)
            
            return {"table_created": table.name}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/CreateTable"

        super().__init__(handler=_create_table, metadata=metadata, **kwargs)


class DropTable(Tool):
    name: str = "lancedb_drop_table"
    description: str | None = "Drop (delete) a table from LanceDB Cloud."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _drop_table(
            table_name: str = Field(..., description="Name of the table to drop"),
            db_uri: str | None = Field(None, description="LanceDB database URI. If not provided, uses integration default")
        ) -> Any:
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            db.drop_table(table_name)
            
            return {"dropped": True, "table": table_name}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DropTable"

        super().__init__(handler=_drop_table, metadata=metadata, **kwargs)


class InsertRecords(Tool):
    name: str = "lancedb_insert_records"
    description: str | None = "Insert or upsert records into a LanceDB Cloud table."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _insert_records(
            table_name: str = Field(..., description="Name of the table to insert into"),
            data: list[dict[str, Any]] = Field(..., description="List of row dictionaries to insert. Each row with a vector column should include a 'vector' key containing a list of floats, e.g. [{'vector': [0.1, 0.2, ...], 'text': '...'}]"),
            mode: str = Field("append", description="Insert mode: 'append' or 'overwrite'"),
            db_uri: str | None = Field(None, description="LanceDB database URI. If not provided, uses integration default")
        ) -> Any:
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            if mode == "overwrite":
                table.add(data, mode="overwrite")
            else:
                table.add(data)
            
            return {"inserted": len(data), "table": table_name, "mode": mode}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/InsertRecords"

        super().__init__(handler=_insert_records, metadata=metadata, **kwargs)


class VectorSearch(Tool):
    name: str = "lancedb_vector_search"
    description: str | None = "Search a LanceDB Cloud table by vector similarity. If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _vector_search(
            table_name: str,
            vector: list[float],
            limit: int = 10,
            metric: str = "cosine",
            vector_column: str = "vector",
            db_uri: str | None = None,
            columns: list[str] | None = None,
            where: str | None = None,
            nprobes: int | None = None,
            refine_factor: int | None = None,
        ) -> Any:
            """
            db_uri: LanceDB Cloud project URL.
            vector: query embedding as a list of floats.
            metric: "cosine", "l2", or "dot".
            vector_column: name of the column containing embeddings (default "vector").
            columns: list of columns to return (default all).
            where: SQL WHERE clause to pre-filter results, e.g. "category = 'news'".
            nprobes: number of IVF partitions to probe (higher = better recall, slower).
            refine_factor: re-rank top-k results using exact distance (higher = better recall).
            """
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            # Build search query
            search_params = {}
            if nprobes is not None:
                search_params["nprobes"] = nprobes
            if refine_factor is not None:
                search_params["refine_factor"] = refine_factor
            
            results = table.search(vector).limit(limit).metric(metric)
            
            if vector_column != "vector":
                results = results.column(vector_column)
            if columns:
                results = results.select(columns)
            if where:
                results = results.where(where)
            if search_params:
                results = results.nprobes(nprobes or 10).refine_factor(refine_factor or 10)
            
            df = results.to_pandas()
            return {"results": df.to_dict("records"), "count": len(df)}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/VectorSearch"

        super().__init__(handler=_vector_search, metadata=metadata, **kwargs)


class FullTextSearch(Tool):
    name: str = "lancedb_full_text_search"
    description: str | None = "Search a LanceDB Cloud table using full-text search (BM25). If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _full_text_search(
            table_name: str,
            query: str,
            db_uri: str | None = None,
            columns: list[str] | None = None,
            limit: int = 10,
            where: str | None = None,
        ) -> Any:
            """
            query: keyword query string for BM25 full-text search.
            columns: list of columns to return (default all).
            where: SQL WHERE clause to pre-filter results.
            """
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            # Build full-text search query
            results = table.search(query).limit(limit)
            
            if columns:
                results = results.select(columns)
            if where:
                results = results.where(where)
            
            df = results.to_pandas()
            return {"results": df.to_dict("records"), "count": len(df)}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/FullTextSearch"

        super().__init__(handler=_full_text_search, metadata=metadata, **kwargs)


class HybridSearch(Tool):
    name: str = "lancedb_hybrid_search"
    description: str | None = "Search a LanceDB Cloud table combining vector similarity and full-text search (hybrid). If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _hybrid_search(
            table_name: str,
            vector: list[float],
            limit: int = 10,
            vector_column: str = "vector",
            metric: str = "cosine",
            db_uri: str | None = None,
            columns: list[str] | None = None,
            where: str | None = None,
        ) -> Any:
            """
            vector: query embedding for the vector similarity component.
            Performs vector search (as proxy for hybrid search).
            """
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            # For hybrid search, we'll combine vector and text search results
            # Note: LanceDB SDK doesn't have built-in hybrid search, so we'll do vector search
            vector_results = table.search(vector).limit(limit * 2).metric(metric)
            
            if vector_column != "vector":
                vector_results = vector_results.column(vector_column)
            if columns:
                vector_results = vector_results.select(columns)
            if where:
                vector_results = vector_results.where(where)
            
            df = vector_results.to_pandas()
            return {"results": df.to_dict("records"), "count": len(df)}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/HybridSearch"

        super().__init__(handler=_hybrid_search, metadata=metadata, **kwargs)


class CreateFTSIndex(Tool):
    name: str = "lancedb_create_fts_index"
    description: str | None = "Create a full-text search (FTS) index on a text column in LanceDB Cloud. If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _create_fts_index(
            table_name: str,
            column_name: str,
            db_uri: str | None = None,
        ) -> Any:
            """
            table_name: name of the table to create index on.
            column_name: name of the text column to index (must contain text data).
            db_uri: LanceDB Cloud project URL (optional, uses integration default).
            """
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            table.create_fts_index(column_name)
            
            return {
                "fts_index_created": True,
                "table": table_name,
                "column": column_name,
                "index_type": "FTS"
            }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/CreateFTSIndex"

        super().__init__(handler=_create_fts_index, metadata=metadata, **kwargs)


class DropFTSIndex(Tool):
    name: str = "lancedb_drop_fts_index"
    description: str | None = "Drop a full-text search (FTS) index from a text column in LanceDB Cloud. If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _drop_fts_index(
            table_name: str,
            column_name: str,
            db_uri: str | None = None,
        ) -> Any:
            """
            table_name: name of the table to drop index from.
            column_name: name of the text column to drop FTS index from.
            db_uri: LanceDB Cloud project URL (optional, uses integration default).
            """
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            # Drop the FTS index - use LanceDB Cloud naming pattern
            index_name = f"{column_name}_idx"
            
            table.drop_index(index_name)
            return {
                "fts_index_dropped": True,
                "table": table_name,
                "column": column_name,
                "index_name_used": index_name,
                "index_type": "FTS"
            }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DropFTSIndex"

        super().__init__(handler=_drop_fts_index, metadata=metadata, **kwargs)


class DeleteRecords(Tool):
    name: str = "lancedb_delete_records"
    description: str | None = "Delete records from a LanceDB Cloud table matching a SQL WHERE predicate. If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _delete_records(
            table_name: str,
            where: str,
            db_uri: str | None = None
        ) -> Any:
            """
            where: SQL WHERE predicate to select rows for deletion, e.g. "id = '123'" or "date < '2024-01-01'".
            """
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            table.delete(where)
            
            return {"deleted": True, "table": table_name, "where": where}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DeleteRecords"

        super().__init__(handler=_delete_records, metadata=metadata, **kwargs)


class DescribeTable(Tool):
    name: str = "lancedb_describe_table"
    description: str | None = "Get metadata and schema information for a LanceDB Cloud table. If db_uri is None, uses the integration's default db_uri."
    integration: Annotated[str, Integration("lancedb")] | None = None
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
        async def _describe_table(table_name: str = Field(..., description="Name of the table to describe"), db_uri: str | None = Field(None, description="LanceDB database URI. If not provided, uses integration default")) -> Any:
            api_key = await _resolve_api_key(self)
            
            if db_uri is None:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    assert "db_uri" in credentials
                    db_uri = credentials["db_uri"]
                else:
                    raise ValueError("db_uri not provided and no integration configured")

            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            
            schema = table.schema
            
            return {
                "table_name": table_name,
                "schema": schema.to_string(),
                "columns": schema.names
            }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "LanceDB/DescribeTable"

        super().__init__(handler=_describe_table, metadata=metadata, **kwargs)
