import os
from typing import Annotated, Any

import lancedb
import pyarrow as pa
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_api_key(tool: Any) -> str:
    """Resolve LanceDB API key from integration, explicit field, or env var."""
    if isinstance(getattr(tool, "integration", None), Integration):
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


async def _resolve_db_uri(tool: Any, db_uri: str | None) -> str:
    """Resolve LanceDB db_uri from argument or integration credentials."""
    if db_uri is not None:
        return db_uri
    if isinstance(getattr(tool, "integration", None), Integration):
        credentials = await tool.integration.resolve()
        if "db_uri" in credentials:
            return credentials["db_uri"]
    raise ValueError("db_uri not provided and no integration configured with db_uri.")


class LanceDBListTables(Tool):
    name: str = "lancedb_list_tables"
    description: str | None = (
        "List all tables in a LanceDB Cloud database. "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tables(
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL, e.g. 'db://default-p73jfr'. If not provided, uses integration's db_uri.",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table_names = db.table_names()
            return {"tables": [{"name": name} for name in table_names], "count": len(table_names)}

        super().__init__(handler=_list_tables, **kwargs)


class LanceDBCreateTable(Tool):
    name: str = "lancedb_create_table"
    description: str | None = "Create a new table in LanceDB Cloud."
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_table(
            table_name: str = Field(..., description="Name of the table to create"),
            schema: dict[str, Any] | list[dict[str, str]] = Field(
                ...,
                description="Arrow schema definition (dict or list format)",
            ),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL (optional, uses integration default)",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)

            if isinstance(schema, list):
                schema_dict: dict[str, Any] = {}
                for field in schema:
                    if isinstance(field, dict) and "name" in field and "type" in field:
                        schema_dict[field["name"]] = field["type"]
                schema = schema_dict

            if isinstance(schema, dict):
                arrow_fields = []
                for field_name, field_type in schema.items():
                    if field_type == "string":
                        arrow_fields.append(pa.field(field_name, pa.string()))
                    elif field_type in ("int", "integer"):
                        arrow_fields.append(pa.field(field_name, pa.int64()))
                    elif field_type == "float":
                        arrow_fields.append(pa.field(field_name, pa.float32()))
                    elif isinstance(field_type, str):
                        # vector:N or vector: N (e.g. vector:3, vector:1536)
                        if field_type.startswith("vector:") and field_type[7:].strip().isdigit():
                            try:
                                dim = int(field_type.split(":")[1].strip())
                                arrow_fields.append(pa.field(field_name, pa.list_(pa.float32(), list_size=dim)))
                            except (ValueError, IndexError):
                                arrow_fields.append(pa.field(field_name, pa.string()))
                        # float[N] or float32[N] (e.g. float[3], float32[1536])
                        elif "[" in field_type and field_type.endswith("]"):
                            try:
                                dim = int(field_type[field_type.index("[") + 1 : -1])
                                arrow_fields.append(pa.field(field_name, pa.list_(pa.float32(), list_size=dim)))
                            except ValueError:
                                arrow_fields.append(pa.field(field_name, pa.string()))
                        else:
                            arrow_fields.append(pa.field(field_name, pa.string()))
                    else:
                        arrow_fields.append(pa.field(field_name, pa.string()))
                schema = pa.schema(arrow_fields)

            table = db.create_table(table_name, schema=schema)
            return {"table_created": table.name}

        super().__init__(handler=_create_table, **kwargs)


class LanceDBDropTable(Tool):
    name: str = "lancedb_drop_table"
    description: str | None = "Drop (delete) a table from LanceDB Cloud."
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _drop_table(
            table_name: str = Field(..., description="Name of the table to drop"),
            db_uri: str | None = Field(
                None,
                description="LanceDB database URI. If not provided, uses integration default",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            db.drop_table(table_name)
            return {"dropped": True, "table": table_name}

        super().__init__(handler=_drop_table, **kwargs)


class LanceDBInsertRecords(Tool):
    name: str = "lancedb_insert_records"
    description: str | None = "Insert or upsert records into a LanceDB Cloud table."
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_records(
            table_name: str = Field(..., description="Name of the table to insert into"),
            data: list[dict[str, Any]] = Field(
                ...,
                description="List of row dictionaries to insert. Each row with a vector column should include a 'vector' key containing a list of floats.",
            ),
            mode: str = Field("append", description="Insert mode: 'append' or 'overwrite'"),
            db_uri: str | None = Field(
                None,
                description="LanceDB database URI. If not provided, uses integration default",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            if mode == "overwrite":
                table.add(data, mode="overwrite")
            else:
                table.add(data)
            return {"inserted": len(data), "table": table_name, "mode": mode}

        super().__init__(handler=_insert_records, **kwargs)


class LanceDBVectorSearch(Tool):
    name: str = "lancedb_vector_search"
    description: str | None = (
        "Search a LanceDB Cloud table by vector similarity. "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _vector_search(
            table_name: str = Field(..., description="Name of the table to search"),
            vector: list[float] = Field(..., description="Query embedding as a list of floats"),
            limit: int = Field(10, description="Maximum number of results to return"),
            metric: str = Field("cosine", description='"cosine", "l2", or "dot"'),
            vector_column: str = Field("vector", description="Name of the column containing embeddings"),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL. If not provided, uses integration default.",
            ),
            columns: list[str] | None = Field(None, description="List of columns to return (default all)"),
            where: str | None = Field(None, description="SQL WHERE clause to pre-filter results"),
            nprobes: int | None = Field(None, description="Number of IVF partitions to probe"),
            refine_factor: int | None = Field(None, description="Re-rank top-k using exact distance"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)

            results = table.search(vector).limit(limit).metric(metric)
            if vector_column != "vector":
                results = results.column(vector_column)
            if columns:
                results = results.select(columns)
            if where:
                results = results.where(where)
            if nprobes is not None or refine_factor is not None:
                results = results.nprobes(nprobes or 10).refine_factor(refine_factor or 10)

            df = results.to_pandas()
            return {"results": df.to_dict("records"), "count": len(df)}

        super().__init__(handler=_vector_search, **kwargs)


class LanceDBFullTextSearch(Tool):
    name: str = "lancedb_full_text_search"
    description: str | None = (
        "Search a LanceDB Cloud table using full-text search (BM25). "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _full_text_search(
            table_name: str = Field(..., description="Name of the table to search"),
            query: str = Field(..., description="Keyword query string for BM25 full-text search"),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL. If not provided, uses integration default.",
            ),
            columns: list[str] | None = Field(None, description="List of columns to return (default all)"),
            limit: int = Field(10, description="Maximum number of results to return"),
            where: str | None = Field(None, description="SQL WHERE clause to pre-filter results"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)

            results = table.search(query).limit(limit)
            if columns:
                results = results.select(columns)
            if where:
                results = results.where(where)

            df = results.to_pandas()
            return {"results": df.to_dict("records"), "count": len(df)}

        super().__init__(handler=_full_text_search, **kwargs)


class LanceDBHybridSearch(Tool):
    name: str = "lancedb_hybrid_search"
    description: str | None = (
        "Search a LanceDB Cloud table combining vector similarity and full-text search (hybrid). "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _hybrid_search(
            table_name: str = Field(..., description="Name of the table to search"),
            vector: list[float] = Field(..., description="Query embedding for the vector similarity component"),
            limit: int = Field(10, description="Maximum number of results to return"),
            vector_column: str = Field("vector", description="Name of the column containing embeddings"),
            metric: str = Field("cosine", description='"cosine", "l2", or "dot"'),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL. If not provided, uses integration default.",
            ),
            columns: list[str] | None = Field(None, description="List of columns to return (default all)"),
            where: str | None = Field(None, description="SQL WHERE clause to pre-filter results"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)

            vector_results = table.search(vector).limit(limit * 2).metric(metric)
            if vector_column != "vector":
                vector_results = vector_results.column(vector_column)
            if columns:
                vector_results = vector_results.select(columns)
            if where:
                vector_results = vector_results.where(where)

            df = vector_results.to_pandas()
            return {"results": df.to_dict("records"), "count": len(df)}

        super().__init__(handler=_hybrid_search, **kwargs)


class LanceDBCreateFTSIndex(Tool):
    name: str = "lancedb_create_fts_index"
    description: str | None = (
        "Create a full-text search (FTS) index on a text column in LanceDB Cloud. "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_fts_index(
            table_name: str = Field(..., description="Name of the table to create index on"),
            column_name: str = Field(..., description="Name of the text column to index"),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL. If not provided, uses integration default.",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            table.create_fts_index(column_name)
            return {
                "fts_index_created": True,
                "table": table_name,
                "column": column_name,
                "index_type": "FTS",
            }

        super().__init__(handler=_create_fts_index, **kwargs)


class LanceDBDropFTSIndex(Tool):
    name: str = "lancedb_drop_fts_index"
    description: str | None = (
        "Drop a full-text search (FTS) index from a text column in LanceDB Cloud. "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _drop_fts_index(
            table_name: str = Field(..., description="Name of the table to drop index from"),
            column_name: str = Field(..., description="Name of the text column to drop FTS index from"),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL. If not provided, uses integration default.",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            index_name = f"{column_name}_idx"
            table.drop_index(index_name)
            return {
                "fts_index_dropped": True,
                "table": table_name,
                "column": column_name,
                "index_name_used": index_name,
                "index_type": "FTS",
            }

        super().__init__(handler=_drop_fts_index, **kwargs)


class LanceDBDeleteRecords(Tool):
    name: str = "lancedb_delete_records"
    description: str | None = (
        "Delete records from a LanceDB Cloud table matching a SQL WHERE predicate. "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_records(
            table_name: str = Field(..., description="Name of the table to delete from"),
            where: str = Field(
                ...,
                description="SQL WHERE predicate to select rows for deletion, e.g. \"id = '123'\" or \"date < '2024-01-01'\"",
            ),
            db_uri: str | None = Field(
                None,
                description="LanceDB Cloud project URL. If not provided, uses integration default.",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            table.delete(where)
            return {"deleted": True, "table": table_name, "where": where}

        super().__init__(handler=_delete_records, **kwargs)


class LanceDBDescribeTable(Tool):
    name: str = "lancedb_describe_table"
    description: str | None = (
        "Get metadata and schema information for a LanceDB Cloud table. "
        "If db_uri is None, uses the integration's default db_uri."
    )
    integration: Annotated[str, Integration("lancedb")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _describe_table(
            table_name: str = Field(..., description="Name of the table to describe"),
            db_uri: str | None = Field(
                None,
                description="LanceDB database URI. If not provided, uses integration default",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            db_uri = await _resolve_db_uri(self, db_uri)
            db = lancedb.connect(db_uri, api_key=api_key)
            table = db.open_table(table_name)
            schema = table.schema
            return {
                "table_name": table_name,
                "schema": schema.to_string(),
                "columns": schema.names,
            }

        super().__init__(handler=_describe_table, **kwargs)
