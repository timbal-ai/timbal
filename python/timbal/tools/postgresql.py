import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


def _rest_url(host: str, table: str) -> str:
    host = host.rstrip("/")
    return f"{host}/rest/v1/{table}"


def _rpc_url(host: str, function_name: str) -> str:
    host = host.rstrip("/")
    return f"{host}/rest/v1/rpc/{function_name}"


def _build_filter_params(filters: dict[str, Any]) -> dict[str, str]:
    """Convert a filters dict to PostgREST query parameters.

    Each key is a column name. The value can be:
      - a plain value → column=eq.{value}
      - a dict with operator → {"$gt": 5} → column=gt.5
        Supported operators: $eq, $neq, $gt, $gte, $lt, $lte, $like, $ilike, $in, $is
    """
    _op_map = {
        "$eq": "eq", "$neq": "neq", "$gt": "gt", "$gte": "gte",
        "$lt": "lt", "$lte": "lte", "$like": "like", "$ilike": "ilike",
        "$in": "in", "$is": "is",
    }
    params: dict[str, str] = {}
    for col, val in filters.items():
        if isinstance(val, dict):
            for mongo_op, pg_op in _op_map.items():
                if mongo_op in val:
                    raw = val[mongo_op]
                    if pg_op == "in" and isinstance(raw, list):
                        params[col] = f"in.({','.join(str(v) for v in raw)})"
                    else:
                        params[col] = f"{pg_op}.{raw}"
                    break
        else:
            params[col] = f"eq.{val}"
    return params


async def _resolve_token(tool: Any) -> str:
    """Resolve PostgREST/Supabase API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("POSTGRESQL_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "PostgreSQL API key not found. Set POSTGRESQL_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


class PostgresqlInsertRow(Tool):
    name: str = "postgresql_insert_row"
    description: str | None = "Insert a new row into a PostgreSQL table via PostgREST."
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_insert_row(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance, e.g. https://xyz.supabase.co",
            ),
            table: str = Field(..., description="Table name"),
            data: dict[str, Any] = Field(
                ...,
                description="Column-value pairs to insert, e.g. {'name': 'Alice', 'email': 'alice@example.com'}",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                        "Prefer": "return=representation",
                    },
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_insert_row, **kwargs)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class PostgresqlUpsertRow(Tool):
    name: str = "postgresql_upsert_row"
    description: str | None = (
        "Insert a row or update it if a row with the same conflict column already exists."
    )
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_upsert_row(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            table: str = Field(..., description="Table name"),
            data: dict[str, Any] = Field(..., description="Column-value pairs for the row"),
            on_conflict: str = Field(
                "id",
                description="Comma-separated column(s) for conflict resolution, e.g. 'id' or 'email'",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                        "Prefer": "return=representation,resolution=merge-duplicates",
                        "Resolution": "merge-duplicates",
                    },
                    params={"on_conflict": on_conflict},
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_upsert_row, **kwargs)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class PostgresqlUpdateRow(Tool):
    name: str = "postgresql_update_row"
    description: str | None = "Update one or more rows in a PostgreSQL table that match the given filters."
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_update_row(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            table: str = Field(..., description="Table name"),
            filters: dict[str, Any] = Field(
                ...,
                description="Columns to match, e.g. {'id': 42} or {'status': {'$eq': 'pending'}}",
            ),
            data: dict[str, Any] = Field(
                ...,
                description="Column-value pairs to set on matching rows",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx
            
            params = _build_filter_params(filters)
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                        "Prefer": "return=representation",
                    },
                    params=params,
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_update_row, **kwargs)


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------


class PostgresqlFindRow(Tool):
    name: str = "postgresql_find_row"
    description: str | None = (
        "Find a single row in a PostgreSQL table by matching a specific column value."
    )
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_find_row(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            table: str = Field(..., description="Table name"),
            column: str = Field(..., description="Column to look up, e.g. 'id' or 'email'"),
            value: Any = Field(..., description="Value to match"),
            select: str = Field("*", description="Comma-separated columns to return, e.g. 'id,name,email'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                        "Prefer": "return=representation",
                    },
                    params={column: f"eq.{value}", "select": select, "limit": 1},
                )
                response.raise_for_status()
                rows = response.json()
                return rows[0] if rows else None

        super().__init__(handler=_postgresql_find_row, **kwargs)


class PostgresqlFindRowWithCustomQuery(Tool):
    name: str = "postgresql_find_row_with_custom_query"
    description: str | None = (
        "Find rows in a PostgreSQL table using multiple filter conditions, "
        "custom column selection, ordering, and pagination."
    )
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_find_row_with_custom_query(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            table: str = Field(..., description="Table name"),
            filters: dict[str, Any] | None = Field(
                None,
                description="Column conditions, e.g. {'status': 'active', 'age': {'$gte': 18}}. Omit to return all rows.",
            ),
            select: str = Field(
                "*",
                description="Columns to return, e.g. 'id,name,email'. Supports embedded joins: 'id,name,orders(id,total)'",
            ),
            order: str | None = Field(
                None,
                description="Sort expression, e.g. 'created_at.desc' or 'name.asc,age.desc'",
            ),
            limit: int = Field(100, description="Max rows to return"),
            offset: int = Field(0, description="Rows to skip for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"select": select, "limit": limit, "offset": offset}
            if filters:
                params.update(_build_filter_params(filters))
            if order:
                params["order"] = order
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_find_row_with_custom_query, **kwargs)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class PostgresqlDeleteRows(Tool):
    name: str = "postgresql_delete_rows"
    description: str | None = "Delete one or more rows from a PostgreSQL table matching the given filters."
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_delete_rows(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            table: str = Field(..., description="Table name"),
            filters: dict[str, Any] = Field(
                ...,
                description="Conditions to select rows, e.g. {'id': 42} or {'status': 'archived'}. At least one required.",
            ),
        ) -> Any:
            if not filters:
                raise ValueError("At least one filter is required to prevent deleting all rows.")
            token = await _resolve_token(self)
            import httpx

            params = _build_filter_params(filters)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                        "Prefer": "return=representation",
                    },
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_delete_rows, **kwargs)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


class PostgresqlExecuteSQLQuery(Tool):
    name: str = "postgresql_execute_sql_query"
    description: str | None = (
        "Execute a raw SQL SELECT query and return the result rows. "
        "Requires the PostgREST RPC endpoint to expose a sql execution function, "
        "or a Supabase project with the pg_net extension / service-role key."
    )
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_execute_sql_query(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            query: str = Field(
                ...,
                description="SQL SELECT statement, e.g. 'SELECT * FROM users WHERE active = true LIMIT 10'",
            ),
            params: list[Any] | None = Field(
                None,
                description="Optional positional parameters for parameterized queries",
            ),
            rpc_function: str = Field(
                "exec_sql",
                description="Name of the PostgREST RPC function that executes SQL",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {"query": query}
            if params:
                body["params"] = params
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _rpc_url(host, rpc_function),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                    },
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_execute_sql_query, **kwargs)


class PostgresqlQuerySQLDatabase(Tool):
    name: str = "postgresql_query_sql_database"
    description: str | None = (
        "Execute any SQL statement (INSERT, UPDATE, DELETE, SELECT, DDL) "
        "against a PostgreSQL database via a service-role RPC endpoint."
    )
    integration: Annotated[str, Integration("postgresql")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _postgresql_query_sql_database(
            host: str = Field(
                ...,
                description="Base URL of your PostgREST or Supabase instance",
            ),
            sql: str = Field(
                ...,
                description="Any valid SQL statement, e.g. 'INSERT INTO orders (user_id, total) VALUES (1, 99.99)'",
            ),
            rpc_function: str = Field(
                "exec_sql",
                description="Name of the PostgREST RPC function that accepts raw SQL",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _rpc_url(host, rpc_function),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                    },
                    json={"query": sql},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_postgresql_query_sql_database, **kwargs)
