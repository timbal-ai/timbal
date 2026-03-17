from typing import Annotated, Any

import httpx

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


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


class InsertRow(Tool):
    name: str = "postgresql_insert_row"
    description: str | None = "Insert a new row into a PostgreSQL table via PostgREST."
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_row(
            host: str,
            table: str,
            data: dict[str, Any],
        ) -> Any:
            """
            host: base URL of your PostgREST or Supabase instance,
                  e.g. "https://xyz.supabase.co" or "http://localhost:3000".
            table: the table name.
            data: column-value pairs to insert, e.g. {"name": "Alice", "email": "alice@example.com"}.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/InsertRow"
        super().__init__(handler=_insert_row, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class UpsertRow(Tool):
    name: str = "postgresql_upsert_row"
    description: str | None = (
        "Insert a row or update it if a row with the same conflict column already exists."
    )
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert_row(
            host: str,
            table: str,
            data: dict[str, Any],
            on_conflict: str = "id",
        ) -> Any:
            """
            data: column-value pairs for the row.
            on_conflict: comma-separated column(s) that define uniqueness for conflict resolution,
                         e.g. "id" or "email" or "user_id,project_id".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _rest_url(host, table),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": token,
                        "Prefer": f"return=representation,resolution=merge-duplicates",
                        "Resolution": f"merge-duplicates",
                    },
                    params={"on_conflict": on_conflict},
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/UpsertRow"
        super().__init__(handler=_upsert_row, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class UpdateRow(Tool):
    name: str = "postgresql_update_row"
    description: str | None = "Update one or more rows in a PostgreSQL table that match the given filters."
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_row(
            host: str,
            table: str,
            filters: dict[str, Any],
            data: dict[str, Any],
        ) -> Any:
            """
            filters: columns to match, e.g. {"id": 42} or {"status": {"$eq": "pending"}}.
                     Supported operators in value dicts: $eq, $neq, $gt, $gte, $lt, $lte,
                     $like, $ilike, $in, $is.
            data: column-value pairs to set on matching rows.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/UpdateRow"
        super().__init__(handler=_update_row, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------


class FindRow(Tool):
    name: str = "postgresql_find_row"
    description: str | None = (
        "Find a single row in a PostgreSQL table by matching a specific column value."
    )
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_row(
            host: str,
            table: str,
            column: str,
            value: Any,
            select: str = "*",
        ) -> Any:
            """
            column: the column to look up, e.g. "id" or "email".
            value: the value to match.
            select: comma-separated columns to return, e.g. "id,name,email". Defaults to "*".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/FindRow"
        super().__init__(handler=_find_row, metadata=metadata, **kwargs)


class FindRowWithCustomQuery(Tool):
    name: str = "postgresql_find_row_with_custom_query"
    description: str | None = (
        "Find rows in a PostgreSQL table using multiple filter conditions, "
        "custom column selection, ordering, and pagination."
    )
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_row_with_custom_query(
            host: str,
            table: str,
            filters: dict[str, Any] | None = None,
            select: str = "*",
            order: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ) -> Any:
            """
            filters: column conditions, e.g. {"status": "active", "age": {"$gte": 18}}.
                     Supported operators: $eq, $neq, $gt, $gte, $lt, $lte,
                     $like, $ilike, $in, $is. Omit to return all rows.
            select: columns to return, e.g. "id,name,email" or "*".
                    Supports embedded resource joins: "id,name,orders(id,total)".
            order: sort expression, e.g. "created_at.desc" or "name.asc,age.desc".
            limit: max rows to return (default 100).
            offset: rows to skip for pagination.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/FindRowWithCustomQuery"
        super().__init__(handler=_find_row_with_custom_query, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class DeleteRows(Tool):
    name: str = "postgresql_delete_rows"
    description: str | None = "Delete one or more rows from a PostgreSQL table matching the given filters."
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_rows(
            host: str,
            table: str,
            filters: dict[str, Any],
        ) -> Any:
            """
            filters: conditions to select rows for deletion, e.g. {"id": 42} or
                     {"status": "archived", "created_at": {"$lt": "2023-01-01"}}.
                     At least one filter is required to prevent accidental full-table deletes.
            """
            if not filters:
                raise ValueError("At least one filter is required to prevent deleting all rows.")

            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/DeleteRows"
        super().__init__(handler=_delete_rows, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


class ExecuteSQLQuery(Tool):
    name: str = "postgresql_execute_sql_query"
    description: str | None = (
        "Execute a raw SQL SELECT query and return the result rows. "
        "Requires the PostgREST RPC endpoint to expose a sql execution function, "
        "or a Supabase project with the pg_net extension / service-role key."
    )
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _execute_sql_query(
            host: str,
            query: str,
            params: list[Any] | None = None,
            rpc_function: str = "exec_sql",
        ) -> Any:
            """
            host: base URL of your PostgREST / Supabase instance.
            query: the SQL SELECT statement to execute,
                   e.g. "SELECT * FROM users WHERE active = true LIMIT 10".
            params: optional positional parameters for parameterized queries,
                    e.g. ["alice@example.com"] for "WHERE email = $1".
            rpc_function: name of the PostgREST RPC function that executes SQL
                          (default "exec_sql"). The function must accept a "query"
                          text argument and optionally a "params" JSON argument.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/ExecuteSQLQuery"
        super().__init__(handler=_execute_sql_query, metadata=metadata, **kwargs)


class QuerySQLDatabase(Tool):
    name: str = "postgresql_query_sql_database"
    description: str | None = (
        "Execute any SQL statement (INSERT, UPDATE, DELETE, SELECT, DDL) "
        "against a PostgreSQL database via a service-role RPC endpoint."
    )
    integration: Annotated[str, Integration("postgresql")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query_sql_database(
            host: str,
            sql: str,
            rpc_function: str = "exec_sql",
        ) -> Any:
            """
            host: base URL of your PostgREST / Supabase instance.
            sql: any valid SQL statement, e.g.:
                 "INSERT INTO orders (user_id, total) VALUES (1, 99.99)"
                 "UPDATE users SET last_login = NOW() WHERE id = 5"
                 "CREATE INDEX IF NOT EXISTS idx_email ON users(email)"
            rpc_function: name of the PostgREST RPC function that accepts raw SQL
                          (default "exec_sql"). Must be granted execute permission
                          to the authenticated role.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "PostgreSQL/QuerySQLDatabase"
        super().__init__(handler=_query_sql_database, metadata=metadata, **kwargs)
