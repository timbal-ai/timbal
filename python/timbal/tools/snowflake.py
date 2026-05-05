from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


def _normalize_account(account: str) -> str:
    raw = account.strip()
    raw = raw.removeprefix("https://").removeprefix("http://")
    raw = raw.removesuffix(".snowflakecomputing.com")
    return raw.strip("/")


async def _resolve_connection(tool: Any) -> dict[str, str | None]:
    account: str | None = None
    user: str | None = None
    password: str | None = None
    private_key: str | None = None
    private_key_passphrase: str | None = None
    totp: str | None = None
    token: str | None = None
    warehouse: str | None = None
    database: str | None = None
    schema: str | None = None
    role: str | None = None

    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        account = credentials.get("account") or credentials.get("account_identifier")
        user = credentials.get("user") or credentials.get("login_name")
        password = credentials.get("password")
        private_key = credentials.get("private_key")
        private_key_passphrase = credentials.get("private_key_passphrase")
        totp = credentials.get("totp")
        token = credentials.get("token") or credentials.get("access_token")
        warehouse = credentials.get("warehouse")
        database = credentials.get("database")
        schema = credentials.get("schema")
        role = credentials.get("role")

    if account is None:
        account = getattr(tool, "account", None)
    if user is None:
        user = getattr(tool, "user", None)
    if password is None and getattr(tool, "password", None) is not None:
        password = tool.password.get_secret_value()
    if private_key is None and getattr(tool, "private_key", None) is not None:
        private_key = tool.private_key.get_secret_value()
    if private_key_passphrase is None and getattr(tool, "private_key_passphrase", None) is not None:
        private_key_passphrase = tool.private_key_passphrase.get_secret_value()
    if totp is None and getattr(tool, "totp", None) is not None:
        totp = tool.totp.get_secret_value()
    if token is None and getattr(tool, "token", None) is not None:
        token = tool.token.get_secret_value()
    if warehouse is None:
        warehouse = getattr(tool, "warehouse", None)
    if database is None:
        database = getattr(tool, "database", None)
    if schema is None:
        schema = getattr(tool, "schema_name", None)
    if role is None:
        role = getattr(tool, "role", None)

    account = account or os.getenv("SNOWFLAKE_ACCOUNT")
    user = user or os.getenv("SNOWFLAKE_USER")
    password = password or os.getenv("SNOWFLAKE_PASSWORD")
    private_key = private_key or os.getenv("SNOWFLAKE_PRIVATE_KEY")
    private_key_passphrase = private_key_passphrase or os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    key_file = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
    if private_key is None and key_file and key_file.strip():
        private_key = Path(key_file.strip()).expanduser().read_text(encoding="utf-8").strip()

    passphrase_file = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE_FILE")
    if private_key_passphrase is None and passphrase_file and passphrase_file.strip():
        private_key_passphrase = Path(passphrase_file.strip()).expanduser().read_text(encoding="utf-8").strip()
    totp = totp or os.getenv("SNOWFLAKE_TOTP")
    token = token or os.getenv("SNOWFLAKE_TOKEN") or os.getenv("SNOWFLAKE_ACCESS_TOKEN")
    warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE")
    database = database or os.getenv("SNOWFLAKE_DATABASE")
    schema = schema or os.getenv("SNOWFLAKE_SCHEMA")
    role = role or os.getenv("SNOWFLAKE_ROLE")

    if not account:
        raise ValueError(
            "Snowflake account is required. Set account on the integration or SNOWFLAKE_ACCOUNT in the environment."
        )

    has_token = bool(token)
    has_password_or_key = bool(password or private_key)
    if has_token:
        pass
    elif has_password_or_key:
        if not user:
            raise ValueError(
                "Snowflake user is required for password or key-pair authentication. "
                "Set user on the integration or SNOWFLAKE_USER."
            )
    else:
        raise ValueError(
            "Snowflake credentials not found. Configure a snowflake integration with "
            "account plus (OAuth/JWT token or user with password/private_key), set env vars, "
            "or pass these fields on the tool."
        )

    return {
        "account": _normalize_account(account),
        "user": user,
        "password": password,
        "private_key": private_key,
        "private_key_passphrase": private_key_passphrase,
        "totp": totp,
        "token": token,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
        "role": role,
    }


async def _execute_statement(
    tool: Any,
    statement: str,
    *,
    timeout: int = 60,
    database: str | None = None,
    schema_name: str | None = None,
    warehouse: str | None = None,
    role: str | None = None,
) -> Any:
    conn = await _resolve_connection(tool)
    if conn["token"]:
        return await _execute_statement_via_sql_api(
            conn=conn,
            statement=statement,
            timeout=timeout,
            database=database,
            schema_name=schema_name,
            warehouse=warehouse,
            role=role,
        )
    return await _execute_statement_via_connector(
        conn=conn,
        statement=statement,
        database=database,
        schema_name=schema_name,
        warehouse=warehouse,
        role=role,
    )


async def _execute_statement_via_sql_api(
    *,
    conn: dict[str, str | None],
    statement: str,
    timeout: int,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
    role: str | None,
) -> Any:
    import httpx

    body: dict[str, Any] = {
        "statement": statement,
        "timeout": timeout,
    }
    if database or conn["database"]:
        body["database"] = database or conn["database"]
    if schema_name or conn["schema"]:
        body["schema"] = schema_name or conn["schema"]
    if warehouse or conn["warehouse"]:
        body["warehouse"] = warehouse or conn["warehouse"]
    if role or conn["role"]:
        body["role"] = role or conn["role"]

    url = f"https://{conn['account']}.snowflakecomputing.com/api/v2/statements"
    headers = {
        "Authorization": f"Bearer {conn['token']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=body, timeout=httpx.Timeout(120.0))
        response.raise_for_status()
        return response.json()


async def _execute_statement_via_connector(
    *,
    conn: dict[str, str | None],
    statement: str,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
    role: str | None,
) -> Any:
    import asyncio

    return await asyncio.to_thread(
        _execute_statement_via_connector_sync,
        conn,
        statement,
        database,
        schema_name,
        warehouse,
        role,
    )


def _execute_statement_via_connector_sync(
    conn: dict[str, str | None],
    statement: str,
    database: str | None,
    schema_name: str | None,
    warehouse: str | None,
    role: str | None,
) -> Any:
    import snowflake.connector

    connect_kwargs: dict[str, Any] = {
        "account": conn["account"],
        "user": conn["user"],
        "warehouse": warehouse or conn["warehouse"],
        "database": database or conn["database"],
        "schema": schema_name or conn["schema"],
        "role": role or conn["role"],
    }

    if conn["private_key"]:
        private_key_bytes = _decode_private_key(
            private_key_pem=conn["private_key"],
            passphrase=conn["private_key_passphrase"],
        )
        connect_kwargs["private_key"] = private_key_bytes
    else:
        connect_kwargs["password"] = conn["password"]
        if conn["totp"]:
            connect_kwargs["passcode"] = conn["totp"]

    with snowflake.connector.connect(**{k: v for k, v in connect_kwargs.items() if v}) as cnx:
        with cnx.cursor() as cur:
            cur.execute(statement)
            rows = cur.fetchall() if cur.description is not None else []
            columns = [col[0] for col in cur.description] if cur.description else []
            return {
                "columns": columns,
                "rows": rows,
                "rowcount": cur.rowcount,
                "sfqid": cur.sfqid,
            }


def _decode_private_key(*, private_key_pem: str, passphrase: str | None) -> bytes:
    from cryptography.hazmat.primitives import serialization

    try:
        key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=passphrase.encode() if passphrase else None,
        )
    except ValueError as e:
        raise ValueError("Invalid Snowflake private_key PEM or wrong private_key_passphrase.") from e
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _quote_ident(identifier: str) -> str:
    safe = identifier.strip().replace('"', '""')
    return f'"{safe}"'


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _connection_config(tool: Any) -> dict[str, Any]:
    return {
        "integration": tool.integration,
        "account": tool.account,
        "user": tool.user,
        "password": tool.password,
        "private_key": tool.private_key,
        "private_key_passphrase": tool.private_key_passphrase,
        "token": tool.token,
        "warehouse": tool.warehouse,
        "database": tool.database,
        "schema_name": tool.schema_name,
        "role": tool.role,
        "totp": tool.totp,
    }


class SnowflakeDescribeTable(Tool):
    name: str = "snowflake_describe_table"
    description: str | None = "Describe the structure of a Snowflake table."
    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None
    role: str | None = None
    totp: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(_connection_config(self)),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            table_name: str = Field(..., description="Target table name."),
            schema_name: str | None = Field(None, description="Override schema for this request."),
            database: str | None = Field(None, description="Override database for this request."),
        ) -> Any:
            sql = f"DESCRIBE TABLE {_quote_ident(table_name)}"
            return await _execute_statement(self, sql, database=database, schema_name=schema_name)

        super().__init__(handler=_handler, **kwargs)


class SnowflakeExecuteQuery(Tool):
    name: str = "snowflake_execute_query"
    description: str | None = (
        "Execute Snowflake SQL via SQL API when using OAuth/JWT token, "
        "or via the Snowflake connector when using password or key-pair authentication."
    )
    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None
    role: str | None = None
    totp: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(_connection_config(self)),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            query: str = Field(..., description="Valid Snowflake SQL statement."),
            database: str | None = Field(None, description="Override database for this request."),
            schema_name: str | None = Field(None, description="Override schema for this request."),
            warehouse: str | None = Field(None, description="Override warehouse for this request."),
            role: str | None = Field(None, description="Override role for this request."),
            timeout: int = Field(60, description="Query timeout in seconds.", ge=1, le=300),
        ) -> Any:
            statement = query.strip()
            if not statement:
                raise ValueError("query cannot be empty.")
            return await _execute_statement(
                self,
                statement,
                timeout=timeout,
                database=database,
                schema_name=schema_name,
                warehouse=warehouse,
                role=role,
            )

        super().__init__(handler=_handler, **kwargs)


class SnowflakeStageData(Tool):
    name: str = "snowflake_stage_data"
    description: str | None = "Bulk load data into a table with COPY INTO from a Snowflake stage."
    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None
    role: str | None = None
    totp: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(_connection_config(self)),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            table_name: str = Field(..., description="Target table for loading data."),
            stage_path: str = Field(..., description="Snowflake stage path, e.g. @my_stage/path/file.csv."),
            file_format: str = Field(
                "TYPE = CSV SKIP_HEADER = 1",
                description="File format clause contents used inside FILE_FORMAT = (...).",
            ),
            on_error: str = Field("CONTINUE", description="COPY error behavior, e.g. CONTINUE or ABORT_STATEMENT."),
            purge: bool = Field(False, description="If true, purge staged files after successful load."),
        ) -> Any:
            sql = (
                f"COPY INTO {_quote_ident(table_name)} "
                f"FROM {stage_path} "
                f"FILE_FORMAT = ({file_format}) "
                f"ON_ERROR = {_sql_literal(on_error)} "
                f"PURGE = {'TRUE' if purge else 'FALSE'}"
            )
            return await _execute_statement(self, sql)

        super().__init__(handler=_handler, **kwargs)


class SnowflakeInsertSingleRow(Tool):
    name: str = "snowflake_insert_single_row"
    description: str | None = "Insert a single row into a Snowflake table."
    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None
    role: str | None = None
    totp: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(_connection_config(self)),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            table_name: str = Field(..., description="Target table name."),
            row: dict[str, Any] = Field(..., description="Column/value map to insert."),
        ) -> Any:
            if not row:
                raise ValueError("row cannot be empty.")
            columns = ", ".join(_quote_ident(col) for col in row.keys())
            values = ", ".join(_sql_literal(val) for val in row.values())
            sql = f"INSERT INTO {_quote_ident(table_name)} ({columns}) VALUES ({values})"
            return await _execute_statement(self, sql)

        super().__init__(handler=_handler, **kwargs)


class SnowflakeInsertMultipleRows(Tool):
    name: str = "snowflake_insert_multiple_rows"
    description: str | None = "Insert multiple rows into a Snowflake table."
    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None
    role: str | None = None
    totp: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(_connection_config(self)),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            table_name: str = Field(..., description="Target table name."),
            rows: list[dict[str, Any]] = Field(..., description="Rows to insert."),
        ) -> Any:
            if not rows:
                raise ValueError("rows cannot be empty.")
            columns = list(rows[0].keys())
            if not columns:
                raise ValueError("rows must contain at least one column.")
            for row in rows:
                if list(row.keys()) != columns:
                    raise ValueError("All rows must have the same columns in the same order.")
            columns_sql = ", ".join(_quote_ident(col) for col in columns)
            values_sql = ", ".join(f"({', '.join(_sql_literal(row[col]) for col in columns)})" for row in rows)
            sql = f"INSERT INTO {_quote_ident(table_name)} ({columns_sql}) VALUES {values_sql}"
            return await _execute_statement(self, sql)

        super().__init__(handler=_handler, **kwargs)


class SnowflakeQuerySQLDatabase(Tool):
    name: str = "snowflake_query_sql_database"
    description: str | None = "Execute a SQL query against Snowflake (Query SQL Database action)."
    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None
    role: str | None = None
    totp: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(_connection_config(self)),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            query: str = Field(..., description="Valid Snowflake SQL query."),
            timeout: int = Field(60, description="Query timeout in seconds.", ge=1, le=300),
        ) -> Any:
            statement = query.strip()
            if not statement:
                raise ValueError("query cannot be empty.")
            return await _execute_statement(self, statement, timeout=timeout)

        super().__init__(handler=_handler, **kwargs)
