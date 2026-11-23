"""SAP HANA Cloud and published ABAP SQL service bindings."""

import base64
import os
import re
from collections.abc import Callable
from contextlib import closing
from datetime import date, time
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, model_validator

from ...platform.integrations import Integration
from ._access import Ref, SAPCall, SAPTool, Scalar, json_size, parse_host

_QUOTED = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*" """, re.VERBOSE)
_MUTATIONS = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|UPSERT|MERGE|CALL|INTO|ALTER|CREATE|DROP|TRUNCATE|EXEC|EXECUTE|NEXTVAL)\b", re.I
)


def _statement(sql: str, *, read: bool) -> str:
    sql = sql.strip().removesuffix(";").rstrip()
    code = _QUOTED.sub(" ", sql)
    if any(mark in code for mark in (";", "--", "/*", "*/", "'", '"', "\0")):
        raise ValueError("Use one SQL statement without comments.")
    if read:
        if not re.match(r"^(SELECT|WITH)\b", code.lstrip(), re.I) or _MUTATIONS.search(code):
            raise ValueError("A single read-only SELECT or WITH statement is required.")
    elif not re.match(r"^(INSERT|UPDATE|DELETE|UPSERT|MERGE)\b", code.lstrip(), re.I):
        raise ValueError("Register a single parameterized DML statement.")
    return sql


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _value(value: Any, budget: int) -> Any:
    if hasattr(value, "read"):
        # hdbcli fetchone(True) supplies streaming LOB handles. Read at most a
        # page plus a sentinel rather than materializing an entire CLOB/BLOB.
        with closing(value):
            value = value.read(size=budget + 1)
    if isinstance(value, date | time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes | memoryview):
        return base64.b64encode(value).decode("ascii")
    return value


def _page(call: SAPCall, cursor: Any, size: int, offset: int | None, *, hana: bool = False) -> dict:
    columns = [column[0] for column in cursor.description]
    rows = []
    result = {"columns": columns, "rows": rows, "truncated": False}
    if offset is not None:
        result["next_offset"] = None
    used = json_size(result)
    while True:
        call.check_active()
        record = cursor.fetchone(True) if hana else cursor.fetchone()
        if record is None:
            return result
        next_offset = offset + len(rows) if offset is not None else None
        if len(rows) < size:
            row = [_value(v, call.access.max_bytes) for v in record]
            added = json_size(row) + bool(rows)
            # Leave room for the continuation field if the next row does not fit.
            reserve = max(0, len(str(next_offset + 1)) - 5) if next_offset is not None else 0
            if used + added + reserve <= call.access.max_bytes:
                rows.append(row)
                used += added
                continue
        if not rows:
            raise ValueError("One SAP result row exceeds the configured byte limit.")
        result["truncated"] = True
        if offset is not None:
            result["next_offset"] = next_offset
        return result


class _HANATool(SAPTool):
    # Optional with a bound connection, which names the host; when both are set they must match.
    host: Ref | None = None
    # Without a bound connection: the dedicated user, else SAP_HANA_USER / SAP_HANA_PASSWORD.
    user: str | None = Field(default=None, exclude=True)
    password: SecretStr | None = Field(default=None, exclude=True)
    # A ``sap_hana`` connection holding host, user and password.
    integration: Annotated[str, Integration("sap_hana")] | None = None

    @model_validator(mode="after")
    def _check_credential_source(self):
        if self.integration is not None and (self.user or self.password):
            raise ValueError("The bound sap_hana connection supplies the HANA user and password; do not set them.")
        if self.integration is None and not self.host:
            raise ValueError("Configure the HANA host, or bind a sap_hana connection.")
        return self

    async def _sql(
        self, call: SAPCall, sql: str, params: list, *, read: bool, size: int = 0, offset: int | None = None
    ) -> dict:
        call.account_input([sql, params])
        if self.integration is not None:
            creds = await self.integration.resolve()
            host, _ = parse_host(creds.get("host"))
            user, password = creds.get("user"), creds.get("password")
            if self.host and self.host.lower() != host:
                raise PermissionError("The configured HANA host differs from the connection's host.")
        else:
            host = self.host
            user = self.user or os.getenv("SAP_HANA_USER")
            password = self.password.get_secret_value() if self.password else os.getenv("SAP_HANA_PASSWORD")
        if not user or not password:
            raise ValueError("Configure the dedicated SAP HANA user and password.")

        def run():
            try:
                from hdbcli import dbapi
            except ImportError as exc:
                raise ImportError("Install SAP's hdbcli package to use HANA Cloud tools.") from exc
            call.check_active()
            timeout_ms = self.access.timeout_seconds * 1_000
            with closing(
                dbapi.connect(
                    address=host,
                    port=443,
                    user=user,
                    password=password,
                    encrypt=True,
                    sslValidateCertificate=True,
                    sslMinProtocolVersion="TLS12",
                    connectTimeout=timeout_ms,
                    communicationTimeout=timeout_ms,
                )
            ) as conn:
                conn.setautocommit(False)
                # No APPLICATIONUSER: the platform does not yet supply a user SAP may trust.
                conn.setclientinfo("APPLICATION", call.application)
                with closing(conn.cursor()) as cursor:
                    if read:
                        cursor.execute("SET TRANSACTION READ ONLY")
                    call.check_active()
                    cursor.execute(sql, tuple(params))
                    if read:
                        return _page(call, cursor, size, offset, hana=True)
                    call.check_active()
                    count = cursor.rowcount
                    conn.commit()
                    return {"affected_rows": count}

        return await call.blocking(run)


class SAPHANARead(_HANATool):
    """A paged projection of one configured HANA Cloud view."""

    name: str = "sap_hana_read"
    description: str = "Read a page from the configured SAP HANA Cloud view using equality filters."
    read_only: Literal[True] = True
    schema_name: Ref
    view: Ref
    columns: Annotated[tuple[Ref, ...], Field(min_length=1)]
    order_by: Annotated[tuple[Ref, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _check_order(self):
        if not set(self.order_by) <= set(self.columns):
            raise ValueError("order_by must use selected columns and uniquely order the view.")
        return self

    async def _invoke(
        self, filters: dict[str, Scalar] | None = None, limit: int | None = None, offset: int = 0
    ) -> dict:
        target = f"{_ident(self.schema_name)}.{_ident(self.view)}"

        async def run(call):
            size = self.access.max_rows if limit is None else limit
            if not 1 <= size <= self.access.max_rows or offset < 0:
                raise ValueError("Invalid SAP page size or offset.")
            predicates, params = [], []
            for key, value in (filters or {}).items():
                if key not in self.columns:
                    raise ValueError("Filter column is outside the configured projection.")
                predicates.append(_ident(key) + (" IS NULL" if value is None else " = ?"))
                if value is not None:
                    params.append(value)
            sql = f"SELECT {', '.join(map(_ident, self.columns))} FROM {target}"
            if predicates:
                sql += " WHERE " + " AND ".join(predicates)
            sql += f" ORDER BY {', '.join(map(_ident, self.order_by))} LIMIT ? OFFSET ?"
            return await self._sql(call, sql, [*params, size + 1, offset], read=True, size=size, offset=offset)

        return await self.access.invoke(self.name, target, run, route="hana_view")


class SAPHANAQuery(_HANATool):
    name: str = "sap_hana_query"
    description: str = "Execute a single SELECT or WITH query using the scoped read-only HANA user."
    read_only: Literal[True] = True

    async def _invoke(self, query: str, bindings: list[Scalar] | None = None) -> dict:
        async def run(call):
            return await self._sql(
                call, _statement(query, read=True), bindings or [], read=True, size=self.access.max_rows
            )

        return await self.access.invoke(self.name, self.host or "sql", run, route="hana_sql")


class SAPHANAWrite(_HANATool):
    name: str = "sap_hana_write"
    description: str = "Execute the configured HANA Cloud write using positional bindings."
    statement: Ref
    object_name: Ref
    read_only: Literal[False] = False

    async def _invoke(self, bindings: list[Scalar]) -> dict:
        async def run(call):
            return await self._sql(call, _statement(self.statement, read=False), bindings, read=False)

        return await self.access.invoke(self.name, self.object_name, run, route="hana_write", document_types=())


class SAPCDSRead(SAPTool):
    """Prepared CDS query using SAP's ODBC driver for the ABAP SQL service."""

    name: str = "sap_cds_read"
    description: str = "Read the configured published CDS projection with positional bindings."
    read_only: Literal[True] = True
    view: Ref
    statement: Ref
    # The deployment owns the DSN, credential retrieval and TLS settings. A fresh
    # DB-API connection is required each time; the tool owns and closes it.
    connect: Callable[[], Any] = Field(exclude=True)

    async def _invoke(self, bindings: list[Scalar] | None = None) -> dict:
        async def run(call):
            call.account_input(bindings)
            sql = _statement(self.statement, read=True)

            def query():
                call.check_active()
                with closing(self.connect()) as conn:
                    conn.timeout = self.access.timeout_seconds
                    with closing(conn.cursor()) as cursor:
                        call.check_active()
                        cursor.execute(sql, *(bindings or []))
                        return _page(call, cursor, self.access.max_rows, None)

            return await call.blocking(query)

        return await self.access.invoke(self.name, self.view, run, route="cds_view")
