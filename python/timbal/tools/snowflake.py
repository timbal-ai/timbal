"""Snowflake tools over the SQL API v2 (``POST /api/v2/statements``).

Pure ``httpx`` — no ``snowflake-connector-python``. Key-pair JWTs are signed
with ``pyjwt[crypto]``, which ``mcp`` already pulls in.

Auth (first configured wins): OAuth ``token`` → ``programmatic_access_token``
→ key pair (``user`` + ``private_key``). Password auth is not offered by the
SQL API.

Each value resolves explicit tool field → ``Integration("snowflake")`` →
env var (``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_USER``, ``SNOWFLAKE_PRIVATE_KEY``,
``SNOWFLAKE_PRIVATE_KEY_PATH``, ``SNOWFLAKE_PRIVATE_KEY_PASSPHRASE``,
``SNOWFLAKE_TOKEN``, ``SNOWFLAKE_PAT``, ``SNOWFLAKE_HOST``, ``SNOWFLAKE_ROLE``,
``SNOWFLAKE_WAREHOUSE``, ``SNOWFLAKE_DATABASE``, ``SNOWFLAKE_SCHEMA``).

Integration credentials (type: credentials):
- account: account identifier (``myorg-myaccount`` preferred; legacy
  ``xy12345.eu-west-1`` locators work too)
- user: user the key pair / token belongs to
- private_key: PKCS#8 PEM (optionally encrypted)
- private_key_passphrase: passphrase for an encrypted ``private_key``
- token / programmatic_access_token: alternatives to the key pair
- host: hostname override (PrivateLink)
- role, warehouse, database, schema: session defaults

The Snowflake role is the real permission boundary. ``SnowflakeQuery``'s
read-only check is a guardrail against an agent picking the wrong tool, not a
security control; give the service user a role with only the grants it needs.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration

_USER_AGENT = "timbal-snowflake/1"
_EPOCH = datetime(1970, 1, 1)
_EPOCH_DATE = date(1970, 1, 1)
_JWT_LIFETIME_SECS = 59 * 60
_JWT_REFRESH_MARGIN_SECS = 5 * 60
_MAX_RESULT_ROWS = 10_000
_MAX_INSERT_ROWS = 1_000
_RETRYABLE_STATUS = {429, 502, 503, 504}

_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_READ_ONLY_KEYWORDS = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "LIST", "LS"}

# (account, user, fingerprint) -> (jwt, expires_at epoch secs)
_JWT_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}


class SnowflakeError(RuntimeError):
    """A statement the SQL API rejected, carrying Snowflake's own error fields."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        sql_state: str | None = None,
        statement_handle: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.sql_state = sql_state
        self.statement_handle = statement_handle
        super().__init__(message)


# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    text = str(value).strip()
    return text or None


def _parse_account(account: str) -> tuple[str, str]:
    """Return ``(jwt_account, host)`` for an account identifier or URL.

    JWT claims want the bare account: ``ORG-ACCOUNT`` or the locator without
    region/cloud/privatelink segments. The host keeps everything the user gave.
    """
    raw = account.strip().lower()
    raw = raw.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
    raw = raw.removesuffix(".snowflakecomputing.com")
    if not raw:
        raise ValueError("Snowflake account identifier is empty.")

    segments = raw.split(".")
    first = segments[0]
    if ".global" in raw:
        # Replication `locator-<suffix>.global` hosts sign as the bare locator.
        jwt_account = first.split("-", 1)[0]
        host = raw
    elif "-" in first or len(segments) == 1:
        # `org-account[.privatelink]` or a bare legacy locator.
        jwt_account = first
        host = raw
    elif "-" in segments[1]:
        # `locator.region[.cloud][.privatelink]`.
        jwt_account = first
        host = raw
    else:
        # `org.account` — the SQL-style spelling of `org-account`.
        jwt_account = f"{first}-{segments[1]}"
        host = ".".join([jwt_account, *segments[2:]])

    # Hostnames can't carry underscores; Snowflake serves account names with
    # them on the hyphenated host. The JWT keeps the real (underscored) name.
    return jwt_account.upper(), f"{host.replace('_', '-')}.snowflakecomputing.com"


async def _resolve_connection(tool: Any) -> dict[str, Any]:
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve() or {}

    def pick(attr: str, keys: tuple[str, ...], env_vars: tuple[str, ...]) -> str | None:
        value = _secret_value(getattr(tool, attr, None))
        if value:
            return value
        for key in keys:
            value = _secret_value(creds.get(key))
            if value:
                return value
        for env_var in env_vars:
            value = _secret_value(os.getenv(env_var))
            if value:
                return value
        return None

    account = pick("account", ("account", "account_identifier"), ("SNOWFLAKE_ACCOUNT",))
    host_override = pick("host", ("host",), ("SNOWFLAKE_HOST",))
    user = pick("user", ("user", "username"), ("SNOWFLAKE_USER",))
    token = pick("token", ("token", "access_token"), ("SNOWFLAKE_TOKEN",))
    pat = pick(
        "programmatic_access_token",
        ("programmatic_access_token", "pat"),
        ("SNOWFLAKE_PAT", "SNOWFLAKE_PROGRAMMATIC_ACCESS_TOKEN"),
    )
    private_key = pick("private_key", ("private_key",), ("SNOWFLAKE_PRIVATE_KEY",))
    if private_key is None:
        key_path = _secret_value(os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"))
        if key_path:
            private_key = Path(key_path).expanduser().read_text(encoding="utf-8")
    passphrase = pick(
        "private_key_passphrase", ("private_key_passphrase", "passphrase"), ("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",)
    )

    if not account and not host_override:
        raise CredentialNotAvailable("Snowflake", missing=["account"], env_vars=["SNOWFLAKE_ACCOUNT"])

    if account:
        jwt_account, host = _parse_account(account)
    else:
        jwt_account, host = _parse_account(host_override or "")
    if host_override:
        host = host_override.removeprefix("https://").removeprefix("http://").split("/", 1)[0]

    if token:
        auth = ("OAUTH", token)
    elif pat:
        auth = ("PROGRAMMATIC_ACCESS_TOKEN", pat)
    elif private_key:
        if not user:
            raise CredentialNotAvailable("Snowflake", missing=["user"], env_vars=["SNOWFLAKE_USER"])
        auth = ("KEYPAIR_JWT", _keypair_jwt(jwt_account, user, private_key, passphrase))
    else:
        raise CredentialNotAvailable(
            "Snowflake",
            missing=["private_key", "programmatic_access_token", "token"],
            env_vars=["SNOWFLAKE_PRIVATE_KEY", "SNOWFLAKE_PAT", "SNOWFLAKE_TOKEN"],
        )

    return {
        "host": host,
        "token_type": auth[0],
        "token": auth[1],
        "role": pick("role", ("role",), ("SNOWFLAKE_ROLE",)),
        "warehouse": pick("warehouse", ("warehouse",), ("SNOWFLAKE_WAREHOUSE",)),
        "database": pick("database", ("database",), ("SNOWFLAKE_DATABASE",)),
        "schema": pick("schema_name", ("schema", "schema_name"), ("SNOWFLAKE_SCHEMA",)),
    }


# ---------------------------------------------------------------------------
# Key-pair JWT
# ---------------------------------------------------------------------------


def _normalize_pem(pem: str, encrypted: bool) -> str:
    """Repair keys mangled by single-line form fields or env vars."""
    text = pem.strip().replace("\\n", "\n")
    match = re.match(r"^(-----BEGIN [A-Z ]+-----)(.*?)(-----END [A-Z ]+-----)$", text, re.S)
    if match:
        header, body, footer = match.groups()
    else:
        kind = "ENCRYPTED PRIVATE KEY" if encrypted else "PRIVATE KEY"
        header, body, footer = f"-----BEGIN {kind}-----", text, f"-----END {kind}-----"
    body = "".join(body.split())
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return "\n".join([header, *lines, footer]) + "\n"


def _load_private_key(pem: str, passphrase: str | None) -> Any:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    data = _normalize_pem(pem, encrypted=bool(passphrase)).encode()
    try:
        key = serialization.load_pem_private_key(data, password=passphrase.encode() if passphrase else None)
    except TypeError as e:
        # cryptography raises TypeError for "encrypted but no password" and the reverse.
        if passphrase:
            raise ValueError("Snowflake private_key is not encrypted; remove private_key_passphrase.") from e
        raise ValueError("Snowflake private_key is encrypted; set private_key_passphrase.") from e
    except ValueError as e:
        raise ValueError(
            "Snowflake private_key could not be parsed. Expected a PKCS#8 PEM "
            "(-----BEGIN PRIVATE KEY----- or -----BEGIN ENCRYPTED PRIVATE KEY-----), "
            "or the passphrase is wrong."
        ) from e
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("Snowflake key-pair auth requires an RSA private key.")
    return key


def _public_key_fingerprint(key: Any) -> str:
    """``SHA256:<base64>`` of the DER SubjectPublicKeyInfo — matches ``DESC USER``'s RSA_PUBLIC_KEY_FP."""
    from cryptography.hazmat.primitives import serialization

    der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "SHA256:" + base64.b64encode(hashlib.sha256(der).digest()).decode()


def _keypair_jwt(account: str, user: str, private_key: str, passphrase: str | None) -> str:
    import jwt

    key = _load_private_key(private_key, passphrase)
    fingerprint = _public_key_fingerprint(key)
    qualified_user = f"{account}.{user.upper()}"
    cache_key = (account, user.upper(), fingerprint)

    now = time.time()
    cached = _JWT_CACHE.get(cache_key)
    if cached and cached[1] - now > _JWT_REFRESH_MARGIN_SECS:
        return cached[0]

    issued_at = int(now)
    expires_at = issued_at + _JWT_LIFETIME_SECS
    token = jwt.encode(
        {"iss": f"{qualified_user}.{fingerprint}", "sub": qualified_user, "iat": issued_at, "exp": expires_at},
        key,
        algorithm="RS256",
    )
    _JWT_CACHE[cache_key] = (token, expires_at)
    return token


# ---------------------------------------------------------------------------
# Identifiers, bindings, value decoding
# ---------------------------------------------------------------------------


def _split_identifier(name: str) -> list[tuple[str, bool]]:
    """Split ``db.schema."Mixed Case"`` into ``[(part, quoted), ...]``."""
    parts: list[tuple[str, bool]] = []
    i, text = 0, name.strip()
    while i < len(text):
        if text[i] == '"':
            j, buf = i + 1, []
            while True:
                if j >= len(text):
                    raise ValueError(f"Unterminated quoted identifier in {name!r}.")
                if text[j] == '"':
                    if j + 1 < len(text) and text[j + 1] == '"':
                        buf.append('"')
                        j += 2
                        continue
                    break
                buf.append(text[j])
                j += 1
            parts.append(("".join(buf), True))
            i = j + 1
        else:
            j = text.find(".", i)
            j = len(text) if j == -1 else j
            part = text[i:j].strip()
            if not _SIMPLE_IDENT.match(part):
                raise ValueError(f"Invalid identifier {part!r} in {name!r}; wrap it in double quotes.")
            parts.append((part, False))
            i = j
        if i < len(text):
            if text[i] != ".":
                raise ValueError(f"Unexpected character {text[i]!r} in identifier {name!r}.")
            i += 1
            if i == len(text):
                raise ValueError(f"Identifier {name!r} ends with '.'.")
    if not parts:
        raise ValueError("Identifier cannot be empty.")
    return parts


def _render_part(value: str, quoted: bool) -> str:
    return '"' + value.replace('"', '""') + '"' if quoted else value


def _render_identifier(name: str, max_parts: int = 3) -> str:
    """Safe SQL for a (possibly qualified) identifier. Unquoted parts keep Snowflake's case folding."""
    parts = _split_identifier(name)
    if len(parts) > max_parts:
        raise ValueError(f"Identifier {name!r} has more than {max_parts} parts.")
    return ".".join(_render_part(value, quoted) for value, quoted in parts)


def _context_name(value: str) -> str:
    """Session context (role/warehouse/database/schema) is matched case-sensitively by the SQL API."""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text.upper() if _SIMPLE_IDENT.match(text) else text


def _catalog_name(value: str, quoted: bool) -> str:
    """How a name part is stored in INFORMATION_SCHEMA (unquoted identifiers fold to upper case)."""
    return value if quoted else value.upper()


def _binding(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "TEXT", "value": None}
    if isinstance(value, bool):
        return {"type": "BOOLEAN", "value": "true" if value else "false"}
    if isinstance(value, int):
        return {"type": "FIXED", "value": str(value)}
    if isinstance(value, float):
        return {"type": "REAL", "value": repr(value)}
    if isinstance(value, (dict, list)):
        return {"type": "TEXT", "value": json.dumps(value)}
    return {"type": "TEXT", "value": str(value)}


def _bindings(values: list[Any] | None) -> dict[str, Any] | None:
    if not values:
        return None
    return {str(i): _binding(v) for i, v in enumerate(values, start=1)}


def _epoch_micros(value: str) -> int:
    return int((Decimal(value) * 1_000_000).to_integral_value(rounding=ROUND_FLOOR))


def _decode_value(value: Any, column: dict[str, Any]) -> Any:
    """SQL API cells are strings; dates/times arrive as epoch offsets."""
    if value is None:
        return None
    kind = str(column.get("type", "")).lower()
    try:
        if kind == "fixed":
            return int(value) if not column.get("scale") else float(value)
        if kind == "real":
            return float(value)
        if kind == "boolean":
            return str(value).lower() in ("true", "1")
        if kind == "date":
            return (_EPOCH_DATE + timedelta(days=int(value))).isoformat()
        if kind == "time":
            return (datetime.min + timedelta(microseconds=_epoch_micros(value))).time().isoformat()
        if kind == "timestamp_ntz":
            return (_EPOCH + timedelta(microseconds=_epoch_micros(value))).isoformat()
        if kind == "timestamp_ltz":
            return (_EPOCH + timedelta(microseconds=_epoch_micros(value))).replace(tzinfo=UTC).isoformat()
        if kind == "timestamp_tz":
            seconds, offset = str(value).split()
            tz = timezone(timedelta(minutes=int(offset) - 1440))
            return (
                (_EPOCH + timedelta(microseconds=_epoch_micros(seconds))).replace(tzinfo=UTC).astimezone(tz).isoformat()
            )
        if kind in ("variant", "object", "array"):
            return json.loads(value)
    except (ValueError, InvalidOperation, OverflowError, json.JSONDecodeError):
        return value
    return value


def _strip_leading_comments(sql: str) -> str:
    text = sql.lstrip()
    while True:
        if text.startswith("--"):
            newline = text.find("\n")
            text = "" if newline == -1 else text[newline + 1 :].lstrip()
        elif text.startswith("/*"):
            end = text.find("*/")
            text = "" if end == -1 else text[end + 2 :].lstrip()
        elif text.startswith("("):
            text = text[1:].lstrip()
        else:
            return text


def _is_read_only(sql: str) -> bool:
    match = re.match(r"[A-Za-z]+", _strip_leading_comments(sql))
    return bool(match) and match.group(0).upper() in _READ_ONLY_KEYWORDS


# ---------------------------------------------------------------------------
# SQL API transport
# ---------------------------------------------------------------------------


def _error_from_response(response: Any) -> SnowflakeError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    message = body.get("message") or response.text or f"HTTP {response.status_code}"
    code = body.get("code")
    sql_state = body.get("sqlState")
    detail = f"Snowflake error {code}" if code else f"Snowflake HTTP {response.status_code}"
    if sql_state:
        detail += f" (SQLSTATE {sql_state})"
    hint = ""
    if code == "390422":
        hint = " A Snowflake network policy blocks this IP; add it to the user's or account's allowed list."
    elif response.status_code == 401:
        hint = (
            " Check that the account identifier and user are right and, for key-pair auth, that the user's "
            "RSA_PUBLIC_KEY matches private_key (compare `DESC USER` RSA_PUBLIC_KEY_FP)."
        )
    elif response.status_code == 403:
        hint = " The user or role lacks access, or a network policy blocks this IP."
    return SnowflakeError(
        f"{detail}: {message}{hint}",
        status_code=response.status_code,
        code=code,
        sql_state=sql_state,
        statement_handle=body.get("statementHandle"),
    )


async def _request(client: Any, method: str, url: str, **kwargs: Any) -> Any:
    delay = 1.0
    for attempt in range(4):
        response = await client.request(method, url, **kwargs)
        if response.status_code not in _RETRYABLE_STATUS or attempt == 3:
            return response
        retry_after = response.headers.get("Retry-After")
        await asyncio.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
        delay *= 2
        if method == "POST" and "params" in kwargs:
            kwargs["params"] = {**kwargs["params"], "retry": "true"}
    return response


async def _cancel(client: Any, base_url: str, handle: str | None) -> None:
    if not handle:
        return
    try:
        await client.post(f"{base_url}/{handle}/cancel")
    except Exception:  # noqa: BLE001 — best effort; the caller is already failing
        pass


async def _execute(
    conn: dict[str, Any],
    statement: str,
    *,
    bindings: list[Any] | None = None,
    timeout: int = 60,
    max_rows: int = 200,
    database: str | None = None,
    schema: str | None = None,
    warehouse: str | None = None,
) -> dict[str, Any]:
    import httpx

    statement = statement.strip()
    if not statement:
        raise ValueError("SQL statement cannot be empty.")

    base_url = f"https://{conn['host']}/api/v2/statements"
    headers = {
        "Authorization": f"Bearer {conn['token']}",
        "X-Snowflake-Authorization-Token-Type": conn["token_type"],
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }
    body: dict[str, Any] = {"statement": statement, "timeout": timeout}
    for key, value in (
        ("database", database or conn["database"]),
        ("schema", schema or conn["schema"]),
        ("warehouse", warehouse or conn["warehouse"]),
        ("role", conn["role"]),
    ):
        if value:
            body[key] = _context_name(value)
    if bound := _bindings(bindings):
        body["bindings"] = bound

    deadline = time.monotonic() + timeout + 30
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 30.0, connect=15.0), headers=headers) as client:
        response = await _request(client, "POST", base_url, params={"requestId": str(uuid.uuid4())}, json=body)

        poll_delay = 0.5
        while response.status_code == 202:
            handle = response.json().get("statementHandle")
            if time.monotonic() > deadline:
                await _cancel(client, base_url, handle)
                raise TimeoutError(f"Snowflake statement {handle} did not finish within {timeout}s.")
            try:
                await asyncio.sleep(poll_delay)
                poll_delay = min(poll_delay * 2, 4.0)
                response = await _request(client, "GET", f"{base_url}/{handle}")
            except asyncio.CancelledError:
                # The warehouse keeps billing for an orphaned statement.
                await asyncio.shield(_cancel(client, base_url, handle))
                raise

        if response.status_code != 200:
            raise _error_from_response(response)

        result = response.json()
        meta = result.get("resultSetMetaData") or {}
        columns = meta.get("rowType") or []
        raw_rows: list[list[Any]] = list(result.get("data") or [])
        handle = result.get("statementHandle")

        partitions = meta.get("partitionInfo") or []
        for index in range(1, len(partitions)):
            if len(raw_rows) >= max_rows:
                break
            part = await _request(client, "GET", f"{base_url}/{handle}", params={"partition": index})
            if part.status_code != 200:
                raise _error_from_response(part)
            raw_rows.extend(part.json().get("data") or [])

    total = meta.get("numRows", len(raw_rows))
    names = [c.get("name") for c in columns]
    rows = [
        {name: _decode_value(cell, column) for name, cell, column in zip(names, row, columns, strict=False)}
        for row in raw_rows[:max_rows]
    ]
    out: dict[str, Any] = {
        "columns": [{"name": c.get("name"), "type": c.get("type"), "nullable": c.get("nullable")} for c in columns],
        "rows": rows,
        "row_count": total,
        "truncated": total > len(rows),
        "statement_handle": handle,
    }
    if result.get("stats"):
        out["stats"] = result["stats"]
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _max_rows_field(default: int = 200) -> Any:
    return Field(default, ge=1, le=_MAX_RESULT_ROWS, description="Maximum rows to return.")


def _timeout_field() -> Any:
    return Field(60, ge=1, le=3600, description="Statement timeout in seconds.")


class _SnowflakeTool(Tool):
    """Shared connection fields for every Snowflake tool."""

    integration: Annotated[str, Integration("snowflake")] | None = None
    account: str | None = None
    user: str | None = None
    private_key: SecretStr | None = None
    private_key_passphrase: SecretStr | None = None
    token: SecretStr | None = None
    programmatic_access_token: SecretStr | None = None
    host: str | None = None
    role: str | None = None
    warehouse: str | None = None
    database: str | None = None
    schema_name: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "account": self.account,
                    "user": self.user,
                    "private_key": self.private_key,
                    "private_key_passphrase": self.private_key_passphrase,
                    "token": self.token,
                    "programmatic_access_token": self.programmatic_access_token,
                    "host": self.host,
                    "role": self.role,
                    "warehouse": self.warehouse,
                    "database": self.database,
                    "schema_name": self.schema_name,
                }
            ),
        }


class SnowflakeQuery(_SnowflakeTool):
    name: str = "snowflake_query"
    description: str | None = (
        "Run a read-only Snowflake SQL statement (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN, LIST) and return rows. "
        "One statement per call. Use ? placeholders with bindings for values."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _snowflake_query(
            query: str = Field(..., description="A single read-only Snowflake SQL statement."),
            bindings: list[Any] | None = Field(None, description="Values for ? placeholders, in order."),
            max_rows: int = _max_rows_field(),
            timeout: int = _timeout_field(),
            database: str | None = Field(None, description="Override the default database."),
            schema_name: str | None = Field(None, description="Override the default schema."),
            warehouse: str | None = Field(None, description="Override the default warehouse."),
        ) -> Any:
            if not _is_read_only(query):
                raise ValueError(
                    "snowflake_query only runs read-only statements (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN, LIST). "
                    "Use snowflake_execute for writes or DDL."
                )
            conn = await _resolve_connection(self)
            return await _execute(
                conn,
                query,
                bindings=bindings,
                timeout=timeout,
                max_rows=max_rows,
                database=database,
                schema=schema_name,
                warehouse=warehouse,
            )

        super().__init__(handler=_snowflake_query, **kwargs)


class SnowflakeExecute(_SnowflakeTool):
    name: str = "snowflake_execute"
    description: str | None = (
        "Run any single Snowflake SQL statement, including INSERT, UPDATE, DELETE, MERGE, COPY INTO and DDL. "
        "Use ? placeholders with bindings for values. Prefer snowflake_query for reads."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _snowflake_execute(
            statement: str = Field(..., description="A single Snowflake SQL statement."),
            bindings: list[Any] | None = Field(None, description="Values for ? placeholders, in order."),
            max_rows: int = _max_rows_field(),
            timeout: int = _timeout_field(),
            database: str | None = Field(None, description="Override the default database."),
            schema_name: str | None = Field(None, description="Override the default schema."),
            warehouse: str | None = Field(None, description="Override the default warehouse."),
        ) -> Any:
            conn = await _resolve_connection(self)
            return await _execute(
                conn,
                statement,
                bindings=bindings,
                timeout=timeout,
                max_rows=max_rows,
                database=database,
                schema=schema_name,
                warehouse=warehouse,
            )

        super().__init__(handler=_snowflake_execute, **kwargs)


class SnowflakeListTables(_SnowflakeTool):
    name: str = "snowflake_list_tables"
    description: str | None = "List tables and views in a Snowflake database with type, row count, size and comment."

    def __init__(self, **kwargs: Any) -> None:
        async def _snowflake_list_tables(
            database: str | None = Field(None, description="Database to inspect. Defaults to the configured one."),
            schema_name: str | None = Field(None, description="Only list objects in this schema."),
            name_like: str | None = Field(None, description="Case-insensitive name filter, e.g. '%ORDER%'."),
            max_rows: int = _max_rows_field(500),
        ) -> Any:
            conn = await _resolve_connection(self)
            db = database or conn["database"]
            if not db:
                raise ValueError("No database given and none configured. Pass database, e.g. 'ANALYTICS'.")
            db_sql = _render_identifier(db, max_parts=1)

            sql = (
                "SELECT table_schema, table_name, table_type, row_count, bytes, comment "
                f"FROM {db_sql}.INFORMATION_SCHEMA.TABLES WHERE table_schema <> 'INFORMATION_SCHEMA'"
            )
            binds: list[Any] = []
            if schema_name:
                [(schema_value, quoted)] = _split_identifier(schema_name)
                sql += " AND table_schema = ?"
                binds.append(_catalog_name(schema_value, quoted))
            if name_like:
                sql += " AND table_name ILIKE ?"
                binds.append(name_like)
            sql += " ORDER BY table_schema, table_name"
            return await _execute(conn, sql, bindings=binds, max_rows=max_rows)

        super().__init__(handler=_snowflake_list_tables, **kwargs)


class SnowflakeDescribeTable(_SnowflakeTool):
    name: str = "snowflake_describe_table"
    description: str | None = (
        "Show the columns of a Snowflake table or view: name, type, nullability, default and comment."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _snowflake_describe_table(
            table: str = Field(..., description="Table or view name: TABLE, SCHEMA.TABLE or DATABASE.SCHEMA.TABLE."),
        ) -> Any:
            conn = await _resolve_connection(self)
            parts = _split_identifier(table)
            if len(parts) > 3:
                raise ValueError(f"Table name {table!r} has more than 3 parts.")
            table_part = parts[-1]
            schema_part = parts[-2] if len(parts) >= 2 else None
            db_part = parts[-3] if len(parts) == 3 else None

            if db_part:
                db_sql = _render_part(*db_part)
            elif conn["database"]:
                db_sql = _render_identifier(conn["database"], max_parts=1)
            else:
                raise ValueError("No database in the table name and none configured. Use DATABASE.SCHEMA.TABLE.")

            if schema_part:
                schema_value = _catalog_name(*schema_part)
            elif conn["schema"]:
                [(value, quoted)] = _split_identifier(conn["schema"])
                schema_value = _catalog_name(value, quoted)
            else:
                raise ValueError("No schema in the table name and none configured. Use SCHEMA.TABLE.")

            sql = (
                "SELECT column_name, data_type, is_nullable, column_default, comment, "
                "character_maximum_length, numeric_precision, numeric_scale "
                f"FROM {db_sql}.INFORMATION_SCHEMA.COLUMNS "
                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position"
            )
            result = await _execute(
                conn, sql, bindings=[schema_value, _catalog_name(*table_part)], max_rows=_MAX_RESULT_ROWS
            )
            if not result["rows"]:
                raise ValueError(f"Table {table!r} not found, or the role {conn['role'] or '(default)'} cannot see it.")
            return result

        super().__init__(handler=_snowflake_describe_table, **kwargs)


class SnowflakeInsertRows(_SnowflakeTool):
    name: str = "snowflake_insert_rows"
    description: str | None = (
        f"Insert up to {_MAX_INSERT_ROWS} rows into a Snowflake table. Every row must have the same keys."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _snowflake_insert_rows(
            table: str = Field(..., description="Target table: TABLE, SCHEMA.TABLE or DATABASE.SCHEMA.TABLE."),
            rows: list[dict[str, Any]] = Field(..., description="Rows as column -> value objects."),
        ) -> Any:
            if not rows:
                raise ValueError("rows cannot be empty.")
            if len(rows) > _MAX_INSERT_ROWS:
                raise ValueError(f"At most {_MAX_INSERT_ROWS} rows per call; got {len(rows)}.")
            columns = list(rows[0].keys())
            if not columns:
                raise ValueError("rows must contain at least one column.")
            for index, row in enumerate(rows):
                if set(row.keys()) != set(columns):
                    raise ValueError(f"Row {index} has different columns than row 0.")

            columns_sql = ", ".join(_render_identifier(column, max_parts=1) for column in columns)
            placeholders = "(" + ", ".join("?" for _ in columns) + ")"
            sql = (
                f"INSERT INTO {_render_identifier(table)} ({columns_sql}) "
                f"VALUES {', '.join(placeholders for _ in rows)}"
            )
            binds = [row[column] for row in rows for column in columns]
            conn = await _resolve_connection(self)
            return await _execute(conn, sql, bindings=binds)

        super().__init__(handler=_snowflake_insert_rows, **kwargs)
