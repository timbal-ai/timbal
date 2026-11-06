"""Tests for Snowflake SQL API tools (mocked; no network)."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
import timbal.tools.snowflake as sf
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.tools.snowflake import (
    SnowflakeAuth,
    SnowflakeError,
    SnowflakeInsertRows,
    SnowflakeQuery,
    _decode_value,
    _execute,
    _is_read_only,
    _keypair_jwt,
    _parse_account,
    _render_identifier,
    _resolve_connection,
    _split_identifier,
)

_ENV_VARS = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PRIVATE_KEY",
    "SNOWFLAKE_PRIVATE_KEY_PATH",
    "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
    "SNOWFLAKE_TOKEN",
    "SNOWFLAKE_PAT",
    "SNOWFLAKE_PROGRAMMATIC_ACCESS_TOKEN",
    "SNOWFLAKE_HOST",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)


@pytest.fixture(autouse=True)
def _clean(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    sf._JWT_CACHE.clear()
    sf._KEY_CACHE.clear()
    if request.node.get_closest_marker("integration"):
        return
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key: rsa.RSAPrivateKey, passphrase: str | None = None) -> str:
    encryption = (
        serialization.BestAvailableEncryption(passphrase.encode()) if passphrase else serialization.NoEncryption()
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    ).decode()


def _expected_fingerprint(key: rsa.RSAPrivateKey) -> str:
    der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "SHA256:" + base64.b64encode(hashlib.sha256(der).digest()).decode()


def _tool(**kwargs: Any) -> SimpleNamespace:
    fields = {
        "integration": None,
        "account": None,
        "user": None,
        "private_key": None,
        "private_key_passphrase": None,
        "token": None,
        "programmatic_access_token": None,
        "host": None,
        "role": None,
        "warehouse": None,
        "database": None,
        "schema_name": None,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _conn(**overrides: Any) -> dict[str, Any]:
    conn = {
        "host": "myorg-acct.snowflakecomputing.com",
        "token_type": "KEYPAIR_JWT",
        "token": "jwt-token",
        "role": "ANALYST",
        "warehouse": "WH",
        "database": "DB",
        "schema": "PUBLIC",
    }
    conn.update(overrides)
    return conn


@pytest.fixture
def mock_api(monkeypatch: pytest.MonkeyPatch):
    """Route httpx.AsyncClient through a handler; record every request."""
    requests: list[httpx.Request] = []

    def install(handler):
        def recording(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        real_client = httpx.AsyncClient

        def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(recording)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        monkeypatch.setattr(sf.asyncio, "sleep", AsyncMock())
        return requests

    return install


# ---------------------------------------------------------------------------
# Account identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("account", "jwt_account", "host"),
    [
        ("myorg-my_account", "MYORG-MY_ACCOUNT", "myorg-my-account.snowflakecomputing.com"),
        ("MYORG-ACCT", "MYORG-ACCT", "myorg-acct.snowflakecomputing.com"),
        ("myorg.acct", "MYORG-ACCT", "myorg-acct.snowflakecomputing.com"),
        ("xy12345", "XY12345", "xy12345.snowflakecomputing.com"),
        ("xy12345.eu-west-1", "XY12345", "xy12345.eu-west-1.snowflakecomputing.com"),
        ("xy12345.us-east-2.aws", "XY12345", "xy12345.us-east-2.aws.snowflakecomputing.com"),
        ("https://myorg-acct.snowflakecomputing.com/", "MYORG-ACCT", "myorg-acct.snowflakecomputing.com"),
        ("myorg-acct.privatelink", "MYORG-ACCT", "myorg-acct.privatelink.snowflakecomputing.com"),
        (
            "xy12345.eu-central-1.privatelink",
            "XY12345",
            "xy12345.eu-central-1.privatelink.snowflakecomputing.com",
        ),
        ("xy12345-abcdef.global", "XY12345", "xy12345-abcdef.global.snowflakecomputing.com"),
    ],
)
def test_parse_account(account: str, jwt_account: str, host: str) -> None:
    assert _parse_account(account) == (jwt_account, host)


# ---------------------------------------------------------------------------
# Key-pair JWT
# ---------------------------------------------------------------------------


def test_keypair_jwt_claims(rsa_key: rsa.RSAPrivateKey) -> None:
    token = _keypair_jwt("MYORG-ACCT", "svc_user", _pem(rsa_key), None)
    claims = jwt.decode(token, rsa_key.public_key(), algorithms=["RS256"])
    assert claims["sub"] == "MYORG-ACCT.SVC_USER"
    assert claims["iss"] == f"MYORG-ACCT.SVC_USER.{_expected_fingerprint(rsa_key)}"
    assert 0 < claims["exp"] - claims["iat"] <= 3600


def test_keypair_jwt_is_cached(rsa_key: rsa.RSAPrivateKey) -> None:
    pem = _pem(rsa_key)
    assert _keypair_jwt("A", "U", pem, None) == _keypair_jwt("A", "U", pem, None)


@pytest.mark.parametrize(
    "mangle",
    [
        lambda pem: pem.replace("\n", "\\n"),
        lambda pem: pem.replace("\n", " "),
        lambda pem: "".join(pem.strip().splitlines()[1:-1]),
    ],
    ids=["escaped-newlines", "single-line", "body-only"],
)
def test_mangled_pem_is_repaired(rsa_key: rsa.RSAPrivateKey, mangle) -> None:
    token = _keypair_jwt("A", "U", mangle(_pem(rsa_key)), None)
    jwt.decode(token, rsa_key.public_key(), algorithms=["RS256"])


def test_encrypted_key(rsa_key: rsa.RSAPrivateKey) -> None:
    pem = _pem(rsa_key, passphrase="s3cret")
    jwt.decode(_keypair_jwt("A", "U", pem, "s3cret"), rsa_key.public_key(), algorithms=["RS256"])
    with pytest.raises(ValueError, match="set private_key_passphrase"):
        _keypair_jwt("A", "U2", pem, None)
    with pytest.raises(ValueError, match="passphrase is wrong"):
        _keypair_jwt("A", "U3", pem, "nope")


def test_passphrase_on_plain_key(rsa_key: rsa.RSAPrivateKey) -> None:
    with pytest.raises(ValueError, match="remove private_key_passphrase"):
        _keypair_jwt("A", "U", _pem(rsa_key), "unneeded")


# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_from_integration(rsa_key: rsa.RSAPrivateKey) -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(
        return_value={
            "account": "myorg-acct",
            "user": "svc",
            "private_key": _pem(rsa_key),
            "role": "RF_ROLE",
            "warehouse": "WH",
            "database": "DB",
            "schema": "SCH",
        }
    )
    conn = await _resolve_connection(_tool(integration=integration))
    assert conn["token_type"] == "KEYPAIR_JWT"
    assert conn["host"] == "myorg-acct.snowflakecomputing.com"
    assert (conn["role"], conn["warehouse"], conn["database"], conn["schema"]) == ("RF_ROLE", "WH", "DB", "SCH")
    claims = jwt.decode(conn["token"], rsa_key.public_key(), algorithms=["RS256"])
    assert claims["sub"] == "MYORG-ACCT.SVC"


@pytest.mark.asyncio
async def test_explicit_fields_beat_integration() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"account": "other", "role": "FROM_INTEGRATION"})
    conn = await _resolve_connection(
        _tool(integration=integration, account="myorg-acct", token=SecretStr("oauth"), role="EXPLICIT")
    )
    assert conn["host"] == "myorg-acct.snowflakecomputing.com"
    assert (conn["token_type"], conn["token"], conn["role"]) == ("OAUTH", "oauth", "EXPLICIT")


@pytest.mark.asyncio
async def test_pat_from_env_and_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "xy12345.eu-west-1")
    monkeypatch.setenv("SNOWFLAKE_PAT", "pat-value")
    monkeypatch.setenv("SNOWFLAKE_HOST", "xy12345.eu-west-1.privatelink.snowflakecomputing.com")
    conn = await _resolve_connection(_tool())
    assert conn["token_type"] == "PROGRAMMATIC_ACCESS_TOKEN"
    assert conn["host"] == "xy12345.eu-west-1.privatelink.snowflakecomputing.com"


@pytest.mark.asyncio
async def test_missing_credentials() -> None:
    with pytest.raises(CredentialNotAvailable):
        await _resolve_connection(_tool())
    with pytest.raises(CredentialNotAvailable):
        await _resolve_connection(_tool(account="myorg-acct"))


@pytest.mark.asyncio
async def test_key_pair_requires_user(rsa_key: rsa.RSAPrivateKey) -> None:
    with pytest.raises(CredentialNotAvailable, match="user"):
        await _resolve_connection(_tool(account="myorg-acct", private_key=SecretStr(_pem(rsa_key))))


# ---------------------------------------------------------------------------
# Identifiers and statement classification
# ---------------------------------------------------------------------------


def test_split_identifier() -> None:
    assert _split_identifier('db.sch."My ""Odd"" Table"') == [
        ("db", False),
        ("sch", False),
        ('My "Odd" Table', True),
    ]
    assert _split_identifier('"a.b".c') == [("a.b", True), ("c", False)]


@pytest.mark.parametrize("bad", ["t; DROP TABLE x", "a..b", "a.", '"open', "a b", ""])
def test_bad_identifiers_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        _render_identifier(bad)


def test_render_identifier() -> None:
    assert _render_identifier('db.sch."x""y"') == 'db.sch."x""y"'
    with pytest.raises(ValueError):
        _render_identifier("a.b.c.d")


@pytest.mark.parametrize(
    "sql",
    [
        "select 1",
        "  WITH x AS (SELECT 1) SELECT * FROM x",
        "-- note\nSHOW TABLES",
        "/* c */ (SELECT 1)",
        "desc table t",
    ],
)
def test_read_only_statements(sql: str) -> None:
    assert _is_read_only(sql)


@pytest.mark.parametrize(
    "sql", ["INSERT INTO t VALUES (1)", "delete from t", "CALL p()", "/* select */ DROP TABLE t", ""]
)
def test_write_statements(sql: str) -> None:
    assert not _is_read_only(sql)


# ---------------------------------------------------------------------------
# Value decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "column", "expected"),
    [
        ("42", {"type": "fixed", "scale": 0}, 42),
        ("12.50", {"type": "fixed", "scale": 2}, 12.5),
        ("1.5e3", {"type": "real"}, 1500.0),
        ("true", {"type": "boolean"}, True),
        ("19723", {"type": "date"}, "2024-01-01"),
        ("45296.5", {"type": "time"}, "12:34:56.500000"),
        ("1704067200.123456789", {"type": "timestamp_ntz"}, "2024-01-01T00:00:00.123456"),
        ("-0.5", {"type": "timestamp_ntz"}, "1969-12-31T23:59:59.500000"),
        ("1704067200.000000000", {"type": "timestamp_ltz"}, "2024-01-01T00:00:00+00:00"),
        ("1704067200.000000000 1500", {"type": "timestamp_tz"}, "2024-01-01T01:00:00+01:00"),
        ('{"a": [1, 2]}', {"type": "variant"}, {"a": [1, 2]}),
        ("hello", {"type": "text"}, "hello"),
        (None, {"type": "fixed", "scale": 0}, None),
        ("not-a-number", {"type": "fixed", "scale": 0}, "not-a-number"),
    ],
)
def test_decode_value(value: Any, column: dict[str, Any], expected: Any) -> None:
    assert _decode_value(value, column) == expected


# ---------------------------------------------------------------------------
# SQL API transport
# ---------------------------------------------------------------------------

_ROW_TYPE = [
    {"name": "ID", "type": "fixed", "scale": 0, "nullable": False},
    {"name": "CREATED", "type": "date", "nullable": True},
]


@pytest.mark.asyncio
async def test_execute_polls_and_fetches_partitions(mock_api) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"statementHandle": "h1", "statementStatusUrl": "/api/v2/statements/h1"})
        if request.url.params.get("partition") == "1":
            return httpx.Response(200, json={"data": [["3", "19724"]]})
        return httpx.Response(
            200,
            json={
                "statementHandle": "h1",
                "resultSetMetaData": {
                    "numRows": 3,
                    "rowType": _ROW_TYPE,
                    "partitionInfo": [{"rowCount": 2}, {"rowCount": 1}],
                },
                "data": [["1", "19723"], ["2", None]],
            },
        )

    requests = mock_api(handler)
    result = await _execute(_conn(), "SELECT id, created FROM t WHERE x = ?", bindings=["a"])

    assert result["rows"] == [
        {"ID": 1, "CREATED": "2024-01-01"},
        {"ID": 2, "CREATED": None},
        {"ID": 3, "CREATED": "2024-01-02"},
    ]
    assert result["row_count"] == 3 and not result["truncated"]

    post = requests[0]
    assert post.url.path == "/api/v2/statements"
    assert post.headers["Authorization"] == "Bearer jwt-token"
    assert post.headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"
    body = json.loads(post.content)
    assert body["role"] == "ANALYST" and body["warehouse"] == "WH" and body["schema"] == "PUBLIC"
    assert body["bindings"] == {"1": {"type": "TEXT", "value": "a"}}
    assert [r.method for r in requests] == ["POST", "GET", "GET"]


@pytest.mark.asyncio
async def test_context_names_are_case_normalized(mock_api) -> None:
    requests = mock_api(
        lambda _request: httpx.Response(200, json={"statementHandle": "h", "resultSetMetaData": {}, "data": []})
    )
    await _execute(_conn(role="rf_reader", warehouse='"Mixed WH"', database="db"), "SELECT 1", schema="raw")
    body = json.loads(requests[0].content)
    assert (body["role"], body["warehouse"], body["database"], body["schema"]) == ("RF_READER", "Mixed WH", "DB", "RAW")


@pytest.mark.asyncio
async def test_cancelled_poll_cancels_statement(mock_api, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={})
        return httpx.Response(202, json={"statementHandle": "slow"})

    requests = mock_api(handler)
    monkeypatch.setattr(sf.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await _execute(_conn(), "SELECT SYSTEM$WAIT(60)")
    assert requests[-1].url.path == "/api/v2/statements/slow/cancel"


@pytest.mark.asyncio
async def test_network_policy_hint(mock_api) -> None:
    mock_api(lambda _request: httpx.Response(403, json={"code": "390422", "message": "IP not allowed."}))
    with pytest.raises(SnowflakeError, match="network policy"):
        await _execute(_conn(), "SELECT 1")


@pytest.mark.asyncio
async def test_execute_stops_fetching_at_max_rows(mock_api) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST", "partition 1 must not be fetched"
        return httpx.Response(
            200,
            json={
                "statementHandle": "h2",
                "resultSetMetaData": {"numRows": 5, "rowType": _ROW_TYPE, "partitionInfo": [{}, {}]},
                "data": [["1", None], ["2", None], ["3", None]],
            },
        )

    mock_api(handler)
    result = await _execute(_conn(), "SELECT 1", max_rows=2)
    assert [r["ID"] for r in result["rows"]] == [1, 2]
    assert result["truncated"] and result["row_count"] == 5


@pytest.mark.asyncio
async def test_execute_surfaces_snowflake_error(mock_api) -> None:
    mock_api(
        lambda _request: httpx.Response(
            422,
            json={
                "code": "002003",
                "sqlState": "02000",
                "message": "Object 'T' does not exist or not authorized.",
                "statementHandle": "h3",
            },
        )
    )
    with pytest.raises(SnowflakeError, match=r"002003 \(SQLSTATE 02000\): Object 'T' does not exist") as exc:
        await _execute(_conn(), "SELECT * FROM t")
    assert exc.value.code == "002003" and exc.value.statement_handle == "h3"


@pytest.mark.asyncio
async def test_auth_error_has_hint(mock_api) -> None:
    mock_api(lambda _request: httpx.Response(401, json={"code": "390144", "message": "JWT token is invalid."}))
    with pytest.raises(SnowflakeError, match="RSA_PUBLIC_KEY_FP"):
        await _execute(_conn(), "SELECT 1")


@pytest.mark.asyncio
async def test_retries_on_503_with_retry_flag(mock_api) -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"statementHandle": "h", "resultSetMetaData": {"numRows": 0}, "data": []})

    requests = mock_api(handler)
    await _execute(_conn(), "SELECT 1")
    first, second = requests
    assert first.url.params["requestId"] == second.url.params["requestId"]
    assert "retry" not in first.url.params and second.url.params["retry"] == "true"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_tool_rejects_writes_before_network() -> None:
    tool = SnowflakeQuery(account="myorg-acct", token=SecretStr("t"))
    out = await tool(query="DELETE FROM t").collect()
    assert out.status.code == "error"
    assert "snowflake_execute" in str(out.error)


@pytest.mark.asyncio
async def test_insert_rows_uses_bindings(mock_api) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "statementHandle": "h",
                "resultSetMetaData": {"numRows": 1, "rowType": [{"name": "number of rows inserted", "type": "fixed"}]},
                "data": [["2"]],
                "stats": {"numRowsInserted": 2},
            },
        )

    requests = mock_api(handler)
    tool = SnowflakeInsertRows(account="myorg-acct", token=SecretStr("t"))
    out = await tool(
        table='sales."Orders"',
        rows=[{"id": 1, "note": "it's"}, {"note": None, "id": 2}],
    ).collect()
    assert out.status.code == "success", out.error
    result = out.output
    body = json.loads(requests[0].content)
    assert body["statement"] == 'INSERT INTO sales."Orders" (id, note) VALUES (?, ?), (?, ?)'
    assert [b["value"] for b in body["bindings"].values()] == ["1", "it's", "2", None]
    assert result["stats"] == {"numRowsInserted": 2}


@pytest.mark.asyncio
async def test_insert_rows_rejects_mismatched_columns() -> None:
    tool = SnowflakeInsertRows(account="myorg-acct", token=SecretStr("t"))
    out = await tool(table="t", rows=[{"a": 1}, {"b": 2}]).collect()
    assert out.status.code == "error"
    assert "Row 1" in str(out.error)


def test_lazy_export() -> None:
    from timbal.tools import SnowflakeQuery as exported

    assert exported is SnowflakeQuery


# ---------------------------------------------------------------------------
# SnowflakeAuth (managed MCP servers)
# ---------------------------------------------------------------------------


def _auth_client(auth: SnowflakeAuth, handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=auth)


_CALLER = contextvars.ContextVar("caller", default=None)


@pytest.mark.asyncio
async def test_auth_prepared_in_caller_context_serves_context_free_requests(rsa_key: rsa.RSAPrivateKey) -> None:
    """MCPServer sends from an owner task with an empty context; the integration must resolve in the caller's."""

    async def resolve() -> dict[str, Any]:
        assert _CALLER.get() == "run-1", "Integration.resolve ran outside the caller's context"
        return {"account": "myorg-acct", "user": "svc", "private_key": _pem(rsa_key)}

    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(side_effect=resolve)
    auth = SnowflakeAuth(integration)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    async def send_twice() -> None:
        async with _auth_client(auth, handler) as client:
            await client.post("https://myorg-acct.snowflakecomputing.com/api/v2/x/mcp-servers/s", json={})
            await client.post("https://myorg-acct.snowflakecomputing.com/api/v2/x/mcp-servers/s", json={})

    _CALLER.set("run-1")
    await auth.prepare()
    await asyncio.get_running_loop().create_task(send_twice(), context=contextvars.Context())

    assert integration.resolve.await_count == 1
    assert all(r.headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT" for r in seen)
    claims = jwt.decode(seen[0].headers["Authorization"].split()[1], rsa_key.public_key(), algorithms=["RS256"])
    assert claims["sub"] == "MYORG-ACCT.SVC"


@pytest.mark.asyncio
async def test_unprepared_integration_auth_refuses_without_run_context() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={})
    async with _auth_client(SnowflakeAuth(integration), lambda _r: httpx.Response(200)) as client:
        with pytest.raises(RuntimeError, match="caller's run context"):
            await client.get("https://h/api/v2/a")
    integration.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_request_prepares_auth_in_caller_task() -> None:
    from timbal.core import MCPServer

    class _Auth(httpx.Auth):
        def __init__(self) -> None:
            self.prepared_in: list[Any] = []

        async def prepare(self) -> None:
            self.prepared_in.append(_CALLER.get())

    auth = _Auth()
    server = MCPServer(transport="http", url="https://h/mcp", auth=auth)
    session = SimpleNamespace(list_tools=AsyncMock(return_value="tools"))
    server._connect = AsyncMock(return_value=session)  # type: ignore[method-assign]

    _CALLER.set("run-2")
    assert await server._request(lambda s: s.list_tools(), what="tools/list") == "tools"
    assert auth.prepared_in == ["run-2"]


@pytest.mark.asyncio
async def test_auth_remints_before_jwt_expiry(rsa_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr(sf.time, "time", lambda: now[0])
    tokens: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens.append(request.headers["Authorization"])
        return httpx.Response(200)

    auth = SnowflakeAuth(account="myorg-acct", user="svc", private_key=_pem(rsa_key))
    async with _auth_client(auth, handler) as client:
        await client.get("https://h/api/v2/a")
        now[0] += 30 * 60
        await client.get("https://h/api/v2/a")
        now[0] += 25 * 60
        await client.get("https://h/api/v2/a")

    assert tokens[0] == tokens[1] != tokens[2]


@pytest.mark.asyncio
async def test_auth_retries_once_on_401(rsa_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr(sf.time, "time", lambda: now[0])
    tokens: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens.append(request.headers["Authorization"])
        now[0] += 1
        return httpx.Response(401)

    auth = SnowflakeAuth(account="myorg-acct", user="svc", private_key=_pem(rsa_key))
    async with _auth_client(auth, handler) as client:
        response = await client.get("https://h/api/v2/a")

    assert response.status_code == 401
    assert len(tokens) == 2 and tokens[0] != tokens[1]


@pytest.mark.asyncio
async def test_auth_static_token_has_no_expiry() -> None:
    seen: list[httpx.Request] = []
    auth = SnowflakeAuth(account="myorg-acct", programmatic_access_token=SecretStr("pat"))
    async with _auth_client(auth, lambda r: seen.append(r) or httpx.Response(200)) as client:
        await client.get("https://h/api/v2/a")
    assert seen[0].headers["Authorization"] == "Bearer pat"
    assert seen[0].headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"


def test_mcp_server_accepts_auth_but_not_with_service_api() -> None:
    from timbal.core import MCPServer

    auth = SnowflakeAuth(account="myorg-acct", token="t")
    server = MCPServer(transport="http", url="https://h/mcp", auth=auth)
    assert server.auth is auth
    assert "auth" not in server.model_dump()
    with pytest.raises(ValueError, match="auth"):
        MCPServer(transport="http", service="api", auth=auth)


# ---------------------------------------------------------------------------
# Live Snowflake (requires credentials + network; excluded from default pytest)
# ---------------------------------------------------------------------------


def _skip_if_snowflake_live_env_not_set() -> None:
    import os

    has_auth = any(
        os.getenv(v)
        for v in ("SNOWFLAKE_PRIVATE_KEY", "SNOWFLAKE_PRIVATE_KEY_PATH", "SNOWFLAKE_PAT", "SNOWFLAKE_TOKEN")
    )
    if not (os.getenv("SNOWFLAKE_ACCOUNT") and has_auth):
        pytest.skip(
            "Snowflake live integration: set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER and SNOWFLAKE_PRIVATE_KEY_PATH "
            "(or SNOWFLAKE_PAT). Optional: SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_MCP_URL."
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_query_context_and_null_binding() -> None:
    _skip_if_snowflake_live_env_not_set()
    out = await SnowflakeQuery()(
        query="SELECT CURRENT_USER() AS u, CURRENT_ROLE() AS r, ? IS NULL AS null_bound, ?::DATE AS d",
        bindings=[None, "2024-01-01"],
    ).collect()
    assert out.status.code == "success", out.error
    row = out.output["rows"][0]
    assert row["NULL_BOUND"] is True
    assert row["D"] == "2024-01-01"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_managed_mcp_lists_tools() -> None:
    import os

    _skip_if_snowflake_live_env_not_set()
    url = os.getenv("SNOWFLAKE_MCP_URL")
    if not url:
        pytest.skip("Set SNOWFLAKE_MCP_URL to a Snowflake-managed MCP server endpoint.")

    from timbal.core import MCPServer

    server = MCPServer(name="snowflake", transport="http", url=url, auth=SnowflakeAuth())
    try:
        tools = await server.resolve()
    finally:
        await server.close()
    assert tools, "managed MCP server returned no tools"
