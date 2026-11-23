import asyncio
import base64
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs
from xml.etree import ElementTree as ET

import httpx
import pytest
from mcp import types
from pydantic import BaseModel, ConfigDict, ValidationError
from timbal import Agent, Tool, Workflow
from timbal.core.mcp import MCPServer
from timbal.core.test_model import TestModel
from timbal.platform.integrations import Integration
from timbal.state import get_run_context, set_run_context
from timbal.tools import (
    SAPBAPI,
    SAPAccess,
    SAPAgentGetTask,
    SAPAgentSend,
    SAPCDSRead,
    SAPCustomizing,
    SAPGUIScript,
    SAPHANAQuery,
    SAPHANARead,
    SAPHANAWrite,
    SAPMCPBinding,
    SAPMCPServer,
    SAPODataRead,
    SAPODataWrite,
    SAPSOAPOperation,
    SAPTokenExchange,
)
from timbal.tools.sap import _http as sap_http
from timbal.tools.sap._access import SAPHTTPError
from timbal.tools.sap._native import _gui_worker
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

SECRET = "s3cret-client-secret"
TOKEN_URL = "https://auth.example/oauth/token"


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("SAP_HANA_USER", raising=False)
    monkeypatch.delenv("SAP_HANA_PASSWORD", raising=False)
    sap_http._TOKEN_CACHE.clear()
    set_run_context(None)
    yield
    set_run_context(None)
    sap_http._TOKEN_CACHE.clear()


@pytest.fixture
def access():
    return SAPAccess()


def sap_creds(**overrides):
    creds = {
        "type": "credentials",
        "host": "sap.example",
        "port": 443,
        "token_url": TOKEN_URL,
        "client_id": "client",
        "client_secret": SECRET,
        "sap_client": "100",
    }
    creds.update(overrides)
    return creds


def bound(creds):
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value=creds)
    return integration


def token_response(token="tok-1", expires_in=3600, token_type="bearer"):
    body = {"access_token": token, "token_type": token_type}
    if expires_in is not None:
        body["expires_in"] = expires_in
    return httpx.Response(200, json=body)


@pytest.fixture
def hana(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.description = [("ID",)]
    cursor.fetchone.side_effect = [(1,), (2,), (3,), None]
    connect = MagicMock(return_value=connection)
    monkeypatch.setitem(sys.modules, "hdbcli", SimpleNamespace(dbapi=SimpleNamespace(connect=connect)))
    return connect, connection, cursor


def hana_tool(access, cls=SAPHANARead, **kwargs):
    config = {"access": access}
    if "integration" not in kwargs:
        config.update(host="hana.example", user="tech", password="test")
    if cls is SAPHANARead:
        config.update(schema_name="REPORTING", view="OPEN_ORDERS", columns=["ID"], order_by=["ID"])
    return cls(**config, **kwargs)


async def test_hana_framework_entrypoint_pagination(access, hana):
    connect, connection, cursor = hana
    tool = hana_tool(access)
    r = await tool(limit=2, filters={"ID": "x' OR 1=1"}, offset=3).collect()
    assert r.output == {"columns": ["ID"], "rows": [[1], [2]], "truncated": True, "next_offset": 5}
    options = connect.call_args.kwargs
    assert (options["address"], options["user"], options["password"]) == ("hana.example", "tech", "test")
    assert (options["port"], options["encrypt"], options["sslValidateCertificate"]) == (443, True, True)
    assert options["sslMinProtocolVersion"] == "TLS12"
    # APPLICATION names the tool; no end user is invented for APPLICATIONUSER.
    assert [c.args for c in connection.setclientinfo.call_args_list] == [("APPLICATION", "sap_hana_read")]
    assert cursor.execute.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)
    assert cursor.execute.call_args.args == (
        'SELECT "ID" FROM "REPORTING"."OPEN_ORDERS" WHERE "ID" = ? ORDER BY "ID" LIMIT ? OFFSET ?',
        ("x' OR 1=1", 3, 3),
    )
    assert set(tool.params_model_schema["properties"]) == {"filters", "limit", "offset"}
    assert r.metadata["sap"] == {"selected_by": "workflow", "route": "hana_view", "object": '"REPORTING"."OPEN_ORDERS"'}
    cursor.close.assert_called_once()
    connection.close.assert_called_once()


async def test_hana_filter_guards(access, hana):
    tool = hana_tool(access)
    for bad in ({"filters": {"SECRET": 1}}, {"limit": 1001}, {"offset": -1}):
        with pytest.raises(ValueError):
            await tool.handler(**bad)
    hana[0].assert_not_called()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM X",
        "SELECT 1; SELECT 2",
        "WITH t AS (DELETE FROM x) SELECT * FROM t",
        "SELECT 1 INTO newtable FROM t",
        "SELECT * FROM t FOR UPDATE",
    ],
)
async def test_hana_freeform_statement_and_read_transaction(access, hana, sql):
    tool = hana_tool(access, SAPHANAQuery)
    with pytest.raises(ValueError):
        await tool.handler(query=sql)
    hana[0].assert_not_called()
    await tool.handler(query="SELECT 'delete;drop' FROM \"orders\";")
    assert hana[2].execute.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)


async def test_hana_write_is_preconfigured_and_transactional(access, hana):
    tool = hana_tool(access, SAPHANAWrite, statement='INSERT INTO "S"."T" VALUES (?)', object_name="S.T")
    hana[2].rowcount = 1
    assert await tool.handler(bindings=[1]) == {"affected_rows": 1}
    hana[1].commit.assert_called_once()
    hana[2].execute.side_effect = RuntimeError("upstream failure")
    with pytest.raises(RuntimeError):
        await tool.handler(bindings=[1])
    hana[1].commit.assert_called_once()
    assert hana[1].close.call_count == 2


async def test_hana_credentials_from_the_integration_or_the_environment(access, hana, monkeypatch):
    connect, _, cursor = hana
    view = dict(schema_name="R", view="V", columns=["ID"], order_by=["ID"])
    monkeypatch.setenv("SAP_HANA_USER", "env-user")
    monkeypatch.setenv("SAP_HANA_PASSWORD", "env-secret")
    await SAPHANARead(access=access, host="hana.example", **view).handler()
    assert (connect.call_args.kwargs["user"], connect.call_args.kwargs["password"]) == ("env-user", "env-secret")

    creds = {"type": "credentials", "host": "HANA.example", "user": "vended-user", "password": "vended-secret"}
    cursor.fetchone.side_effect = [(1,), None]
    await SAPHANARead(access=access, integration=bound(creds), **view).handler()
    options = connect.call_args.kwargs
    assert (options["address"], options["user"], options["password"]) == (
        "hana.example",
        "vended-user",
        "vended-secret",
    )
    # A bound integration never falls back to the environment.
    with pytest.raises(ValueError, match="user and password"):
        await SAPHANARead(access=access, integration=bound({**creds, "password": ""}), **view).handler()
    elsewhere = SAPHANARead(access=access, integration=bound(creds), host="other.example", **view)
    with pytest.raises(PermissionError, match="host"):
        await elsewhere.handler()
    assert connect.call_count == 2
    with pytest.raises(ValidationError, match="supplies the HANA user"):
        SAPHANARead(access=access, integration=bound(creds), user="tech", password="x", **view)
    with pytest.raises(ValidationError, match="host"):
        SAPHANARead(access=access, **view)


async def test_byte_limited_page_continues_at_first_unreturned_row(access, hana):
    access.max_bytes = 145
    hana[2].fetchone.side_effect = [("x" * 45,), ("y" * 45,), None]
    page = await hana_tool(access).handler()
    assert page["next_offset"] == 1
    assert page["rows"] == [["x" * 45]]
    hana[2].fetchone.side_effect = [("z" * 200,), None]
    with pytest.raises(ValueError, match="One SAP result row"):
        await hana_tool(access).handler()


async def test_abap_sql_service(access, hana):
    connect, connection, cursor = hana
    tool = SAPCDSRead(
        access=access, view="ReleasedView", statement="SELECT ID FROM ReleasedView WHERE ID = ?", connect=connect
    )
    assert (await tool.handler(bindings=["10"]))["columns"] == ["ID"]
    cursor.execute.assert_called_once_with("SELECT ID FROM ReleasedView WHERE ID = ?", "10")
    assert connection.timeout == 60
    connection.close.assert_called_once()


@pytest.fixture
def http(monkeypatch):
    requests, responses = [], []

    def dispatch(request):
        requests.append(request)
        return responses.pop(0)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_class(transport=httpx.MockTransport(dispatch), **kw))
    return requests, responses


def http_tool(cls, access, endpoint="https://sap.example/service/Orders", **kwargs):
    kwargs.setdefault("token_provider", AsyncMock(return_value="exchanged"))
    return cls(access=access, endpoint=endpoint, object_name="Orders", **kwargs)


def odata_read(access, **kwargs):
    return http_tool(SAPODataRead, access, properties=["ID"], order_by=["ID"], **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        {"value": [{"ID": "1", "hidden": "redact"}], "@odata.nextLink": "https://other.example/x?$skiptoken=next"},
        {"d": {"results": [{"ID": "1"}], "__next": "Orders?$skiptoken=next"}},
    ],
)
async def test_odata_read_v2_v4_pinned_target(access, http, data):
    requests, responses = http
    responses.extend([httpx.Response(200, json=data), httpx.Response(200, json={"value": []})])
    tool = http_tool(SAPODataRead, access, properties=["ID"], order_by=["ID"], cds=True)
    page = await tool.handler(filters={"ID": "a'b"})
    assert page == {"rows": [{"ID": "1"}], "next_skiptoken": "next"}
    assert requests[0].url.params["$filter"] == "ID eq 'a''b'"
    await tool.handler(skiptoken=page["next_skiptoken"])
    assert requests[1].url.host == "sap.example"
    assert requests[1].url.params["$skiptoken"] == "next"
    assert requests[1].headers["Authorization"] == "Bearer exchanged"


async def test_client_credentials_grant_shape_and_cache(access, http):
    requests, responses = http
    tool = odata_read(access, token_provider=None, integration=bound(sap_creds()))
    responses.extend(
        [token_response(), httpx.Response(200, json={"value": []}), httpx.Response(200, json={"value": []})]
    )
    await tool.handler()
    grant, read = requests
    assert (grant.method, str(grant.url)) == ("POST", TOKEN_URL)
    assert grant.headers["Authorization"] == "Basic " + base64.b64encode(f"client:{SECRET}".encode()).decode()
    assert grant.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert parse_qs(grant.content.decode()) == {"grant_type": ["client_credentials"]}
    assert (read.url.host, read.headers["Authorization"]) == ("sap.example", "Bearer tok-1")
    await tool.handler()
    assert [r.url.host for r in requests] == ["auth.example", "sap.example", "sap.example"]
    # A rotated secret is a different cache entry.
    rotated = odata_read(access, token_provider=None, integration=bound(sap_creds(client_secret="rotated")))
    responses.extend([token_response("tok-2"), httpx.Response(200, json={"value": []})])
    await rotated.handler()
    assert requests[-1].headers["Authorization"] == "Bearer tok-2"


@pytest.mark.parametrize(("expires_in", "renew_at"), [(3600, 3480), (100, 50)])
async def test_token_is_renewed_before_it_expires(access, http, monkeypatch, expires_in, renew_at):
    requests, responses = http
    now = [1000.0]
    monkeypatch.setattr(sap_http, "_clock", lambda: now[0])
    tool = odata_read(access, token_provider=None, integration=bound(sap_creds()))
    responses.extend([token_response("tok-1", expires_in), httpx.Response(200, json={"value": []})])
    await tool.handler()
    now[0] = 1000 + renew_at - 1
    responses.append(httpx.Response(200, json={"value": []}))
    await tool.handler()
    now[0] = 1000 + renew_at + 1
    responses.extend([token_response("tok-2", expires_in), httpx.Response(200, json={"value": []})])
    await tool.handler()
    grants = [r for r in requests if r.url.host == "auth.example"]
    assert len(grants) == 2
    assert requests[-1].headers["Authorization"] == "Bearer tok-2"


async def test_token_grant_refusals_never_reveal_the_secret(access, http):
    requests, responses = http
    access.failure_threshold = 100  # Each refusal counts as an upstream failure.
    tool = odata_read(access, token_provider=None, integration=bound(sap_creds()))
    refusals = [
        (token_response(token_type="mac"), RuntimeError, "bearer"),
        (token_response(token_type=None), RuntimeError, "bearer"),
        (httpx.Response(302, headers={"Location": "https://evil.example/token"}), RuntimeError, "redirect"),
        (httpx.Response(401, json={"error": "invalid_client", "secret": SECRET}), SAPHTTPError, "HTTP 401"),
        (httpx.Response(200, text="not json"), RuntimeError, "access token"),
    ]
    for response, error, message in refusals:
        responses.append(response)
        with pytest.raises(error, match=message) as caught:
            await tool.handler()
        assert SECRET not in str(caught.value) and "tok-1" not in str(caught.value)
    # Nothing was cached, no redirect was followed and SAP was never called.
    assert [r.url.host for r in requests] == ["auth.example"] * len(refusals)
    assert not sap_http._TOKEN_CACHE
    insecure = odata_read(access, token_provider=None, integration=bound(sap_creds(token_url="http://auth.example/t")))
    with pytest.raises(ValueError, match="HTTPS"):
        await insecure.handler()
    incomplete = odata_read(access, token_provider=None, integration=bound(sap_creds(client_secret="")))
    with pytest.raises(ValueError, match="token_url, client_id and client_secret"):
        await incomplete.handler()
    assert len(requests) == len(refusals)


async def test_rejected_token_is_not_reused_and_never_traced(access, http):
    requests, responses = http
    tool = odata_read(access, token_provider=None, integration=bound(sap_creds()))
    responses.extend([token_response("tok-1"), httpx.Response(401)])
    failed = await tool().collect()
    assert "401" in str(failed.error)
    responses.extend([token_response("tok-2"), httpx.Response(200, json={"value": [{"ID": "1"}]})])
    ok = await tool().collect()
    assert ok.output == {"rows": [{"ID": "1"}]}
    assert requests[-1].headers["Authorization"] == "Bearer tok-2"
    traced = json.dumps([failed.model_dump(), ok.model_dump()], default=str)
    assert SECRET not in traced and "tok-1" not in traced and "tok-2" not in traced
    assert SECRET not in tool.model_dump_json()


@pytest.mark.parametrize(
    "creds", [sap_creds(host="other.example"), sap_creds(port=8443)], ids=["other-host", "other-port"]
)
async def test_requests_stay_on_the_connection_host(access, http, creds):
    requests, responses = http
    # Neither a granted token nor one from the deployment's provider leaves the connection's host.
    for token_provider in (None, AsyncMock(return_value="exchanged")):
        tool = odata_read(access, token_provider=token_provider, integration=bound(creds))
        with pytest.raises(PermissionError, match="connection's host"):
            await tool.handler()
    assert not requests
    on_host = odata_read(
        access,
        endpoint="https://SAP.example:8443/service/Orders",
        integration=bound(sap_creds(host="https://sap.example:8443", port=None)),
    )
    responses.append(httpx.Response(200, json={"value": []}))
    await on_host.handler()
    assert (requests[0].url.host, requests[0].url.port) == ("sap.example", 8443)
    with pytest.raises(ValueError, match="disagree"):
        await odata_read(access, integration=bound(sap_creds(host="sap.example:8443", port=443))).handler()


async def test_sap_client_is_added_when_absent(access, http):
    requests, responses = http
    responses.extend([httpx.Response(200, json={"value": []}) for _ in range(3)])
    await odata_read(access, integration=bound(sap_creds())).handler()
    assert requests[0].url.params.get_list("sap-client") == ["100"]
    explicit = odata_read(
        access, endpoint="https://sap.example/service/Orders?sap-client=200", integration=bound(sap_creds())
    )
    await explicit.handler()
    assert requests[1].url.params.get_list("sap-client") == ["200"]
    await odata_read(access, integration=bound(sap_creds(sap_client=""))).handler()
    assert "sap-client" not in requests[2].url.params
    with pytest.raises(ValueError, match="three digits"):
        await odata_read(access, integration=bound(sap_creds(sap_client="1000"))).handler()


async def test_odata_write_schema_csrf_cookie_no_replay(access, http):
    requests, responses = http
    tool = http_tool(
        SAPODataWrite,
        access,
        input_model=Item,
        method="POST",
        document_types=["order"],
        csrf_endpoint="https://sap.example/service/",
    )
    responses.extend(
        [
            httpx.Response(200, headers={"X-CSRF-Token": "csrf", "Set-Cookie": "sid=test; Path=/"}),
            httpx.Response(201, json={"ID": "42"}),
        ]
    )
    output = await tool(payload={"id": "42"}).collect()
    assert output.output == {"ID": "42"}
    assert output.metadata["sap"] == {
        "selected_by": "workflow",
        "route": "published_api",
        "object": "Orders",
        "document_types": ["order"],
    }
    assert [r.method for r in requests] == ["GET", "POST"]
    assert requests[0].headers["X-CSRF-Token"] == "Fetch"
    assert (requests[1].headers["X-CSRF-Token"], requests[1].headers["cookie"]) == ("csrf", "sid=test")
    assert json.loads(requests[1].content) == {"id": "42"}
    with pytest.raises(ValidationError):
        await tool.handler(payload={"id": "42", "unknown": "value"})
    with pytest.raises(ValidationError, match="same origin"):
        http_tool(
            SAPODataWrite,
            access,
            input_model=Item,
            method="POST",
            document_types=[],
            csrf_endpoint="https://other.test/",
        )


async def test_odata_csrf_fetch_skips_service_document(access, http):
    requests, responses = http
    access.max_bytes = 32

    async def service_document():
        yield b"x" * 1024

    fetch = httpx.Response(
        200, headers={"X-CSRF-Token": "csrf", "Set-Cookie": "session=abc; Path=/"}, content=service_document()
    )
    responses.extend([fetch, httpx.Response(201, json={"id": "42"})])
    tool = http_tool(
        SAPODataWrite, access, input_model=Item, method="POST", document_types=[], csrf_endpoint="https://sap.example/"
    )

    assert await tool.handler(payload={"id": "42"}) == {"id": "42"}
    assert fetch.is_closed and not fetch.is_stream_consumed
    assert requests[1].headers["X-CSRF-Token"] == "csrf"
    assert requests[1].headers["Cookie"] == "session=abc"


async def test_soap_published_operation_and_fault(access, http):
    requests, responses = http
    tag = "{urn:example}GetItem"

    def build(payload):
        element = ET.Element(tag)
        ET.SubElement(element, "id").text = payload.id
        return element

    tool = http_tool(
        SAPSOAPOperation,
        access,
        operation_tag=tag,
        action="urn:get",
        input_model=Item,
        read_only=True,
        document_types=[],
        build_element=build,
    )
    envelope = '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body>{}</s:Body></s:Envelope>'
    responses.extend(
        [httpx.Response(200, text=envelope.format("<Item/>")), httpx.Response(200, text=envelope.format("<s:Fault/>"))]
    )
    assert "Item" in await tool.handler(payload={"id": "<item>"})
    assert b"&lt;item&gt;" in requests[0].content
    assert requests[0].headers["SOAPAction"] == '"urn:get"'
    with pytest.raises(RuntimeError, match="SOAP fault"):
        await tool.handler(payload={"id": "42"})


async def test_bapi_return_and_commit(access):
    conn = MagicMock()
    tool = SAPBAPI(
        access=access,
        function="BAPI_EXAMPLE",
        connect=lambda: conn,
        read_only=False,
        commit=True,
        document_types=["material"],
        input_model=Item,
    )
    conn.call.side_effect = [{"RETURN": [{"TYPE": "E"}]}, {"RETURN": {"TYPE": "S"}}, {"RETURN": {}}]
    with pytest.raises(RuntimeError, match="application error"):
        await tool.handler(payload={"id": "1"})
    result = await tool.handler(payload={"id": "1"})
    assert result["RETURN"]["TYPE"] == "S"
    assert [c.args[0] for c in conn.call.call_args_list] == ["BAPI_EXAMPLE", "BAPI_EXAMPLE", "BAPI_TRANSACTION_COMMIT"]
    assert conn.close.call_count == 2


async def test_agent_gateway_http_json(access, http):
    requests, responses = http
    send = http_tool(
        SAPAgentSend,
        access,
        document_types=["order"],
        request_metadata={"scenario": "orders"},
        callback_url="https://worker.example/completion",
    )
    responses.extend([httpx.Response(200, json={"task": {"id": "123"}}), httpx.Response(200, json={"id": "123"})])
    result = await send.handler(text="create", context_id="ctx")
    assert result["task"]["id"] == "123"
    sent = json.loads(requests[0].content)
    assert sent["metadata"] == {"scenario": "orders"}
    assert sent["configuration"]["pushNotification"]["url"] == "https://worker.example/completion"
    msg = sent["message"]
    assert (msg["role"], msg["contextId"], msg["parts"]) == ("ROLE_USER", "ctx", [{"text": "create"}])
    task = http_tool(SAPAgentGetTask, access)
    assert await task.handler(task_id="123") == {"id": "123"}
    assert requests[1].url.path.endswith("/123")
    with pytest.raises(ValueError):
        await task.handler(task_id="../other")


async def test_exchange_resolves_each_user_without_forwarding_the_subject(access, http):
    requests, responses = http
    provider = SAPTokenExchange(
        endpoint="https://identity.example/oauth2/token",
        resource="urn:sap:example",
        client_id="client",
        client_secret="secret",
        subject_token=AsyncMock(side_effect=["user-one", "user-two"]),
    )
    tool = SAPAgentSend(
        access=access,
        endpoint="https://sap.example/v1/message:send",
        token_provider=provider,
        document_types=[],
        object_name="agent",
    )
    for i in range(2):
        responses.extend(
            [
                httpx.Response(200, json={"access_token": f"exchanged-{i}"}),
                httpx.Response(200, json={"message": {"text": "ok"}}),
            ]
        )
        await tool.handler(text="hello")
    assert [r.url.host for r in requests] == ["identity.example", "sap.example"] * 2
    assert "user-one" in requests[0].content.decode()
    assert "user-two" in requests[2].content.decode()
    assert [r.headers["authorization"] for r in requests[1::2]] == ["Bearer exchanged-0", "Bearer exchanged-1"]
    assert "secret" not in provider.model_dump_json()


async def test_local_hana_mcp_requires_read_only_manifest(access):
    config = dict(transport="stdio", command="hana-ai-worker", access=access, kind="hana-ai")
    with pytest.raises(ValidationError, match="read-only"):
        SAPMCPServer(**config, manifest={"write": {"read_only": False, "document_types": []}})
    server = SAPMCPServer(**config, manifest={"lookup": {"read_only": True, "document_types": []}})
    with pytest.raises(ValueError, match="writable"):
        server._make_tool(
            types.Tool(
                name="lookup", inputSchema={"type": "object"}, annotations=types.ToolAnnotations(readOnlyHint=False)
            )
        )
    await server.close()


async def test_native_worker_retains_slot_after_timeout(access):
    access.timeout_seconds = 1
    access.concurrency = 1
    started, release = threading.Event(), threading.Event()

    def worker():
        started.set()
        release.wait(5)

    async def run(call):
        return await call.blocking(worker)

    pending = asyncio.create_task(access.invoke("read", "v", run, route="published_api"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(TimeoutError):
            await pending
        with pytest.raises(RuntimeError, match="concurrency"):
            await access.invoke("read", "v", run, route="published_api")
    finally:
        release.set()
    async with asyncio.timeout(2):
        while access._active:
            await asyncio.sleep(0.01)


async def test_managed_mcp_allowlist_and_read_only(access, monkeypatch):
    allowed = types.Tool(name="lookup", inputSchema={"type": "object"})
    create = types.Tool(name="create", inputSchema={"type": "object"})
    other = types.Tool(name="other", inputSchema={"type": "object"})
    server = SAPMCPServer(
        name="sap",
        access=access,
        kind="gateway",
        transport="http",
        url="https://sap.example/mcp",
        headers={"Authorization": "Bearer example"},
        manifest={
            "lookup": SAPMCPBinding(read_only=True, document_types=[]),
            "create": SAPMCPBinding(read_only=False, document_types=["order"]),
        },
    )
    result = types.CallToolResult(content=[types.TextContent(type="text", text="ok")])
    request = AsyncMock(side_effect=[types.ListToolsResult(tools=[allowed, create, other]), result, result])
    monkeypatch.setattr(MCPServer, "_request", request)
    create_tool, lookup = await server.resolve()
    assert (create_tool.name, lookup.name) == ("sap__create", "sap__lookup")
    assert (create_tool.read_only, lookup.read_only) == (False, True)
    assert lookup._log_events == set() and create_tool._log_events == set()
    assert await create_tool.handler() == "ok"
    assert await lookup.handler() == "ok"
    assert request.await_count == 3
    await server.close()


async def test_gui_scripts_run_only_in_deterministic_workflows(access):
    change = SAPCustomizing(nonproduction_system="QAS", client="100", transport_request="QASK0001")
    script = MagicMock(return_value="ok")
    config = dict(
        access=access,
        nonproduction_system="QAS",
        user="TECH",
        customizing=change,
        script=script,
        document_types=[],
        input_model=Item,
    )
    with pytest.raises(ValidationError):
        SAPGUIScript(client="200", **config)
    tool = SAPGUIScript(client="100", **config)
    if sys.platform != "win32":
        with pytest.raises(RuntimeError, match="Windows"):
            await tool.handler(payload={"id": "1"})
    access.agent_selected = True
    with pytest.raises(PermissionError, match="deterministic"):
        await tool.handler(payload={"id": "1"})
    access.agent_selected = False

    first = Message(
        role="assistant",
        content=[ToolUseContent(id="c", name=tool.name, input={"payload": {"id": "1"}})],
        stop_reason="tool_use",
    )
    agent = Agent(name="test", model=TestModel(responses=[first, "done"]), tools=[tool], max_iter=3)
    await agent(prompt="run").collect()
    [span] = [s for s in get_run_context()._trace.values() if s.metadata.get("sap")]
    assert span.metadata["sap"]["selected_by"] == "agent"
    assert "deterministic" in str(span.error)
    script.assert_not_called()


def test_gui_worker_matches_session_and_transport(monkeypatch):
    pipe = MagicMock()
    session = SimpleNamespace(Info=SimpleNamespace(SystemName="QAS", Client="100", User="TECH"))
    engine = SimpleNamespace(Connections=[SimpleNamespace(Sessions=[session])])
    monkeypatch.setitem(sys.modules, "pythoncom", MagicMock())
    monkeypatch.setitem(
        sys.modules, "win32com.client", SimpleNamespace(GetObject=lambda _: SimpleNamespace(GetScriptingEngine=engine))
    )
    script = MagicMock(return_value={"ok": True})
    payload = Item(id="123")
    _gui_worker(pipe, ("QAS", "200", "TECH"), script, payload, "TR", 1024)
    script.assert_not_called()
    _gui_worker(pipe, ("QAS", "100", "TECH"), script, payload, "TR", 1024)
    script.assert_called_once_with(session, payload, "TR")
    assert pipe.send_bytes.call_args.args == (b'S{"ok":true}',)


async def test_agent_selection_is_recorded_on_the_span(access, http):
    _, responses = http
    tool = odata_read(access)
    first = Message(
        role="assistant", content=[ToolUseContent(id="c", name=tool.name, input={})], stop_reason="tool_use"
    )
    agent = Agent(name="reader", model=TestModel(responses=[first, "done"]), tools=[tool], max_iter=3)
    responses.append(httpx.Response(200, json={"value": [{"ID": "42"}]}))
    await agent(prompt="read").collect()
    [span] = [s for s in get_run_context()._trace.values() if s.metadata.get("sap")]
    assert span.metadata["sap"] == {
        "selected_by": "agent",
        "route": "published_api",
        "object": "Orders",
        "agent": "reader",
    }
    set_run_context(None)
    workflow = Workflow(name="lookup")
    workflow.step(tool)
    responses.append(httpx.Response(200, json={"value": []}))
    await workflow().collect()
    [span] = [s for s in get_run_context()._trace.values() if s.metadata.get("sap")]
    assert span.metadata["sap"]["selected_by"] == "workflow"


async def test_agent_selection_is_detected_through_workflows(access, http, monkeypatch):
    from timbal.tools.sap import _access

    _, responses = http
    recorded = []
    original = _access._record
    monkeypatch.setattr(_access, "_record", lambda details: (recorded.append(details), original(details)))
    tool = odata_read(access)
    workflow = Workflow(name="lookup")
    workflow.step(tool)

    async def wrapper():
        return (await workflow().collect()).output

    offered = Tool(name="wrapper", handler=wrapper)
    first = Message(
        role="assistant", content=[ToolUseContent(id="c", name=offered.name, input={})], stop_reason="tool_use"
    )
    agent = Agent(name="reader", model=TestModel(responses=[first, "done"]), tools=[offered], max_iter=3)
    responses.append(httpx.Response(200, json={"value": []}))
    await agent(prompt="read").collect()
    assert [(d["selected_by"], d["agent"]) for d in recorded] == [("agent", "reader")]


async def test_quota_concurrency_circuit_and_size(access, http):
    requests, responses = http
    tool = http_tool(SAPODataRead, access, properties=["ID"], order_by=["ID"])
    access.calls_per_minute = 1
    responses.append(httpx.Response(200, json={"value": []}))
    await tool.handler()
    with pytest.raises(RuntimeError, match="rate limit"):
        await tool.handler()
    access.calls_per_minute = 20
    access.max_bytes = 8
    responses.append(httpx.Response(200, content=b"x" * 100))
    with pytest.raises(ValueError, match="byte limit"):
        await tool.handler()
    access.failure_threshold = 1
    responses.append(httpx.Response(401))
    with pytest.raises(RuntimeError, match="401"):
        await tool.handler()
    with pytest.raises(RuntimeError, match="cooling"):
        await tool.handler()
    assert len(requests) == 3


async def test_concurrent_calls_share_quota_and_release_on_cancel(access):
    access.concurrency = 1
    started = asyncio.Event()

    async def block(_call):
        started.set()
        await asyncio.Future()

    pending = asyncio.create_task(access.invoke("one", "v", block, route="published_api"))
    await started.wait()
    with pytest.raises(RuntimeError, match="concurrency"):
        await access.invoke("two", "v", block, route="published_api")
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not access._active


async def test_hana_lob_is_bounded_and_closed(access, hana):
    access.max_bytes = 256
    lob = MagicMock()
    lob.read.return_value = "x" * 257
    hana[2].fetchone.side_effect = [(lob,), None]
    with pytest.raises(ValueError, match="byte limit"):
        await hana_tool(access).handler()
    hana[2].fetchone.assert_called_once_with(True)
    lob.read.assert_called_once_with(size=257)
    lob.close.assert_called_once()


async def test_customizing_pins_client_transport_and_origin(access, http):
    requests, responses = http
    change = SAPCustomizing(nonproduction_system="https://sap.example", client="100", transport_request="TR17")
    config = dict(
        method="POST", input_model=Item, document_types=[], customizing=change, transport_query_parameter="transport"
    )
    tool = http_tool(SAPODataWrite, access, integration=bound(sap_creds(sap_client="200")), **config)
    responses.append(httpx.Response(204))
    output = await tool(payload={"id": "1"}).collect()
    assert dict(requests[0].url.params) == {"sap-client": "100", "transport": "TR17"}
    assert output.metadata["sap"]["customizing"] == {
        "nonproduction_system": "https://sap.example",
        "client": "100",
        "transport_request": "TR17",
    }
    with pytest.raises(ValidationError, match="origin"):
        SAPODataWrite(access=access, endpoint="https://other.example/changes", object_name="Change", **config)


async def test_native_bapi_customizing_checks_actual_destination(access):
    connection = MagicMock()
    change = SAPCustomizing(nonproduction_system="QAS", client="100", transport_request="TR17")
    connection.get_connection_attributes.return_value = {"sysId": "PRD", "client": "100"}
    connection.call.return_value = {"RETURN": {"TYPE": "S"}}
    tool = SAPBAPI(
        access=access,
        function="BAPI_CUSTOMIZING",
        connect=lambda: connection,
        read_only=False,
        document_types=[],
        input_model=Item,
        customizing=change,
        transport_parameter="TRANSPORT",
    )
    with pytest.raises(PermissionError, match="destination"):
        await tool.handler(payload={"id": "123"})
    connection.call.assert_not_called()
    connection.get_connection_attributes.return_value["sysId"] = "QAS"
    await tool.handler(payload={"id": "123"})
    connection.call.assert_called_once_with("BAPI_CUSTOMIZING", id="123", TRANSPORT="TR17")


@pytest.mark.parametrize("kind", ["third-party-hana", "third-party-api", "third-party-cds"])
async def test_third_party_mcp_validates_local_schema(access, monkeypatch, kind):
    config = dict(
        kind=kind,
        access=access,
        transport="http",
        url="https://worker.example/mcp",
        headers={"Authorization": "Bearer token"},
    )
    with pytest.raises(ValidationError, match="input model"):
        SAPMCPServer(**config, manifest={"read": {"read_only": True, "document_types": []}})
    server = SAPMCPServer(
        **config, manifest={"read": SAPMCPBinding(read_only=True, document_types=[], input_model=Item)}
    )
    request = AsyncMock(return_value=types.CallToolResult(content=[types.TextContent(type="text", text="ok")]))
    monkeypatch.setattr(MCPServer, "_request", request)
    tool = server._make_tool(types.Tool(name="read", inputSchema={"type": "object"}))
    assert tool.params_model_schema["required"] == ["id"]
    assert tool.read_only
    with pytest.raises(ValidationError):
        await tool.handler(id="1", unknown="x")
    request.assert_not_awaited()
    assert await tool.handler(id="1") == "ok"
    await server.close()


async def test_odata_update_binds_keys_to_the_configured_collection(access, http):
    requests, responses = http

    class Update(Item):
        quantity: int

    update = http_tool(
        SAPODataWrite, access, method="PATCH", key_fields=["id"], input_model=Update, etag='"v1"', document_types=[]
    )
    responses.append(httpx.Response(204))
    await update.handler(payload={"id": "a/b'c", "quantity": 2})
    assert requests[0].url.raw_path == b"/service/Orders(id=%27a%2Fb%27%27c%27)"
    assert requests[0].headers["if-match"] == '"v1"'
    assert json.loads(requests[0].content) == {"quantity": 2}


def test_read_only_is_declared_on_every_tool(access):
    def build(_payload):
        return ET.Element("{urn:example}Op")

    soap = dict(
        operation_tag="{urn:example}Op", action="urn:op", input_model=Item, document_types=[], build_element=build
    )
    bapi = dict(function="BAPI_X", connect=MagicMock(), document_types=[], input_model=Item)
    tools = [
        (odata_read(access), True),
        (http_tool(SAPODataWrite, access, method="POST", input_model=Item, document_types=[]), False),
        (http_tool(SAPSOAPOperation, access, read_only=True, **soap), True),
        (http_tool(SAPSOAPOperation, access, read_only=False, **soap), False),
        (http_tool(SAPAgentSend, access, document_types=[]), False),
        (http_tool(SAPAgentGetTask, access), True),
        (hana_tool(access), True),
        (hana_tool(access, SAPHANAQuery), True),
        (hana_tool(access, SAPHANAWrite, statement="DELETE FROM T WHERE ID = ?", object_name="T"), False),
        (SAPCDSRead(access=access, view="V", statement="SELECT 1 FROM V", connect=MagicMock()), True),
        (SAPBAPI(access=access, read_only=True, **bapi), True),
        (SAPBAPI(access=access, read_only=False, **bapi), False),
        (
            SAPGUIScript(
                access=access,
                nonproduction_system="QAS",
                client="100",
                user="TECH",
                script=MagicMock(),
                document_types=[],
                input_model=Item,
            ),
            False,
        ),
    ]
    for tool, read_only in tools:
        assert tool.read_only is read_only, tool.name
        # A read-only tool policy offers exactly the readers.
        assert Agent._tool_allowed(tool, "read_only") is read_only, tool.name
    with pytest.raises(ValidationError):
        hana_tool(access, SAPHANAWrite, statement="DELETE FROM T", object_name="T", read_only=True)
    with pytest.raises(ValidationError):
        odata_read(access, read_only=False)
    with pytest.raises(ValidationError):
        http_tool(SAPSOAPOperation, access, read_only=True, **{**soap, "document_types": ["order"]})


def test_sap_tools_keep_payloads_out_of_event_logs(access):
    assert odata_read(access)._log_events == set()
    assert hana_tool(access)._log_events == set()
    assert Tool(name="plain", handler=lambda: None)._log_events
