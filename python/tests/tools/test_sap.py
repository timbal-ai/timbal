import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree as ET

import httpx
import pytest
from mcp import types
from pydantic import BaseModel, ConfigDict, ValidationError
from timbal import Agent, Tool, Workflow
from timbal.core.mcp import MCPServer
from timbal.core.test_model import TestModel
from timbal.state import set_run_context
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
from timbal.tools.sap._native import _gui_worker
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str


@pytest.fixture
def access():
    return SAPAccess(catalog_id="catalog@1", scope_reference="accepted-scope")


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
    config = {"host": "hana.example", "user": "tech", "password": "test", "access": access}
    if cls is SAPHANARead:
        config.update(schema_name="REPORTING", view="OPEN_ORDERS", columns=["ID"], order_by=["ID"])
    return cls(**config, **kwargs)


async def test_hana_framework_entrypoint_pagination(access, hana):
    connect, connection, cursor = hana
    tool = hana_tool(access)
    r = await tool(limit=2, filters={"ID": "x' OR 1=1"}, offset=3).collect()
    assert r.output == {"columns": ["ID"], "rows": [[1], [2]], "truncated": True, "next_offset": 5}
    options = connect.call_args.kwargs
    assert (options["port"], options["encrypt"], options["sslValidateCertificate"]) == (443, True, True)
    assert options["sslMinProtocolVersion"] == "TLS12"
    assert cursor.execute.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)
    assert cursor.execute.call_args.args == (
        'SELECT "ID" FROM "REPORTING"."OPEN_ORDERS" WHERE "ID" = ? ORDER BY "ID" LIMIT ? OFFSET ?',
        ("x' OR 1=1", 3, 3),
    )
    assert set(tool.params_model_schema["properties"]) == {"filters", "limit", "offset"}
    cursor.close.assert_called_once()
    connection.close.assert_called_once()


async def test_hana_scope_and_filter_guards(access, hana):
    tool = hana_tool(access)
    access.scope_reference = None
    with pytest.raises(PermissionError, match="scope"):
        await tool.handler()
    access.scope_reference = "ref"
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
async def test_hana_freeform_gate_and_read_transaction(access, hana, sql):
    tool = hana_tool(access, SAPHANAQuery)
    with pytest.raises(PermissionError, match="B"):
        await tool.handler(query="SELECT 1")
    access.authorizations["B(i)(b)"] = "ref"
    with pytest.raises(ValueError):
        await tool.handler(query=sql)
    hana[0].assert_not_called()
    await tool.handler(query="SELECT 'delete;drop' FROM \"orders\";")
    assert hana[2].execute.call_args_list[0].args == ("SET TRANSACTION READ ONLY",)


async def test_hana_write_is_preconfigured_and_transactional(access, hana):
    tool = hana_tool(access, SAPHANAWrite, statement='INSERT INTO "S"."T" VALUES (?)', object_name="S.T")
    with pytest.raises(PermissionError):
        await tool.handler(bindings=[1])
    access.authorizations["B(i)(c)"] = "ref"
    hana[2].rowcount = 1
    assert await tool.handler(bindings=[1]) == {"affected_rows": 1}
    hana[1].commit.assert_called_once()
    hana[2].execute.side_effect = RuntimeError("upstream failure")
    with pytest.raises(RuntimeError):
        await tool.handler(bindings=[1])
    hana[1].commit.assert_called_once()
    assert hana[1].close.call_count == 2


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


def http_tool(cls, access, **kwargs):
    return cls(
        access=access,
        endpoint="https://sap.example/service/Orders",
        object_name="Orders",
        token_provider=AsyncMock(return_value="exchanged"),
        **kwargs,
    )


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


async def test_odata_write_schema_document_gate_csrf_cookie_no_replay(access, http):
    requests, responses = http
    tool = http_tool(
        SAPODataWrite,
        access,
        input_model=Item,
        method="POST",
        document_types=["order"],
        csrf_endpoint="https://sap.example/service/",
    )
    with pytest.raises(PermissionError, match="B\\(iv\\)"):
        await tool.handler(payload={"id": "42"})
    assert not requests
    access.document_authorizations["order"] = "accepted-order-scope"
    responses.extend(
        [
            httpx.Response(200, headers={"X-CSRF-Token": "csrf", "Set-Cookie": "sid=test; Path=/"}),
            httpx.Response(201, json={"ID": "42"}),
        ]
    )
    output = await tool(payload={"id": "42"}).collect()
    assert output.output == {"ID": "42"}
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
    with pytest.raises(PermissionError):
        await tool.handler(payload={"id": "1"})
    access.document_authorizations["material"] = "reference"
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
    with pytest.raises(PermissionError):
        await send.handler(text="create")
    access.document_authorizations["order"] = "ref"
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
    import threading

    access.timeout_seconds = 1
    access.concurrency = 1
    started, release = threading.Event(), threading.Event()

    def worker():
        started.set()
        release.wait(5)

    async def run(call):
        return await call.blocking(worker)

    pending = asyncio.create_task(access.invoke("read", "v", run, route="A(ii)"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(TimeoutError):
            await pending
        with pytest.raises(RuntimeError, match="concurrency"):
            await access.invoke("read", "v", run, route="A(ii)")
    finally:
        release.set()
    async with asyncio.timeout(2):
        while access._active:
            await asyncio.sleep(0.01)


async def test_managed_mcp_allowlist_and_document_gate(access, monkeypatch):
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
    request = AsyncMock(
        side_effect=[
            types.ListToolsResult(tools=[allowed, create, other]),
            types.CallToolResult(content=[types.TextContent(type="text", text="ok")]),
        ]
    )
    monkeypatch.setattr(MCPServer, "_request", request)
    create_tool, lookup = await server.resolve()
    assert (create_tool.name, lookup.name) == ("sap__create", "sap__lookup")
    with pytest.raises(PermissionError):
        await create_tool.handler()
    assert await lookup.handler() == "ok"
    assert request.await_count == 2
    await server.close()


async def test_gui_requires_deterministic_nonproduction_scope(access):
    change = SAPCustomizing(nonproduction_system="QAS", client="100", transport_request="QASK0001", authorization="ref")
    config = dict(
        access=access,
        nonproduction_system="QAS",
        user="TECH",
        customizing=change,
        script=lambda *_: "ok",
        document_types=[],
        input_model=Item,
    )
    with pytest.raises(ValidationError):
        SAPGUIScript(client="200", **config)
    tool = SAPGUIScript(client="100", **config)
    with pytest.raises(PermissionError, match=r"B\(v\)"):
        await tool.handler(payload={"id": "1"})
    access.authorizations["B(v)"] = "ref"
    access.agent_selected = True
    with pytest.raises(PermissionError, match="deterministic"):
        await tool.handler(payload={"id": "1"})


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


@pytest.mark.parametrize("nested", [False, True])
async def test_agent_route_through_normal_dispatch(access, http, nested):
    requests, responses = http
    tool = http_tool(SAPODataRead, access, properties=["ID"], order_by=["ID"])
    workflow = Workflow(name="lookup")
    workflow.step(tool)

    async def wrapper():
        return (await workflow().collect()).output

    offered = Tool(name="wrapper", handler=wrapper) if nested else tool
    first = Message(
        role="assistant", content=[ToolUseContent(id="c", name=offered.name, input={})], stop_reason="tool_use"
    )
    agent = Agent(name="test", model=TestModel(responses=[first, "done"]), tools=[offered], max_iter=3)
    await agent(prompt="read").collect()
    assert not requests
    set_run_context(None)  # A new conversation must replay the test model's first tool call.
    access.authorizations["B(ii)"] = "written"
    responses.append(httpx.Response(200, json={"value": [{"ID": "42"}]}))
    await agent(prompt="read").collect()
    assert len(requests) == 1


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

    pending = asyncio.create_task(access.invoke("one", "v", block, route="A(ii)"))
    await started.wait()
    with pytest.raises(RuntimeError, match="concurrency"):
        await access.invoke("two", "v", block, route="A(ii)")
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
    change = SAPCustomizing(
        nonproduction_system="https://sap.example",
        client="100",
        transport_request="TR17",
        authorization="accepted-change",
    )
    config = dict(
        method="POST", input_model=Item, document_types=[], customizing=change, transport_query_parameter="transport"
    )
    tool = http_tool(SAPODataWrite, access, **config)
    responses.append(httpx.Response(204))
    await tool.handler(payload={"id": "1"})
    assert dict(requests[0].url.params) == {"sap-client": "100", "transport": "TR17"}
    with pytest.raises(ValidationError, match="origin"):
        SAPODataWrite(access=access, endpoint="https://other.example/changes", object_name="Change", **config)


async def test_native_bapi_customizing_checks_actual_destination(access):
    connection = MagicMock()
    change = SAPCustomizing(nonproduction_system="QAS", client="100", transport_request="TR17", authorization="ref")
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


@pytest.mark.parametrize(
    ("kind", "route"), [("third-party-hana", "B(i)(a)"), ("third-party-api", "B(ii)"), ("third-party-cds", "B(iii)")]
)
async def test_third_party_mcp_requires_route_and_validates_local_schema(access, monkeypatch, kind, route):
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
    with pytest.raises(PermissionError):
        await tool.handler(id="1")
    access.authorizations[route] = "scope"
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
