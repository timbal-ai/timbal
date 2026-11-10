"""Tests for Qlik Cloud tools: personal (user OAuth) + service (M2M) auth, REST wiring, Engine hypercube."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.tools import qlik
from timbal.tools.qlik import (
    QlikAuth,
    QlikEngineError,
    QlikGetAppFields,
    QlikListApps,
    QlikQuery,
    QlikWhoAmI,
    _resolve_auth,
    build_hypercube_def,
    engine_hypercube,
    normalize_tenant_url,
    page_plan,
)

TENANT = "https://example.eu.qlikcloud.com"


def _tool(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "integration": None,
        "mode": None,
        "tenant_url": TENANT,
        "token": None,
        "refresh_token": None,
        "client_id": None,
        "client_secret": None,
        "api_key": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _integration(creds: dict[str, Any]) -> MagicMock:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value=creds)
    return integration


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "QLIK_TENANT_URL",
        "QLIK_ACCESS_TOKEN",
        "QLIK_REFRESH_TOKEN",
        "QLIK_CLIENT_ID",
        "QLIK_CLIENT_SECRET",
        "QLIK_API_KEY",
        "QLIK_AUTH_MODE",
        "QLIK_SERVICE_SCOPES",
    ):
        monkeypatch.delenv(var, raising=False)
    qlik._TOKEN_CACHE.clear()


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    """Route every httpx.AsyncClient the module opens through `handler`; returns the requests seen."""
    seen: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(record), **kwargs)
    )
    return seen


# ---------------------------------------------------------------------------
# Tenant URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["example.eu.qlikcloud.com", "https://example.eu.qlikcloud.com/", " https://example.eu.qlikcloud.com/hub "],
)
def test_normalize_tenant_url(raw: str) -> None:
    assert normalize_tenant_url(raw) == TENANT


def test_normalize_tenant_url_rejects_http() -> None:
    with pytest.raises(ValueError, match="https"):
        normalize_tenant_url("http://example.eu.qlikcloud.com")


@pytest.mark.asyncio
async def test_tenant_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLIK_TENANT_URL", "example.eu.qlikcloud.com")
    auth = await _resolve_auth(_tool(tenant_url=None, token=SecretStr("user")))
    assert auth.tenant_url == TENANT


@pytest.mark.asyncio
async def test_tenant_url_from_integration() -> None:
    integration = _integration({"tenant_url": TENANT, "access_token": "user"})
    auth = await _resolve_auth(_tool(tenant_url=None, integration=integration))
    assert auth == QlikAuth(TENANT, "user", "personal")


@pytest.mark.asyncio
async def test_missing_tenant_raises() -> None:
    with pytest.raises(CredentialNotAvailable) as exc_info:
        await _resolve_auth(_tool(tenant_url=None, token=SecretStr("user")))
    assert exc_info.value.provider_name == "Qlik"
    assert exc_info.value.missing == ["tenant_url"]


# ---------------------------------------------------------------------------
# Personal mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_user_token() -> None:
    assert await _resolve_auth(_tool(token=SecretStr("user-oauth"))) == QlikAuth(TENANT, "user-oauth", "personal")


@pytest.mark.asyncio
async def test_user_token_from_integration() -> None:
    auth = await _resolve_auth(_tool(integration=_integration({"token": "platform-user-token"})))
    assert auth.token == "platform-user-token"
    assert auth.mode == "personal"


@pytest.mark.asyncio
async def test_user_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLIK_ACCESS_TOKEN", "env-user")
    assert (await _resolve_auth(_tool())).token == "env-user"


@pytest.mark.asyncio
async def test_refresh_token_exchanged_with_client(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange = AsyncMock(return_value="refreshed-access")
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_refresh_token", exchange)
    tool = _tool(refresh_token=SecretStr("rt"), client_id="cid", client_secret=SecretStr("secret"))
    assert await _resolve_auth(tool) == QlikAuth(TENANT, "refreshed-access", "personal")
    exchange.assert_awaited_once_with(TENANT, "cid", "secret", "rt")


@pytest.mark.asyncio
async def test_user_token_beats_service_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    cc = AsyncMock(return_value="service-token")
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_client_credentials", cc)
    tool = _tool(token=SecretStr("user"), client_id="cid", client_secret=SecretStr("secret"), api_key=SecretStr("k"))
    assert (await _resolve_auth(tool)).mode == "personal"
    cc.assert_not_awaited()


@pytest.mark.asyncio
async def test_personal_mode_never_falls_back_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    cc = AsyncMock(return_value="service-token")
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_client_credentials", cc)
    tool = _tool(mode="personal", client_id="cid", client_secret=SecretStr("secret"), api_key=SecretStr("k"))
    with pytest.raises(CredentialNotAvailable):
        await _resolve_auth(tool)
    cc.assert_not_awaited()


@pytest.mark.asyncio
async def test_mode_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLIK_AUTH_MODE", "personal")
    monkeypatch.setenv("QLIK_API_KEY", "service-key")
    with pytest.raises(CredentialNotAvailable):
        await _resolve_auth(_tool())


# ---------------------------------------------------------------------------
# Service mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_credentials_from_tool_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    cc = AsyncMock(return_value="service-token")
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_client_credentials", cc)
    tool = _tool(client_id="cid", client_secret=SecretStr("secret"))
    assert await _resolve_auth(tool) == QlikAuth(TENANT, "service-token", "service")
    cc.assert_awaited_once_with(TENANT, "cid", "secret")


@pytest.mark.asyncio
async def test_client_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLIK_TENANT_URL", TENANT)
    monkeypatch.setenv("QLIK_CLIENT_ID", "cid")
    monkeypatch.setenv("QLIK_CLIENT_SECRET", "secret")
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_client_credentials", AsyncMock(return_value="svc"))
    assert await _resolve_auth(_tool(tenant_url=None)) == QlikAuth(TENANT, "svc", "service")


@pytest.mark.asyncio
async def test_client_credentials_from_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_client_credentials", AsyncMock(return_value="svc"))
    integration = _integration({"tenant_url": TENANT, "client_id": "cid", "client_secret": "secret"})
    assert (await _resolve_auth(_tool(tenant_url=None, integration=integration))).mode == "service"


@pytest.mark.asyncio
async def test_api_key_is_service_mode() -> None:
    assert await _resolve_auth(_tool(api_key=SecretStr("qlik-api-key"))) == QlikAuth(TENANT, "qlik-api-key", "service")


@pytest.mark.asyncio
async def test_service_mode_ignores_user_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timbal.tools.qlik.get_token_from_client_credentials", AsyncMock(return_value="svc"))
    tool = _tool(mode="service", token=SecretStr("user"), client_id="cid", client_secret=SecretStr("secret"))
    assert (await _resolve_auth(tool)).token == "svc"


@pytest.mark.asyncio
async def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="personal' or 'service"):
        await _resolve_auth(_tool(mode="robot", token=SecretStr("user")))


@pytest.mark.asyncio
async def test_missing_credentials_raises() -> None:
    with pytest.raises(CredentialNotAvailable) as exc_info:
        await _resolve_auth(_tool())
    assert exc_info.value.provider_name == "Qlik"
    assert "QLIK_CLIENT_ID" in exc_info.value.env_vars


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_credentials_grant_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"access_token": "svc", "expires_in": 3600}))
    assert await qlik.get_token_from_client_credentials(TENANT, "cid", "secret") == "svc"
    assert await qlik.get_token_from_client_credentials(TENANT, "cid", "secret") == "svc"
    assert len(seen) == 1
    assert str(seen[0].url) == f"{TENANT}/oauth/token"
    body = json.loads(seen[0].content)
    assert body == {
        "grant_type": "client_credentials",
        "client_id": "cid",
        "client_secret": "secret",
        "scope": "user_default",
    }


@pytest.mark.asyncio
async def test_expired_cached_token_is_renewed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"access_token": "svc", "expires_in": 3600}))
    await qlik.get_token_from_client_credentials(TENANT, "cid", "secret")
    key = next(iter(qlik._TOKEN_CACHE))
    token, _, refresh = qlik._TOKEN_CACHE[key]
    qlik._TOKEN_CACHE[key] = (token, 0.0, refresh)
    await qlik.get_token_from_client_credentials(TENANT, "cid", "secret")
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_rotated_refresh_token_is_used_next_time(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            {"access_token": "a1", "refresh_token": "rt2", "expires_in": 3600},
            {"access_token": "a2", "refresh_token": "rt3", "expires_in": 3600},
        ]
    )
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json=next(responses)))
    assert await qlik.get_token_from_refresh_token(TENANT, "cid", "secret", "rt1") == "a1"
    key = next(iter(qlik._TOKEN_CACHE))
    token, _, refresh = qlik._TOKEN_CACHE[key]
    qlik._TOKEN_CACHE[key] = (token, 0.0, refresh)  # force a renewal
    assert await qlik.get_token_from_refresh_token(TENANT, "cid", "secret", "rt1") == "a2"
    assert [json.loads(r.content)["refresh_token"] for r in seen] == ["rt1", "rt2"]


@pytest.mark.asyncio
async def test_oauth5_explains_trusted_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(
        monkeypatch,
        lambda _: httpx.Response(400, json={"errors": [{"code": "OAUTH-5", "title": "unauthorized client"}]}),
    )
    with pytest.raises(ValueError, match="Trusted") as exc_info:
        await qlik.get_token_from_client_credentials(TENANT, "cid", "secret")
    assert "OAUTH-5" in str(exc_info.value)


@pytest.mark.asyncio
async def test_failed_refresh_names_the_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(400, json={"error": "invalid_grant"}))
    with pytest.raises(ValueError, match="refresh_token.*invalid_grant"):
        await qlik.get_token_from_refresh_token(TENANT, "cid", "secret", "rt")


# ---------------------------------------------------------------------------
# REST tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLIK_TENANT_URL", TENANT)
    monkeypatch.setenv("QLIK_ACCESS_TOKEN", "user")
    seen = _mock_http(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "data": [
                    {"resourceId": "app-1", "name": "Sales", "spaceId": "sp", "ownerId": "u1", "updatedAt": "2026"}
                ],
                "links": {"next": {"href": f"{TENANT}/api/v1/items?resourceType=app&next=CURSOR2"}},
            },
        ),
    )
    out = await QlikListApps()(query=" sales ", limit=500).collect()
    assert out.status.code == "success", out.error
    assert out.output == {
        "apps": [
            {
                "app_id": "app-1",
                "name": "Sales",
                "description": None,
                "space_id": "sp",
                "owner_id": "u1",
                "updated_at": "2026",
            }
        ],
        "count": 1,
        "next": "CURSOR2",
    }
    request = seen[0]
    assert request.url.path == "/api/v1/items"
    assert dict(request.url.params) == {"resourceType": "app", "limit": "100", "noActions": "true", "query": "sales"}
    assert request.headers["Authorization"] == "Bearer user"


@pytest.mark.asyncio
async def test_get_app_fields_groups_fields_by_table(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {
        "has_section_access": True,
        "tables": [
            {"name": "Sales", "no_of_rows": 1200, "is_system": False},
            {"name": "$$SysTable", "no_of_rows": 3, "is_system": True},
        ],
        "fields": [
            {"name": "Customer", "src_tables": ["Sales"], "cardinal": 40, "tags": ["$key", "$text"]},
            {"name": "Amount", "src_tables": ["Sales"], "cardinal": 900, "tags": ["$numeric"], "is_hidden": True},
            {"name": "$Field", "src_tables": ["$$SysTable"], "is_system": True},
        ],
    }
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json=metadata))
    out = await QlikGetAppFields(token=SecretStr("user"), tenant_url=TENANT)(app_id=" app-1 ").collect()
    assert out.status.code == "success", out.error
    assert seen[0].url.path == "/api/v1/apps/app-1/data/metadata"
    assert out.output == {
        "app_id": "app-1",
        "has_section_access": True,
        "tables": [
            {
                "name": "Sales",
                "rows": 1200,
                "fields": [
                    {"name": "Customer", "distinct_values": 40, "tags": []},
                    {"name": "Amount", "distinct_values": 900, "tags": [], "hidden": True},
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_get_app_fields_no_access(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(403, json={}))
    out = await QlikGetAppFields(token=SecretStr("user"), tenant_url=TENANT)(app_id="app-1").collect()
    assert out.status.code == "error"
    assert "may not open it" in out.error["message"]


@pytest.mark.asyncio
async def test_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "id": "u1",
                "name": "Ada",
                "email": "ada@example.com",
                "tenantId": "t1",
                "status": "active",
                "assignedRoles": [{"name": "TenantAdmin"}],
            },
        ),
    )
    out = await QlikWhoAmI(token=SecretStr("user"), tenant_url=TENANT)().collect()
    assert out.status.code == "success", out.error
    assert out.output["name"] == "Ada"
    assert out.output["roles"] == ["TenantAdmin"]
    assert out.output["mode"] == "personal"


@pytest.mark.asyncio
async def test_401_tells_a_person_to_sign_in_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(401, json={}))
    out = await QlikWhoAmI(token=SecretStr("expired"), tenant_url=TENANT)().collect()
    assert out.status.code == "error"
    assert "Sign in to Qlik again" in out.error["message"]


def test_config_exposes_both_modes() -> None:
    config = QlikQuery().get_config()
    for key in ("integration", "mode", "tenant_url", "token", "refresh_token", "client_id", "client_secret", "api_key"):
        assert key in config


# ---------------------------------------------------------------------------
# Engine API
# ---------------------------------------------------------------------------


def test_build_hypercube_def() -> None:
    cube = build_hypercube_def(["Region"], ["Sum(Sales)"])
    assert cube["qDimensions"] == [{"qDef": {"qFieldDefs": ["Region"]}, "qNullSuppression": False}]
    assert cube["qMeasures"] == [{"qDef": {"qDef": "Sum(Sales)", "qLabel": "Sum(Sales)"}}]
    assert cube["qInterColumnSortOrder"] == [0, 1]


def test_page_plan_respects_cell_limit() -> None:
    pages = page_plan(width=3, max_rows=8000, total_rows=20000)
    assert [p["qHeight"] for p in pages] == [3333, 3333, 1334]
    assert [p["qTop"] for p in pages] == [0, 3333, 6666]
    assert all(p["qWidth"] * p["qHeight"] <= 10_000 for p in pages)


def test_page_plan_stops_at_total() -> None:
    assert page_plan(width=2, max_rows=100, total_rows=7) == [{"qTop": 0, "qLeft": 0, "qWidth": 2, "qHeight": 7}]
    assert page_plan(width=2, max_rows=100, total_rows=0) == []


class _FakeEngine:
    """Answers the JSON-RPC calls engine_hypercube makes, after an OnConnected notification."""

    def __init__(self, *, total: int = 2, error_on: str | None = None) -> None:
        self.total = total
        self.error_on = error_on
        self.sent: list[dict[str, Any]] = []
        self.outbox: list[str] = [json.dumps({"jsonrpc": "2.0", "method": "OnConnected", "params": {}})]
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def connect(self, url: str, **kwargs: Any) -> "_FakeEngine":
        self.url = url
        self.headers = kwargs.get("additional_headers")
        return self

    async def __aenter__(self) -> "_FakeEngine":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        method = message["method"]
        if method == self.error_on:
            reply: dict[str, Any] = {"error": {"code": 11001, "message": "Bad expression", "parameter": "Sum(X"}}
        elif method == "OpenDoc":
            reply = {"result": {"qReturn": {"qHandle": 1}}}
        elif method == "CreateSessionObject":
            reply = {"result": {"qReturn": {"qHandle": 2}}}
        elif method == "GetLayout":
            reply = {
                "result": {
                    "qLayout": {
                        "qHyperCube": {
                            "qSize": {"qcx": 2, "qcy": self.total},
                            "qDimensionInfo": [{"qFallbackTitle": "Region"}],
                            "qMeasureInfo": [{"qFallbackTitle": "Sum(Sales)"}],
                        }
                    }
                }
            }
        else:
            reply = {
                "result": {
                    "qDataPages": [
                        {
                            "qMatrix": [
                                [{"qText": "North", "qNum": "NaN"}, {"qText": "1.5k", "qNum": 1500}],
                                [{"qText": "South", "qNum": "NaN"}, {"qText": "-", "qNum": "NaN"}],
                            ]
                        }
                    ]
                }
            }
        self.outbox.append(json.dumps({"jsonrpc": "2.0", "id": message["id"], **reply}))

    async def recv(self) -> str:
        return self.outbox.pop(0)


@pytest.mark.asyncio
async def test_engine_hypercube(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _FakeEngine(total=5)
    monkeypatch.setattr("websockets.asyncio.client.connect", engine.connect)
    result = await engine_hypercube(QlikAuth(TENANT, "tok", "personal"), "app-1", ["Region"], ["Sum(Sales)"], 2)
    assert engine.url == "wss://example.eu.qlikcloud.com/app/app-1"
    assert engine.headers == {"Authorization": "Bearer tok"}
    assert [m["method"] for m in engine.sent] == ["OpenDoc", "CreateSessionObject", "GetLayout", "GetHyperCubeData"]
    assert engine.sent[3]["params"] == ["/qHyperCubeDef", [{"qTop": 0, "qLeft": 0, "qWidth": 2, "qHeight": 2}]]
    assert result == {
        "columns": [{"name": "Region", "kind": "dimension"}, {"name": "Sum(Sales)", "kind": "measure"}],
        "rows": [["North", 1500], ["South", "-"]],
        "total_rows": 5,
        "truncated": True,
    }


@pytest.mark.asyncio
async def test_engine_error_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _FakeEngine(error_on="GetLayout")
    monkeypatch.setattr("websockets.asyncio.client.connect", engine.connect)
    with pytest.raises(QlikEngineError, match="Bad expression"):
        await engine_hypercube(QlikAuth(TENANT, "tok", "service"), "app-1", ["Region"], ["Sum(X"], 10)


@pytest.mark.asyncio
async def test_query_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _FakeEngine(total=2)
    monkeypatch.setattr("websockets.asyncio.client.connect", engine.connect)
    out = await QlikQuery(token=SecretStr("user"), tenant_url=TENANT)(
        app_id="app-1", dimensions=["Region", " "], measures=["Sum(Sales)"], max_rows=50
    ).collect()
    assert out.status.code == "success", out.error
    assert out.output["app_id"] == "app-1"
    assert out.output["truncated"] is False
    assert engine.sent[1]["params"][0]["qHyperCubeDef"]["qDimensions"][0]["qDef"]["qFieldDefs"] == ["Region"]


@pytest.mark.asyncio
async def test_query_tool_needs_a_dimension_or_measure() -> None:
    out = await QlikQuery(token=SecretStr("user"), tenant_url=TENANT)(app_id="app-1").collect()
    assert out.status.code == "error"
    assert "at least one dimension or one measure" in out.error["message"]
