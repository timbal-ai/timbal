"""Qlik Cloud (Qlik Sense) tools: apps, app data model, hypercube queries, whoami.

Two connection modes, the same split as Power BI's:

- **personal** (user-delegated OAuth): the tools act as one person, with that person's
  Qlik roles, space access and Section Access. Token from ``Integration("qlik")``
  (``token`` / ``access_token``), tool field ``token``, or env ``QLIK_ACCESS_TOKEN``;
  or a ``refresh_token`` (integration, tool field, env ``QLIK_REFRESH_TOKEN``) that is
  exchanged with the OAuth client's ``client_id`` + ``client_secret``.
- **service** (machine-to-machine): the tools act as the OAuth client itself.
  ``client_id`` + ``client_secret`` go through the ``client_credentials`` grant, or a
  Qlik API key (``api_key`` / env ``QLIK_API_KEY``) is sent as the Bearer token.
  The OAuth client must allow machine-to-machine access and use the "Trusted"
  consent method; otherwise Qlik answers the grant with ``OAUTH-5``.

Every mode needs the tenant URL (``https://<tenant>.<region>.qlikcloud.com``): tool
field ``tenant_url``, integration ``tenant_url``, or env ``QLIK_TENANT_URL``.

``mode`` (tool field, integration ``mode``, or env ``QLIK_AUTH_MODE``) pins one of the
two. Unset, the first credential found wins and personal sources are tried before
service ones. Set ``mode="personal"`` wherever a user without a token must be refused
rather than served through the wider service identity.

Integration credentials (type: oauth or credentials):
- tenant_url: Qlik Cloud tenant URL
- token / access_token: a user's OAuth access token (personal)
- refresh_token: a user's OAuth refresh token (personal, needs client_id + client_secret)
- client_id / client_secret: the Qlik OAuth client
- api_key: a Qlik API key (service)
- mode: "personal" or "service"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from typing import Annotated, Any, Literal, NamedTuple
from urllib.parse import parse_qs, urlparse

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration

#: Scopes a personal sign-in asks for; ``offline_access`` yields the refresh token.
PERSONAL_SCOPES = "user_default offline_access"
#: Scope requested by the ``client_credentials`` grant.
SERVICE_SCOPES = "user_default"

#: The Engine API serves at most 10,000 cells per GetHyperCubeData page.
_ENGINE_PAGE_CELLS = 10_000
_MAX_QUERY_ROWS = 10_000
#: Seconds before expiry at which a cached token is renewed.
_TOKEN_SKEW = 60

_ENV_VARS = [
    "QLIK_TENANT_URL",
    "QLIK_ACCESS_TOKEN",
    "QLIK_REFRESH_TOKEN",
    "QLIK_CLIENT_ID",
    "QLIK_CLIENT_SECRET",
    "QLIK_API_KEY",
]

Mode = Literal["personal", "service"]


class QlikAuth(NamedTuple):
    tenant_url: str
    token: str
    mode: Mode


class QlikEngineError(ValueError):
    """The Qlik Engine answered a JSON-RPC call with an error."""


# (tenant, client, hash of the secret material) -> (access token, expires at, latest refresh token)
_TOKEN_CACHE: dict[str, tuple[str, float, str | None]] = {}


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    text = str(value).strip()
    return text or None


def _first(*values: Any) -> str | None:
    for value in values:
        text = _secret_value(value)
        if text:
            return text
    return None


def normalize_tenant_url(value: str) -> str:
    """``tenant.eu.qlikcloud.com`` or ``https://tenant.eu.qlikcloud.com/`` -> ``https://tenant.eu.qlikcloud.com``."""
    text = value.strip().rstrip("/")
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Qlik tenant URL must be an https URL, got {value!r}.")
    return f"https://{parsed.netloc}"


def _tenant_url(tool: Any, creds: dict[str, Any]) -> str:
    raw = _first(
        getattr(tool, "tenant_url", None),
        creds.get("tenant_url"),
        creds.get("tenant"),
        os.getenv("QLIK_TENANT_URL"),
    )
    if not raw:
        raise CredentialNotAvailable("Qlik", missing=["tenant_url"], env_vars=["QLIK_TENANT_URL"])
    return normalize_tenant_url(raw)


def _mode(tool: Any, creds: dict[str, Any]) -> Mode | None:
    raw = _first(getattr(tool, "mode", None), creds.get("mode"), os.getenv("QLIK_AUTH_MODE"))
    if raw is None:
        return None
    mode = raw.lower()
    if mode not in ("personal", "service"):
        raise ValueError(f"Qlik mode must be 'personal' or 'service', got {raw!r}.")
    return mode  # type: ignore[return-value]


def _client_parts(tool: Any, creds: dict[str, Any]) -> tuple[str, str] | None:
    client_id = _first(getattr(tool, "client_id", None), creds.get("client_id"), os.getenv("QLIK_CLIENT_ID"))
    client_secret = _first(
        getattr(tool, "client_secret", None), creds.get("client_secret"), os.getenv("QLIK_CLIENT_SECRET")
    )
    if client_id and client_secret:
        return client_id, client_secret
    return None


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _token_error(response: Any) -> str:
    """Qlik's error code and title from a failed /oauth/token response."""
    try:
        body = response.json()
    except ValueError:
        return (getattr(response, "text", "") or "")[:200]
    errors = body.get("errors") if isinstance(body, dict) else None
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        first = errors[0]
        return " ".join(str(v) for v in (first.get("code"), first.get("title"), first.get("detail")) if v)
    if isinstance(body, dict):
        return " ".join(str(v) for v in (body.get("error"), body.get("error_description")) if v)
    return str(body)[:200]


async def _token_request(tenant_url: str, payload: dict[str, str]) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(
            f"{tenant_url}/oauth/token",
            json=payload,
            headers={"Accept": "application/json"},
        )
    if response.status_code >= 400:
        detail = _token_error(response)
        if "OAUTH-5" in detail and payload.get("grant_type") == "client_credentials":
            raise ValueError(
                "Qlik refused the client_credentials grant (OAUTH-5). Service mode needs an OAuth client "
                "that allows machine-to-machine access and uses the 'Trusted' consent method "
                f"(Administration > OAuth). Upstream: {detail}"
            )
        raise ValueError(f"Qlik token request ({payload.get('grant_type')}) failed: {response.status_code} {detail}")
    body = response.json()
    if not isinstance(body.get("access_token"), str):
        raise ValueError("Qlik token response did not include access_token.")
    return body


def _expires_at(body: dict[str, Any]) -> float:
    try:
        lifetime = int(body.get("expires_in") or 3600)
    except (TypeError, ValueError):
        lifetime = 3600
    return time.time() + max(60, lifetime)


async def get_token_from_client_credentials(tenant_url: str, client_id: str, client_secret: str) -> str:
    """Service mode: an access token for the OAuth client itself, cached until shortly before it expires."""
    key = _cache_key("client_credentials", tenant_url, client_id, client_secret)
    cached = _TOKEN_CACHE.get(key)
    if cached and cached[1] - time.time() > _TOKEN_SKEW:
        return cached[0]
    body = await _token_request(
        tenant_url,
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": os.getenv("QLIK_SERVICE_SCOPES", SERVICE_SCOPES),
        },
    )
    _TOKEN_CACHE[key] = (body["access_token"], _expires_at(body), None)
    return body["access_token"]


async def get_token_from_refresh_token(tenant_url: str, client_id: str, client_secret: str, refresh_token: str) -> str:
    """Personal mode: a user's access token from their refresh token.

    Qlik may rotate the refresh token on every use, so the newest one is kept in the
    cache entry and used for the next exchange. Persist it yourself if the process
    restarts often (``exchange_refresh_token`` returns the whole token response).
    """
    key = _cache_key("refresh_token", tenant_url, client_id, client_secret, refresh_token)
    cached = _TOKEN_CACHE.get(key)
    if cached and cached[1] - time.time() > _TOKEN_SKEW:
        return cached[0]
    current = cached[2] if cached and cached[2] else refresh_token
    body = await exchange_refresh_token(tenant_url, client_id, client_secret, current)
    _TOKEN_CACHE[key] = (body["access_token"], _expires_at(body), body.get("refresh_token") or current)
    return body["access_token"]


async def exchange_refresh_token(
    tenant_url: str, client_id: str, client_secret: str, refresh_token: str
) -> dict[str, Any]:
    """Raw refresh_token grant: the whole token response (access_token, refresh_token, expires_in)."""
    return await _token_request(
        tenant_url,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )


async def _resolve_auth(tool: Any) -> QlikAuth:
    """Tenant URL and Bearer token for this tool (see the module docstring for the order)."""
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve() or {}

    tenant_url = _tenant_url(tool, creds)
    mode = _mode(tool, creds)

    if mode != "service":
        token = _first(
            getattr(tool, "token", None),
            creds.get("token"),
            creds.get("access_token"),
            os.getenv("QLIK_ACCESS_TOKEN"),
        )
        if token:
            return QlikAuth(tenant_url, token, "personal")
        refresh = _first(
            getattr(tool, "refresh_token", None), creds.get("refresh_token"), os.getenv("QLIK_REFRESH_TOKEN")
        )
        client = _client_parts(tool, creds)
        if refresh and client:
            token = await get_token_from_refresh_token(tenant_url, *client, refresh)
            return QlikAuth(tenant_url, token, "personal")

    if mode != "personal":
        api_key = _first(getattr(tool, "api_key", None), creds.get("api_key"), os.getenv("QLIK_API_KEY"))
        if api_key:
            return QlikAuth(tenant_url, api_key, "service")
        client = _client_parts(tool, creds)
        if client:
            token = await get_token_from_client_credentials(tenant_url, *client)
            return QlikAuth(tenant_url, token, "service")

    if mode == "personal":
        missing = ["token", "refresh_token + client_id + client_secret"]
    elif mode == "service":
        missing = ["client_id + client_secret", "api_key"]
    else:
        missing = ["token", "refresh_token", "client_id + client_secret", "api_key"]
    raise CredentialNotAvailable("Qlik", missing=missing, env_vars=_ENV_VARS)


async def _qlik_get(auth: QlikAuth, path: str, params: dict[str, Any] | None = None) -> Any:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.get(
            f"{auth.tenant_url}{path}",
            params=params or None,
            headers={"Authorization": f"Bearer {auth.token}", "Accept": "application/json"},
        )
    if response.status_code == 401:
        hint = (
            "Sign in to Qlik again (the user's token expired or was revoked)."
            if auth.mode == "personal"
            else "Check the OAuth client or API key."
        )
        raise ValueError(f"Qlik rejected the {auth.mode} credentials (401). {hint}")
    if response.status_code in (403, 404) and path.startswith("/api/v1/apps/"):
        raise ValueError(
            f"Qlik answered {response.status_code}: the app does not exist or this identity may not open it."
        )
    response.raise_for_status()
    return response.json() if response.content else {}


def _next_cursor(body: dict[str, Any]) -> str | None:
    href = ((body.get("links") or {}).get("next") or {}).get("href")
    if not href:
        return None
    values = parse_qs(urlparse(href).query).get("next")
    return values[0] if values else None


def build_hypercube_def(dimensions: list[str], measures: list[str]) -> dict[str, Any]:
    """qHyperCubeDef for dimensions (field names or ``=expressions``) by measures (Qlik expressions)."""
    width = len(dimensions) + len(measures)
    return {
        "qDimensions": [{"qDef": {"qFieldDefs": [d]}, "qNullSuppression": False} for d in dimensions],
        "qMeasures": [{"qDef": {"qDef": m, "qLabel": m}} for m in measures],
        "qInitialDataFetch": [],
        "qSuppressZero": False,
        "qSuppressMissing": False,
        "qInterColumnSortOrder": list(range(width)),
    }


def page_plan(width: int, max_rows: int, total_rows: int) -> list[dict[str, int]]:
    """GetHyperCubeData pages covering ``min(max_rows, total_rows)`` rows, each within the 10,000-cell limit."""
    wanted = max(0, min(max_rows, total_rows))
    if width <= 0 or wanted == 0:
        return []
    height = max(1, _ENGINE_PAGE_CELLS // width)
    return [
        {"qTop": top, "qLeft": 0, "qWidth": width, "qHeight": min(height, wanted - top)}
        for top in range(0, wanted, height)
    ]


def _cell(cell: dict[str, Any], is_measure: bool) -> Any:
    if is_measure:
        number = cell.get("qNum")
        if isinstance(number, int | float) and not (isinstance(number, float) and math.isnan(number)):
            return number
    return cell.get("qText")


async def engine_hypercube(
    auth: QlikAuth,
    app_id: str,
    dimensions: list[str],
    measures: list[str],
    max_rows: int,
    *,
    timeout: float = 60,
) -> dict[str, Any]:
    """Open the app over the Engine API (JSON-RPC on a websocket) and read one hypercube."""
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "qlik_query needs the `websockets` package: pip install websockets (or timbal[server])."
        ) from exc

    host = urlparse(auth.tenant_url).netloc
    cube = build_hypercube_def(dimensions, measures)
    width = len(dimensions) + len(measures)
    next_id = 0

    async def call(ws: Any, handle: int, method: str, params: Any) -> dict[str, Any]:
        nonlocal next_id
        next_id += 1
        request_id = next_id
        await ws.send(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "handle": handle, "method": method, "params": params})
        )
        while True:
            reply = json.loads(await ws.recv())
            if reply.get("id") != request_id:
                # Notifications (OnConnected, OnAuthenticationInformation, change pushes).
                if reply.get("method") == "OnAuthenticationInformation" and (reply.get("params") or {}).get(
                    "mustAuthenticate"
                ):
                    raise QlikEngineError("Qlik Engine did not accept the token for this app.")
                continue
            if "error" in reply:
                error = reply["error"]
                message = error.get("message") or error
                parameter = error.get("parameter")
                raise QlikEngineError(f"{method}: {message}{f' ({parameter})' if parameter else ''}"[:400])
            return reply.get("result") or {}

    async def run(ws: Any) -> dict[str, Any]:
        doc = (await call(ws, -1, "OpenDoc", [app_id]))["qReturn"]["qHandle"]
        created = await call(
            ws, doc, "CreateSessionObject", [{"qInfo": {"qType": "timbal-hypercube"}, "qHyperCubeDef": cube}]
        )
        obj = created["qReturn"]["qHandle"]
        layout = (await call(ws, obj, "GetLayout", []))["qLayout"]["qHyperCube"]
        if layout.get("qError"):
            raise QlikEngineError(f"Hypercube error: {layout['qError']}")
        total = int((layout.get("qSize") or {}).get("qcy", 0))
        matrix: list[list[dict[str, Any]]] = []
        for page in page_plan(width, max_rows, total):
            data = await call(ws, obj, "GetHyperCubeData", ["/qHyperCubeDef", [page]])
            for data_page in data.get("qDataPages", []):
                matrix.extend(data_page.get("qMatrix", []))
        dim_info = layout.get("qDimensionInfo", [])
        measure_info = layout.get("qMeasureInfo", [])
        columns = [
            *[{"name": d.get("qFallbackTitle"), "kind": "dimension"} for d in dim_info],
            *[{"name": m.get("qFallbackTitle"), "kind": "measure"} for m in measure_info],
        ]
        n_dims = len(dim_info)
        rows = [[_cell(c, i >= n_dims) for i, c in enumerate(row)] for row in matrix]
        return {"columns": columns, "rows": rows, "total_rows": total, "truncated": total > len(rows)}

    async def session() -> dict[str, Any]:
        async with connect(
            f"wss://{host}/app/{app_id}",
            additional_headers={"Authorization": f"Bearer {auth.token}"},
            open_timeout=20,
            max_size=64 * 1024 * 1024,
        ) as ws:
            return await run(ws)

    return await asyncio.wait_for(session(), timeout=timeout)


class _QlikTool(Tool):
    """Shared auth fields for Qlik tools (personal OAuth + service M2M)."""

    integration: Annotated[str, Integration("qlik")] | None = None
    mode: Mode | None = None
    tenant_url: str | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "mode": self.mode,
                    "tenant_url": self.tenant_url,
                    "token": self.token,
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "api_key": self.api_key,
                }
            ),
        }


class QlikWhoAmI(_QlikTool):
    name: str = "qlik_whoami"
    description: str | None = (
        "Show which Qlik Cloud user these tools act as (name, email, roles) and whether that is a "
        "person (personal) or the service identity."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _whoami() -> Any:
            auth = await _resolve_auth(self)
            body = await _qlik_get(auth, "/api/v1/users/me")
            roles = [r.get("name") for r in body.get("assignedRoles", []) if isinstance(r, dict) and r.get("name")]
            return {
                "id": body.get("id"),
                "name": body.get("name"),
                "email": body.get("email"),
                "subject": body.get("subject"),
                "status": body.get("status"),
                "tenant_id": body.get("tenantId"),
                "roles": roles or body.get("roles") or [],
                "mode": auth.mode,
                "tenant_url": auth.tenant_url,
            }

        super().__init__(handler=_whoami, **kwargs)


class QlikListApps(_QlikTool):
    name: str = "qlik_list_apps"
    description: str | None = "List the Qlik Sense apps this identity can open, optionally filtered by name or space."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_apps(
            query: str | None = Field(None, description="Words of the app name to search for. Empty lists all apps."),
            space_id: str | None = Field(None, description="Only apps in this space ID."),
            limit: int = Field(25, description="Maximum apps to return (1-100)."),
            next: str | None = Field(None, description="Cursor from a previous call's `next` to read the next page."),
        ) -> Any:
            auth = await _resolve_auth(self)
            params: dict[str, Any] = {
                "resourceType": "app",
                "limit": max(1, min(int(limit), 100)),
                "noActions": "true",
            }
            if query and query.strip():
                params["query"] = query.strip()
            if space_id:
                params["spaceId"] = space_id
            if next:
                params["next"] = next
            body = await _qlik_get(auth, "/api/v1/items", params)
            apps = [
                {
                    "app_id": item.get("resourceId"),
                    "name": item.get("name"),
                    "description": item.get("description") or None,
                    "space_id": item.get("spaceId") or None,
                    "owner_id": item.get("ownerId"),
                    "updated_at": item.get("updatedAt"),
                }
                for item in body.get("data", [])
            ]
            return {"apps": apps, "count": len(apps), "next": _next_cursor(body)}

        super().__init__(handler=_list_apps, **kwargs)


class QlikGetAppFields(_QlikTool):
    name: str = "qlik_get_app_fields"
    description: str | None = (
        "Read a Qlik app's data model: its tables, row counts and fields. Call this before qlik_query "
        "so dimensions and measures use real field names."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_app_fields(
            app_id: str = Field(..., description="App ID (`app_id` from qlik_list_apps)."),
            include_system: bool = Field(False, description="Include Qlik system tables and fields ($Field, ...)."),
        ) -> Any:
            auth = await _resolve_auth(self)
            app_id = app_id.strip()
            body = await _qlik_get(auth, f"/api/v1/apps/{app_id}/data/metadata")

            tables: dict[str, dict[str, Any]] = {}
            for table in body.get("tables", []):
                if table.get("is_system") and not include_system:
                    continue
                tables[table.get("name")] = {"name": table.get("name"), "rows": table.get("no_of_rows"), "fields": []}
            for field in body.get("fields", []):
                if field.get("is_system") and not include_system:
                    continue
                entry = {
                    "name": field.get("name"),
                    "distinct_values": field.get("cardinal"),
                    "tags": [t for t in field.get("tags", []) if not str(t).startswith("$") or include_system],
                }
                if field.get("is_hidden"):
                    entry["hidden"] = True
                for table_name in field.get("src_tables") or []:
                    if table_name in tables:
                        tables[table_name]["fields"].append(entry)
            return {
                "app_id": app_id,
                "has_section_access": body.get("has_section_access"),
                "tables": list(tables.values()),
            }

        super().__init__(handler=_get_app_fields, **kwargs)


class QlikQuery(_QlikTool):
    name: str = "qlik_query"
    description: str | None = (
        "Compute figures from a Qlik app through the Engine API: a table of dimensions (field names) by "
        "measures (Qlik expressions such as Sum(Sales) or Sum({<Year={2026}>} Sales)). Results follow the "
        "identity's own access, including Section Access."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _query(
            app_id: str = Field(..., description="App ID (`app_id` from qlik_list_apps)."),
            dimensions: list[str] = Field(
                default_factory=list,
                description="Fields to group by, exactly as qlik_get_app_fields lists them (or `=expression`).",
            ),
            measures: list[str] = Field(
                default_factory=list,
                description="Qlik aggregation expressions, e.g. `Sum(Sales)`, `Count(distinct Customer)`.",
            ),
            max_rows: int = Field(100, description=f"Maximum rows to return (1-{_MAX_QUERY_ROWS})."),
        ) -> Any:
            dims = [d.strip() for d in dimensions if d and d.strip()]
            meas = [m.strip() for m in measures if m and m.strip()]
            if not dims and not meas:
                raise ValueError("Pass at least one dimension or one measure.")
            auth = await _resolve_auth(self)
            app_id = app_id.strip()
            rows = max(1, min(int(max_rows), _MAX_QUERY_ROWS))
            try:
                result = await engine_hypercube(auth, app_id, dims, meas, rows)
            except TimeoutError as exc:
                raise TimeoutError(
                    "Qlik Engine did not answer in time. Try fewer dimensions or a narrower measure."
                ) from exc
            return {"app_id": app_id, **result}

        super().__init__(handler=_query, **kwargs)
