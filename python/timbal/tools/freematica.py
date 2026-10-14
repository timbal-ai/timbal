"""Freemática ERP REST tools (deterministic, not an MCP server).

Auth (first match wins), sent as the five ``x-auth-*`` headers Freemática requires.
There is no login endpoint; the credentials are issued out of band:

- ``Integration("freematica")`` fields ``auth_token``, ``auth_company``,
  ``auth_organization``, ``auth_app``, ``auth_session`` (optional ``base_url``).
- The same names as tool fields.
- Env ``FREEMATICA_AUTH_TOKEN``, ``FREEMATICA_AUTH_COMPANY``,
  ``FREEMATICA_AUTH_ORGANIZATION``, ``FREEMATICA_AUTH_APP``, ``FREEMATICA_AUTH_SESSION``.
- ``FREEMATICA_BASE_URL`` (default ``https://api-p01.clientservicepanel.com/restsat/api``).

Every response is the envelope ``{errorCode, errorMessage, data}``. ``errorCode``
``"200"`` is success. List payloads ``{total, items, rowHeight}`` become
``{items, total, count}``.

Filters are typed parameters. FIQL (``rquery``) is built internally; the model
never writes it. ``page`` is 1-indexed and must be >= 1: the API treats ``page=0``
as "return the entire dataset". Detail ids are the opaque base64 ``idReg`` from a
list, not natural codes such as ``COD_CLI``.

Behavior is ported from the public client at
https://github.com/nubiia-dev/mcp-freematica (filters, quirks, write field names).
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
from typing import Annotated, Any, Literal
from urllib.parse import quote

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration

_DEFAULT_BASE_URL = "https://api-p01.clientservicepanel.com/restsat/api"
_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
_MAX_JITTER_SECONDS = 0.2
_RETRYABLE = frozenset({"server_error", "network_error", "rate_limit_exceeded"})

_AUTH_FIELDS = (
    ("auth_token", "FREEMATICA_AUTH_TOKEN", "x-auth-token"),
    ("auth_company", "FREEMATICA_AUTH_COMPANY", "x-auth-company"),
    ("auth_organization", "FREEMATICA_AUTH_ORGANIZATION", "x-auth-organization"),
    ("auth_app", "FREEMATICA_AUTH_APP", "x-auth-app"),
    ("auth_session", "FREEMATICA_AUTH_SESSION", "x-auth-session"),
)

RANGO_NO_SOPORTADO = (
    "The Freemática API cannot combine a from-date and a to-date on the same field "
    "in this endpoint (it returns 0 rows). Use only one bound per query."
)

_FIQL_OPS = {
    "eq": "==",
    "ne": "!=",
    "gt": "=gt=",
    "lt": "=lt=",
    "ge": "=ge=",
    "le": "=le=",
    "in": "=in=",
}
_FIQL_ESCAPE = str.maketrans(
    {";": "%3B", ",": "%2C", "(": "%28", ")": "%29", '"': "%22", "'": "%27", "=": "%3D", "!": "%21"}
)

MasterDataCatalog = Literal[
    "tipos-contrato",
    "tipo-instalacion",
    "clases-servicios",
    "tipos-casos",
    "subtipos-casos",
    "tipos-oportunidad-negocio",
    "tipos-impuestos",
    "naturalezas-abono",
    "incidencecode",
    "claves-facturacion",
    "paises",
    "nacionalidades",
    "provincias",
    "poblaciones",
    "empresas",
    "delegaciones",
    "lineas-negocio",
    "cargos-clientes",
    "calendarios",
    "series",
    "familias",
    "subfamilias",
    "lineas",
    "bancos",
]

_CATALOG_ENDPOINTS: dict[str, str] = {
    "tipos-contrato": "/ppre/v2/tipos-contrato",
    "tipo-instalacion": "/ppre/v1/tipo-instalacion",
    "clases-servicios": "/pvss/v1/clases-servicios",
    "tipos-casos": "/pcrm/v2/tipos-casos",
    "subtipos-casos": "/pcrm/v2/subtipos-casos",
    "tipos-oportunidad-negocio": "/pcrm/v2/tipos-oportunidad-negocio",
    "tipos-impuestos": "/pgrl/v2/tipos-impuestos",
    "naturalezas-abono": "/pven/v1/naturalezas-abono",
    "incidencecode": "/pvss/v2/incidencecode",
    "claves-facturacion": "/pvss/v2/claves-facturacion",
    "paises": "/pgrl/v1/paises",
    "nacionalidades": "/pgrl/v1/nacionalidades",
    "provincias": "/pgrl/v1/provincias",
    "poblaciones": "/pgrl/v2/poblaciones",
    "empresas": "/pgrl/v1/empresas",
    "delegaciones": "/pgrl/v1/delegaciones/agrupcod",
    "lineas-negocio": "/pgrl/v2/lineas-negocio",
    "cargos-clientes": "/pgrl/v2/cargos-clientes",
    "calendarios": "/pgrl/v1/calendarios",
    "series": "/pgrl/v2/series",
    "familias": "/part/v1/familias",
    "subfamilias": "/part/v1/subfamilias",
    "lineas": "/part/v1/lineas",
    "bancos": "/pgrl/v2/bancos",
}

_LOCATIONS: dict[str, dict[str, Any]] = {
    "cobro": {
        "path": "/pgrl/v2/localizaciones-cobro-clientes",
        "get_path": "/pgrl/v2/localizaciones-cobro-clientes",
        "cod": "LOC_CLI_COB",
        "nombre": "NOM_LOC_COB",
        "domicilio": "DOMICILIO",
        "contacto": "COB_CONTACTO",
        "nombre_required": True,
        "forma_pago": "COD_FORMA_COBRO",
    },
    "envio": {
        "path": "/pgrl/v2/localizaciones-envio-clientes",
        "get_path": "/pgrl/v2/localizaciones-envio-clientes",
        "cod": "LOC_CLI_ENV",
        "nombre": "NOM_LOC_ENV",
        "domicilio": "DOMICILIO_ENVIO",
        "contacto": "ENV_CONTACTO",
        "nombre_required": False,
        "forma_pago": None,
    },
    "factura": {
        "path": "/pgrl/v2/localizaciones-factura-clientes",
        "get_path": "/pgrl/v1/localizaciones-factura-clientes",
        "cod": "LOC_CLI_FAC",
        "nombre": "NOM_LOC_FAC",
        "domicilio": "DOMICILIO",
        "contacto": "FAC_CONTACTO",
        "nombre_required": True,
        "forma_pago": None,
    },
    "servicio": {
        "path": "/pgrl/v2/localizaciones-servicio-clientes",
        "get_path": "/pgrl/v2/localizaciones-servicio-clientes",
        "cod": "LOC_CLI_SERV",
        "nombre": "DESCRIPCION",
        "domicilio": "DOMICILIO",
        "contacto": "CONTACTO",
        "nombre_required": True,
        "forma_pago": None,
    },
}

LocalizacionTipo = Literal["cobro", "envio", "factura", "servicio"]


class _RetryableFreematicaError(Exception):
    """Transient upstream failure. ``retry_after`` is seconds from the Retry-After header."""

    def __init__(self, code: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


def _secret(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    text = str(value).strip()
    return text or None


def _quote_fiql(value: Any) -> str:
    """Quote a FIQL scalar. Spaces stay literal; reserved characters are percent-encoded."""
    return "'" + str(value).translate(_FIQL_ESCAPE) + "'"


def build_fiql(filters: dict[str, Any]) -> str:
    """Build an AND-joined FIQL string. ``None`` values are skipped.

    A value may be a scalar (``==``) or ``{"op": ..., "value": ...}`` with
    ``op`` in eq, ne, gt, lt, ge, le, in. Empty input returns ``""``.
    """
    parts: list[str] = []
    for key, raw in filters.items():
        if raw is None:
            continue
        if isinstance(raw, dict) and "op" in raw:
            op = _FIQL_OPS[str(raw["op"])]
            value = raw["value"]
            if raw["op"] == "in":
                inner = ",".join(_quote_fiql(item) for item in value)
                parts.append(f"{key}{op}({inner})")
            else:
                parts.append(f"{key}{op}{_quote_fiql(value)}")
        else:
            parts.append(f"{key}=={_quote_fiql(raw)}")
    return ";".join(parts)


def next_day_iso(iso_date: str) -> str:
    """Return the next UTC calendar day. Rejects non-dates such as 2026-02-30."""
    from datetime import datetime, timedelta

    try:
        parsed = datetime.strptime(iso_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date: {iso_date}. Use a real calendar date as YYYY-MM-DD.") from exc
    if parsed.strftime("%Y-%m-%d") != iso_date:
        raise ValueError(f"Invalid date: {iso_date}. Use a real calendar date as YYYY-MM-DD.")
    return (parsed + timedelta(days=1)).strftime("%Y-%m-%d")


def _decode_idreg(idreg: str, expected: int, label: str) -> list[str]:
    try:
        decoded = base64.b64decode(idreg, validate=False).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Invalid {label} idReg (not base64): {idreg}") from exc
    parts = decoded.split("__")
    if len(parts) != expected or any(not part for part in parts):
        raise ValueError(f"Invalid {label} idReg: expected base64 of {expected} '__'-separated parts, got {decoded!r}.")
    return parts


def decode_contrato_idreg(idreg: str) -> dict[str, str]:
    """Decode a contract idReg (base64 of ``EMP__DELEG__COD``)."""
    empresa, delegacion, cod = _decode_idreg(idreg, 3, "contract")
    return {"empresa": empresa, "delegacion": delegacion, "cod_contrato": cod}


def decode_servicio_idreg(idreg: str) -> dict[str, str]:
    """Decode a service idReg (base64 of ``EMP__DELEG__CTRT__COD``)."""
    empresa, delegacion, contrato, servicio = _decode_idreg(idreg, 4, "service")
    return {
        "empresa": empresa,
        "delegacion": delegacion,
        "cod_contrato": contrato,
        "cod_servicio": servicio,
    }


def next_prefix(prefix: str) -> str | None:
    """Next lexical prefix so ``>= prefix AND < next`` matches the prefix. None if it overflows."""
    chars = list(prefix)
    index = len(chars) - 1
    while index >= 0:
        if chars[index] == "9":
            chars[index] = "0"
            index -= 1
            continue
        chars[index] = chr(ord(chars[index]) + 1)
        return "".join(chars)
    return None


def cliente_idreg(grupo: str, cod_cliente: str) -> str:
    """idReg Freemática derives for a new client: base64 of ``GRUPO__COD``."""
    return base64.b64encode(f"{grupo}__{cod_cliente}".encode()).decode("ascii")


def _settings_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


async def resolve_freematica_auth(tool: Any) -> tuple[str, dict[str, str]]:
    """Return ``(base_url, x-auth headers)`` for a Freemática tool."""
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    headers: dict[str, str] = {}
    missing: list[str] = []
    env_vars: list[str] = []
    for field, env_name, header in _AUTH_FIELDS:
        value = _secret(creds.get(field)) or _secret(getattr(tool, field, None)) or _secret(os.getenv(env_name))
        if not value:
            missing.append(field)
            env_vars.append(env_name)
        else:
            headers[header] = value
    if missing:
        raise CredentialNotAvailable("Freematica", missing=missing, env_vars=env_vars)

    base_url = (
        _secret(creds.get("base_url"))
        or _secret(getattr(tool, "base_url", None))
        or _secret(os.getenv("FREEMATICA_BASE_URL"))
        or _DEFAULT_BASE_URL
    )
    return base_url.rstrip("/"), headers


def _error(code: str, message: str) -> ValueError:
    return ValueError(f"{code}: {message}")


def _map_status(status: int, retry_after: str | None) -> _RetryableFreematicaError | ValueError:
    if status == 401:
        return _error("invalid_token", "Authentication failed (401). The x-auth-* credentials are invalid or expired.")
    if status == 403:
        return _error("forbidden", "Insufficient permissions (403).")
    if status == 404:
        return _error("not_found", "Resource not found (404).")
    if status == 429:
        seconds = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
        return _RetryableFreematicaError("rate_limit_exceeded", "Rate limit exceeded (429).", seconds)
    if status >= 500:
        return _RetryableFreematicaError("server_error", f"Upstream server error ({status}).")
    return _error("unexpected_error", f"Freemática returned HTTP {status}.")


def _map_envelope(error_code: str, error_message: str) -> ValueError | _RetryableFreematicaError:
    message = error_message or f"API error (errorCode={error_code})"
    if error_code == "401":
        return _error("invalid_token", message)
    if error_code == "403":
        return _error("forbidden", message)
    if error_code == "404":
        return _error("not_found", message)
    if error_code == "429":
        return _RetryableFreematicaError("rate_limit_exceeded", message)
    if error_code.startswith("5"):
        return _RetryableFreematicaError("server_error", message)
    return _error("unexpected_error", message)


def _normalize(data: Any) -> Any:
    if isinstance(data, dict) and "items" in data and "total" in data:
        items = data.get("items") or []
        try:
            total = int(data["total"])
        except (TypeError, ValueError):
            total = 0
        return {"items": items, "total": total, "count": len(items)}
    return data


def _retry_delay(error: _RetryableFreematicaError, attempt: int) -> float:
    if error.retry_after and error.retry_after > 0:
        return error.retry_after
    base = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
    return base + random.random() * _MAX_JITTER_SECONDS


async def _freematica_request(
    tool: Any,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: Any = None,
) -> Any:
    """Call one Freemática endpoint and return the unwrapped ``data``."""
    import httpx

    base_url, auth_headers = await resolve_freematica_auth(tool)
    timeout = _settings_int("FREEMATICA_TIMEOUT_MS", 30_000) / 1000
    max_retries = _settings_int("FREEMATICA_MAX_RETRIES", 3)
    headers = {**auth_headers, "Content-Type": "application/json", "Accept": "application/json"}
    clean_params = {key: value for key, value in (params or {}).items() if value is not None} or None
    url = f"{base_url}{path}"
    last: _RetryableFreematicaError | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        for attempt in range(max_retries + 1):
            if attempt and last is not None:
                await asyncio.sleep(_retry_delay(last, attempt - 1))
            try:
                response = await client.request(method, url, headers=headers, params=clean_params, json=body)
            except httpx.TimeoutException as exc:
                last = _RetryableFreematicaError("network_error", f"Request timed out: {exc}")
                continue
            except httpx.RequestError as exc:
                last = _RetryableFreematicaError("network_error", f"Network error: {exc}")
                continue

            if response.status_code >= 400:
                mapped = _map_status(response.status_code, response.headers.get("Retry-After"))
                if isinstance(mapped, _RetryableFreematicaError):
                    last = mapped
                    continue
                raise mapped

            try:
                payload = response.json()
            except ValueError as exc:
                raise _error("unexpected_error", f"Freemática returned a non-JSON body: {exc}") from exc
            if not isinstance(payload, dict) or "errorCode" not in payload:
                return payload
            if str(payload.get("errorCode")) != "200":
                mapped = _map_envelope(str(payload.get("errorCode")), str(payload.get("errorMessage") or ""))
                if isinstance(mapped, _RetryableFreematicaError):
                    last = mapped
                    continue
                raise mapped
            return _normalize(payload.get("data"))

    assert last is not None
    raise _error(last.code, str(last))


def _first_item(result: Any, what: str, idreg: str) -> dict[str, Any]:
    items = result.get("items") if isinstance(result, dict) else None
    if not items:
        raise _error("not_found", f"{what} not found: {idreg}")
    return items[0]


def _guard_size(result: dict[str, Any]) -> dict[str, Any]:
    """Truncate ``export_asientos`` when the JSON exceeds FREEMATICA_MAX_RESPONSE_SIZE_MB."""
    import json

    max_bytes = _settings_int("FREEMATICA_MAX_RESPONSE_SIZE_MB", 10) * 1024 * 1024
    items = list(result.get("items") or [])
    if len(json.dumps(items).encode()) <= max_bytes:
        return {**result, "truncated": False}
    kept: list[Any] = []
    for item in items:
        trial = kept + [item]
        if len(json.dumps(trial).encode()) > max_bytes:
            break
        kept = trial
    return {
        **result,
        "items": kept,
        "count": len(kept),
        "truncated": True,
        "warning": (
            f"Response truncated to {len(kept)} of {len(items)} items because it exceeded "
            f"{max_bytes // (1024 * 1024)} MB. Use a shorter date range."
        ),
    }


def _page() -> Any:
    return Field(
        1,
        ge=1,
        le=999,
        description="Page to fetch, 1-indexed. page=0 is rejected: the API treats it as the full dataset.",
    )


def _items() -> Any:
    return Field(20, ge=1, le=50, description="Items per page (1-50, default 20).")


def _list_params(page: int, items: int, fiql: str = "", **native: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"page": page, "items": items, **native}
    if fiql:
        params["rquery"] = fiql
    return params


class _FreematicaTool(Tool):
    """Shared credential fields for every Freemática tool."""

    integration: Annotated[str, Integration("freematica")] | None = None
    auth_token: SecretStr | None = None
    auth_company: str | None = None
    auth_organization: str | None = None
    auth_app: str | None = None
    auth_session: str | None = None
    base_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "auth_token": self.auth_token,
                    "auth_company": self.auth_company,
                    "auth_organization": self.auth_organization,
                    "auth_app": self.auth_app,
                    "auth_session": self.auth_session,
                    "base_url": self.base_url,
                }
            ),
        }


class FreematicaGetMasterData(_FreematicaTool):
    """Return one of the 24 master-data catalogs (types, geography, org, inventory, banks)."""

    name: str = "freematica_get_master_data"
    description: str | None = (
        "Return a Freemática master-data catalog so cryptic codes from other tools can be resolved "
        "to names. catalog is one of: tipos-contrato, tipo-instalacion, clases-servicios, tipos-casos, "
        "subtipos-casos, tipos-oportunidad-negocio, tipos-impuestos, naturalezas-abono, incidencecode, "
        "claves-facturacion, paises, nacionalidades, provincias, poblaciones, empresas, delegaciones, "
        "lineas-negocio, cargos-clientes, calendarios, series, familias, subfamilias, lineas, bancos."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_master_data(
            catalog: MasterDataCatalog = Field(..., description="Which master-data catalog to load."),
        ) -> Any:
            result = await _freematica_request(self, "GET", _CATALOG_ENDPOINTS[catalog])
            if isinstance(result, dict):
                result = {**result, "catalog": catalog}
            return result

        super().__init__(handler=_get_master_data, **kwargs)


class FreematicaListClientes(_FreematicaTool):
    """Paginated client list. Each item includes the opaque idReg used by get_cliente."""

    name: str = "freematica_list_clientes"
    description: str | None = (
        "List Freemática clients (COD_CLI, NOMBRE_CLI, NIF, ...). Use the opaque idReg field, "
        "not COD_CLI, with freematica_get_cliente."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_clientes(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(self, "GET", "/pgrl/v2/clientes", params=_list_params(page, items))

        super().__init__(handler=_list_clientes, **kwargs)


class FreematicaGetCliente(_FreematicaTool):
    """Client detail. idreg is the opaque idReg from freematica_list_clientes."""

    name: str = "freematica_get_cliente"
    description: str | None = (
        "Get one client. idreg MUST be the opaque base64 idReg from freematica_list_clientes. "
        "A natural code such as COD_CLI returns not_found."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_cliente(
            idreg: str = Field(..., description="Opaque idReg from freematica_list_clientes."),
        ) -> Any:
            return await _freematica_request(self, "GET", f"/pgrl/v2/clientes/{quote(idreg, safe='')}")

        super().__init__(handler=_get_cliente, **kwargs)


class FreematicaCreateCliente(_FreematicaTool):
    """Create a client. idReg is derived as base64 of GRUPO__COD."""

    name: str = "freematica_create_cliente"
    description: str | None = (
        "Create a Freemática client (POST /pgrl/v2/clientes). Required: grupo_cliente, cod_cliente, "
        "nombre, nif, tipo_impuesto, divisa, tipo_facturacion (D daily, M monthly, Q biweekly). "
        "idReg is computed as base64 of GRUPO__COD. Extra native columns go in campos_adicionales."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_cliente(
            grupo_cliente: str = Field(..., description="Client group (COD_GRUPO_CLI)."),
            cod_cliente: str = Field(..., description="Client code (COD_CLI)."),
            nombre: str = Field(..., description="Client name (NOMBRE_CLI, 40 chars)."),
            nif: str = Field(..., description="Tax id (NIF)."),
            tipo_impuesto: str = Field(..., description="Tax type (TIPO_IMPTO)."),
            divisa: str = Field(..., description="Currency code, e.g. EUR (COD_DIVISA)."),
            tipo_facturacion: Literal["D", "M", "Q"] = Field(
                ..., description="Billing cadence: D daily, M monthly, Q biweekly (TIPO_FACT)."
            ),
            campos_adicionales: dict[str, Any] | None = Field(
                None, description="Extra native VoClientes columns, e.g. COD_TARIFA or CREDITO."
            ),
        ) -> Any:
            body = _drop_none(
                {
                    "idReg": cliente_idreg(grupo_cliente, cod_cliente),
                    "COD_GRUPO_CLI": grupo_cliente,
                    "COD_CLI": cod_cliente,
                    "NOMBRE_CLI": nombre,
                    "NIF": nif,
                    "TIPO_IMPTO": tipo_impuesto,
                    "COD_DIVISA": divisa,
                    "TIPO_FACT": tipo_facturacion,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(self, "POST", "/pgrl/v2/clientes", body=body)

        super().__init__(handler=_create_cliente, **kwargs)


class FreematicaUpdateCliente(_FreematicaTool):
    """Partial client update: fetch the current record, merge, then PUT the full object."""

    name: str = "freematica_update_cliente"
    description: str | None = (
        "Partially update a client. The current record is fetched and merged before PUT, "
        "because the API requires the mandatory fields on every update. idreg is the opaque idReg."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_cliente(
            idreg: str = Field(..., description="Opaque idReg of the client."),
            nombre: str | None = Field(None, description="New NOMBRE_CLI."),
            nif: str | None = Field(None, description="New NIF."),
            tipo_impuesto: str | None = Field(None, description="New TIPO_IMPTO."),
            divisa: str | None = Field(None, description="New COD_DIVISA."),
            tipo_facturacion: Literal["D", "M", "Q"] | None = Field(None, description="New TIPO_FACT."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra native columns to merge."),
        ) -> Any:
            path = f"/pgrl/v2/clientes/{quote(idreg, safe='')}"
            current = await _freematica_request(self, "GET", path)
            if not isinstance(current, dict):
                raise _error("unexpected_error", "Client detail was not an object.")
            changes = _drop_none(
                {
                    "NOMBRE_CLI": nombre,
                    "NIF": nif,
                    "TIPO_IMPTO": tipo_impuesto,
                    "COD_DIVISA": divisa,
                    "TIPO_FACT": tipo_facturacion,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(self, "PUT", path, body={**current, **changes})

        super().__init__(handler=_update_cliente, **kwargs)


class FreematicaListContactosClientes(_FreematicaTool):
    """Paginated client contacts."""

    name: str = "freematica_list_contactos_clientes"
    description: str | None = "List client contacts. Each item includes an opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_contactos(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/pgrl/v2/contactos-clientes", params=_list_params(page, items)
            )

        super().__init__(handler=_list_contactos, **kwargs)


class FreematicaCreateContactoCliente(_FreematicaTool):
    """Create a client contact. grupo_cliente and cod_cliente are required."""

    name: str = "freematica_create_contacto_cliente"
    description: str | None = (
        "Create a client contact (POST /pgrl/v2/contactos-clientes). Required: grupo_cliente and cod_cliente."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_contacto(
            grupo_cliente: str = Field(..., description="Client group (CC_GRUPO_CLI)."),
            cod_cliente: str = Field(..., description="Client code (CC_CLI)."),
            nombre_apellidos: str | None = Field(None, description="Contact name."),
            cargo: str | None = Field(None, description="Job title."),
            telefono: str | None = Field(None, description="Phone."),
            movil: str | None = Field(None, description="Mobile."),
            email: str | None = Field(None, description="Email."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra native CC_* columns."),
        ) -> Any:
            body = _drop_none(
                {
                    "CC_GRUPO_CLI": grupo_cliente,
                    "CC_CLI": cod_cliente,
                    "CC_NOM": nombre_apellidos,
                    "CC_CARGO": cargo,
                    "CC_TELEFONO": telefono,
                    "CC_MOVIL": movil,
                    "CC_EMAIL": email,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(self, "POST", "/pgrl/v2/contactos-clientes", body=body)

        super().__init__(handler=_create_contacto, **kwargs)


class FreematicaUpdateContactoCliente(_FreematicaTool):
    """Partial contact update via fetch (v1 detail) + merge + PUT."""

    name: str = "freematica_update_contacto_cliente"
    description: str | None = (
        "Partially update a client contact. The v1 detail is fetched and merged before PUT. "
        "idreg comes from freematica_list_contactos_clientes."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_contacto(
            idreg: str = Field(..., description="Opaque idReg of the contact."),
            nombre_apellidos: str | None = Field(None, description="Contact name."),
            cargo: str | None = Field(None, description="Job title."),
            telefono: str | None = Field(None, description="Phone."),
            movil: str | None = Field(None, description="Mobile."),
            email: str | None = Field(None, description="Email."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra native columns."),
        ) -> Any:
            current = await _freematica_request(self, "GET", f"/pgrl/v1/contactos-clientes/{quote(idreg, safe='')}")
            if not isinstance(current, dict):
                raise _error("unexpected_error", "Contact detail was not an object.")
            changes = _drop_none(
                {
                    "CC_NOM": nombre_apellidos,
                    "CC_CARGO": cargo,
                    "CC_TELEFONO": telefono,
                    "CC_MOVIL": movil,
                    "CC_EMAIL": email,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(
                self, "PUT", f"/pgrl/v2/contactos-clientes/{quote(idreg, safe='')}", body={**current, **changes}
            )

        super().__init__(handler=_update_contacto, **kwargs)


class FreematicaListOportunidadesNegocio(_FreematicaTool):
    """Paginated CRM opportunities."""

    name: str = "freematica_list_oportunidades_negocio"
    description: str | None = "List business opportunities. Use idReg with the get tools, not the natural id."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_oportunidades(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/pcrm/v2/oportunidades-negocio", params=_list_params(page, items)
            )

        super().__init__(handler=_list_oportunidades, **kwargs)


class FreematicaGetOportunidadNegocio(_FreematicaTool):
    """Opportunity detail by opaque idReg."""

    name: str = "freematica_get_oportunidad_negocio"
    description: str | None = "Get one opportunity. idreg is the opaque idReg from the list."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_oportunidad(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pcrm/v2/oportunidades-negocio/{quote(idreg, safe='')}")

        super().__init__(handler=_get_oportunidad, **kwargs)


class FreematicaGetOportunidadNegocioDatosAmpliados(_FreematicaTool):
    """Extended opportunity data. not_found is normal when there is no extra data."""

    name: str = "freematica_get_oportunidad_negocio_datos_ampliados"
    description: str | None = (
        "Extended data for an opportunity. not_found is a normal outcome when the opportunity has none."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_datos(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(
                self, "GET", f"/pcrm/v2/oportunidades-negocio/{quote(idreg, safe='')}/datos-ampliados"
            )

        super().__init__(handler=_get_datos, **kwargs)


def _location_body(tipo: str, values: dict[str, Any]) -> dict[str, Any]:
    spec = _LOCATIONS[tipo]
    body = _drop_none(
        {
            spec["cod"]: values.get("cod_localizacion"),
            spec["nombre"]: values.get("nombre"),
            spec["domicilio"]: values.get("domicilio"),
            spec["contacto"]: values.get("contacto"),
            "COD_GRUPO_CLI": values.get("grupo_cliente"),
            "COD_CLI": values.get("cod_cliente"),
            "COD_PAIS": values.get("cod_pais"),
            "COD_PROVINCIA": values.get("cod_provincia"),
            "COD_POBLACION": values.get("cod_poblacion"),
            "COD_POSTAL": values.get("cod_postal"),
        }
    )
    if spec["forma_pago"] and values.get("forma_pago") is not None:
        body[spec["forma_pago"]] = values["forma_pago"]
    body.update(values.get("campos_adicionales") or {})
    return body


class FreematicaListLocalizacionesCobroClientes(_FreematicaTool):
    """Client collection locations (SEPA, direct debit)."""

    name: str = "freematica_list_localizaciones_cobro_clientes"
    description: str | None = "List client collection locations (SEPA / direct debit)."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(self, "GET", _LOCATIONS["cobro"]["path"], params=_list_params(page, items))

        super().__init__(handler=_list, **kwargs)


class FreematicaListLocalizacionesEnvioClientes(_FreematicaTool):
    """Client shipping locations."""

    name: str = "freematica_list_localizaciones_envio_clientes"
    description: str | None = "List client shipping locations."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(self, "GET", _LOCATIONS["envio"]["path"], params=_list_params(page, items))

        super().__init__(handler=_list, **kwargs)


class FreematicaListLocalizacionesFacturaClientes(_FreematicaTool):
    """Client invoice locations."""

    name: str = "freematica_list_localizaciones_factura_clientes"
    description: str | None = "List client invoice locations."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            # The list is v1. v2 exists for create/update only (Postman + the verified client).
            return await _freematica_request(
                self, "GET", _LOCATIONS["factura"]["get_path"], params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaListLocalizacionesServicioClientes(_FreematicaTool):
    """Client service locations."""

    name: str = "freematica_list_localizaciones_servicio_clientes"
    description: str | None = "List client service locations."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", _LOCATIONS["servicio"]["path"], params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaListLocalizacionesPagoProveedores(_FreematicaTool):
    """Supplier payment locations."""

    name: str = "freematica_list_localizaciones_pago_proveedores"
    description: str | None = "List supplier payment locations."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/pgrl/v2/localizaciones-pago-proveedores", params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaCreateLocalizacionCliente(_FreematicaTool):
    """Create a client location of type cobro, envio, factura or servicio."""

    name: str = "freematica_create_localizacion_cliente"
    description: str | None = (
        "Create a client location. tipo is cobro, envio, factura or servicio. "
        "forma_pago is only sent for cobro. nombre is required except for envio."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            tipo: LocalizacionTipo = Field(..., description="cobro, envio, factura or servicio."),
            grupo_cliente: str = Field(..., description="Client group."),
            cod_cliente: str = Field(..., description="Client code."),
            cod_localizacion: str = Field(..., description="Location code."),
            nombre: str | None = Field(None, description="Location name. Required except for envio."),
            domicilio: str | None = Field(None, description="Address."),
            contacto: str | None = Field(None, description="Contact person."),
            forma_pago: str | None = Field(None, description="Payment form. Only used for cobro."),
            cod_pais: int | None = Field(None, description="Country code."),
            cod_provincia: str | None = Field(None, description="Province code."),
            cod_poblacion: str | None = Field(None, description="Town code."),
            cod_postal: str | None = Field(None, description="Postal code."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra native columns."),
        ) -> Any:
            spec = _LOCATIONS[tipo]
            if spec["nombre_required"] and not nombre:
                raise ValueError(f"nombre is required for a {tipo} location.")
            body = _location_body(tipo, locals())
            return await _freematica_request(self, "POST", spec["path"], body=body)

        super().__init__(handler=_create, **kwargs)


class FreematicaUpdateLocalizacionCliente(_FreematicaTool):
    """Partial location update: fetch (v1 for factura) + merge + PUT."""

    name: str = "freematica_update_localizacion_cliente"
    description: str | None = (
        "Partially update a client location. The current record is fetched and merged before PUT. "
        "idreg comes from the list tool of the same tipo."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            tipo: LocalizacionTipo = Field(..., description="cobro, envio, factura or servicio."),
            idreg: str = Field(..., description="Opaque idReg of the location."),
            nombre: str | None = Field(None, description="Location name."),
            domicilio: str | None = Field(None, description="Address."),
            contacto: str | None = Field(None, description="Contact person."),
            forma_pago: str | None = Field(None, description="Payment form. Only used for cobro."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra native columns."),
        ) -> Any:
            spec = _LOCATIONS[tipo]
            current = await _freematica_request(self, "GET", f"{spec['get_path']}/{quote(idreg, safe='')}")
            if not isinstance(current, dict):
                raise _error("unexpected_error", "Location detail was not an object.")
            changes = _location_body(
                tipo,
                {
                    "nombre": nombre,
                    "domicilio": domicilio,
                    "contacto": contacto,
                    "forma_pago": forma_pago,
                    "campos_adicionales": campos_adicionales,
                },
            )
            return await _freematica_request(
                self, "PUT", f"{spec['path']}/{quote(idreg, safe='')}", body={**current, **changes}
            )

        super().__init__(handler=_update, **kwargs)
