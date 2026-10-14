"""Freemática tools: auth, FIQL, envelope/errors, retries, and one route per tool.

Live calls are marked ``integration`` and skip unless ``FREEMATICA_AUTH_*`` is set.
"""

from __future__ import annotations

import asyncio
import base64
import random
from typing import Any, Self
from urllib.parse import quote

import httpx
import pytest
from pydantic import SecretStr
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.tools import freematica as fm
from timbal.tools.freematica import (
    RANGO_NO_SOPORTADO,
    _freematica_request,
    _guard_size,
    build_fiql,
    cliente_idreg,
    decode_contrato_idreg,
    decode_servicio_idreg,
    next_day_iso,
    next_prefix,
    resolve_freematica_auth,
)
from timbal.tools.freematica_contracts import (
    FreematicaCreateContrato,
    FreematicaCreateContratoOpcionales,
    FreematicaCreateServicioContrato,
    FreematicaCreateServicioFacturacionTxt,
    FreematicaCreateServicioHistoricoPrecios,
    FreematicaGetContrato,
    FreematicaGetContratoOpcionales,
    FreematicaGetServicioContrato,
    FreematicaListContratos,
    FreematicaListContratosOpcionales,
    FreematicaListResultadosFacturacion,
    FreematicaListServiciosContrato,
    FreematicaUpdateContrato,
    FreematicaUpdateContratoOpcionales,
    FreematicaUpdateServicioFacturacion,
    FreematicaUpdateServicioFechas,
    FreematicaUpdateServicioHistoricoPrecios,
)
from timbal.tools.freematica_people import (
    FreematicaGetFichaPrevCliente,
    FreematicaGetPersona,
    FreematicaGetVigilanciaSalud,
    FreematicaListCalendarioPeriodos,
    FreematicaListCalendarios,
    FreematicaListPersonal,
    FreematicaListVigilanciaSalud,
)
from timbal.tools.freematica_purchasing import (
    FreematicaExportAsientos,
    FreematicaGetArticulo,
    FreematicaGetFacturaCompra,
    FreematicaGetPedidoCompra,
    FreematicaGetPrecioArticulo,
    FreematicaGetProveedor,
    FreematicaListArticulos,
    FreematicaListCuentasAnaliticas,
    FreematicaListCuentasContables,
    FreematicaListFacturasCompras,
    FreematicaListMaterialesAsignadosServicios,
    FreematicaListPedidosCompra,
    FreematicaListProveedores,
)
from timbal.tools.freematica_sales import (
    FreematicaGetAlbaranFactura,
    FreematicaGetAlbaranVenta,
    FreematicaGetCarteraCliente,
    FreematicaGetEdicomInfo,
    FreematicaGetFacturaCabecera,
    FreematicaGetFacturaDocumento,
    FreematicaGetFacturaElectronica,
    FreematicaGetFacturaLog,
    FreematicaListAlbaranesFactura,
    FreematicaListAlbaranesVentas,
    FreematicaListCarteraClientes,
    FreematicaListFacturaIva,
    FreematicaListFacturaLineas,
    FreematicaListFacturasCabecera,
    FreematicaListFacturasDocumentos,
    FreematicaListFacturasElectronicas,
    FreematicaListFacturaVencimientos,
)

BASE = "https://example.test/api"
ID = "abc"
CID = base64.b64encode(b"E1__D1__C1").decode()
PAGE = {"page": 1, "items": 20}
AUTH_ENV = {
    "FREEMATICA_AUTH_TOKEN": "tok",
    "FREEMATICA_AUTH_COMPANY": "co",
    "FREEMATICA_AUTH_ORGANIZATION": "org",
    "FREEMATICA_AUTH_APP": "app",
    "FREEMATICA_AUTH_SESSION": "ses",
    "FREEMATICA_BASE_URL": BASE,
}
AUTH_HEADERS = {
    "x-auth-token": "tok",
    "x-auth-company": "co",
    "x-auth-organization": "org",
    "x-auth-app": "app",
    "x-auth-session": "ses",
}

_LIST_ROW = {"CTRT_COD": "C1", "CTRT_EMP": "E1", "CTRT_DELEG": "D1", "ok": True, "NOMBRE_CLI": "old"}


def _qid(value: str = ID) -> str:
    return quote(value, safe="")


class _Resp:
    def __init__(self, status: int = 200, payload: Any = None, headers: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _ScriptedClient:
    """httpx stand-in. ``script`` is a list of responses or exceptions, last one repeats."""

    def __init__(self, script: list[Any], calls: list[dict[str, Any]]) -> None:
        self.script = script
        self.calls = calls
        self._i = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def request(
        self, method: str, url: str, headers: dict | None = None, params: Any = None, json: Any = None
    ) -> _Resp:
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "params": params, "json": json})
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return item


def _ok(data: Any | None = None) -> _Resp:
    if data is None:
        data = {"items": [_LIST_ROW], "total": "1"}
    return _Resp(200, {"errorCode": "200", "errorMessage": "", "data": data})


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def factory(*_a: Any, **_k: Any) -> _ScriptedClient:
        return _ScriptedClient([_ok()], calls)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    return calls


def _tool(**fields: Any) -> Any:
    class _T:
        integration = fields.get("integration")
        auth_token = fields.get("auth_token")
        auth_company = fields.get("auth_company")
        auth_organization = fields.get("auth_organization")
        auth_app = fields.get("auth_app")
        auth_session = fields.get("auth_session")
        base_url = fields.get("base_url")

    return _T()


def _filled(**overrides: Any) -> Any:
    values = {
        "auth_token": SecretStr("field-token"),
        "auth_company": "field-co",
        "auth_organization": "field-org",
        "auth_app": "field-app",
        "auth_session": "field-ses",
    }
    values.update(overrides)
    return _tool(**values)


# --- FIQL, dates, idReg -------------------------------------------------------


def test_fiql_quoting_escaping_and_in() -> None:
    assert build_fiql({}) == ""
    assert build_fiql({"A": None, "B": None}) == ""
    assert build_fiql({"NOM": "a b"}) == "NOM=='a b'"
    encoded = build_fiql({"X": "';DROP,=!"})
    assert encoded == "X=='%27%3BDROP%2C%3D%21'"
    assert ";" not in encoded.split("==", 1)[1]
    assert build_fiql({"A": {"op": "in", "value": ["1", "2"]}}) == "A=in=('1','2')"
    assert build_fiql({"A": {"op": "ge", "value": "2026-01-01"}, "B": "x"}) == "A=ge='2026-01-01';B=='x'"


def test_next_day_and_prefix() -> None:
    assert next_day_iso("2026-01-31") == "2026-02-01"
    with pytest.raises(ValueError, match="Invalid date"):
        next_day_iso("2026-02-30")
    assert next_prefix("43") == "44"
    assert next_prefix("19") == "20"
    assert next_prefix("99") is None


def test_idreg_helpers() -> None:
    encoded = cliente_idreg("G", "C")
    assert base64.b64decode(encoded).decode() == "G__C"
    assert decode_contrato_idreg(base64.b64encode(b"E__D__C").decode()) == {
        "empresa": "E",
        "delegacion": "D",
        "cod_contrato": "C",
    }
    assert decode_servicio_idreg(base64.b64encode(b"E__D__C__S").decode())["cod_servicio"] == "S"
    with pytest.raises(ValueError, match="not base64"):
        decode_contrato_idreg(base64.b64encode(b"\xff\xfe").decode())
    with pytest.raises(ValueError, match="3 '__'-separated"):
        decode_contrato_idreg(base64.b64encode(b"only-one").decode())


# --- auth ---------------------------------------------------------------------


async def test_auth_precedence_integration_over_fields_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    integration = Integration("freematica", "org-1")

    async def _resolve() -> dict[str, str]:
        return {
            "auth_token": "from-integration",
            "auth_company": "ic",
            "auth_organization": "io",
            "auth_app": "ia",
            "auth_session": "is",
            "base_url": "https://integration.example/api/",
        }

    integration.resolve = _resolve  # type: ignore[method-assign]
    base, headers = await resolve_freematica_auth(_filled(integration=integration))
    assert base == "https://integration.example/api"
    assert headers["x-auth-token"] == "from-integration"


async def test_auth_fields_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    base, headers = await resolve_freematica_auth(_filled())
    assert headers["x-auth-token"] == "field-token"
    assert base == BASE

    empty = _tool()
    base, headers = await resolve_freematica_auth(empty)
    assert headers == AUTH_HEADERS
    assert base == BASE


async def test_auth_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in AUTH_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(CredentialNotAvailable) as exc:
        await resolve_freematica_auth(_tool())
    assert exc.value.provider_name == "Freematica"
    assert "auth_token" in exc.value.missing


# --- envelope, errors, retries, size ------------------------------------------


def _patch_client(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def factory(*_a: Any, **_k: Any) -> _ScriptedClient:
        return _ScriptedClient(script, calls)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return calls


async def test_envelope_unwrap_and_list_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    _patch_client(monkeypatch, [_ok({"items": [{}, {}], "total": "9", "rowHeight": 1})])
    result = await _freematica_request(_tool(), "GET", "/pgrl/v2/clientes")
    assert result == {"items": [{}, {}], "total": 9, "count": 2}


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "invalid_token"), (403, "forbidden"), (404, "not_found"), (400, "unexpected_error")],
)
async def test_http_error_mapping(monkeypatch: pytest.MonkeyPatch, status: int, code: str) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    calls = _patch_client(monkeypatch, [_Resp(status, {"error": True})])
    with pytest.raises(ValueError, match=f"^{code}:"):
        await _freematica_request(_tool(), "GET", "/x")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("error_code", "code"),
    [("401", "invalid_token"), ("403", "forbidden"), ("404", "not_found"), ("418", "unexpected_error")],
)
async def test_envelope_error_mapping(monkeypatch: pytest.MonkeyPatch, error_code: str, code: str) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    _patch_client(monkeypatch, [_Resp(200, {"errorCode": error_code, "errorMessage": "nope", "data": None})])
    with pytest.raises(ValueError, match=f"^{code}:"):
        await _freematica_request(_tool(), "GET", "/x")


async def test_retry_backoff_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FREEMATICA_MAX_RETRIES", "3")
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    monkeypatch.setattr(random, "random", lambda: 0)
    calls = _patch_client(monkeypatch, [_Resp(500, {}), _Resp(502, {}), _ok({"hello": 1})])
    assert await _freematica_request(_tool(), "GET", "/x") == {"hello": 1}
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


async def test_retry_after_and_no_retry_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    calls = _patch_client(monkeypatch, [_Resp(429, {}, headers={"Retry-After": "7"}), _ok({"ok": True})])
    assert (await _freematica_request(_tool(), "GET", "/x"))["ok"] is True
    assert sleeps == [7.0]
    assert len(calls) == 2

    calls4 = _patch_client(monkeypatch, [_Resp(400, {})])
    with pytest.raises(ValueError, match="^unexpected_error:"):
        await _freematica_request(_tool(), "GET", "/x")
    assert len(calls4) == 1


async def test_network_error_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: _async_none())
    monkeypatch.setattr(random, "random", lambda: 0)
    calls = _patch_client(monkeypatch, [httpx.ConnectError("down"), _ok({"ok": True})])
    assert (await _freematica_request(_tool(), "GET", "/x"))["ok"] is True
    assert len(calls) == 2


async def _async_none() -> None:
    return None


def test_size_guard_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREEMATICA_MAX_RESPONSE_SIZE_MB", "0")
    result = _guard_size({"items": [{"a": "x" * 50}, {"b": "y"}], "total": 2, "count": 2})
    assert result["truncated"] is True
    assert result["count"] == 0
    assert "warning" in result


# --- routes -------------------------------------------------------------------

_CLIENT = {
    "grupo_cliente": "G",
    "cod_cliente": "C",
    "nombre": "N",
    "nif": "NIF",
    "tipo_impuesto": "IVA",
    "divisa": "EUR",
    "tipo_facturacion": "M",
}


def _routes() -> list[tuple[Any, dict[str, Any], list[tuple[str, str, dict | None]]]]:
    """(tool class, kwargs, [(method, path, param subset or None)])."""
    qid = _qid()
    qcid = _qid(CID)
    page = PAGE
    return [
        (fm.FreematicaGetMasterData, {"catalog": "empresas"}, [("GET", "/pgrl/v1/empresas", None)]),
        (fm.FreematicaGetMasterData, {"catalog": "delegaciones"}, [("GET", "/pgrl/v1/delegaciones/agrupcod", None)]),
        (fm.FreematicaListClientes, {}, [("GET", "/pgrl/v2/clientes", page)]),
        (fm.FreematicaGetCliente, {"idreg": ID}, [("GET", f"/pgrl/v2/clientes/{qid}", None)]),
        (fm.FreematicaCreateCliente, _CLIENT, [("POST", "/pgrl/v2/clientes", None)]),
        (
            fm.FreematicaUpdateCliente,
            {"idreg": ID, "nombre": "New"},
            [("GET", f"/pgrl/v2/clientes/{qid}", None), ("PUT", f"/pgrl/v2/clientes/{qid}", None)],
        ),
        (fm.FreematicaListContactosClientes, {}, [("GET", "/pgrl/v2/contactos-clientes", page)]),
        (
            fm.FreematicaCreateContactoCliente,
            {"grupo_cliente": "G", "cod_cliente": "C"},
            [("POST", "/pgrl/v2/contactos-clientes", None)],
        ),
        (
            fm.FreematicaUpdateContactoCliente,
            {"idreg": ID, "nombre_apellidos": "Ann"},
            [("GET", f"/pgrl/v1/contactos-clientes/{qid}", None), ("PUT", f"/pgrl/v2/contactos-clientes/{qid}", None)],
        ),
        (fm.FreematicaListOportunidadesNegocio, {}, [("GET", "/pcrm/v2/oportunidades-negocio", page)]),
        (fm.FreematicaGetOportunidadNegocio, {"idreg": ID}, [("GET", f"/pcrm/v2/oportunidades-negocio/{qid}", None)]),
        (
            fm.FreematicaGetOportunidadNegocioDatosAmpliados,
            {"idreg": ID},
            [("GET", f"/pcrm/v2/oportunidades-negocio/{qid}/datos-ampliados", None)],
        ),
        (fm.FreematicaListLocalizacionesCobroClientes, {}, [("GET", "/pgrl/v2/localizaciones-cobro-clientes", page)]),
        (fm.FreematicaListLocalizacionesEnvioClientes, {}, [("GET", "/pgrl/v2/localizaciones-envio-clientes", page)]),
        (
            fm.FreematicaListLocalizacionesFacturaClientes,
            {},
            [("GET", "/pgrl/v1/localizaciones-factura-clientes", page)],
        ),
        (
            fm.FreematicaListLocalizacionesServicioClientes,
            {},
            [("GET", "/pgrl/v2/localizaciones-servicio-clientes", page)],
        ),
        (
            fm.FreematicaListLocalizacionesPagoProveedores,
            {},
            [("GET", "/pgrl/v2/localizaciones-pago-proveedores", page)],
        ),
        (
            fm.FreematicaCreateLocalizacionCliente,
            {
                "tipo": "cobro",
                "grupo_cliente": "G",
                "cod_cliente": "C",
                "cod_localizacion": "L",
                "nombre": "N",
                "forma_pago": "TR",
            },
            [("POST", "/pgrl/v2/localizaciones-cobro-clientes", None)],
        ),
        (
            fm.FreematicaUpdateLocalizacionCliente,
            {"tipo": "factura", "idreg": ID, "nombre": "N"},
            [
                ("GET", f"/pgrl/v1/localizaciones-factura-clientes/{qid}", None),
                ("PUT", f"/pgrl/v2/localizaciones-factura-clientes/{qid}", None),
            ],
        ),
        (FreematicaListPersonal, {}, [("GET", "/pers/v1/personal", page)]),
        (FreematicaGetPersona, {"idreg": ID}, [("GET", f"/pers/v1/personal/{qid}", None)]),
        (FreematicaListCalendarios, {}, [("GET", "/pgrl/v1/calendarios", page)]),
        (FreematicaListCalendarioPeriodos, {"idreg": ID}, [("GET", f"/pgrl/v1/calendarios/{qid}/periodos", page)]),
        (
            FreematicaGetFichaPrevCliente,
            {"cod_cliente": "C"},
            [("GET", "/pprl/v2/ficha-prev-cliente", {"codCli": "C"})],
        ),
        (
            FreematicaListVigilanciaSalud,
            {"idreg_persona": ID},
            [("GET", "/pprl/v1/vigilancia-salud", {**page, "idRegPersona": ID})],
        ),
        (FreematicaGetVigilanciaSalud, {"idreg": ID}, [("GET", f"/pprl/v1/vigilancia-salud/{qid}", None)]),
        (
            FreematicaListProveedores,
            {"cod_proveedor": "P"},
            [("GET", "/pgrl/v2/proveedores", {**page, "rquery": "COD_PRO=='P'"})],
        ),
        (FreematicaGetProveedor, {"idreg": ID}, [("GET", f"/pgrl/v2/proveedores/{qid}", None)]),
        (
            FreematicaListFacturasCompras,
            {"exportado": "not_exported"},
            [("GET", "/pcmp/v2/facturas-compras", {**page, "exportado": "not_exported"})],
        ),
        (FreematicaGetFacturaCompra, {"idreg": ID}, [("GET", f"/pcmp/v2/facturas-compras/{qid}", None)]),
        (
            FreematicaListPedidosCompra,
            {"empresa": "E", "fecha_pedido_desde": "2026-01-01"},
            [("GET", "/pcmp/v2/pedidos", {**page, "codEmpresa": "E", "desdeFecha": "2026-01-01"})],
        ),
        (FreematicaGetPedidoCompra, {"idreg": ID}, [("GET", f"/pcmp/v2/pedidos/{qid}", None)]),
        (
            FreematicaListCuentasContables,
            {"prefijo_cuenta": "43"},
            [("GET", "/pcon/v2/cuentas", {"rquery": "COD_CTA=ge='43';COD_CTA=lt='44'"})],
        ),
        (FreematicaListCuentasAnaliticas, {}, [("GET", "/pcon/v2/cuentas-analiticas", None)]),
        (
            FreematicaExportAsientos,
            {"empresa": "E", "cal": "2026"},
            [("GET", "/pcon/v2/export-asientos", {"empresa": "E", "cal": "2026"})],
        ),
        (
            FreematicaListArticulos,
            {"activo": True},
            [("GET", "/part/v1/articulos", {**page, "rquery": "MOTIVO_BAJA==''"})],
        ),
        (FreematicaGetArticulo, {"idreg": ID}, [("GET", f"/part/v1/articulos/{qid}", None)]),
        (FreematicaGetPrecioArticulo, {"idreg": ID}, [("GET", f"/pgrl/v1/precio-articulo/{qid}", None)]),
        (FreematicaListMaterialesAsignadosServicios, {}, [("GET", "/pvss/v2/contratos-servicios-material", None)]),
        (
            FreematicaListFacturasCabecera,
            {"fecha_factura_hasta": "2026-01-31"},
            [("GET", "/pven/v1/facturas-cabecera", {**page, "rquery": "FVC_FCHFAC=lt='2026-02-01'"})],
        ),
        (FreematicaGetFacturaCabecera, {"idreg": ID}, [("GET", f"/pven/v1/facturas-cabecera/{qid}", None)]),
        (FreematicaListFacturaLineas, {"idreg": ID}, [("GET", f"/pven/v1/facturas-cabecera/{qid}/lineas", page)]),
        (FreematicaListFacturaIva, {"idreg": ID}, [("GET", f"/pven/v1/facturas-cabecera/{qid}/iva", page)]),
        (
            FreematicaListFacturaVencimientos,
            {"idreg": ID},
            [("GET", f"/pven/v1/facturas-cabecera/{qid}/vencimientos", page)],
        ),
        (
            FreematicaListAlbaranesVentas,
            {"empresa": "E", "fecha_desde": "2026-01-01"},
            [("GET", "/pven/v2/albaranes-ventas", {**page, "codEmpresa": "E", "desdeFecha": "2026-01-01"})],
        ),
        (FreematicaGetAlbaranVenta, {"idreg": ID}, [("GET", f"/pven/v2/albaranes-ventas/{qid}", None)]),
        (FreematicaListAlbaranesFactura, {}, [("GET", "/pven/v2/albaranes-facturas", page)]),
        (FreematicaGetAlbaranFactura, {"idreg": ID}, [("GET", f"/pven/v2/albaranes-facturas/{qid}", None)]),
        (
            FreematicaListCarteraClientes,
            {"estado": "pendiente", "solo_impagados": True},
            [
                (
                    "GET",
                    "/pcar/v1/cartera-clientes",
                    {**page, "rquery": "CARCL_SITCAR=='1';CARCL_FECIMPAG=ge='1900-01-01'"},
                )
            ],
        ),
        (FreematicaGetCarteraCliente, {"idreg": ID}, [("GET", f"/pcar/v1/cartera-clientes/{qid}", None)]),
        (FreematicaListFacturasElectronicas, {}, [("GET", "/pven/v1/facturas", page)]),
        (FreematicaGetFacturaElectronica, {"idreg": ID}, [("GET", f"/pven/v1/facturas/{qid}", None)]),
        (FreematicaGetFacturaDocumento, {"idreg": ID}, [("GET", f"/pven/v1/facturas/{qid}/documento", None)]),
        (FreematicaGetFacturaLog, {"idreg": ID}, [("GET", f"/pven/v1/facturas/{qid}/log", page)]),
        (FreematicaGetEdicomInfo, {}, [("GET", "/pven/v1/facturas/edicominfo", page)]),
        (FreematicaListFacturasDocumentos, {}, [("GET", "/pven/v1/facturas/download", page)]),
        (FreematicaListContratos, {"empresa": "E"}, [("GET", "/pvss/v1/contratos", {**page, "codEmpresa": "E"})]),
        (
            FreematicaGetContrato,
            {"empresa": "E1", "cod_contrato": "C1"},
            [("GET", "/pvss/v1/contratos", {"codEmpresa": "E1", "page": 1, "items": 200})],
        ),
        (
            FreematicaListServiciosContrato,
            {"id_contrato": CID},
            [("GET", f"/pvss/v1/contratos/{qcid}/servicios", page)],
        ),
        (FreematicaGetServicioContrato, {"idreg": ID}, [("GET", f"/pvss/v2/contratos-servicios/{qid}", None)]),
        (FreematicaListContratosOpcionales, {}, [("GET", "/ppre/v2/contratos/opcionales", page)]),
        (FreematicaGetContratoOpcionales, {"idreg": ID}, [("GET", f"/ppre/v2/contratos/opcionales/{qid}", None)]),
        (FreematicaListResultadosFacturacion, {}, [("GET", "/pvss/v1/facturacion-resultados", page)]),
        (
            FreematicaCreateContrato,
            {"delegacion": "D", "descripcion": "d", "fecha": "2026-01-01", "cod_cliente": "C"},
            [("POST", "/pvss/v2/contratos", None)],
        ),
        (FreematicaUpdateContrato, {"idreg": ID, "descripcion": "d"}, [("PUT", f"/pvss/v2/contratos/{qid}", None)]),
        (
            FreematicaCreateServicioContrato,
            {"id_contrato": CID, "descripcion": "svc"},
            [("POST", f"/pvss/v2/contratos/{qcid}/servicios", None)],
        ),
        (
            FreematicaUpdateServicioFechas,
            {"id_contrato": CID, "id_servicio": ID, "fecha_fin": "2026-02-01"},
            [("PUT", f"/pvss/v2/contratos/{qcid}/servicio/{qid}", None)],
        ),
        (
            FreematicaCreateServicioHistoricoPrecios,
            {"id_contrato": CID, "id_servicio": ID, "campos": {"SERVHPR_PRECIO": 1}},
            [("POST", f"/pvss/v2/contratos/{qcid}/servicios-historico-precios/{qid}", None)],
        ),
        (
            FreematicaUpdateServicioHistoricoPrecios,
            {"id_contrato": CID, "id_servicio": ID, "campos": {"SERVHPR_PRECIO": 2}},
            [("PUT", f"/pvss/v2/contratos/{qcid}/servicios-historico-precios/{qid}", None)],
        ),
        (
            FreematicaCreateServicioFacturacionTxt,
            {"id_contrato": CID, "id_servicio": ID, "texto": "hola"},
            [("POST", f"/pvss/v2/contratos/{qcid}/servicios-facturacion-txt/{qid}", None)],
        ),
        (
            FreematicaUpdateServicioFacturacion,
            {"id_contrato": CID, "id_servicio": ID, "campos": {"CTRTF_PRECIO": 1}},
            [("PUT", f"/pvss/v2/contratos/{qcid}/servicios-facturacion/{qid}", None)],
        ),
        (
            FreematicaCreateContratoOpcionales,
            {"empresa": "E", "delegacion": "D", "tipo_contrato": "T", "num_contrato": "1", "fecha": "2026-01-01"},
            [("POST", "/ppre/v2/contratos/opcionales", None)],
        ),
        (
            FreematicaUpdateContratoOpcionales,
            {"idreg": ID, "campos": {"CON2_OBS": "x"}},
            [("PUT", f"/ppre/v2/contratos/opcionales/{qid}", None)],
        ),
    ]


async def _call(tool: Any, **kwargs: Any) -> Any:
    return await tool(**kwargs).collect()


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    _routes(),
    ids=[f"{cls.__name__}-{index}" for index, (cls, _kwargs, _expected) in enumerate(_routes())],
)
async def test_route(http: list[dict[str, Any]], cls: Any, kwargs: dict[str, Any], expected: list[tuple]) -> None:
    if not isinstance(cls, type):
        pytest.skip("id row")
    result = await _call(cls(), **kwargs)
    assert result.status.code == "success", result.error
    assert len(http) == len(expected)
    for got, (method, path, params) in zip(http, expected, strict=True):
        assert got["method"] == method
        assert got["url"] == f"{BASE}{path}"
        for header, value in AUTH_HEADERS.items():
            assert got["headers"][header] == value
        if params is not None:
            assert got["params"] == params


async def test_create_cliente_body_and_update_merge(http: list[dict[str, Any]]) -> None:
    created = await _call(fm.FreematicaCreateCliente(), **_CLIENT)
    assert created.status.code == "success"
    body = http[-1]["json"]
    assert body["COD_CLI"] == "C"
    assert body["TIPO_FACT"] == "M"
    assert body["idReg"] == cliente_idreg("G", "C")

    http.clear()
    updated = await _call(fm.FreematicaUpdateCliente(), idreg=ID, nombre="New")
    assert updated.status.code == "success"
    assert http[0]["method"] == "GET"
    merged = http[1]["json"]
    assert merged["NOMBRE_CLI"] == "New"
    assert merged["items"][0]["ok"] is True


async def test_create_servicio_decodes_contract_id(http: list[dict[str, Any]]) -> None:
    result = await _call(FreematicaCreateServicioContrato(), id_contrato=CID, descripcion="svc")
    assert result.status.code == "success"
    body = http[0]["json"]
    assert body["CTRTS_EMP"] == "E1"
    assert body["CTRTS_DELEG"] == "D1"
    assert body["CTRTS_CTRT"] == "C1"
    assert body["CTRTS_DES"] == "svc"


async def test_location_body_uses_tipo_fields(http: list[dict[str, Any]]) -> None:
    result = await _call(
        fm.FreematicaCreateLocalizacionCliente(),
        tipo="cobro",
        grupo_cliente="G",
        cod_cliente="C",
        cod_localizacion="L",
        nombre="N",
        forma_pago="TR",
    )
    assert result.status.code == "success"
    body = http[0]["json"]
    assert body["LOC_CLI_COB"] == "L"
    assert body["NOM_LOC_COB"] == "N"
    assert body["COD_FORMA_COBRO"] == "TR"


async def test_rango_no_soportado_and_bad_date(http: list[dict[str, Any]]) -> None:
    result = await _call(
        FreematicaListFacturasCabecera(), fecha_factura_desde="2026-01-01", fecha_factura_hasta="2026-01-31"
    )
    assert result.status.code == "error"
    assert RANGO_NO_SOPORTADO in result.error["message"]
    assert http == []

    bad = await _call(FreematicaListFacturasCabecera(), fecha_factura_hasta="2026-02-30")
    assert bad.status.code == "error"
    assert "Invalid date" in bad.error["message"]


async def test_page_zero_rejected() -> None:
    result = await _call(fm.FreematicaListClientes(), page=0)
    assert result.status.code == "error"
    assert result.error["type"] == "ValidationError"
    assert "greater than or equal to 1" in result.error["message"]


async def test_cartera_rejects_unknown_estado(http: list[dict[str, Any]]) -> None:
    result = await _call(FreematicaListCarteraClientes(), estado="pagado")
    assert result.status.code == "error"
    assert "estado" in result.error["message"]
    assert http == []


async def test_proveedores_activo_filters_client_side(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    rows = [{"FECHA_BAJA": ""}, {"FECHA_BAJA": "2020-01-01"}, {"FECHA_BAJA": None}]
    _patch_client(monkeypatch, [_ok({"items": rows, "total": "3"})])
    result = await _call(FreematicaListProveedores(), activo=True)
    assert result.status.code == "success"
    assert result.output["count"] == 2
    assert result.output["total"] == 3


async def test_export_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in AUTH_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FREEMATICA_MAX_RESPONSE_SIZE_MB", "0")
    _patch_client(monkeypatch, [_ok({"items": [{"a": "x" * 20}], "total": "1"})])
    result = await _call(FreematicaExportAsientos(), empresa="E", cal="2026")
    assert result.status.code == "success"
    assert result.output["truncated"] is True
    assert "warning" in result.output


def test_registry_lists_every_freematica_tool() -> None:
    from timbal import tools

    names = [name for name in tools.__all__ if name.startswith("Freematica")]
    assert len(names) == 72
    for name in names:
        module = tools._LAZY_IMPORTS[name]
        assert module == ".freematica" or module.startswith(".freematica_")
        cls = getattr(tools, name)
        assert cls.__name__ == name
        assert cls().name.startswith("freematica_")


# --- live (skipped without credentials) ---------------------------------------


def _live_ready() -> bool:
    import os

    return all(
        os.getenv(key)
        for key in (
            "FREEMATICA_AUTH_TOKEN",
            "FREEMATICA_AUTH_COMPANY",
            "FREEMATICA_AUTH_ORGANIZATION",
            "FREEMATICA_AUTH_APP",
            "FREEMATICA_AUTH_SESSION",
        )
    )


@pytest.mark.integration
async def test_live_list_clientes_and_master_data() -> None:
    if not _live_ready():
        pytest.skip("FREEMATICA_AUTH_* is not set")
    clients = await _call(fm.FreematicaListClientes(), page=1, items=1)
    assert clients.status.code == "success"
    assert "items" in clients.output
    empresas = await _call(fm.FreematicaGetMasterData(), catalog="empresas")
    assert empresas.status.code == "success"
    items = clients.output["items"]
    if not items or not items[0].get("idReg"):
        pytest.skip("client list returned no idReg to open a detail")
    detail = await _call(fm.FreematicaGetCliente(), idreg=items[0]["idReg"])
    assert detail.status.code == "success"
