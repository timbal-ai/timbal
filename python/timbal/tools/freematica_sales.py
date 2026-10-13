"""Freemática tools: sales invoices, delivery notes, receivables, and e-invoices.

Auth and request handling live in ``freematica.py``.
"""

from typing import Any
from urllib.parse import quote

from pydantic import Field

from .freematica import (
    RANGO_NO_SOPORTADO,
    _freematica_request,
    _FreematicaTool,
    _items,
    _list_params,
    _page,
    build_fiql,
    next_day_iso,
)

_ESTADO_CARTERA = {"pendiente": "1", "cancelado": "2", "derivado": "3"}


class FreematicaListFacturasCabecera(_FreematicaTool):
    """Sales invoice headers. From-date and to-date cannot be combined."""

    name: str = "freematica_list_facturas_cabecera"
    description: str | None = (
        "List sales invoice headers. Do not set both fecha_factura_desde and fecha_factura_hasta: "
        "the API returns 0 rows. traspasado_contabilidad false finds invoices not posted to accounting."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            empresa: str | None = Field(None, description="Company (FVC_CODEMP)."),
            cod_cliente: str | None = Field(None, description="Client code (FVC_CODCLI)."),
            representante: str | None = Field(None, description="Sales rep (FVC_CODREPRES)."),
            serie: str | None = Field(None, description="Invoice series (FVC_SERIEFRA)."),
            num_factura: str | None = Field(None, description="Invoice number (FVC_NUMFRA)."),
            forma_pago: str | None = Field(None, description="Payment form (FVC_FPAGO)."),
            delegacion: str | None = Field(None, description="Branch (FVC_DELEG)."),
            fecha_factura_desde: str | None = Field(None, description="Invoice date from, YYYY-MM-DD (FVC_FCHFAC >=)."),
            fecha_factura_hasta: str | None = Field(
                None, description="Invoice date to, inclusive (emulated as < next day)."
            ),
            traspasado_contabilidad: bool | None = Field(
                None, description="Posted to accounting (FVC_TRASP_CONTAB 1/0)."
            ),
        ) -> Any:
            if fecha_factura_desde and fecha_factura_hasta:
                raise ValueError(RANGO_NO_SOPORTADO)
            parts = [
                build_fiql(
                    {
                        "FVC_CODEMP": empresa,
                        "FVC_CODCLI": cod_cliente,
                        "FVC_CODREPRES": representante,
                        "FVC_SERIEFRA": serie,
                        "FVC_NUMFRA": num_factura,
                        "FVC_FPAGO": forma_pago,
                        "FVC_DELEG": delegacion,
                    }
                )
            ]
            if fecha_factura_desde:
                parts.append(build_fiql({"FVC_FCHFAC": {"op": "ge", "value": fecha_factura_desde}}))
            if fecha_factura_hasta:
                parts.append(build_fiql({"FVC_FCHFAC": {"op": "lt", "value": next_day_iso(fecha_factura_hasta)}}))
            if traspasado_contabilidad is not None:
                parts.append(build_fiql({"FVC_TRASP_CONTAB": "1" if traspasado_contabilidad else "0"}))
            return await _freematica_request(
                self,
                "GET",
                "/pven/v1/facturas-cabecera",
                params=_list_params(page, items, ";".join(p for p in parts if p)),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetFacturaCabecera(_FreematicaTool):
    """One sales invoice header."""

    name: str = "freematica_get_factura_cabecera"
    description: str | None = "Get one sales invoice header by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pven/v1/facturas-cabecera/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListFacturaLineas(_FreematicaTool):
    """Lines of one sales invoice."""

    name: str = "freematica_list_factura_lineas"
    description: str | None = "List the lines of one sales invoice."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            idreg: str = Field(..., description="Invoice idReg."), page: int = _page(), items: int = _items()
        ) -> Any:
            return await _freematica_request(
                self,
                "GET",
                f"/pven/v1/facturas-cabecera/{quote(idreg, safe='')}/lineas",
                params=_list_params(page, items),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaListFacturaIva(_FreematicaTool):
    """VAT lines of one sales invoice."""

    name: str = "freematica_list_factura_iva"
    description: str | None = "List the VAT lines of one sales invoice."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            idreg: str = Field(..., description="Invoice idReg."), page: int = _page(), items: int = _items()
        ) -> Any:
            return await _freematica_request(
                self, "GET", f"/pven/v1/facturas-cabecera/{quote(idreg, safe='')}/iva", params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaListFacturaVencimientos(_FreematicaTool):
    """Due dates of one sales invoice."""

    name: str = "freematica_list_factura_vencimientos"
    description: str | None = "List the collection due dates of one sales invoice."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            idreg: str = Field(..., description="Invoice idReg."), page: int = _page(), items: int = _items()
        ) -> Any:
            return await _freematica_request(
                self,
                "GET",
                f"/pven/v1/facturas-cabecera/{quote(idreg, safe='')}/vencimientos",
                params=_list_params(page, items),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaListAlbaranesVentas(_FreematicaTool):
    """Sales delivery notes. empresa is required by the API."""

    name: str = "freematica_list_albaranes_ventas"
    description: str | None = "List sales delivery notes. empresa is required. Filters are native query params."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            empresa: str = Field(..., description="Company code (required by the API, native codEmpresa)."),
            page: int = _page(),
            items: int = _items(),
            delegacion: str | None = Field(None, description="Branch (native codDelegacion)."),
            cod_cliente: str | None = Field(None, description="Client code (native codCliente)."),
            cod_documento: str | None = Field(None, description="Document code (native codDocumento)."),
            fecha_desde: str | None = Field(None, description="From date YYYY-MM-DD (native desdeFecha)."),
            fecha_hasta: str | None = Field(None, description="To date YYYY-MM-DD (native hastaFecha)."),
            order: str | None = Field(None, description="Native order expression."),
        ) -> Any:
            return await _freematica_request(
                self,
                "GET",
                "/pven/v2/albaranes-ventas",
                params=_list_params(
                    page,
                    items,
                    "",
                    codEmpresa=empresa,
                    codDelegacion=delegacion,
                    codCliente=cod_cliente,
                    codDocumento=cod_documento,
                    desdeFecha=fecha_desde,
                    hastaFecha=fecha_hasta,
                    order=order,
                ),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetAlbaranVenta(_FreematicaTool):
    """One sales delivery note."""

    name: str = "freematica_get_albaran_venta"
    description: str | None = "Get one sales delivery note by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pven/v2/albaranes-ventas/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListAlbaranesFactura(_FreematicaTool):
    """Links between a delivery note and an invoice."""

    name: str = "freematica_list_albaranes_factura"
    description: str | None = "List delivery-note to invoice links."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            idreg: str | None = Field(None, description="Optional exact idReg filter (native, not FIQL)."),
        ) -> Any:
            params = _list_params(page, items)
            if idreg:
                params["idReg"] = idreg
            return await _freematica_request(self, "GET", "/pven/v2/albaranes-facturas", params=params)

        super().__init__(handler=_list, **kwargs)


class FreematicaGetAlbaranFactura(_FreematicaTool):
    """One delivery-note/invoice link."""

    name: str = "freematica_get_albaran_factura"
    description: str | None = "Get one delivery-note/invoice link by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pven/v2/albaranes-facturas/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListCarteraClientes(_FreematicaTool):
    """Client receivables. estado maps to CARCL_SITCAR (1 pendiente, 2 cancelado, 3 derivado)."""

    name: str = "freematica_list_cartera_clientes"
    description: str | None = (
        "List client receivables. estado is pendiente, cancelado or derivado. "
        "solo_impagados finds documents with an unpaid date. Do not combine fecha_doc_desde and fecha_doc_hasta."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            empresa: str | None = Field(None, description="Company (CARCL_EMP)."),
            cod_cliente: str | None = Field(None, description="Client code (CARCL_CODAUX)."),
            grupo_cliente: str | None = Field(None, description="Client group (CARCL_GRUPAUX)."),
            estado: str | None = Field(None, description="pendiente, cancelado or derivado (CARCL_SITCAR)."),
            referencia: str | None = Field(None, description="Reference (CARCL_REFCAR)."),
            forma_pago: str | None = Field(None, description="Payment form (CARCL_CODFPAG)."),
            fecha_doc_desde: str | None = Field(None, description="Document date from, YYYY-MM-DD."),
            fecha_doc_hasta: str | None = Field(None, description="Document date to, inclusive."),
            fecha_vencimiento_desde: str | None = Field(None, description="Due date from. A to-date is not supported."),
            solo_impagados: bool | None = Field(None, description="Only documents with an unpaid date."),
        ) -> Any:
            if fecha_doc_desde and fecha_doc_hasta:
                raise ValueError(RANGO_NO_SOPORTADO)
            if estado is not None and estado not in _ESTADO_CARTERA:
                raise ValueError("estado must be pendiente, cancelado or derivado.")
            parts = [
                build_fiql(
                    {
                        "CARCL_EMP": empresa,
                        "CARCL_CODAUX": cod_cliente,
                        "CARCL_GRUPAUX": grupo_cliente,
                        "CARCL_SITCAR": _ESTADO_CARTERA.get(estado) if estado else None,
                        "CARCL_REFCAR": referencia,
                        "CARCL_CODFPAG": forma_pago,
                    }
                )
            ]
            if fecha_doc_desde:
                parts.append(build_fiql({"CARCL_FECDOC": {"op": "ge", "value": fecha_doc_desde}}))
            if fecha_doc_hasta:
                parts.append(build_fiql({"CARCL_FECDOC": {"op": "lt", "value": next_day_iso(fecha_doc_hasta)}}))
            if fecha_vencimiento_desde:
                parts.append(build_fiql({"CARCL_FECVCTO": {"op": "ge", "value": fecha_vencimiento_desde}}))
            if solo_impagados:
                parts.append(build_fiql({"CARCL_FECIMPAG": {"op": "ge", "value": "1900-01-01"}}))
            return await _freematica_request(
                self,
                "GET",
                "/pcar/v1/cartera-clientes",
                params=_list_params(page, items, ";".join(p for p in parts if p)),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetCarteraCliente(_FreematicaTool):
    """One receivables document."""

    name: str = "freematica_get_cartera_cliente"
    description: str | None = "Get one receivables document by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pcar/v1/cartera-clientes/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListFacturasElectronicas(_FreematicaTool):
    """Electronic invoices."""

    name: str = "freematica_list_facturas_electronicas"
    description: str | None = "List electronic invoices (GET /pven/v1/facturas)."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(self, "GET", "/pven/v1/facturas", params=_list_params(page, items))

        super().__init__(handler=_list, **kwargs)


class FreematicaGetFacturaElectronica(_FreematicaTool):
    """One electronic invoice."""

    name: str = "freematica_get_factura_electronica"
    description: str | None = "Get one electronic invoice by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pven/v1/facturas/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaGetFacturaDocumento(_FreematicaTool):
    """Electronic invoice document payload."""

    name: str = "freematica_get_factura_documento"
    description: str | None = "Get the document payload of an electronic invoice."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pven/v1/facturas/{quote(idreg, safe='')}/documento")

        super().__init__(handler=_get, **kwargs)


class FreematicaGetFacturaLog(_FreematicaTool):
    """Transmission log of an electronic invoice."""

    name: str = "freematica_get_factura_log"
    description: str | None = "List the transmission log of an electronic invoice."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(
            idreg: str = Field(..., description="Opaque idReg."), page: int = _page(), items: int = _items()
        ) -> Any:
            return await _freematica_request(
                self, "GET", f"/pven/v1/facturas/{quote(idreg, safe='')}/log", params=_list_params(page, items)
            )

        super().__init__(handler=_get, **kwargs)


class FreematicaGetEdicomInfo(_FreematicaTool):
    """Edicom connector info for electronic invoicing."""

    name: str = "freematica_get_edicom_info"
    description: str | None = "Get Edicom electronic-invoicing connector info."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/pven/v1/facturas/edicominfo", params=_list_params(page, items)
            )

        super().__init__(handler=_get, **kwargs)


class FreematicaListFacturasDocumentos(_FreematicaTool):
    """Downloadable electronic-invoice documents."""

    name: str = "freematica_list_facturas_documentos"
    description: str | None = "List downloadable electronic-invoice documents (GET /pven/v1/facturas/download)."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/pven/v1/facturas/download", params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)
