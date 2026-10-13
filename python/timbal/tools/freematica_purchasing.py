"""Freemática tools: suppliers, purchases, accounting, and the article catalog.

Auth and request handling live in ``freematica.py``.
"""

from typing import Any, Literal
from urllib.parse import quote

from pydantic import Field

from .freematica import (
    RANGO_NO_SOPORTADO,
    _first_item,
    _freematica_request,
    _FreematicaTool,
    _guard_size,
    _items,
    _list_params,
    _page,
    build_fiql,
    next_day_iso,
    next_prefix,
)


class FreematicaListProveedores(_FreematicaTool):
    """Suppliers. nombre is an exact match. activo=true is filtered client-side."""

    name: str = "freematica_list_proveedores"
    description: str | None = (
        "List suppliers. nombre is an exact match (partial search returns 400 or 0 rows). "
        "activo true drops deactivated rows from the page; the reported total is unfiltered."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            cod_proveedor: str | None = Field(None, description="Supplier code (COD_PRO)."),
            grupo_proveedor: str | None = Field(None, description="Supplier group (COD_GRUPO_PRO)."),
            nif: str | None = Field(None, description="Tax id (NIF)."),
            nombre: str | None = Field(None, description="Exact name (NOMBRE_PRO)."),
            activo: bool | None = Field(None, description="True keeps rows with an empty FECHA_BAJA."),
            tipo_ident: str | None = Field(None, description="Id type (CMP_TIPO_IDENT)."),
            cod_provincia: str | None = Field(None, description="Province (COD_PROVINCIA)."),
            cod_pais: str | None = Field(None, description="Country (COD_PAIS)."),
        ) -> Any:
            filters: dict[str, Any] = {
                "COD_PRO": cod_proveedor,
                "COD_GRUPO_PRO": grupo_proveedor,
                "NIF": nif,
                "NOMBRE_PRO": nombre,
                "CMP_TIPO_IDENT": tipo_ident,
                "COD_PROVINCIA": cod_provincia,
                "COD_PAIS": cod_pais,
            }
            if activo is False:
                filters["FECHA_BAJA"] = {"op": "ge", "value": "1900-01-01"}
            result = await _freematica_request(
                self, "GET", "/pgrl/v2/proveedores", params=_list_params(page, items, build_fiql(filters))
            )
            if activo is True and isinstance(result, dict):
                kept = [
                    row for row in result.get("items") or [] if not str((row or {}).get("FECHA_BAJA") or "").strip()
                ]
                result = {**result, "items": kept, "count": len(kept)}
            return result

        super().__init__(handler=_list, **kwargs)


class FreematicaGetProveedor(_FreematicaTool):
    """One supplier."""

    name: str = "freematica_get_proveedor"
    description: str | None = "Get one supplier by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pgrl/v2/proveedores/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListFacturasCompras(_FreematicaTool):
    """Purchase invoices. exportado is a native enum, not FIQL."""

    name: str = "freematica_list_facturas_compras"
    description: str | None = (
        "List purchase invoices. exportado is all or not_exported (native param). "
        "Other filters are FIQL on FCC_* columns."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            empresa: str | None = Field(None, description="Company (FCC_CODEMP)."),
            cod_proveedor: str | None = Field(None, description="Supplier (FCC_CODPRO)."),
            serie: str | None = Field(None, description="Series (FCC_SERIEFRA)."),
            num_factura: str | None = Field(None, description="Number (FCC_NUMFRA)."),
            forma_pago: str | None = Field(None, description="Payment form (FCC_FPAGO)."),
            delegacion: str | None = Field(None, description="Branch (FCC_DELEG)."),
            fecha_desde: str | None = Field(None, description="Invoice date from (FCC_FCHFAC >=)."),
            fecha_hasta: str | None = Field(None, description="Invoice date to (FCC_FCHFAC <=)."),
            traspasado_contabilidad: bool | None = Field(None, description="Posted to accounting."),
            exportado: Literal["all", "not_exported"] | None = Field(None, description="Native exportado filter."),
        ) -> Any:
            parts = [
                build_fiql(
                    {
                        "FCC_CODEMP": empresa,
                        "FCC_CODPRO": cod_proveedor,
                        "FCC_SERIEFRA": serie,
                        "FCC_NUMFRA": num_factura,
                        "FCC_FPAGO": forma_pago,
                        "FCC_DELEG": delegacion,
                        "FCC_TRASP_CONTAB": None
                        if traspasado_contabilidad is None
                        else ("1" if traspasado_contabilidad else "0"),
                    }
                ),
                build_fiql({"FCC_FCHFAC": {"op": "ge", "value": fecha_desde}}) if fecha_desde else "",
                build_fiql({"FCC_FCHFAC": {"op": "le", "value": fecha_hasta}}) if fecha_hasta else "",
            ]
            params = _list_params(page, items, ";".join(part for part in parts if part))
            if exportado:
                params["exportado"] = exportado
            return await _freematica_request(self, "GET", "/pcmp/v2/facturas-compras", params=params)

        super().__init__(handler=_list, **kwargs)


class FreematicaGetFacturaCompra(_FreematicaTool):
    """One purchase invoice."""

    name: str = "freematica_get_factura_compra"
    description: str | None = "Get one purchase invoice by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pcmp/v2/facturas-compras/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListPedidosCompra(_FreematicaTool):
    """Purchase orders. Some filters are native query params, the rest are FIQL."""

    name: str = "freematica_list_pedidos_compra"
    description: str | None = (
        "List purchase orders. empresa, cod_proveedor and order dates are native params; "
        "num_pedido, delegacion and forma_pago are FIQL."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            empresa: str | None = Field(None, description="Company (native codEmpresa)."),
            cod_proveedor: str | None = Field(None, description="Supplier (native codProveedor)."),
            fecha_pedido_desde: str | None = Field(None, description="Order date from (native desdeFecha)."),
            fecha_pedido_hasta: str | None = Field(None, description="Order date to (native hastaFecha)."),
            num_pedido: str | None = Field(None, description="Order number (ALCC_NUMDOC)."),
            delegacion: str | None = Field(None, description="Branch (ALCC_DELEG)."),
            forma_pago: str | None = Field(None, description="Payment form (ALCC_FPAGO)."),
            referencia: str | None = Field(None, description="Reference (ALCC_REFERENCIA)."),
        ) -> Any:
            fiql = build_fiql(
                {
                    "ALCC_NUMDOC": num_pedido,
                    "ALCC_DELEG": delegacion,
                    "ALCC_FPAGO": forma_pago,
                    "ALCC_REFERENCIA": referencia,
                }
            )
            return await _freematica_request(
                self,
                "GET",
                "/pcmp/v2/pedidos",
                params=_list_params(
                    page,
                    items,
                    fiql,
                    codEmpresa=empresa,
                    codProveedor=cod_proveedor,
                    desdeFecha=fecha_pedido_desde,
                    hastaFecha=fecha_pedido_hasta,
                ),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetPedidoCompra(_FreematicaTool):
    """One purchase order, header plus lines."""

    name: str = "freematica_get_pedido_compra"
    description: str | None = "Get one purchase order (header, supplier and lines) by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pcmp/v2/pedidos/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListCuentasContables(_FreematicaTool):
    """Chart of accounts. prefijo_cuenta is a lexical prefix on COD_CTA."""

    name: str = "freematica_list_cuentas_contables"
    description: str | None = (
        "List general-ledger accounts. prefijo_cuenta matches COD_CTA by range (>= prefix and < the next prefix)."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            cod_plan: str | None = Field(None, description="Chart code (COD_PLAN)."),
            prefijo_cuenta: str | None = Field(None, description="Account-code prefix (COD_CTA)."),
            activa: bool | None = Field(None, description="Active flag (CTA_ACTIVA 1/0)."),
            grupo_cuenta: str | None = Field(None, description="Account group (COD_GRUPO_CTA)."),
        ) -> Any:
            parts = [
                build_fiql(
                    {
                        "COD_PLAN": cod_plan,
                        "CTA_ACTIVA": None if activa is None else ("1" if activa else "0"),
                        "COD_GRUPO_CTA": grupo_cuenta,
                    }
                )
            ]
            if prefijo_cuenta:
                parts.append(build_fiql({"COD_CTA": {"op": "ge", "value": prefijo_cuenta}}))
                nxt = next_prefix(prefijo_cuenta)
                if nxt:
                    parts.append(build_fiql({"COD_CTA": {"op": "lt", "value": nxt}}))
            fiql = ";".join(part for part in parts if part)
            params = {"rquery": fiql} if fiql else None
            return await _freematica_request(self, "GET", "/pcon/v2/cuentas", params=params)

        super().__init__(handler=_list, **kwargs)


class FreematicaListCuentasAnaliticas(_FreematicaTool):
    """Analytical accounts."""

    name: str = "freematica_list_cuentas_analiticas"
    description: str | None = "List analytical accounts. prefijo_cuenta is a prefix on COD_CTA_ANL."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            cod_plan: str | None = Field(None, description="Chart code (COD_PLAN)."),
            prefijo_cuenta: str | None = Field(None, description="Account prefix (COD_CTA_ANL)."),
            activa: bool | None = Field(None, description="Active flag (CTA_ACTIVA_ANL)."),
            grupo_cuenta: str | None = Field(None, description="Group (COD_GRUPO_ANL)."),
            area: str | None = Field(None, description="Area (AREA_ANL)."),
            delegacion: str | None = Field(None, description="Branch (DELEG)."),
        ) -> Any:
            parts = [
                build_fiql(
                    {
                        "COD_PLAN": cod_plan,
                        "CTA_ACTIVA_ANL": None if activa is None else ("1" if activa else "0"),
                        "COD_GRUPO_ANL": grupo_cuenta,
                        "AREA_ANL": area,
                        "DELEG": delegacion,
                    }
                )
            ]
            if prefijo_cuenta:
                parts.append(build_fiql({"COD_CTA_ANL": {"op": "ge", "value": prefijo_cuenta}}))
                nxt = next_prefix(prefijo_cuenta)
                if nxt:
                    parts.append(build_fiql({"COD_CTA_ANL": {"op": "lt", "value": nxt}}))
            fiql = ";".join(part for part in parts if part)
            return await _freematica_request(
                self, "GET", "/pcon/v2/cuentas-analiticas", params={"rquery": fiql} if fiql else None
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaExportAsientos(_FreematicaTool):
    """Export journal entries. Responses above FREEMATICA_MAX_RESPONSE_SIZE_MB are truncated."""

    name: str = "freematica_export_asientos"
    description: str | None = (
        "Export journal entries (GET /pcon/v2/export-asientos). empresa and cal are required. "
        "Do not set both fecha_desde and fecha_hasta. Large responses are truncated with a warning."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _export(
            empresa: str = Field(..., description="Company (native empresa)."),
            cal: str = Field(..., description="Accounting calendar (native cal)."),
            periodo: str | None = Field(None, description="Period (native periodo)."),
            fecha_desde: str | None = Field(None, description="Entry date from (ASI_FCHASI >=)."),
            fecha_hasta: str | None = Field(None, description="Entry date to, inclusive (emulated as < next day)."),
            diario: str | None = Field(None, description="Journal (ASI_DIARIO)."),
            borrador: bool | None = Field(None, description="True keeps draft entries (ASI_BORR not empty)."),
        ) -> Any:
            if fecha_desde and fecha_hasta:
                raise ValueError(RANGO_NO_SOPORTADO)
            parts: list[str] = []
            if fecha_desde:
                parts.append(build_fiql({"ASI_FCHASI": {"op": "ge", "value": fecha_desde}}))
            if fecha_hasta:
                parts.append(build_fiql({"ASI_FCHASI": {"op": "lt", "value": next_day_iso(fecha_hasta)}}))
            if diario:
                parts.append(build_fiql({"ASI_DIARIO": diario}))
            if borrador is True:
                parts.append(build_fiql({"ASI_BORR": {"op": "ne", "value": ""}}))
            elif borrador is False:
                parts.append(build_fiql({"ASI_BORR": ""}))
            params: dict[str, Any] = {"empresa": empresa, "cal": cal}
            if periodo:
                params["periodo"] = periodo
            if parts:
                params["rquery"] = ";".join(parts)
            result = await _freematica_request(self, "GET", "/pcon/v2/export-asientos", params=params)
            if not isinstance(result, dict):
                return result
            return _guard_size(result)

        super().__init__(handler=_export, **kwargs)


class FreematicaListArticulos(_FreematicaTool):
    """Article catalog. Filters are exact. activo maps to MOTIVO_BAJA."""

    name: str = "freematica_list_articulos"
    description: str | None = (
        "List articles, materials and consumables. Filters are exact (partial search is rejected). "
        "COD_ARTICULO may include leading spaces. activo true/false maps to an empty/non-empty MOTIVO_BAJA."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            cod_articulo: str | None = Field(None, description="Exact article code, spaces included (COD_ARTICULO)."),
            tipo_codigo: str | None = Field(None, description="Code type (TIPO_CODIGO)."),
            cod_proveedor: str | None = Field(None, description="Supplier (COD_PROVEEDOR)."),
            linea: str | None = Field(None, description="Line (COD_LIN_ART)."),
            familia: str | None = Field(None, description="Family (COD_FAMILIA)."),
            subfamilia: str | None = Field(None, description="Subfamily (COD_SUBFAM)."),
            descripcion: str | None = Field(None, description="Exact description (DESC_ART)."),
            codigo_barras: str | None = Field(None, description="Barcode (CODIGO_BARRAS)."),
            activo: bool | None = Field(None, description="True: empty MOTIVO_BAJA. False: non-empty."),
        ) -> Any:
            filters: dict[str, Any] = {
                "COD_ARTICULO": cod_articulo,
                "TIPO_CODIGO": tipo_codigo,
                "COD_PROVEEDOR": cod_proveedor,
                "COD_LIN_ART": linea,
                "COD_FAMILIA": familia,
                "COD_SUBFAM": subfamilia,
                "DESC_ART": descripcion,
                "CODIGO_BARRAS": codigo_barras,
            }
            if activo is True:
                filters["MOTIVO_BAJA"] = ""
            elif activo is False:
                filters["MOTIVO_BAJA"] = {"op": "ne", "value": ""}
            return await _freematica_request(
                self, "GET", "/part/v1/articulos", params=_list_params(page, items, build_fiql(filters))
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetArticulo(_FreematicaTool):
    """One article. The API returns a one-item list."""

    name: str = "freematica_get_articulo"
    description: str | None = "Get one article. idreg is the opaque idReg from freematica_list_articulos."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            result = await _freematica_request(self, "GET", f"/part/v1/articulos/{quote(idreg, safe='')}")
            return _first_item(result, "Article", idreg)

        super().__init__(handler=_get, **kwargs)


class FreematicaGetPrecioArticulo(_FreematicaTool):
    """Sale prices of one article."""

    name: str = "freematica_get_precio_articulo"
    description: str | None = "Get sale prices (PRECIO_VENTA, DESCUENTO, FACTURABLE) for an article idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Article idReg from freematica_list_articulos.")) -> Any:
            return await _freematica_request(self, "GET", f"/pgrl/v1/precio-articulo/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListMaterialesAsignadosServicios(_FreematicaTool):
    """Materials already assigned to contract services, not the article catalog."""

    name: str = "freematica_list_materiales_asignados_servicios"
    description: str | None = (
        "List materials already assigned to contract services. For the reference catalog use freematica_list_articulos."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _freematica_request(self, "GET", "/pvss/v2/contratos-servicios-material")

        super().__init__(handler=_list, **kwargs)
