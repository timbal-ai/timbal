"""Freemática tools: contracts, services, optionals, and their create/update writes.

There is no delete tool. Auth and request handling live in ``freematica.py``.
"""

from typing import Any
from urllib.parse import quote

from pydantic import Field

from .freematica import (
    _drop_none,
    _error,
    _freematica_request,
    _FreematicaTool,
    _items,
    _list_params,
    _page,
    decode_contrato_idreg,
)


class FreematicaListContratos(_FreematicaTool):
    """Contract headers. This endpoint ignores FIQL; only native company/branch filters work."""

    name: str = "freematica_list_contratos"
    description: str | None = (
        "List contract headers (CTRT_*). FIQL is ignored here: only empresa and delegacion filter. "
        "Use freematica_get_contrato to find one contract by its natural code."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            empresa: str | None = Field(None, description="Company (native codEmpresa)."),
            delegacion: str | None = Field(None, description="Branch (native codDelegacion)."),
            order: str | None = Field(None, description="Native order expression."),
        ) -> Any:
            return await _freematica_request(
                self,
                "GET",
                "/pvss/v1/contratos",
                params=_list_params(page, items, "", codEmpresa=empresa, codDelegacion=delegacion, order=order),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetContrato(_FreematicaTool):
    """Find a contract by natural codes, paging the list because FIQL is ignored."""

    name: str = "freematica_get_contrato"
    description: str | None = (
        "Find a contract by empresa + cod_contrato (optional delegacion). "
        "The list endpoint ignores FIQL, so this scans pages of 200 until it matches CTRT_COD."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get(
            empresa: str = Field(..., description="Company (CTRT_EMP)."),
            cod_contrato: str = Field(..., description="Natural contract code (CTRT_COD)."),
            delegacion: str | None = Field(None, description="Branch (CTRT_DELEG)."),
        ) -> Any:
            page_size = 200
            for page in range(1, 101):
                result = await _freematica_request(
                    self,
                    "GET",
                    "/pvss/v1/contratos",
                    params=_list_params(page, page_size, "", codEmpresa=empresa, codDelegacion=delegacion),
                )
                items = result.get("items") if isinstance(result, dict) else []
                for row in items or []:
                    same_code = str(row.get("CTRT_COD")) == cod_contrato and str(row.get("CTRT_EMP")) == empresa
                    same_branch = delegacion is None or str(row.get("CTRT_DELEG")) == delegacion
                    if same_code and same_branch:
                        return row
                if len(items or []) < page_size:
                    break
            raise _error(
                "not_found",
                f"Contract not found: empresa={empresa} delegacion={delegacion or '*'} cod_contrato={cod_contrato}",
            )

        super().__init__(handler=_get, **kwargs)


class FreematicaListServiciosContrato(_FreematicaTool):
    """Services of one contract."""

    name: str = "freematica_list_servicios_contrato"
    description: str | None = "List services of a contract. id_contrato is the contract idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            id_contrato: str = Field(..., description="Contract idReg."),
            page: int = _page(),
            items: int = _items(),
        ) -> Any:
            return await _freematica_request(
                self,
                "GET",
                f"/pvss/v1/contratos/{quote(id_contrato, safe='')}/servicios",
                params=_list_params(page, items),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetServicioContrato(_FreematicaTool):
    """One contract service."""

    name: str = "freematica_get_servicio_contrato"
    description: str | None = "Get one contract service by its opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Service idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pvss/v2/contratos-servicios/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListContratosOpcionales(_FreematicaTool):
    """Contract optional records."""

    name: str = "freematica_list_contratos_opcionales"
    description: str | None = "List contract optional records (CON2_*)."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/ppre/v2/contratos/opcionales", params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetContratoOpcionales(_FreematicaTool):
    """One contract optional record."""

    name: str = "freematica_get_contrato_opcionales"
    description: str | None = "Get one contract optional record by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/ppre/v2/contratos/opcionales/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)


class FreematicaListResultadosFacturacion(_FreematicaTool):
    """Results of the automatic billing batch."""

    name: str = "freematica_list_resultados_facturacion"
    description: str | None = "List results of the automatic service-billing batch."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(
                self, "GET", "/pvss/v1/facturacion-resultados", params=_list_params(page, items)
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaCreateContrato(_FreematicaTool):
    """Create a contract header. There is no delete."""

    name: str = "freematica_create_contrato"
    description: str | None = (
        "Create a contract header (POST /pvss/v2/contratos). "
        "Required by the API: delegacion, descripcion, fecha, cod_cliente."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            delegacion: str = Field(..., description="Branch (CTRT_DELEG)."),
            descripcion: str = Field(..., description="Description (CTRT_DES)."),
            fecha: str = Field(..., description="Date YYYY-MM-DD (CTRT_FECHA)."),
            cod_cliente: str = Field(..., description="Client code (CTRT_COD_CLI)."),
            empresa: str | None = Field(None, description="Company (CTRT_EMP)."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra CTRT_* columns."),
        ) -> Any:
            body = _drop_none(
                {
                    "CTRT_DELEG": delegacion,
                    "CTRT_DES": descripcion,
                    "CTRT_FECHA": fecha,
                    "CTRT_COD_CLI": cod_cliente,
                    "CTRT_EMP": empresa,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(self, "POST", "/pvss/v2/contratos", body=body)

        super().__init__(handler=_create, **kwargs)


class FreematicaUpdateContrato(_FreematicaTool):
    """Update a contract header."""

    name: str = "freematica_update_contrato"
    description: str | None = "Update a contract header (PUT /pvss/v2/contratos/{idReg})."

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            idreg: str = Field(..., description="Contract idReg."),
            descripcion: str | None = Field(None, description="Description (CTRT_DES)."),
            fecha: str | None = Field(None, description="Date (CTRT_FECHA)."),
            cod_cliente: str | None = Field(None, description="Client code (CTRT_COD_CLI)."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra CTRT_* columns."),
        ) -> Any:
            body = _drop_none(
                {
                    "CTRT_DES": descripcion,
                    "CTRT_FECHA": fecha,
                    "CTRT_COD_CLI": cod_cliente,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(self, "PUT", f"/pvss/v2/contratos/{quote(idreg, safe='')}", body=body)

        super().__init__(handler=_update, **kwargs)


class FreematicaCreateServicioContrato(_FreematicaTool):
    """Create a service. Identity fields are derived from the contract idReg."""

    name: str = "freematica_create_servicio_contrato"
    description: str | None = (
        "Create a service on a contract. CTRTS_EMP, CTRTS_DELEG and CTRTS_CTRT are decoded from id_contrato."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            id_contrato: str = Field(..., description="Contract idReg (base64 EMP__DELEG__COD)."),
            descripcion: str | None = Field(None, description="Service description."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra CTRTS_* columns."),
        ) -> Any:
            parts = decode_contrato_idreg(id_contrato)
            body = _drop_none(
                {
                    "CTRTS_EMP": parts["empresa"],
                    "CTRTS_DELEG": parts["delegacion"],
                    "CTRTS_CTRT": parts["cod_contrato"],
                    "CTRTS_DES": descripcion,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(
                self, "POST", f"/pvss/v2/contratos/{quote(id_contrato, safe='')}/servicios", body=body
            )

        super().__init__(handler=_create, **kwargs)


class FreematicaUpdateServicioFechas(_FreematicaTool):
    """Set a service start/end date. fecha_fin is how a service is closed; there is no delete."""

    name: str = "freematica_update_servicio_fechas"
    description: str | None = (
        "Update a service start and end date. Setting fecha_fin is the supported way to close a service."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            id_contrato: str = Field(..., description="Contract idReg."),
            id_servicio: str = Field(..., description="Service idReg."),
            fecha_alta: str | None = Field(None, description="Start date YYYY-MM-DD."),
            fecha_fin: str | None = Field(None, description="End date YYYY-MM-DD."),
        ) -> Any:
            body = _drop_none({"CTRTS_FCHALTA": fecha_alta, "CTRTS_FCHFIN": fecha_fin})
            path = f"/pvss/v2/contratos/{quote(id_contrato, safe='')}/servicio/{quote(id_servicio, safe='')}"
            return await _freematica_request(self, "PUT", path, body=body)

        super().__init__(handler=_update, **kwargs)


class FreematicaCreateServicioHistoricoPrecios(_FreematicaTool):
    """Create a versioned price row for a service."""

    name: str = "freematica_create_servicio_historico_precios"
    description: str | None = "Create a service price-history row (SERVHPR_*)."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            id_contrato: str = Field(..., description="Contract idReg."),
            id_servicio: str = Field(..., description="Service idReg."),
            campos: dict[str, Any] = Field(..., description="Native SERVHPR_* columns."),
        ) -> Any:
            path = (
                f"/pvss/v2/contratos/{quote(id_contrato, safe='')}"
                f"/servicios-historico-precios/{quote(id_servicio, safe='')}"
            )
            return await _freematica_request(self, "POST", path, body=campos)

        super().__init__(handler=_create, **kwargs)


class FreematicaUpdateServicioHistoricoPrecios(_FreematicaTool):
    """Update a service price-history row."""

    name: str = "freematica_update_servicio_historico_precios"
    description: str | None = "Update a service price-history row (SERVHPR_*)."

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            id_contrato: str = Field(..., description="Contract idReg."),
            id_servicio: str = Field(..., description="Service idReg."),
            campos: dict[str, Any] = Field(..., description="Native SERVHPR_* columns to send."),
        ) -> Any:
            path = (
                f"/pvss/v2/contratos/{quote(id_contrato, safe='')}"
                f"/servicios-historico-precios/{quote(id_servicio, safe='')}"
            )
            return await _freematica_request(self, "PUT", path, body=campos)

        super().__init__(handler=_update, **kwargs)


class FreematicaCreateServicioFacturacionTxt(_FreematicaTool):
    """Create an invoice text line on a service."""

    name: str = "freematica_create_servicio_facturacion_txt"
    description: str | None = "Create an invoice text line for a service (CTRTFL_*)."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            id_contrato: str = Field(..., description="Contract idReg."),
            id_servicio: str = Field(..., description="Service idReg."),
            texto: str = Field(..., description="Invoice text."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra CTRTFL_* columns."),
        ) -> Any:
            path = (
                f"/pvss/v2/contratos/{quote(id_contrato, safe='')}"
                f"/servicios-facturacion-txt/{quote(id_servicio, safe='')}"
            )
            body = _drop_none({"CTRTFL_TXT": texto, **(campos_adicionales or {})})
            return await _freematica_request(self, "POST", path, body=body)

        super().__init__(handler=_create, **kwargs)


class FreematicaUpdateServicioFacturacion(_FreematicaTool):
    """Update service billing data (hourly prices, payment form)."""

    name: str = "freematica_update_servicio_facturacion"
    description: str | None = "Update service billing data (CTRTF_*: hourly prices, payment form, amounts)."

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            id_contrato: str = Field(..., description="Contract idReg."),
            id_servicio: str = Field(..., description="Service idReg."),
            campos: dict[str, Any] = Field(..., description="Native CTRTF_* columns."),
        ) -> Any:
            path = (
                f"/pvss/v2/contratos/{quote(id_contrato, safe='')}/servicios-facturacion/{quote(id_servicio, safe='')}"
            )
            return await _freematica_request(self, "PUT", path, body=campos)

        super().__init__(handler=_update, **kwargs)


class FreematicaCreateContratoOpcionales(_FreematicaTool):
    """Create a contract optional record."""

    name: str = "freematica_create_contrato_opcionales"
    description: str | None = (
        "Create a contract optional record. Required: empresa, delegacion, tipo_contrato, num_contrato, fecha."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            empresa: str = Field(..., description="Company (CON2_CODEMP)."),
            delegacion: str = Field(..., description="Branch (CON2_DELEG)."),
            tipo_contrato: str = Field(..., description="Contract type (CON2_TIPOCONT)."),
            num_contrato: str = Field(..., description="Contract number (CON2_NUMCONT)."),
            fecha: str = Field(..., description="Date YYYY-MM-DD (CON2_FCHCONT)."),
            campos_adicionales: dict[str, Any] | None = Field(None, description="Extra CON2_* columns."),
        ) -> Any:
            body = _drop_none(
                {
                    "CON2_CODEMP": empresa,
                    "CON2_DELEG": delegacion,
                    "CON2_TIPOCONT": tipo_contrato,
                    "CON2_NUMCONT": num_contrato,
                    "CON2_FCHCONT": fecha,
                    **(campos_adicionales or {}),
                }
            )
            return await _freematica_request(self, "POST", "/ppre/v2/contratos/opcionales", body=body)

        super().__init__(handler=_create, **kwargs)


class FreematicaUpdateContratoOpcionales(_FreematicaTool):
    """Update a contract optional record."""

    name: str = "freematica_update_contrato_opcionales"
    description: str | None = "Update a contract optional record (CON2_*) by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            idreg: str = Field(..., description="Optional-record idReg."),
            campos: dict[str, Any] = Field(..., description="Native CON2_* columns to send."),
        ) -> Any:
            return await _freematica_request(
                self, "PUT", f"/ppre/v2/contratos/opcionales/{quote(idreg, safe='')}", body=campos
            )

        super().__init__(handler=_update, **kwargs)
