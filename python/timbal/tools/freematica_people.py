"""Freemática tools: employees, work calendars, and occupational health (PRL).

Auth and request handling live in ``freematica.py``.
"""

from typing import Any
from urllib.parse import quote

from pydantic import Field

from .freematica import (
    _first_item,
    _freematica_request,
    _FreematicaTool,
    _items,
    _list_params,
    _page,
    build_fiql,
)


class FreematicaListPersonal(_FreematicaTool):
    """Paginated employees. Filters are exact matches; partial search is not supported."""

    name: str = "freematica_list_personal"
    description: str | None = (
        "List employees from GET /pers/v1/personal (the v2 sync endpoint is not used). "
        "Filters are exact. Use idReg with freematica_get_persona."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            empresa: str | None = Field(None, description="Company (VSSPER_EMP)."),
            delegacion: str | None = Field(None, description="Branch (VSSPER_DELEG)."),
            cod_persona: str | None = Field(None, description="Person code (VSSPER_COD)."),
            nombre: str | None = Field(None, description="Exact first name (VSSPER_NOM)."),
            apellido: str | None = Field(None, description="Exact surname (VSSPER_APELL1)."),
            nif: str | None = Field(None, description="Tax id (VSSPER_NIF)."),
            situacion: str | None = Field(None, description="Status (VSSPER_SIT)."),
            departamento: str | None = Field(None, description="Department (VSSPER_DPTO)."),
            seccion: str | None = Field(None, description="Section (VSSPER_SECCION)."),
        ) -> Any:
            fiql = build_fiql(
                {
                    "VSSPER_EMP": empresa,
                    "VSSPER_DELEG": delegacion,
                    "VSSPER_COD": cod_persona,
                    "VSSPER_NOM": nombre,
                    "VSSPER_APELL1": apellido,
                    "VSSPER_NIF": nif,
                    "VSSPER_SIT": situacion,
                    "VSSPER_DPTO": departamento,
                    "VSSPER_SECCION": seccion,
                }
            )
            return await _freematica_request(self, "GET", "/pers/v1/personal", params=_list_params(page, items, fiql))

        super().__init__(handler=_list, **kwargs)


class FreematicaGetPersona(_FreematicaTool):
    """Employee detail. The API returns a one-item list, which is unwrapped."""

    name: str = "freematica_get_persona"
    description: str | None = "Get one employee. idreg is the opaque idReg from freematica_list_personal."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            result = await _freematica_request(self, "GET", f"/pers/v1/personal/{quote(idreg, safe='')}")
            return _first_item(result, "Employee", idreg)

        super().__init__(handler=_get, **kwargs)


class FreematicaListCalendarios(_FreematicaTool):
    """Paginated work calendars."""

    name: str = "freematica_list_calendarios"
    description: str | None = "List work calendars."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(page: int = _page(), items: int = _items()) -> Any:
            return await _freematica_request(self, "GET", "/pgrl/v1/calendarios", params=_list_params(page, items))

        super().__init__(handler=_list, **kwargs)


class FreematicaListCalendarioPeriodos(_FreematicaTool):
    """Periods of one work calendar."""

    name: str = "freematica_list_calendario_periodos"
    description: str | None = "List periods of a work calendar. idreg is the calendar idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            idreg: str = Field(..., description="Opaque calendar idReg."),
            page: int = _page(),
            items: int = _items(),
        ) -> Any:
            return await _freematica_request(
                self,
                "GET",
                f"/pgrl/v1/calendarios/{quote(idreg, safe='')}/periodos",
                params=_list_params(page, items),
            )

        super().__init__(handler=_list, **kwargs)


class FreematicaGetFichaPrevCliente(_FreematicaTool):
    """Client occupational-health file. Filters are native query params, not FIQL."""

    name: str = "freematica_get_ficha_prev_cliente"
    description: str | None = (
        "Get a client's PRL file (GET /pprl/v2/ficha-prev-cliente). "
        "At least one of cod_cliente, grupo_cliente, cod_localizacion_servicio, codigo_ficha is required."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get(
            cod_cliente: str | None = Field(None, description="Client code (codCli)."),
            grupo_cliente: int | None = Field(None, description="Client group (grpCli)."),
            cod_localizacion_servicio: int | None = Field(None, description="Service location (locServ)."),
            codigo_ficha: str | None = Field(None, description="File code (codigo)."),
        ) -> Any:
            if all(value is None for value in (cod_cliente, grupo_cliente, cod_localizacion_servicio, codigo_ficha)):
                raise ValueError("Provide at least one filter.")
            return await _freematica_request(
                self,
                "GET",
                "/pprl/v2/ficha-prev-cliente",
                params={
                    "codCli": cod_cliente,
                    "grpCli": grupo_cliente,
                    "locServ": cod_localizacion_servicio,
                    "codigo": codigo_ficha,
                },
            )

        super().__init__(handler=_get, **kwargs)


class FreematicaListVigilanciaSalud(_FreematicaTool):
    """Health-surveillance records. FIQL is ignored by this endpoint; only idRegPersona filters."""

    name: str = "freematica_list_vigilancia_salud"
    description: str | None = (
        "List occupational health-surveillance records. The only working filter is idreg_persona "
        "(the employee idReg); FIQL is ignored by this endpoint."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list(
            page: int = _page(),
            items: int = _items(),
            idreg_persona: str | None = Field(None, description="Employee idReg (native idRegPersona)."),
        ) -> Any:
            params = _list_params(page, items)
            if idreg_persona is not None:
                params["idRegPersona"] = idreg_persona
            return await _freematica_request(self, "GET", "/pprl/v1/vigilancia-salud", params=params)

        super().__init__(handler=_list, **kwargs)


class FreematicaGetVigilanciaSalud(_FreematicaTool):
    """One health-surveillance record."""

    name: str = "freematica_get_vigilancia_salud"
    description: str | None = "Get one health-surveillance record by opaque idReg."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(idreg: str = Field(..., description="Opaque idReg.")) -> Any:
            return await _freematica_request(self, "GET", f"/pprl/v1/vigilancia-salud/{quote(idreg, safe='')}")

        super().__init__(handler=_get, **kwargs)
