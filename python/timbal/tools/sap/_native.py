"""Bindings for published native BAPIs and deterministic SAP GUI for Windows scripts."""

import json
import multiprocessing
import sys
import threading
from collections.abc import Callable
from contextlib import closing
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints, model_validator
from pydantic_core import to_json

from ._access import Ref, SAPCustomizing, SAPTool


class _NativeOperation(SAPTool):
    input_model: type[BaseModel] = Field(exclude=True)
    document_types: tuple[Ref, ...]


def _check_return(result: dict) -> None:
    messages = result.get("RETURN", [])
    if isinstance(messages, dict):
        messages = [messages]
    if any(m.get("TYPE") in ("E", "A", "X") for m in messages):
        raise RuntimeError("SAP BAPI reported an application error.")


class SAPBAPI(_NativeOperation):
    """One published BAPI using an authenticated RFC SDK connection owned by the deployment."""

    name: str = "sap_bapi"
    description: str = "Call the registered published SAP BAPI with its declared input fields."
    function: Ref
    read_only: bool
    commit: bool = False
    customizing: SAPCustomizing | None = None
    transport_parameter: Ref | None = None
    connect: Callable[[], Any] = Field(exclude=True)

    @model_validator(mode="after")
    def _check_commit(self):
        if self.read_only and self.commit:
            raise ValueError("A read-only BAPI cannot request a transaction commit.")
        if self.customizing and not self.transport_parameter:
            raise ValueError("Configure the published BAPI transport parameter for customizing.")
        return self

    async def _invoke(self, payload: BaseModel):
        params = payload.model_dump(by_alias=True, exclude_unset=True)

        async def run(call):
            if self.customizing:
                params[self.transport_parameter] = self.customizing.transport_request
            call.account_input(params)

            def invoke():
                call.check_active()
                with closing(self.connect()) as connection:
                    if self.customizing:
                        attrs = connection.get_connection_attributes()
                        if (attrs["sysId"], attrs["client"]) != (
                            self.customizing.nonproduction_system,
                            self.customizing.client,
                        ):
                            raise PermissionError(
                                "BAPI destination differs from the designated non-production system/client."
                            )
                    call.check_active()
                    result = connection.call(self.function, **params)
                    _check_return(result)
                    if self.commit:
                        call.check_active()
                        self.access._charge(call.consumer, start=False)
                        _check_return(connection.call("BAPI_TRANSACTION_COMMIT", WAIT="X"))
                    return result

            return await call.blocking(invoke)

        return await self.access.invoke(
            self.name,
            self.function,
            run,
            route="A(ii)",
            agent_route="B(ii)",
            document_types=self.document_types,
            customizing=self.customizing,
        )


_gui_lock = threading.Lock()


def _gui_worker(pipe, target, script, payload, transport, max_bytes):
    """Spawned Windows process. Only trusted, importable deployment code is passed in."""
    try:
        import pythoncom
        from win32com.client import GetObject

        pythoncom.CoInitialize()
        try:
            engine = GetObject("SAPGUI").GetScriptingEngine
            matches = [
                session
                for connection in engine.Connections
                for session in connection.Sessions
                if (session.Info.SystemName, session.Info.Client, session.Info.User) == target
            ]
            if len(matches) != 1:
                pipe.send_bytes(b"EExpected exactly one authenticated SAP GUI session matching the designation.")
                return
            data = to_json(script(matches[0], payload, transport), bytes_mode="base64")
            if len(data) > max_bytes:
                pipe.send_bytes(b"ESAP GUI result exceeds the configured byte limit.")
                return
            pipe.send_bytes(b"S" + data)
        finally:
            pythoncom.CoUninitialize()
    except ImportError:
        pipe.send_bytes(b"ESAP GUI Scripting requires SAP GUI for Windows and pywin32.")
    except Exception:
        pipe.send_bytes(b"ESAP GUI script failed.")
    finally:
        pipe.close()


class SAPGUIScript(_NativeOperation):
    """Run installed deterministic Python scripting code in one designated SAP GUI session."""

    name: str = "sap_gui_script"
    description: str = "Run a preconfigured SAP GUI script in its designated non-production client."
    nonproduction_system: Ref
    client: Annotated[str, StringConstraints(pattern=r"^\d{3}$")]
    user: Ref = Field(exclude=True)
    script: Callable[[Any, BaseModel, str | None], Any] = Field(exclude=True)
    customizing: SAPCustomizing | None = None

    @model_validator(mode="after")
    def _check_target(self):
        if self.customizing and (self.nonproduction_system, self.client) != (
            self.customizing.nonproduction_system,
            self.customizing.client,
        ):
            raise ValueError("SAP GUI session must match the designated customizing system and client.")
        return self

    async def _invoke(self, payload: BaseModel):
        def execute(call):
            if sys.platform != "win32":
                raise RuntimeError("SAP GUI Scripting requires a Windows worker.")
            if not _gui_lock.acquire(blocking=False):
                raise RuntimeError("The SAP GUI scripting worker is busy.")
            try:
                ctx = multiprocessing.get_context("spawn")
                reader, writer = ctx.Pipe(duplex=False)
                target = (self.nonproduction_system, self.client, self.user)
                transport = self.customizing.transport_request if self.customizing else None
                child = ctx.Process(
                    target=_gui_worker, args=(writer, target, self.script, payload, transport, self.access.max_bytes)
                )
                try:
                    call.check_active()
                    child.start()
                    writer.close()
                    while not reader.poll(0.05):
                        call.check_active()
                    data = reader.recv_bytes(maxlength=max(256, self.access.max_bytes + 1))
                    if not data.startswith(b"S"):
                        raise RuntimeError(data[1:].decode())
                    return json.loads(data[1:])
                finally:
                    if child.is_alive():
                        child.terminate()
                    if child.pid is not None:
                        child.join()
                    child.close()
                    reader.close()
                    writer.close()
            finally:
                _gui_lock.release()

        async def run(call):
            call.account_input(payload)
            return await call.blocking(lambda: execute(call))

        return await self.access.invoke(
            self.name,
            f"{self.nonproduction_system}/{self.client}",
            run,
            route="B(v)",
            document_types=self.document_types,
            customizing=self.customizing,
        )
