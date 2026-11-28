"""SAP integration MCP transports restricted to a deployment-owned tool manifest."""

from typing import Literal

from mcp import types
from pydantic import BaseModel, Field, model_validator

from ...core.mcp import MCPServer, MCPTool
from ._access import Ref, Route, SAPAccess
from ._http import https_endpoint

_ROUTES: dict[str, Route] = {
    "hana-ai": "hana_toolkit",
    "gateway": "sap_mcp_gateway",
    "third-party-hana": "third_party_hana",
    "third-party-api": "third_party_api",
    "third-party-cds": "third_party_cds",
}


class SAPMCPBinding(BaseModel, frozen=True):
    read_only: bool
    document_types: tuple[Ref, ...]
    input_model: type[BaseModel] | None = Field(default=None, exclude=True)


class SAPMCPServer(MCPServer):
    """Use one server instance per tenant and authenticated principal."""

    access: SAPAccess = Field(exclude=True)
    env: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    headers: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)

    follow_redirects: Literal[False] = False

    kind: Literal["hana-ai", "gateway", "third-party-hana", "third-party-api", "third-party-cds"]
    manifest: dict[Ref, SAPMCPBinding] = Field(min_length=1)
    timeout: float = Field(default=60, gt=0)
    connect_timeout: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def _check_sap_transport(self):
        if self.service or (self.transport == "stdio" and self.kind != "hana-ai"):
            raise ValueError("Use authenticated HTTPS, or stdio for the local SAP hana-ai server.")
        if self.transport == "http":
            https_endpoint(self.url or "")
            if self.auth is None and not any(
                k.lower() == "authorization" and v.strip() for k, v in self.headers.items()
            ):
                raise ValueError("Configure authentication for the SAP MCP server.")
        if self.kind in ("hana-ai", "third-party-hana", "third-party-cds") and any(
            not op.read_only for op in self.manifest.values()
        ):
            raise ValueError("The SAP HANA and CDS MCP manifests require read-only tools.")
        if self.kind.startswith("third-party") and any(op.input_model is None for op in self.manifest.values()):
            raise ValueError("Each third-party MCP operation needs a deployment-owned input model.")
        if any(op.read_only and op.document_types for op in self.manifest.values()):
            raise ValueError("A read-only MCP tool cannot create SAP documents.")
        return self

    def _make_tool(self, mcp_tool: types.Tool) -> MCPTool:
        tool = super()._make_tool(mcp_tool)
        binding = self.manifest.get(mcp_tool.name)
        if binding is None:
            return tool
        if binding.read_only and mcp_tool.annotations and mcp_tool.annotations.readOnlyHint is False:
            raise ValueError(
                f"SAP MCP server declares {mcp_tool.name} writable but the manifest declares it read-only."
            )
        dispatch = tool.handler

        async def invoke(**arguments):
            async def run(call):
                if self.manifest.get(mcp_tool.name) != binding:
                    raise PermissionError("Reconnect the SAP MCP server after changing its tool manifest.")
                payload = arguments
                if binding.input_model is not None:
                    model = binding.input_model.model_validate(arguments)
                    payload = model.model_dump(mode="json", by_alias=True, exclude_unset=True)
                call.account_input(payload)
                return await dispatch(**payload)

            return await self.access.invoke(
                tool.name,
                mcp_tool.name,
                run,
                route=_ROUTES[self.kind],
                document_types=None if binding.read_only else binding.document_types,
            )

        changes = {"handler": invoke, "read_only": binding.read_only}
        if binding.input_model is not None:
            changes["input_schema"] = binding.input_model.model_json_schema()
        bound = tool.model_copy(update=changes)
        # SAP inputs and results stay out of the host's event logs.
        bound._log_events = set()
        return bound

    async def resolve(self):
        tools = await super().resolve()
        allowed = {self._qualified_tool_name(name) for name in self.manifest}
        return [tool for tool in tools if tool.name in allowed]
