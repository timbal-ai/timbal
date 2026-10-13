"""Microsoft Fabric REST API tools.

Auth (first match wins):

- **User delegated (OAuth)**: ``token`` / ``access_token`` from
  ``Integration("fabric")``, tool field ``token``, or env ``FABRIC_ACCESS_TOKEN``.
- **App delegated (service principal)**: ``api_key`` as a Bearer token, or Azure AD
  client credentials (``tenant_id`` + ``client_id`` + ``client_secret``) with scope
  ``https://api.fabric.microsoft.com/.default``. Also accepts ``FABRIC_API_KEY`` or
  ``FABRIC_TENANT_ID`` / ``FABRIC_CLIENT_ID`` / ``FABRIC_CLIENT_SECRET``.

Every Fabric item type shares the generic Items API (``fabric_list_items``,
``fabric_create_item``, ``fabric_get_item_definition``, ...), so per-type CRUD endpoints
(lakehouses, notebooks, warehouses, ...) are reached through it with the ``type`` field.
The type-specific tools cover everything else (tables, jobs, mirroring, spark, ...) and live in
the sibling ``fabric_*.py`` modules, which reuse the auth and request helpers defined here.

Long running operations (HTTP 202) are polled until they finish, up to
``_LRO_MAX_WAIT_SECONDS``; if they are still running, the operation handle is returned so it
can be followed with ``fabric_get_operation_state`` / ``fabric_get_operation_result``. Job runs
return immediately with the job instance id (jobs can take hours).
"""

import asyncio
import base64
import os
import re
from typing import Annotated, Any
from urllib.parse import quote, urlparse

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration

_BASE_URL = "https://api.fabric.microsoft.com/v1"
_FABRIC_HOST = "api.fabric.microsoft.com"
_FABRIC_SCOPE = "https://api.fabric.microsoft.com/.default"
_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

_LRO_MAX_WAIT_SECONDS = 90.0
_LRO_DEFAULT_POLL_SECONDS = 2.0
_LRO_MAX_POLL_SECONDS = 10.0
_MAX_RATE_LIMIT_RETRIES = 3
_MAX_RATE_LIMIT_WAIT_SECONDS = 30.0
_JOB_INSTANCE_URL = re.compile(r"/jobs/[^?]*instances/")
_OPERATION_URL = re.compile(r"/v1/operations/[^/]+$")


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    text = str(value).strip()
    return text or None


async def _get_token_from_client_credentials(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Obtain an app-only token via the Azure AD client-credentials flow."""
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(
            _TOKEN_URL.format(tenant_id=tenant_id),
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": _FABRIC_SCOPE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError("Fabric service principal token response did not include access_token.")
        return str(token)


def _service_principal_parts(tool: Any, creds: dict[str, Any]) -> tuple[str, str, str] | None:
    tenant_id = (
        _secret_value(creds.get("tenant_id"))
        or _secret_value(getattr(tool, "tenant_id", None))
        or _secret_value(os.getenv("FABRIC_TENANT_ID"))
    )
    client_id = (
        _secret_value(creds.get("client_id"))
        or _secret_value(getattr(tool, "client_id", None))
        or _secret_value(os.getenv("FABRIC_CLIENT_ID"))
    )
    client_secret = (
        _secret_value(creds.get("client_secret"))
        or _secret_value(getattr(tool, "client_secret", None))
        or _secret_value(os.getenv("FABRIC_CLIENT_SECRET"))
    )
    if tenant_id and client_id and client_secret:
        return tenant_id, client_id, client_secret
    return None


async def _resolve_token(tool: Any) -> str:
    """Return a Fabric REST Bearer token (user-delegated OAuth or app-delegated)."""
    explicit_key = _secret_value(getattr(tool, "api_key", None))
    if explicit_key:
        return explicit_key
    explicit_token = _secret_value(getattr(tool, "token", None))
    if explicit_token:
        return explicit_token

    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    oauth = _secret_value(creds.get("token")) or _secret_value(creds.get("access_token"))
    if oauth:
        return oauth

    api_key = _secret_value(creds.get("api_key"))
    if api_key:
        return api_key

    sp = _service_principal_parts(tool, creds)
    if sp:
        return await _get_token_from_client_credentials(*sp)

    env_key = _secret_value(os.getenv("FABRIC_API_KEY"))
    if env_key:
        return env_key
    env_oauth = _secret_value(os.getenv("FABRIC_ACCESS_TOKEN"))
    if env_oauth:
        return env_oauth

    raise CredentialNotAvailable(
        "Fabric",
        missing=["token", "api_key"],
        env_vars=[
            "FABRIC_ACCESS_TOKEN",
            "FABRIC_API_KEY",
            "FABRIC_TENANT_ID",
            "FABRIC_CLIENT_ID",
            "FABRIC_CLIENT_SECRET",
        ],
    )


def _quote(value: Any) -> str:
    return quote(str(value), safe="")


def _quote_path(value: Any) -> str:
    """Quote a multi-segment path parameter, keeping its slashes."""
    return quote(str(value), safe="/")


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    cleaned: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        cleaned[key] = value
    return cleaned or None


def _upload_bytes(content: str | None, content_base64: str | None) -> bytes:
    if (content is None) == (content_base64 is None):
        raise ValueError("Provide exactly one of content (UTF-8 text) or content_base64 (binary).")
    if content_base64 is not None:
        return base64.b64decode(content_base64)
    return str(content).encode("utf-8")


def _retry_after(response: Any, default: float) -> float:
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return default


def _raise_for_fabric(response: Any) -> None:
    """Raise a ValueError carrying the Fabric error code/message instead of a bare HTTP status."""
    status = response.status_code
    if status < 400:
        return
    detail = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else {}
        code = body.get("errorCode") or error.get("code")
        message = body.get("message") or error.get("message")
        detail = ": ".join(str(part) for part in (code, message) if part)
        if body.get("requestId"):
            detail += f" (requestId {body['requestId']})"
    if not detail:
        detail = (getattr(response, "text", "") or "")[:300]
    hint = ""
    if status in (401, 403):
        hint = " Check that the OAuth token is valid and grants the Fabric delegated scopes this API requires."
    raise ValueError(f"Fabric API returned {status}: {detail}.{hint}".replace("..", "."))


def _parse_response(response: Any) -> Any:
    if response.status_code == 204 or not response.content:
        return {"status": "success", "status_code": response.status_code}
    content_type = response.headers.get("content-type", "")
    if "json" in content_type or not content_type:
        try:
            return response.json()
        except ValueError:
            pass
    raw = response.content
    payload: dict[str, Any] = {
        "content_type": content_type,
        "size": len(raw),
        "content_base64": base64.b64encode(raw).decode("ascii"),
    }
    if content_type.startswith("text/") or any(s in content_type for s in ("json", "xml", "yaml", "csv")):
        try:
            payload["content"] = raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return payload


async def _send(
    client: Any,
    method: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    body: Any = None,
    content: bytes | None = None,
    content_type: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {"headers": headers, "params": params}
    if body is not None:
        kwargs["json"] = body
    elif content is not None:
        kwargs["content"] = content
        kwargs["headers"] = {**headers, "Content-Type": content_type or "application/octet-stream"}
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        response = await client.request(method, url, **kwargs)
        if response.status_code != 429 or attempt == _MAX_RATE_LIMIT_RETRIES:
            break
        await asyncio.sleep(min(_retry_after(response, 2.0), _MAX_RATE_LIMIT_WAIT_SECONDS))
    _raise_for_fabric(response)
    return response


async def _wait_for_operation(client: Any, response: Any, headers: dict[str, str]) -> Any:
    """Follow a 202 Accepted response until it completes (or hand back the operation handle)."""
    location = response.headers.get("Location")
    operation_id = response.headers.get("x-ms-operation-id")
    retry_after = _retry_after(response, _LRO_DEFAULT_POLL_SECONDS)
    accepted: dict[str, Any] = {
        "status": "Accepted",
        "status_code": 202,
        "operation_id": operation_id,
        "location": location,
        "retry_after_seconds": retry_after,
    }
    if not location or urlparse(location).hostname != _FABRIC_HOST:
        return accepted
    if _JOB_INSTANCE_URL.search(urlparse(location).path):
        accepted["job_instance_id"] = urlparse(location).path.rstrip("/").rsplit("/", 1)[-1]
        return accepted

    waited = 0.0
    while waited < _LRO_MAX_WAIT_SECONDS:
        delay = min(max(retry_after, 1.0), _LRO_MAX_POLL_SECONDS)
        await asyncio.sleep(delay)
        waited += delay
        state_response = await _send(client, "GET", location, headers)
        state = _parse_response(state_response)
        status = str(state.get("status", "")).lower() if isinstance(state, dict) else ""
        if status in ("succeeded", "completed"):
            if not _OPERATION_URL.search(urlparse(location).path):
                return state
            try:
                result_response = await _send(client, "GET", f"{location}/result", headers)
            except ValueError:
                return state
            return _parse_response(result_response)
        if status in ("failed", "canceled", "cancelled"):
            error = state.get("error") if isinstance(state, dict) else None
            raise ValueError(f"Fabric operation {operation_id or location} {state.get('status')}: {error or state}")
        retry_after = _retry_after(state_response, retry_after)

    accepted["status"] = "Running"
    accepted["message"] = "Still running. Follow it with fabric_get_operation_state and fabric_get_operation_result."
    return accepted


async def _fabric_request(
    tool: Any,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: Any = None,
    content: bytes | None = None,
    content_type: str | None = None,
) -> Any:
    """Call a Fabric REST endpoint and return the parsed response."""
    token = await _resolve_token(tool)
    import httpx

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        response = await _send(
            client, method, f"{_BASE_URL}{path}", headers, _clean_params(params), body, content, content_type
        )
        if response.status_code == 202:
            return await _wait_for_operation(client, response, headers)
        return _parse_response(response)


class _FabricTool(Tool):
    """Shared auth fields for Microsoft Fabric tools (OAuth user-delegated + service principal)."""

    integration: Annotated[str, Integration("fabric")] | None = None
    api_key: SecretStr | None = None
    token: SecretStr | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_key": self.api_key,
                    "token": self.token,
                    "tenant_id": self.tenant_id,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            ),
        }


class FabricListCapacities(_FabricTool):
    name: str = "fabric_list_capacities"
    description: str | None = (
        "Returns a list of capacities the principal can access (either administrator or a contributor)."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_capacities(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/capacities",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_capacities, **kwargs)


class FabricGetCapacity(_FabricTool):
    name: str = "fabric_get_capacity"
    description: str | None = "Returns specified capacity information."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_capacity(
            capacity_id: str = Field(..., description="The capacity ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/capacities/{capacity_id}",
            )

        super().__init__(handler=_get_capacity, **kwargs)


class FabricGetCapacitySurgeProtection(_FabricTool):
    name: str = "fabric_get_capacity_surge_protection"
    description: str | None = (
        "Returns the surge protection configuration for the specified capacity. Surge protection lets a capacity "
        "administrator set utilization thresholds that proactively reject background operations to prevent the "
        "capacity from entering deep throttling states. A capacity with no configured surge protection returns "
        "`state` set to `Disabled` with the threshold properties omitted."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_capacity_surge_protection(
            capacity_id: str = Field(..., description="The capacity ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/capacities/{capacity_id}/surgeProtection",
            )

        super().__init__(handler=_get_capacity_surge_protection, **kwargs)


class FabricUpdateCapacitySurgeProtection(_FabricTool):
    name: str = "fabric_update_capacity_surge_protection"
    description: str | None = "Updates the surge protection configuration for the specified capacity."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_capacity_surge_protection(
            capacity_id: str = Field(..., description="The capacity ID."),
            state: str | None = Field(
                None,
                description=(
                    "The state of surge protection on a capacity. Additional `SurgeProtectionState` values may be added "
                    "over time. Allowed values: Enabled, Disabled."
                ),
            ),
            rejection_threshold: int | None = Field(
                None,
                description=(
                    "The 24-hour background utilization percentage at which background operations are rejected. Valid "
                    "range: 15-100."
                ),
            ),
            recovery_threshold: int | None = Field(
                None,
                description=(
                    "The 24-hour background utilization percentage at which surge protection deactivates. Must be greater "
                    "than 5 and strictly less than `rejectionThreshold`."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/capacities/{capacity_id}/surgeProtection",
                body=_drop_none(
                    {"state": state, "rejectionThreshold": rejection_threshold, "recoveryThreshold": recovery_threshold}
                ),
            )

        super().__init__(handler=_update_capacity_surge_protection, **kwargs)


class FabricSearchCatalog(_FabricTool):
    name: str = "fabric_search_catalog"
    description: str | None = (
        "The Catalog Search API enables programmatic discovery of OneLake catalog entries across workspaces. It "
        "supports cross-workspace search over catalog metadata and returns results filtered to entries the calling "
        "principal is authorized to access. Search results include stable identifiers that are intended to be used "
        "with complementary Fabric APIs to retrieve additional details or perform supported actions. Preview API: may "
        "change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _search_catalog(
            search: str | None = Field(
                None,
                description=(
                    "The text query for the search. This field supports searching across the display name, workspace "
                    "display name, and description of the CatalogEntry. The search field supports the following "
                    'operators: * " " : Double quotes; match the enclosed words as an exact phrase. * \\* : Asterisk; '
                    "wildcard matching zero or more characters. * ? : Question mark; wildcard matching a single "
                    "character. * && : Double ampersands; require every search term. * Underscores are treated as part of "
                    "a search term; no escaping is required. * Other special characters are ignored and treated as "
                    "separators."
                ),
            ),
            page_size: int | None = Field(
                None,
                description=(
                    "The page size that needs to be returned. Page size must be between 1 and 1000. Defaults to 50."
                ),
            ),
            filter: str | None = Field(
                None,
                description=(
                    "The filter for the search. Additional filter options may be added over time. The filter parameter "
                    "supports the following properties: * **Type**: Matches the `type` property of a catalog entry, for "
                    "example `Report`, `Lakehouse`, or `Workspace`. A single filter can specify at most 500 Type values, "
                    "each up to 50 characters long. * **WorkspaceId**: The ID of the workspace that contains the catalog "
                    "entry. A single filter can specify at most 12 WorkspaceId values, each of which must be a valid "
                    "GUID. The filter parameter supports the following operators to refine results: * eq: Equals; matches "
                    "the exact value. * ne: Not Equals; excludes the specified value. * and: Logical AND; matches only if "
                    "all of the conditions are true. * or: Logical OR; matches if any of the conditions are true. * ( ): "
                    "Parentheses; groups expressions to define logical hierarchy."
                ),
            ),
            continuation_token: str | None = Field(
                None,
                description=(
                    "The continuation token for the next page. A continuation token carries forward the `search`, "
                    "`filter`, and `pageSize` of the original request, so it may be sent on its own. It must not be "
                    "combined with `search` or `filter` in the same request; doing so returns an error."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/catalog/search",
                body=_drop_none(
                    {"search": search, "pageSize": page_size, "filter": filter, "continuationToken": continuation_token}
                ),
            )

        super().__init__(handler=_search_catalog, **kwargs)


class FabricListDomains(_FabricTool):
    name: str = "fabric_list_domains"
    description: str | None = "Returns a list of all the tenant's domains."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_domains(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/domains",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_domains, **kwargs)


class FabricGetDomain(_FabricTool):
    name: str = "fabric_get_domain"
    description: str | None = "Returns specified domain information."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_domain(
            domain_id: str = Field(..., description="The domain ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/domains/{domain_id}",
            )

        super().__init__(handler=_get_domain, **kwargs)


class FabricGetOperationState(_FabricTool):
    name: str = "fabric_get_operation_state"
    description: str | None = (
        "Returns the current state of the long running operation. You get the operationId from x-ms-operation-id "
        "header return by the API that initiated the operation. Once the operation status is 'Succeeded' use the Get "
        "Operation Result API to retrieve the result."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_operation_state(
            operation_id: str = Field(..., description="The operation ID"),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/operations/{operation_id}",
            )

        super().__init__(handler=_get_operation_state, **kwargs)


class FabricGetOperationResult(_FabricTool):
    name: str = "fabric_get_operation_result"
    description: str | None = (
        "Returns the result of the long running operation. You get the operationId from x-ms-operation-id header "
        "return by the API that initiated the operation."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_operation_result(
            operation_id: str = Field(..., description="The operation ID"),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/operations/{operation_id}/result",
            )

        super().__init__(handler=_get_operation_result, **kwargs)


class FabricListTags(_FabricTool):
    name: str = "fabric_list_tags"
    description: str | None = "Returns a list of all the tenant's tags."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tags(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/tags",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_tags, **kwargs)


class FabricListWorkspaces(_FabricTool):
    name: str = "fabric_list_workspaces"
    description: str | None = (
        "Returns a list of workspaces the principal can access. Use the roles query parameter to filter results by "
        "the principal workspace role."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspaces(
            roles: str | None = Field(
                None,
                description=(
                    "A list of roles. Separate values using a comma. If not provided, all workspaces are returned."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            prefer_workspace_specific_endpoints: bool | None = Field(
                None,
                description=(
                    "A setting that controls whether to include the workspace-specific API endpoint per workspace. True - "
                    "Include the workspace-specific API endpoint, False - Do not include the workspace-specific API "
                    "endpoint."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/workspaces",
                params={
                    "roles": roles,
                    "continuationToken": continuation_token,
                    "preferWorkspaceSpecificEndpoints": prefer_workspace_specific_endpoints,
                },
            )

        super().__init__(handler=_list_workspaces, **kwargs)


class FabricCreateWorkspace(_FabricTool):
    name: str = "fabric_create_workspace"
    description: str | None = "Creates a new workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_workspace(
            display_name: str = Field(
                ...,
                description=(
                    "The workspace display name.<br>The display name cannot contain more than 256 characters.<br>Only "
                    'unused workspace names are allowed.<br>"Admin monitoring" is a reserved workspace name.'
                ),
            ),
            description: str | None = Field(
                None,
                description="The workspace description.<br>The description cannot contain more than 4000 characters.",
            ),
            capacity_id: str | None = Field(None, description="The ID of the capacity to assign the workspace to."),
            domain_id: str | None = Field(None, description="The ID of the domain to assign the workspace to."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/workspaces",
                body=_drop_none(
                    {
                        "displayName": display_name,
                        "description": description,
                        "capacityId": capacity_id,
                        "domainId": domain_id,
                    }
                ),
            )

        super().__init__(handler=_create_workspace, **kwargs)


class FabricDeleteWorkspace(_FabricTool):
    name: str = "fabric_delete_workspace"
    description: str | None = "Deletes the specified workspace and the items under it."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_workspace(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}",
            )

        super().__init__(handler=_delete_workspace, **kwargs)


class FabricGetWorkspace(_FabricTool):
    name: str = "fabric_get_workspace"
    description: str | None = "Returns specified workspace information."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace(
            workspace_id: str = Field(..., description="The workspace ID."),
            prefer_workspace_specific_endpoints: bool | None = Field(
                None,
                description=(
                    "A setting that controls whether to include workspace-specific or general public endpoints for API "
                    "and OneLake access. True - Include workspace-specific endpoints for API and OneLake access, False - "
                    "Include general public endpoints for OneLake access."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}",
                params={"preferWorkspaceSpecificEndpoints": prefer_workspace_specific_endpoints},
            )

        super().__init__(handler=_get_workspace, **kwargs)


class FabricUpdateWorkspace(_FabricTool):
    name: str = "fabric_update_workspace"
    description: str | None = "Updates the specified workspace properties."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_workspace(
            workspace_id: str = Field(..., description="The workspace ID."),
            display_name: str | None = Field(
                None,
                description=(
                    "The workspace display name.<br>The display name cannot contain more than 256 "
                    'characters.<br>Workspace names must be unique within the tenant.<br>"Admin monitoring" is a reserved '
                    "workspace name."
                ),
            ),
            description: str | None = Field(
                None,
                description="The workspace description.<br>The description cannot contain more than 4000 characters.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}",
                body=_drop_none({"displayName": display_name, "description": description}),
            )

        super().__init__(handler=_update_workspace, **kwargs)


class FabricApplyWorkspaceTags(_FabricTool):
    name: str = "fabric_apply_workspace_tags"
    description: str | None = "Apply tags to a workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _apply_workspace_tags(
            workspace_id: str = Field(..., description="The workspace ID."),
            tags: list[Any] = Field(..., description="An array of tag IDs to apply."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/applyTags",
                body=_drop_none({"tags": tags}),
            )

        super().__init__(handler=_apply_workspace_tags, **kwargs)


class FabricAssignWorkspaceToCapacity(_FabricTool):
    name: str = "fabric_assign_workspace_to_capacity"
    description: str | None = "Assigns the specified workspace to the specified capacity."

    def __init__(self, **kwargs: Any) -> None:
        async def _assign_workspace_to_capacity(
            workspace_id: str = Field(..., description="The workspace ID."),
            capacity_id: str = Field(..., description="The ID of the capacity the workspace should be assigned to."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/assignToCapacity",
                body=_drop_none({"capacityId": capacity_id}),
            )

        super().__init__(handler=_assign_workspace_to_capacity, **kwargs)


class FabricAssignWorkspaceToDomain(_FabricTool):
    name: str = "fabric_assign_workspace_to_domain"
    description: str | None = "Assigns the specified workspace to the specified domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _assign_workspace_to_domain(
            workspace_id: str = Field(..., description="The workspace ID."),
            domain_id: str = Field(..., description="The ID of the domain the workspace should be assigned to."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/assignToDomain",
                body=_drop_none({"domainId": domain_id}),
            )

        super().__init__(handler=_assign_workspace_to_domain, **kwargs)


class FabricListFolders(_FabricTool):
    name: str = "fabric_list_folders"
    description: str | None = "Returns a list of folders from the specified workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_folders(
            workspace_id: str = Field(..., description="The workspace ID."),
            root_folder_id: str | None = Field(
                None,
                description=(
                    "This parameter allows users to filter folders based on a specific root folder. If not provided, the "
                    "workspace is used as the root folder."
                ),
            ),
            recursive: bool | None = Field(
                None,
                description=(
                    "Lists folders in a folder and its nested folders, or just a folder only. True - All folders in the "
                    "folder and its nested folders are listed, False - Only folders in the folder are listed. The default "
                    "value is true."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/folders",
                params={
                    "rootFolderId": root_folder_id,
                    "recursive": recursive,
                    "continuationToken": continuation_token,
                },
            )

        super().__init__(handler=_list_folders, **kwargs)


class FabricCreateFolder(_FabricTool):
    name: str = "fabric_create_folder"
    description: str | None = "Creates a folder in the specified workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_folder(
            workspace_id: str = Field(..., description="The workspace ID."),
            display_name: str = Field(
                ..., description="The folder display name. The name must meet Folder name requirements"
            ),
            parent_folder_id: str | None = Field(
                None,
                description=(
                    "The parent folder ID. If not specified or null, the folder is created with the workspace as its "
                    "parent folder."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/folders",
                body=_drop_none({"displayName": display_name, "parentFolderId": parent_folder_id}),
            )

        super().__init__(handler=_create_folder, **kwargs)


class FabricDeleteFolder(_FabricTool):
    name: str = "fabric_delete_folder"
    description: str | None = "Deletes the specified folder. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_folder(
            workspace_id: str = Field(..., description="The workspace ID."),
            folder_id: str = Field(..., description="The folder ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/folders/{folder_id}",
            )

        super().__init__(handler=_delete_folder, **kwargs)


class FabricGetFolder(_FabricTool):
    name: str = "fabric_get_folder"
    description: str | None = "Returns the properties of the specified folder. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_folder(
            workspace_id: str = Field(..., description="The workspace ID."),
            folder_id: str = Field(..., description="The folder ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/folders/{folder_id}",
            )

        super().__init__(handler=_get_folder, **kwargs)


class FabricUpdateFolder(_FabricTool):
    name: str = "fabric_update_folder"
    description: str | None = "Updates the properties of the specified folder. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_folder(
            workspace_id: str = Field(..., description="The workspace ID."),
            folder_id: str = Field(..., description="The folder ID."),
            display_name: str = Field(
                ..., description="The folder display name. The name must meet Folder name requirements"
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/folders/{folder_id}",
                body=_drop_none({"displayName": display_name}),
            )

        super().__init__(handler=_update_folder, **kwargs)


class FabricMoveFolder(_FabricTool):
    name: str = "fabric_move_folder"
    description: str | None = "Moves the specified folder within the same workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _move_folder(
            workspace_id: str = Field(..., description="The workspace ID."),
            folder_id: str = Field(..., description="The folder ID."),
            target_folder_id: str | None = Field(
                None,
                description="The destination folder ID. If not provided, the workspace is used as the destination folder.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/folders/{folder_id}/move",
                body=_drop_none({"targetFolderId": target_folder_id}),
            )

        super().__init__(handler=_move_folder, **kwargs)


class FabricGitCommit(_FabricTool):
    name: str = "fabric_git_commit"
    description: str | None = (
        "Commits the changes made in the workspace to the connected remote branch. To use this API, the caller's Git "
        "credentials must be configured using Update My Git Credentials API. You can use the Get My Git Credentials "
        "API to check the Git credentials configuration. You can choose to commit all changes, specific items, or "
        "specific files within items using the FileLevelSelective mode. To sync the workspace for the first time, use "
        "this API after the Connect and Initialize Connection APIs."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_commit(
            workspace_id: str = Field(..., description="The workspace ID."),
            mode: str = Field(
                ...,
                description=(
                    "Modes for the commit operation. Additional modes may be added over time. Allowed values: All, "
                    "Selective, FileLevelSelective."
                ),
            ),
            workspace_head: str | None = Field(
                None,
                description=(
                    "Full SHA hash that the workspace is synced to. The hash can be retrieved from the Git Status API."
                ),
            ),
            comment: str | None = Field(
                None,
                description=(
                    "Caller-free comment for this commit. Maximum length is 300 characters. If no comment is provided by "
                    "the caller, use the default Git provider comment."
                ),
            ),
            items: list[Any] | None = Field(
                None,
                description=(
                    "Specific items to commit. This is relevant only for the Selective commit mode. Mutually exclusive "
                    "with itemsWithFileSelection. The items can be retrieved from the Git Status API."
                ),
            ),
            items_with_file_selection: list[Any] | None = Field(
                None,
                description=(
                    "Items with per-file selection for the FileLevelSelective commit mode. Mutually exclusive with items. "
                    "Each entry specifies an item and optionally a list of file paths to commit. If selectedFiles is null "
                    "or empty for an item, all files for that item are committed."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/git/commitToGit",
                body=_drop_none(
                    {
                        "mode": mode,
                        "workspaceHead": workspace_head,
                        "comment": comment,
                        "items": items,
                        "itemsWithFileSelection": items_with_file_selection,
                    }
                ),
            )

        super().__init__(handler=_git_commit, **kwargs)


class FabricGitConnect(_FabricTool):
    name: str = "fabric_git_connect"
    description: str | None = (
        "Connect a specific workspace to a git repository and branch. This operation does not sync between the "
        "workspace and the connected branch. To complete the sync, use the Initialize Connection operation and follow "
        "with either the Commit To Git or the Update From Git operation. To get started with GitHub, see: Get started "
        "with Git integration. To get the connection ID, see Automate Git integration."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_connect(
            workspace_id: str = Field(..., description="The workspace ID."),
            git_provider_details: dict[str, Any] = Field(..., description="The Git provider details."),
            my_git_credentials: dict[str, Any] | None = Field(None, description="The Git credentials."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/git/connect",
                body=_drop_none({"gitProviderDetails": git_provider_details, "myGitCredentials": my_git_credentials}),
            )

        super().__init__(handler=_git_connect, **kwargs)


class FabricGitGetConnection(_FabricTool):
    name: str = "fabric_git_get_connection"
    description: str | None = "Returns git connection details for the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _git_get_connection(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/git/connection",
            )

        super().__init__(handler=_git_get_connection, **kwargs)


class FabricGitDisconnect(_FabricTool):
    name: str = "fabric_git_disconnect"
    description: str | None = "Disconnect a specific workspace from the Git repository and branch it is connected to."

    def __init__(self, **kwargs: Any) -> None:
        async def _git_disconnect(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/git/disconnect",
            )

        super().__init__(handler=_git_disconnect, **kwargs)


class FabricGitInitializeConnection(_FabricTool):
    name: str = "fabric_git_initialize_connection"
    description: str | None = (
        "Initialize a connection for a workspace that's connected to Git. To use this API, the caller's Git "
        "credentials must be configured using Update My Git Credentials API. You can use the Get My Git Credentials "
        "API to check the Git credentials configuration. This API should be called after a successful call to the "
        "Connect API. To complete a full sync of the workspace, use the Required Action operation to call the "
        "relevant sync operation, either Commit To Git or Update From Git."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_initialize_connection(
            workspace_id: str = Field(..., description="The workspace ID."),
            initialization_strategy: str | None = Field(
                None,
                description=(
                    "The strategy required for an initialization process when content exists on both the remote side and "
                    "the workspace side. Additional strategies may be added over time. Allowed values: None, "
                    "PreferRemote, PreferWorkspace."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/git/initializeConnection",
                body=_drop_none({"initializationStrategy": initialization_strategy}),
            )

        super().__init__(handler=_git_initialize_connection, **kwargs)


class FabricGitGetMyCredentials(_FabricTool):
    name: str = "fabric_git_get_my_credentials"
    description: str | None = (
        "Returns the user's Git credentials configuration details. Indicates how the user's credentials are obtained "
        "for accessing the relevant Git provider, automatically or through configured connection. If the user's "
        "credentials aren't configured, go to Update My Git Credentials API."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_get_my_credentials(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/git/myGitCredentials",
            )

        super().__init__(handler=_git_get_my_credentials, **kwargs)


class FabricGitUpdateMyCredentials(_FabricTool):
    name: str = "fabric_git_update_my_credentials"
    description: str | None = (
        "Updates the user's Git credentials configuration details. Each user in the workspace has their own "
        "configured Git credentials. You can use Get My Git Credentials API to get the Git credentials configuration. "
        "To get the connection ID, see Automate Git integration."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_update_my_credentials(
            workspace_id: str = Field(..., description="The workspace ID."),
            source: str = Field(
                ...,
                description=(
                    "The Git credentials source. Additional Git credentials sources may be added over time. Allowed "
                    "values: ConfiguredConnection, Automatic, None."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/git/myGitCredentials",
                body=_drop_none({"source": source}),
            )

        super().__init__(handler=_git_update_my_credentials, **kwargs)


class FabricGitGetStatus(_FabricTool):
    name: str = "fabric_git_get_status"
    description: str | None = (
        "Returns the `Git status` of items in the workspace, that can be committed to Git. The status indicates "
        "changes to items since the last workspace and remote branch sync. If the remote and workspace items were "
        "both modified, the API flags a conflict. The API should not be called while Update From Git operation is "
        "executing. To use this API, the caller's Git credentials must be configured using Update My Git Credentials "
        "API. You can use the Get My Git Credentials API to check the Git credentials configuration."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_get_status(
            workspace_id: str = Field(..., description="The workspace ID."),
            include_files_details: bool | None = Field(
                None,
                description=(
                    "When true, each item change in the response includes a fileChanges array with per-file change "
                    "details."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/git/status",
                params={"includeFilesDetails": include_files_details},
            )

        super().__init__(handler=_git_get_status, **kwargs)


class FabricGitUpdateFromGit(_FabricTool):
    name: str = "fabric_git_update_from_git"
    description: str | None = (
        "Updates the workspace with commits pushed to the connected branch. To use this API, the caller's Git "
        "credentials must be configured using Update My Git Credentials API. You can use the Get My Git Credentials "
        "API to check the Git credentials configuration. The update only affects items in the workspace that were "
        "changed in those commits. If called after the Connect and Initialize Connection APIs, it will perform a full "
        "update of the entire workspace."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_update_from_git(
            workspace_id: str = Field(..., description="The workspace ID."),
            remote_commit_hash: str = Field(..., description="Remote full SHA commit hash."),
            workspace_head: str | None = Field(
                None,
                description=(
                    "Full SHA hash that the workspace is synced to. This value may be null only after Initialize "
                    "Connection. In other cases, the system will validate that the given value is aligned with the head "
                    "known to the system."
                ),
            ),
            conflict_resolution: dict[str, Any] | None = Field(None, description="The basic conflict resolution data."),
            options: dict[str, Any] | None = Field(
                None, description="Contains the options that are enabled for the update from Git."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/git/updateFromGit",
                body=_drop_none(
                    {
                        "workspaceHead": workspace_head,
                        "remoteCommitHash": remote_commit_hash,
                        "conflictResolution": conflict_resolution,
                        "options": options,
                    }
                ),
            )

        super().__init__(handler=_git_update_from_git, **kwargs)


class FabricGitListWorkspaceRelations(_FabricTool):
    name: str = "fabric_git_list_workspace_relations"
    description: str | None = (
        "Returns a list of workspace relations for the specified workspace. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_list_workspace_relations(
            workspace_id: str = Field(..., description="The workspace ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/git/workspaceRelations",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_git_list_workspace_relations, **kwargs)


class FabricGitCreateWorkspaceRelation(_FabricTool):
    name: str = "fabric_git_create_workspace_relation"
    description: str | None = (
        "Creates a workspace relation between the specified workspace and a related workspace. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _git_create_workspace_relation(
            workspace_id: str = Field(..., description="The workspace ID."),
            related_workspace_id: str = Field(..., description="The related workspace ID."),
            relation_type: str = Field(
                ...,
                description=(
                    "The type of the related workspace in the relation. Additional related workspace types may be added "
                    "over time. Allowed values: Base, Branch, RelatedWorkspace."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/git/workspaceRelations",
                body=_drop_none({"relatedWorkspaceId": related_workspace_id, "relationType": relation_type}),
            )

        super().__init__(handler=_git_create_workspace_relation, **kwargs)


class FabricGitDeleteWorkspaceRelation(_FabricTool):
    name: str = "fabric_git_delete_workspace_relation"
    description: str | None = "Deletes a workspace relation. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _git_delete_workspace_relation(
            workspace_id: str = Field(..., description="The workspace ID."),
            workspace_relation_id: str = Field(..., description="The workspace relation ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/git/workspaceRelations/{workspace_relation_id}",
            )

        super().__init__(handler=_git_delete_workspace_relation, **kwargs)


class FabricListItems(_FabricTool):
    name: str = "fabric_list_items"
    description: str | None = "Returns a list of items from the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_items(
            workspace_id: str = Field(..., description="The workspace ID."),
            type: str | None = Field(None, description="The item's type."),
            recursive: bool | None = Field(
                None,
                description=(
                    "Lists items in a folder and its nested folders, or just a folder only. True - All items in the "
                    "folder and its nested folders are listed, False - Only items in the folder are listed. The default "
                    "value is true."
                ),
            ),
            root_folder_id: str | None = Field(
                None,
                description=(
                    "This parameter allows users to filter items based on a specific root folder. If not provided, the "
                    "workspace is used as the root folder."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            include: list[str] | None = Field(
                None,
                description=(
                    "Specifies which item properties to include in the response as a comma-separated list. Additional "
                    "values may be added over time."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items",
                params={
                    "type": type,
                    "recursive": recursive,
                    "rootFolderId": root_folder_id,
                    "continuationToken": continuation_token,
                    "include": include,
                },
            )

        super().__init__(handler=_list_items, **kwargs)


class FabricCreateItem(_FabricTool):
    name: str = "fabric_create_item"
    description: str | None = (
        "Creates an item in the specified workspace. This API is supported for a number of item types, find the "
        "supported item types in Item management overview. You can use Get item definition API to get an item "
        "definition."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            display_name: str = Field(
                ...,
                description="The item display name. The display name must follow naming rules according to item type.",
            ),
            type: str = Field(
                ...,
                description=(
                    "The type of the item. Additional item types may be added over time. Allowed values: Dashboard, "
                    "Report, SemanticModel, PaginatedReport, Datamart, Lakehouse, Eventhouse, Environment, KQLDatabase, "
                    "KQLQueryset, KQLDashboard, DataPipeline, Notebook, SparkJobDefinition, MLExperiment, MLModel, "
                    "Warehouse, Eventstream, SQLEndpoint, MirroredWarehouse, MirroredDatabase, Reflex, GraphQLApi, "
                    "MountedDataFactory, SQLDatabase, CopyJob, VariableLibrary, Dataflow, ApacheAirflowJob, "
                    "WarehouseSnapshot, DigitalTwinBuilder, DigitalTwinBuilderFlow, MirroredAzureDatabricksCatalog, Map, "
                    "AnomalyDetector, UserDataFunction, GraphModel, GraphQuerySet, SnowflakeDatabase, OperationsAgent, "
                    "CosmosDBDatabase, Ontology, EventSchemaSet, DataAgent, MirroredCatalog, AppBackend, OrgApp, "
                    "OrgAppAudience, DataBuildToolJob, AzureDatabricksStorage, Plan."
                ),
            ),
            description: str | None = Field(
                None, description="The item description. Maximum length is 256 characters."
            ),
            folder_id: str | None = Field(
                None,
                description=(
                    "The folder ID. If not specified or null, the item is created with the workspace as its folder."
                ),
            ),
            definition: dict[str, Any] | None = Field(None, description="An item definition object."),
            creation_payload: dict[str, Any] | None = Field(
                None,
                description=(
                    "A set of properties used to create the item. The *Create Item* page of the relevant type indicates "
                    "whether `creationPayload` is supported and lists the item's properties. Use `creationPayload` or "
                    "`definition`. You can't use both at the same time."
                ),
            ),
            sensitivity_label_settings: dict[str, Any] | None = Field(
                None, description="The sensitivity label settings."
            ),
            options: dict[str, Any] | None = Field(
                None,
                description=(
                    "Options for the item being created. Only honored when a `definition` is also provided; supplying "
                    "`options` without `definition` results in a `BadRequest` error. Property names are option names "
                    "valid for the item type; values are the option values. For the options supported by a specific item "
                    "type, see that item type's `Update-Definition` API documentation."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items",
                body=_drop_none(
                    {
                        "displayName": display_name,
                        "description": description,
                        "type": type,
                        "folderId": folder_id,
                        "definition": definition,
                        "creationPayload": creation_payload,
                        "sensitivityLabelSettings": sensitivity_label_settings,
                        "options": options,
                    }
                ),
            )

        super().__init__(handler=_create_item, **kwargs)


class FabricBulkExportItemDefinitions(_FabricTool):
    name: str = "fabric_bulk_export_item_definitions"
    description: str | None = (
        "Bulk export item definitions from the workspace. Exports item definitions from multiple items in a workspace "
        "in a single operation. You can export all supported item definitions or selectively export specific items "
        "from the specified workspace. To see the supported item types, see Item definition overview."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_export_item_definitions(
            workspace_id: str = Field(..., description="The workspace ID."),
            mode: str = Field(
                ...,
                description=(
                    "Modes for the bulk export item definitions operation. Additional modes may be added over time. "
                    "Allowed values: All, Selective."
                ),
            ),
            items: list[Any] | None = Field(None, description="A list of item identifiers to retrieve"),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/bulkExportDefinitions",
                body=_drop_none({"items": items, "mode": mode}),
            )

        super().__init__(handler=_bulk_export_item_definitions, **kwargs)


class FabricBulkImportItemDefinitions(_FabricTool):
    name: str = "fabric_bulk_import_item_definitions"
    description: str | None = (
        "Bulk import item definitions into the workspace. Imports item definitions for multiple items into a "
        "workspace in a single operation. Each item definition in the request will be processed, creating new items "
        "or updating existing ones as determined by the system based on whether the item already exists. To see the "
        "supported item types, see Item definition overview."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_import_item_definitions(
            workspace_id: str = Field(..., description="The workspace ID."),
            definition_parts: list[Any] = Field(..., description="A list of items definitions parts to import"),
            options: dict[str, Any] | None = Field(None, description="The import items request's configuration."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/bulkImportDefinitions",
                body=_drop_none({"definitionParts": definition_parts, "options": options}),
            )

        super().__init__(handler=_bulk_import_item_definitions, **kwargs)


class FabricBulkMoveItems(_FabricTool):
    name: str = "fabric_bulk_move_items"
    description: str | None = (
        "Moves multiple items to a folder. Child items are moved with their parent items. You can't move a child item "
        "without its parent item."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_move_items(
            workspace_id: str = Field(..., description="The workspace ID."),
            items: list[Any] = Field(..., description="The IDs of requested items to move."),
            target_folder_id: str | None = Field(
                None,
                description="The destination folder ID. If not provided, the workspace is used as the destination folder.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/bulkMove",
                body=_drop_none({"targetFolderId": target_folder_id, "items": items}),
            )

        super().__init__(handler=_bulk_move_items, **kwargs)


class FabricDeleteItem(_FabricTool):
    name: str = "fabric_delete_item"
    description: str | None = "Deletes the specified item."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            hard_delete: bool | None = Field(
                None,
                description=(
                    "Specifies whether to perform a hard delete. When set to `true`, the item is permanently deleted and "
                    "cannot be recovered. When set to `false` or not specified, the item is soft-deleted if the item type "
                    "supports it."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/items/{item_id}",
                params={"hardDelete": hard_delete},
            )

        super().__init__(handler=_delete_item, **kwargs)


class FabricGetItem(_FabricTool):
    name: str = "fabric_get_item"
    description: str | None = (
        "Returns properties of the specified item. This API is supported for a number of item types, find the "
        "supported item types in Item management overview. For retrieving additional type specific properties, refer "
        "to the get API reference page of the specific item type."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            include: list[str] | None = Field(
                None,
                description=(
                    "Specifies which item properties to include in the response as a comma-separated list. Additional "
                    "values may be added over time."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}",
                params={"include": include},
            )

        super().__init__(handler=_get_item, **kwargs)


class FabricUpdateItem(_FabricTool):
    name: str = "fabric_update_item"
    description: str | None = (
        "Updates the properties of the specified item. This API is supported for a number of item types, find the "
        "supported item types and information about their definition structure in Item management overview."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            display_name: str | None = Field(
                None,
                description="The item display name. The display name must follow naming rules according to item type.",
            ),
            description: str | None = Field(
                None, description="The item description. Maximum length is 256 characters."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/items/{item_id}",
                body=_drop_none({"displayName": display_name, "description": description}),
            )

        super().__init__(handler=_update_item, **kwargs)


class FabricApplyItemTags(_FabricTool):
    name: str = "fabric_apply_item_tags"
    description: str | None = "Apply tags on an item."

    def __init__(self, **kwargs: Any) -> None:
        async def _apply_item_tags(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            tags: list[Any] = Field(..., description="The array of tag IDs."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/applyTags",
                body=_drop_none({"tags": tags}),
            )

        super().__init__(handler=_apply_item_tags, **kwargs)


class FabricGetItemDefinition(_FabricTool):
    name: str = "fabric_get_item_definition"
    description: str | None = (
        "Returns the specified item definition. This API is supported for a number of item types, find the supported "
        "item types and information about their definition structure in Item definition overview. When you get an "
        "item's definition, the sensitivity label is not a part of the definition."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            format: str | None = Field(None, description="The format of the item definition."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/getDefinition",
                params={"format": format},
            )

        super().__init__(handler=_get_item_definition, **kwargs)


class FabricListItemJobInstances(_FabricTool):
    name: str = "fabric_list_item_job_instances"
    description: str | None = "Returns a list of job instances for the specified item."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_item_job_instances(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/instances",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_item_job_instances, **kwargs)


class FabricGetItemJobInstance(_FabricTool):
    name: str = "fabric_get_item_job_instance"
    description: str | None = "Get one item's job instance."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_job_instance(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_instance_id: str = Field(..., description="The job instance ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}",
            )

        super().__init__(handler=_get_item_job_instance, **kwargs)


class FabricCancelItemJobInstance(_FabricTool):
    name: str = "fabric_cancel_item_job_instance"
    description: str | None = "Cancel an item's job instance."

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_item_job_instance(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_instance_id: str = Field(..., description="The job instance ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}/cancel",
            )

        super().__init__(handler=_cancel_item_job_instance, **kwargs)


class FabricRunItemJob(_FabricTool):
    name: str = "fabric_run_item_job"
    description: str | None = "Run on-demand item job instance."

    def __init__(self, **kwargs: Any) -> None:
        async def _run_item_job(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_type: str = Field(..., description="Job type"),
            execution_data: dict[str, Any] | None = Field(
                None,
                description=(
                    "The execution data for an on-demand job. This is fixed static data, defined by the specific item job "
                    "type."
                ),
            ),
            parameters: list[Any] | None = Field(
                None,
                description=(
                    "The parameter list for an on-demand job. These are per-run, user-defined inputs that tailor this "
                    "invocation. Note: This property is not broadly supported. If the API returns an error with errorCode "
                    "`FeatureNotAvailable` and errorMessage `Parameter is not allowed for this item type or this item job "
                    "type`, the `parameters` property is not supported for the specified item type or item job type."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/{_quote(job_type)}/instances",
                body=_drop_none({"executionData": execution_data, "parameters": parameters}),
            )

        super().__init__(handler=_run_item_job, **kwargs)


class FabricListItemSchedules(_FabricTool):
    name: str = "fabric_list_item_schedules"
    description: str | None = "Get scheduling settings for one specific item."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_item_schedules(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_type: str = Field(..., description="The job type."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/{_quote(job_type)}/schedules",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_item_schedules, **kwargs)


class FabricCreateItemSchedule(_FabricTool):
    name: str = "fabric_create_item_schedule"
    description: str | None = "Create a new schedule for an item. An item can create maximum 20 schedulers."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_item_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_type: str = Field(..., description="The job type."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
            execution_data: dict[str, Any] | None = Field(
                None,
                description=(
                    "The execution data for a scheduled job. This is fixed static data, defined by the specific item job "
                    "type."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/{_quote(job_type)}/schedules",
                body=_drop_none({"enabled": enabled, "configuration": configuration, "executionData": execution_data}),
            )

        super().__init__(handler=_create_item_schedule, **kwargs)


class FabricDeleteItemSchedule(_FabricTool):
    name: str = "fabric_delete_item_schedule"
    description: str | None = "Delete an existing schedule for an item."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_type: str = Field(..., description="The job type."),
            schedule_id: str = Field(..., description="The item schedule ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/{_quote(job_type)}/schedules/{schedule_id}",
            )

        super().__init__(handler=_delete_item_schedule, **kwargs)


class FabricGetItemSchedule(_FabricTool):
    name: str = "fabric_get_item_schedule"
    description: str | None = "Get an existing schedule for an item."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_type: str = Field(..., description="The job type."),
            schedule_id: str = Field(..., description="The item schedule ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/{_quote(job_type)}/schedules/{schedule_id}",
            )

        super().__init__(handler=_get_item_schedule, **kwargs)


class FabricUpdateItemSchedule(_FabricTool):
    name: str = "fabric_update_item_schedule"
    description: str | None = "Update an existing schedule for an item."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_item_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            job_type: str = Field(..., description="The job type."),
            schedule_id: str = Field(..., description="The item schedule ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
            execution_data: dict[str, Any] | None = Field(
                None,
                description=(
                    "The execution data for a scheduled job. This is fixed static data, defined by the specific item job "
                    "type."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/items/{item_id}/jobs/{_quote(job_type)}/schedules/{schedule_id}",
                body=_drop_none({"enabled": enabled, "configuration": configuration, "executionData": execution_data}),
            )

        super().__init__(handler=_update_item_schedule, **kwargs)


class FabricUpdateItemLogicalId(_FabricTool):
    name: str = "fabric_update_item_logical_id"
    description: str | None = (
        "Updates the logical ID for the specified item. Corresponding item instances share the same logical ID. Use "
        "this operation to resolve logical ID conflicts. Before changing the logical ID, review any automations that "
        "rely on its current value and update them as needed. Otherwise, those automations might fail to locate the "
        "item or fail because of a duplicate display name. For more information, see Resolve logical ID conflicts in "
        "Microsoft Fabric."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_item_logical_id(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            logical_id: str = Field(..., description="The logical ID to update for the item."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/items/{item_id}/logicalId",
                body=_drop_none({"logicalId": logical_id}),
            )

        super().__init__(handler=_update_item_logical_id, **kwargs)


class FabricMoveItem(_FabricTool):
    name: str = "fabric_move_item"
    description: str | None = (
        "Moves the specified item to a folder. Moves the specified item to a folder within the same workspace. Child "
        "items are moved with their parent item. You can't move a child item without its parent item."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _move_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            target_folder_id: str | None = Field(
                None,
                description="The destination folder ID. If not provided, the workspace is used as the destination folder.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/move",
                body=_drop_none({"targetFolderId": target_folder_id}),
            )

        super().__init__(handler=_move_item, **kwargs)


class FabricGetItemDownstreamRelations(_FabricTool):
    name: str = "fabric_get_item_downstream_relations"
    description: str | None = "Gets downstream relations for an item. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_downstream_relations(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/relations/downstream",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_item_downstream_relations, **kwargs)


class FabricGetItemUpstreamRelations(_FabricTool):
    name: str = "fabric_get_item_upstream_relations"
    description: str | None = "Gets upstream relations for an item. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_upstream_relations(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/relations/upstream",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_item_upstream_relations, **kwargs)


class FabricUnapplyItemTags(_FabricTool):
    name: str = "fabric_unapply_item_tags"
    description: str | None = "Unapply tags from an item."

    def __init__(self, **kwargs: Any) -> None:
        async def _unapply_item_tags(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            tags: list[Any] = Field(..., description="The array of tag IDs."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/unapplyTags",
                body=_drop_none({"tags": tags}),
            )

        super().__init__(handler=_unapply_item_tags, **kwargs)


class FabricUpdateItemDefinition(_FabricTool):
    name: str = "fabric_update_item_definition"
    description: str | None = (
        "Overrides the definition for the specified item. This API is supported for a number of item types, find the "
        "supported item types and information about their definition structure in Item definition overview. Updating "
        "the item's definition, does not affect its sensitivity label."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_item_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            definition: dict[str, Any] = Field(..., description="An item definition object."),
            update_metadata: bool | None = Field(
                None,
                description=(
                    "When set to true and the .platform file is provided as part of the definition, the item's metadata "
                    "is updated using the metadata in the .platform file"
                ),
            ),
            options: dict[str, Any] | None = Field(
                None,
                description=(
                    "Options for the item whose definition is being updated. Property names are option names valid for "
                    "the item type; values are the option values. For the options supported by a specific item type, see "
                    "that item type's `Update-Definition` API documentation."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/updateDefinition",
                params={"updateMetadata": update_metadata},
                body=_drop_none({"definition": definition, "options": options}),
            )

        super().__init__(handler=_update_item_definition, **kwargs)


class FabricListRecoverableItems(_FabricTool):
    name: str = "fabric_list_recoverable_items"
    description: str | None = "Lists recoverable items in a workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_recoverable_items(
            workspace_id: str = Field(..., description="The workspace ID."),
            type: str | None = Field(None, description="The recoverable item's type."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            recoverable_by_me: bool | None = Field(
                None,
                description="When set to true, only items that can be immediately recovered by the caller are returned.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/recoverableItems",
                params={"type": type, "continuationToken": continuation_token, "recoverableByMe": recoverable_by_me},
            )

        super().__init__(handler=_list_recoverable_items, **kwargs)


class FabricDeleteRecoverableItem(_FabricTool):
    name: str = "fabric_delete_recoverable_item"
    description: str | None = "Permanently deletes a recoverable item."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_recoverable_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The recoverable item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/recoverableItems/{item_id}",
            )

        super().__init__(handler=_delete_recoverable_item, **kwargs)


class FabricRecoverRecoverableItem(_FabricTool):
    name: str = "fabric_recover_recoverable_item"
    description: str | None = (
        "Recovers a soft-deleted item in the specified workspace. Recovers a soft-deleted Fabric item to the Active "
        "provision state in the specified workspace. If the item has child items, all of them are recovered as well."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _recover_recoverable_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The recoverable item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/recoverableItems/{item_id}/recover",
            )

        super().__init__(handler=_recover_recoverable_item, **kwargs)


class FabricUnapplyWorkspaceTags(_FabricTool):
    name: str = "fabric_unapply_workspace_tags"
    description: str | None = "Unapply tags from a workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _unapply_workspace_tags(
            workspace_id: str = Field(..., description="The workspace ID."),
            tags: list[Any] = Field(..., description="An array of tag IDs to unapply."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/unapplyTags",
                body=_drop_none({"tags": tags}),
            )

        super().__init__(handler=_unapply_workspace_tags, **kwargs)


class FabricUnassignWorkspaceFromCapacity(_FabricTool):
    name: str = "fabric_unassign_workspace_from_capacity"
    description: str | None = "Unassigns the specified workspace from capacity."

    def __init__(self, **kwargs: Any) -> None:
        async def _unassign_workspace_from_capacity(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/unassignFromCapacity",
            )

        super().__init__(handler=_unassign_workspace_from_capacity, **kwargs)


class FabricUnassignWorkspaceFromDomain(_FabricTool):
    name: str = "fabric_unassign_workspace_from_domain"
    description: str | None = "Unassigns the specified workspace from domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _unassign_workspace_from_domain(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/unassignFromDomain",
            )

        super().__init__(handler=_unassign_workspace_from_domain, **kwargs)
