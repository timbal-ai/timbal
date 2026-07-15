"""Aircall Public API tools.

Auth (first match wins):

- **Basic Auth** (Aircall customers): ``api_id`` + ``api_token`` from
  ``Integration("aircall")``, tool fields, or env ``AIRCALL_API_ID`` / ``AIRCALL_API_TOKEN``.
- **OAuth Bearer** (technology partners): ``access_token`` from integration, tool field, or
  env ``AIRCALL_ACCESS_TOKEN``.

Integration credentials (type: credentials):
- api_id: API key ID (Basic auth username)
- api_token: API key token (Basic auth password)
- access_token: OAuth access token (Bearer; alternative to Basic)
"""

from __future__ import annotations

import base64
import os
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_ROOT = "https://api.aircall.io"
_API_PREFIX = "/v1"


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    text = str(value).strip()
    return text or None


def _join_path(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    if normalized.startswith(_API_PREFIX):
        return f"{_API_ROOT}{normalized}"
    return f"{_API_ROOT}{_API_PREFIX}{normalized}"


async def _resolve_bearer_token(tool: Any) -> str | None:
    if getattr(tool, "access_token", None) is not None:
        return _secret_value(tool.access_token)

    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()
        token = _secret_value(creds.get("access_token"))
        if token:
            return token

    return _secret_value(os.getenv("AIRCALL_ACCESS_TOKEN"))


async def _resolve_basic_credentials(tool: Any) -> tuple[str, str] | None:
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    api_id = (
        _secret_value(creds.get("api_id"))
        or _secret_value(creds.get("api_key"))
        or _secret_value(getattr(tool, "api_id", None))
        or _secret_value(os.getenv("AIRCALL_API_ID"))
    )
    api_token = (
        _secret_value(creds.get("api_token"))
        or _secret_value(creds.get("api_secret"))
        or _secret_value(getattr(tool, "api_token", None))
        or _secret_value(os.getenv("AIRCALL_API_TOKEN"))
    )
    if api_id and api_token:
        return api_id, api_token
    return None


def _basic_auth_headers(api_id: str, api_token: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{api_id}:{api_token}".encode()).decode("ascii")
    return {
        "Authorization": f"Basic {encoded}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _bearer_auth_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _auth_headers(tool: Any) -> dict[str, str]:
    bearer = await _resolve_bearer_token(tool)
    if bearer:
        return _bearer_auth_headers(bearer)

    basic = await _resolve_basic_credentials(tool)
    if basic:
        return _basic_auth_headers(*basic)

    raise ValueError(
        "Aircall credentials not found. Configure integration with api_id + api_token, "
        "or access_token, or set AIRCALL_API_ID/AIRCALL_API_TOKEN or AIRCALL_ACCESS_TOKEN."
    )


async def _aircall_request(
    tool: Any,
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    import httpx

    url = _join_path(path)
    headers = await _auth_headers(tool)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.request(
            method.upper(),
            url,
            headers=headers,
            params=params or None,
            json=json_body,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _aircall_config_fields(tool: Any) -> dict[str, Any]:
    return {
        "integration": tool.integration,
        "api_id": tool.api_id,
        "api_token": tool.api_token,
        "access_token": tool.access_token,
    }


class AircallRequest(Tool):
    """Call any Aircall Public API endpoint."""

    name: str = "aircall_request"
    description: str | None = (
        "Call any Aircall Public API endpoint with method, path, optional query params and JSON body."
    )
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _request(
            method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = Field(
                "GET",
                description="HTTP method.",
            ),
            path: str = Field(..., description="API path relative to /v1 (e.g. 'calls' or '/users')."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query string parameters."),
            body: dict[str, Any] | None = Field(None, description="Optional JSON request body."),
        ) -> Any:
            return await _aircall_request(
                self,
                method=method,
                path=path,
                params=query_params,
                json_body=body,
            )

        super().__init__(handler=_request, **kwargs)


class AircallPing(Tool):
    """Verify Aircall API credentials via GET /v1/ping."""

    name: str = "aircall_ping"
    description: str | None = "Verify Aircall API credentials via GET /v1/ping."
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _ping() -> Any:
            return await _aircall_request(self, method="GET", path="/ping")

        super().__init__(handler=_ping, **kwargs)


class AircallListCalls(Tool):
    """List calls for the Aircall company (six months of history)."""

    name: str = "aircall_list_calls"
    description: str | None = "List calls for the Aircall company. Up to six months of history."
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_calls(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(20, description="Results per page (1-50)."),
            from_timestamp: int | None = Field(
                None,
                description="Minimal call creation date (UNIX timestamp). Query param: from.",
            ),
            to_timestamp: int | None = Field(
                None,
                description="Maximal call creation date (UNIX timestamp). Query param: to.",
            ),
            order: Literal["asc", "desc"] | None = Field(
                None,
                description="Order by created_at. Default asc; use desc for most recent calls first.",
            ),
            fetch_contact: bool | None = Field(None, description="Include contact details in each call."),
            fetch_short_urls: bool | None = Field(None, description="Include short URLs in each call."),
            fetch_call_timeline: bool | None = Field(None, description="Include ivr_options_selected in each call."),
        ) -> Any:
            params: dict[str, Any] = {"page": page, "per_page": per_page}
            if from_timestamp is not None:
                params["from"] = from_timestamp
            if to_timestamp is not None:
                params["to"] = to_timestamp
            if order is not None:
                params["order"] = order
            if fetch_contact is not None:
                params["fetch_contact"] = fetch_contact
            if fetch_short_urls is not None:
                params["fetch_short_urls"] = fetch_short_urls
            if fetch_call_timeline is not None:
                params["fetch_call_timeline"] = fetch_call_timeline
            return await _aircall_request(self, method="GET", path="/calls", params=params)

        super().__init__(handler=_list_calls, **kwargs)


class AircallGetCall(Tool):
    """Retrieve a single call by ID."""

    name: str = "aircall_get_call"
    description: str | None = "Retrieve a single call by ID."
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_call(
            call_id: int = Field(..., description="Call ID."),
            fetch_contact: bool | None = Field(None, description="Include contact details."),
            fetch_short_urls: bool | None = Field(None, description="Include short URLs."),
            fetch_call_timeline: bool | None = Field(None, description="Include ivr_options_selected."),
        ) -> Any:
            params: dict[str, Any] = {}
            if fetch_contact is not None:
                params["fetch_contact"] = fetch_contact
            if fetch_short_urls is not None:
                params["fetch_short_urls"] = fetch_short_urls
            if fetch_call_timeline is not None:
                params["fetch_call_timeline"] = fetch_call_timeline
            return await _aircall_request(self, method="GET", path=f"/calls/{call_id}", params=params or None)

        super().__init__(handler=_get_call, **kwargs)


class AircallSearchCalls(Tool):
    """Search calls by user, phone number, tags, direction, or date range."""

    name: str = "aircall_search_calls"
    description: str | None = (
        "Search calls by user_id, phone_number, tags, direction, or date range. "
        "Transferred calls appear for the receiving number, not the original."
    )
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _search_calls(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(20, description="Results per page (1-50)."),
            from_timestamp: int | None = Field(None, description="Minimal call creation date (UNIX timestamp)."),
            to_timestamp: int | None = Field(None, description="Maximal call creation date (UNIX timestamp)."),
            order: Literal["asc", "desc"] | None = Field(None, description="Order by created_at."),
            direction: Literal["inbound", "outbound"] | None = Field(None, description="Call direction filter."),
            user_id: int | None = Field(None, description="Filter by user who made or received the call."),
            phone_number: str | None = Field(None, description="Filter by calling or receiving phone number."),
            tags: list[int] | None = Field(None, description="Tag IDs (AND filter — all tags must match)."),
            fetch_contact: bool | None = Field(None, description="Include contact details."),
            fetch_short_urls: bool | None = Field(None, description="Include short URLs."),
            fetch_call_timeline: bool | None = Field(None, description="Include ivr_options_selected."),
        ) -> Any:
            params: dict[str, Any] = {"page": page, "per_page": per_page}
            if from_timestamp is not None:
                params["from"] = from_timestamp
            if to_timestamp is not None:
                params["to"] = to_timestamp
            if order is not None:
                params["order"] = order
            if direction is not None:
                params["direction"] = direction
            if user_id is not None:
                params["user_id"] = user_id
            if phone_number is not None:
                params["phone_number"] = phone_number
            if tags is not None:
                params["tags"] = tags
            if fetch_contact is not None:
                params["fetch_contact"] = fetch_contact
            if fetch_short_urls is not None:
                params["fetch_short_urls"] = fetch_short_urls
            if fetch_call_timeline is not None:
                params["fetch_call_timeline"] = fetch_call_timeline
            return await _aircall_request(self, method="GET", path="/calls/search", params=params)

        super().__init__(handler=_search_calls, **kwargs)


class AircallListContacts(Tool):
    """List shared contacts for the Aircall company."""

    name: str = "aircall_list_contacts"
    description: str | None = "List shared contacts for the Aircall company."
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_contacts(
            page: int = Field(1, description="Page number (1-based)."),
            per_page: int = Field(20, description="Results per page (1-50)."),
            from_timestamp: int | None = Field(None, description="Minimal contact creation date (UNIX timestamp)."),
            to_timestamp: int | None = Field(None, description="Maximal contact creation date (UNIX timestamp)."),
            order: Literal["asc", "desc"] | None = Field(None, description="Sort order."),
            order_by: Literal["created_at", "updated_at"] | None = Field(None, description="Sort field."),
        ) -> Any:
            params: dict[str, Any] = {"page": page, "per_page": per_page}
            if from_timestamp is not None:
                params["from"] = from_timestamp
            if to_timestamp is not None:
                params["to"] = to_timestamp
            if order is not None:
                params["order"] = order
            if order_by is not None:
                params["order_by"] = order_by
            return await _aircall_request(self, method="GET", path="/contacts", params=params)

        super().__init__(handler=_list_contacts, **kwargs)


class AircallGetContact(Tool):
    """Retrieve a shared contact by ID."""

    name: str = "aircall_get_contact"
    description: str | None = "Retrieve a shared contact by ID."
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_contact(
            contact_id: int = Field(..., description="Contact ID."),
        ) -> Any:
            return await _aircall_request(self, method="GET", path=f"/contacts/{contact_id}")

        super().__init__(handler=_get_contact, **kwargs)


class AircallCreateContact(Tool):
    """Create a shared contact. At least one phone number is required."""

    name: str = "aircall_create_contact"
    description: str | None = "Create a shared contact. At least one phone number is required."
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _create_contact(
            phone_numbers: list[dict[str, str]] = Field(
                ...,
                description="Phone numbers. Each item needs label and value (e.g. {'label': 'Work', 'value': '+1...'}).",
            ),
            first_name: str | None = Field(None, description="Contact first name."),
            last_name: str | None = Field(None, description="Contact last name."),
            company_name: str | None = Field(None, description="Company name."),
            information: str | None = Field(None, description="Free-form information (e.g. external CRM id)."),
            emails: list[dict[str, str]] | None = Field(
                None,
                description="Emails. Each item needs label and value.",
            ),
        ) -> Any:
            body: dict[str, Any] = {"phone_numbers": phone_numbers}
            if first_name is not None:
                body["first_name"] = first_name
            if last_name is not None:
                body["last_name"] = last_name
            if company_name is not None:
                body["company_name"] = company_name
            if information is not None:
                body["information"] = information
            if emails is not None:
                body["emails"] = emails
            return await _aircall_request(self, method="POST", path="/contacts", json_body=body)

        super().__init__(handler=_create_contact, **kwargs)


class AircallUpdateContact(Tool):
    """Update a shared contact (POST /v1/contacts/:id). Phone/email updates use dedicated endpoints."""

    name: str = "aircall_update_contact"
    description: str | None = (
        "Update a shared contact name, company, or information. "
        "To change phone numbers or emails use the dedicated contact detail endpoints."
    )
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _update_contact(
            contact_id: int = Field(..., description="Contact ID."),
            first_name: str | None = Field(None, description="Updated first name."),
            last_name: str | None = Field(None, description="Updated last name."),
            company_name: str | None = Field(None, description="Updated company name."),
            information: str | None = Field(None, description="Updated information."),
        ) -> Any:
            body: dict[str, Any] = {}
            if first_name is not None:
                body["first_name"] = first_name
            if last_name is not None:
                body["last_name"] = last_name
            if company_name is not None:
                body["company_name"] = company_name
            if information is not None:
                body["information"] = information
            return await _aircall_request(
                self,
                method="POST",
                path=f"/contacts/{contact_id}",
                json_body=body,
            )

        super().__init__(handler=_update_contact, **kwargs)


class AircallGetCallTranscription(Tool):
    """Retrieve AI transcription for a call. Requires AI Assist on the Aircall account."""

    name: str = "aircall_get_call_transcription"
    description: str | None = (
        "Retrieve AI-generated transcription for a call. Requires AI Assist (or AI Assist Pro) on the account."
    )
    integration: Annotated[str, Integration("aircall")] | None = None
    api_id: SecretStr | None = None
    api_token: SecretStr | None = None
    access_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_aircall_config_fields(self))}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_transcription(
            call_id: int = Field(..., description="Call ID."),
        ) -> Any:
            return await _aircall_request(self, method="GET", path=f"/calls/{call_id}/transcription")

        super().__init__(handler=_get_transcription, **kwargs)
