"""ServiceNow Table API tools for record CRUD operations."""

import base64
import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Resolve ServiceNow instance and auth from integration or env vars.

    Returns (base_url, auth_header_value) where auth_header_value is
    'Basic {base64(username:password)}'.
    """
    instance: str | None = None
    username: str | None = None
    password: str | None = None

    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        instance = credentials.get("instance")
        username = credentials.get("username")
        password = credentials.get("password")

    if instance is None:
        instance = tool.instance or os.getenv("SERVICENOW_INSTANCE")
    if username is None:
        username = (tool.username.get_secret_value() if tool.username else None) or os.getenv("SERVICENOW_USERNAME")
    if password is None:
        password = (tool.password.get_secret_value() if tool.password else None) or os.getenv("SERVICENOW_PASSWORD")

    if not instance or not username or not password:
        raise ValueError(
            "ServiceNow credentials not found. Set SERVICENOW_INSTANCE, SERVICENOW_USERNAME, and "
            "SERVICENOW_PASSWORD environment variables, pass them in config, or configure an integration."
        )

    auth_str = f"{username}:{password}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()
    base_url = f"https://{instance}.service-now.com/api/now/table"
    return base_url, f"Basic {auth_b64}"


# --- Record ---


class ServiceNowCreateRecord(Tool):
    name: str = "servicenow_create_record"
    description: str | None = "Create a new record in a ServiceNow table."
    integration: Annotated[str, Integration("servicenow")] | None = None
    instance: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "instance": self.instance,
                "username": self.username,
                "password": self.password,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_record(
            table: str = Field(..., description="Table name (e.g. incident, task, sys_user)"),
            record: dict[str, Any] = Field(..., description="Record fields as key-value pairs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/{table}",
                    headers={
                        "Authorization": auth,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=record,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_record, **kwargs)


class ServiceNowDeleteRecord(Tool):
    name: str = "servicenow_delete_record"
    description: str | None = "Delete a record from a ServiceNow table by sys_id."
    integration: Annotated[str, Integration("servicenow")] | None = None
    instance: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "instance": self.instance,
                "username": self.username,
                "password": self.password,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_record(
            table: str = Field(..., description="Table name (e.g. incident, task)"),
            sys_id: str = Field(..., description="Record sys_id to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/{table}/{sys_id}",
                    headers={"Authorization": auth, "Accept": "application/json"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted", "sys_id": sys_id}

        super().__init__(handler=_delete_record, **kwargs)


class ServiceNowGetRecordById(Tool):
    name: str = "servicenow_get_record_by_id"
    description: str | None = "Get a single ServiceNow record by sys_id."
    integration: Annotated[str, Integration("servicenow")] | None = None
    instance: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "instance": self.instance,
                "username": self.username,
                "password": self.password,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_record_by_id(
            table: str = Field(..., description="Table name (e.g. incident, task)"),
            sys_id: str = Field(..., description="Record sys_id"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/{table}/{sys_id}",
                    headers={"Authorization": auth, "Accept": "application/json"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_record_by_id, **kwargs)


class ServiceNowGetRecords(Tool):
    name: str = "servicenow_get_records"
    description: str | None = "Get multiple records from a ServiceNow table with optional query and pagination."
    integration: Annotated[str, Integration("servicenow")] | None = None
    instance: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "instance": self.instance,
                "username": self.username,
                "password": self.password,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_records(
            table: str = Field(..., description="Table name (e.g. incident, task)"),
            sysparm_limit: int = Field(10, description="Max records to return (1-10000)"),
            sysparm_offset: int = Field(0, description="Offset for pagination"),
            sysparm_query: str | None = Field(None, description="Encoded query (e.g. active=true^state=1)"),
            sysparm_order_by: str | None = Field(None, description="Sort field (prefix - for desc)"),
            sysparm_fields: str | None = Field(None, description="Comma-separated fields to return"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {
                "sysparm_limit": sysparm_limit,
                "sysparm_offset": sysparm_offset,
            }
            if sysparm_query:
                params["sysparm_query"] = sysparm_query
            if sysparm_order_by:
                params["sysparm_order_by"] = sysparm_order_by
            if sysparm_fields:
                params["sysparm_fields"] = sysparm_fields

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/{table}",
                    headers={"Authorization": auth, "Accept": "application/json"},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_records, **kwargs)


class ServiceNowPartiallyUpdateRecord(Tool):
    name: str = "servicenow_partially_update_record"
    description: str | None = "Partially update a ServiceNow record (PATCH) - only specified fields are changed."
    integration: Annotated[str, Integration("servicenow")] | None = None
    instance: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "instance": self.instance,
                "username": self.username,
                "password": self.password,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _partially_update_record(
            table: str = Field(..., description="Table name (e.g. incident, task)"),
            sys_id: str = Field(..., description="Record sys_id to update"),
            record: dict[str, Any] = Field(..., description="Fields to update (only these are changed)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{base_url}/{table}/{sys_id}",
                    headers={
                        "Authorization": auth,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=record,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_partially_update_record, **kwargs)


class ServiceNowUpdateRecord(Tool):
    name: str = "servicenow_update_record"
    description: str | None = "Full update of a ServiceNow record (PUT) - replaces the entire record."
    integration: Annotated[str, Integration("servicenow")] | None = None
    instance: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "instance": self.instance,
                "username": self.username,
                "password": self.password,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_record(
            table: str = Field(..., description="Table name (e.g. incident, task)"),
            sys_id: str = Field(..., description="Record sys_id to update"),
            record: dict[str, Any] = Field(..., description="Full record data (replaces existing)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/{table}/{sys_id}",
                    headers={
                        "Authorization": auth,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=record,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_record, **kwargs)
