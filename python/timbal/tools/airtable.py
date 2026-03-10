import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_AIRTABLE_API_BASE = "https://api.airtable.com/v0"
_AIRTABLE_META_BASE = "https://api.airtable.com/v0/meta"

_BATCH_SIZE = 10  # Airtable hard limit per request


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Airtable API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("AIRTABLE_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Airtable API key not found. Set AIRTABLE_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


def _validate_airtable_id(value: str, expected_prefix: str, param_name: str) -> None:
    """Raise ValueError early when a human-readable name is passed instead of an Airtable ID."""
    if not value.startswith(expected_prefix):
        hint = "Use airtable_list_bases to look up the correct ID." if expected_prefix == "app" else ""
        raise ValueError(
            f"'{param_name}' must be an Airtable ID starting with '{expected_prefix}' "
            f"(e.g. '{expected_prefix}xxxxxxxxxxxxx'), got: '{value}'. {hint}".strip()
        )


def _normalize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Normalize field values so {\"name\": \"...\"}  single-key dicts become plain strings.

    Airtable's singleSelect field expects a bare string, not an object with a 'name' key.
    This handles both formats transparently so callers never need to know the difference.
    """
    result: dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, dict) and tuple(v.keys()) == ("name",):
            result[k] = v["name"]
        else:
            result[k] = v
    return result


async def _raise_for_status(response: httpx.Response) -> None:
    """Like response.raise_for_status() but always includes the response body in the message."""
    if response.is_error:
        try:
            body: Any = response.json()
        except Exception:
            body = response.text
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code} {response.request.method} {response.url} — {body}",
            request=response.request,
            response=response,
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class ListRecords(Tool):
    name: str = "airtable_list_records"
    description: str | None = "List records in an Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_records(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id_or_name: str = Field(..., description="Airtable table ID or name"),
            fields: list[str] | None = Field(None, description="List of field names to retrieve"),
            filter_by_formula: str | None = Field(None, description="Filter formula to apply"),
            max_records: int | None = Field(None, description="Maximum number of records to return"),
            page_size: int = Field(100, description="Number of records per page"),
            sort: list[dict[str, str]] | None = Field(None, description="List of sort criteria. {'field': 'Name', 'direction': 'asc' | 'desc'}"),
            view: str | None = Field(None, description="View ID or name to filter by"),
            offset: str | None = Field(None, description="Offset token for pagination"),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            _validate_airtable_id(table_id_or_name, "tbl", "table_id_or_name")
            sort: list[dict[str, str]] | None = None,
            view: str | None = None,
            offset: str | None = None,
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if fields:
                params["fields[]"] = fields
            if filter_by_formula:
                params["filterByFormula"] = filter_by_formula
            if max_records:
                params["maxRecords"] = max_records
            if view:
                params["view"] = view
            if offset:
                params["offset"] = offset
            if sort:
                for i, s in enumerate(sort):
                    params[f"sort[{i}][field]"] = s["field"]
                    params[f"sort[{i}][direction]"] = s.get("direction", "asc")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_AIRTABLE_API_BASE}/{base_id}/{table_id_or_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/ListRecords"

        super().__init__(handler=_list_records, metadata=metadata, **kwargs)


class GetRecord(Tool):
    name: str = "airtable_get_record"
    description: str | None = "Get a single record by its ID from an Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_record(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id_or_name: str = Field(..., description="Airtable table ID or name"),
            record_id: str = Field(..., description="Airtable record ID starting with 'rec'"),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            _validate_airtable_id(record_id, "rec", "record_id")
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_AIRTABLE_API_BASE}/{base_id}/{table_id_or_name}/{record_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/GetRecord"

        super().__init__(handler=_get_record, metadata=metadata, **kwargs)


class CreateRecords(Tool):
    name: str = "airtable_create_records"
    description: str | None = "Create new records in an Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_records(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id_or_name: str = Field(..., description="Airtable table ID or name"),
            records: list[dict[str, Any]] = Field(..., description="List of records to create ({'fields': {'FieldName': value, ...}}). Each record should be a dictionary with 'fields' key. singleSelect values can be passed as a plain string or {"name": "..."} — both are accepted."),
            typecast: bool = Field(False, description="If True, Airtable will attempt to convert string values to the correct type."),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            normalized = [{"fields": _normalize_fields(r["fields"])} for r in records]
            all_created: list[Any] = []

            async with httpx.AsyncClient() as client:
                for i in range(0, len(normalized), _BATCH_SIZE):
                    batch = normalized[i : i + _BATCH_SIZE]
                    response = await client.post(
                        f"{_AIRTABLE_API_BASE}/{base_id}/{table_id_or_name}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"records": batch, "typecast": typecast},
                    )
                    await _raise_for_status(response)
                    all_created.extend(response.json().get("records", []))

            return {"records": all_created}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/CreateRecords"

        super().__init__(handler=_create_records, metadata=metadata, **kwargs)


class UpdateRecords(Tool):
    name: str = "airtable_update_records"
    description: str | None = "Update existing records in an Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_records(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id_or_name: str = Field(..., description="Airtable table ID or name"),
            records: list[dict[str, Any]] = Field(..., description="List of records to update ({'id': 'recXXX', 'fields': {'FieldName': value, ...}}). Each record should be a dictionary with 'id' and 'fields' keys. singleSelect values can be passed as a plain string or {'name': '...'} — both are accepted."),
            typecast: bool = Field(False, description="If True, Airtable will attempt to convert string values to the correct type."),
            destructive: bool = Field(False, description="If True, uses PUT (replaces all fields); if False, uses PATCH (merges fields)."),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            normalized = [{"id": r["id"], "fields": _normalize_fields(r["fields"])} for r in records]
            method = "put" if destructive else "patch"

            async with httpx.AsyncClient() as client:
                response = await getattr(client, method)(
                    f"{_AIRTABLE_API_BASE}/{base_id}/{table_id_or_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"records": normalized, "typecast": typecast},
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/UpdateRecords"

        super().__init__(handler=_update_records, metadata=metadata, **kwargs)


class DeleteRecords(Tool):
    name: str = "airtable_delete_records"
    description: str | None = "Delete one or more records from an Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_records(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id_or_name: str = Field(..., description="Airtable table ID or name"),
            record_ids: list[str] = Field(..., description="List of record IDs to delete (each starting with 'rec')."),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_AIRTABLE_API_BASE}/{base_id}/{table_id_or_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=[("records[]", rid) for rid in record_ids],
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/DeleteRecords"

        super().__init__(handler=_delete_records, metadata=metadata, **kwargs)


class ListComments(Tool):
    name: str = "airtable_list_comments"
    description: str | None = "List all comments for a specific Airtable record from newest to oldest."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_comments(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id_or_name: str = Field(..., description="Airtable table ID or name"),
            record_id: str = Field(..., description="Airtable record ID starting with 'rec'"),
            page_size: int = Field(100, description="Number of comments to return per page (max 100)"),
            offset: str | None = Field(None, description="Offset cursor for pagination"),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            _validate_airtable_id(record_id, "rec", "record_id")
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if offset:
                params["offset"] = offset

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_AIRTABLE_API_BASE}/{base_id}/{table_id_or_name}/{record_id}/comments",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/ListComments"

        super().__init__(handler=_list_comments, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


class ListBases(Tool):
    name: str = "airtable_list_bases"
    description: str | None = "List all accessible Airtable bases with their ID, name, and permission level."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_bases(offset: str | None = Field(None, description="Pagination offset for listing bases")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {}
            if offset:
                params["offset"] = offset

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_AIRTABLE_META_BASE}/bases",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/ListBases"

        super().__init__(handler=_list_bases, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class ListTables(Tool):
    name: str = "airtable_list_tables"
    description: str | None = "List all tables in a given Airtable base."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tables(base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name). Use airtable_list_bases first if you only have the base name.")) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_AIRTABLE_META_BASE}/bases/{base_id}/tables",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"include[]": "visibleFieldIds"},
                )
                await _raise_for_status(response)
                data = response.json()
                return [
                    {"id": t["id"], "name": t["name"], "description": t.get("description")}
                    for t in data.get("tables", [])
                ]

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/ListTables"

        super().__init__(handler=_list_tables, metadata=metadata, **kwargs)


class BaseSchema(Tool):
    name: str = "airtable_base_schema"
    description: str | None = "Get complete schema for all tables in an Airtable base, including fields and views."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _base_schema(base_id: str = Field(..., description="Airtable base ID starting with 'app'")) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_AIRTABLE_META_BASE}/bases/{base_id}/tables",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/BaseSchema"

        super().__init__(handler=_base_schema, metadata=metadata, **kwargs)


class CreateTable(Tool):
    name: str = "airtable_create_table"
    description: str | None = "Create a new table in an Airtable base with specified fields."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_table(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            name: str = Field(..., description="Name of the table to create"),
            fields: list[dict[str, Any]] = Field(..., description="List of field definitions. The first field must be the primary field and cannot be a computed type. [{'name': 'Name', 'type': 'singleLineText'}, {'name': 'Status', 'type': 'singleSelect', 'options': {'choices': [{'name': 'Todo'}]}}] The first field must be the primary field and cannot be a computed type."),
            description: str | None = Field(None, description="Optional description for the table"),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {"name": name, "fields": fields}
            if description:
                body["description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_AIRTABLE_META_BASE}/bases/{base_id}/tables",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/CreateTable"

        super().__init__(handler=_create_table, metadata=metadata, **kwargs)


class UpdateTable(Tool):
    name: str = "airtable_update_table"
    description: str | None = "Update an existing Airtable table's name or description."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_table(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id: str = Field(..., description="Airtable table ID starting with 'tbl' (not the table name)"),
            name: str | None = Field(None, description="New name for the table"),
            description: str | None = Field(None, description="New description for the table"),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            _validate_airtable_id(table_id, "tbl", "table_id")
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {}
            if name:
                body["name"] = name
            if description is not None:
                body["description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_AIRTABLE_META_BASE}/bases/{base_id}/tables/{table_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/UpdateTable"

        super().__init__(handler=_update_table, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


class CreateField(Tool):
    name: str = "airtable_create_field"
    description: str | None = "Add a new field (column) to an existing Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_field(
            base_id: str = Field(..., description="Airtable base ID starting with 'app' (not the base name)"),
            table_id: str = Field(..., description="Airtable table ID starting with 'tbl' (not the table name)"),
            name: str = Field(..., description="Name of the field to create"),
            type: str = Field(..., description="Airtable field type, e.g. 'singleLineText', 'number', 'singleSelect', 'multipleSelects', 'date', 'checkbox', 'url', 'email', 'phoneNumber', 'currency', 'percent', 'duration', 'rating', 'multilineText', etc."),
            description: str | None = Field(None, description="Optional description for the field"),
            options: dict[str, Any] | None = Field(None, description="Optional field-specific options (e.g., choices for select fields) e.g. {'choices': [{'name': 'Option A'}]} for singleSelect."),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            _validate_airtable_id(table_id, "tbl", "table_id")
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {"name": name, "type": type}
            if description:
                body["description"] = description
            if options:
                body["options"] = options

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_AIRTABLE_META_BASE}/bases/{base_id}/tables/{table_id}/fields",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/CreateField"

        super().__init__(handler=_create_field, metadata=metadata, **kwargs)


class UpdateField(Tool):
    name: str = "airtable_update_field"
    description: str | None = "Update a field's metadata (name, description, or options) in an Airtable table."
    integration: Annotated[str, Integration("airtable")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_field(
            base_id: str = Field(..., description="Airtable base ID"),
            table_id: str = Field(..., description="Airtable table ID"),
            field_id: str = Field(..., description="Airtable field ID"),
            name: str | None = Field(None, description="New field name"),
            description: str | None = Field(None, description="Field description"),
            options: dict[str, Any] | None = Field(None, description="Field options for select/multiSelect fields"),
        ) -> Any:
            _validate_airtable_id(base_id, "app", "base_id")
            _validate_airtable_id(table_id, "tbl", "table_id")
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {}
            if name:
                body["name"] = name
            if description is not None:
                body["description"] = description
            if options:
                body["options"] = options

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_AIRTABLE_META_BASE}/bases/{base_id}/tables/{table_id}/fields/{field_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                await _raise_for_status(response)
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Airtable/UpdateField"

        super().__init__(handler=_update_field, metadata=metadata, **kwargs)
