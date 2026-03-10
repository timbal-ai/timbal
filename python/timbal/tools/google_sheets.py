import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_SHEETS_API_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
_DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Google Sheets API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("GOOGLE_SHEETS_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Google Sheets API key not found. Set GOOGLE_SHEETS_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class CreateSheet(Tool):
    name: str = "google_sheets_create_sheet"
    description: str | None = "Create a new Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _create_sheet(
            title: str = Field(..., description="Title of the new spreadsheet"),
            sheet_names: list[str] | None = Field(None, description="List of sheet names to create in the spreadsheet"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {"properties": {"title": title}}
            if sheet_names:
                body["sheets"] = [{"properties": {"title": name}} for name in sheet_names]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _SHEETS_API_BASE,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/CreateSheet"

        super().__init__(handler=_create_sheet, metadata=metadata, **kwargs)


class GetSpreadsheetInfo(Tool):
    name: str = "google_sheets_get_spreadsheet_info"
    description: str | None = "Get metadata for a Google Sheets spreadsheet, including title, locale, and sheet list."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _get_spreadsheet_info(spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"fields": "spreadsheetId,properties,sheets.properties"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/GetSpreadsheetInfo"

        super().__init__(handler=_get_spreadsheet_info, metadata=metadata, **kwargs)


class GetSheetNames(Tool):
    name: str = "google_sheets_get_sheet_names"
    description: str | None = "List all sheet (tab) names in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _get_sheet_names(spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"fields": "sheets.properties.title,sheets.properties.sheetId"},
                )
                response.raise_for_status()
                data = response.json()
                return [s["properties"] for s in data.get("sheets", [])]

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/GetSheetNames"

        super().__init__(handler=_get_sheet_names, metadata=metadata, **kwargs)


class BatchGet(Tool):
    name: str = "google_sheets_batch_get"
    description: str | None = "Get values from multiple ranges in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _batch_get(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            ranges: list[str] = Field(..., description="List of ranges to retrieve values from (e.g., ['Sheet1!A1:B10', 'Sheet2!C1:D10'])"),
            major_dimension: str = Field("ROWS", description="Dimension to use for the range values: 'ROWS' or 'COLUMNS'"),
            value_render_option: str = Field("FORMATTED_VALUE", description="How to render the values: 'FORMATTED_VALUE', 'UNFORMATTED_VALUE', or 'FORMULA'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: list[tuple[str, str]] = [(r, r) for r in ranges]
            query: list[tuple[str, str]] = [("ranges", r) for r in ranges]
            query.append(("majorDimension", major_dimension))
            query.append(("valueRenderOption", value_render_option))

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values:batchGet",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=query,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/BatchGet"

        super().__init__(handler=_batch_get, metadata=metadata, **kwargs)


class BatchUpdate(Tool):
    name: str = "google_sheets_batch_update"
    description: str | None = "Update values in multiple ranges of a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _batch_update(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            data: list[dict[str, Any]] = Field(..., description="List of value ranges to update, each with 'range' and 'values' keys: list of {\"range\": \"Sheet1!A1:B2\", \"values\": [[...], [...]]}"),
            value_input_option: str = Field("USER_ENTERED", description="How to interpret the input values: 'RAW' or 'USER_ENTERED'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body = {"valueInputOption": value_input_option, "data": data}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values:batchUpdate",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/BatchUpdate"

        super().__init__(handler=_batch_update, metadata=metadata, **kwargs)


class AppendValues(Tool):
    name: str = "google_sheets_append_values"
    description: str | None = "Append rows of values to a Google Sheets range."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _append_values(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            range: str = Field(..., description="Range to append to, e.g., 'Sheet1!A1'"),
            values: list[list[Any]] = Field(..., description="2D array of values to append"),
            value_input_option: str = Field("USER_ENTERED", description="How to interpret the input values: 'RAW' or 'USER_ENTERED'"),
            insert_data_option: str = Field("INSERT_ROWS", description="How to insert data: 'OVERWRITE' or 'INSERT_ROWS'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body = {"values": values}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values/{range}:append",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={
                        "valueInputOption": value_input_option,
                        "insertDataOption": insert_data_option,
                    },
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/AppendValues"

        super().__init__(handler=_append_values, metadata=metadata, **kwargs)


class LookupRow(Tool):
    name: str = "google_sheets_lookup_row"
    description: str | None = "Find the first row where a column contains a given value in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _lookup_row(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            range: str = Field(..., description="Range to search in, e.g., 'Sheet1!A1:Z1000'"),
            column_index: int = Field(..., description="Column index (1-based) to search in"),
            value: str = Field(..., description="Value to search for"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values/{range}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"majorDimension": "ROWS", "valueRenderOption": "FORMATTED_VALUE"},
                )
                response.raise_for_status()
                data = response.json()

            rows = data.get("values", [])
            for row_index, row in enumerate(rows):
                if len(row) > column_index and str(row[column_index]) == value:
                    return {"row_index": row_index, "row": row}

            return {"row_index": None, "row": None}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/LookupRow"

        super().__init__(handler=_lookup_row, metadata=metadata, **kwargs)


class ClearValues(Tool):
    name: str = "google_sheets_clear_values"
    description: str | None = "Clear all values from a range in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _clear_values(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            range: str = Field(..., description="Range to clear, e.g., 'Sheet1!A1:Z1000'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values/{range}:clear",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/ClearValues"

        super().__init__(handler=_clear_values, metadata=metadata, **kwargs)


class CopySheet(Tool):
    name: str = "google_sheets_copy_sheet"
    description: str | None = "Copy a sheet from one Google Sheets spreadsheet to another."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _copy_sheet(
            source_spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID to copy from"),
            sheet_id: int = Field(..., description="ID of the sheet to copy"),
            destination_spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID to copy to"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body = {"destinationSpreadsheetId": destination_spreadsheet_id}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{source_spreadsheet_id}/sheets/{sheet_id}:copyTo",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/CopySheet"

        super().__init__(handler=_copy_sheet, metadata=metadata, **kwargs)


class AddSheet(Tool):
    name: str = "google_sheets_add_sheet"
    description: str | None = "Add a new sheet (tab) to an existing Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _add_sheet(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            title: str = Field(..., description="Title for the new sheet"),
            index: int | None = Field(None, description="Index where the new sheet should be inserted (0-based)"),
            row_count: int = Field(1000, description="Number of rows in the new sheet"),
            column_count: int = Field(26, description="Number of columns in the new sheet"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            sheet_properties: dict[str, Any] = {
                "title": title,
                "gridProperties": {"rowCount": row_count, "columnCount": column_count},
            }
            if index is not None:
                sheet_properties["index"] = index

            body = {"requests": [{"addSheet": {"properties": sheet_properties}}]}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}:batchUpdate",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/AddSheet"

        super().__init__(handler=_add_sheet, metadata=metadata, **kwargs)


class ShareSpreadsheet(Tool):
    name: str = "google_sheets_share_spreadsheet"
    description: str | None = "Share a Google Sheets spreadsheet with a user, group, domain, or make it public."
    integration: Annotated[str, Integration("google_sheets")] | None = None
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
        async def _share_spreadsheet(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            role: str = Field(..., description="Role to grant: 'reader', 'commenter', 'writer', 'owner'"),
            type: str = Field(..., description="Type of grant: 'user', 'group', 'domain', 'anyone'"),
            email_address: str | None = Field(None, description="Email address (required for 'user' or 'group')"),
            domain: str | None = Field(None, description="Domain (required for 'domain')"),
            send_notification_email: bool = True,
            email_message: str | None = None,
        ) -> Any:
            """
            role: "reader", "commenter", "writer", "owner"
            type: "user", "group", "domain", "anyone"
            email_address: required when type is "user" or "group"
            domain: required when type is "domain"
            """
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {"role": role, "type": type}
            if email_address:
                body["emailAddress"] = email_address
            if domain:
                body["domain"] = domain

            params: dict[str, Any] = {"sendNotificationEmail": send_notification_email}
            if email_message:
                params["emailMessage"] = email_message

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DRIVE_API_BASE}/files/{spreadsheet_id}/permissions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/ShareSpreadsheet"

        super().__init__(handler=_share_spreadsheet, metadata=metadata, **kwargs)
