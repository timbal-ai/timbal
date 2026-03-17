from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_SHEETS_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
_DRIVE_BASE = "https://www.googleapis.com/drive/v3"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError(
        "Google Sheets credentials not found. Configure an integration or pass token."
    )


class GoogleSheetsCreateSheet(Tool):
    name: str = "google_sheets_create_sheet"
    description: str | None = "Create a new Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_sheet(
            title: str = Field(..., description="Title of the new spreadsheet"),
            sheet_names: list[str] | None = Field(None, description="List of sheet names to create in the spreadsheet"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {"properties": {"title": title}}
            if sheet_names:
                body["sheets"] = [{"properties": {"title": name}} for name in sheet_names]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _SHEETS_BASE,
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_sheet, **kwargs)


class GoogleSheetsGetSpreadsheetInfo(Tool):
    name: str = "google_sheets_get_spreadsheet_info"
    description: str | None = "Get metadata for a Google Sheets spreadsheet, including title, locale, and sheet list."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_spreadsheet_info(spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_BASE}/{spreadsheet_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"fields": "spreadsheetId,properties,sheets.properties"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_spreadsheet_info, **kwargs)


class GoogleSheetsGetSheetNames(Tool):
    name: str = "google_sheets_get_sheet_names"
    description: str | None = "List all sheet (tab) names in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sheet_names(spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_BASE}/{spreadsheet_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"fields": "sheets.properties.title,sheets.properties.sheetId"},
                )
                response.raise_for_status()
                data = response.json()
                return [s["properties"] for s in data.get("sheets", [])]

        super().__init__(handler=_get_sheet_names, **kwargs)


class GoogleSheetsBatchGet(Tool):
    name: str = "google_sheets_batch_get"
    description: str | None = "Get values from multiple ranges in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _batch_get(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            ranges: list[str] = Field(..., description="List of ranges to retrieve values from (e.g., ['Sheet1!A1:B10', 'Sheet2!C1:D10'])"),
            major_dimension: str = Field("ROWS", description="Dimension to use for the range values: 'ROWS' or 'COLUMNS'"),
            value_render_option: str = Field("FORMATTED_VALUE", description="How to render the values: 'FORMATTED_VALUE', 'UNFORMATTED_VALUE', or 'FORMULA'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            query: list[tuple[str, str]] = [("ranges", r) for r in ranges]
            query.append(("majorDimension", major_dimension))
            query.append(("valueRenderOption", value_render_option))

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_BASE}/{spreadsheet_id}/values:batchGet",
                    headers={"Authorization": f"Bearer {token}"},
                    params=query,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_batch_get, **kwargs)


class GoogleSheetsBatchUpdate(Tool):
    name: str = "google_sheets_batch_update"
    description: str | None = "Update values in multiple ranges of a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _batch_update(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            data: list[dict[str, Any]] = Field(..., description="List of value ranges to update, each with 'range' and 'values' keys: list of {\"range\": \"Sheet1!A1:B2\", \"values\": [[...], [...]]}"),
            value_input_option: str = Field("USER_ENTERED", description="How to interpret the input values: 'RAW' or 'USER_ENTERED'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body = {"valueInputOption": value_input_option, "data": data}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_BASE}/{spreadsheet_id}/values:batchUpdate",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_batch_update, **kwargs)


class GoogleSheetsAppendValues(Tool):
    name: str = "google_sheets_append_values"
    description: str | None = "Append rows of values to a Google Sheets range."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _append_values(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            range: str = Field(..., description="Range to append to, e.g., 'Sheet1!A1'"),
            values: list[list[Any]] = Field(..., description="2D array of values to append"),
            value_input_option: str = Field("USER_ENTERED", description="How to interpret the input values: 'RAW' or 'USER_ENTERED'"),
            insert_data_option: str = Field("INSERT_ROWS", description="How to insert data: 'OVERWRITE' or 'INSERT_ROWS'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body = {"values": values}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_BASE}/{spreadsheet_id}/values/{range}:append",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "valueInputOption": value_input_option,
                        "insertDataOption": insert_data_option,
                    },
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_append_values, **kwargs)


class GoogleSheetsLookupRow(Tool):
    name: str = "google_sheets_lookup_row"
    description: str | None = "Find the first row where a column contains a given value in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _lookup_row(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            range: str = Field(..., description="Range to search in, e.g., 'Sheet1!A1:Z1000'"),
            column_index: int = Field(..., description="Column index (1-based) to search in"),
            value: str = Field(..., description="Value to search for"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_BASE}/{spreadsheet_id}/values/{range}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"majorDimension": "ROWS", "valueRenderOption": "FORMATTED_VALUE"},
                )
                response.raise_for_status()
                data = response.json()

            rows = data.get("values", [])
            for row_index, row in enumerate(rows):
                if len(row) > column_index and str(row[column_index]) == value:
                    return {"row_index": row_index, "row": row}

            return {"row_index": None, "row": None}

        super().__init__(handler=_lookup_row, **kwargs)


class GoogleSheetsClearValues(Tool):
    name: str = "google_sheets_clear_values"
    description: str | None = "Clear all values from a range in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _clear_values(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            range: str = Field(..., description="Range to clear, e.g., 'Sheet1!A1:Z1000'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_BASE}/{spreadsheet_id}/values/{range}:clear",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_clear_values, **kwargs)


class GoogleSheetsCopySheet(Tool):
    name: str = "google_sheets_copy_sheet"
    description: str | None = "Copy a sheet from one Google Sheets spreadsheet to another."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _copy_sheet(
            source_spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID to copy from"),
            sheet_id: int = Field(..., description="ID of the sheet to copy"),
            destination_spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID to copy to"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body = {"destinationSpreadsheetId": destination_spreadsheet_id}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_BASE}/{source_spreadsheet_id}/sheets/{sheet_id}:copyTo",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_copy_sheet, **kwargs)


class GoogleSheetsAddSheet(Tool):
    name: str = "google_sheets_add_sheet"
    description: str | None = "Add a new sheet (tab) to an existing Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_sheet(
            spreadsheet_id: str = Field(..., description="Google Sheets spreadsheet ID"),
            title: str = Field(..., description="Title for the new sheet"),
            index: int | None = Field(None, description="Index where the new sheet should be inserted (0-based)"),
            row_count: int = Field(1000, description="Number of rows in the new sheet"),
            column_count: int = Field(26, description="Number of columns in the new sheet"),
        ) -> Any:
            token = await _resolve_token(self)
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
                    f"{_SHEETS_BASE}/{spreadsheet_id}:batchUpdate",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_add_sheet, **kwargs)


class GoogleSheetsShareSpreadsheet(Tool):
    name: str = "google_sheets_share_spreadsheet"
    description: str | None = "Share a Google Sheets spreadsheet with a user, group, domain, or make it public."
    integration: Annotated[str, Integration("google_sheets")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
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
            token = await _resolve_token(self)
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
                    f"{_DRIVE_BASE}/files/{spreadsheet_id}/permissions",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_share_spreadsheet, **kwargs)
