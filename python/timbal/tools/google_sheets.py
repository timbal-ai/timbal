from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_SHEETS_API_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
_DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


class CreateSheet(Tool):
    name: str = "google_sheets_create_sheet"
    description: str | None = "Create a new Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_sheet(
            title: str,
            sheet_names: list[str] | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"properties": {"title": title}}
            if sheet_names:
                body["sheets"] = [{"properties": {"title": name}} for name in sheet_names]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _SHEETS_API_BASE,
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_spreadsheet_info(spreadsheet_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}",
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sheet_names(spreadsheet_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}",
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _batch_get(
            spreadsheet_id: str,
            ranges: list[str],
            major_dimension: str = "ROWS",
            value_render_option: str = "FORMATTED_VALUE",
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: list[tuple[str, str]] = [(r, r) for r in ranges]
            query: list[tuple[str, str]] = [("ranges", r) for r in ranges]
            query.append(("majorDimension", major_dimension))
            query.append(("valueRenderOption", value_render_option))

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values:batchGet",
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _batch_update(
            spreadsheet_id: str,
            data: list[dict[str, Any]],
            value_input_option: str = "USER_ENTERED",
        ) -> Any:
            """
            data: list of {"range": "Sheet1!A1:B2", "values": [[...], [...]]}
            value_input_option: "RAW" or "USER_ENTERED"
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body = {"valueInputOption": value_input_option, "data": data}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values:batchUpdate",
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _append_values(
            spreadsheet_id: str,
            range: str,
            values: list[list[Any]],
            value_input_option: str = "USER_ENTERED",
            insert_data_option: str = "INSERT_ROWS",
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body = {"values": values}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values/{range}:append",
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _lookup_row(
            spreadsheet_id: str,
            range: str,
            column_index: int,
            value: str,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values/{range}",
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/LookupRow"

        super().__init__(handler=_lookup_row, metadata=metadata, **kwargs)


class ClearValues(Tool):
    name: str = "google_sheets_clear_values"
    description: str | None = "Clear all values from a range in a Google Sheets spreadsheet."
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _clear_values(
            spreadsheet_id: str,
            range: str,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{spreadsheet_id}/values/{range}:clear",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/ClearValues"

        super().__init__(handler=_clear_values, metadata=metadata, **kwargs)


class CopySheet(Tool):
    name: str = "google_sheets_copy_sheet"
    description: str | None = "Copy a sheet from one Google Sheets spreadsheet to another."
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _copy_sheet(
            source_spreadsheet_id: str,
            sheet_id: int,
            destination_spreadsheet_id: str,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body = {"destinationSpreadsheetId": destination_spreadsheet_id}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SHEETS_API_BASE}/{source_spreadsheet_id}/sheets/{sheet_id}:copyTo",
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_sheet(
            spreadsheet_id: str,
            title: str,
            index: int | None = None,
            row_count: int = 1000,
            column_count: int = 26,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
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
    integration: Annotated[str, Integration("google_sheets")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _share_spreadsheet(
            spreadsheet_id: str,
            role: str,
            type: str,
            email_address: str | None = None,
            domain: str | None = None,
            send_notification_email: bool = True,
            email_message: str | None = None,
        ) -> Any:
            """
            role: "reader", "commenter", "writer", "owner"
            type: "user", "group", "domain", "anyone"
            email_address: required when type is "user" or "group"
            domain: required when type is "domain"
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleSheets/ShareSpreadsheet"

        super().__init__(handler=_share_spreadsheet, metadata=metadata, **kwargs)
