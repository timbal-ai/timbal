from typing import Annotated, Any
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


async def _resolve_token(tool: Any) -> str:
    """Resolve Excel OAuth token from integration."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    raise ValueError("Excel integration not configured.")


def _wb(drive_id: str | None, workbook_id: str) -> str:
    if drive_id:
        return f"{_GRAPH_BASE}/drives/{drive_id}/items/{workbook_id}/workbook"
    return f"{_GRAPH_BASE}/me/drive/items/{workbook_id}/workbook"


def _sheet(drive_id: str | None, workbook_id: str, sheet_name: str) -> str:
    return f"{_wb(drive_id, workbook_id)}/worksheets/{sheet_name}"


# ---------------------------------------------------------------------------
# Row operations
# ---------------------------------------------------------------------------


class WriteToSheet(Tool):
    name: str = "excel_write_to_sheet"
    description: str | None = (
        "Write one or more rows of values to a worksheet starting at a given cell address."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _write_to_sheet(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            address: str = Field(..., description="Top-left cell address to start writing, e.g. 'A1' or 'B3'."),
            values: list[list[Any]] = Field(..., description="2-D array of values, e.g. [['Name', 'Age'], ['Alice', 30]]."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            rows = len(values)
            cols = max(len(r) for r in values) if values else 1
            range_addr = _expand_address(address, rows, cols)

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/range(address='{range_addr}')",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"values": values},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/WriteToSheet"
        super().__init__(handler=_write_to_sheet, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Sheet operations
# ---------------------------------------------------------------------------


class ClearSheet(Tool):
    name: str = "excel_clear_sheet"
    description: str | None = "Clear all content and formatting from a worksheet."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _clear_sheet(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/range/clear",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"applyTo": "All"},
                )
                response.raise_for_status()
                return {"cleared": True, "sheet": sheet_name}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/ClearSheet"
        super().__init__(handler=_clear_sheet, metadata=metadata, **kwargs)


class CreateSheet(Tool):
    name: str = "excel_create_sheet"
    description: str | None = "Add a new worksheet to an existing workbook."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_sheet(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_wb(drive_id, workbook_id)}/worksheets/add",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"name": sheet_name},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/CreateSheet"
        super().__init__(handler=_create_sheet, metadata=metadata, **kwargs)


class DeleteSheet(Tool):
    name: str = "excel_delete_sheet"
    description: str | None = "Delete a worksheet from a workbook by name."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_sheet(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return {"deleted": True, "sheet": sheet_name}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/DeleteSheet"
        super().__init__(handler=_delete_sheet, metadata=metadata, **kwargs)


class ListSheets(Tool):
    name: str = "excel_list_sheets"
    description: str | None = "List all worksheets in a workbook."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sheets(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_wb(drive_id, workbook_id)}/worksheets",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/ListSheets"
        super().__init__(handler=_list_sheets, metadata=metadata, **kwargs)


class ReadFromSheet(Tool):
    name: str = "excel_read_from_sheet"
    description: str | None = (
        "Read all used values from a worksheet, or a specific cell range."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _read_from_sheet(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            address: str | None = Field(None, description="Optional range address, e.g. 'A1:D10'. If omitted, reads the used range."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            if address:
                url = f"{_sheet(drive_id, workbook_id, sheet_name)}/range(address='{address}')"
            else:
                url = f"{_sheet(drive_id, workbook_id, sheet_name)}/usedRange"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/ReadFromSheet"
        super().__init__(handler=_read_from_sheet, metadata=metadata, **kwargs)


class UpdateCell(Tool):
    name: str = "excel_update_cell"
    description: str | None = "Update the value of a single cell in a worksheet."
    integration: Annotated[str, Integration("excel")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_cell(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            address: str = Field(..., description="Cell address, e.g. 'B4'."),
            value: Any = Field(..., description="The new value to set (string, number, bool, or formula starting with '=')."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/range(address='{address}')",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"values": [[value]]},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/UpdateCell"
        super().__init__(handler=_update_cell, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Spreadsheet operations
# ---------------------------------------------------------------------------


class FindRow(Tool):
    name: str = "excel_find_row"
    description: str | None = (
        "Find the first row in a worksheet where a specified column matches a given value."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_row(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            column: str = Field(..., description="Column letter to search, e.g. 'A' or 'C'."),
            value: str = Field(..., description="The value to look for (case-insensitive string match)."),
            has_header: bool = Field(True, description="Whether the first row is a header row (skipped during search)."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/usedRange",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                data = response.json()

            values: list[list[Any]] = data.get("values", [])
            headers = values[0] if has_header and values else []
            col_idx = _col_letter_to_index(column)
            start_row = 1 if has_header else 0

            for i in range(start_row, len(values)):
                cell_val = str(values[i][col_idx]) if col_idx < len(values[i]) else ""
                if cell_val.lower() == value.lower():
                    row_dict = dict(zip(headers, values[i])) if headers else {"values": values[i]}
                    return {"row_index": i, "row": row_dict}

            return {"row_index": None, "row": None}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/FindRow"
        super().__init__(handler=_find_row, metadata=metadata, **kwargs)


class GetCellsInRange(Tool):
    name: str = "excel_get_cells_in_range"
    description: str | None = "Get all cell values within a specified range address."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_cells_in_range(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            address: str = Field(..., description="Range address, e.g. 'A1:E20'."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/range(address='{address}')",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/GetCellsInRange"
        super().__init__(handler=_get_cells_in_range, metadata=metadata, **kwargs)


class GetRowByIndex(Tool):
    name: str = "excel_get_row_by_index"
    description: str | None = "Get a single row from a worksheet by its 0-based row index."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_row_by_index(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            row_index: int = Field(..., description="0-based index of the data row to retrieve (header row not counted when has_header=True)."),
            has_header: bool = Field(True, description="Whether the first row is a header row (skipped during indexing)."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/usedRange",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                data = response.json()

            values: list[list[Any]] = data.get("values", [])
            headers = values[0] if has_header and values else []
            actual_idx = row_index + (1 if has_header else 0)

            if actual_idx >= len(values):
                return {"row_index": row_index, "row": None}

            row_vals = values[actual_idx]
            row_dict = dict(zip(headers, row_vals)) if headers else {"values": row_vals}
            return {"row_index": row_index, "row": row_dict}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/GetRowByIndex"
        super().__init__(handler=_get_row_by_index, metadata=metadata, **kwargs)


class UpdateRow(Tool):
    name: str = "excel_update_row"
    description: str | None = (
        "Update all values in an existing row by its 0-based row index."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_row(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            row_index: int = Field(..., description="0-based index of the data row to update (header not counted when has_header=True)."),
            values: list[Any] = Field(..., description="List of cell values for the row, left to right."),
            has_header: bool = Field(True, description="Whether the first row is a header row (skipped during indexing)."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            actual_idx = row_index + (1 if has_header else 0)
            excel_row = actual_idx + 1  # Excel rows are 1-based
            cols = len(values)
            end_col = _col_index_to_letter(cols - 1)
            address = f"A{excel_row}:{end_col}{excel_row}"

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/range(address='{address}')",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"values": [values]},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/UpdateRow"
        super().__init__(handler=_update_row, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Table operations
# ---------------------------------------------------------------------------


class AddDataToTable(Tool):
    name: str = "excel_add_data_to_table"
    description: str | None = "Append one or more rows to an Excel table."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_data_to_table(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            table_name: str = Field(..., description="Name of the Excel table (ListObject)."),
            rows: list[list[Any]] = Field(..., description="List of row arrays to append, e.g. [['Alice', 30], ['Bob', 25]]."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/tables/{table_name}/rows/add",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"values": rows},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/AddDataToTable"
        super().__init__(handler=_add_data_to_table, metadata=metadata, **kwargs)


class CreateTable(Tool):
    name: str = "excel_create_table"
    description: str | None = (
        "Convert a range into a named Excel table (ListObject) with optional headers."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_table(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="Name or index of the worksheet (e.g. 'Sheet1' or '0')."),
            address: str = Field(..., description="Range to convert to a table, e.g. 'A1:D10'."),
            has_headers: bool = Field(True, description="Whether the first row of the range contains column headers."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}/tables/add",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"address": address, "hasHeaders": has_headers},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/CreateTable"
        super().__init__(handler=_create_table, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Workbook operations
# ---------------------------------------------------------------------------


class CreateWorkbook(Tool):
    name: str = "excel_create_workbook"
    description: str | None = (
        "Create a new Excel workbook (.xlsx) in OneDrive at the specified path."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_workbook(
            name: str = Field(..., description="Filename for the new workbook, e.g. 'report.xlsx'."),
            folder_path: str = Field("/", description="OneDrive folder path, e.g. '/Documents/Reports'. Defaults to root."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint; omit for personal OneDrive."),
        ) -> Any:
            
            token = await _resolve_token(self)
            import httpx

            if not name.endswith(".xlsx"):
                name = f"{name}.xlsx"

            folder_path = folder_path.rstrip("/")
            if drive_id:
                if folder_path:
                    url = f"{_GRAPH_BASE}/drives/{drive_id}/root:{folder_path}:/{name}/content"
                else:
                    url = f"{_GRAPH_BASE}/drives/{drive_id}/root:/{name}:/content"
            else:
                if folder_path:
                    url = f"{_GRAPH_BASE}/me/drive/root:{folder_path}/{name}:/content"
                else:
                    url = f"{_GRAPH_BASE}/me/drive/root:/{name}:/content"

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    },
                    content=b"",
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/CreateWorkbook"
        super().__init__(handler=_create_workbook, metadata=metadata, **kwargs)


class ListWorkbooks(Tool):
    name: str = "excel_list_workbooks"
    description: str | None = (
        "List Excel workbooks (.xlsx files) in a OneDrive folder."
    )
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workbooks(
            folder_path: str = Field("/", description="OneDrive folder path to search in. Defaults to root."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            folder_path = folder_path.rstrip("/")
            if drive_id:
                if folder_path:
                    url = f"{_GRAPH_BASE}/drives/{drive_id}/root:{folder_path}:/children"
                else:
                    url = f"{_GRAPH_BASE}/drives/{drive_id}/root/children"
            else:
                if folder_path:
                    url = f"{_GRAPH_BASE}/me/drive/root:{folder_path}:/children"
                else:
                    url = f"{_GRAPH_BASE}/me/drive/root/children"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"$filter": "endswith(name, '.xlsx')", "$select": "id,name,size,lastModifiedDateTime,webUrl"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/ListWorkbooks"
        super().__init__(handler=_list_workbooks, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Worksheet operations
# ---------------------------------------------------------------------------


class GetSheetById(Tool):
    name: str = "excel_get_sheet_by_id"
    description: str | None = "Get worksheet metadata by its persistent ID."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sheet_by_id(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_id: str = Field(..., description="The persistent worksheet ID (not the display name)."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_wb(drive_id, workbook_id)}/worksheets/{sheet_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/GetSheetById"
        super().__init__(handler=_get_sheet_by_id, metadata=metadata, **kwargs)


class RenameSheet(Tool):
    name: str = "excel_rename_sheet"
    description: str | None = "Rename a worksheet in a workbook."
    integration: Annotated[str, Integration("excel")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _rename_sheet(
            workbook_id: str = Field(..., description="The ID of the workbook file in OneDrive/SharePoint."),
            sheet_name: str = Field(..., description="The current display name of the worksheet."),
            new_name: str = Field(..., description="The new display name for the worksheet."),
            drive_id: str | None = Field(None, description="Optional Drive ID for SharePoint drives; omit for personal OneDrive."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_sheet(drive_id, workbook_id, sheet_name)}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"name": new_name},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Excel/RenameSheet"
        super().__init__(handler=_rename_sheet, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col_letter_to_index(col: str) -> int:
    col = col.upper().strip()
    result = 0
    for ch in col:
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result - 1


def _col_index_to_letter(idx: int) -> str:
    result = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        result = chr(ord("A") + rem) + result
    return result


def _expand_address(start: str, rows: int, cols: int) -> str:
    """Convert a start address like 'B3' and dimensions into a range like 'B3:D5'."""
    col_str = ""
    row_str = ""
    for ch in start.upper():
        if ch.isalpha():
            col_str += ch
        else:
            row_str += ch
    start_col_idx = _col_letter_to_index(col_str)
    start_row = int(row_str)
    end_col = _col_index_to_letter(start_col_idx + cols - 1)
    end_row = start_row + rows - 1
    return f"{col_str}{start_row}:{end_col}{end_row}"
