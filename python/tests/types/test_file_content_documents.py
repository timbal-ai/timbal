"""Spreadsheets and Word documents are pasted into the prompt as text; an unreadable one never fails the run."""
import io
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook
from timbal.types import File
from timbal.types.content import FileContent

CONVERTERS = ("to_openai_responses_input", "to_openai_chat_completions_input", "to_anthropic_input")

# A custom document property with no name, as some writers emit; openpyxl rejects it.
NAMELESS_CUSTOM_PROPS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
    'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
    '<property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2"><vt:lpwstr>x</vt:lpwstr></property>'
    "</Properties>"
)


def _workbook(path: Path, sheets: dict[str, list[list[str]]], custom_props: str | None = None) -> Path:
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    if custom_props is None:
        path.write_bytes(buf.getvalue())
        return path
    src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(
                    b"</Types>",
                    b'<Override PartName="/docProps/custom.xml" '
                    b'ContentType="application/vnd.openxmlformats-officedocument.custom-properties+xml"/></Types>',
                )
            out.writestr(item, data)
        out.writestr("docProps/custom.xml", custom_props)
    return path


def _text(content: FileContent, converter: str) -> str:
    return getattr(FileContent(file=content.file), converter)()["text"]


@pytest.mark.parametrize("converter", CONVERTERS)
def test_workbook_with_a_nameless_custom_property_is_read(tmp_path: Path, converter: str) -> None:
    path = _workbook(tmp_path / "pack.xlsx", {"People": [["name", "unit"], ["Ali", "Traffic"]]}, NAMELESS_CUSTOM_PROPS)
    text = _text(FileContent(file=File.validate(str(path))), converter)
    assert text == "name,unit\r\nAli,Traffic\r\n"


@pytest.mark.parametrize("converter", CONVERTERS)
def test_every_sheet_is_read_under_its_title(tmp_path: Path, converter: str) -> None:
    path = _workbook(tmp_path / "pack.xlsx", {"People": [["name"], ["Ali"]], "Units": [["unit"], ["Traffic"]]})
    text = _text(FileContent(file=File.validate(str(path))), converter)
    assert text == "## Sheet: People\nname\r\nAli\r\n\n## Sheet: Units\nunit\r\nTraffic\r\n"


@pytest.mark.parametrize("converter", CONVERTERS)
@pytest.mark.parametrize(("name", "kind"), [("broken.xlsx", "spreadsheet"), ("broken.docx", "Word document")])
def test_an_unreadable_document_reaches_the_model_as_a_file_error(
    tmp_path: Path, converter: str, name: str, kind: str
) -> None:
    path = tmp_path / name
    path.write_bytes(b"PK\x03\x04 not really a zip")
    text = _text(FileContent(file=File.validate(str(path))), converter)
    assert text.startswith(f"[FILE ERROR: The {kind} could not be read.")
