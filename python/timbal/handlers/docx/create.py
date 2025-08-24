from io import BytesIO
from urllib.parse import urlparse

import requests
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
from docx.shared import Inches
from pydantic import BaseModel, Field

from ...types.file import File
from ...utils import resolve_default

# Alignment mappings
PARAGRAPH_ALIGNMENTS = {
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "left": WD_ALIGN_PARAGRAPH.LEFT
}

TABLE_ALIGNMENTS = {
    "center": WD_TABLE_ALIGNMENT.CENTER,
    "right": WD_TABLE_ALIGNMENT.RIGHT,
    "left": WD_TABLE_ALIGNMENT.LEFT
}


class DocxText(BaseModel):
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False


class DocxImage(BaseModel):
    uri: str
    width: float | None = None  # Width in pixels (will be converted to inches)
    height: float | None = None  # Height in pixels (will be converted to inches)
    caption: str | None = None
    alignment: str = "left"


class DocxTable(BaseModel):
    data: list[list[str | DocxImage]]
    style: str = "Table Grid"
    header_row: bool = False
    alignment: str = "left"
    borders: bool = True
    cell_alignment: str = "left"
    column_widths: list[float] | None = None


class DocxSection(BaseModel):
    heading: str | None = None
    level: int = 1
    paragraphs: list[str | DocxText] = []
    images: list[DocxImage] = []
    tables: list[DocxTable] = []


def _load_image(uri: str) -> BytesIO | str:
    """Load image from URI (URL or file path)."""
    parsed = urlparse(uri)
    if parsed.scheme in ('http', 'https'):
        response = requests.get(uri, timeout=30)
        response.raise_for_status()
        return BytesIO(response.content)
    return uri


def _add_picture_with_size(run_or_doc, image_stream, width_px: float | None = None, height_px: float | None = None):
    """Add picture with optional width/height dimensions (converts pixels to inches)."""
    # Convert pixels to inches using 96 DPI (standard screen resolution)
    # Also apply reasonable max limits for document images
    MAX_WIDTH_INCHES = 6.5  # Standard page width minus margins
    MAX_HEIGHT_INCHES = 9.0  # Reasonable max height
    
    width_in = min(width_px / 96, MAX_WIDTH_INCHES) if width_px else None
    height_in = min(height_px / 96, MAX_HEIGHT_INCHES) if height_px else None
    
    if width_in and height_in:
        return run_or_doc.add_picture(image_stream, width=Inches(width_in), height=Inches(height_in))
    elif width_in:
        return run_or_doc.add_picture(image_stream, width=Inches(width_in))
    elif height_in:
        return run_or_doc.add_picture(image_stream, height=Inches(height_in))
    else:
        return run_or_doc.add_picture(image_stream)


def _set_alignment(element, alignment: str, alignment_map: dict):
    """Set alignment on an element using the provided alignment map."""
    if alignment.lower() in alignment_map:
        element.alignment = alignment_map[alignment.lower()]


def _populate_table_cell(cell, cell_content, table_data: DocxTable):
    """Populate a single table cell with content and styling."""
    if isinstance(cell_content, DocxImage):
        cell.text = ""
        cell_paragraph = cell.paragraphs[0]
        
        try:
            image_stream = _load_image(cell_content.uri)
            run = cell_paragraph.runs[0] if cell_paragraph.runs else cell_paragraph.add_run()
            _add_picture_with_size(run, image_stream, cell_content.width, cell_content.height)
            _set_alignment(cell_paragraph, cell_content.alignment, PARAGRAPH_ALIGNMENTS)
        except Exception as e:
            cell.text = f"[Image error: {str(e)}]"
    else:
        cell.text = str(cell_content)
    
    # Apply cell alignment
    cell_alignment = table_data.cell_alignment.lower()
    if cell_alignment == "center":
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for paragraph in cell.paragraphs:
            if not any(run.element.xpath('.//pic:pic') for run in paragraph.runs):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif cell_alignment == "right":
        for paragraph in cell.paragraphs:
            if not any(run.element.xpath('.//pic:pic') for run in paragraph.runs):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _apply_table_styling(table, table_data: DocxTable, rows: int, cols: int):
    """Apply styling options to the table."""
    # Set table style and alignment
    table.style = table_data.style
    _set_alignment(table, table_data.alignment, TABLE_ALIGNMENTS)
    
    # Header row styling
    if table_data.header_row and rows > 0:
        for j in range(cols):
            try:
                cell = table.cell(0, j)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            except Exception:
                continue
    
    # Column widths
    if table_data.column_widths:
        for j, width in enumerate(table_data.column_widths):
            if j < cols:
                try:
                    for i in range(rows):
                        table.cell(i, j).width = Inches(width)
                except Exception:
                    continue
    
    # Borders
    if table_data.borders:
        try:
            tbl = table._tbl
            tblPr = tbl.tblPr
            
            tblBorders = OxmlElement('w:tblBorders')
            border_types = ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']
            
            for border_type in border_types:
                border = OxmlElement(f'w:{border_type}')
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '4')
                border.set(qn('w:space'), '0')
                border.set(qn('w:color'), '000000')
                tblBorders.append(border)
            
            tblPr.append(tblBorders)
        except Exception:
            pass


def add_table(doc, table_data: DocxTable):
    """Create a single table with all styling and content options."""
    data = table_data.data
    if not data or not isinstance(data, list):
        raise ValueError("Table data is empty or not a list")
        
    rows = len(data)
    cols = len(data[0]) if rows > 0 and isinstance(data[0], list) else 0
    
    if rows == 0 or cols == 0:
        raise ValueError("Table has no rows or columns")
    
    table = doc.add_table(rows=rows, cols=cols)
    
    # Populate cells
    for i, row in enumerate(data):
        if not isinstance(row, list):
            continue
        for j, cell_content in enumerate(row):
            if j < cols:
                try:
                    cell = table.cell(i, j)
                    _populate_table_cell(cell, cell_content, table_data)
                except Exception:
                    continue
    
    # Apply styling
    _apply_table_styling(table, table_data, rows, cols)


def create_docx(
    title: str | None = Field(
        None,
        description="Title for the document",
    ),
    sections: list[DocxSection] = Field(
        ...,
        description="List of sections in the document",
    )
) -> File:
    """
    Generates a Word document (.docx) from a structured document model containing title, sections, 
    images, and enhanced tables. Supports text formatting, image embedding, and advanced table styling.
    
    Args:
        title: Title for the document
        sections: List of sections in the document
    
    Returns:
        File: File object containing the Word document bytes with .docx extension
    """
    title = resolve_default("title", title)
    sections = resolve_default("sections", sections)

    filename = title + ".docx" if title else "generated_document.docx"
    doc = Document()
    
    if title:
        doc.add_heading(title, 0)
    
    for section in sections:
        if section.heading:
            doc.add_heading(section.heading, section.level)
        
        for paragraph in section.paragraphs:
            p = doc.add_paragraph()
            if isinstance(paragraph, str):
                p.add_run(paragraph)
            elif isinstance(paragraph, DocxText):
                run = p.add_run(paragraph.text)
                run.bold = paragraph.bold
                run.italic = paragraph.italic
                run.underline = paragraph.underline
        
        for image in section.images:
            try:
                image_stream = _load_image(image.uri)
                _add_picture_with_size(doc, image_stream, image.width, image.height)
                
                last_paragraph = doc.paragraphs[-1]
                _set_alignment(last_paragraph, image.alignment, PARAGRAPH_ALIGNMENTS)
                
                if image.caption:
                    caption_para = doc.add_paragraph(image.caption)
                    caption_para.alignment = last_paragraph.alignment
                    
            except Exception as e:
                doc.add_paragraph(f"[Image could not be loaded: {str(e)}]")
        
        for table_data in section.tables:
            try:
                add_table(doc, table_data)
            except Exception as e:
                doc.add_paragraph(f"[Error creating table: {str(e)}]")
    
    buffer = BytesIO()
    doc.save(buffer)
    file = File.validate(
        buffer.getvalue(), 
        info={
            "extension": ".docx", 
            "name": filename,
        },
    )
    return file
