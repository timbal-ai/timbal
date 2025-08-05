import base64
import json
from io import BytesIO
from urllib.parse import urlparse

import requests
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
from docx.shared import Inches

from ...types.field import Field
from ...types.file import File

DOCUMENT_FORMAT_SPEC = """JSON string with document structure. Expected format:
{
    "title": "Document Title" (optional),
    "sections": [
        {
            "heading": "Section Heading" (optional),
            "level": 1 (heading level, optional, default=1),
            "paragraphs": [
                "Plain text paragraph",
                {"text": "Bold text", "bold": true},
                {"text": "Italic text", "italic": true},
                {"text": "Underlined text", "underline": true}
            ] (optional),
            "images": [
                {
                    "path": "/path/to/image.jpg or https://example.com/image.jpg" (local path or URL),
                    "base64_data": "base64-encoded-image-data" (alternative to path),
                    "width": 6.0 (inches, optional),
                    "height": 4.0 (inches, optional),
                    "caption": "Image caption" (optional),
                    "alignment": "center" (left/center/right, optional)
                }
            ] (optional),
                            "tables": [
                    {
                        "data": [
                            ["Header1", "Header2"], 
                            ["Row1Col1", "Row1Col2"],
                            ["Text", {"path": "/path/to/image.jpg", "width": 2.0, "alignment": "center"}]
                        ],
                        "style": "Table Grid" (optional),
                        "header_row": true (optional),
                        "alignment": "center" (optional),
                        "borders": true (optional),
                        "cell_alignment": "center" (optional)
                    }
                ] (optional)
        }
    ] (optional),
}"""


def create_table(doc, table_data):
    """Create a single table with all styling and content options."""
    try:
        if "data" not in table_data:
            doc.add_paragraph("[Error: Table data missing 'data' field]")
            return
            
        data = table_data["data"]
        if not data or not isinstance(data, list):
            doc.add_paragraph("[Error: Table data is empty or not a list]")
            return
            
        rows = len(data)
        cols = len(data[0]) if rows > 0 and isinstance(data[0], list) else 0
        
        if rows == 0 or cols == 0:
            doc.add_paragraph("[Error: Table has no rows or columns]")
            return
        
        table = doc.add_table(rows=rows, cols=cols)
        
        table_style = table_data.get("style", "Table Grid")
        try:
            table.style = table_style
        except Exception:
            try:
                table.style = "Table Grid"
            except Exception:
                pass
        
        alignment = table_data.get("alignment", "left").lower()
        try:
            if alignment == "center":
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
            elif alignment == "right":
                table.alignment = WD_TABLE_ALIGNMENT.RIGHT
        except Exception:
            pass
        
        for i, row in enumerate(data):
            if not isinstance(row, list):
                continue
            for j, cell_content in enumerate(row):
                if j < cols:
                    try:
                        cell = table.cell(i, j)
                        if isinstance(cell_content, dict) and ("path" in cell_content or "base64_data" in cell_content):
                            cell.text = ""
                            cell_paragraph = cell.paragraphs[0]
                            
                            try:
                                if "base64_data" in cell_content:
                                    image_data = base64.b64decode(cell_content["base64_data"])
                                    image_stream = BytesIO(image_data)
                                elif "path" in cell_content:
                                    path = cell_content["path"]
                                    parsed = urlparse(path)
                                    if parsed.scheme in ('http', 'https'):
                                        response = requests.get(path, timeout=30)
                                        response.raise_for_status()
                                        image_stream = BytesIO(response.content)
                                    else:
                                        image_stream = path
                                else:
                                    cell.text = str(cell_content)
                                    continue

                                width = cell_content.get("width")
                                height = cell_content.get("height")
                                
                                if width and height:
                                    run = cell_paragraph.runs[0] if cell_paragraph.runs else cell_paragraph.add_run()
                                    run.add_picture(image_stream, width=Inches(width), height=Inches(height))
                                elif width:
                                    run = cell_paragraph.runs[0] if cell_paragraph.runs else cell_paragraph.add_run()
                                    run.add_picture(image_stream, width=Inches(width))
                                elif height:
                                    run = cell_paragraph.runs[0] if cell_paragraph.runs else cell_paragraph.add_run()
                                    run.add_picture(image_stream, height=Inches(height))
                                else:
                                    run = cell_paragraph.runs[0] if cell_paragraph.runs else cell_paragraph.add_run()
                                    run.add_picture(image_stream)
                                
                                img_alignment = cell_content.get("alignment", "left").lower()
                                if img_alignment == "center":
                                    cell_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                elif img_alignment == "right":
                                    cell_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                                
                            except Exception as e:
                                cell.text = f"[Image error: {str(e)}]"
                        else:
                            cell.text = str(cell_content)
                        
                        cell_alignment = table_data.get("cell_alignment", "left").lower()
                        if cell_alignment == "center":
                            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                            for paragraph in cell.paragraphs:
                                if not any(run.element.xpath('.//pic:pic') for run in paragraph.runs):
                                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        elif cell_alignment == "right":
                            for paragraph in cell.paragraphs:
                                if not any(run.element.xpath('.//pic:pic') for run in paragraph.runs):
                                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    except Exception:
                        continue
        
        if table_data.get("header_row", False) and rows > 0:
            try:
                for j in range(cols):
                    cell = table.cell(0, j)
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.bold = True
            except Exception:
                pass
        
        if "column_widths" in table_data:
            try:
                widths = table_data["column_widths"]
                for j, width in enumerate(widths):
                    if j < cols:
                        for i in range(rows):
                            table.cell(i, j).width = Inches(width)
            except Exception:
                pass
        
        if table_data.get("borders", True):
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
                
    except Exception as e:
        doc.add_paragraph(f"[Error creating table: {str(e)}]")


def create_docx(
    content: str = Field(description=DOCUMENT_FORMAT_SPEC), 
    filename: str = Field(description="Optional filename for the generated document. Without .docx extension", default="document")
) -> File:
    """
    Generates a Word document (.docx) from a JSON structure containing title, sections, 
    images, and enhanced tables. Supports text formatting, image embedding, and advanced table styling.
    
    Args:
        content: JSON string with document structure. Enhanced format supports:
            - Basic text formatting (bold, italic, underline)
            - Image embedding from file paths or base64 data
            - Advanced table styling with borders, alignment, column widths
            - Image captions and positioning
        filename: Optional filename for the generated document (default: "document.docx")
    
    Returns:
        File: File object containing the Word document bytes with .docx extension
        
    Raises:
        json.JSONDecodeError: If content is not valid JSON
        KeyError: If required table data structure is missing
        FileNotFoundError: If image path is specified but file doesn't exist
    """
    doc_data = json.loads(content)
    doc = Document()
    
    if "title" in doc_data:
        doc.add_heading(doc_data["title"], 0)
    
    if "sections" in doc_data:
        for section in doc_data["sections"]:
            if "heading" in section:
                level = section.get("level", 1)
                doc.add_heading(section["heading"], level)
            
            if "paragraphs" in section:
                for para in section["paragraphs"]:
                    p = doc.add_paragraph()
                    if isinstance(para, str):
                        p.add_run(para)
                    elif isinstance(para, dict):
                        text = para.get("text", "")
                        run = p.add_run(text)
                        if para.get("bold"):
                            run.bold = True
                        if para.get("italic"):
                            run.italic = True
                        if para.get("underline"):
                            run.underline = True
            
            if "images" in section:
                for img_data in section["images"]:
                    try:
                        if "base64_data" in img_data:
                            image_data = base64.b64decode(img_data["base64_data"])
                            image_stream = BytesIO(image_data)
                        elif "path" in img_data:
                            path = img_data["path"]
                            parsed = urlparse(path)
                            if parsed.scheme in ('http', 'https'):
                                response = requests.get(path, timeout=30)
                                response.raise_for_status()
                                image_stream = BytesIO(response.content)
                            else:
                                image_stream = path
                        else:
                            continue
                        
                        width = img_data.get("width")
                        height = img_data.get("height")
                        
                        if width and height:
                            doc.add_picture(image_stream, width=Inches(width), height=Inches(height))
                        elif width:
                            doc.add_picture(image_stream, width=Inches(width))
                        elif height:
                            doc.add_picture(image_stream, height=Inches(height))
                        else:
                            doc.add_picture(image_stream)
                        
                        alignment = img_data.get("alignment", "left").lower()
                        last_paragraph = doc.paragraphs[-1]
                        if alignment == "center":
                            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        elif alignment == "right":
                            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                        
                        if "caption" in img_data:
                            caption_para = doc.add_paragraph(img_data["caption"])
                            caption_para.alignment = last_paragraph.alignment
                            
                    except Exception as e:
                        doc.add_paragraph(f"[Image could not be loaded: {str(e)}]")
            
            if "tables" in section:
                for table_data in section["tables"]:
                    create_table(doc, table_data)
    
    if "tables" in doc_data:
        for table_data in doc_data["tables"]:
            create_table(doc, table_data)
    
    buffer = BytesIO()
    doc.save(buffer)
    file_obj = File.validate(buffer.getvalue(), info={"extension": ".docx", "name": filename})
    return file_obj
