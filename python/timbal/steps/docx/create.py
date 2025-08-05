import json
from io import BytesIO

from docx import Document

from ...types.file import File
from ...types.field import Field


def create_docx(content: str = Field(description=("""JSON string with document structure. Expected format:
    {
        "title": "Document Title" (optional),
        "sections": [
            {
                "heading": "Section Heading" (optional),
                "level": 1 (heading level, optional, default=1),
                "paragraphs": [
                    "Plain text paragraph",
                    {"text": "Bold text", "bold": true},
                    {"text": "Italic text", "italic": true}
                ] (optional)
            }
        ] (optional),
        "tables": [
            {
                "data": [
                    ["Header1", "Header2"],
                    ["Row1Col1", "Row1Col2"]
                ]
            }
        ] (optional)
    }"""))) -> File:
    """
    Generates a Word document (.docx) from a JSON structure containing title, sections, 
    and tables. Supports basic text formatting including bold and italic styling.
    
    Args:
        content: JSON string with document structure. Expected format:
            {
                "title": "Document Title" (optional),
                "sections": [
                    {
                        "heading": "Section Heading" (optional),
                        "level": 1 (heading level, optional, default=1),
                        "paragraphs": [
                            "Plain text paragraph",
                            {"text": "Bold text", "bold": true},
                            {"text": "Italic text", "italic": true}
                        ] (optional)
                    }
                ] (optional),
                "tables": [
                    {
                        "data": [
                            ["Header1", "Header2"],
                            ["Row1Col1", "Row1Col2"]
                        ]
                    }
                ] (optional)
            }
    
    Returns:
        File: File object containing the Word document bytes with .docx extension
        
    Raises:
        json.JSONDecodeError: If content is not valid JSON
        KeyError: If required table data structure is missing
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
                        # Handle formatted text
                        text = para.get("text", "")
                        if para.get("bold"):
                            p.add_run(text).bold = True
                        elif para.get("italic"):
                            p.add_run(text).italic = True
                        else:
                            p.add_run(text)
    
    if "tables" in doc_data:
        for table_data in doc_data["tables"]:
            rows = len(table_data["data"])
            cols = len(table_data["data"][0]) if rows > 0 else 0
            table = doc.add_table(rows=rows, cols=cols)
            
            for i, row in enumerate(table_data["data"]):
                for j, cell_text in enumerate(row):
                    table.cell(i, j).text = str(cell_text)
    
    buffer = BytesIO()
    doc.save(buffer)
    
    file_obj = File.validate(buffer.getvalue(), info={"extension": ".docx"})
    return file_obj
