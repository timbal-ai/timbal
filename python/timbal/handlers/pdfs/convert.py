import tempfile
from pathlib import Path

import fitz

from ...types.field import Field
from ...types.file import File


def convert_pdf_to_images(
    pdf: File = Field(description="The PDF file to convert to images."),
    dpi: int = 200,
) -> list[File]:

    pdf.seek(0)
    doc = fitz.Document(stream=pdf.read())

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        # TODO Use File.validate(bytes, {"extension": ".png"})
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        pix.save(tmp_path)
        pix_file = File.validate(tmp_path)
        pages.append(pix_file)

    return pages
