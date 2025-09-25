import base64
import io
from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import pandas as pd
import structlog
from docx import Document

from ..file import File
from .base import BaseContent

logger = structlog.get_logger("timbal.types.content")


AVAILABLE_ENCODINGS = [
    "utf-8",
    "cp1252",
    "iso-8859-1",
    "utf-16",
]


def pdf_to_images(pdf: File, dpi: int = 200) -> list[File]:
    """Convert a PDF file to a list of images files."""
    import tempfile
    from pathlib import Path

    import fitz
    pdf.seek(0) # Ensure the pointer is at the start of the file
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


class FileContent(BaseContent):
    """File content type for chat messages."""
    type: Literal["file"] = "file"
    file: File
    # TODO Change this to cached properties
    # Cached openai and anthropic inputs (some conversions are costly, e.g. audio transcriptions).
    _cached_openai_responses_input: Any | None = None
    _cached_openai_chat_completions_input: Any | None = None
    _cached_anthropic_input: Any | None = None

    @override
    def to_openai_responses_input(self, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
        """See base class."""
        if self._cached_openai_responses_input is not None:
            return self._cached_openai_responses_input

        mime = self.file.__content_type__
        
        # Ensure the file pointer is at the start of the file if we need to read it.
        current_position = self.file.tell()
        if current_position != 0:
            self.file.seek(0)

        if (
            (mime and mime.startswith("text/")) or 
            # Files like .jsonl don't have a mime type.
            (self.file.__source_extension__ in [".json", ".jsonl"])
        ):
            # Attempt to decode the binary content into a string guessing the encoding.
            raw_bytes = self.file.read()
            content = None
            for encoding in AVAILABLE_ENCODINGS:
                try:
                    content = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                raise ValueError(f"Could not decode file {self.file} with any of: {AVAILABLE_ENCODINGS}")
            openai_responses_input = {
                "type": "input_text",
                "text": content,
            }
            self._cached_openai_responses_input = openai_responses_input
            return openai_responses_input
        elif self.file.__source_extension__ == ".xlsx":
            df = pd.read_excel(io.BytesIO(self.file.read()))
            openai_responses_input = {
                "type": "input_text",
                "text": df.to_csv(index=False, header=True, sep=","),
            }
            self._cached_openai_responses_input = openai_responses_input
            return openai_responses_input
        elif self.file.__source_extension__ == ".docx":
            doc = Document(io.BytesIO(self.file.read()))
            text_content = []
            # Process document elements in order to maintain structure
            for element in doc.element.body:
                if element.tag.endswith("p"):
                    paragraph = doc.paragraphs[len([e for e in doc.element.body[:doc.element.body.index(element)] if e.tag.endswith("p")])]
                    if paragraph.text.strip():
                        text_content.append(paragraph.text.strip())
                elif element.tag.endswith("tbl"):
                    table = doc.tables[len([e for e in doc.element.body[:doc.element.body.index(element)] if e.tag.endswith('tbl')])]
                    table_content = []
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            table_content.append(",".join(row_text))
                    text_content.append("\n".join(table_content))
            text_content = "\n".join(text_content)
            openai_responses_input = {
                "type": "input_text",
                "text": text_content,
            }
            self._cached_openai_responses_input = openai_responses_input
            return openai_responses_input
        elif mime and mime.startswith("image/"):
            url = self.file.to_data_url()
            openai_responses_input = {
                "type": "input_image", 
                "image_url": url,
            }
            self._cached_openai_responses_input = openai_responses_input
            return openai_responses_input
        elif mime == "application/pdf":
            # TODO Review this one
            url = self.file.to_data_url()
            openai_responses_input = {
                "type": "input_file", 
                "filename": self.file.name,
                "file_data": url,
            }
            self._cached_openai_responses_input = openai_responses_input
            return openai_responses_input
        elif mime and mime.startswith("audio/"):
            # Some gpt models accept audio files as input. If the user is using a model that doesn't have this capability, the sdk will raise an error
            if self.file.__source_scheme__ == "data":
                base64_data = self.file.__source__.split(",", 1)[1]
            else:
                base64_data = base64.b64encode(self.file.read()).decode("utf-8")
            if "mp3" not in mime and "wav" not in mime:
                raise ValueError(f"Unsupported audio format: {mime}. Must be one of: mp3, wav")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data, 
                    "format": "wav" if "wav" in mime else "mp3",
                },
            }
        raise ValueError(f"Unsupported file {self.file}.")

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
        """See base class."""
        if self._cached_openai_chat_completions_input is not None:
            return self._cached_openai_chat_completions_input

        mime = self.file.__content_type__
        
        # Ensure the file pointer is at the start of the file if we need to read it.
        current_position = self.file.tell()
        if current_position != 0:
            self.file.seek(0)

        if (
            (mime and mime.startswith("text/")) or 
            # Files like .jsonl don't have a mime type.
            (self.file.__source_extension__ in [".json", ".jsonl"])
        ):
            # Attempt to decode the binary content into a string guessing the encoding.
            raw_bytes = self.file.read()
            content = None
            for encoding in AVAILABLE_ENCODINGS:
                try:
                    content = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                raise ValueError(f"Could not decode file {self.file} with any of: {AVAILABLE_ENCODINGS}")
            openai_input = {
                "type": "text",
                "text": content,
            }
            self._cached_openai_chat_completions_input = openai_input
            return openai_input
        elif self.file.__source_extension__ == ".xlsx":
            df = pd.read_excel(io.BytesIO(self.file.read()))
            openai_input = {
                "type": "text",
                "text": df.to_csv(index=False, header=True, sep=","),
            }
            self._cached_openai_chat_completions_input = openai_input
            return openai_input
        elif self.file.__source_extension__ == ".docx":
            doc = Document(io.BytesIO(self.file.read()))
            text_content = []
            # Process document elements in order to maintain structure
            for element in doc.element.body:
                if element.tag.endswith("p"):
                    paragraph = doc.paragraphs[len([e for e in doc.element.body[:doc.element.body.index(element)] if e.tag.endswith("p")])]
                    if paragraph.text.strip():
                        text_content.append(paragraph.text.strip())
                elif element.tag.endswith("tbl"):
                    table = doc.tables[len([e for e in doc.element.body[:doc.element.body.index(element)] if e.tag.endswith('tbl')])]
                    table_content = []
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            table_content.append(",".join(row_text))
                    text_content.append("\n".join(table_content))
            text_content = "\n".join(text_content)
            openai_input = {
                "type": "text",
                "text": text_content,
            }
            self._cached_openai_chat_completions_input = openai_input
            return openai_input
        elif mime and mime.startswith("image/"):
            url = self.file.to_data_url()
            return {
                "type": "image_url", 
                "image_url": {"url": url},
            }
        elif mime == "application/pdf":
            # TODO Review openai sdk "file" type
            # OpenAI internally gets an image of each page and complements it with the extracted text.
            # ? For all the use cases we've used, extracting just the images has worked well.
            pages = pdf_to_images(self.file)
            logger.info(
                "pdf_to_images", 
                n_pages=len(pages), 
                description=".to_openai_chat_completions_input() implicitly converting input pdf to images...",
            )
            pages_input = []
            for page in pages:
                url = page.to_data_url()
                pages_input.append({
                    "type": "image_url", 
                    "image_url": {"url": url},
                })
            self._cached_openai_chat_completions_input = pages_input
            return pages_input
        elif mime and mime.startswith("audio/"):
            # Some gpt models accept audio files as input. If the user is using a model that doesn't have this capability, the sdk will raise an error
            if self.file.__source_scheme__ == "data":
                base64_data = self.file.__source__.split(",", 1)[1]
            else:
                base64_data = base64.b64encode(self.file.read()).decode("utf-8")
            if "mp3" not in mime and "wav" not in mime:
                raise ValueError(f"Unsupported audio format: {mime}. Must be one of: mp3, wav")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data, 
                    "format": "wav" if "wav" in mime else "mp3",
                },
            }
        raise ValueError(f"Unsupported file {self.file}.")

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
        """See base class."""
        if self._cached_anthropic_input is not None:
            return self._cached_anthropic_input

        mime = self.file.__content_type__

        # Ensure the file pointer is at the start of the file if we need to read it.
        current_position = self.file.tell()
        if current_position != 0:
            self.file.seek(0)

        if (
            (mime and mime.startswith("text/")) or 
            # Files like .jsonl don't have a mime type.
            (self.file.__source_extension__ in [".json", ".jsonl"])
        ):
            # Attempt to decode the binary content into a string guessing the encoding.
            raw_bytes = self.file.read()
            content = None
            for encoding in AVAILABLE_ENCODINGS:
                try:
                    content = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                raise ValueError(f"Could not decode file {self.file} with any of: {AVAILABLE_ENCODINGS}")
            anthropic_input = {
                "type": "text",
                "text": content,
            }
            self._cached_anthropic_input = anthropic_input
            return anthropic_input
        elif self.file.__source_extension__ == ".xlsx":
            df = pd.read_excel(io.BytesIO(self.file.read()))
            anthropic_input = {
                "type": "text",
                "text": df.to_csv(index=False, header=True, sep=","),
            }
            self._cached_anthropic_input = anthropic_input
            return anthropic_input
        elif self.file.__source_extension__ == ".docx":
            doc = Document(io.BytesIO(self.file.read()))
            text_content = []
            # Process document elements in order to maintain structure
            for element in doc.element.body:
                if element.tag.endswith("p"):
                    paragraph = doc.paragraphs[len([e for e in doc.element.body[:doc.element.body.index(element)] if e.tag.endswith('p')])]
                    if paragraph.text.strip():
                        text_content.append(paragraph.text.strip())
                elif element.tag.endswith("tbl"):
                    table = doc.tables[len([e for e in doc.element.body[:doc.element.body.index(element)] if e.tag.endswith('tbl')])]
                    table_content = []
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            table_content.append(",".join(row_text))
                    text_content.append("\n".join(table_content))
            text_content = "\n".join(text_content)
            anthropic_input = {
                "type": "text",
                "text": text_content,
            }
            self._cached_anthropic_input = anthropic_input
            return anthropic_input
        elif mime and mime.startswith("image/"):
            # TODO Review this one
            url = self.file.to_data_url()
            if url.startswith("data:"):
                base64_data = url.split(",", 1)[1]
                return {
                    "type": "image", 
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": base64_data,
                    }
                }
            else:
                return {
                    "type": "image", 
                    "source": {
                        "type": "url",
                        "url": url,
                    }
                }
        elif mime == "application/pdf":
            # TODO Review this one
            url = self.file.to_data_url()
            if url.startswith("data:"):
                base64_data = url.split(",", 1)[1]
                return {
                    "type": "document", 
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": base64_data,
                    }
                }
            else:
                return {
                    "type": "document", 
                    "source": {
                        "type": "url",
                        "url": url,
                    }
                }
        raise ValueError(f"Unsupported file {self.file}.")
