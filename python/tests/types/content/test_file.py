import base64
import email.mime.multipart
import email.mime.text
import io
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from timbal.errors import ImageProcessingError, PDFProcessingError
from timbal.types import File
from timbal.types.content import FileContent, content_factory
from timbal.types.content.file import (
    _extract_email_attachments,
    _extract_email_body,
)


def test_basic_file_content_validation(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content)
    content = content_factory({"type": "file", "file": File.validate(str(test_file))})
    assert isinstance(content, FileContent)
    assert isinstance(content.file, File)
    assert content.type == "file"

    # file must be a File
    with pytest.raises(ValueError):
        content_factory({"type": "file", "file": "not a file"})


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_fc(data: bytes, extension: str) -> FileContent:
    """Create a fresh FileContent with the given bytes and extension."""
    file = File.validate(data, {"extension": extension})
    return FileContent(file=file)


# ---------------------------------------------------------------------------
# TestExtractEmailBody
# ---------------------------------------------------------------------------

class TestExtractEmailBody:
    def test_single_part_plain_text(self):
        raw = "Content-Type: text/plain\r\n\r\nHello world"
        result = _extract_email_body(raw)
        assert result == "Hello world"

    def test_multipart_prefers_plain_text(self):
        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg.attach(email.mime.text.MIMEText("plain body", "plain"))
        msg.attach(email.mime.text.MIMEText("<html>html body</html>", "html"))
        result = _extract_email_body(msg.as_string())
        assert result == "plain body"

    def test_multipart_html_fallback(self):
        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg.attach(email.mime.text.MIMEText("<html>only html</html>", "html"))
        result = _extract_email_body(msg.as_string())
        assert "<html>only html</html>" in result

    def test_empty_email_returns_empty_string(self):
        raw = "Content-Type: text/plain\r\n\r\n"
        result = _extract_email_body(raw)
        assert result == ""


# ---------------------------------------------------------------------------
# TestExtractEmailAttachments
# ---------------------------------------------------------------------------

class TestExtractEmailAttachments:
    def test_no_attachments(self):
        raw = "Content-Type: text/plain\r\n\r\nHello"
        result = _extract_email_attachments(raw)
        assert result == []

    def test_single_attachment(self):
        msg = email.mime.multipart.MIMEMultipart()
        msg.attach(email.mime.text.MIMEText("body", "plain"))
        attachment = email.mime.text.MIMEText("attachment content", "plain")
        attachment.add_header("Content-Disposition", "attachment", filename="test.txt")
        msg.attach(attachment)
        result = _extract_email_attachments(msg.as_string())
        assert len(result) == 1
        assert result[0]["filename"] == "test.txt"
        assert "content_type" in result[0]
        assert "data" in result[0]

    def test_attachment_with_content_id(self):
        msg = email.mime.multipart.MIMEMultipart()
        msg.attach(email.mime.text.MIMEText("body", "plain"))
        attachment = email.mime.text.MIMEText("image data", "plain")
        attachment.add_header("Content-Disposition", "attachment", filename="image.png")
        attachment["Content-ID"] = "<unique-image-id>"
        msg.attach(attachment)
        result = _extract_email_attachments(msg.as_string())
        assert len(result) == 1
        assert "content_id" in result[0]
        assert result[0]["content_id"] == "unique-image-id"


# ---------------------------------------------------------------------------
# TestValidatePdf
# ---------------------------------------------------------------------------

class TestValidatePdf:
    def test_valid_pdf_no_exception(self):
        mock_fitz = MagicMock()
        mock_doc = MagicMock()
        mock_fitz.Document.return_value = mock_doc
        file = File.validate(b"%PDF-1.4 fake", {"extension": ".pdf"})
        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            from timbal.types.content.file import validate_pdf
            # Should not raise
            validate_pdf(file)
        mock_doc.close.assert_called_once()

    def test_invalid_pdf_raises(self):
        mock_fitz = MagicMock()
        mock_fitz.Document.side_effect = Exception("bad pdf")
        file = File.validate(b"not a pdf", {"extension": ".pdf"})
        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            from timbal.types.content.file import validate_pdf
            with pytest.raises(PDFProcessingError):
                validate_pdf(file)


# ---------------------------------------------------------------------------
# TestValidateImage
# ---------------------------------------------------------------------------

class TestValidateImage:
    def test_jpeg(self):
        file = File.validate(b"\xff\xd8\xff" + b"\x00" * 20, {"extension": ".jpg"})
        from timbal.types.content.file import validate_image
        validate_image(file)  # should not raise

    def test_png(self):
        file = File.validate(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, {"extension": ".png"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_gif87a(self):
        file = File.validate(b"GIF87a" + b"\x00" * 20, {"extension": ".gif"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_gif89a(self):
        file = File.validate(b"GIF89a" + b"\x00" * 20, {"extension": ".gif"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_webp(self):
        file = File.validate(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10, {"extension": ".webp"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_bmp(self):
        file = File.validate(b"BM" + b"\x00" * 20, {"extension": ".bmp"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_tiff_little_endian(self):
        file = File.validate(b"II\x2a\x00" + b"\x00" * 20, {"extension": ".tiff"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_tiff_big_endian(self):
        file = File.validate(b"MM\x00\x2a" + b"\x00" * 20, {"extension": ".tiff"})
        from timbal.types.content.file import validate_image
        validate_image(file)

    def test_empty_file_raises(self):
        file = File.validate(b"", {"extension": ".png"})
        from timbal.types.content.file import validate_image
        with pytest.raises(ImageProcessingError, match="empty"):
            validate_image(file)

    def test_unrecognised_format_raises(self):
        file = File.validate(b"not-an-image-at-all", {"extension": ".png"})
        from timbal.types.content.file import validate_image
        with pytest.raises(ImageProcessingError, match="Unrecognised"):
            validate_image(file)

    def test_webp_riff_without_webp_marker_raises(self):
        # RIFF header but not WebP — should not validate
        file = File.validate(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 10, {"extension": ".webp"})
        from timbal.types.content.file import validate_image
        with pytest.raises(ImageProcessingError):
            validate_image(file)


# ---------------------------------------------------------------------------
# TestPdfToImages
# ---------------------------------------------------------------------------

class TestPdfToImages:
    def test_pdf_with_one_page(self, tmp_path):
        # Create a real temporary PNG so pix.save works
        fake_png = tmp_path / "page.png"
        fake_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_fitz = MagicMock()
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()

        mock_fitz.Document.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_page.get_pixmap.return_value = mock_pix

        # Make pix.save write bytes to the tmp file path passed to it
        def fake_save(path):
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_pix.save.side_effect = fake_save

        file = File.validate(b"%PDF-1.4 fake", {"extension": ".pdf"})
        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            from timbal.types.content.file import pdf_to_images
            pages = pdf_to_images(file)

        assert len(pages) == 1
        assert isinstance(pages[0], File)

    def test_invalid_pdf_raises(self):
        mock_fitz = MagicMock()
        mock_fitz.Document.side_effect = Exception("corrupt")
        file = File.validate(b"not-a-pdf", {"extension": ".pdf"})
        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            from timbal.types.content.file import pdf_to_images
            with pytest.raises(PDFProcessingError):
                pdf_to_images(file)


# ---------------------------------------------------------------------------
# TestExtractDocxContent
# ---------------------------------------------------------------------------

class TestExtractDocxContent:
    def test_docx_with_paragraphs(self):
        mock_docx_module = MagicMock()
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Hello paragraph"

        # Simulate body elements with one paragraph tag
        mock_element = MagicMock()
        mock_element.tag = "some_ns}p"  # ends with 'p'
        mock_doc.element.body = [mock_element]
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []
        mock_docx_module.Document.return_value = mock_doc

        file = File.validate(b"fake-docx", {"extension": ".docx"})
        with patch.dict(sys.modules, {"docx": mock_docx_module}):
            from timbal.types.content.file import _extract_docx_content
            result = _extract_docx_content(file)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestFileContentOpenAIResponses
# ---------------------------------------------------------------------------

class TestFileContentOpenAIResponses:
    def test_plain_text_file(self):
        fc = _make_fc(b"hello world", ".txt")
        result = fc.to_openai_responses_input()
        assert result["type"] == "input_text"
        assert "hello world" in result["text"]

    def test_image_valid(self):
        fc = _make_fc(b"fake-png-data", ".png")
        with patch("timbal.types.content.file.validate_image"):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_image"
        assert "image_url" in result

    def test_image_invalid_returns_error_text(self):
        fc = _make_fc(b"bad-png", ".png")
        with patch("timbal.types.content.file.validate_image", side_effect=ImageProcessingError("corrupt")):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_text"
        assert "[FILE ERROR:" in result["text"]

    def test_pdf_valid(self):
        file = File.validate(b"%PDF-1.4", {"extension": ".pdf", "name": "test.pdf"})
        fc = FileContent(file=file)
        with patch("timbal.types.content.file.validate_pdf"):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_file"
        assert "file_data" in result

    def test_pdf_invalid_returns_error_text(self):
        fc = _make_fc(b"bad-pdf", ".pdf")
        with patch("timbal.types.content.file.validate_pdf", side_effect=PDFProcessingError("bad")):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_text"
        assert "[FILE ERROR:" in result["text"]

    def test_audio_wav(self):
        fc = _make_fc(b"RIFF" + b"\x00" * 40, ".wav")
        result = fc.to_openai_responses_input()
        assert result["type"] == "input_audio"
        assert result["input_audio"]["format"] == "wav"

    def test_eml_raises_not_implemented(self):
        fc = _make_fc(b"Content-Type: text/plain\r\n\r\nHello", ".eml")
        with pytest.raises(NotImplementedError):
            fc.to_openai_responses_input()

    def test_docx_returns_text(self):
        fc = _make_fc(b"fake-docx", ".docx")
        with patch("timbal.types.content.file._extract_docx_content", return_value="doc content"):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_text"
        assert result["text"] == "doc content"

    def test_cached_result_is_same_object(self):
        fc = _make_fc(b"hello", ".txt")
        first = fc.to_openai_responses_input()
        second = fc.to_openai_responses_input()
        assert first is second


# ---------------------------------------------------------------------------
# TestFileContentOpenAIChatCompletions
# ---------------------------------------------------------------------------

class TestFileContentOpenAIChatCompletions:
    def test_plain_text_file(self):
        fc = _make_fc(b"hello world", ".txt")
        result = fc.to_openai_chat_completions_input()
        assert result["type"] == "text"
        assert "hello world" in result["text"]

    def test_image_valid(self):
        fc = _make_fc(b"fake-png-data", ".png")
        with patch("timbal.types.content.file.validate_image"):
            result = fc.to_openai_chat_completions_input()
        assert result["type"] == "image_url"
        assert "image_url" in result

    def test_image_invalid_returns_error_text(self):
        fc = _make_fc(b"bad-png", ".png")
        with patch("timbal.types.content.file.validate_image", side_effect=ImageProcessingError("corrupt")):
            result = fc.to_openai_chat_completions_input()
        assert result["type"] == "text"
        assert "[FILE ERROR:" in result["text"]

    def test_pdf_valid_returns_list_of_image_urls(self):
        fc = _make_fc(b"%PDF-1.4", ".pdf")
        mock_page = File.validate(b"fake-png", {"extension": ".png"})
        with patch("timbal.types.content.file.pdf_to_images", return_value=[mock_page]):
            result = fc.to_openai_chat_completions_input()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_pdf_invalid_returns_error_text(self):
        fc = _make_fc(b"bad-pdf", ".pdf")
        with patch("timbal.types.content.file.pdf_to_images", side_effect=PDFProcessingError("bad")):
            result = fc.to_openai_chat_completions_input()
        assert result["type"] == "text"
        assert "[FILE ERROR:" in result["text"]

    def test_eml_plain_text_returns_list(self):
        raw_eml = b"Content-Type: text/plain\r\n\r\nHello world"
        fc = _make_fc(raw_eml, ".eml")
        result = fc.to_openai_chat_completions_input()
        assert isinstance(result, list)
        assert any("Hello world" in str(item) for item in result)

    def test_cached_result_is_same_object(self):
        fc = _make_fc(b"hello", ".txt")
        first = fc.to_openai_chat_completions_input()
        second = fc.to_openai_chat_completions_input()
        assert first is second


# ---------------------------------------------------------------------------
# TestFileContentAnthropic
# ---------------------------------------------------------------------------

class TestFileContentAnthropic:
    def test_plain_text_file(self):
        fc = _make_fc(b"hello world", ".txt")
        result = fc.to_anthropic_input()
        assert result["type"] == "text"
        assert "hello world" in result["text"]

    def test_image_valid_base64(self):
        fc = _make_fc(b"fake-png-data", ".png")
        with patch("timbal.types.content.file.validate_image"):
            result = fc.to_anthropic_input()
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"

    def test_pdf_base64(self):
        fc = _make_fc(b"%PDF-1.4 fake content", ".pdf")
        result = fc.to_anthropic_input()
        assert result["type"] == "document"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "application/pdf"

    def test_docx_returns_text(self):
        fc = _make_fc(b"fake-docx", ".docx")
        with patch("timbal.types.content.file._extract_docx_content", return_value="docx content"):
            result = fc.to_anthropic_input()
        assert result["type"] == "text"
        assert result["text"] == "docx content"

    def test_cached_result_is_same_object(self):
        fc = _make_fc(b"hello", ".txt")
        first = fc.to_anthropic_input()
        second = fc.to_anthropic_input()
        assert first is second
