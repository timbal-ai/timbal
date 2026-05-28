import base64
import email.mime.multipart
import email.mime.text
import io
import json
import os
import pathlib
import struct
import sys
import zlib
from unittest.mock import MagicMock, patch

import pytest
from timbal.errors import ImageProcessingError, PDFProcessingError
from timbal.types import File
from timbal.types.content import FileContent, content_factory
from timbal.types.content.file import (
    _extract_email_attachments,
    _extract_email_body,
)


def _make_solid_png(size: int = 32, rgb: tuple[int, int, int] = (255, 128, 64)) -> bytes:
    """Generate a valid solid-color RGB PNG without external deps.

    Anthropic rejects images below a few pixels with ``Could not process image``,
    so use 32x32 as the minimum for cross-provider tests.
    """
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    pixel = bytes(rgb)
    raw = b"".join(b"\x00" + pixel * size for _ in range(size))
    idat = _chunk(b"IDAT", zlib.compress(raw, 9))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_PNG_BYTES = _make_solid_png()

# Real-API integration test matrix: (model, max_tokens, env_key)
# Mirrors the pattern used in test_output_model.py so the same models get coverage.
_REAL_LLM_MATRIX = [
    pytest.param("openai/gpt-4o-mini", None, "OPENAI_API_KEY", id="openai"),
    pytest.param("anthropic/claude-haiku-4-5", 1024, "ANTHROPIC_API_KEY", id="anthropic"),
    pytest.param("google/gemini-2.5-flash-lite", None, "GEMINI_API_KEY", id="google"),
]


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
        assert result["filename"] == "test.pdf"

    def test_pdf_anonymous_bytes_filename_is_string(self):
        """Regression: anonymous bytes PDFs must still send a string filename.

        ``_safe_file_name`` returns None for bytes without a name attr and for
        data URLs, but the OpenAI Responses API rejects null filenames.
        """
        fc = _make_fc(b"%PDF-1.4 stub", ".pdf")
        assert fc.name is None
        with patch("timbal.types.content.file.validate_pdf"):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_file"
        assert isinstance(result["filename"], str) and result["filename"], (
            "filename must be a non-empty string even when display name is unknown"
        )
        assert result["filename"].endswith(".pdf")

    def test_pdf_data_url_filename_is_string(self):
        """Regression: data-URL PDFs must still send a string filename."""
        data_url = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4 stub").decode()
        fc = FileContent(file=File(data_url))
        assert fc.name is None
        with patch("timbal.types.content.file.validate_pdf"):
            result = fc.to_openai_responses_input()
        assert result["type"] == "input_file"
        assert isinstance(result["filename"], str) and result["filename"]
        assert result["filename"].endswith(".pdf")

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


# ---------------------------------------------------------------------------
# TestFileContentName — name defaulting must not trigger File I/O
# ---------------------------------------------------------------------------

class TestFileContentName:
    def test_name_defaults_from_local_path(self, tmp_path: pathlib.Path) -> None:
        test_file = tmp_path / "Q3 Report.pdf"
        test_file.write_bytes(b"%PDF-1.4 stub")
        fc = FileContent(file=File.validate(str(test_file)))
        assert fc.name == "Q3 Report.pdf"

    def test_name_defaults_from_url_basename(self) -> None:
        fc = FileContent(file=File("https://content.timbal.ai/tmp/abc/Q3_Report.pdf"))
        assert fc.name == "Q3_Report.pdf"

    def test_name_url_basename_is_url_decoded(self) -> None:
        fc = FileContent(file=File("https://example.com/files/Q3%20Report.pdf"))
        assert fc.name == "Q3 Report.pdf"

    def test_explicit_name_wins_over_file_name(self, tmp_path: pathlib.Path) -> None:
        test_file = tmp_path / "raw_uuid.pdf"
        test_file.write_bytes(b"%PDF-1.4 stub")
        fc = FileContent(file=File.validate(str(test_file)), name="Display Name.pdf")
        assert fc.name == "Display Name.pdf"

    def test_name_none_for_anonymous_bytes(self) -> None:
        fc = FileContent(file=File.validate(b"hello", {"extension": ".txt"}))
        assert fc.name is None

    def test_name_from_named_bytes(self) -> None:
        bio = io.BytesIO(b"hello")
        bio.name = "greeting.txt"
        fc = FileContent(file=File(bio, extension=".txt"))
        assert fc.name == "greeting.txt"

    def test_name_none_for_data_url(self) -> None:
        data_url = "data:text/plain;base64," + base64.b64encode(b"hi").decode()
        fc = FileContent(file=File(data_url))
        assert fc.name is None

    def test_construction_does_not_trigger_fetcher_for_local_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Regression: building FileContent must NOT read the file from disk.

        Reading ``self.file.name`` proxies through File.__wrapped__ which calls
        the fetcher and slurps the entire file. For URLs that means a sync
        network round-trip. The validator must extract the name from source
        metadata instead.
        """
        test_file = tmp_path / "big.bin"
        test_file.write_bytes(b"\x00" * 1024)
        file_obj = File.validate(str(test_file))

        assert object.__getattribute__(file_obj, "__fileobj__") is None
        _ = FileContent(file=file_obj)
        assert object.__getattribute__(file_obj, "__fileobj__") is None, (
            "FileContent construction must not load the file"
        )

    def test_construction_does_not_trigger_fetcher_for_url(self) -> None:
        file_obj = File("https://example.com/some/file.pdf")
        assert object.__getattribute__(file_obj, "__fileobj__") is None
        _ = FileContent(file=file_obj)
        assert object.__getattribute__(file_obj, "__fileobj__") is None, (
            "FileContent construction must not fetch the URL"
        )


# ---------------------------------------------------------------------------
# TestFileContentSerialization — dump shape + round-trip via content_factory
# ---------------------------------------------------------------------------

class TestFileContentSerialization:
    async def test_dump_shape_includes_name(self, tmp_path: pathlib.Path) -> None:
        from timbal.utils.serialization import dump

        test_file = tmp_path / "Q3 Report.pdf"
        test_file.write_bytes(b"%PDF-1.4 stub")
        fc = FileContent(file=File.validate(str(test_file)))

        dumped = await dump(fc)
        assert dumped["type"] == "file"
        assert dumped["name"] == "Q3 Report.pdf"
        assert isinstance(dumped["file"], str)
        assert set(dumped.keys()) == {"type", "name", "file"}, (
            "dump must not leak private attrs like _cached_*"
        )

    async def test_dump_omits_name_when_none(self) -> None:
        from timbal.utils.serialization import dump

        fc = FileContent(file=File.validate(b"hello", {"extension": ".txt"}))
        assert fc.name is None

        dumped = await dump(fc)
        assert "name" not in dumped
        assert dumped["type"] == "file"
        assert isinstance(dumped["file"], str)

    async def test_dump_explicit_name_preserved(self) -> None:
        from timbal.utils.serialization import dump

        fc = FileContent(
            file=File.validate(b"hello", {"extension": ".txt"}),
            name="hello.txt",
        )
        dumped = await dump(fc)
        assert dumped["name"] == "hello.txt"

    async def test_dump_does_not_leak_cached_attrs(self) -> None:
        """Calling to_*_input populates the _cached_* private attrs.

        Those must not appear in dump output regardless of cache state.
        """
        from timbal.utils.serialization import dump

        fc = FileContent(
            file=File.validate(b"hello world", {"extension": ".txt"}),
            name="hello.txt",
        )
        _ = fc.to_anthropic_input()
        _ = fc.to_openai_responses_input()
        _ = fc.to_openai_chat_completions_input()
        assert fc._cached_anthropic_input is not None
        assert fc._cached_openai_responses_input is not None
        assert fc._cached_openai_chat_completions_input is not None

        dumped = await dump(fc)
        assert set(dumped.keys()) == {"type", "name", "file"}

    async def test_round_trip_preserves_name(self, tmp_path: pathlib.Path) -> None:
        from timbal.utils.serialization import dump

        test_file = tmp_path / "Q3 Report.pdf"
        test_file.write_bytes(b"%PDF-1.4 stub")
        fc = FileContent(
            file=File.validate(str(test_file)),
            name="Quarterly Report Q3.pdf",
        )

        dumped = await dump(fc)
        restored = content_factory(dumped)
        assert isinstance(restored, FileContent)
        assert restored.name == "Quarterly Report Q3.pdf"

    def test_content_factory_accepts_name_field(self) -> None:
        restored = content_factory({
            "type": "file",
            "name": "Display.pdf",
            "file": "https://example.com/x/y.pdf",
        })
        assert isinstance(restored, FileContent)
        assert restored.name == "Display.pdf"

    def test_content_factory_backward_compat_old_traces(self) -> None:
        """Old traces written before the name field exists must still load.

        Name defaults from the URL basename via the validator.
        """
        restored = content_factory({
            "type": "file",
            "file": "https://example.com/files/legacy.pdf",
        })
        assert isinstance(restored, FileContent)
        assert restored.name == "legacy.pdf"

    async def test_dump_inside_message(self) -> None:
        """End-to-end: FileContent inside a Message round-trips with name."""
        from timbal.types.message import Message
        from timbal.utils.serialization import dump

        fc = FileContent(
            file=File.validate(b"data", {"extension": ".pdf"}),
            name="Q3.pdf",
        )
        msg = Message(role="user", content=[fc])

        dumped = await dump(msg)
        assert dumped["role"] == "user"
        assert len(dumped["content"]) == 1
        assert dumped["content"][0]["type"] == "file"
        assert dumped["content"][0]["name"] == "Q3.pdf"
        assert "_cached_anthropic_input" not in dumped["content"][0]

        restored = Message.validate(dumped)
        assert isinstance(restored.content[0], FileContent)
        assert restored.content[0].name == "Q3.pdf"


# ---------------------------------------------------------------------------
# TestFileContentSessionChainIntegration
# Full prod-like pipeline: Agent → JSONL trace → reload → next turn.
# Exercises every layer that can drop the `name` field: serialization,
# JSONL round-trip, content_factory, Message rebuild, second LLM call.
# ---------------------------------------------------------------------------

class TestFileContentSessionChainIntegration:
    async def test_files_persist_through_jsonl_session_chain(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Turn 1 sends text + 3 files of different source schemes, gets persisted
        to a JSONL trace, then turn 2 reloads from disk via ``parent_id`` and the
        LLM sees the original FileContent objects (with names) in its memory.

        Catches regressions in:
          - serialization.dump() for FileContent (name + no cache leakage)
          - JSONL write/read of nested file metadata
          - content_factory accepting the new shape on reload
          - Message validation preserving FileContent.name
          - Agent memory chain across runs not stripping content metadata
        """
        from timbal import Agent
        from timbal.core.test_model import TestModel
        from timbal.state.tracing.providers import JsonlTracingProvider
        from timbal.types.content import TextContent
        from timbal.types.message import Message

        local_pdf = tmp_path / "Q3 Report.pdf"
        local_pdf.write_bytes(b"%PDF-1.4 stub")

        captured_turns: list[list[Message]] = []

        def handler(messages: list[Message]) -> str:
            captured_turns.append(list(messages))
            return f"acknowledged turn {len(captured_turns)}"

        traces_path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces_path)

        agent = Agent(
            name="file_agent",
            model=TestModel(handler=handler),
            tracing_provider=provider,
        )

        turn1_msg = Message(
            role="user",
            content=[
                TextContent(text="Please review these documents."),
                # local path — name auto-defaults to file basename
                FileContent(file=File.validate(str(local_pdf))),
                # anonymous bytes with explicit display name (the platform upload case)
                FileContent(
                    file=File.validate(b"raw bytes", {"extension": ".txt"}),
                    name="user_uploaded_notes.txt",
                ),
                # URL — name auto-defaults from (decoded) URL basename
                FileContent(file=File("https://example.com/files/spec%20v2.pdf")),
            ],
        )
        out1 = await agent(messages=[turn1_msg]).collect()
        assert out1.status.code == "success", out1.error

        # ----- inspect the on-disk JSONL trace after turn 1 -----
        assert traces_path.exists()
        records = [
            json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()
        ]
        assert len(records) == 1
        agent_span = next(s for s in records[0]["spans"] if s["path"] == "file_agent")
        stored_memory = agent_span["memory"]

        stored_user = stored_memory[0]
        assert stored_user["role"] == "user"
        file_blocks = [c for c in stored_user["content"] if c["type"] == "file"]
        assert len(file_blocks) == 3

        assert sorted(b.get("name") for b in file_blocks) == [
            "Q3 Report.pdf",
            "spec v2.pdf",
            "user_uploaded_notes.txt",
        ]
        for b in file_blocks:
            assert set(b.keys()) == {"type", "name", "file"}, (
                f"unexpected keys leaked to JSONL (private attrs?): {sorted(b.keys())}"
            )
            assert isinstance(b["file"], str) and b["file"], (
                "file ref must serialize to a non-empty string"
            )

        # ----- turn 2: provider reloads turn 1 from disk via parent_id -----
        out2 = await agent(
            prompt="Anything else I should know?",
            parent_id=out1.run_id,
        ).collect()
        assert out2.status.code == "success", out2.error
        assert out2.error is None

        # Two LLM invocations
        assert len(captured_turns) == 2

        # Turn 2 messages: [reloaded turn-1 user, reloaded turn-1 assistant, new turn-2 user]
        turn2_messages = captured_turns[1]
        assert len(turn2_messages) == 3
        reloaded_user = turn2_messages[0]
        assert reloaded_user.role == "user"

        reloaded_files = [c for c in reloaded_user.content if isinstance(c, FileContent)]
        assert len(reloaded_files) == 3
        assert sorted(fc.name for fc in reloaded_files) == [
            "Q3 Report.pdf",
            "spec v2.pdf",
            "user_uploaded_notes.txt",
        ], "filename metadata lost across JSONL reload"

        # Reloaded FileContent must be usable downstream — to_*_input must not raise.
        # The bytes file was persisted to a temp path on dump; the local pdf likewise
        # round-trips as a local path. Skip the URL one (would trigger fetch).
        local_fc = next(fc for fc in reloaded_files if fc.name == "Q3 Report.pdf")
        bytes_fc = next(fc for fc in reloaded_files if fc.name == "user_uploaded_notes.txt")

        with patch("timbal.types.content.file.validate_pdf"):
            local_res = local_fc.to_openai_responses_input()
        assert local_res["filename"] == "Q3 Report.pdf", (
            "explicit name on reloaded FileContent must flow into provider input"
        )

        bytes_res = bytes_fc.to_anthropic_input()
        assert bytes_res["type"] == "text"

        # ----- verify turn 2 disk record still has all the file metadata -----
        records = [
            json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()
        ]
        turn2_record = next(r for r in records if r["run_id"] == out2.run_id)
        turn2_agent_span = next(s for s in turn2_record["spans"] if s["path"] == "file_agent")
        turn2_memory = turn2_agent_span["memory"]
        assert len(turn2_memory) == 4, (
            f"expected 4 messages (turn1 u/a, turn2 u/a), got {len(turn2_memory)}"
        )

        turn2_files = [c for c in turn2_memory[0]["content"] if c["type"] == "file"]
        assert len(turn2_files) == 3
        assert sorted(b["name"] for b in turn2_files) == [
            "Q3 Report.pdf",
            "spec v2.pdf",
            "user_uploaded_notes.txt",
        ]
        for b in turn2_files:
            assert set(b.keys()) == {"type", "name", "file"}


# ---------------------------------------------------------------------------
# TestFileContentRealLLMIntegration
# Hits real OpenAI / Anthropic / Gemini APIs. Skipped when API keys are absent.
# Run explicitly with:  uv run pytest -m integration python/tests/types/content/test_file.py
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFileContentRealLLMIntegration:
    @pytest.mark.parametrize("model,max_tokens,env_key", _REAL_LLM_MATRIX)
    async def test_image_persists_through_jsonl_session_chain_real_llm(
        self,
        tmp_path: pathlib.Path,
        model: str,
        max_tokens: int | None,
        env_key: str,
    ) -> None:
        """End-to-end with a real LLM provider.

        Sends an image with an explicit display name, persists to JSONL, then
        runs a second turn that forces the provider to reload turn 1 from disk
        and convert the reloaded ``FileContent`` back into provider-native input.
        If serialization, ``content_factory``, or the agent memory chain ever
        loses the file ref or its ``name``, turn 2 will fail (provider error or
        missing-key crash) and this test will catch it.
        """
        if not os.getenv(env_key):
            pytest.skip(f"{env_key} not set")

        from timbal import Agent
        from timbal.state.tracing.providers import JsonlTracingProvider
        from timbal.types.content import TextContent
        from timbal.types.message import Message

        traces_path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces_path)

        agent_kwargs: dict = {
            "name": "real_llm_file_agent",
            "model": model,
            "tracing_provider": provider,
        }
        if max_tokens is not None:
            agent_kwargs["max_tokens"] = max_tokens

        agent = Agent(**agent_kwargs)

        display_name = "company_logo.png"
        turn1_msg = Message(
            role="user",
            content=[
                TextContent(
                    text="I'm attaching an image. Reply with one short sentence acknowledging it."
                ),
                FileContent(
                    file=File.validate(_PNG_BYTES, {"extension": ".png"}),
                    name=display_name,
                ),
            ],
        )
        out1 = await agent(messages=[turn1_msg]).collect()
        assert out1.status.code == "success", out1.error
        assert out1.error is None

        # JSONL: the image block must have the display name and no leaked attrs.
        records = [
            json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()
        ]
        agent_span = next(
            s for s in records[-1]["spans"] if s["path"] == "real_llm_file_agent"
        )
        user_msg = agent_span["memory"][0]
        file_blocks = [c for c in user_msg["content"] if c["type"] == "file"]
        assert len(file_blocks) == 1
        assert file_blocks[0]["name"] == display_name
        assert set(file_blocks[0].keys()) == {"type", "name", "file"}, (
            f"private attrs leaked into trace: {sorted(file_blocks[0].keys())}"
        )

        # Turn 2: forces JSONL reload + real provider re-ingestion of the file.
        out2 = await agent(
            prompt="Just say 'ok'.",
            parent_id=out1.run_id,
        ).collect()
        assert out2.status.code == "success", out2.error
        assert out2.error is None, (
            f"second turn failed — file ref likely broken across reload: {out2.error}"
        )

        # JSONL after turn 2: original file metadata must still be intact.
        records = [
            json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()
        ]
        turn2_record = next(r for r in records if r["run_id"] == out2.run_id)
        turn2_agent_span = next(
            s for s in turn2_record["spans"] if s["path"] == "real_llm_file_agent"
        )
        turn2_memory = turn2_agent_span["memory"]
        assert len(turn2_memory) >= 4, (
            f"expected turn1 u/a + turn2 u/a in memory, got {len(turn2_memory)}"
        )

        reloaded_file_blocks = [
            c for c in turn2_memory[0]["content"] if c["type"] == "file"
        ]
        assert len(reloaded_file_blocks) == 1
        assert reloaded_file_blocks[0]["name"] == display_name, (
            "display name lost across two-turn JSONL round-trip"
        )
        assert set(reloaded_file_blocks[0].keys()) == {"type", "name", "file"}
