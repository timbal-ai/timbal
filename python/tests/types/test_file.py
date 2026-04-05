import pathlib
from io import BytesIO

import pytest
from timbal.types import File


def test_file_validate_none() -> None:
    with pytest.raises(ValueError):
        File.validate(None)


def test_file_validate_any() -> None:
    with pytest.raises(ValueError):
        File.validate(object())


def test_file_serialize_none() -> None:
    assert File.serialize(None) is None


def test_file_serialize_any() -> None:
    with pytest.raises(ValueError):
        File.serialize(object())


def test_file_validate_bytes() -> None:
    test_content = b"Hello, World!"
    file = File.validate(test_content)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "bytes"
    assert file.__source_extension__ is None
    assert file.read() == test_content


def test_file_validate_bytearray() -> None:
    test_content = bytearray(b"Hello, World!")
    file = File.validate(test_content)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "bytes"
    assert file.__source_extension__ is None
    assert file.read() == bytes(test_content)


def test_file_validate_io_base() -> None:
    test_content = b"Hello, World!"
    test_io = BytesIO(test_content)
    file = File.validate(test_io)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "bytes"
    assert file.__source_extension__ is None
    assert file.read() == test_content


def test_file_serialize_byteslike() -> None:
    test_content = b"Hello, World!"
    test_io = BytesIO(test_content)

    file = File.validate(test_io)

    assert file.to_data_url() == "data:application/octet-stream;base64,SGVsbG8sIFdvcmxkIQ=="


def test_file_validate_local_path_from_str(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    file = File.validate(str(test_file))

    assert isinstance(file, File)
    assert file.__source_scheme__ == "local_path"
    assert file.__source_extension__ == test_file.suffix
    assert file.read() == test_content


def test_file_validate_local_path_from_path(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    file = File.validate(test_file)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "local_path"
    assert file.__source_extension__ == test_file.suffix
    assert file.read() == test_content


def test_file_serialize_local_path(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    file = File.validate(str(test_file))

    assert file.to_data_url() == "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ=="


def test_file_validate_data_url() -> None:
    test_content = b"Hello, World!"
    data_url = "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ=="

    file = File.validate(data_url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "data_url"
    assert file.__source_extension__ == ".txt"
    assert file.read() == test_content


def test_file_validate_data_url_with_custom_mime_type() -> None:
    test_content = b'{"key": "value"}'
    data_url = "data:timbal/jsonl;base64,eyJrZXkiOiAidmFsdWUifQ=="

    file = File.validate(data_url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "data_url"
    assert file.__source_extension__ == ".jsonl"
    assert file.read() == test_content


def test_file_serialize_data_url() -> None:
    data_url = "data:timbal/jsonl;base64,eyJrZXkiOiAidmFsdWUifQ=="

    file = File.validate(data_url)

    assert file.to_data_url() == data_url


def test_file_validate_url() -> None:
    test_content = b"Hello, World!\n"
    url = "https://content.timbal.ai/assets/hello_world.txt"

    file = File.validate(url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "url"
    assert file.__source_extension__ == ".txt"
    assert file.read() == test_content


def test_file_serialize_url() -> None:
    url = "https://content.timbal.ai/assets/hello_world.txt"

    file = File.validate(url)

    assert file.to_data_url() == "data:text/plain;base64,SGVsbG8sIFdvcmxkIQo="


# ---------------------------------------------------------------------------
# TestFileInit — content_type detection for bytes-sourced files
# ---------------------------------------------------------------------------


class TestFileInit:
    def test_bytes_with_md_extension_content_type(self) -> None:
        """Line 85: .md extension falls back to text/markdown on Python 3.11."""
        file = File.validate(b"# Hello", info={"extension": ".md"})
        assert file.__content_type__ == "text/markdown"

    def test_bytes_with_unknown_extension_content_type(self) -> None:
        """Line 87: Unknown extension falls back to application/octet-stream."""
        file = File.validate(b"binary data", info={"extension": ".zzz_unknown"})
        assert file.__content_type__ == "application/octet-stream"

    def test_bytes_with_known_extension_content_type(self) -> None:
        """Lines 73-74: Known extension (e.g. .txt) resolves correctly."""
        file = File.validate(b"hello", info={"extension": ".txt"})
        assert file.__content_type__ == "text/plain"

    def test_str_repr_bytes_with_extension(self) -> None:
        """Lines 92-93: __str__ for bytes-sourced file with extension."""
        file = File.validate(b"hello", info={"extension": ".txt"})
        assert str(file) == "io.IOBase(.txt)"
        assert repr(file) == "File(source=io.IOBase(.txt))"

    def test_str_repr_bytes_without_extension(self) -> None:
        """Lines 92-93: __str__ for bytes-sourced file without extension."""
        file = File.validate(b"hello")
        assert str(file) == "io.IOBase()"
        assert repr(file) == "File(source=io.IOBase())"


# ---------------------------------------------------------------------------
# TestFileMagicMethods — proxy __getattr__, __setattr__, __delattr__, __iter__,
#                        __deepcopy__, __wrapped__, readable/writable/seekable
# ---------------------------------------------------------------------------


class TestFileMagicMethods:
    def test_getattr_extension_alias(self) -> None:
        """Line 102: .extension is an alias for __source_extension__."""
        file = File.validate(b"data", info={"extension": ".csv"})
        assert file.extension == ".csv"

    def test_getattr_persisted_alias(self) -> None:
        """Line 104: .persisted is an alias for __persisted__."""
        file = File.validate(b"data")
        assert file.persisted is None

    def test_getattr_proxied_to_wrapped(self) -> None:
        """Lines 106-107: Unknown attr is proxied to the underlying BytesIO."""
        bio = BytesIO(b"hello")
        bio.custom_attr = "proxied_value"
        file = File.validate(bio)
        assert file.custom_attr == "proxied_value"

    def test_setattr_proxied_to_wrapped(self) -> None:
        """Lines 113-114: Setting unknown attr proxies to the underlying BytesIO."""
        file = File.validate(b"data")
        file.name = "test_name.bin"
        # The name should now be on the underlying BytesIO
        assert file.__wrapped__.name == "test_name.bin"

    def test_delattr_proxied_to_wrapped(self) -> None:
        """Lines 120-121: Deleting an unknown attr proxies to the underlying BytesIO."""
        bio = BytesIO(b"data")
        bio.my_temp_attr = "to_be_deleted"
        file = File.validate(bio)
        del file.my_temp_attr
        assert not hasattr(bio, "my_temp_attr")

    def test_iter_lines(self) -> None:
        """Line 125: __iter__ delegates to the underlying file object."""
        content = b"line1\nline2\nline3\n"
        file = File.validate(content)
        lines = list(file)
        assert lines == [b"line1\n", b"line2\n", b"line3\n"]

    def test_deepcopy_returns_self(self) -> None:
        """Line 145: __deepcopy__ returns self."""
        import copy

        file = File.validate(b"data")
        copied = copy.deepcopy(file)
        assert copied is file

    def test_wrapped_raises_without_fetcher(self) -> None:
        """Line 155: __wrapped__ raises ValueError when neither fileobj nor fetcher is set."""
        from timbal.types.file import File as _File

        # Manually construct a File with no fileobj and no fetcher
        f = object.__new__(_File)
        object.__setattr__(f, "__source__", "fake")
        object.__setattr__(f, "__source_scheme__", "url")
        object.__setattr__(f, "__source_extension__", None)
        object.__setattr__(f, "__fileobj__", None)
        object.__setattr__(f, "__fetcher__", None)
        object.__setattr__(f, "__persisted__", None)
        object.__setattr__(f, "__content_type__", "application/octet-stream")

        with pytest.raises(ValueError, match="not properly initialized"):
            _ = f.__wrapped__

    def test_readable_seekable_writable(self) -> None:
        """Lines 168-175: readable(), writable(), seekable() proxy correctly."""
        file = File.validate(b"data")
        assert file.readable() is True
        assert file.seekable() is True
        # BytesIO is not writable via io.IOBase.writable() but the proxy call must work
        assert isinstance(file.writable(), bool)


# ---------------------------------------------------------------------------
# TestFileValidateInfoDict — bytes / IOBase + info dict with extension / name
# ---------------------------------------------------------------------------


class TestFileValidateInfoDict:
    def test_bytes_with_info_extension(self) -> None:
        """Lines 200-201: bytes validated with info dict containing 'extension'."""
        file = File.validate(b"hello csv", info={"extension": ".csv"})
        assert file.__source_extension__ == ".csv"

    def test_bytes_with_info_name(self) -> None:
        """Lines 200-204: bytes validated with info dict containing 'name' sets source.name."""
        file = File.validate(b"hello", info={"name": "my_file.bin"})
        assert file.__wrapped__.name == "my_file.bin"

    def test_bytes_with_info_extension_and_name(self) -> None:
        """Lines 200-204: bytes validated with both 'extension' and 'name' in info dict."""
        file = File.validate(b"hello", info={"extension": ".txt", "name": "readme.txt"})
        assert file.__source_extension__ == ".txt"
        assert file.__wrapped__.name == "readme.txt"

    def test_iobase_with_info_extension(self) -> None:
        """Lines 215-216: io.IOBase validated with info dict containing 'extension'."""
        bio = BytesIO(b"data")
        file = File.validate(bio, info={"extension": ".json"})
        assert file.__source_extension__ == ".json"

    def test_iobase_with_info_name(self) -> None:
        """Lines 215-219: io.IOBase validated with info dict containing 'name'."""
        bio = BytesIO(b"data")
        file = File.validate(bio, info={"name": "output.json"})
        assert file.__wrapped__.name == "output.json"

    def test_iobase_with_info_extension_and_name(self) -> None:
        """Lines 215-219: io.IOBase validated with both keys in info dict."""
        bio = BytesIO(b"data")
        file = File.validate(bio, info={"extension": ".jsonl", "name": "output.jsonl"})
        assert file.__source_extension__ == ".jsonl"
        assert file.__wrapped__.name == "output.jsonl"


# ---------------------------------------------------------------------------
# TestFileDataUrlErrors — malformed data URLs
# ---------------------------------------------------------------------------


class TestFileDataUrlErrors:
    def test_data_url_missing_mimetype(self) -> None:
        """Line 251: Data URL without a mimetype separator raises ValueError."""
        with pytest.raises(ValueError, match="must specify a mimetype"):
            File.validate("data:base64,SGVsbG8=")

    def test_data_url_unsupported_custom_mimetype(self) -> None:
        """Line 257: Unknown MIME type that doesn't start with 'timbal/' raises ValueError."""
        with pytest.raises(ValueError, match="Custom mimetypes must be specified as 'timbal/"):
            File.validate("data:application/x-unknown-type-xyz;base64,SGVsbG8=")

    def test_invalid_scheme_raises(self) -> None:
        """Line 276: Unsupported scheme (e.g. ftp://) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid file source"):
            File.validate("ftp://example.com/file.txt")


# ---------------------------------------------------------------------------
# TestFilePersist — async persist() method
# ---------------------------------------------------------------------------


class TestFilePersist:
    async def test_persist_already_persisted(self) -> None:
        """Line 309: Returns immediately if __persisted__ is already set."""
        file = File.validate(b"data")
        object.__setattr__(file, "__persisted__", "already/persisted/url")
        result = await file.persist()
        assert result == "already/persisted/url"

    async def test_persist_url_no_platform_config(self) -> None:
        """Line 316: URL file with no platform config returns the URL directly."""
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        ctx = RunContext(platform_config=None, tracing_provider=None)
        set_run_context(ctx)

        url = "https://example.com/file.txt"
        file = File.validate(url)
        result = await file.persist()
        assert result == url

    async def test_persist_url_cdn_url_with_platform_config(self) -> None:
        """Lines 318-319: CDN URL with matching platform config cdn is stored and returned."""
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.config import PlatformConfig, PlatformAuth, PlatformAuthType

        cdn = "content.timbal.ai"
        platform_config = PlatformConfig(
            host="https://api.timbal.ai",
            cdn=cdn,
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="token123"),
        )
        ctx = RunContext(platform_config=platform_config, tracing_provider=None)
        set_run_context(ctx)

        url = f"https://{cdn}/assets/file.txt"
        file = File.validate(url)
        result = await file.persist()
        assert result == url
        assert file.__persisted__ == url

    async def test_persist_local_path_no_platform_config(self, tmp_path: pathlib.Path) -> None:
        """Lines 322-325: Local path file without platform config returns the path string."""
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        ctx = RunContext(platform_config=None, tracing_provider=None)
        set_run_context(ctx)

        test_file = tmp_path / "hello.txt"
        test_file.write_bytes(b"hello")
        file = File.validate(str(test_file))
        result = await file.persist()
        assert result == str(test_file)
        assert file.__persisted__ == str(test_file)

    async def test_persist_bytes_no_platform_config_writes_temp(self) -> None:
        """Lines 327-335: Bytes file without platform config writes to a temp file."""
        import os
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        ctx = RunContext(platform_config=None, tracing_provider=None)
        set_run_context(ctx)

        content = b"temp file content"
        file = File.validate(content, info={"extension": ".bin"})
        result = await file.persist()
        assert result is not None
        assert result.endswith(".bin")
        assert os.path.exists(result)
        assert open(result, "rb").read() == content

    async def test_persist_bytes_no_extension_writes_temp(self) -> None:
        """Lines 327-335: Bytes file without extension writes temp file without suffix."""
        import os
        from timbal.state import set_run_context
        from timbal.state.context import RunContext

        ctx = RunContext(platform_config=None, tracing_provider=None)
        set_run_context(ctx)

        content = b"no extension content"
        file = File.validate(content)
        result = await file.persist()
        assert result is not None
        assert os.path.exists(result)


# ---------------------------------------------------------------------------
# TestFileSerialize — serialize() with persisted value
# ---------------------------------------------------------------------------


class TestFileSerialize:
    def test_serialize_persisted_file_returns_persisted_url(self) -> None:
        """Lines 373-374: serialize() returns __persisted__ when it's set."""
        file = File.validate(b"data")
        object.__setattr__(file, "__persisted__", "https://cdn.example.com/file.bin")
        result = File.serialize(file)
        assert result == "https://cdn.example.com/file.bin"

    def test_serialize_unpersisted_file_returns_data_url(self) -> None:
        """Lines 375-376: serialize() returns to_data_url() when not yet persisted."""
        file = File.validate(b"hello")
        result = File.serialize(file)
        assert result.startswith("data:")


# ---------------------------------------------------------------------------
# TestFileFetchHttp — Content-Disposition filename parsing
# ---------------------------------------------------------------------------


class TestFileFetchHttp:
    def test_fetch_http_uses_content_disposition_filename(self) -> None:
        """Lines 406-410: filename is parsed from Content-Disposition header."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.content = b"file content"
        mock_response.headers = {
            "Content-Disposition": 'attachment; filename="report.pdf"',
            "Content-Type": "application/pdf",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            fileobj = File._fetch_http_file(url="https://example.com/download")

        assert fileobj.name == "report.pdf"

    def test_fetch_http_content_disposition_single_quotes(self) -> None:
        """Lines 406-410: filename parsing with single-quoted name."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.content = b"file content"
        mock_response.headers = {
            "Content-Disposition": "attachment; filename='report.csv'",
            "Content-Type": "text/csv",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            fileobj = File._fetch_http_file(url="https://example.com/download")

        assert fileobj.name == "report.csv"

    def test_fetch_http_falls_back_to_url_when_no_content_disposition(self) -> None:
        """Lines 411-413: Falls back to URL-based filename when no Content-Disposition."""
        from unittest.mock import MagicMock, patch

        headers_data = {
            "Content-Type": "application/octet-stream",
        }

        mock_headers = MagicMock()
        mock_headers.get = lambda key, default=None: headers_data.get(key, default)

        mock_response = MagicMock()
        mock_response.content = b"file content"
        mock_response.headers = mock_headers
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            fileobj = File._fetch_http_file(url="https://example.com/path/data.json")

        assert fileobj.name == "data.json"


# ---------------------------------------------------------------------------
# TestFilePydanticSchema — JSON schema generation
# ---------------------------------------------------------------------------


class TestFilePydanticSchema:
    def test_get_pydantic_json_schema_structure(self) -> None:
        """Lines 428-433: __get_pydantic_json_schema__ returns expected schema dict."""
        schema = File.__get_pydantic_json_schema__(None, None)
        assert schema["type"] == "string"
        assert schema["format"] == "uri"
        assert "description" in schema
