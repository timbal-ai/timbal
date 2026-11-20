import asyncio
import contextlib
import io
import os
import pathlib
import threading
import time
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
    url = "https://timbalusercontent.com/assets/hello_world.txt"

    file = File.validate(url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "url"
    assert file.__source_extension__ == ".txt"
    assert file.read() == test_content


def test_file_serialize_url() -> None:
    url = "https://timbalusercontent.com/assets/hello_world.txt"

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
        # File subclasses io.IOBase; Python 3.11+ requires IOBase.__new__ for bypass init.
        f = io.IOBase.__new__(_File)
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

    async def test_persist_url_usercontent_host_short_circuits(self) -> None:
        """timbalusercontent.com URLs are platform-persisted with the default
        platform_config.cdn — no re-upload should happen."""
        from timbal.state import set_run_context
        from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig
        from timbal.state.context import RunContext

        platform_config = PlatformConfig(
            host="https://api.timbal.ai",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="token123"),
        )
        ctx = RunContext(platform_config=platform_config, tracing_provider=None)
        set_run_context(ctx)

        url = "https://timbalusercontent.com/tmp/019f285e/Screenshot%202026-07-03%20at%2016.23.06.png"
        file = File.validate(url)
        result = await file.persist()
        assert result == url
        assert file.__persisted__ == url

    async def test_persist_url_cdn_with_explicit_port_short_circuits(self) -> None:
        """CDN URLs with an explicit port (or uppercase host) still match the host check."""
        from timbal.state import set_run_context
        from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig
        from timbal.state.context import RunContext

        platform_config = PlatformConfig(
            host="https://api.timbal.ai",
            cdn="content.timbal.ai",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="token123"),
        )
        set_run_context(RunContext(platform_config=platform_config, tracing_provider=None))

        for url in (
            "https://content.timbal.ai:443/assets/file.txt",
            "https://Content.Timbal.AI/assets/file.txt",
            "https://timbalusercontent.com:443/tmp/019f/file.png",
        ):
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

    def _platform_ctx(self, org_id: str = "org-1"):
        from timbal.state.context import RunContext
        from timbal.state.config import (
            PlatformAuth,
            PlatformAuthType,
            PlatformConfig,
            PlatformSubject,
        )

        platform_config = PlatformConfig(
            host="https://api.timbal.ai",
            cdn="content.timbal.ai",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="token123"),
            subject=PlatformSubject(org_id=org_id, app_id="app-9"),
        )
        return RunContext(platform_config=platform_config, tracing_provider=None)

    async def test_persist_upload_counts_against_the_run_org(self) -> None:
        """POST /files carries the run's org_id so the org's upload limit applies."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from timbal.state import set_run_context

        set_run_context(self._platform_ctx(org_id="42"))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content_length": 4,
            "content_type": "application/octet-stream",
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "name": "a.bin",
            "url": "https://timbalusercontent.com/tmp/019f/a.bin",
        }

        file = File.validate(b"data", {"extension": ".bin"})
        with patch("timbal.platform.utils._request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await file.persist()

        assert mock_request.await_args.args == ("POST", "files")
        assert mock_request.await_args.kwargs["params"] == {"org_id": "42"}

    async def test_persist_upload_omits_non_numeric_org_id(self) -> None:
        """The platform rejects a non-numeric org_id, so persist() leaves it out."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from timbal.state import set_run_context

        set_run_context(self._platform_ctx(org_id="org-1"))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content_length": 4,
            "content_type": "application/octet-stream",
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "name": "a.bin",
            "url": "https://timbalusercontent.com/tmp/019f/a.bin",
        }

        file = File.validate(b"data", {"extension": ".bin"})
        with patch("timbal.platform.utils._request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await file.persist()

        assert mock_request.await_args.kwargs["params"] is None

    async def test_persist_upload_returns_unsigned_url(self, tmp_path: pathlib.Path) -> None:
        """POST /files returns a fully-encoded url; persist() must use it verbatim."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from timbal.state import set_run_context

        set_run_context(self._platform_ctx())

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content_length": 4,
            "content_type": "application/octet-stream",
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "name": "my file.bin",
            "url": "https://timbalusercontent.com/tmp/019f/my%20file.bin",
        }

        test_file = tmp_path / "my file.bin"
        test_file.write_bytes(b"data")
        file = File.validate(str(test_file))
        with patch(
            "timbal.platform.utils._request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response
            result = await file.persist()

        assert result == "https://timbalusercontent.com/tmp/019f/my%20file.bin"
        assert file.__persisted__ == result

    async def test_persist_anonymous_bytes_with_platform_config_synthesizes_filename(self) -> None:
        """Anonymous bytes File has no initial BytesIO.name; persist() must synthesize one."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from timbal.state import set_run_context

        set_run_context(self._platform_ctx())

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content_length": 4,
            "content_type": "image/png",
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "name": "generated.png",
            "url": "https://timbalusercontent.com/tmp/019f/generated.png",
        }

        file = File.validate(b"data", {"extension": ".png"})
        with patch("timbal.platform.utils._request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await file.persist()

        assert result == "https://timbalusercontent.com/tmp/019f/generated.png"
        uploaded_name, uploaded_bytes, uploaded_mime = mock_request.await_args.kwargs["files"]["file"]
        assert uploaded_name.endswith(".png")
        assert uploaded_bytes == b"data"
        assert uploaded_mime == "image/png"

    async def test_persist_bytes_with_file_content_name_uses_display_name(self) -> None:
        """FileContent(..., name='company_logo.png') propagates its name to an unnamed bytes File."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from timbal.state import set_run_context
        from timbal.types.content import FileContent

        set_run_context(self._platform_ctx())

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content_length": 4,
            "content_type": "image/png",
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "name": "company_logo.png",
            "url": "https://timbalusercontent.com/tmp/019f/company_logo.png",
        }

        fc = FileContent(file=File.validate(b"data", {"extension": ".png"}), name="company_logo.png")
        with patch("timbal.platform.utils._request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await fc.file.persist()

        assert result == "https://timbalusercontent.com/tmp/019f/company_logo.png"
        uploaded_name, uploaded_bytes, uploaded_mime = mock_request.await_args.kwargs["files"]["file"]
        assert uploaded_name == "company_logo.png"
        assert uploaded_bytes == b"data"
        assert uploaded_mime == "image/png"


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
        """filename is parsed from Content-Disposition header."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.content = b"file content"
        mock_response.headers = {
            "Content-Disposition": 'attachment; filename="report.pdf"',
            "Content-Type": "application/pdf",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fileobj = File._fetch_http_file(url="https://example.com/download")

        assert fileobj.name == "report.pdf"

    def test_fetch_http_content_disposition_single_quotes(self) -> None:
        """filename parsing with single-quoted name."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.content = b"file content"
        mock_response.headers = {
            "Content-Disposition": "attachment; filename='report.csv'",
            "Content-Type": "text/csv",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fileobj = File._fetch_http_file(url="https://example.com/download")

        assert fileobj.name == "report.csv"

    def test_fetch_http_falls_back_to_url_when_no_content_disposition(self) -> None:
        """Falls back to URL-based filename when no Content-Disposition."""
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

        with patch("httpx.get", return_value=mock_response):
            fileobj = File._fetch_http_file(url="https://example.com/path/data.json")

        assert fileobj.name == "data.json"


# ---------------------------------------------------------------------------
# TestFilePydanticSchema — JSON schema generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestFileConstructor — File(source) smart constructor
# ---------------------------------------------------------------------------


class TestFileConstructor:
    def test_bytes(self) -> None:
        file = File(b"hello")
        assert file.__source_scheme__ == "bytes"
        assert file.read() == b"hello"

    def test_bytes_with_extension_and_name(self) -> None:
        file = File(b"hello", extension=".txt", name="readme.txt")
        assert file.__source_extension__ == ".txt"
        assert file.__wrapped__.name == "readme.txt"
        assert file.__content_type__ == "text/plain"

    def test_bytearray(self) -> None:
        file = File(bytearray(b"data"))
        assert file.__source_scheme__ == "bytes"
        assert file.read() == b"data"

    def test_io_base(self) -> None:
        bio = BytesIO(b"content")
        file = File(bio)
        assert file.__source_scheme__ == "bytes"
        assert file.read() == b"content"

    def test_local_path_str(self, tmp_path: pathlib.Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello")
        file = File(str(test_file))
        assert file.__source_scheme__ == "local_path"
        assert file.read() == b"hello"

    def test_local_path_obj(self, tmp_path: pathlib.Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello")
        file = File(test_file)
        assert file.__source_scheme__ == "local_path"
        assert file.read() == b"hello"

    def test_data_url(self) -> None:
        file = File("data:text/plain;base64,SGVsbG8=")
        assert file.__source_scheme__ == "data_url"
        assert file.__source_extension__ == ".txt"

    def test_http_url(self) -> None:
        file = File("https://example.com/file.pdf")
        assert file.__source_scheme__ == "url"
        assert file.__source_extension__ == ".pdf"

    def test_identity_existing_file(self) -> None:
        original = File(b"data")
        same = File(original)
        assert same is original

    def test_invalid_source(self) -> None:
        with pytest.raises(ValueError):
            File(12345)

    def test_invalid_scheme(self) -> None:
        with pytest.raises(ValueError):
            File("ftp://example.com/file.txt")

    def test_backwards_compat_internal_init(self, tmp_path: pathlib.Path) -> None:
        """Internal callers passing source_scheme still work."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello")
        file = File(
            str(test_file),
            source_scheme="local_path",
            source_extension=".txt",
            fetcher=lambda: File._fetch_local_file(path=test_file),
        )
        assert file.__source_scheme__ == "local_path"
        assert file.read() == b"hello"


# ---------------------------------------------------------------------------
# TestFileLoad — async load() method
# ---------------------------------------------------------------------------


class TestFileLoad:
    async def test_load_bytes_is_noop(self) -> None:
        file = File(b"already loaded")
        await file.load()
        assert file.read() == b"already loaded"

    async def test_load_local_path(self, tmp_path: pathlib.Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"local content")
        file = File(str(test_file))
        await file.load()
        assert file.read() == b"local content"

    async def test_load_url_async(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.content = b"fetched content"
        mock_response.headers = {
            "Content-Type": "text/plain",
            "Content-Disposition": 'attachment; filename="doc.txt"',
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        file = File("https://example.com/doc.txt")
        await file.load(client=mock_client)

        assert file.read() == b"fetched content"
        assert file.__content_type__ == "text/plain"
        # Client should NOT be closed when passed in (caller owns it)
        mock_client.aclose.assert_not_called()

    async def test_load_idempotent(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        file = File("https://example.com/file.bin")
        await file.load(client=mock_client)
        await file.load(client=mock_client)

        # Only fetched once
        assert mock_client.get.call_count == 1


# ---------------------------------------------------------------------------
# TestMessageLoad — Message.load() concurrent file loading
# ---------------------------------------------------------------------------


class TestMessageLoad:
    async def test_load_no_files(self) -> None:
        from timbal.types.message import Message

        msg = Message.validate({"role": "user", "content": "hello"})
        await msg.load()  # Should not raise

    async def test_load_with_file_content(self, tmp_path: pathlib.Path) -> None:
        from timbal.types.content import FileContent
        from timbal.types.message import Message

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"content")
        file = File(str(test_file))
        msg = Message(role="user", content=[FileContent(file=file)])
        await msg.load()
        file.seek(0)
        assert file.read() == b"content"


# ---------------------------------------------------------------------------
# TestFilePydanticSchema — JSON schema generation
# ---------------------------------------------------------------------------


class TestFilePydanticSchema:
    def test_get_pydantic_json_schema_structure(self) -> None:
        schema = File.__get_pydantic_json_schema__(None, None)
        assert schema["type"] == "string"
        assert schema["format"] == "uri"
        assert "description" in schema


# ---------------------------------------------------------------------------
# TestLazyFetchSafety — never-materialized files must not hit the network
# ---------------------------------------------------------------------------


class TestLazyFetchSafety:
    def test_close_unfetched_url_file_does_not_fetch(self) -> None:
        from unittest.mock import patch

        with patch("httpx.get") as mock_get:
            file = File("https://cdn.invalid/report.pdf")
            file.close()
        mock_get.assert_not_called()

    def test_gc_of_unfetched_url_file_does_not_fetch(self) -> None:
        import gc
        from unittest.mock import patch

        with patch("httpx.get") as mock_get:
            file = File("https://cdn.invalid/report.pdf")
            del file
            gc.collect()
        mock_get.assert_not_called()

    def test_close_closes_materialized_fileobj(self) -> None:
        file = File.validate(b"hello", {"extension": ".txt"})
        assert file.read() == b"hello"
        file.close()
        assert file.__wrapped__.closed

    def test_failed_fetch_is_cached_and_not_retried(self) -> None:
        from unittest.mock import patch

        boom = ConnectionError("no route to host")
        with patch("httpx.get", side_effect=boom) as mock_get:
            file = File("https://cdn.invalid/report.pdf")
            with pytest.raises(ConnectionError):
                file.read()
            with pytest.raises(ConnectionError):
                file.read()
        assert mock_get.call_count == 1

    def test_half_constructed_file_repr_and_close_are_safe(self) -> None:
        """A File whose __init__ raised before _setup() (invalid source) leaves an
        object with none of its internal slots set. repr()/str()/close() run on it
        during GC/unraisable-hook handling must not recurse or raise — this was a
        RecursionError on Windows/py3.13."""
        f = io.IOBase.__new__(File)  # bypass __init__: no slots set at all
        assert str(f) == "File(uninitialized)"
        assert repr(f) == "File(source=File(uninitialized))"
        f.close()  # must be a no-op, not raise AttributeError into __del__
        with pytest.raises(AttributeError):
            _ = f.__source_scheme__

    def test_invalid_source_gc_does_not_raise_unraisable(self, recwarn) -> None:
        """Constructing File with an invalid source raises, and the half-built
        object being finalized must not surface an unraisable exception."""
        import gc

        for _ in range(50):
            with pytest.raises(ValueError):
                File("not://a/real/scheme/at/all")
        gc.collect()  # would trigger the RecursionError via __del__ -> close/repr


# ---------------------------------------------------------------------------
# TestSignedContentResign — expired / revoked platform CDN URLs re-sign on read
# ---------------------------------------------------------------------------

_KB_OBJECT = "https://timbalusercontent.com/orgs/42/k2/7/files/9/report.pdf"


def _signed_url(expires_in: float, signature: str = "old", base: str = _KB_OBJECT) -> str:
    expires = int(time.time() + expires_in)
    return f"{base}?Expires={expires}&Signature={signature}&Key-Pair-Id=K2WLME83EQGDL3&Hash-Algorithm=SHA256"


def _sign_response(signed_url: str | None, url: str = _KB_OBJECT):
    from unittest.mock import MagicMock

    res = MagicMock()
    res.json.return_value = {"signed_url": signed_url, "url": url}
    return res


class _FakeCdn:
    """Serves every URL in ``ok``, raises a connection error for those in ``unreachable`` and answers
    ``fail_status`` (403) to anything else, recording each GET."""

    def __init__(self, *ok: str) -> None:
        self.ok = set(ok)
        self.unreachable: set[str] = set()
        self.fail_status = 403
        self.calls: list[str] = []

    def get_sync(self, url: str, **_kwargs):
        import httpx

        self.calls.append(url)
        if url in self.unreachable:
            raise httpx.ConnectError("All connection attempts failed", request=httpx.Request("GET", url))
        if url in self.ok:
            return httpx.Response(
                200, content=b"%PDF-1.7", headers={"Content-Type": "application/pdf"}, request=httpx.Request("GET", url)
            )
        return httpx.Response(self.fail_status, content=b"<Error/>", request=httpx.Request("GET", url))

    async def get(self, url: str, **kwargs):
        return self.get_sync(url, **kwargs)


def _platform_ctx(org_id: str = "42", token: str = "token123", subject: bool = True):
    from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
    from timbal.state.context import RunContext

    return RunContext(
        platform_config=PlatformConfig(
            host="api.timbal.ai",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token=token),
            subject=PlatformSubject(org_id=org_id, app_id="1") if subject else None,
        ),
        tracing_provider=None,
    )


@contextlib.contextmanager
def _sign_endpoint(mode: str, *results):
    """Patch the platform request the given read path signs through; each call takes the next result."""
    from unittest.mock import AsyncMock, MagicMock, patch

    if mode == "async":
        target, mock = "timbal.platform.utils._request", AsyncMock(side_effect=list(results))
    else:
        target, mock = "timbal.platform.utils._request_sync", MagicMock(side_effect=list(results))
    with patch(target, mock):
        yield mock


def _platform_response(status: int, json: dict | None = None, headers: dict | None = None):
    import httpx

    request = httpx.Request("POST", "https://api.timbal.ai/x")
    return httpx.Response(status, json=json if json is not None else {}, headers=headers, request=request)


class _FakePlatform:
    """Stands in for the platform's HTTP client, below ``_request``: each request takes the next scripted
    result and the last one repeats. A result is a response, an exception to raise or a coroutine function."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.requests: list[tuple[str, str, dict]] = []

    @contextlib.asynccontextmanager
    async def client(self, _timeout):
        yield self

    async def request(self, method: str, url: str, **kwargs):
        self.requests.append((method, url, kwargs))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return await result()
        return result

    def patched(self):
        from unittest.mock import patch

        return patch("timbal.platform.utils.request_client", self.client)


async def _read(url: str, cdn: _FakeCdn, mode: str) -> bytes:
    """Read a File through the async load() path or the sync lazy-fetch path."""
    from unittest.mock import patch

    file = File(url)
    if mode == "async":
        await file.load(client=cdn)
    else:
        # Lazy fetch inside a running event loop: the sync path must not touch the loop.
        with patch("httpx.get", side_effect=cdn.get_sync):
            file.read()
    file.seek(0)
    return file.read()


@pytest.fixture
def platform_run():
    from timbal.state import get_run_context, set_run_context
    from timbal.types.file import _clear_mint_cache

    previous = get_run_context()
    set_run_context(_platform_ctx())
    _clear_mint_cache()
    yield
    _clear_mint_cache()
    set_run_context(previous)


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.usefixtures("platform_run")
class TestSignedContentResign:
    async def test_fresh_url_is_fetched_without_signing(self, mode: str) -> None:
        stored = _signed_url(3600)
        cdn = _FakeCdn(stored)
        with _sign_endpoint(mode) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        sign.assert_not_called()
        assert cdn.calls == [stored]

    async def test_url_near_its_expiry_is_fetched_without_signing(self, mode: str) -> None:
        # Not past its Expires yet: it can still serve, so the read does not wait on a sign call.
        stored = _signed_url(30)
        cdn = _FakeCdn(stored)
        with _sign_endpoint(mode) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        sign.assert_not_called()
        assert cdn.calls == [stored]

    async def test_expired_url_is_resigned_before_fetch(self, mode: str) -> None:
        stored = _signed_url(-3600)  # clearly expired: no wasted GET on it
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn(stored, fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert sign.call_args.args == ("POST", "orgs/42/content/sign")
        assert sign.call_args.kwargs["json"] == {"url": stored}
        assert cdn.calls == [fresh]

    async def test_url_just_past_its_expires_is_still_fetched_first(self, mode: str) -> None:
        from timbal.errors import PlatformError

        # Expired a couple of seconds ago by this machine's clock, which may be ahead of the CDN's:
        # the CDN may still accept it, and a sign call first could outlast what it has left.
        stored = _signed_url(-2)
        cdn = _FakeCdn(stored)
        with _sign_endpoint(mode, PlatformError("unavailable", status_code=503)) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        sign.assert_not_called()
        assert cdn.calls == [stored]

        # Really expired: the CDN says so, and the read re-signs as after any 403.
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [stored, fresh]

    async def test_clock_skew_grace_decides_what_is_fetched_first(self, mode: str) -> None:
        import types
        from unittest.mock import patch

        from timbal.types.file import _CLOCK_SKEW_GRACE_SECS, _clear_mint_cache

        start = time.time()
        now = [start]
        stored = _signed_url(0)
        expires = int(start)
        fresh = _signed_url(7200, signature="new")
        with patch("timbal.types.file.time", types.SimpleNamespace(time=lambda: now[0])):
            for past_expires, first in ((_CLOCK_SKEW_GRACE_SECS - 1, stored), (_CLOCK_SKEW_GRACE_SECS + 1, fresh)):
                now[0] = expires + past_expires
                _clear_mint_cache()
                cdn = _FakeCdn(stored, fresh)
                with _sign_endpoint(mode, _sign_response(fresh)):
                    await _read(stored, cdn, mode)
                assert cdn.calls == [first]

    async def test_sign_call_goes_through_the_platform_request_helpers(self, mode: str) -> None:
        # Only the HTTP client is faked: the sync path really runs its sign call on a helper
        # thread while this test's event loop is running.
        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        platform = _FakePlatform(_platform_response(200, {"signed_url": fresh, "url": _KB_OBJECT}))
        cdn = _FakeCdn(fresh)
        with platform.patched():
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        ((method, url, kwargs),) = platform.requests
        assert (method, url) == ("POST", "https://api.timbal.ai/orgs/42/content/sign")
        assert kwargs["headers"]["Authorization"] == "Bearer token123"
        assert kwargs["json"] == {"url": stored}
        assert cdn.calls == [fresh]

    async def test_expired_url_falls_back_to_legacy_url_field(self, mode: str) -> None:
        stored = _signed_url(-7200)
        cdn = _FakeCdn(_KB_OBJECT)
        with _sign_endpoint(mode, _sign_response(None, url=_KB_OBJECT)):
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert cdn.calls == [_KB_OBJECT]

    async def test_403_resigns_once_and_retries(self, mode: str) -> None:
        # Not expired by its Expires, but signed with a revoked key.
        stored = _signed_url(3600)
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [stored, fresh]

    async def test_403_after_resign_raises_the_fetch_error(self, mode: str) -> None:
        import httpx

        stored = _signed_url(3600)
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn()
        with _sign_endpoint(mode, _sign_response(fresh)) as sign, pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _read(stored, cdn, mode)
        assert exc_info.value.response.status_code == 403
        assert sign.call_count == 1
        assert cdn.calls == [stored, fresh]

    async def test_sign_endpoint_403_raises_fetch_error_without_looping(self, mode: str) -> None:
        import httpx
        from timbal.errors import PlatformError

        stored = _signed_url(3600)
        cdn = _FakeCdn()
        denied = PlatformError("forbidden", status_code=403)
        with _sign_endpoint(mode, denied, denied) as sign, pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _read(stored, cdn, mode)
        assert exc_info.value.response.status_code == 403
        assert exc_info.value.__cause__ is denied
        assert sign.call_count == 1
        assert cdn.calls == [stored]

    async def test_failed_sign_of_an_expired_url_fetches_it_once(self, mode: str) -> None:
        import httpx
        from timbal.errors import PlatformError

        denied = PlatformError("forbidden", status_code=403)
        # Past its Expires and no fresh URL to be had: one sign attempt, one GET, no second sign on the 403.
        stored = _signed_url(-3600)
        cdn = _FakeCdn()
        with _sign_endpoint(mode, denied, denied) as sign, pytest.raises(httpx.HTTPStatusError) as exc_info:
            await _read(stored, cdn, mode)
        assert exc_info.value.response.status_code == 403
        assert exc_info.value.__cause__ is denied
        assert sign.call_count == 1
        assert cdn.calls == [stored]

        # The CDN has the last word: this machine's clock may be ahead of it.
        cdn = _FakeCdn(stored)
        with _sign_endpoint(mode, denied) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [stored]

    async def test_unexpired_url_does_not_depend_on_the_sign_endpoint(self, mode: str) -> None:
        from timbal.errors import PlatformError

        # 30 s left. Whatever the sign endpoint would do (fail, or answer a URL the CDN refuses),
        # the stored URL serves the read and the endpoint is not called.
        stored = _signed_url(30)
        for answer in (PlatformError("unavailable", status_code=503), _sign_response(None, url=_KB_OBJECT)):
            cdn = _FakeCdn(stored)
            with _sign_endpoint(mode, answer) as sign:
                assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            sign.assert_not_called()
            assert cdn.calls == [stored]

    async def test_valid_stored_url_is_fetched_before_a_memoized_mint(self, mode: str) -> None:
        first = _signed_url(3600, signature="first")
        stored = _signed_url(3600)
        cdn = _FakeCdn(first)
        with _sign_endpoint(mode, _sign_response(first)) as sign:
            await _read(_signed_url(-3600), cdn, mode)  # memoizes `first` for the object
            cdn.ok = {stored}  # the memoized mint would now be refused
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls[1:] == [stored]

    @pytest.mark.parametrize("status", [404, 500])
    async def test_error_other_than_403_on_an_unexpired_url_is_final(self, mode: str, status: int) -> None:
        import httpx

        fresh = _signed_url(3600, signature="new")
        stored = _signed_url(3600)
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            await _read(_signed_url(-3600), cdn, mode)  # a memoized mint that would serve
            cdn.fail_status = status
            cdn.calls.clear()
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await _read(stored, cdn, mode)
        assert exc_info.value.response.status_code == status
        assert sign.call_count == 1
        assert cdn.calls == [stored]

    async def test_url_without_a_usable_expires_is_fetched_first(self, mode: str) -> None:
        fresh = _signed_url(3600, signature="new")
        for query in ("Policy=eyJ9&Signature=old&Key-Pair-Id=K2", "Expires=soon&Signature=old&Key-Pair-Id=K2"):
            stored = f"{_KB_OBJECT}?{query}"
            # Nothing says it is expired: it is fetched as it is...
            cdn = _FakeCdn(stored)
            with _sign_endpoint(mode) as sign:
                assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            sign.assert_not_called()
            assert cdn.calls == [stored]
            # ...and re-signed only once the CDN refuses it.
            cdn = _FakeCdn(fresh)
            with _sign_endpoint(mode, _sign_response(fresh)) as sign:
                assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            assert sign.call_count == 1
            assert cdn.calls == [stored, fresh]
            from timbal.types.file import _clear_mint_cache

            _clear_mint_cache()

    async def test_failed_refresh_is_not_reused_by_later_reads(self, mode: str) -> None:
        import httpx

        # The endpoint answers only the legacy url (no Expires), which 404s: the read fails...
        cdn = _FakeCdn()
        cdn.fail_status = 404
        with _sign_endpoint(mode, _sign_response(None, url=_KB_OBJECT)) as sign:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await _read(_signed_url(-3600), cdn, mode)
            assert exc_info.value.response.status_code == 404
            # ...and a later read of the same object with a working URL is not sent to it.
            working = _signed_url(3600, signature="ok")
            cdn.ok = {working}
            cdn.calls.clear()
            assert await _read(working, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [working]

    async def test_cached_mint_failing_with_any_status_is_dropped(self, mode: str) -> None:
        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        second = _signed_url(3600, signature="second")
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh), _sign_response(second)) as sign:
            await _read(stored, cdn, mode)  # memoizes `fresh`
            cdn.ok, cdn.fail_status = {second}, 404
            cdn.calls.clear()
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 2
        assert cdn.calls == [fresh, second, second]

    async def test_memoized_mint_is_used_until_it_is_clearly_expired(self, mode: str) -> None:
        import types
        from unittest.mock import patch

        from timbal.types.file import _CLOCK_SKEW_GRACE_SECS

        start = time.time()
        now = [start]
        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        expires = int(start + 3600)
        second = _signed_url(3 * 3600, signature="second")
        cdn = _FakeCdn(fresh, second)
        with (
            patch("timbal.types.file.time", types.SimpleNamespace(time=lambda: now[0])),
            _sign_endpoint(mode, _sign_response(fresh), _sign_response(second)) as sign,
        ):
            await _read(stored, cdn, mode)  # memoizes `fresh`, good for an hour
            now[0] = expires - 10  # seconds from its Expires: still the URL to use
            await _read(stored, cdn, mode)
            now[0] = expires + _CLOCK_SKEW_GRACE_SECS - 1  # just past it here; the CDN's clock may be behind
            await _read(stored, cdn, mode)
            assert sign.call_count == 1
            assert cdn.calls == [fresh, fresh, fresh]
            now[0] = expires + _CLOCK_SKEW_GRACE_SECS + 1  # clearly expired: not fetched again
            await _read(stored, cdn, mode)
        assert sign.call_count == 2
        assert cdn.calls == [fresh, fresh, fresh, second]

    async def test_unreachable_cached_mint_does_not_stop_a_valid_stored_url(self, mode: str) -> None:
        fresh = _signed_url(3600, signature="new")
        stored = _signed_url(3600)
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            await _read(_signed_url(-3600), cdn, mode)  # memoizes `fresh`
            cdn.ok, cdn.unreachable = {stored}, {fresh}  # the memoized URL now raises a connection error
            cdn.calls.clear()
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [stored, stored]

    async def test_unreachable_cached_mint_is_replaced_when_the_stored_url_expired(self, mode: str) -> None:
        stored = _signed_url(-3600)
        first = _signed_url(3600, signature="first")
        second = _signed_url(3600, signature="second")
        cdn = _FakeCdn(first)
        with _sign_endpoint(mode, _sign_response(first), _sign_response(second)) as sign:
            await _read(stored, cdn, mode)
            cdn.ok, cdn.unreachable = {second}, {first}
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 2
        assert cdn.calls == [first, first, second, second]

    async def test_unreachable_minted_url_raises_and_is_not_memoized(self, mode: str) -> None:
        import httpx

        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn()
        cdn.unreachable = {fresh}
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            with pytest.raises(httpx.ConnectError):
                await _read(stored, cdn, mode)
            assert cdn.calls == [fresh, stored]  # the stored URL was still tried before failing
            # A later read of the same object with a working URL is not sent to the mint.
            working = _signed_url(3600, signature="ok")
            cdn.ok = {working}
            cdn.calls.clear()
            assert await _read(working, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [working]

    async def test_unreachable_stored_url_raises_without_signing(self, mode: str) -> None:
        import httpx

        # A connection error on a URL that is not past its Expires is what it was before re-signing existed.
        stored = _signed_url(3600)
        cdn = _FakeCdn()
        cdn.unreachable = {stored}
        with _sign_endpoint(mode) as sign, pytest.raises(httpx.ConnectError):
            await _read(stored, cdn, mode)
        sign.assert_not_called()
        assert cdn.calls == [stored]

    async def test_failed_sign_then_unreachable_stored_url_raises_both(self, mode: str) -> None:
        import httpx
        from timbal.errors import PlatformError

        denied = PlatformError("forbidden", status_code=403)
        stored = _signed_url(-3600)
        cdn = _FakeCdn()
        cdn.unreachable = {stored}
        with _sign_endpoint(mode, denied) as sign, pytest.raises(httpx.ConnectError) as exc_info:
            await _read(stored, cdn, mode)
        assert exc_info.value.__cause__ is denied
        assert sign.call_count == 1
        assert cdn.calls == [stored]

    async def test_stored_url_is_the_last_resort_after_a_refused_refresh(self, mode: str) -> None:
        import httpx

        stored = _signed_url(-3600)  # past its Expires here; this machine's clock may be ahead of the CDN's
        for answer, refreshed in (
            (_sign_response(_signed_url(3600, signature="new")), _signed_url(3600, signature="new")),
            (_sign_response(None, url=_KB_OBJECT), _KB_OBJECT),  # signing unavailable: only the legacy url
        ):
            # The CDN still serves the stored URL: the read is served by it.
            cdn = _FakeCdn(stored)
            with _sign_endpoint(mode, answer) as sign:
                assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            assert sign.call_count == 1
            assert cdn.calls == [refreshed, stored]
            # It does not: the error raised is the refreshed URL's.
            cdn = _FakeCdn()
            with _sign_endpoint(mode, answer), pytest.raises(httpx.HTTPStatusError) as exc_info:
                await _read(stored, cdn, mode)
            assert str(exc_info.value.request.url) == refreshed
            assert cdn.calls == [refreshed, stored]

    async def test_sign_answering_the_stored_url_is_not_memoized(self, mode: str) -> None:
        from timbal.types.file import _SignedContentUrl

        stored = _signed_url(-3600)
        cdn = _FakeCdn(stored)
        with _sign_endpoint(mode, _sign_response(stored)):
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert _SignedContentUrl.resolve(stored).cached() is None

    async def test_sign_call_has_a_total_budget(self, mode: str) -> None:
        import httpx
        from timbal.types.file import _SIGN_BUDGET_SECS

        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            await _read(stored, _FakeCdn(fresh), mode)
        assert sign.call_args.kwargs["total_timeout"] == _SIGN_BUDGET_SECS
        assert sign.call_args.kwargs["timeout"] == httpx.Timeout(_SIGN_BUDGET_SECS)

    async def test_without_platform_subject_behaviour_is_unchanged(self, mode: str) -> None:
        import httpx
        from timbal.state import set_run_context

        for ctx in (None, _platform_ctx(subject=False)):
            set_run_context(ctx)
            for stored in (_signed_url(-3600), _signed_url(3600)):
                cdn = _FakeCdn()
                with _sign_endpoint(mode) as sign, pytest.raises(httpx.HTTPStatusError):
                    await _read(stored, cdn, mode)
                sign.assert_not_called()
                assert cdn.calls == [stored]

    async def test_unsigned_and_foreign_urls_are_never_signed(self, mode: str) -> None:
        import httpx

        for stored in (
            "https://timbalusercontent.com/orgs/42/assets/logo.png",  # public content
            "https://timbalusercontent.com/tmp/019f/a.png",
            "https://content.timbal.ai/orgs/42/k2/7/files/9/report.pdf",  # legacy unsigned link
            _signed_url(-3600, base="https://example.com/report.pdf"),  # not a platform host
        ):
            cdn = _FakeCdn()
            with _sign_endpoint(mode) as sign, pytest.raises(httpx.HTTPStatusError):
                await _read(stored, cdn, mode)
            sign.assert_not_called()
            assert cdn.calls == [stored]

    async def test_second_read_is_served_from_the_mint_cache(self, mode: str) -> None:
        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
            # Another File for the same object (e.g. the next turn reloading memory).
            assert await _read(_signed_url(-1800), cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 1
        assert cdn.calls == [fresh, fresh]

    async def test_revoked_key_second_read_reuses_the_mint(self, mode: str) -> None:
        stored = _signed_url(3600)  # not past its Expires, so it is asked first on every read
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)) as sign:
            await _read(stored, cdn, mode)
            await _read(stored, cdn, mode)
        assert sign.call_count == 1
        assert cdn.calls == [stored, fresh, stored, fresh]

    async def test_cached_mint_that_is_refused_is_replaced(self, mode: str) -> None:
        stored = _signed_url(-3600)
        first = _signed_url(3600, signature="first")
        second = _signed_url(3600, signature="second")
        cdn = _FakeCdn(first)
        with _sign_endpoint(mode, _sign_response(first), _sign_response(second)) as sign:
            await _read(stored, cdn, mode)
            cdn.ok = {second}  # e.g. the key behind the cached mint was revoked
            assert await _read(stored, cdn, mode) == b"%PDF-1.7"
        assert sign.call_count == 2
        assert cdn.calls == [first, first, second]

    async def test_mints_are_not_shared_across_callers(self, mode: str) -> None:
        from timbal.state import set_run_context

        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, *[_sign_response(fresh)] * 3) as sign:
            await _read(stored, cdn, mode)
            set_run_context(_platform_ctx(token="someone-else"))
            await _read(stored, cdn, mode)
            set_run_context(_platform_ctx(org_id="43"))
            await _read(stored, cdn, mode)
        assert sign.call_count == 3

    async def test_persist_keeps_the_stored_url(self, mode: str) -> None:
        stored = _signed_url(-3600)
        fresh = _signed_url(3600, signature="new")
        file = File(stored)
        cdn = _FakeCdn(fresh)
        with _sign_endpoint(mode, _sign_response(fresh)):
            if mode == "async":
                await file.load(client=cdn)
            else:
                from unittest.mock import patch

                with patch("httpx.get", side_effect=cdn.get_sync):
                    file.read()
        assert await file.persist() == stored
        assert file.name == "report.pdf"


@pytest.mark.usefixtures("platform_run")
class TestMintMemo:
    def test_forget_only_drops_the_url_that_failed(self) -> None:
        from timbal.types.file import _SignedContentUrl

        target = _SignedContentUrl.resolve(_signed_url(-3600))
        newer = _signed_url(3600, signature="newer")
        target.remember(newer)
        # Another read replaced the mint meanwhile: the failure of the old one must not evict the new one.
        target.forget(_signed_url(3600, signature="older"), 403)
        assert target.cached() == newer
        target.forget(newer, 403)
        assert target.cached() is None

    def test_cache_is_capped(self, monkeypatch) -> None:
        from timbal.types.file import _SignedContentUrl

        monkeypatch.setattr("timbal.types.file._MINT_CACHE_MAX", 2)
        targets = [_SignedContentUrl.resolve(_signed_url(-3600, base=f"{_KB_OBJECT}.{i}")) for i in range(3)]
        for target in targets:
            target.remember(_signed_url(3600, signature="new", base=target.object_url))
        assert [target.cached() is not None for target in targets] == [False, True, True]


def _dns_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "timbal-request-sync-dns"]


def _until(condition, timeout: float = 10.0) -> None:
    """Wait for a state another thread is about to reach (never a fixed sleep)."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the condition"
        time.sleep(0.001)


class TestSyncDnsPool:
    """The executor behind every ``_request_sync`` helper loop: bounded, with no idle threads."""

    def test_runs_work_and_leaves_no_thread_behind(self) -> None:
        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(2)
        futures = [pool.submit(pow, 2, exponent) for exponent in range(20)]
        assert [future.result(10) for future in futures] == [2**exponent for exponent in range(20)]
        _until(lambda: pool._threads_alive == 0)
        assert not pool._queue

    def test_never_runs_more_at_once_than_its_limit(self) -> None:
        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(2)
        gate = threading.Event()
        running = threading.Semaphore(0)

        def blocked() -> str:
            running.release()
            gate.wait(30)
            return threading.current_thread().name

        try:
            futures = [pool.submit(blocked) for _ in range(5)]
            assert running.acquire(timeout=10) and running.acquire(timeout=10)
            # Two are running; the other three wait for one of those threads, they do not get their own.
            assert pool._threads_alive == 2
            assert len(pool._queue) == 3
            assert not running.acquire(blocking=False)
        finally:
            gate.set()
        assert {future.result(10) for future in futures} == {"timbal-request-sync-dns"}
        _until(lambda: pool._threads_alive == 0)

    def test_work_cancelled_while_queued_never_runs(self) -> None:
        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(1)
        gate = threading.Event()
        ran: list[int] = []
        try:
            first = pool.submit(gate.wait, 30)
            queued = [pool.submit(ran.append, n) for n in range(3)]
            assert queued[0].cancel() and queued[2].cancel()
        finally:
            gate.set()
        assert first.result(10) is True
        assert queued[1].result(10) is None
        _until(lambda: pool._threads_alive == 0)
        assert ran == [1]

    def test_cancelled_work_does_not_pile_up_behind_stuck_threads(self) -> None:
        from concurrent.futures import wait

        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(2)
        gate = threading.Event()
        dropped = []
        try:
            for _ in range(2):
                pool.submit(gate.wait, 30)
            for _ in range(50):  # calls that reached their deadline while the pool was stuck
                dropped.append(pool.submit(int))
                assert dropped[-1].cancel()
            assert len(pool._queue) <= 3
        finally:
            gate.set()
        _until(lambda: pool._threads_alive == 0)
        # Dropped from the queue or reached by a thread, each one ends as any cancelled future does.
        assert not wait(dropped, timeout=10).not_done

    def test_an_exception_reaches_the_future(self) -> None:
        import socket

        from timbal.platform.utils import _SyncDnsPool

        def lookup() -> None:
            raise socket.gaierror(8, "no such host")

        pool = _SyncDnsPool(1)
        assert isinstance(pool.submit(lookup).exception(10), socket.gaierror)
        assert pool.submit(int).result(10) == 0  # and the pool goes on working

    def test_shutdown_by_a_loop_does_not_stop_it(self) -> None:
        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(1)
        pool.shutdown(wait=True)  # what closing an event loop does to its default executor
        assert pool.submit(int).result(10) == 0

    def test_work_is_not_left_queued_when_no_thread_can_start(self) -> None:
        from unittest.mock import patch

        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(1)
        with patch("threading.Thread.start", side_effect=RuntimeError("can't start new thread")):
            with pytest.raises(RuntimeError):
                pool.submit(int)
        assert not pool._queue and pool._threads_alive == 0
        assert pool.submit(int).result(10) == 0

    def test_work_is_failed_when_every_concurrent_thread_start_fails(self, monkeypatch) -> None:
        from concurrent.futures import Future

        from timbal.platform.utils import _SyncDnsPool

        submitters = 8
        pool = _SyncDnsPool(submitters)
        under_way = threading.Barrier(submitters)  # every start is under way before the first one fails
        real_start = threading.Thread.start

        def start(thread: threading.Thread) -> None:
            if thread.name != "timbal-request-sync-dns":
                return real_start(thread)
            under_way.wait(10)
            raise RuntimeError("can't start new thread")

        outcomes: list[Future | RuntimeError] = []

        def submit() -> None:
            try:
                outcomes.append(pool.submit(int))
            except RuntimeError as exc:
                outcomes.append(exc)

        monkeypatch.setattr(threading.Thread, "start", start)
        threads = [threading.Thread(target=submit) for _ in range(submitters)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        # No thread exists to run any of it: each lookup failed there and then, none is left waiting for one.
        assert len(outcomes) == submitters
        for outcome in outcomes:
            failure = outcome if isinstance(outcome, RuntimeError) else outcome.exception(timeout=5)
            assert isinstance(failure, RuntimeError)
        assert not pool._queue and pool._threads_alive == 0
        monkeypatch.undo()
        assert pool.submit(int).result(10) == 0

    def test_cancelled_work_stays_cancelled_when_the_queue_is_failed(self, monkeypatch) -> None:
        from concurrent.futures import Future

        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(2)
        go = [threading.Event(), threading.Event()]  # one per thread start, in order of arrival
        arrived = threading.Semaphore(0)
        arrivals: list[threading.Thread] = []
        lock = threading.Lock()
        real_start = threading.Thread.start

        def start(thread: threading.Thread) -> None:
            if thread.name != "timbal-request-sync-dns":
                return real_start(thread)
            with lock:
                arrivals.append(threading.current_thread())
                mine = go[len(arrivals) - 1]
            arrived.release()
            assert mine.wait(10)
            raise RuntimeError("can't start new thread")

        outcomes: dict[threading.Thread, Future | RuntimeError] = {}

        def submit() -> None:
            try:
                outcomes[threading.current_thread()] = pool.submit(int)
            except RuntimeError as exc:
                outcomes[threading.current_thread()] = exc

        monkeypatch.setattr(threading.Thread, "start", start)
        threads = [threading.Thread(target=submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        assert arrived.acquire(timeout=10) and arrived.acquire(timeout=10)
        first, second = arrivals
        go[0].set()  # the first start fails while the second is still under way: its lookup stays queued
        first.join(10)
        queued = outcomes[first]
        assert isinstance(queued, Future) and not queued.done()
        assert queued.cancel()  # its call reached its deadline meanwhile
        go[1].set()  # the second fails too: nothing is left to run the queue
        second.join(10)
        assert isinstance(outcomes[second], RuntimeError)
        assert queued.cancelled()
        assert not pool._queue and pool._threads_alive == 0

    def test_work_a_running_thread_got_to_is_returned_though_its_own_thread_did_not_start(self, monkeypatch) -> None:
        from concurrent.futures import Future

        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(2)
        gate = threading.Event()
        starting, fail_now = threading.Event(), threading.Event()
        real_start = threading.Thread.start
        hold = [False]

        def start(thread: threading.Thread) -> None:
            if hold[0] and thread.name == "timbal-request-sync-dns":
                starting.set()
                assert fail_now.wait(10)
                raise RuntimeError("can't start new thread")
            return real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
        outcomes: list[Future | RuntimeError] = []

        def submit() -> None:
            try:
                outcomes.append(pool.submit(lambda: "looked up"))
            except RuntimeError as exc:
                outcomes.append(exc)

        try:
            busy = pool.submit(gate.wait, 30)  # the pool's first thread
            hold[0] = True
            submitter = threading.Thread(target=submit)
            submitter.start()
            assert starting.wait(10)  # a second thread is being started for the queued lookup...
        finally:
            gate.set()
        assert busy.result(10) is True
        _until(lambda: pool._threads_alive == 1)  # ...but the first got to it, and has exited
        fail_now.set()  # only now does the second thread fail to start
        submitter.join(10)
        (outcome,) = outcomes
        assert isinstance(outcome, Future) and outcome.result(10) == "looked up"
        assert not pool._queue and pool._threads_alive == 0

    def test_work_waits_for_a_running_thread_when_another_cannot_start(self, monkeypatch) -> None:
        from timbal.platform.utils import _SyncDnsPool

        pool = _SyncDnsPool(2)
        gate = threading.Event()
        real_start = threading.Thread.start
        fail = [False]

        def start(thread: threading.Thread) -> None:
            if fail[0] and thread.name == "timbal-request-sync-dns":
                raise RuntimeError("can't start new thread")
            return real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
        try:
            busy = pool.submit(gate.wait, 30)
            fail[0] = True
            queued = pool.submit(int)  # no second thread to be had, but the first one is alive
            assert not queued.done()
        finally:
            gate.set()
        assert busy.result(10) is True
        assert queued.result(10) == 0
        _until(lambda: pool._threads_alive == 0)


@pytest.mark.usefixtures("platform_run")
class TestRequestSync:
    """``_request_sync`` is ``_request`` run on a helper thread: same auth, retries and errors."""

    def test_client_error_raises_platform_error_without_retry(self) -> None:
        from timbal.errors import PlatformError
        from timbal.platform.utils import _request_sync

        platform = _FakePlatform(_platform_response(403, {"error": "no access"}))
        with platform.patched(), pytest.raises(PlatformError) as exc_info:
            _request_sync("POST", "orgs/42/content/sign", json={"url": "x"})
        assert exc_info.value.status_code == 403
        ((method, url, kwargs),) = platform.requests
        assert (method, url) == ("POST", "https://api.timbal.ai/orgs/42/content/sign")
        # The caller's run context (platform config) reached the helper thread.
        assert kwargs["headers"]["Authorization"] == "Bearer token123"
        assert kwargs["json"] == {"url": "x"}

    def test_server_error_is_retried(self) -> None:
        from unittest.mock import AsyncMock, patch

        from timbal.platform.utils import _request_sync

        platform = _FakePlatform(_platform_response(503), _platform_response(200, {"ok": True}))
        with platform.patched(), patch("asyncio.sleep", AsyncMock()) as sleep:
            res = _request_sync("POST", "x", json={})
        assert res.json() == {"ok": True}
        assert len(platform.requests) == 2
        sleep.assert_awaited_once()

    async def test_runs_while_the_callers_event_loop_is_running(self) -> None:
        from timbal.platform.utils import _request_sync

        # A lazy File read can happen on an event loop's thread, where asyncio.run() would raise.
        platform = _FakePlatform(_platform_response(200, {"ok": True}))
        with platform.patched():
            assert _request_sync("POST", "x", json={}).json() == {"ok": True}

    def test_retry_after_past_the_budget_raises_without_sleeping(self) -> None:
        from unittest.mock import AsyncMock, patch

        from timbal.errors import PlatformError
        from timbal.platform.utils import _request_sync

        platform = _FakePlatform(_platform_response(429, headers={"Retry-After": "120"}))
        with platform.patched(), patch("asyncio.sleep", AsyncMock()) as sleep, pytest.raises(PlatformError) as exc_info:
            _request_sync("POST", "x", json={}, total_timeout=30)
        assert exc_info.value.status_code == 429
        assert len(platform.requests) == 1
        sleep.assert_not_awaited()

    def test_short_retries_still_run_inside_the_budget(self) -> None:
        from unittest.mock import AsyncMock, patch

        from timbal.platform.utils import _request_sync

        platform = _FakePlatform(_platform_response(429, headers={"Retry-After": "1"}), _platform_response(200))
        with platform.patched(), patch("asyncio.sleep", AsyncMock()) as sleep:
            _request_sync("POST", "x", json={}, total_timeout=30)
        assert len(platform.requests) == 2
        sleep.assert_awaited_once_with(1.0)

    def test_caller_is_released_at_the_deadline_if_the_helper_thread_is_stuck(self) -> None:
        from unittest.mock import patch

        from timbal.errors import PlatformError
        from timbal.platform.utils import _request_sync

        entered, release = threading.Event(), threading.Event()

        async def stuck(*_args, **_kwargs):
            entered.set()
            release.wait(10)  # blocks the helper thread's loop, so no deadline can fire in it

        with patch("timbal.platform.utils._request", stuck):
            try:
                with pytest.raises(PlatformError, match="total timeout"):
                    _request_sync("POST", "x", json={}, total_timeout=0.05)
                # The helper thread may start late: keep the fake in place until it has picked it up.
                assert entered.wait(10)
            finally:
                release.set()


    def test_stalled_dns_lookups_leave_no_helper_threads_and_fill_only_the_pool(self, monkeypatch) -> None:
        import socket
        from unittest.mock import patch

        from timbal.errors import PlatformError
        from timbal.platform import utils

        pool = utils._SyncDnsPool(2)
        monkeypatch.setattr(utils, "_sync_dns", pool)
        release = threading.Event()
        lookups: list[str] = []

        def stalled(host, *_args, **_kwargs):
            lookups.append(host)
            release.wait(30)
            raise socket.gaierror(8, "resolver stalled")

        async def lookup(*_args, total_timeout, **_kwargs):
            # A request's first step, on the helper's own loop. The name only ever reaches the stub above.
            try:
                async with asyncio.timeout(total_timeout):
                    await asyncio.get_running_loop().getaddrinfo("resolver.invalid", 443)
            except TimeoutError:
                raise PlatformError("lookup timed out") from None

        def helpers() -> list[threading.Thread]:
            return [thread for thread in threading.enumerate() if thread.name == "timbal-request-sync"]

        with patch("timbal.platform.utils._request", lookup), patch("socket.getaddrinfo", stalled):
            try:
                for _ in range(5):
                    with pytest.raises(PlatformError):
                        utils._request_sync("POST", "x", json={}, total_timeout=0.05)
                    # The helper thread ends with its request, though its lookup is still in the resolver.
                    for helper in helpers():
                        helper.join(10)
                    assert not helpers()
                # Five calls timed out; no more lookups are stuck than the pool has threads.
                assert len(lookups) <= 2
                assert len(_dns_threads()) <= 2
            finally:
                release.set()
            # Once the resolver answers, the pool's threads are gone too...
            _until(lambda: not _dns_threads())
            # ...and the lookups queued behind the stuck ones were cancelled with their calls: none ran.
            assert len(lookups) <= 2

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    def test_dns_pool_is_replaced_in_a_forked_child(self) -> None:
        import subprocess
        import sys
        import textwrap

        import timbal

        # In a fresh interpreter: forking this one, with its event loop and threads, proves nothing.
        script = textwrap.dedent(
            """
            import os, sys, threading
            from timbal.platform import utils

            utils._sync_dns = utils._SyncDnsPool(1)
            stuck = threading.Event()
            utils._sync_dns.submit(lambda: (stuck.set(), threading.Event().wait()))  # the pool's one thread, stuck
            assert stuck.wait(10)
            pid = os.fork()
            if pid == 0:
                # That thread does not exist here: the inherited pool would queue this lookup for ever.
                try:
                    ok = utils._sync_dns.submit(int).result(10) == 0
                except Exception:
                    ok = False
                os._exit(0 if ok else 1)
            sys.exit(os.waitpid(pid, 0)[1] >> 8)
            """
        )
        env = {**os.environ, "PYTHONPATH": str(pathlib.Path(timbal.__file__).parents[1])}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr


class _Clock:
    """A monotonic clock that only moves when a test (or a patched sleep) moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds

    def patched(self):
        """Give the request helpers this clock, leaving the event loop's own clock alone."""
        import types
        from unittest.mock import patch

        stack = contextlib.ExitStack()
        stack.enter_context(patch("timbal.platform.utils.time", types.SimpleNamespace(monotonic=self.monotonic)))
        stack.enter_context(patch("asyncio.sleep", self.sleep))
        return stack


@pytest.mark.usefixtures("platform_run")
class TestRetryBudget:
    """``total_timeout`` is a deadline for the whole call: every attempt in full, and the waits between them."""

    @pytest.mark.parametrize("mode", ["async", "sync"])
    async def test_deadline_cancels_a_response_that_drips_in(self, mode: str) -> None:
        from unittest.mock import patch

        import httpx
        from timbal.errors import PlatformError
        from timbal.platform.utils import _request, _request_sync

        cancelled = threading.Event()

        class Drip(httpx.AsyncByteStream):
            # Three reads 0.2 s apart: no single read comes near the 5 s read timeout.
            async def __aiter__(self):
                try:
                    for _ in range(3):
                        await asyncio.sleep(0.2)
                        yield b" "
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        transport = httpx.MockTransport(lambda _req: httpx.Response(200, stream=Drip()))

        @contextlib.asynccontextmanager
        async def request_client(timeout):
            async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
                yield client

        with patch("timbal.platform.utils.request_client", request_client):
            with pytest.raises(PlatformError, match="total timeout"):
                if mode == "async":
                    await _request("POST", "x", json={}, timeout=httpx.Timeout(5.0), total_timeout=0.05)
                else:
                    _request_sync("POST", "x", json={}, timeout=httpx.Timeout(5.0), total_timeout=0.05)
            # The request was cancelled, not left to finish in the background. Waited for inside the
            # patch: the sync path's helper thread may only reach the fake client after the caller raised.
            assert cancelled.wait(10)

    async def test_retry_runs_only_until_the_deadline(self) -> None:
        from unittest.mock import patch

        import httpx
        from timbal.errors import PlatformError
        from timbal.platform.utils import _request

        clock = _Clock()
        deadlines = []
        real_timeout = asyncio.timeout

        def timeout(delay):
            deadlines.append(delay)
            return real_timeout(delay)

        async def slow_429():
            clock.now = 25.0  # the first attempt took 25 s
            return _platform_response(429, headers={"Retry-After": "1"})

        platform = _FakePlatform(slow_429, _platform_response(429, headers={"Retry-After": "120"}))
        with (
            clock.patched(),
            patch("asyncio.timeout", timeout),
            platform.patched(),
            pytest.raises(PlatformError) as exc_info,
        ):
            await _request("POST", "x", json={}, timeout=httpx.Timeout(30.0), total_timeout=30)
        # The retry starts at 26 s and is cancelled 4 s later, not after another 30.
        assert deadlines == [30.0, 4.0]
        # The second 429 asks for 120 s, past the deadline: raised, not slept.
        assert exc_info.value.status_code == 429
        assert len(platform.requests) == 2
        assert clock.now == 26.0

    async def test_no_retry_once_the_attempts_spent_the_budget(self) -> None:
        from timbal.errors import PlatformError
        from timbal.platform.utils import _request

        clock = _Clock()

        async def slow_503():
            clock.now = 29.95  # no 100 ms backoff fits in what is left
            return _platform_response(503)

        platform = _FakePlatform(slow_503)
        with clock.patched(), platform.patched(), pytest.raises(PlatformError) as exc_info:
            await _request("POST", "x", json={}, total_timeout=30)
        assert exc_info.value.status_code == 503
        assert len(platform.requests) == 1
        assert clock.now == 29.95
