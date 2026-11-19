import base64
import hashlib
import io
import mimetypes
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import urlopen

import structlog
from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    # SerializationInfo,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)
from pydantic_core import CoreSchema, core_schema
from uuid_extensions import uuid7

from .. import __version__
from ..platform.types import UploadFileResponse
from ..state import get_or_create_run_context, get_run_context

logger = structlog.get_logger("timbal.types.file")


def _is_local_path(source: str) -> bool:
    """True when ``source`` is a filesystem path rather than a URL.

    ``urlparse`` treats ``C:\\foo`` and ``C:/foo`` as scheme ``c``, so drive-letter
    paths must be detected before URL parsing.
    """
    if source.startswith(("/", "./", "../", "~")):
        return True
    return len(source) >= 2 and source[0].isalpha() and source[1] == ":" and (
        len(source) == 2 or source[2] in ("\\", "/")
    )


# Hosts serving platform-persisted content. URLs on these hosts are already
# durable — re-uploading them via POST /files is wasteful and (worse) renames
# the object using the encoded basename. `platform_config.cdn` is checked too;
# content.timbal.ai is kept here so URLs minted before the timbalusercontent.com
# move (or configs still pinning the old host) are still recognized.
_PLATFORM_CDN_HOSTS = ("timbalusercontent.com", "content.timbal.ai")


def _is_platform_cdn_host(host: str | None, platform_config: Any) -> bool:
    """True when ``host`` (a parsed ``.hostname``) serves platform-persisted content."""
    if not host:
        return False
    # .hostname (unlike .netloc) drops port/userinfo and lowercases, so e.g.
    # "content.timbal.ai:443" still matches. The configured cdn may carry a
    # port too — strip it the same way.
    cdn_host = urlparse(f"//{platform_config.cdn}").hostname
    return host == cdn_host or host in _PLATFORM_CDN_HOSTS


# ---------------------------------------------------------------------------
# Re-signing platform content URLs on read
# ---------------------------------------------------------------------------
# Private platform content is served from the CDN with a CloudFront canned-policy
# signature on the query string (Expires, Signature, Key-Pair-Id). Such a URL
# dies at its Expires (1h by default, 24h at most) or when its signing key is
# revoked, yet persist() keeps it verbatim as a durable reference, so memories
# and traces hold URLs that stop working. Reads heal them through
# POST /orgs/{org_id}/content/sign, which resolves the URL's path back to the
# object, re-checks the caller's access and mints a fresh URL. It ignores the old
# signature, so expired URLs and ones signed with a revoked key both re-sign.
# Only reads change: the stored reference stays the URL that was stored.
#
# Two rules keep a read from ever doing worse than it did before re-signing:
# - A stored URL that has not reached its Expires is the read's first request,
#   exactly as before. Replacement URLs and the sign endpoint come into play
#   only after the CDN refused it (403) or once it is past its Expires.
# - No read fails without having fetched the stored URL: this machine's clock
#   decides what is tried first, never whether the stored URL is tried at all.

_SIGN_BUDGET_SECS = 10.0
"""Deadline for one sign call, retries included (``total_timeout``).

Short on purpose: the sync read path blocks its thread (possibly an event
loop's) for the whole call."""

_MINT_CACHE_MAX = 1_000
"""Cap on memoized mints; the least recently stored entry is evicted past it."""

# (caller, org_id, object URL without query) -> (minted URL, its Expires or None).
_mint_cache: "OrderedDict[tuple[str, str, str], tuple[str, float | None]]" = OrderedDict()
_mint_cache_lock = threading.Lock()  # the sync read path can run on worker threads


def _clear_mint_cache() -> None:
    """Drop every memoized mint (tests; next read of a stale URL re-signs)."""
    with _mint_cache_lock:
        _mint_cache.clear()


def _signed_url_expires(url: str) -> float | None:
    """The ``Expires`` epoch seconds on a signed URL, or None when absent/unparseable."""
    values = parse_qs(urlparse(url).query).get("Expires")
    if not values:
        return None
    try:
        return float(values[0])
    except ValueError:
        return None


def _unexpired(expires: float | None) -> bool:
    """Not past ``expires`` by this machine's clock (or no Expires to go by): the URL may still serve."""
    return expires is None or expires > time.time()


def _failure_label(failure: Any) -> Any:
    """How a failed fetch is logged: its status code, else the exception's type (never a signed URL)."""
    if isinstance(failure, int):
        return failure
    response = getattr(failure, "response", None)
    return getattr(response, "status_code", None) or type(failure).__name__


class _SignedContentUrl:
    """A platform-signed CDN URL being read inside a run that can re-sign it.

    :meth:`resolve` returns None — leave the read alone — unless the URL is on a
    platform CDN host, carries CloudFront signing params, and the current run
    has a platform config with an org subject. Unsigned content (public assets,
    tmp files, legacy content.timbal.ai links) never re-signs.
    """

    __slots__ = ("url", "org_id", "expires", "_cache_key")

    def __init__(self, url: str, org_id: str, cache_key: tuple[str, str, str]) -> None:
        self.url = url
        self.org_id = org_id
        self.expires = _signed_url_expires(url)
        self._cache_key = cache_key

    @classmethod
    def resolve(cls, url: str) -> "_SignedContentUrl | None":
        run_context = get_run_context()
        platform_config = run_context.platform_config if run_context is not None else None
        subject = platform_config.subject if platform_config is not None else None
        if subject is None or not subject.org_id:
            return None
        parsed = urlparse(url)
        if not _is_platform_cdn_host(parsed.hostname, platform_config):
            return None
        query = parse_qs(parsed.query)
        if "Signature" not in query and "Key-Pair-Id" not in query:
            return None
        # The endpoint re-checks the caller's access, so one caller's mint must
        # never serve another caller's read: key the cache by platform + auth.
        caller = hashlib.sha256(
            f"{platform_config.host}\0{platform_config.auth.header_key}\0{platform_config.auth.header_value}".encode()
        ).hexdigest()
        object_url = parsed._replace(query="", fragment="").geturl()
        return cls(url, subject.org_id, (caller, subject.org_id, object_url))

    @property
    def sign_path(self) -> str:
        return f"orgs/{quote(self.org_id, safe='')}/content/sign"

    @property
    def object_url(self) -> str:
        """The URL without its query string — safe to log (no signature)."""
        return self._cache_key[2]

    def unexpired(self) -> bool:
        """The stored URL has not reached its Expires (or carries none), so it may still serve."""
        return _unexpired(self.expires)

    def cached(self) -> str | None:
        """A memoized mint for this object that has not reached its own Expires, if any."""
        with _mint_cache_lock:
            entry = _mint_cache.get(self._cache_key)
        if entry is None or not _unexpired(entry[1]):
            return None
        return entry[0]

    @staticmethod
    def minted_url(sign_response: Any) -> str:
        """The usable URL in a sign response (``signed_url``, else the legacy ``url``)."""
        minted = None
        if isinstance(sign_response, dict):
            minted = sign_response.get("signed_url") or sign_response.get("url")
        if not isinstance(minted, str) or not minted:
            raise ValueError("The content sign endpoint returned no URL.")
        return minted

    def remember(self, minted: str) -> None:
        """Memoize a mint that just served a download, until its own Expires (if it has one)."""
        with _mint_cache_lock:
            _mint_cache.pop(self._cache_key, None)
            if len(_mint_cache) >= _MINT_CACHE_MAX:
                _mint_cache.popitem(last=False)
            _mint_cache[self._cache_key] = (minted, _signed_url_expires(minted))

    def forget(self, url: str, failure: Any) -> None:
        """Drop the memoized mint if it is ``url``, which just failed to serve.

        ``failure`` is the error status it answered or the exception its fetch raised.
        """
        logger.warning(
            "Dropping a memoized platform content URL that failed to download",
            url=self.object_url,
            failure=_failure_label(failure),
        )
        with _mint_cache_lock:
            entry = _mint_cache.get(self._cache_key)
            if entry is not None and entry[0] == url:
                del _mint_cache[self._cache_key]

    def mint_failed(self, exc: Exception) -> Exception:
        """The freshly minted URL did not download (error status, or the fetch raised)."""
        logger.warning(
            "Re-signed platform content URL failed to download",
            url=self.object_url,
            failure=_failure_label(exc),
        )
        return exc

    def sign_failed(self, exc: Exception) -> Exception:
        logger.warning(
            "Could not re-sign platform content URL",
            url=self.object_url,
            status_code=getattr(exc, "status_code", None),
            error=str(exc),
        )
        return exc


def _raise_for_status(res: Any, sign_error: Exception) -> None:
    """raise_for_status(), chaining the sign failure that left the URL unrefreshed."""
    import httpx

    try:
        res.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise exc from sign_error


# The two drivers below are the same flow, one awaiting and one blocking:
# 1. A stored URL that has not reached its Expires is fetched first, exactly as a
#    read that never re-signs would. Its answer is final (the download, or the
#    error raised) unless it is a 403: expired by the CDN's clock, or a revoked
#    key.
# 2. Otherwise a memoized mint for the object that has not reached its own
#    Expires is fetched, and serves the read if it downloads. If it fails in any
#    way (an error status, or the fetch raises: connection refused, timeout, ...)
#    it is dropped.
# 3. Otherwise sign once and fetch the minted URL. A mint is memoized only once
#    it has served a download, so a URL that fails is never reused by later reads.
# 4. If none of that served the read and step 1 skipped the stored URL (it is
#    past its Expires here, but this machine's clock may run ahead of the CDN's),
#    the stored URL is fetched: the CDN has the last word.
# 5. If nothing served: a failed sign call raises the stored URL's error chained
#    to the sign failure; a minted URL that failed to download raises its own
#    error.
# At most one sign call per read, so a sign endpoint that refuses (403) can't
# loop.


async def _get_signed_content(target: _SignedContentUrl, get: Callable[[str], Awaitable[Any]]) -> Any:
    import httpx

    from ..platform.utils import _request

    async def mint() -> str:
        res = await _request(
            "POST",
            target.sign_path,
            json={"url": target.url},
            timeout=httpx.Timeout(_SIGN_BUDGET_SECS),
            total_timeout=_SIGN_BUDGET_SECS,
        )
        return target.minted_url(res.json())

    # 1. The stored URL, while it may still serve.
    stored = None  # its error response, once the CDN has answered it
    if target.unexpired():
        res = await get(target.url)
        if res.status_code != 403:
            res.raise_for_status()
            return res
        stored = res

    # 2. A memoized mint.
    cached = target.cached()
    if cached is not None:
        try:
            res = await get(cached)
        except Exception as exc:
            target.forget(cached, exc)
        else:
            if res.is_success:
                return res
            target.forget(cached, res.status_code)

    # 3. A fresh mint.
    sign_error = mint_error = None
    try:
        minted = await mint()
    except Exception as exc:
        sign_error = target.sign_failed(exc)
    else:
        try:
            res = await get(minted)
            res.raise_for_status()
        except Exception as exc:
            mint_error = target.mint_failed(exc)
        else:
            if minted != target.url:
                target.remember(minted)
            return res

    # 4. The stored URL as the last resort, if step 1 skipped it.
    if stored is None:
        try:
            res = await get(target.url)
        except Exception as exc:
            if mint_error is None:
                raise exc from sign_error
        else:
            if res.is_success:
                return res
            stored = res

    # 5. Nothing served.
    if mint_error is not None:
        raise mint_error
    _raise_for_status(stored, sign_error)


def _get_signed_content_sync(target: _SignedContentUrl, get: Callable[[str], Any]) -> Any:
    import httpx

    from ..platform.utils import _request_sync

    def mint() -> str:
        res = _request_sync(
            "POST",
            target.sign_path,
            json={"url": target.url},
            timeout=httpx.Timeout(_SIGN_BUDGET_SECS),
            total_timeout=_SIGN_BUDGET_SECS,
        )
        return target.minted_url(res.json())

    # 1. The stored URL, while it may still serve.
    stored = None  # its error response, once the CDN has answered it
    if target.unexpired():
        res = get(target.url)
        if res.status_code != 403:
            res.raise_for_status()
            return res
        stored = res

    # 2. A memoized mint.
    cached = target.cached()
    if cached is not None:
        try:
            res = get(cached)
        except Exception as exc:
            target.forget(cached, exc)
        else:
            if res.is_success:
                return res
            target.forget(cached, res.status_code)

    # 3. A fresh mint.
    sign_error = mint_error = None
    try:
        minted = mint()
    except Exception as exc:
        sign_error = target.sign_failed(exc)
    else:
        try:
            res = get(minted)
            res.raise_for_status()
        except Exception as exc:
            mint_error = target.mint_failed(exc)
        else:
            if minted != target.url:
                target.remember(minted)
            return res

    # 4. The stored URL as the last resort, if step 1 skipped it.
    if stored is None:
        try:
            res = get(target.url)
        except Exception as exc:
            if mint_error is None:
                raise exc from sign_error
        else:
            if res.is_success:
                return res
            stored = res

    # 5. Nothing served.
    if mint_error is not None:
        raise mint_error
    _raise_for_status(stored, sign_error)


def _extract_filename(headers: dict, url: str) -> str:
    """Extract filename from Content-Disposition header, falling back to URL path."""
    filename = None
    content_disposition = headers.get("Content-Disposition")
    if content_disposition and "filename=" in content_disposition:
        try:
            filename = content_disposition.split("filename=")[1].strip("\"'")
        except (IndexError, AttributeError):
            pass
    if not filename:
        # Decode the basename: URL paths are percent-encoded, but the filename is
        # used as-is for re-uploads — keeping it encoded double-encodes downstream.
        filename = unquote(urlparse(url).path.split("/")[-1])
    return filename


class File(io.IOBase):
    """A wrapper class that provides a uniform file-like interface for various file sources.

    This class can handle local files, URLs, S3 URIs, and data URLs, providing a consistent
    interface for file operations regardless of the source. It implements the io.IOBase interface
    and provides lazy loading of file contents.

    Attributes:
        __source__: The original source identifier (path, URL, etc.)
        __source_scheme__: The scheme of the source (local_path, url, s3, data_url, bytes)
        __source_extension__: The file extension if available
        __fileobj__: The actual file object once loaded
        __fetcher__: Function to fetch/load the file content
        __persisted__: The url of the file once persisted on the platform
        __content_type__: The content type of the file
    """

    __slots__ = (
        "__source__",
        "__source_scheme__",
        "__source_extension__",
        "__fileobj__",
        "__fetcher__",
        "__persisted__",
        "__content_type__",
    )

    _UNSET = object()

    def __new__(cls, source: Any, *args, **kwargs):  # noqa: ARG004
        if isinstance(source, cls):
            return source  # identity shortcut — same instance, no copy
        return super().__new__(cls)

    def __init__(
        self,
        source: Any,
        source_scheme: Any = _UNSET,
        source_extension: str | None = None,
        fetcher: Callable[[], io.IOBase] | None = None,
        *,
        extension: str | None = None,
        name: str | None = None,
    ) -> None:
        # Guard: __new__ returned an existing File, already initialized
        try:
            object.__getattribute__(self, "__source__")
            return
        except AttributeError:
            pass

        if source_scheme is not File._UNSET:
            # Internal direct path — preserve full backwards compat
            self._setup(source, source_scheme, source_extension, fetcher)
            return

        # Smart routing path — what validate() used to do
        if isinstance(source, bytes | bytearray):
            bio = io.BytesIO(source)
            if name is not None:
                bio.name = name
            self._setup(bio, "bytes", extension)
            return

        if isinstance(source, io.IOBase):
            if name is not None:
                source.name = name
            self._setup(source, "bytes", extension)
            return

        if isinstance(source, Path):
            source = source.expanduser().resolve().as_posix()

        if not isinstance(source, str):
            raise ValueError("File must be a string, path, or file-like object.")

        if _is_local_path(source):
            path = Path(source).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise ValueError("Invalid file path.")
            self._setup(
                source, "local_path", path.suffix,
                fetcher=lambda: self._fetch_local_file(path=path),
            )
            return

        parsed_url = urlparse(source)

        # Local path
        if parsed_url.scheme == "":
            path = Path(source).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise ValueError("Invalid file path.")
            self._setup(
                source, "local_path", path.suffix,
                fetcher=lambda: self._fetch_local_file(path=path),
            )
            return

        # Data URL
        if parsed_url.scheme == "data":
            parsed_url_mimetype = parsed_url.path.split(";", 1)
            if len(parsed_url_mimetype) != 2:
                raise ValueError("Data url must specify a mimetype.")
            parsed_url_mimetype = parsed_url_mimetype[0]
            parsed_url_extension = mimetypes.guess_extension(parsed_url_mimetype)
            if parsed_url_extension is None:
                if not parsed_url_mimetype.startswith("timbal/"):
                    raise ValueError("Custom mimetypes must be specified as 'timbal/<extension>'.")
                parsed_url_extension = "." + parsed_url_mimetype[7:]
            self._setup(
                source, "data_url", parsed_url_extension,
                fetcher=lambda: self._fetch_data_url_file(data_url=source),
            )
            return

        # HTTP/HTTPS URL
        if parsed_url.scheme in ("http", "https"):
            parsed_url_name = parsed_url.path.split("/")[-1]
            parsed_url_extension = f".{parsed_url_name.split('.')[-1]}" if "." in parsed_url_name else ""
            self._setup(
                source, "url", parsed_url_extension,
                fetcher=lambda: self._fetch_http_file(url=source),
            )
            return

        raise ValueError("Invalid file source. Must be a local file path, data URL, or HTTP/HTTPS url.")

    def _setup(
        self,
        source: Any,
        source_scheme: str,
        source_extension: str | None = None,
        fetcher: Callable[[], io.IOBase] | None = None,
    ) -> None:
        """Initialize internal slots. This is the original __init__ body."""
        object.__setattr__(self, "__source__", source)
        object.__setattr__(self, "__source_scheme__", source_scheme)
        object.__setattr__(self, "__source_extension__", source_extension)
        object.__setattr__(self, "__fetcher__", fetcher)
        object.__setattr__(self, "__fetch_error__", None)
        object.__setattr__(self, "__persisted__", None)

        if isinstance(source, io.IOBase):
            object.__setattr__(self, "__fileobj__", source)
        else:
            object.__setattr__(self, "__fileobj__", None)

        if source_scheme == "bytes":
            if self.__source_extension__:
                content_type, _ = mimetypes.guess_type(f"tmp{self.__source_extension__}")
                # Python 3.11 doesn't recognize .md files, add fallback
                if content_type is None and self.__source_extension__ == ".md":
                    content_type = "text/markdown"
                elif content_type is None:
                    content_type = "application/octet-stream"
            else:
                content_type = "application/octet-stream"
        else:
            content_type, _ = mimetypes.guess_type(str(self))
            # Python 3.11 doesn't recognize .md files, add fallback
            if content_type is None and str(self).endswith(".md"):
                content_type = "text/markdown"
            elif content_type is None:
                content_type = "application/octet-stream"
        object.__setattr__(self, "__content_type__", content_type)

    def __str__(self) -> str:
        try:
            if self.__source_scheme__ == "bytes":
                ext_info = f"{self.__source_extension__}" if self.__source_extension__ else ""
                return f"io.IOBase({ext_info})"
            return self.__source__
        except AttributeError:
            # Half-constructed File (e.g. __init__ raised before _setup). Keep
            # repr() safe so finalization/unraisable-hook paths don't blow up.
            return "File(uninitialized)"

    def __repr__(self) -> str:
        return f"File(source={str(self)})"

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access through to the wrapped file object."""
        if name.startswith("__"):
            # Never route a dunder / internal slot to the lazy fetcher: the
            # private slots (__fileobj__, __source_scheme__, __wrapped__, ...) and
            # CPython's own finalizer probes (__IOBase_closed, which starts with
            # "__" but does not end with "__") all land here only when genuinely
            # unset — e.g. a File whose __init__ raised before _setup() ran, being
            # repr'd or finalized. Fetching would be wrong, and re-entering via
            # __wrapped__ (whose getter itself raises AttributeError when
            # __fileobj__ is unset) recurses until RecursionError. Proxied file
            # attributes (read, name, seek, ...) never start with "__".
            raise AttributeError(name)
        if name == "extension":
            return object.__getattribute__(self, "__source_extension__")
        elif name == "persisted":
            return object.__getattribute__(self, "__persisted__")
        elif name == "name":
            wrapped = self.__wrapped__
            if not hasattr(wrapped, "name"):
                fileext = object.__getattribute__(self, "__source_extension__") or ""
                wrapped.name = f"{uuid7()}{fileext}"
            return wrapped.name
        # TODO Add more aliases here
        else:
            return getattr(self.__wrapped__, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute assignment through to the wrapped file object."""
        if name in ("__IOBase_closed", "_finalizing"):
            # Set by io.IOBase's C finalizer/close(); keep these on this
            # instance rather than proxying, which would trigger the lazy fetcher.
            object.__setattr__(self, name, value)
            return
        if hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__wrapped__, name, value)

    def __delattr__(self, name: str) -> None:
        """Proxy attribute deletion through to the wrapped file object."""
        if hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.__wrapped__, name)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over the wrapped file object."""
        return iter(self.__wrapped__)

    def __copy__(self) -> "File":
        """Return self for shallow copy - File objects are immutable references."""
        return self

    def __deepcopy__(self, memo: dict) -> "File":
        """Return self for deep copy - File objects are immutable references.

        File objects represent immutable references to file sources and should not
        be deeply copied. The underlying file content and metadata remain the same,
        so we return the same instance to avoid issues with copying file handles
        and other system resources.

        Args:
            memo: Dictionary used by deepcopy to track already copied objects

        Returns:
            File: The same File instance
        """
        return self

    # TODO Remove the fetching from here. Move to validation
    @property
    def __wrapped__(self) -> Any:
        """Get the underlying file object, fetching it if necessary."""
        fileobj = object.__getattribute__(self, "__fileobj__")
        if fileobj is None:
            # A failed fetch is cached and re-raised: every proxied attribute access
            # lands here, so without this a broken URL is re-fetched on each access.
            # getattr with default: an AttributeError escaping this property would
            # send Python back through __getattr__ and recurse. Also covers Files
            # constructed without _setup (e.g. via __new__ in tests).
            try:
                fetch_error = object.__getattribute__(self, "__fetch_error__")
            except AttributeError:
                fetch_error = None
            if fetch_error is not None:
                raise fetch_error
            fetcher = object.__getattribute__(self, "__fetcher__")
            if fetcher is None:
                raise ValueError("File object is not properly initialized.")
            try:
                fileobj = fetcher()
            except Exception as e:
                object.__setattr__(self, "__fetch_error__", e)
                raise
            # Some libraries (e.g. openai) require file-like objects to have the name property defined.
            if not hasattr(fileobj, "name"):
                fileext = object.__getattribute__(self, "__source_extension__")
                if fileext:
                    fileobj.name = f"{uuid7()}{fileext}"
            if hasattr(fileobj, "__content_type__"):
                object.__setattr__(self, "__content_type__", fileobj.__content_type__)
            object.__setattr__(self, "__fileobj__", fileobj)
        return fileobj

    def close(self) -> None:
        """Close the wrapped file object if it was ever materialized.

        io.IOBase.__del__ calls close() at garbage collection; this must never
        trigger the lazy fetcher (a real network request for URL sources) just
        to close content that was never loaded, nor raise for a File whose
        __init__ failed before _setup() ran — an exception escaping close()
        propagates into __del__ and becomes an unraisable exception.
        """
        try:
            fileobj = object.__getattribute__(self, "__fileobj__")
        except AttributeError:
            return
        if fileobj is not None:
            fileobj.close()

    # Overwrite some io.IOBase methods to proxy to the wrapped file object (explicitly).
    def readable(self) -> bool:
        return self.__getattr__("readable")()

    def writable(self) -> bool:
        return self.__getattr__("writable")()

    def seekable(self) -> bool:
        return self.__getattr__("seekable")()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.__getattr__("seek")(offset, whence)

    async def load(self, client: Any = None) -> None:
        """Eagerly load the file content. For URL sources this fetches async via httpx.

        Args:
            client: Optional shared httpx.AsyncClient for connection reuse.
                    If None, a temporary client is created and closed after the request.
        """
        if object.__getattribute__(self, "__fileobj__") is not None:
            return
        if self.__source_scheme__ != "url":
            # Local/bytes/data_url — trigger existing sync fetcher (cheap, no network I/O)
            _ = self.__wrapped__
            return
        import httpx

        url = str(self)
        headers = {"User-Agent": f"Timbal/{__version__}"}
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

        async def get(fetch_url: str) -> httpx.Response:
            return await client.get(fetch_url, headers=headers, follow_redirects=True)

        try:
            signed = _SignedContentUrl.resolve(url)
            if signed is None:
                res = await get(url)
                res.raise_for_status()
            else:
                res = await _get_signed_content(signed, get)
        finally:
            if own_client:
                await client.aclose()
        fileobj = io.BytesIO(res.content)
        fileobj.name = _extract_filename(res.headers, url)
        server_content_type = res.headers.get("Content-Type", "application/octet-stream")
        if server_content_type not in ("application/octet-stream", "binary/octet-stream"):
            fileobj.__content_type__ = server_content_type
        if hasattr(fileobj, "__content_type__"):
            object.__setattr__(self, "__content_type__", fileobj.__content_type__)
        object.__setattr__(self, "__fileobj__", fileobj)

    @classmethod
    def validate(
        cls,
        value: ValidatorFunctionWrapHandler,
        info: dict | ValidationInfo | None = None,  # noqa: ARG003
    ) -> "File":
        """Pydantic-compat shim. Delegates to File(source, extension=..., name=...)."""
        extension = info.get("extension") if isinstance(info, dict) else None
        name = info.get("name") if isinstance(info, dict) else None
        return cls(value, extension=extension, name=name)

    def to_disk(self, path: Path) -> None:
        """Save the file to disk."""
        # Ensure the file obj has the pointer at the start of the file.
        self.seek(0)
        content = self.read()
        with open(path, "wb") as f:
            f.write(content)

    def to_data_url(self) -> str:
        """Serialize the file to a data url string."""
        if self.__source_scheme__ == "data_url":
            return str(self)

        # Ensure the file obj has the pointer at the start of the file.
        self.seek(0)
        content = self.read()

        bs64_content = base64.b64encode(content).decode("utf-8")
        return f"data:{self.__content_type__};base64,{bs64_content}"

    async def persist(self) -> str | None:
        """Persist the file to some storage.
        If there's no run context or valid platform config, the file will be persisted to local disk.
        If there's a platform configuration, the file will be uploaded to the platform.
        If the file is already persisted, it will be returned as is.
        """
        if self.__persisted__ is not None:
            return self.__persisted__

        run_context = get_or_create_run_context()

        if self.__source_scheme__ == "url":
            url = str(self)
            if not run_context.platform_config:
                return url
            # Kept verbatim, signature included: reads re-sign it once it
            # expires (see _SignedContentUrl), so stored references heal.
            if _is_platform_cdn_host(urlparse(url).hostname, run_context.platform_config):
                object.__setattr__(self, "__persisted__", url)
                return url

        if not run_context.platform_config or not run_context.platform_config.subject:
            if self.__source_scheme__ == "local_path":
                local_path = str(self)
                object.__setattr__(self, "__persisted__", local_path)
                return local_path

            temp_dir = tempfile.gettempdir()
            temp_name = str(uuid7())
            if self.__source_extension__:
                temp_name += self.__source_extension__
            temp_path = Path(temp_dir) / temp_name
            self.to_disk(temp_path)
            temp_path = temp_path.as_posix()
            object.__setattr__(self, "__persisted__", temp_path)
            return temp_path

        self.seek(0)
        content = self.read()
        self.seek(0)  # Return the pointer to the beginning of the file

        path = "files"
        files = {"file": (self.name, content, self.__content_type__)}
        # The platform counts the upload against this org's files_upload_rate
        # instead of the per-user default. It only accepts numeric org ids;
        # platforms without the parameter ignore it.
        org_id = run_context.platform_config.subject.org_id
        params = {"org_id": org_id} if org_id.isdigit() else None

        from ..platform.utils import _request

        res = await _request("POST", path, params=params, files=files)
        upload_response = UploadFileResponse.model_validate(res.json())
        # The platform returns a fully percent-encoded URL — use it verbatim.
        # Reconstructing it from `name` (raw, unencoded) breaks whenever the name
        # needs encoding: removesuffix() finds no match, so the encoded name got
        # appended to the already-complete URL, yielding a duplicated, doubly
        # encoded path that 403s on the CDN.
        url = upload_response.url
        object.__setattr__(self, "__persisted__", url)
        return url

    @classmethod
    def serialize(
        cls,
        value: Any,
        *args,  # noqa: ARG003
        **kwargs,  # noqa: ARG003
    ) -> str:
        """Serialize the file to a data url string. Bytes-like files are not supported will have octet-stream as content type."""
        # When creating a model with fields with File type that are nullable,
        # pydantic will pass None as the value to File.serialize.
        if value is None:
            return None

        if not isinstance(value, cls):
            raise ValueError("Cannot serialize a non-file object.")

        if value.__persisted__ is not None:
            return value.__persisted__
        else:
            return value.to_data_url()

    @classmethod
    def _fetch_local_file(cls, path: Path) -> io.IOBase:
        """Fetch a local file from a valid path in the system."""
        # We don't use path.open("rb") because it used the full path as the name,
        # and we weren't able to et the name property after the fact.
        fileobj = io.BytesIO(path.read_bytes())
        fileobj.name = path.name
        return fileobj

    @classmethod
    def _fetch_data_url_file(cls, data_url: str) -> io.IOBase:
        """Parse a data URL and return a file-like object."""
        res = urlopen(data_url)  # noqa: S310
        return io.BytesIO(res.read())

    @classmethod
    def _fetch_http_file(cls, url: str) -> io.IOBase:
        """Fetch a file from a HTTP/HTTPS URL (sync fallback)."""
        import httpx

        headers = {"User-Agent": f"Timbal/{__version__}"}

        def get(fetch_url: str) -> httpx.Response:
            return httpx.get(fetch_url, headers=headers, follow_redirects=True, timeout=30.0)

        signed = _SignedContentUrl.resolve(url)
        if signed is None:
            res = get(url)
            res.raise_for_status()
        else:
            res = _get_signed_content_sync(signed, get)
        fileobj = io.BytesIO(res.content)
        fileobj.name = _extract_filename(res.headers, url)
        # Set content type from server response
        # NOTE: This will overwrite the File's extension-based content type detection
        server_content_type = res.headers.get("Content-Type", "application/octet-stream")
        # Only set __content_type__ if server provides a specific type (not generic binary)
        if server_content_type not in ("application/octet-stream", "binary/octet-stream"):
            fileobj.__content_type__ = server_content_type
        # For generic types, let the File class keep its extension-based detection
        return fileobj

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler) -> dict[str, Any]:
        """Defines what this type should be in openapi.json."""
        # https://docs.pydantic.dev/2.8/errors/usage_errors/#custom-json-schema
        json_schema = {
            "type": "string",
            "format": "uri",
            "description": "A file reference which can be a local path, a URL, or a data URL.",
        }
        return json_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Defines how to serialize this type in the core schema."""
        return core_schema.with_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize,
                info_arg=False,
                when_used="always",
            ),
        )
