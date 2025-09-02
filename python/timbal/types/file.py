import base64
import io
import mimetypes
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import requests
from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    # SerializationInfo,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)
from pydantic_core import CoreSchema, core_schema
from uuid_extensions import uuid7

from ..state import get_or_create_run_context
from ..utils import _platform_api_call


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



    def __init__(
        self,
        source: Any,
        source_scheme: str,
        source_extension: str | None = None,
        fetcher: Callable[[], io.IOBase] | None = None,
    ) -> None:
        object.__setattr__(self, "__source__", source)
        object.__setattr__(self, "__source_scheme__", source_scheme)
        object.__setattr__(self, "__source_extension__", source_extension)
        object.__setattr__(self, "__fetcher__", fetcher)
        object.__setattr__(self, "__persisted__", None)

        if isinstance(source, io.IOBase):
            object.__setattr__(self, "__fileobj__", source)
        else:
            object.__setattr__(self, "__fileobj__", None)

        if source_scheme == "bytes":
            if self.__source_extension__:
                content_type, _ = mimetypes.guess_type(f"tmp{self.__source_extension__}")
                if content_type is None:
                    content_type = "application/octet-stream"
            else:
                content_type = "application/octet-stream"
        else:
            content_type, _ = mimetypes.guess_type(str(self))
            if content_type is None:
                content_type = "application/octet-stream"
        object.__setattr__(self, "__content_type__", content_type)

    
    def __str__(self) -> str:
        if self.__source_scheme__ == "bytes":
            ext_info = f"{self.__source_extension__}" if self.__source_extension__ else ""
            return f"io.IOBase({ext_info})"
        return self.__source__


    def __repr__(self) -> str:
        return f"File(source={str(self)})"


    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access through to the wrapped file object."""
        if name == "extension":
            return object.__getattribute__(self, "__source_extension__")
        elif name == "persisted":
            return object.__getattribute__(self, "__persisted__")
        # TODO Add more aliases here
        else:
            return getattr(self.__wrapped__, name)


    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute assignment through to the wrapped file object."""
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
            fetcher = object.__getattribute__(self, "__fetcher__")
            if fetcher is None:
                raise ValueError("File object is not properly initialized.")
            fileobj = fetcher()
            # Some libraries (e.g. openai) require file-like objects to have the name property defined.
            if not hasattr(fileobj, "name"):
                fileext = object.__getattribute__(self, "__source_extension__")
                if fileext:
                    fileobj.name = f"{uuid7()}{fileext}"
            if hasattr(fileobj, "__content_type__"):
                object.__setattr__(self, "__content_type__", fileobj.__content_type__)
            object.__setattr__(self, "__fileobj__", fileobj)
        return fileobj


    # Overwrite some io.IOBase methods to proxy to the wrapped file object (explicitly).
    def readable(self) -> bool:
        return self.__getattr__("readable")()


    def writable(self) -> bool:
        return self.__getattr__("writable")()


    def seekable(self) -> bool:
        return self.__getattr__("seekable")()


    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.__getattr__("seek")(offset, whence)


    @classmethod
    def validate(
        cls, 
        value: ValidatorFunctionWrapHandler, 
        info: dict | ValidationInfo | None = None, # noqa: ARG003
    ) -> "File":
        """Create a new Field instance validating a local path, an url, an s3 uri, a data url or a file-like object.
        Validation info can be used to pass context information to the file fetcher:
            >>> model_instance = model.model_validate(
            ...     {**model_params_dict},
            ...     context={...},
            ... )
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, bytes | bytearray):
            source_extension = None
            source_name = None
            if isinstance(info, dict):
                source_extension = info.get("extension")
                source_name = info.get("name")
            source = io.BytesIO(value)
            if source_name is not None:
                source.name = source_name
            return File(
                source, 
                source_scheme="bytes", 
                source_extension=source_extension,
            )

        if isinstance(value, io.IOBase):
            source_extension = None
            source_name = None
            if isinstance(info, dict):
                source_extension = info.get("extension")
                source_name = info.get("name")
            source = value
            if source_name is not None:
                source.name = source_name
            return File(
                source, 
                source_scheme="bytes", 
                source_extension=source_extension,
            )

        if isinstance(value, Path):
            value = value.expanduser().resolve().as_posix()

        if not isinstance(value, str):
            raise ValueError("File must be a string, path, or file-like object.")

        parsed_url = urlparse(value)

        # Check for valid local path.
        if parsed_url.scheme == "":
            path = Path(value).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise ValueError("Invalid file path.")
            parsed_url_extension = path.suffix
            return File(
                value,
                source_scheme="local_path",
                source_extension=parsed_url_extension,
                fetcher=lambda: cls._fetch_local_file(path=path),
            )

        # Data urls: data:[<mediatype>][;base64],<data>.
        if parsed_url.scheme == "data":
            parsed_url_mimetype = parsed_url.path.split(";", 1)
            if len(parsed_url_mimetype) != 2:
                raise ValueError("Data url must specify a mimetype.")
            parsed_url_mimetype = parsed_url_mimetype[0]
            parsed_url_extension = mimetypes.guess_extension(parsed_url_mimetype)
            if parsed_url_extension is None:
                # Custom mimetypes must be specified as ^timbal\/\w*$
                if not parsed_url_mimetype.startswith("timbal/"):
                    raise ValueError("Custom mimetypes must be specified as 'timbal/<extension>'.")
                parsed_url_extension = "." + parsed_url_mimetype[7:]
            return File(
                value,
                source_scheme="data_url",
                source_extension=parsed_url_extension,
                fetcher=lambda: cls._fetch_data_url_file(data_url=value),
            )

        if parsed_url.scheme == "http" or parsed_url.scheme == "https":
            parsed_url_name = parsed_url.path.split("/")[-1]
            parsed_url_extension = f".{parsed_url_name.split('.')[-1]}" if "." in parsed_url_name else ""
            return File(
                value,
                source_scheme="url",
                source_extension=parsed_url_extension,
                fetcher=lambda: cls._fetch_http_file(url=value),
            )

        raise ValueError("Invalid file source. Must be a local file path, data URL, or HTTP/HTTPS url.")

    
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

    
    async def persist(
        self, 
        org_id: str | None = None,
        # ? Add more resource specifiers
    ) -> str | None:
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
            elif url.startswith(f"https://{run_context.platform_config.cdn}"):
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

        subject = run_context.platform_config.subject
        org_id = org_id or subject.org_id
        # ? We could add more subject info here

        self.seek(0)
        content = self.read()

        path = f"orgs/{org_id}/files"
        files = {"file": (self.name, content, self.__content_type__)}

        res = await _platform_api_call("POST", path, files=files)
        res_json = res.json()
        # ? We could use an UploadFileResponse pydantic model
        url = res_json["url"]
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


    # TODO Make async
    @classmethod
    def _fetch_http_file(cls, url: str) -> io.IOBase:
        """Fetch a file from a HTTP/HTTPS URL."""
        res = requests.get(url, stream=True)
        res.raise_for_status()
        fileobj = io.BytesIO(res.content)
        # Try to get a meaningful filename from URL or Content-Disposition header
        filename = None
        # Check Content-Disposition header first
        content_disposition = res.headers.get("Content-Disposition")
        if content_disposition and "filename=" in content_disposition:
            try:
                filename = content_disposition.split("filename=")[1].strip('"\'')
            except (IndexError, AttributeError):
                pass
        # Fallback to URL path
        if not filename:
            filename = url.split("/")[-1]
        fileobj.name = filename
        fileobj.__content_type__ = res.headers.get("Content-Type", "application/octet-stream")
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
