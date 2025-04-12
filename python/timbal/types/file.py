"""
This module provides a File class that wraps various types of file sources,
including local files, URLs, enabling uniform interaction through
a file-like interface.

It is inspired by and based on the design found in the Replicate Cog project.
See original implementation at:
https://github.com/replicate/cog/blob/main/python/cog/types.py

Usage:

1. Validating and opening a local file:
   >>> file_instance = File.validate('path/to/local/file.txt')
   >>> print(file_instance.read())

2. Validating and opening a data url:
   >>> file_instance = File.validate('data:text/plain;base64,SGVsbG8sIFdvcmxkIQ%3D%3D')
   >>> print(file_instance.read()) # b'Hello, World!'

3. Opening a file from a URL:
   >>> from PIL import Image
   >>> file_instance = File.validate('https://example.com/image.png')
   >>> image = Image.open(file_instance)
   >>> image.show()

4. Opening a file-like object:
   >>> from io import BytesIO
   >>> file_instance = File.validate(BytesIO(b'Hello, World!'))
   >>> print(file_instance.read())
"""

import base64
import io
import mimetypes
import os
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

from ..state.context import RunContext


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

    __max_file_size__: int | None = (
        int(os.getenv("TIMBAL_MAX_FILE_SIZE")) if os.getenv("TIMBAL_MAX_FILE_SIZE") else None
    )
    """ Maximum allowed file size in bytes"""


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
                    content_type = f"timbal/{self.__source_extension__}"
            else:
                content_type = "application/octet-stream"
        else:
            content_type, _ = mimetypes.guess_type(str(self))
            if content_type is None:
                content_type = f"timbal/{self.__source_extension__}"
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
        if name in ("__source__", "__source_scheme__", "__source_extension__", "__fileobj__", "__fetcher__"):
            raise AttributeError(name)
        elif name == "persisted":
            return object.__getattribute__(self, "__persisted__")
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
    

    @property
    def __wrapped__(self) -> Any:
        """Get the underlying file object, fetching it if necessary."""
        fileobj = object.__getattribute__(self, "__fileobj__")
        if fileobj is None:
            fetcher = object.__getattribute__(self, "__fetcher__")
            if fetcher is None:
                raise ValueError("File object is not properly initialized.")
            fileobj = fetcher()
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
            if isinstance(info, dict):
                source_extension = info.get("extension")
            return File(io.BytesIO(value), source_scheme="bytes", source_extension=source_extension)

        if isinstance(value, io.IOBase):
            source_extension = None
            if isinstance(info, dict):
                source_extension = info.get("extension")
            return File(value, source_scheme="bytes", source_extension=source_extension)

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
            if cls.__max_file_size__ is not None:
                file_size = path.stat().st_size  # bytes
                if file_size > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
            parsed_url_extension = path.suffix
            return File(
                value,
                source_scheme="local_path",
                source_extension=parsed_url_extension,
                fetcher=lambda: cls._fetch_local_file(path=path),
            )

        # Data urls: data:[<mediatype>][;base64],<data>.
        if parsed_url.scheme == "data":
            if cls.__max_file_size__ is not None:
                comma_idx = value.find(",")
                # Redundant check.
                if comma_idx == -1:
                    raise ValueError("Invalid data URL.")
                encoded = value[comma_idx + 1 :]
                # ? We could count the padding if we want to be more precise.
                estimated_size = len(encoded) * 3 // 4
                if estimated_size > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
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
            response = requests.head(value)
            if cls.__max_file_size__ is not None:
                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
            parsed_url_name = parsed_url.path.split("/")[-1]
            parsed_url_extension = f".{parsed_url_name.split('.')[-1]}" if "." in parsed_url_name else ""
            return File(
                value,
                source_scheme="url",
                source_extension=parsed_url_extension,
                fetcher=lambda: cls._fetch_http_file(url=value),
            )

        raise ValueError("Invalid file source. Must be a local file path, data URL, or HTTP/HTTPS url.")

    
    def persist(self, context: RunContext) -> None:
        """Persist the file to the Timbal platform."""
        host = context.timbal_platform_config.host

        auth_config = context.timbal_platform_config.auth_config
        headers = {auth_config.header_key: auth_config.header_value}

        app_config = context.timbal_platform_config.app_config
        org_id = app_config.org_id
        app_id = app_config.app_id
        resource_path = f"orgs/{org_id}/apps/{app_id}/runs/{context.id}"

        # Ensure the file obj has the pointer at the start of the file.
        self.seek(0)
        content = self.read()
        size = len(content)

        body = {
            "name": f"{uuid7()}{self.__source_extension__}",
            "content_type": self.__content_type__,
            "content_length": size,
        }

        res = requests.post(
            f"https://{host}/{resource_path}/files", 
            headers=headers,
            json=body,
        )
        res.raise_for_status()

        res_body = res.json()
        uploader = res_body.get("uploader")
        if uploader is None:
            return

        upload_uri = uploader.get("upload_uri")
        upload_headers = {
            "Content-Length": str(size),
            "Content-Type": self.__content_type__,
        }

        upload_res = requests.put(
            upload_uri, 
            data=content, 
            headers=upload_headers,
        )
        upload_res.raise_for_status()

        content_url = uploader.get("content_url")
        object.__setattr__(self, "__persisted__", content_url)

    
    def to_data_url(self) -> str:
        """Serialize the file to a data url string."""
        if self.__source_scheme__ == "data_url":
            return str(self)

        # Ensure the file obj has the pointer at the start of the file.
        self.seek(0)
        content = self.read()

        bs64_content = base64.b64encode(content).decode("utf-8")
        return f"data:{self.__content_type__};base64,{bs64_content}"


    @classmethod
    def serialize(
        cls, 
        value: Any, 
        context: RunContext | None = None,
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

        if isinstance(context, RunContext) and context.timbal_platform_config is not None:
            if value.__source_scheme__ == "url":
                url = str(value)
                if url.startswith(f"https://{context.timbal_platform_config.cdn}"):
                    value.__persisted__ = url
                    return url
            value.persist(context)
            if value.__persisted__ is not None:
                return value.__persisted__

        return value.to_data_url()


    @classmethod
    def _fetch_local_file(cls, path: Path) -> io.IOBase:
        """Fetch a local file from a valid path in the system."""
        return path.open("rb")


    @classmethod
    def _fetch_data_url_file(cls, data_url: str) -> io.IOBase:
        """Parse a data URL and return a file-like object."""
        res = urlopen(data_url)  # noqa: S310
        return io.BytesIO(res.read())


    @classmethod
    def _fetch_http_file(cls, url: str) -> io.IOBase:
        """Fetch a file from a HTTP/HTTPS URL."""
        res = requests.get(url, stream=True)
        res.raise_for_status()
        return io.BytesIO(res.content)


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
                info_arg=True,
                when_used="always",
            ),
        )
