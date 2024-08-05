"""
This module provides a File class that wraps various types of file sources, 
including local files, URLs, and S3 URIs, enabling uniform interaction through 
a file-like interface. It also provides a Field factory function for creating
Pydantic model fields with extended capabilities such as choice enforcement. 

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

4. Iterating over lines in an S3 file:
   >>> file_instance = File.validate('s3://mybucket/myfile.txt')
   >>> for line in file_instance:
   ...     print(line)

5. Opening a file-like object:
   >>> from io import BytesIO
   >>> file_instance = File.validate(BytesIO(b'Hello, World!'))
   >>> print(file_instance.read())

Note: Ensure appropriate error handling and resource management, especially when dealing with file operations.
"""


import io
import os
import boto3
import pathlib
import requests

from pydantic import (
    Field as PydanticField,
    GetJsonSchemaHandler,
)
from pydantic_core import CoreSchema
from typing import (
    Any, 
    List, 
    Union,
    Iterator,
    Dict,
    Optional,
    Callable,
)
from urllib.parse import urlparse
from urllib.request import urlopen


def Field(
    default: Any = ...,
    description: str = None,
    ge: float = None,
    le: float = None,
    min_length: int = None,
    max_length: int = None,
    regex: str = None,
    choices: List[Union[str, int]] = None,
) -> Any:
    """
    Wrapper of pydantic.Field. Doesn't require a default value to be the first argument.
    Parameters are kept the same as in Replicate's Cog to ensure compatibility.

    If default is not provided, the field is required.
    If default is explicitly set to None, then the field is optional.
    """
    field_info = {
        "description": description,
        "ge": ge,
        "le": le,
        "min_length": min_length,
        "max_length": max_length,
        "pattern": regex,
    }
    # Choices is not implemented in pydantic Field. This will be added in json_schema_extra.
    if choices is not None:
        field_info["choices"] = choices
    return PydanticField(default, **field_info)


class File(io.IOBase):
    __slots__ = ("__source__", "__fileobj__", "__fetcher__")
    __max_file_size__: Optional[int] = int(os.getenv("TIMBAL_MAX_FILE_SIZE")) if os.getenv("TIMBAL_MAX_FILE_SIZE") else None

    def __init__(
        self, 
        source: Any,
        fetcher: Optional[Callable[[], io.IOBase]] = None,
    ) -> None:
        object.__setattr__(self, "__source__", source)
        object.__setattr__(self, "__fetcher__", fetcher)
        if isinstance(source, io.IOBase):
            object.__setattr__(self, "__fileobj__", source)
        else:
            object.__setattr__(self, "__fileobj__", None)

    # Proxy getattr/setattr/delattr through to the wrapped file object.
    def __getattr__(self, name: str) -> Any:
        if name in ("__source__", "__fileobj__", "__fetcher__"):
            raise AttributeError(name)
        else:
            return getattr(self.__wrapped__, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__wrapped__, name, value)

    def __delattr__(self, name: str) -> None:
        if hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.__wrapped__, name)

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.__wrapped__)

    @property
    def __wrapped__(self) -> Any:
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
    def __get_validators__(cls) -> Iterator[Any]:
        """Pydantic calls this method to retrieve the validators when validating data against this class."""
        yield cls.validate
    
    @classmethod 
    def validate(cls, value: Any) -> "File":
        """Create a new Field instance validating a local path, an url, an s3 uri, a data url or a file-like object."""
        if isinstance(value, (bytes, bytearray)):
            return File(io.BytesIO(value))
        if isinstance(value, io.IOBase):
            return File(value)
        if not isinstance(value, str):
            raise ValueError("File must be a string or file-like object.")
        parsed_url = urlparse(value)
        # Check for valid local path.
        if parsed_url.scheme == "":
            path = pathlib.Path(value).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise ValueError("Invalid file path.")
            if cls.__max_file_size__ is not None:
                file_size = path.stat().st_size # bytes
                print(type(file_size), type(cls.__max_file_size__))
                if file_size > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
            # ? File type restrictions
            return File(value, fetcher=lambda: cls._fetch_local_file(path=value))
        # Data urls: data:[<mediatype>][;base64],<data>.
        if parsed_url.scheme == "data":
            if cls.__max_file_size__ is not None:
                comma_idx = value.find(",")
                # Redundant check.
                if comma_idx == -1:
                    raise ValueError("Invalid data URL.")
                encoded = value[comma_idx + 1:]
                # ? We could count the padding if we want to be more precise.
                estimated_size = len(encoded) * 3 // 4
                if estimated_size > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
            # ? File type restrictions
            return File(value, fetcher=lambda: cls._fetch_data_url_file(data_url=value))
        # HTTP/HTTPS URLs.
        if parsed_url.scheme == "http" or parsed_url.scheme == "https":
            response = requests.head(value)
            if cls.__max_file_size__ is not None:
                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
            # ? File type restrictions
            return File(value, fetcher=lambda: cls._fetch_http_file(url=value))
        # S3 URIs.
        if parsed_url.scheme == "s3":
            bucket = parsed_url.netloc
            key = parsed_url.path.lstrip("/")
            s3_client = boto3.client("s3")
            response = s3_client.head_object(Bucket=bucket, Key=key)
            if cls.__max_file_size__ is not None:
                content_length = int(response.get("ContentLength", 0))
                if content_length > cls.__max_file_size__:
                    raise ValueError(f"File size exceeds the maximum allowed size of {cls.__max_file_size__} bytes.")
            # ? File type restrictions
            return File(value, fetcher=lambda: cls._fetch_s3_file(bucket=bucket, key=key))
        # Else the file is invalid.
        raise ValueError("Invalid file source. Must be a local file path, data URL, HTTP URL, or S3 URI.")

    @classmethod
    def _fetch_local_file(cls, path: pathlib.Path) -> io.IOBase:
        """Fetch a local file from a valid path in the system."""
        return path.open("rb")

    @classmethod
    def _fetch_data_url_file(cls, data_url: str) -> io.IOBase:
        """Parse a data URL and return a file-like object."""
        res = urlopen(data_url) # noqa: S310
        return io.BytesIO(res.read())

    @classmethod
    def _fetch_http_file(cls, url: str) -> io.IOBase:
        """Fetch a file from a HTTP/HTTPS URL."""
        res = requests.get(url, stream=True)
        res.raise_for_status()
        return io.BytesIO(res.content)

    @classmethod
    def _fetch_s3_file(cls, bucket: str, key: str) -> io.IOBase:
        """Fetch a file from an S3 bucket."""
        # Pull configuration directly from the environment.
        s3_client = boto3.client("s3")
        res = s3_client.get_object(
            Bucket=bucket,
            Key=key,
        )
        return io.BytesIO(res["Body"].read())

    @classmethod
    def __get_pydantic_json_schema__(lcs, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler) -> Dict[str, Any]:
        """Defines what this type should be in openapi.json."""
        # https://docs.pydantic.dev/2.8/errors/usage_errors/#custom-json-schema
        json_schema = {
            "type": "string",
            "format": "uri",
            "description": "A file reference which can be a local path, a URL, an S3 URI, or a data URL.",
        }
        return json_schema
