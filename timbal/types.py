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
import boto3
import pathlib
import requests

from pydantic import Field as PydanticField
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
    
    # TODO - Validate some stuff like content-length before pulling files.
    @classmethod 
    def validate(cls, value: Any) -> "File":
        """Create a new Field instance validating a local path, an url, an s3 uri, a data url or a file-like object."""
        if isinstance(value, io.IOBase):
            return File(value)
        if not isinstance(value, str):
            raise ValueError("File must be a string or file-like object.")
        parsed_url = urlparse(value)
        # Check for valid local path.
        if parsed_url.scheme == "":
            path = pathlib.Path(value).expanduser().resolve()
            if path.exists() and path.is_file():
                return File(path.open("rb"))
            else:
                raise ValueError("Invalid file path.")
        # Data urls: data:[<mediatype>][;base64],<data>
        if parsed_url.scheme == "data":
            res = urlopen(value) # noqa: S310
            return File(io.BytesIO(res.read()))
        # HTTP/HTTPS URLs.
        if parsed_url.scheme == "http" or parsed_url.scheme == "https":
            return File(value, fetcher=lambda: cls._fetch_http_file(url=value))
        # S3 URIs.
        if parsed_url.scheme == "s3":
            return File(value, fetcher=lambda: cls._fetch_s3_file(bucket=parsed_url.netloc, key=parsed_url.path.lstrip("/")))
        # Else the file is invalid.
        raise ValueError("Invalid file source. Must be a local file path, data URL, HTTP URL, or S3 URI.")

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
        client = boto3.client("s3")
        res = client.get_object(
            Bucket=bucket,
            Key=key,
        )
        return io.BytesIO(res["Body"].read())

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        """Defines what this type should be in openapi.json."""
        # https://json-schema.org/understanding-json-schema/reference/string.html#uri-template
        field_schema.update(type="string", format="uri")
