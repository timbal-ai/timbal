"""Host-side file validation and bounded input reads for RunPython."""

import asyncio
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from ..types import File


def validate_path(name: str) -> str:
    """Require portable relative paths within the input/output directory."""
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
        or "\\" in name
        or ":" in name
        or any(ord(char) < 32 for char in name)
        or len(name.encode()) > 240
    ):
        raise ValueError("File names must be relative paths, without '..', and at most 240 UTF-8 bytes.")
    return name


def validate_file(value: object) -> str | File:
    """Local files require an explicit File object from application code."""
    if isinstance(value, File):
        return value
    if not isinstance(value, str) or urlparse(value).scheme not in ("https", "http", "data"):
        raise ValueError("Supply a file URL or data URL. Application code can pass File objects for local files.")
    # Keep URLs as strings through Timbal's input tracing. Eager File conversion
    # could persist/fetch them before the handler's bounded streaming download.
    return value


async def read_file(file: File, limit: int) -> bytes:
    """Bound the read itself, including HTTP responses without Content-Length."""
    if file.__source_scheme__ == "url":
        data = bytearray()
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            async with client.stream("GET", str(file)) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    if len(data) + len(chunk) > limit:
                        raise ValueError("Inputs exceed max_file_bytes.")
                    data.extend(chunk)
        return bytes(data)

    def read() -> bytes:
        if file.__source_scheme__ == "local_path":
            with Path(str(file)).open("rb") as stream:
                return stream.read(limit + 1)
        if file.__source_scheme__ == "data_url" and len(str(file)) > 4 * limit + 1024:
            raise ValueError("Input data URL exceeds max_file_bytes.")
        position = file.tell()
        try:
            file.seek(0)
            return file.read(limit + 1)
        finally:
            file.seek(position)

    data = await asyncio.to_thread(read)
    if len(data) > limit:
        raise ValueError("Inputs exceed max_file_bytes.")
    return data
