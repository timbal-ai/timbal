"""Host-side file validation and bounded input reads for RunPython."""

import asyncio
import ipaddress
import socket
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from ..types import File

_BLOCKED_IPV4 = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "192.0.0.0/24",
        "192.88.99.0/24",
        "168.63.129.16/32",
    )
)
_GLOBAL_IPV6 = ipaddress.ip_network("2000::/3")
_BLOCKED_IPV6 = tuple(ipaddress.ip_network(cidr) for cidr in ("2001::/23", "2002::/16"))


def _public_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    address = ipaddress.ip_address(value)
    allowed = address.is_global and not (address.is_multicast or address.is_reserved)
    if address.version == 4:
        # Include special-use ranges whose classification differs across supported
        # Python versions, and Azure's platform virtual IP (not ordinary Internet).
        allowed = allowed and not any(address in network for network in _BLOCKED_IPV4)
    else:
        # Only native global unicast: reject mapped IPv4, NAT64, Teredo, and 6to4
        # addresses, which can otherwise embed/translate to an internal IPv4 host.
        allowed = allowed and address in _GLOBAL_IPV6 and not any(address in network for network in _BLOCKED_IPV6)
    if not allowed:
        raise ValueError("Input file URLs must resolve only to public Internet addresses.")
    return address


async def _download_target(url: httpx.URL) -> httpx.URL:
    """Validate all DNS answers and pin the request to one checked numeric IP."""
    if url.scheme not in ("http", "https") or not url.host or url.userinfo or "%" in url.host:
        raise ValueError("Input file URLs must use HTTP(S), without credentials or scoped addresses.")
    host = url.raw_host.decode("ascii")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        answers = await asyncio.get_running_loop().getaddrinfo(
            host,
            url.port or (443 if url.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        addresses = [_public_address(answer[4][0]) for answer in answers]
        if not addresses:
            raise ValueError("Input file URL did not resolve to a public Internet address.") from None
        # Prefer IPv4 when both families exist; every answer has been validated.
        address = min(addresses, key=lambda item: item.version)
    return url.copy_with(host=str(_public_address(str(address))))


async def _download_file(source: str, limit: int) -> bytes:
    url = httpx.URL(source)
    for hop in range(6):
        target = await _download_target(url)
        # A new client per hop avoids pooling two hostnames sharing an IP under
        # one TLS connection. Environment proxies must not re-resolve our URL or
        # redirect traffic into the host's private network.
        async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
            async with client.stream(
                "GET",
                target,
                headers={"Host": url.netloc.decode("ascii")},
                extensions={"sni_hostname": url.raw_host.decode("ascii")},
            ) as response:
                if response.is_redirect and "location" in response.headers:
                    if hop == 5:
                        raise ValueError("Input file URL exceeds five redirects.")
                    # Resolve relative Location against the original hostname,
                    # then validate and pin the next hop before sending anything.
                    url = url.join(response.headers["location"])
                    continue
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    if len(data) + len(chunk) > limit:
                        raise ValueError("Inputs exceed max_file_bytes.")
                    data.extend(chunk)
                return bytes(data)
    raise AssertionError("Redirect limit must end the download.")


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
        return await _download_file(str(file), limit)

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
