"""File downloads must not bridge the host's private network into a sandbox."""

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from timbal.tools import RunPython
from timbal.tools._run_python_transfer import read_file
from timbal.types import File


def answers(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, 443),
        )
        for address in addresses
    ]


@pytest.fixture
async def dns(monkeypatch):
    resolver = AsyncMock(return_value=answers("93.184.215.14"))
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolver)
    return resolver


@pytest.fixture
def http(monkeypatch):
    state = SimpleNamespace(requests=[], options=[], handler=lambda _request: httpx.Response(200, content=b"file"))
    client_type = httpx.AsyncClient

    def handle(request):
        state.requests.append(request)
        return state.handler(request)

    def client(**kwargs):
        state.options.append(kwargs)
        return client_type(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return state


@pytest.mark.parametrize(
    "host",
    [
        "169.254.169.254",
        "169.254.170.2",
        "168.63.129.16",
        "127.0.0.1",
        "0.0.0.0",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "100.100.100.200",
        "192.0.0.8",
        "192.88.99.1",
        "224.0.0.1",
        "240.0.0.1",
        "[::1]",
        "[::]",
        "[fd00:ec2::254]",
        "[fe80::1]",
        "[::ffff:169.254.169.254]",
        "[64:ff9b::a9fe:a9fe]",
        "[2002:a9fe:a9fe::1]",
        "[2001::a9fe:a9fe]",
        "[ff02::1]",
    ],
)
async def test_nonpublic_literals_never_send_request(http, dns, host):
    with pytest.raises(ValueError, match="public Internet"):
        await read_file(File(f"http://{host}/metadata"), 1024)
    assert http.requests == []
    dns.assert_not_awaited()


@pytest.mark.parametrize("host", ["metadata.google.internal", "localhost", "2130706433", "0177.0.0.1", "0x7f000001"])
async def test_nonpublic_dns_and_alternate_ip_encodings(http, dns, host):
    dns.return_value = answers("127.0.0.1")
    # HTTPX rejects octal dotted addresses during URL parsing; other legacy
    # encodings are rejected once the system resolver canonicalizes them.
    with pytest.raises((ValueError, httpx.InvalidURL)):
        await read_file(File(f"http://{host}/file"), 1024)
    assert http.requests == []


async def test_mixed_public_private_dns_is_rejected(http, dns):
    dns.return_value = answers("93.184.215.14", "10.0.0.1")
    with pytest.raises(ValueError, match="public Internet"):
        await read_file(File("https://files.example/data"), 1024)
    assert http.requests == []


async def test_public_url_is_pinned_and_preserves_host_and_tls_name(http, dns, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("ALL_PROXY", "http://169.254.169.254:8080")
    # A DNS rebind would return a private address if the hostname were looked up
    # again by the transport. It receives only the vetted numeric IP instead.
    dns.side_effect = [answers("93.184.215.14"), answers("127.0.0.1")]
    assert await read_file(File("https://files.example:8443/data?signature=abc"), 1024) == b"file"
    dns.assert_awaited_once()
    request = http.requests[0]
    assert str(request.url) == "https://93.184.215.14:8443/data?signature=abc"
    assert request.headers["host"] == "files.example:8443"
    assert request.extensions["sni_hostname"] == "files.example"
    assert "authorization" not in request.headers
    assert all(options["trust_env"] is False and options["follow_redirects"] is False for options in http.options)


async def test_public_ipv6_literal_is_supported(http, dns):
    assert await read_file(File("https://[2606:4700:4700::1111]/file"), 1024) == b"file"
    assert http.requests[0].url.host == "2606:4700:4700::1111"
    assert http.requests[0].headers["host"] == "[2606:4700:4700::1111]"
    dns.assert_not_awaited()


async def test_http_transport_connects_to_pinned_ip_with_verified_tls(dns, monkeypatch):
    import ssl

    from httpcore._backends.auto import AutoBackend

    stream = SimpleNamespace(
        read=AsyncMock(return_value=b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nfile"),
        write=AsyncMock(),
        aclose=AsyncMock(),
        get_extra_info=lambda _name: None,
    )
    stream.start_tls = AsyncMock(return_value=stream)
    connect = AsyncMock(return_value=stream)
    monkeypatch.setattr(AutoBackend, "connect_tcp", connect)
    assert await read_file(File("https://files.example/file"), 1024) == b"file"
    connect.assert_awaited_once()
    assert connect.call_args.kwargs["host"] == "93.184.215.14"
    assert connect.call_args.kwargs["port"] == 443
    tls = stream.start_tls.call_args.kwargs
    assert tls["server_hostname"] == "files.example"
    assert tls["ssl_context"].check_hostname is True
    assert tls["ssl_context"].verify_mode == ssl.CERT_REQUIRED
    assert b"Host: files.example\r\n" in stream.write.call_args_list[0].args[0]
    dns.assert_awaited_once()


@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/secret",
        "http://internal.example/secret",
    ],
)
async def test_redirect_to_private_destination_is_blocked(http, dns, location):
    dns.side_effect = [answers("93.184.215.14"), answers("10.0.0.1")]
    http.handler = lambda _request: httpx.Response(302, headers={"location": location})
    with pytest.raises(ValueError, match="public Internet"):
        await read_file(File("https://public.example/file"), 1024)
    assert len(http.requests) == 1
    assert http.requests[0].url.host == "93.184.215.14"


async def test_same_host_redirect_revalidates_dns(http, dns):
    dns.side_effect = [answers("93.184.215.14"), answers("169.254.169.254")]
    http.handler = lambda _request: httpx.Response(302, headers={"location": "/next"})
    with pytest.raises(ValueError, match="public Internet"):
        await read_file(File("https://public.example/file"), 1024)
    assert len(http.requests) == 1


async def test_public_redirects_keep_original_authority_and_relative_paths(http, dns):
    dns.side_effect = [answers("93.184.215.14"), answers("93.184.215.14"), answers("1.1.1.1")]

    def redirect(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/next"})
        if request.url.path == "/next":
            return httpx.Response(307, headers={"location": "https://cdn.example/final"})
        return httpx.Response(200, content=b"downloaded")

    http.handler = redirect
    assert await read_file(File("https://files.example/start"), 1024) == b"downloaded"
    assert [request.headers["host"] for request in http.requests] == ["files.example", "files.example", "cdn.example"]
    assert http.requests[-1].url.host == "1.1.1.1"


async def test_redirect_loop_is_bounded(http, dns):
    http.handler = lambda _request: httpx.Response(302, headers={"location": "/again"})
    with pytest.raises(ValueError, match="five redirects"):
        await read_file(File("https://files.example/start"), 1024)
    assert len(http.requests) == 6
    assert dns.await_count == 6


@pytest.mark.parametrize(
    "location", ["file:///etc/passwd", "ftp://files.example/file", "http://user:secret@files.example/file"]
)
async def test_redirect_invalid_scheme_or_credentials_is_blocked(http, dns, location):
    http.handler = lambda _request: httpx.Response(302, headers={"location": location})
    with pytest.raises(ValueError, match="HTTP"):
        await read_file(File("https://files.example/start"), 1024)
    assert len(http.requests) == 1
    dns.assert_awaited_once()


async def test_private_input_fails_before_modal_creation(http, dns, monkeypatch):
    import sys

    create = AsyncMock()
    image = SimpleNamespace(debian_slim=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Image=image, Sandbox=SimpleNamespace(create=create)))
    with pytest.raises(ValueError, match="public Internet"):
        await RunPython().handler("1", files={"credentials.txt": "http://169.254.169.254/latest/meta-data/"})
    create.assert_not_called()
    assert http.requests == []
    dns.assert_not_awaited()
