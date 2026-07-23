import asyncio
import socket

import httpx
import pytest
from pydantic import ValidationError
from timbal.core.mcp import MCPServer
from timbal.core.mcp_oauth import (
    FileTokenStorage,
    InMemoryTokenStorage,
    OAuth,
    _CallbackServer,
)

try:
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
except ImportError:  # pragma: no cover
    OAuthClientProvider = None

pytestmark = pytest.mark.skipif(OAuthClientProvider is None, reason="mcp package not installed")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _sample_tokens() -> "OAuthToken":
    return OAuthToken(access_token="abc123", token_type="Bearer", expires_in=3600, refresh_token="ref456")


def _sample_client_info() -> "OAuthClientInformationFull":
    return OAuthClientInformationFull(
        client_id="client-1",
        redirect_uris=["http://localhost:51820/callback"],
    )


class TestInMemoryTokenStorage:
    @pytest.mark.asyncio
    async def test_roundtrip(self):
        storage = InMemoryTokenStorage()
        assert await storage.get_tokens() is None
        assert await storage.get_client_info() is None

        await storage.set_tokens(_sample_tokens())
        await storage.set_client_info(_sample_client_info())

        assert (await storage.get_tokens()).access_token == "abc123"
        assert (await storage.get_client_info()).client_id == "client-1"


class TestFileTokenStorage:
    @pytest.mark.asyncio
    async def test_roundtrip_and_persistence(self, tmp_path):
        url = "https://mcp.example.com/mcp"
        storage = FileTokenStorage(url, storage_dir=tmp_path)
        assert await storage.get_tokens() is None

        await storage.set_tokens(_sample_tokens())
        await storage.set_client_info(_sample_client_info())

        # A fresh instance pointing at the same URL sees the same credentials.
        storage2 = FileTokenStorage(url, storage_dir=tmp_path)
        assert (await storage2.get_tokens()).refresh_token == "ref456"
        assert (await storage2.get_client_info()).client_id == "client-1"

    @pytest.mark.asyncio
    async def test_isolated_per_server_url(self, tmp_path):
        a = FileTokenStorage("https://a.example.com/mcp", storage_dir=tmp_path)
        b = FileTokenStorage("https://b.example.com/mcp", storage_dir=tmp_path)
        await a.set_tokens(_sample_tokens())
        assert await b.get_tokens() is None

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path):
        storage = FileTokenStorage("https://mcp.example.com/mcp", storage_dir=tmp_path)
        await storage.set_tokens(_sample_tokens())
        storage.clear()
        assert await storage.get_tokens() is None


class TestCallbackServer:
    @pytest.mark.asyncio
    async def test_captures_code_and_state(self):
        server = _CallbackServer("localhost", 0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        async def hit_callback():
            async with httpx.AsyncClient() as client:
                return await client.get(f"http://localhost:{port}/callback?code=the-code&state=the-state")

        response, (code, state) = await asyncio.gather(hit_callback(), server.wait_for_code(timeout=5))
        assert response.status_code == 200
        assert b"Authentication complete" in response.content
        assert code == "the-code"
        assert state == "the-state"

    @pytest.mark.asyncio
    async def test_error_callback_raises(self):
        server = _CallbackServer("localhost", 0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        async def hit_callback():
            async with httpx.AsyncClient() as client:
                return await client.get(
                    f"http://localhost:{port}/callback?error=access_denied&error_description=nope"
                )

        hit_task = asyncio.create_task(hit_callback())
        with pytest.raises(RuntimeError, match="access_denied"):
            await server.wait_for_code(timeout=5)
        response = await hit_task
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_path_is_404_and_flow_still_completes(self):
        server = _CallbackServer("localhost", 0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        async def hit():
            async with httpx.AsyncClient() as client:
                r404 = await client.get(f"http://localhost:{port}/favicon.ico")
                r200 = await client.get(f"http://localhost:{port}/callback?code=c&state=s")
                return r404, r200

        (r404, r200), (code, state) = await asyncio.gather(hit(), server.wait_for_code(timeout=5))
        assert r404.status_code == 404
        assert r200.status_code == 200
        assert (code, state) == ("c", "s")

    @pytest.mark.asyncio
    async def test_timeout(self):
        server = _CallbackServer("localhost", 0)
        await server.start()
        with pytest.raises(TimeoutError):
            await server.wait_for_code(timeout=0.05)


class TestOAuthConfig:
    def test_build_provider(self, tmp_path):
        oauth = OAuth(scopes="read write", callback_port=51999, storage_dir=tmp_path)
        provider = oauth.build_provider("https://mcp.example.com/mcp")
        assert isinstance(provider, OAuthClientProvider)

        metadata = provider.context.client_metadata
        assert str(metadata.redirect_uris[0]) == "http://localhost:51999/callback"
        assert metadata.scope == "read write"
        assert metadata.token_endpoint_auth_method == "none"
        assert isinstance(provider.context.storage, FileTokenStorage)

    def test_custom_storage_is_used(self):
        storage = InMemoryTokenStorage()
        provider = OAuth(storage=storage).build_provider("https://mcp.example.com/mcp")
        assert provider.context.storage is storage

    @pytest.mark.asyncio
    async def test_redirect_and_callback_handlers_complete_flow(self, tmp_path, monkeypatch):
        """Simulate the browser: redirect handler opens the portal, then the portal
        redirects the user back to the local callback URL with the auth code."""
        opened_urls: list[str] = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url) or True)

        port = _free_port()
        oauth = OAuth(callback_port=port, storage_dir=tmp_path, flow_timeout=5)
        provider = oauth.build_provider("https://mcp.example.com/mcp")

        # SDK calls redirect_handler first (opens browser)...
        await provider.context.redirect_handler("https://auth.example.com/authorize?state=xyz")
        assert opened_urls == ["https://auth.example.com/authorize?state=xyz"]

        # ...then awaits callback_handler while the user logs in.
        async def simulate_browser_redirect():
            async with httpx.AsyncClient() as client:
                return await client.get(f"http://localhost:{port}/callback?code=ok&state=xyz")

        response, (code, state) = await asyncio.gather(
            simulate_browser_redirect(),
            provider.context.callback_handler(),
        )
        assert response.status_code == 200
        assert (code, state) == ("ok", "xyz")


class TestMCPServerAuthValidation:
    def test_auth_string_coerces_to_oauth(self):
        server = MCPServer(transport="http", url="https://mcp.example.com/mcp", auth="oauth")
        assert isinstance(server.auth, OAuth)

    def test_auth_instance_preserved(self):
        oauth = OAuth(scopes="read")
        server = MCPServer(transport="http", url="https://mcp.example.com/mcp", auth=oauth)
        assert server.auth is oauth

    def test_auth_rejected_for_stdio(self):
        with pytest.raises(ValidationError, match="only supported for http"):
            MCPServer(transport="stdio", command="echo", auth="oauth")

    def test_no_auth_by_default(self):
        server = MCPServer(transport="http", url="https://mcp.example.com/mcp")
        assert server.auth is None
