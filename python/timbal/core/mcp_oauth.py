"""OAuth 2.0 authorization for MCP servers.

Implements the client side of the MCP authorization spec on top of the
official SDK's ``OAuthClientProvider`` (authorization code flow with PKCE,
metadata discovery, dynamic client registration, automatic token refresh).

This module supplies the three pieces the SDK leaves to the application:

- **Browser redirect**: when the server responds 401, the user's default
  browser is opened at the authorization portal so they can log in.
- **Callback capture**: a one-shot local HTTP server listens on
  ``http://localhost:{port}/callback`` and captures the authorization code
  the portal redirects back with.
- **Token persistence**: ``FileTokenStorage`` keeps tokens and the registered
  client info under ``~/.timbal/mcp_oauth/`` (keyed by server URL) so users
  log in once, not on every run. ``InMemoryTokenStorage`` is available for
  ephemeral sessions.

Usage:

    from timbal.core import MCPServer
    from timbal.core.mcp_oauth import OAuth

    server = MCPServer(
        transport="http",
        url="https://mcp.example.com/mcp",
        auth=OAuth(),  # or auth="oauth" for all defaults
    )
"""

import asyncio
import hashlib
import json
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import structlog
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger("timbal.core.mcp_oauth")

DEFAULT_STORAGE_DIR = Path.home() / ".timbal" / "mcp_oauth"

_CALLBACK_SUCCESS_HTML = b"""\
<!DOCTYPE html>
<html>
<head><title>Authentication complete</title></head>
<body style="font-family: sans-serif; text-align: center; padding-top: 4rem;">
<h1>Authentication complete</h1>
<p>You can close this window and return to your application.</p>
</body>
</html>
"""

_CALLBACK_ERROR_HTML = b"""\
<!DOCTYPE html>
<html>
<head><title>Authentication failed</title></head>
<body style="font-family: sans-serif; text-align: center; padding-top: 4rem;">
<h1>Authentication failed</h1>
<p>You can close this window and retry from your application.</p>
</body>
</html>
"""


class InMemoryTokenStorage:
    """Ephemeral token storage. Tokens are lost when the process exits."""

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


class FileTokenStorage:
    """Persist tokens and client registration to disk, keyed by server URL.

    Each server gets its own JSON file under ``storage_dir`` named by a hash
    of the server URL, created with owner-only permissions (0600).
    """

    def __init__(self, server_url: str, storage_dir: Path | None = None) -> None:
        self.server_url = server_url
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        digest = hashlib.sha256(server_url.encode()).hexdigest()[:16]
        host = urlparse(server_url).hostname or "server"
        self.path = self.storage_dir / f"{host}_{digest}.json"

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    async def get_tokens(self) -> OAuthToken | None:
        tokens = self._read().get("tokens")
        return OAuthToken.model_validate(tokens) if tokens else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        client_info = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(client_info) if client_info else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)

    def clear(self) -> None:
        """Delete stored credentials for this server (forces re-authentication)."""
        self.path.unlink(missing_ok=True)


class _CallbackServer:
    """One-shot local HTTP server that captures the OAuth authorization callback.

    Listens on ``http://{host}:{port}{path}`` for a single redirect carrying
    ``code`` and ``state`` query parameters, serves a small confirmation page,
    and hands the values back to the OAuth flow.
    """

    def __init__(self, host: str, port: int, path: str = "/callback") -> None:
        self.host = host
        self.port = port
        self.path = path
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[tuple[str, str | None]] | None = None

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    async def start(self) -> None:
        if self._server is not None:
            return
        self._result = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle_connection, self.host, self.port)
        logger.debug("OAuth callback server listening", redirect_uri=self.redirect_uri)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            # Drain headers so the browser doesn't see a reset connection.
            while await reader.readline() not in (b"\r\n", b"\n", b""):
                pass

            parts = request_line.decode("latin-1").split(" ")
            target = parts[1] if len(parts) >= 2 else "/"
            parsed = urlparse(target)

            if parsed.path != self.path:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return

            params = parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            error = params.get("error", [None])[0]

            if error or not code:
                body = _CALLBACK_ERROR_HTML
                status = b"HTTP/1.1 400 Bad Request"
                if self._result is not None and not self._result.done():
                    description = params.get("error_description", [None])[0]
                    message = f"OAuth authorization failed: {error or 'no authorization code returned'}"
                    if description:
                        message = f"{message} ({description})"
                    self._result.set_exception(RuntimeError(message))
            else:
                body = _CALLBACK_SUCCESS_HTML
                status = b"HTTP/1.1 200 OK"
                if self._result is not None and not self._result.done():
                    self._result.set_result((code, state))

            writer.write(
                status
                + b"\r\nContent-Type: text/html; charset=utf-8"
                + f"\r\nContent-Length: {len(body)}".encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()

    async def wait_for_code(self, timeout: float) -> tuple[str, str | None]:
        assert self._result is not None, "start() must be called before wait_for_code()"
        try:
            return await asyncio.wait_for(asyncio.shield(self._result), timeout)
        except TimeoutError:
            raise TimeoutError(f"Timed out after {timeout:.0f}s waiting for the OAuth callback. ") from None
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._result = None


class OAuth(BaseModel):
    """OAuth 2.0 configuration for an HTTP :class:`~timbal.core.mcp.MCPServer`.

    When the MCP server responds 401, the flow kicks in automatically: the
    user's browser opens at the server's authorization portal, they log in,
    and the resulting tokens are stored (and refreshed) transparently. All
    fields have sensible defaults, so ``OAuth()`` works out of the box.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client_name: str = "Timbal Agent"
    """Client name sent during dynamic client registration (shown on consent screens)."""

    scopes: str | None = None
    """Space-separated OAuth scopes to request. If None, the server's advertised scopes are used."""

    callback_host: str = "localhost"
    """Host the local callback server binds to. Must match what the browser can reach."""

    callback_port: int = 51820
    """Port for the local callback server. The redirect URI becomes http://{host}:{port}/callback."""

    callback_path: str = "/callback"
    """Path component of the redirect URI."""

    flow_timeout: float = 300.0
    """Seconds to wait for the user to complete the login in the browser."""

    client_metadata_url: str | None = None
    """Optional URL-based client ID (CIMD). Skips dynamic registration on servers that support it."""

    storage: Any = None
    """Custom TokenStorage. Defaults to FileTokenStorage under ~/.timbal/mcp_oauth/."""

    storage_dir: Path | None = None
    """Directory for the default FileTokenStorage. Ignored when a custom storage is given."""

    open_browser: bool = True
    """Open the system browser automatically. The URL is always logged for headless environments."""

    def build_provider(self, server_url: str) -> OAuthClientProvider:
        """Build the httpx auth provider that drives the OAuth flow for ``server_url``."""
        callback_server = _CallbackServer(self.callback_host, self.callback_port, self.callback_path)
        storage = self.storage or FileTokenStorage(server_url, storage_dir=self.storage_dir)

        client_metadata = OAuthClientMetadata(
            client_name=self.client_name,
            redirect_uris=[callback_server.redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # public client with PKCE
            scope=self.scopes,
        )

        async def redirect_handler(authorization_url: str) -> None:
            # Start listening *before* opening the browser so the redirect
            # can't race the server startup.
            await callback_server.start()
            logger.info(
                "MCP server requires authentication. Complete the login in your browser.",
                url=authorization_url,
            )
            if self.open_browser:
                opened = webbrowser.open(authorization_url)
                if not opened:
                    logger.warning(
                        "Could not open a browser automatically. Open the authorization URL manually.",
                        url=authorization_url,
                    )

        async def callback_handler() -> tuple[str, str | None]:
            return await callback_server.wait_for_code(self.flow_timeout)

        return OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=self.flow_timeout,
            client_metadata_url=self.client_metadata_url,
        )
