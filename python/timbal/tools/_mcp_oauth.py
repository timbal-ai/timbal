"""Shared plumbing for integrations that wrap a vendor's *official* hosted MCP server.

The vendor owns the tool catalogue (names, schemas, read/write split, confirmation previews); Timbal
only supplies the transport (:class:`~timbal.core.mcp.MCPServer`) and the credential. These servers
use per-user OAuth, so there is no API key to configure: the bearer token comes from a platform
``Integration`` (the platform runs the browser consent and refreshes the token server-side), an
explicit ``token``, or an environment variable, in that order.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import httpx
from pydantic import SecretStr, model_validator

from ..core.mcp import MCPServer
from ..platform.integrations import Integration

_TOKEN_KEYS = ("access_token", "token", "bearer_auth")


class IntegrationBearerAuth(httpx.Auth):
    """``httpx.Auth`` that sends ``Authorization: Bearer <token>`` from an OAuth integration.

    ``MCPServer`` sends requests from a context-free transport task, so it awaits :meth:`prepare` in
    the caller's run context before each request. That is where the ``Integration`` is resolved, under
    the caller's identity (every resolve is a platform round trip that serves the stored token and
    refreshes it when it expires). The auth flow itself only applies the prepared token.

    One ``MCPServer`` session is shared by its callers and uses the most recently prepared token:
    don't share an instance across users.
    """

    def __init__(
        self,
        provider: str,
        integration: str | Integration | None = None,
        *,
        token: str | SecretStr | None = None,
        env_vars: Sequence[str] = (),
    ) -> None:
        if isinstance(integration, str):
            integration = Integration(provider, integration)
        self.provider = provider
        self._integration = integration
        self._explicit = token.get_secret_value() if isinstance(token, SecretStr) else token
        self._env_vars = tuple(env_vars)
        self._token: str | None = None

    async def _resolve(self) -> str:
        if self._explicit:
            return self._explicit
        if isinstance(self._integration, Integration):
            credentials = await self._integration.resolve()
            for key in _TOKEN_KEYS:
                if credentials.get(key):
                    return str(credentials[key])
            raise ValueError(
                f"{self.provider} integration credentials must include an OAuth access token "
                f"({', '.join(_TOKEN_KEYS)}). Reconnect the integration from the platform."
            )
        for var in self._env_vars:
            if os.getenv(var):
                return os.environ[var]
        hint = f"set {' or '.join(self._env_vars)}, " if self._env_vars else ""
        raise ValueError(f"{self.provider} MCP credentials not found: {hint}pass token=, or configure an integration.")

    async def prepare(self) -> None:
        """Resolve the token in the caller's run context. ``MCPServer`` awaits this before every request."""
        self._token = await self._resolve()

    async def _ensure_prepared(self) -> str:
        if self._token is None:
            from ..state import get_run_context

            if isinstance(self._integration, Integration) and get_run_context() is None:
                raise RuntimeError(
                    f"{type(self).__name__} needs its Integration resolved in the caller's run context: "
                    "use it through MCPServer(auth=...) or await auth.prepare() first."
                )
            await self.prepare()
        return self._token  # type: ignore[return-value]

    def sync_auth_flow(self, _request: httpx.Request):
        raise RuntimeError(f"{type(self).__name__} only supports httpx.AsyncClient.")

    async def async_auth_flow(self, request: httpx.Request):
        token = await self._ensure_prepared()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def gate_unless_read_only(tool: Any) -> bool:
    """Approval policy: require approval for every tool the server does not declare read-only.

    Fail-closed for finance data: a tool with no annotations is treated as a write. Pass
    ``approval="destructive"`` (only annotated destructive tools), ``"all"`` or ``None`` to change it.
    """
    annotations = getattr(tool, "annotations", None)
    return getattr(annotations, "readOnlyHint", None) is not True


class OfficialMCPServer(MCPServer):
    """An :class:`MCPServer` preset for a vendor's official hosted MCP server.

    Subclasses set ``PROVIDER``, ``DEFAULT_NAME``, ``DEFAULT_URL`` and ``TOKEN_ENV_VARS``, and add an
    ``integration`` field annotated with ``Integration(<provider>)``. The tools are whatever the server
    lists, exposed as ``<name>__<tool>``; writes go through Timbal's approval gate by default (see
    :func:`gate_unless_read_only`) on top of whatever confirmation the vendor's server asks for.
    """

    PROVIDER: ClassVar[str] = ""
    DEFAULT_NAME: ClassVar[str] = ""
    DEFAULT_URL: ClassVar[str] = ""
    TOKEN_ENV_VARS: ClassVar[tuple[str, ...]] = ()

    transport: Literal["stdio", "http"] = "http"
    approval: Any = gate_unless_read_only
    token: SecretStr | None = None
    integration: Any = None

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data.setdefault("name", cls.DEFAULT_NAME)
            data.setdefault("url", cls.DEFAULT_URL)
        return data

    @model_validator(mode="after")
    def _attach_auth(self) -> OfficialMCPServer:
        if self.transport != "http":
            raise ValueError(f"{self.PROVIDER} MCP is a hosted server; only the http transport is supported")
        if self.auth is None:
            self.auth = IntegrationBearerAuth(
                self.PROVIDER, self.integration, token=self.token, env_vars=self.TOKEN_ENV_VARS
            )
        return self
