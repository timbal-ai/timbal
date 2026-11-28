"""Agicap through its official hosted MCP server (``https://mcp.agicap.com/mcp``).

Agicap publishes this server for AI clients: ask questions about cash position by entity and currency,
overdue invoices, KPIs, debt maturities and variance, and act (import transactions, what-if scenarios,
forecasts, categorisation rules). The tool catalogue is whatever the server lists to the signed-in user,
so Timbal exposes exactly the tools Claude gets, as ``agicap__<tool>``.

    from timbal import Agent
    from timbal.tools import AgicapMCP

    agent = Agent(name="treasury", model="...", tools=[AgicapMCP(integration="<org_integration_id>")])

Credentials resolve: explicit ``token`` → ``Integration("agicap")`` → ``AGICAP_MCP_TOKEN``. The server
uses OAuth through the user's Agicap sign-in (scopes ``mcp:read mcp:write agicap:app offline_access``):
the token is the signed-in user's, and an Agicap admin decides whether the connection is read-only or
read+write. Getting the token is the platform integration's job (browser consent, refresh); request only
``mcp:read`` if the agent must not write.

Writes: Agicap asks for human confirmation on writes; on top of that, every tool the server does not
mark ``readOnlyHint`` goes through Timbal's approval gate (``approval=`` to change it).
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from ..platform.integrations import Integration
from ._mcp_oauth import IntegrationBearerAuth, OfficialMCPServer

AGICAP_MCP_URL = "https://mcp.agicap.com/mcp"
_TOKEN_ENV = ("AGICAP_MCP_TOKEN",)


class AgicapAuth(IntegrationBearerAuth):
    """Bearer auth for Agicap's MCP server: explicit token → integration → ``AGICAP_MCP_TOKEN``."""

    def __init__(self, integration: str | Integration | None = None, *, token=None) -> None:
        super().__init__("agicap", integration, token=token, env_vars=_TOKEN_ENV)


class AgicapMCP(OfficialMCPServer):
    """Agicap's official MCP server as a Timbal tool source (see the module docstring)."""

    PROVIDER: ClassVar[str] = "agicap"
    DEFAULT_NAME: ClassVar[str] = "agicap"
    DEFAULT_URL: ClassVar[str] = AGICAP_MCP_URL
    TOKEN_ENV_VARS: ClassVar[tuple[str, ...]] = _TOKEN_ENV

    integration: Annotated[str, Integration("agicap")] | None = None
