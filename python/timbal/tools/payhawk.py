"""Payhawk through its official hosted MCP server (``https://mcp.payhawk.com/mcp``).

Payhawk publishes this server for AI clients. Its tool catalogue (expenses, documents and receipts,
cards, fund accounts, approvals, suppliers, purchase orders, accounting exports, budgets, policies,
users and teams, comments, insights) is whatever the server lists to the signed-in user, so Timbal
exposes exactly the tools Claude gets, as ``payhawk__<tool>``.

    from timbal import Agent
    from timbal.tools import PayhawkMCP

    agent = Agent(name="finance", model="...", tools=[PayhawkMCP(integration="<org_integration_id>")])

Credentials resolve: explicit ``token`` → ``Integration("payhawk")`` → ``PAYHAWK_MCP_TOKEN``. The server
uses per-user OAuth (authorization code + PKCE, dynamic client registration), so the token is the
signed-in user's and every call runs with that user's Payhawk permissions. Getting it is the platform
integration's job (browser consent, refresh); Payhawk additionally allowlists the OAuth clients that
may register, so the Timbal platform's OAuth client has to be approved by Payhawk before an
integration can be connected. Payhawk's server never moves money, issues cards or changes access.

Writes: the server returns a preview and waits for confirmation; on top of that, every tool the server
does not mark ``readOnlyHint`` goes through Timbal's approval gate (``approval=`` to change it). The
catalogue is large: ``defer_loading=True`` keeps the schemas out of the prompt until the model searches.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from ..platform.integrations import Integration
from ._mcp_oauth import IntegrationBearerAuth, OfficialMCPServer

PAYHAWK_MCP_URL = "https://mcp.payhawk.com/mcp"
_TOKEN_ENV = ("PAYHAWK_MCP_TOKEN",)


class PayhawkAuth(IntegrationBearerAuth):
    """Bearer auth for Payhawk's MCP server: explicit token → integration → ``PAYHAWK_MCP_TOKEN``."""

    def __init__(self, integration: str | Integration | None = None, *, token=None) -> None:
        super().__init__("payhawk", integration, token=token, env_vars=_TOKEN_ENV)


class PayhawkMCP(OfficialMCPServer):
    """Payhawk's official MCP server as a Timbal tool source (see the module docstring)."""

    PROVIDER: ClassVar[str] = "payhawk"
    DEFAULT_NAME: ClassVar[str] = "payhawk"
    DEFAULT_URL: ClassVar[str] = PAYHAWK_MCP_URL
    TOKEN_ENV_VARS: ClassVar[tuple[str, ...]] = _TOKEN_ENV

    integration: Annotated[str, Integration("payhawk")] | None = None
