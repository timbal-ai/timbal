"""Zendesk Support API tools for tickets, users, groups, and comments."""

import base64
import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://{subdomain}.zendesk.com/api/v2"


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Resolve Zendesk subdomain and auth from integration or env vars.

    Returns (base_url, auth_header_value) where auth_header_value is
    'Basic {base64(email/token:api_token)}'.
    """
    subdomain: str | None = None
    email: str | None = None
    api_token: str | None = None

    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        subdomain = credentials.get("subdomain")
        email = credentials.get("email")
        api_token = credentials.get("api_token")

    if subdomain is None:
        subdomain = tool.subdomain or os.getenv("ZENDESK_SUBDOMAIN")
    if email is None:
        email = (tool.email.get_secret_value() if tool.email else None) or os.getenv("ZENDESK_EMAIL")
    if api_token is None:
        api_token = (tool.api_token.get_secret_value() if tool.api_token else None) or os.getenv("ZENDESK_API_TOKEN")

    if not subdomain or not email or not api_token:
        raise ValueError(
            "Zendesk credentials not found. Set ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, and ZENDESK_API_TOKEN "
            "environment variables, pass them in config, or configure an integration."
        )

    auth_str = f"{email}/token:{api_token}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()
    return _BASE_URL.format(subdomain=subdomain), f"Basic {auth_b64}"


# --- Ticket ---


class ZendeskCreateTicket(Tool):
    name: str = "zendesk_create_ticket"
    description: str | None = (
        "Create a new Zendesk support ticket. Requires subject and comment body."
    )
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_ticket(
            subject: str = Field(..., description="Ticket subject"),
            comment_body: str = Field(..., description="First comment / description"),
            requester_email: str | None = Field(None, description="Requester email (creates user if needed)"),
            requester_id: int | None = Field(None, description="Requester user ID"),
            priority: str = Field("normal", description='"urgent", "high", "normal", or "low"'),
            status: str = Field("new", description='"new", "open", "pending", "hold", "solved", or "closed"'),
            ticket_type: str = Field("question", description='"problem", "incident", "question", or "task"'),
            tags: list[str] | None = Field(None, description="Tags for the ticket"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ticket: dict[str, Any] = {
                "subject": subject,
                "comment": {"body": comment_body},
                "priority": priority,
                "status": status,
                "type": ticket_type,
            }
            if requester_email:
                ticket["requester"] = {"email": requester_email}
            elif requester_id:
                ticket["requester_id"] = requester_id
            if tags:
                ticket["tags"] = tags

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tickets.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket": ticket},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_ticket, **kwargs)


class ZendeskListTickets(Tool):
    name: str = "zendesk_list_tickets"
    description: str | None = "List Zendesk tickets with optional filters and pagination."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tickets(
            per_page: int = Field(25, description="Tickets per page (1-100)"),
            page: int = Field(1, description="Page number"),
            sort_by: str = Field("created_at", description="Sort field (e.g. created_at, updated_at)"),
            sort_order: str = Field("desc", description='"asc" or "desc"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page, "sort_by": sort_by, "sort_order": sort_order}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_tickets, **kwargs)


class ZendeskShowTicket(Tool):
    name: str = "zendesk_show_ticket"
    description: str | None = "Get details of a single Zendesk ticket by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_ticket(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_ticket, **kwargs)


class ZendeskUpdateTicket(Tool):
    name: str = "zendesk_update_ticket"
    description: str | None = "Update an existing Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_ticket(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            subject: str | None = Field(None, description="New subject"),
            status: str | None = Field(None, description='"new", "open", "pending", "hold", "solved", or "closed"'),
            priority: str | None = Field(None, description='"urgent", "high", "normal", or "low"'),
            assignee_id: int | None = Field(None, description="Agent ID to assign ticket to"),
            group_id: int | None = Field(None, description="Group ID to assign ticket to"),
            comment_body: str | None = Field(None, description="Add a new comment to the ticket"),
            comment_public: bool = Field(True, description="Whether the new comment is public"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ticket: dict[str, Any] = {}
            if subject is not None:
                ticket["subject"] = subject
            if status is not None:
                ticket["status"] = status
            if priority is not None:
                ticket["priority"] = priority
            if assignee_id is not None:
                ticket["assignee_id"] = assignee_id
            if group_id is not None:
                ticket["group_id"] = group_id
            if comment_body is not None:
                ticket["comment"] = {"body": comment_body, "public": comment_public}

            if not ticket:
                return {"error": "No fields to update"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket": ticket},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_ticket, **kwargs)


class ZendeskSearchTickets(Tool):
    name: str = "zendesk_search_tickets"
    description: str | None = "Search Zendesk tickets by query (Zendesk search syntax)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_tickets(
            query: str = Field(..., description="Search query (e.g. 'status:open type:question')"),
            sort_by: str = Field("created_at", description="Sort field"),
            sort_order: str = Field("desc", description='"asc" or "desc"'),
            per_page: int = Field(25, description="Results per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"query": query, "sort_by": sort_by, "sort_order": sort_order, "per_page": per_page, "page": page}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                data = response.json()
                # Filter to tickets only
                results = data.get("results", [])
                tickets = [r for r in results if r.get("result_type") == "ticket"]
                return {"results": tickets, "count": data.get("count", len(tickets))}

        super().__init__(handler=_search_tickets, **kwargs)


class ZendeskShowManyTickets(Tool):
    name: str = "zendesk_show_many_tickets"
    description: str | None = "Show multiple Zendesk tickets by IDs (comma-separated, up to 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many_tickets(
            ids: list[int] = Field(..., description="Ticket IDs to fetch"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/show_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many_tickets, **kwargs)


class ZendeskUpdateManyTickets(Tool):
    name: str = "zendesk_update_many_tickets"
    description: str | None = "Update multiple Zendesk tickets. Pass ids in query or tickets array in body."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_many_tickets(
            tickets: list[dict[str, Any]] = Field(..., description="List of {id, ...fields} to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/update_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"tickets": tickets},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_many_tickets, **kwargs)


class ZendeskCountTickets(Tool):
    name: str = "zendesk_count_tickets"
    description: str | None = "Get approximate count of tickets in the account."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_tickets() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_tickets, **kwargs)


class ZendeskCreateManyTickets(Tool):
    name: str = "zendesk_create_many_tickets"
    description: str | None = "Create multiple Zendesk tickets in one request."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_many_tickets(
            tickets: list[dict[str, Any]] = Field(..., description="List of ticket objects to create"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tickets/create_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"tickets": tickets},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_many_tickets, **kwargs)


class ZendeskDeleteTicket(Tool):
    name: str = "zendesk_delete_ticket"
    description: str | None = "Delete a Zendesk ticket (soft delete, can be restored)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_ticket(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/tickets/{ticket_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_ticket, **kwargs)


class ZendeskBulkDeleteTickets(Tool):
    name: str = "zendesk_bulk_delete_tickets"
    description: str | None = "Bulk delete Zendesk tickets (soft delete, up to 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_tickets(
            ids: list[int] = Field(..., description="Ticket IDs to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/tickets/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_bulk_delete_tickets, **kwargs)


class ZendeskMarkTicketAsSpam(Tool):
    name: str = "zendesk_mark_ticket_as_spam"
    description: str | None = "Mark a Zendesk ticket as spam and suspend it."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _mark_ticket_as_spam(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}/mark_as_spam.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_mark_ticket_as_spam, **kwargs)


class ZendeskBulkMarkTicketsAsSpam(Tool):
    name: str = "zendesk_bulk_mark_tickets_as_spam"
    description: str | None = "Bulk mark Zendesk tickets as spam (up to 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_mark_tickets_as_spam(
            ids: list[int] = Field(..., description="Ticket IDs to mark as spam"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/mark_many_as_spam.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_mark_tickets_as_spam, **kwargs)


class ZendeskMergeTickets(Tool):
    name: str = "zendesk_merge_tickets"
    description: str | None = "Merge source tickets into a target ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _merge_tickets(
            target_ticket_id: int = Field(..., description="Target ticket ID (ticket to keep)"),
            source_ticket_ids: list[int] = Field(..., description="Source ticket IDs to merge into target"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tickets/{target_ticket_id}/merge.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"source_ticket_ids": source_ticket_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_merge_tickets, **kwargs)


class ZendeskTicketRelatedInformation(Tool):
    name: str = "zendesk_ticket_related_information"
    description: str | None = "Get related information for a ticket (requester, assignee, etc.)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _ticket_related(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/related.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_ticket_related, **kwargs)


class ZendeskListCollaborators(Tool):
    name: str = "zendesk_list_collaborators"
    description: str | None = "List collaborators (CCs) for a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_collaborators(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/collaborators.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_collaborators, **kwargs)


class ZendeskListEmailCcs(Tool):
    name: str = "zendesk_list_email_ccs"
    description: str | None = "List email CCs for a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_email_ccs(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/email_ccs.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_email_ccs, **kwargs)


class ZendeskListFollowers(Tool):
    name: str = "zendesk_list_followers"
    description: str | None = "List followers for a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_followers(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/followers.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_followers, **kwargs)


class ZendeskListDeletedTickets(Tool):
    name: str = "zendesk_list_deleted_tickets"
    description: str | None = "List deleted Zendesk tickets."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deleted_tickets(
            per_page: int = Field(100, description="Per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/deleted_tickets.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_deleted_tickets, **kwargs)


class ZendeskRestoreDeletedTicket(Tool):
    name: str = "zendesk_restore_deleted_ticket"
    description: str | None = "Restore a previously deleted Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _restore_deleted_ticket(
            ticket_id: int = Field(..., description="Deleted ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/deleted_tickets/{ticket_id}/restore.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_restore_deleted_ticket, **kwargs)


class ZendeskRestoreMultipleDeletedTickets(Tool):
    name: str = "zendesk_restore_multiple_deleted_tickets"
    description: str | None = "Restore multiple previously deleted Zendesk tickets."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _restore_multiple_deleted_tickets(
            ids: list[int] = Field(..., description="Deleted ticket IDs to restore"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/deleted_tickets/restore_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_restore_multiple_deleted_tickets, **kwargs)


class ZendeskDeleteTicketPermanently(Tool):
    name: str = "zendesk_delete_ticket_permanently"
    description: str | None = "Permanently delete a Zendesk ticket (cannot be restored)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_ticket_permanently(
            ticket_id: int = Field(..., description="Deleted ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/deleted_tickets/{ticket_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_ticket_permanently, **kwargs)


class ZendeskDeleteMultipleTicketsPermanently(Tool):
    name: str = "zendesk_delete_multiple_tickets_permanently"
    description: str | None = "Permanently delete multiple Zendesk tickets (cannot be restored)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_multiple_tickets_permanently(
            ids: list[int] = Field(..., description="Deleted ticket IDs to permanently delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/deleted_tickets/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_multiple_tickets_permanently, **kwargs)


class ZendeskListTicketIncidents(Tool):
    name: str = "zendesk_list_ticket_incidents"
    description: str | None = "List incidents linked to a problem ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_incidents(
            ticket_id: int = Field(..., description="Problem ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/incidents.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_incidents, **kwargs)


class ZendeskListProblems(Tool):
    name: str = "zendesk_list_problems"
    description: str | None = "List problem-type tickets."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_problems(
            per_page: int = Field(25, description="Per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/problems.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_problems, **kwargs)


class ZendeskAutocompleteProblems(Tool):
    name: str = "zendesk_autocomplete_problems"
    description: str | None = "Autocomplete problem tickets by text."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _autocomplete_problems(
            text: str = Field(..., description="Text to search for in problem subjects"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/problems/autocomplete.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"text": text},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_autocomplete_problems, **kwargs)


class ZendeskCountOrganizationTickets(Tool):
    name: str = "zendesk_count_organization_tickets"
    description: str | None = "Get approximate count of tickets for an organization."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_organization_tickets(
            organization_id: int = Field(..., description="Organization ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/{organization_id}/tickets/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_organization_tickets, **kwargs)


# --- User ---


class ZendeskListUsers(Tool):
    name: str = "zendesk_list_users"
    description: str | None = "List Zendesk users with optional role filter."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_users(
            per_page: int = Field(25, description="Users per page"),
            page: int = Field(1, description="Page number"),
            role: str | None = Field(None, description='"admin", "agent", or "end-user"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if role:
                params["role"] = role

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_users, **kwargs)


class ZendeskShowUser(Tool):
    name: str = "zendesk_show_user"
    description: str | None = "Get details of a single Zendesk user by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_user(
            user_id: int = Field(..., description="Zendesk user ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_user, **kwargs)


# --- Group ---


class ZendeskListGroups(Tool):
    name: str = "zendesk_list_groups"
    description: str | None = "List all Zendesk groups."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_groups() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_groups, **kwargs)


class ZendeskShowGroup(Tool):
    name: str = "zendesk_show_group"
    description: str | None = "Get details of a single Zendesk group by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_group(
            group_id: int = Field(..., description="Zendesk group ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/{group_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_group, **kwargs)


class ZendeskCountGroups(Tool):
    name: str = "zendesk_count_groups"
    description: str | None = "Count all Zendesk groups."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_groups() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_groups, **kwargs)


class ZendeskCreateGroup(Tool):
    name: str = "zendesk_create_group"
    description: str | None = "Create a new Zendesk group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_group(
            name: str = Field(..., description="Group name"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/groups.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"group": {"name": name}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_group, **kwargs)


class ZendeskUpdateGroup(Tool):
    name: str = "zendesk_update_group"
    description: str | None = "Update a Zendesk group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_group(
            group_id: int = Field(..., description="Group ID"),
            name: str = Field(..., description="New group name"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/groups/{group_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"group": {"name": name}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_group, **kwargs)


class ZendeskDeleteGroup(Tool):
    name: str = "zendesk_delete_group"
    description: str | None = "Delete a Zendesk group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_group(
            group_id: int = Field(..., description="Group ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/groups/{group_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json() if response.content else {}

        super().__init__(handler=_delete_group, **kwargs)


class ZendeskListAssignableGroups(Tool):
    name: str = "zendesk_list_assignable_groups"
    description: str | None = "List Zendesk groups that can be assigned to tickets."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_assignable_groups() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/assignable.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_assignable_groups, **kwargs)


# --- Comment ---


class ZendeskListComments(Tool):
    name: str = "zendesk_list_comments"
    description: str | None = "List all comments on a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_comments(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/comments.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_comments, **kwargs)


class ZendeskCountTicketComments(Tool):
    name: str = "zendesk_count_ticket_comments"
    description: str | None = "Count comments on a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_comments(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/comments/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_comments, **kwargs)


class ZendeskMakeCommentPrivate(Tool):
    name: str = "zendesk_make_comment_private"
    description: str | None = "Make a Zendesk ticket comment private."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _make_private(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            ticket_comment_id: int = Field(..., description="Ticket comment ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}/comments/{ticket_comment_id}/make_private.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json() if response.content else {"status": "ok"}

        super().__init__(handler=_make_private, **kwargs)


class ZendeskRedactTicketCommentInAgentWorkspace(Tool):
    name: str = "zendesk_redact_ticket_comment_in_agent_workspace"
    description: str | None = "Redact ticket comment in Agent Workspace (wrap text in <redact> tags)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _redact_comment(
            ticket_comment_id: int = Field(..., description="Ticket comment ID"),
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            html_body: str | None = Field(None, description="HTML body with <redact> tags"),
            external_attachment_urls: list[str] | None = Field(
                None, description="Attachment URLs to redact"
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {"ticket_id": ticket_id}
            if html_body is not None:
                body["html_body"] = html_body
            if external_attachment_urls is not None:
                body["external_attachment_urls"] = external_attachment_urls

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/comment_redactions/{ticket_comment_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_redact_comment, **kwargs)


class ZendeskRedactChatComment(Tool):
    name: str = "zendesk_redact_chat_comment"
    description: str | None = "Redact text in a chat ticket comment (wrap in <redact> tags)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _redact_chat(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            chat_id: str = Field(..., description="Chat ID from ChatStartedEvent"),
            text: str = Field(..., description="Message with <redact> tags"),
            chat_index: int | None = Field(None, description="Chat index (or use message_id)"),
            message_id: str | None = Field(None, description="Message ID (or use chat_index)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {"chat_id": chat_id, "text": text}
            if chat_index is not None:
                body["chat_index"] = chat_index
            if message_id is not None:
                body["message_id"] = message_id

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/chat_redactions/{ticket_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_redact_chat, **kwargs)


class ZendeskRedactChatCommentAttachment(Tool):
    name: str = "zendesk_redact_chat_comment_attachment"
    description: str | None = "Redact chat attachments from a chat ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _redact_chat_attachments(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            chat_id: str = Field(..., description="Chat ID from ChatStartedEvent"),
            chat_indexes: list[int] | None = Field(None, description="Chat indexes (or use message_ids)"),
            message_ids: list[str] | None = Field(None, description="Message IDs (or use chat_indexes)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {"chat_id": chat_id}
            if chat_indexes is not None:
                body["chat_indexes"] = chat_indexes
            if message_ids is not None:
                body["message_ids"] = message_ids

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/chat_file_redactions/{ticket_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_redact_chat_attachments, **kwargs)


class ZendeskRedactStringInComment(Tool):
    name: str = "zendesk_redact_string_in_comment"
    description: str | None = "Redact a string from a ticket comment (legacy endpoint)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _redact_string(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            ticket_comment_id: int = Field(..., description="Ticket comment ID"),
            text: str = Field(..., description="String to redact"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}/comments/{ticket_comment_id}/redact.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"text": text},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_redact_string, **kwargs)


# --- Bookmark ---


class ZendeskListBookmarks(Tool):
    name: str = "zendesk_list_bookmarks"
    description: str | None = "List Zendesk bookmarks."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_bookmarks() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/bookmarks.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_bookmarks, **kwargs)


class ZendeskCreateBookmark(Tool):
    name: str = "zendesk_create_bookmark"
    description: str | None = "Create a Zendesk bookmark for a ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_bookmark(
            ticket_id: int = Field(..., description="Ticket ID to bookmark"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/bookmarks.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"bookmark": {"ticket_id": ticket_id}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_bookmark, **kwargs)


class ZendeskDeleteBookmark(Tool):
    name: str = "zendesk_delete_bookmark"
    description: str | None = "Delete a Zendesk bookmark."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_bookmark(
            bookmark_id: int = Field(..., description="Bookmark ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/bookmarks/{bookmark_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_bookmark, **kwargs)


# --- Workspace ---


class ZendeskListWorkspaces(Tool):
    name: str = "zendesk_list_workspaces"
    description: str | None = "List all Zendesk contextual workspaces."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspaces(
            per_page: int = Field(25, description="Workspaces per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/workspaces.json",
                    headers={"Authorization": auth},
                    params={"per_page": per_page},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_workspaces, **kwargs)


class ZendeskShowWorkspace(Tool):
    name: str = "zendesk_show_workspace"
    description: str | None = "Get details of a single Zendesk workspace by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_workspace(
            workspace_id: int = Field(..., description="Zendesk workspace ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/workspaces/{workspace_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_workspace, **kwargs)


class ZendeskCreateWorkspace(Tool):
    name: str = "zendesk_create_workspace"
    description: str | None = "Create a new Zendesk contextual workspace."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_workspace(
            title: str = Field(..., description="Workspace title"),
            description: str = Field("", description="Workspace description"),
            ticket_form_id: int | None = Field(None, description="Ticket form ID"),
            macro_ids: list[int] | None = Field(None, description="Macro IDs to associate"),
            conditions: dict[str, Any] | None = Field(None, description="Conditions object with all/any arrays"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            workspace: dict[str, Any] = {"title": title, "description": description}
            if ticket_form_id is not None:
                workspace["ticket_form_id"] = ticket_form_id
            if macro_ids is not None:
                workspace["macros"] = macro_ids
            if conditions is not None:
                workspace["conditions"] = conditions

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/workspaces.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"workspace": workspace},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_workspace, **kwargs)


class ZendeskUpdateWorkspace(Tool):
    name: str = "zendesk_update_workspace"
    description: str | None = "Update an existing Zendesk workspace."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_workspace(
            workspace_id: int = Field(..., description="Zendesk workspace ID"),
            title: str | None = Field(None, description="New title"),
            description: str | None = Field(None, description="New description"),
            ticket_form_id: int | None = Field(None, description="Ticket form ID"),
            macro_ids: list[int] | None = Field(None, description="Macro IDs"),
            conditions: dict[str, Any] | None = Field(None, description="Conditions object"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            workspace: dict[str, Any] = {}
            if title is not None:
                workspace["title"] = title
            if description is not None:
                workspace["description"] = description
            if ticket_form_id is not None:
                workspace["ticket_form_id"] = ticket_form_id
            if macro_ids is not None:
                workspace["macro_ids"] = macro_ids
            if conditions is not None:
                workspace["conditions"] = conditions

            if not workspace:
                return {"error": "No fields to update"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/workspaces/{workspace_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"workspace": workspace},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_workspace, **kwargs)


class ZendeskDeleteWorkspace(Tool):
    name: str = "zendesk_delete_workspace"
    description: str | None = "Delete a Zendesk workspace."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_workspace(
            workspace_id: int = Field(..., description="Zendesk workspace ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/workspaces/{workspace_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_workspace, **kwargs)


class ZendeskBulkDeleteWorkspaces(Tool):
    name: str = "zendesk_bulk_delete_workspaces"
    description: str | None = "Bulk delete multiple Zendesk workspaces by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_workspaces(
            ids: list[int] = Field(..., description="Workspace IDs to delete (e.g. [1, 2, 3])"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/workspaces/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_bulk_delete_workspaces, **kwargs)


class ZendeskReorderWorkspaces(Tool):
    name: str = "zendesk_reorder_workspaces"
    description: str | None = "Reorder Zendesk workspaces by providing IDs in desired order."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_workspaces(
            ids: list[int] = Field(..., description="Workspace IDs in desired order (e.g. [12, 32, 48])"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/workspaces/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_workspaces, **kwargs)


# --- View ---


class ZendeskListViews(Tool):
    name: str = "zendesk_list_views"
    description: str | None = "List Zendesk views (shared and personal)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_views(
            per_page: int = Field(50, description="Views per page"),
            page: int = Field(1, description="Page number"),
            sort_by: str = Field("position", description='"alphabetical", "created_at", "updated_at", or "position"'),
            sort_order: str = Field("asc", description='"asc" or "desc"'),
            access: str | None = Field(None, description='"personal", "shared", or "account"'),
            active: bool | None = Field(None, description="Filter by active status"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page, "sort_by": sort_by, "sort_order": sort_order}
            if access:
                params["access"] = access
            if active is not None:
                params["active"] = str(active).lower()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_views, **kwargs)


class ZendeskListViewsCompact(Tool):
    name: str = "zendesk_list_views_compact"
    description: str | None = "List Zendesk views in compact form (max 32 records)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_views_compact() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/compact.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_views_compact, **kwargs)


class ZendeskListViewsById(Tool):
    name: str = "zendesk_list_views_by_id"
    description: str | None = "List Zendesk views by IDs."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_views_by_id(
            ids: list[int] = Field(..., description="View IDs (e.g. [25, 23])"),
            active: bool | None = Field(None, description="Filter by active status"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"ids": ",".join(str(i) for i in ids)}
            if active is not None:
                params["active"] = str(active).lower()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/show_many.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_views_by_id, **kwargs)


class ZendeskListActiveViews(Tool):
    name: str = "zendesk_list_active_views"
    description: str | None = "List active Zendesk views only."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_active_views(
            sort_by: str = Field("position", description="Sort field"),
            sort_order: str = Field("asc", description='"asc" or "desc"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/active.json",
                    headers={"Authorization": auth},
                    params={"sort_by": sort_by, "sort_order": sort_order},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_active_views, **kwargs)


class ZendeskCountViews(Tool):
    name: str = "zendesk_count_views"
    description: str | None = "Get approximate count of Zendesk views."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_views() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_views, **kwargs)


class ZendeskSearchViews(Tool):
    name: str = "zendesk_search_views"
    description: str | None = "Search Zendesk views by title."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_views(
            query: str = Field(..., description="Search query (matches view title)"),
            sort_by: str | None = Field(None, description="Sort field"),
            sort_order: str = Field("asc", description='"asc" or "desc"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query, "sort_order": sort_order}
            if sort_by:
                params["sort_by"] = sort_by

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_views, **kwargs)


class ZendeskShowView(Tool):
    name: str = "zendesk_show_view"
    description: str | None = "Get details of a single Zendesk view by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_view(
            view_id: int = Field(..., description="Zendesk view ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/{view_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_view, **kwargs)


class ZendeskCreateView(Tool):
    name: str = "zendesk_create_view"
    description: str | None = "Create a new Zendesk view. Requires title and at least one condition in 'all'."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_view(
            title: str = Field(..., description="View title"),
            all_conditions: list[dict[str, Any]] = Field(
                ...,
                description="Conditions (all must match). E.g. [{'field':'status','operator':'is','value':'open'}]",
            ),
            any_conditions: list[dict[str, Any]] | None = Field(None, description="Conditions (any can match)"),
            description: str = Field("", description="View description"),
            active: bool = Field(True, description="Whether view is active"),
            output: dict[str, Any] | None = Field(None, description="Output config with columns, sort_by, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            view: dict[str, Any] = {
                "title": title,
                "all": all_conditions,
                "description": description,
                "active": active,
            }
            if any_conditions:
                view["any"] = any_conditions
            if output:
                view["output"] = output

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/views.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"view": view},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_view, **kwargs)


class ZendeskUpdateView(Tool):
    name: str = "zendesk_update_view"
    description: str | None = "Update an existing Zendesk view."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_view(
            view_id: int = Field(..., description="Zendesk view ID"),
            title: str | None = Field(None, description="New title"),
            all_conditions: list[dict[str, Any]] | None = Field(None, description="Replace all conditions"),
            any_conditions: list[dict[str, Any]] | None = Field(None, description="Replace any conditions"),
            active: bool | None = Field(None, description="Active status"),
            output: dict[str, Any] | None = Field(None, description="Output config"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            view: dict[str, Any] = {}
            if title is not None:
                view["title"] = title
            if all_conditions is not None:
                view["all"] = all_conditions
            if any_conditions is not None:
                view["any"] = any_conditions
            if active is not None:
                view["active"] = active
            if output is not None:
                view["output"] = output

            if not view:
                return {"error": "No fields to update"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/views/{view_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"view": view},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_view, **kwargs)


class ZendeskUpdateManyViews(Tool):
    name: str = "zendesk_update_many_views"
    description: str | None = "Update multiple Zendesk views (position, active status)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_many_views(
            views: list[dict[str, Any]] = Field(
                ...,
                description="List of {id, position?, active?} objects. E.g. [{'id':25,'position':3}]",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/views/update_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"views": views},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_many_views, **kwargs)


class ZendeskDeleteView(Tool):
    name: str = "zendesk_delete_view"
    description: str | None = "Delete a Zendesk view."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_view(
            view_id: int = Field(..., description="Zendesk view ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/views/{view_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_view, **kwargs)


class ZendeskBulkDeleteViews(Tool):
    name: str = "zendesk_bulk_delete_views"
    description: str | None = "Bulk delete multiple Zendesk views."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_views(
            ids: list[int] = Field(..., description="View IDs to delete (e.g. [1, 2, 3])"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/views/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_bulk_delete_views, **kwargs)


class ZendeskExecuteView(Tool):
    name: str = "zendesk_execute_view"
    description: str | None = "Execute a Zendesk view and return columns and rows (rate limited: 5/min/view)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _execute_view(
            view_id: int = Field(..., description="Zendesk view ID"),
            sort_by: str | None = Field(None, description="Sort field"),
            sort_order: str = Field("desc", description='"asc" or "desc"'),
            per_page: int = Field(25, description="Results per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page, "sort_order": sort_order}
            if sort_by:
                params["sort_by"] = sort_by

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/{view_id}/execute.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_execute_view, **kwargs)


class ZendeskListTicketsFromView(Tool):
    name: str = "zendesk_list_tickets_from_view"
    description: str | None = "List tickets from a Zendesk view."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tickets_from_view(
            view_id: int = Field(..., description="Zendesk view ID"),
            sort_by: str | None = Field(None, description="Sort field"),
            sort_order: str = Field("desc", description='"asc" or "desc"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"sort_order": sort_order}
            if sort_by:
                params["sort_by"] = sort_by

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/{view_id}/tickets.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_tickets_from_view, **kwargs)


class ZendeskCountTicketsInView(Tool):
    name: str = "zendesk_count_tickets_in_view"
    description: str | None = "Get ticket count for a Zendesk view (rate limited: 5/min/view)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_tickets_in_view(
            view_id: int = Field(..., description="Zendesk view ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/views/{view_id}/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_tickets_in_view, **kwargs)


class ZendeskPreviewViews(Tool):
    name: str = "zendesk_preview_views"
    description: str | None = "Preview a Zendesk view by conditions (rate limited: 5/min/view)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _preview_views(
            view: dict[str, Any] = Field(
                ...,
                description="View object with 'all' conditions and optional 'output'. E.g. {'all':[{'field':'status','operator':'is','value':'open'}]}",
            ),
            per_page: int = Field(25, description="Results per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/views/preview.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"view": view},
                    params={"per_page": per_page, "page": page},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_preview_views, **kwargs)


class ZendeskPreviewTicketCount(Tool):
    name: str = "zendesk_preview_ticket_count"
    description: str | None = "Preview ticket count for view conditions."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _preview_ticket_count(
            view: dict[str, Any] = Field(
                ...,
                description="View object with 'all' conditions. E.g. {'all':[{'field':'status','operator':'is','value':'open'}]}",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/views/preview/count.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"view": view},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_preview_ticket_count, **kwargs)


# --- User (extended) ---


class ZendeskCreateUser(Tool):
    name: str = "zendesk_create_user"
    description: str | None = "Create a new Zendesk user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_user(
            name: str = Field(..., description="User name"),
            user_email: str = Field(..., description="User email address"),
            role: str = Field("end-user", description='"end-user", "agent", or "admin"'),
            verified: bool = Field(True, description="Whether identity is verified"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/users.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"user": {"name": name, "email": user_email, "role": role, "verified": verified}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_user, **kwargs)


class ZendeskUpdateUser(Tool):
    name: str = "zendesk_update_user"
    description: str | None = "Update an existing Zendesk user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_user(
            user_id: int = Field(..., description="Zendesk user ID"),
            name: str | None = Field(None, description="New name"),
            role: str | None = Field(None, description="New role"),
            notes: str | None = Field(None, description="Notes"),
            details: str | None = Field(None, description="Details"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            user: dict[str, Any] = {}
            if name is not None:
                user["name"] = name
            if role is not None:
                user["role"] = role
            if notes is not None:
                user["notes"] = notes
            if details is not None:
                user["details"] = details

            if not user:
                return {"error": "No fields to update"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"user": user},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_user, **kwargs)


class ZendeskDeleteUser(Tool):
    name: str = "zendesk_delete_user"
    description: str | None = "Delete a Zendesk user (soft delete)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_user(
            user_id: int = Field(..., description="Zendesk user ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/users/{user_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_user, **kwargs)


class ZendeskSearchUsers(Tool):
    name: str = "zendesk_search_users"
    description: str | None = "Search Zendesk users by query."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_users(
            query: str = Field(..., description="Search query"),
            sort_by: str = Field("name", description="Sort field (name, created_at, updated_at)"),
            sort_order: str = Field("asc", description='"asc" or "desc"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/search.json",
                    headers={"Authorization": auth},
                    params={"query": query, "sort_by": sort_by, "sort_order": sort_order},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_users, **kwargs)


class ZendeskCountUsers(Tool):
    name: str = "zendesk_count_users"
    description: str | None = "Get approximate count of Zendesk users."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_users() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_users, **kwargs)


class ZendeskShowManyUsers(Tool):
    name: str = "zendesk_show_many_users"
    description: str | None = "Show multiple Zendesk users by IDs."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many_users(
            ids: list[int] = Field(..., description="User IDs (e.g. [1, 2, 3])"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/show_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many_users, **kwargs)


class ZendeskShowSelf(Tool):
    name: str = "zendesk_show_self"
    description: str | None = "Show the currently authenticated user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_self() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/me.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_self, **kwargs)


class ZendeskShowUserRelatedInformation(Tool):
    name: str = "zendesk_show_user_related_information"
    description: str | None = "Show related information for a Zendesk user (requested tickets, etc.)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_user_related(
            user_id: int = Field(..., description="Zendesk user ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_id}/related.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_user_related, **kwargs)


class ZendeskListDeletedUsers(Tool):
    name: str = "zendesk_list_deleted_users"
    description: str | None = "List deleted Zendesk users."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deleted_users(
            per_page: int = Field(50, description="Per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/deleted_users.json",
                    headers={"Authorization": auth},
                    params={"per_page": per_page, "page": page},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_deleted_users, **kwargs)


class ZendeskShowDeletedUser(Tool):
    name: str = "zendesk_show_deleted_user"
    description: str | None = "Show a deleted Zendesk user by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_deleted_user(
            deleted_user_id: int = Field(..., description="Deleted user ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/deleted_users/{deleted_user_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_deleted_user, **kwargs)


class ZendeskCountDeletedUsers(Tool):
    name: str = "zendesk_count_deleted_users"
    description: str | None = "Count deleted Zendesk users."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_deleted_users() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/deleted_users/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_deleted_users, **kwargs)


class ZendeskAutocompleteUsers(Tool):
    name: str = "zendesk_autocomplete_users"
    description: str | None = "Autocomplete Zendesk users by name or email prefix."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _autocomplete_users(
            name: str = Field(..., description="Name or email prefix to search"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/autocomplete.json",
                    headers={"Authorization": auth},
                    params={"name": name},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_autocomplete_users, **kwargs)


class ZendeskBulkDeleteUsers(Tool):
    name: str = "zendesk_bulk_delete_users"
    description: str | None = "Bulk delete multiple Zendesk users."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_users(
            ids: list[int] = Field(..., description="User IDs to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/users/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_bulk_delete_users, **kwargs)


class ZendeskPermanentlyDeleteUser(Tool):
    name: str = "zendesk_permanently_delete_user"
    description: str | None = "Permanently delete a Zendesk user (from deleted users)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _permanently_delete_user(
            deleted_user_id: int = Field(..., description="Deleted user ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/deleted_users/{deleted_user_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "permanently_deleted"}

        super().__init__(handler=_permanently_delete_user, **kwargs)


# --- Category (Ticket Trigger Categories) ---


class ZendeskListTicketTriggerCategories(Tool):
    name: str = "zendesk_list_ticket_trigger_categories"
    description: str | None = "List ticket trigger categories."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_categories(
            include: str | None = Field(None, description="Sideloads e.g. rule_counts"),
            sort: str | None = Field(None, description="Sort: position, name, created_at, updated_at"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if include:
                params["include"] = include
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/trigger_categories.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_categories, **kwargs)


class ZendeskCreateTicketTriggerCategory(Tool):
    name: str = "zendesk_create_ticket_trigger_category"
    description: str | None = "Create a ticket trigger category."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_category(
            trigger_category: dict[str, Any] = Field(
                ...,
                description="Category object: name, position",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/trigger_categories.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"trigger_category": trigger_category},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_category, **kwargs)


class ZendeskCreateBatchJobForTicketTriggerCategories(Tool):
    name: str = "zendesk_create_batch_job_for_ticket_trigger_categories"
    description: str | None = "Create batch job for ticket trigger categories (patch positions, triggers)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_batch_job(
            job: dict[str, Any] = Field(
                ...,
                description="Job object: action (patch), items (trigger_categories, triggers)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/trigger_categories/jobs.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"job": job},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_batch_job, **kwargs)


class ZendeskShowTicketTriggerCategory(Tool):
    name: str = "zendesk_show_ticket_trigger_category"
    description: str | None = "Show a ticket trigger category by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_category(
            trigger_category_id: str = Field(..., description="Trigger category ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/trigger_categories/{trigger_category_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_category, **kwargs)


class ZendeskUpdateTicketTriggerCategory(Tool):
    name: str = "zendesk_update_ticket_trigger_category"
    description: str | None = "Update a ticket trigger category."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_category(
            trigger_category_id: str = Field(..., description="Trigger category ID"),
            trigger_category: dict[str, Any] = Field(
                ...,
                description="Fields to update: name, position",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{base_url}/trigger_categories/{trigger_category_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"trigger_category": trigger_category},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_category, **kwargs)


class ZendeskDeleteTicketTriggerCategory(Tool):
    name: str = "zendesk_delete_ticket_trigger_category"
    description: str | None = "Delete a ticket trigger category."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_category(
            trigger_category_id: str = Field(..., description="Trigger category ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/trigger_categories/{trigger_category_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_category, **kwargs)


# --- Trigger (Ticket Triggers) ---


class ZendeskListTicketTriggers(Tool):
    name: str = "zendesk_list_ticket_triggers"
    description: str | None = "List all Zendesk ticket triggers."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_triggers(
            active: bool | None = Field(None, description="Filter by active status"),
            per_page: int = Field(50, description="Per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if active is not None:
                params["active"] = str(active).lower()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/triggers.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_triggers, **kwargs)


class ZendeskListActiveTicketTriggers(Tool):
    name: str = "zendesk_list_active_ticket_triggers"
    description: str | None = "List active Zendesk ticket triggers only."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_active_triggers() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/triggers/active.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_active_triggers, **kwargs)


class ZendeskSearchTicketTriggers(Tool):
    name: str = "zendesk_search_ticket_triggers"
    description: str | None = "Search Zendesk ticket triggers by title."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_triggers(
            query: str = Field(..., description="Search query (matches trigger title)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/triggers/search.json",
                    headers={"Authorization": auth},
                    params={"query": query},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_triggers, **kwargs)


class ZendeskShowTicketTrigger(Tool):
    name: str = "zendesk_show_ticket_trigger"
    description: str | None = "Show a Zendesk ticket trigger by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_trigger(
            trigger_id: int = Field(..., description="Zendesk trigger ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/triggers/{trigger_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_trigger, **kwargs)


class ZendeskListTicketTriggerDefinitions(Tool):
    name: str = "zendesk_list_ticket_trigger_definitions"
    description: str | None = "List ticket trigger action and condition definitions."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_definitions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/triggers/definitions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_definitions, **kwargs)


class ZendeskListTicketTriggerRevisions(Tool):
    name: str = "zendesk_list_ticket_trigger_revisions"
    description: str | None = "List revisions for a Zendesk ticket trigger."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_revisions(
            trigger_id: int = Field(..., description="Zendesk trigger ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/triggers/{trigger_id}/revisions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_revisions, **kwargs)


class ZendeskCreateTicketTrigger(Tool):
    name: str = "zendesk_create_ticket_trigger"
    description: str | None = "Create a new Zendesk ticket trigger."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_trigger(
            title: str = Field(..., description="Trigger title"),
            actions: list[dict[str, Any]] = Field(..., description="Actions array"),
            conditions: dict[str, Any] = Field(
                ...,
                description="Conditions with all/any arrays. E.g. {'all':[{'field':'status','operator':'is','value':'open'}]}",
            ),
            description: str = Field("", description="Trigger description"),
            active: bool = Field(True, description="Whether trigger is active"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            trigger: dict[str, Any] = {
                "title": title,
                "actions": actions,
                "conditions": conditions,
                "description": description,
                "active": active,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/triggers.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"trigger": trigger},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_trigger, **kwargs)


class ZendeskUpdateTicketTrigger(Tool):
    name: str = "zendesk_update_ticket_trigger"
    description: str | None = "Update a Zendesk ticket trigger."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_trigger(
            trigger_id: int = Field(..., description="Zendesk trigger ID"),
            title: str | None = Field(None, description="New title"),
            actions: list[dict[str, Any]] | None = Field(None, description="New actions"),
            conditions: dict[str, Any] | None = Field(None, description="New conditions"),
            active: bool | None = Field(None, description="Active status"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            trigger: dict[str, Any] = {}
            if title is not None:
                trigger["title"] = title
            if actions is not None:
                trigger["actions"] = actions
            if conditions is not None:
                trigger["conditions"] = conditions
            if active is not None:
                trigger["active"] = active

            if not trigger:
                return {"error": "No fields to update"}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/triggers/{trigger_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"trigger": trigger},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_trigger, **kwargs)


class ZendeskDeleteTicketTrigger(Tool):
    name: str = "zendesk_delete_ticket_trigger"
    description: str | None = "Delete a Zendesk ticket trigger."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_trigger(
            trigger_id: int = Field(..., description="Zendesk trigger ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/triggers/{trigger_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_trigger, **kwargs)


class ZendeskBulkDeleteTicketTriggers(Tool):
    name: str = "zendesk_bulk_delete_ticket_triggers"
    description: str | None = "Bulk delete Zendesk ticket triggers."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_triggers(
            ids: list[int] = Field(..., description="Trigger IDs to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/triggers/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_bulk_delete_triggers, **kwargs)


# --- Target ---


class ZendeskListTargets(Tool):
    name: str = "zendesk_list_targets"
    description: str | None = "List Zendesk targets (notification endpoints for triggers/automations)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_targets() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/targets.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_targets, **kwargs)


class ZendeskShowTarget(Tool):
    name: str = "zendesk_show_target"
    description: str | None = "Get details of a Zendesk target by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_target(
            target_id: int = Field(..., description="Zendesk target ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/targets/{target_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_target, **kwargs)


class ZendeskCreateTarget(Tool):
    name: str = "zendesk_create_target"
    description: str | None = "Create a Zendesk target (e.g. email_target). Requires type, title, and type-specific fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_target(
            target: dict[str, Any] = Field(..., description="Target object: type, title, and type-specific fields (e.g. email, subject for email_target)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/targets.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"target": target},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_target, **kwargs)


class ZendeskUpdateTarget(Tool):
    name: str = "zendesk_update_target"
    description: str | None = "Update a Zendesk target."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_target(
            target_id: int = Field(..., description="Zendesk target ID"),
            target: dict[str, Any] = Field(..., description="Target fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/targets/{target_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"target": target},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_target, **kwargs)


class ZendeskDeleteTarget(Tool):
    name: str = "zendesk_delete_target"
    description: str | None = "Delete a Zendesk target."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_target(
            target_id: int = Field(..., description="Zendesk target ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/targets/{target_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_target, **kwargs)


class ZendeskListTargetFailures(Tool):
    name: str = "zendesk_list_target_failures"
    description: str | None = "List recent Zendesk target failures (25 most recent per target)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_target_failures() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/target_failures.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_target_failures, **kwargs)


class ZendeskShowTargetFailure(Tool):
    name: str = "zendesk_show_target_failure"
    description: str | None = "Get details of a Zendesk target failure by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_target_failure(
            target_failure_id: int = Field(..., description="Zendesk target failure ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/target_failures/{target_failure_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_target_failure, **kwargs)


# --- Automation ---


class ZendeskListAutomations(Tool):
    name: str = "zendesk_list_automations"
    description: str | None = "List Zendesk automations."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_automations(
            active: bool | None = Field(None, description="Filter by active status"),
            per_page: int = Field(50, description="Per page"),
            sort: str | None = Field(None, description="Sort field"),
            include: str | None = Field(None, description="Sideload e.g. usage_24h"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page}
            if active is not None:
                params["active"] = str(active).lower()
            if sort:
                params["sort"] = sort
            if include:
                params["include"] = include

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/automations.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_automations, **kwargs)


class ZendeskListActiveAutomations(Tool):
    name: str = "zendesk_list_active_automations"
    description: str | None = "List active Zendesk automations."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_active() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/automations/active.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_active, **kwargs)


class ZendeskSearchAutomations(Tool):
    name: str = "zendesk_search_automations"
    description: str | None = "Search Zendesk automations by title."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search(
            query: str = Field(..., description="Search query (matches title)"),
            active: bool | None = Field(None, description="Filter by active"),
            sort_by: str | None = Field(None, description="alphabetical, created_at, updated_at, position"),
            sort_order: str | None = Field(None, description="asc or desc"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query}
            if active is not None:
                params["active"] = str(active).lower()
            if sort_by:
                params["sort_by"] = sort_by
            if sort_order:
                params["sort_order"] = sort_order

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/automations/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search, **kwargs)


class ZendeskShowAutomation(Tool):
    name: str = "zendesk_show_automation"
    description: str | None = "Show a Zendesk automation by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show(
            automation_id: int = Field(..., description="Automation ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/automations/{automation_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show, **kwargs)


class ZendeskCreateAutomation(Tool):
    name: str = "zendesk_create_automation"
    description: str | None = "Create a Zendesk automation."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            automation: dict[str, Any] = Field(
                ...,
                description="Automation object: title, all (conditions), actions",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/automations.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"automation": automation},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create, **kwargs)


class ZendeskUpdateAutomation(Tool):
    name: str = "zendesk_update_automation"
    description: str | None = "Update a Zendesk automation."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update(
            automation_id: int = Field(..., description="Automation ID"),
            automation: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/automations/{automation_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"automation": automation},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update, **kwargs)


class ZendeskUpdateManyAutomations(Tool):
    name: str = "zendesk_update_many_automations"
    description: str | None = "Update many Zendesk automations (position, active)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_many(
            automations: list[dict[str, Any]] = Field(
                ...,
                description="List of {id, position?, active?}",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/automations/update_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"automations": automations},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_many, **kwargs)


class ZendeskDeleteAutomation(Tool):
    name: str = "zendesk_delete_automation"
    description: str | None = "Delete a Zendesk automation."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(
            automation_id: int = Field(..., description="Automation ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/automations/{automation_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete, **kwargs)


class ZendeskBulkDeleteAutomations(Tool):
    name: str = "zendesk_bulk_delete_automations"
    description: str | None = "Bulk delete Zendesk automations."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete(
            ids: str = Field(..., description="Comma-separated automation IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/automations/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_bulk_delete, **kwargs)


# --- Tag ---


class ZendeskSearchTags(Tool):
    name: str = "zendesk_search_tags"
    description: str | None = "Search Zendesk tags by name (min 2 characters)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_tags(
            name: str = Field(..., description="Tag name substring (min 2 chars)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/autocomplete/tags.json",
                    headers={"Authorization": auth},
                    params={"name": name},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_tags, **kwargs)


class ZendeskListTags(Tool):
    name: str = "zendesk_list_tags"
    description: str | None = "List Zendesk tags (up to 20k most popular in last 60 days)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tags(
            per_page: int = Field(100, description="Tags per page"),
            page: int = Field(1, description="Page number"),
            sort: str = Field("name", description="Sort field (prefix with - for desc)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page, "sort": sort}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tags.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_tags, **kwargs)


class ZendeskCountTags(Tool):
    name: str = "zendesk_count_tags"
    description: str | None = "Get approximate count of Zendesk tags."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_tags() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tags/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_tags, **kwargs)


class ZendeskListResourceTags(Tool):
    name: str = "zendesk_list_resource_tags"
    description: str | None = "List tags for a ticket (resource tags)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_resource_tags(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/tags.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_resource_tags, **kwargs)


class ZendeskAddTags(Tool):
    name: str = "zendesk_add_tags"
    description: str | None = "Add tags to a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_tags(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            tags: list[str] = Field(..., description="Tags to add"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}/tags.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"tags": tags},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_add_tags, **kwargs)


class ZendeskSetTags(Tool):
    name: str = "zendesk_set_tags"
    description: str | None = "Set (replace) tags on a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_tags(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            tags: list[str] = Field(..., description="Tags to set (replaces existing)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tickets/{ticket_id}/tags.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"tags": tags},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_tags, **kwargs)


class ZendeskRemoveTags(Tool):
    name: str = "zendesk_remove_tags"
    description: str | None = "Remove tags from a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_tags(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            tags: list[str] = Field(..., description="Tags to remove"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/tickets/{ticket_id}/tags.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"tags": tags},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_remove_tags, **kwargs)


# --- Suspended Ticket ---


class ZendeskListSuspendedTickets(Tool):
    name: str = "zendesk_list_suspended_tickets"
    description: str | None = "List Zendesk suspended tickets (emails pending review)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_suspended_tickets(
            per_page: int = Field(25, description="Per page"),
            page: int = Field(1, description="Page number"),
            sort_by: str = Field("created_at", description="Sort by: author_email, cause, created_at, subject"),
            sort_order: str = Field("desc", description='"asc" or "desc"'),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page, "sort_by": sort_by, "sort_order": sort_order}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/suspended_tickets.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_suspended_tickets, **kwargs)


class ZendeskShowSuspendedTicket(Tool):
    name: str = "zendesk_show_suspended_ticket"
    description: str | None = "Get details of a Zendesk suspended ticket by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_suspended_ticket(
            suspended_ticket_id: int = Field(..., description="Suspended ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/suspended_tickets/{suspended_ticket_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_suspended_ticket, **kwargs)


class ZendeskRecoverSuspendedTicket(Tool):
    name: str = "zendesk_recover_suspended_ticket"
    description: str | None = "Recover a single Zendesk suspended ticket (accept as valid ticket)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _recover_suspended_ticket(
            suspended_ticket_id: int = Field(..., description="Suspended ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/suspended_tickets/{suspended_ticket_id}/recover.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_recover_suspended_ticket, **kwargs)


class ZendeskRecoverMultipleSuspendedTickets(Tool):
    name: str = "zendesk_recover_multiple_suspended_tickets"
    description: str | None = "Recover multiple Zendesk suspended tickets (up to 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _recover_multiple_suspended_tickets(
            ids: list[int] = Field(..., description="Suspended ticket IDs to recover"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/suspended_tickets/recover_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_recover_multiple_suspended_tickets, **kwargs)


class ZendeskDeleteSuspendedTicket(Tool):
    name: str = "zendesk_delete_suspended_ticket"
    description: str | None = "Delete a Zendesk suspended ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_suspended_ticket(
            suspended_ticket_id: int = Field(..., description="Suspended ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/suspended_tickets/{suspended_ticket_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_suspended_ticket, **kwargs)


class ZendeskDeleteMultipleSuspendedTickets(Tool):
    name: str = "zendesk_delete_multiple_suspended_tickets"
    description: str | None = "Delete multiple Zendesk suspended tickets (up to 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_multiple_suspended_tickets(
            ids: list[int] = Field(..., description="Suspended ticket IDs to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/suspended_tickets/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_multiple_suspended_tickets, **kwargs)


class ZendeskSuspendedTicketAttachments(Tool):
    name: str = "zendesk_suspended_ticket_attachments"
    description: str | None = "Get attachment tokens from a suspended ticket for use when recovering."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _suspended_ticket_attachments(
            suspended_ticket_id: int = Field(..., description="Suspended ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/suspended_tickets/{suspended_ticket_id}/attachments.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_suspended_ticket_attachments, **kwargs)


class ZendeskExportSuspendedTickets(Tool):
    name: str = "zendesk_export_suspended_tickets"
    description: str | None = "Export suspended tickets as CSV. Zendesk emails a link when done (1 req/min limit)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _export_suspended_tickets() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/suspended_tickets/export.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_export_suspended_tickets, **kwargs)


# --- Organization Subscription ---


class ZendeskShowOrganizationSubscription(Tool):
    name: str = "zendesk_show_organization_subscription"
    description: str | None = "Get details of an organization subscription by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_organization_subscription(
            organization_subscription_id: int = Field(..., description="Organization subscription ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_subscriptions/{organization_subscription_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_organization_subscription, **kwargs)


# --- Custom Ticket Status ---


class ZendeskListCustomTicketStatuses(Tool):
    name: str = "zendesk_list_custom_ticket_statuses"
    description: str | None = "List Zendesk custom ticket statuses."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_custom_statuses(
            active: bool | None = Field(None, description="Filter by active (true/false)"),
            default: bool | None = Field(None, description="Filter by default (true/false)"),
            status_categories: str | None = Field(None, description="Comma-separated status categories"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if active is not None:
                params["active"] = active
            if default is not None:
                params["default"] = default
            if status_categories:
                params["status_categories"] = status_categories
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_statuses.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_custom_statuses, **kwargs)


class ZendeskShowCustomTicketStatus(Tool):
    name: str = "zendesk_show_custom_ticket_status"
    description: str | None = "Get a Zendesk custom ticket status by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_custom_status(
            custom_status_id: int = Field(..., description="Custom status ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_statuses/{custom_status_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_custom_status, **kwargs)


class ZendeskCreateCustomTicketStatus(Tool):
    name: str = "zendesk_create_custom_ticket_status"
    description: str | None = "Create a Zendesk custom ticket status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_custom_status(
            custom_status: dict[str, Any] = Field(..., description="Custom status: agent_label, status_category, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/custom_statuses.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_status": custom_status},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_custom_status, **kwargs)


class ZendeskUpdateCustomTicketStatus(Tool):
    name: str = "zendesk_update_custom_ticket_status"
    description: str | None = "Update a Zendesk custom ticket status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_custom_status(
            custom_status_id: int = Field(..., description="Custom status ID"),
            custom_status: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/custom_statuses/{custom_status_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_status": custom_status},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_custom_status, **kwargs)


class ZendeskBulkUpdateDefaultCustomTicketStatuses(Tool):
    name: str = "zendesk_bulk_update_default_custom_ticket_statuses"
    description: str | None = "Bulk update default custom ticket statuses per category."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_update_default(
            ids: list[int] = Field(..., description="Custom status IDs to set as default for their category"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_str = ",".join(str(i) for i in ids)
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/custom_status/default.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ids": ids_str},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_update_default, **kwargs)


class ZendeskCreateTicketFormStatusesForCustomStatus(Tool):
    name: str = "zendesk_create_ticket_form_statuses_for_custom_status"
    description: str | None = "Create ticket form status associations for a custom status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_ticket_form_statuses(
            custom_status_id: int = Field(..., description="Custom status ID"),
            ticket_form_status: list[dict[str, Any]] = Field(..., description="List of {ticket_form_id: N}"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/custom_statuses/{custom_status_id}/ticket_form_statuses.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_form_status": ticket_form_status},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_ticket_form_statuses, **kwargs)


# --- Job Status ---


class ZendeskListJobStatuses(Tool):
    name: str = "zendesk_list_job_statuses"
    description: str | None = "List Zendesk background job statuses."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_job_statuses() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/job_statuses.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_job_statuses, **kwargs)


class ZendeskShowJobStatus(Tool):
    name: str = "zendesk_show_job_status"
    description: str | None = "Get a Zendesk job status by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_job_status(
            job_status_id: str = Field(..., description="Job status ID (string)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/job_statuses/{job_status_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_job_status, **kwargs)


class ZendeskShowManyJobStatuses(Tool):
    name: str = "zendesk_show_many_job_statuses"
    description: str | None = "Show multiple Zendesk job statuses by IDs."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many_job_statuses(
            ids: list[str] = Field(..., description="Job status IDs (comma-separated)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids_param = ",".join(ids)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/job_statuses/show_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many_job_statuses, **kwargs)


# --- Organization Subscription (extra) ---


class ZendeskListOrganizationSubscriptions(Tool):
    name: str = "zendesk_list_organization_subscriptions"
    description: str | None = "List Zendesk organization subscriptions."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_organization_subscriptions(
            per_page: int = Field(100, description="Per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_subscriptions.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_organization_subscriptions, **kwargs)


class ZendeskCreateOrganizationSubscription(Tool):
    name: str = "zendesk_create_organization_subscription"
    description: str | None = "Create a Zendesk organization subscription."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_organization_subscription(
            organization_id: int = Field(..., description="Organization ID"),
            user_id: int = Field(..., description="User ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organization_subscriptions.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_subscription": {"organization_id": organization_id, "user_id": user_id}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_organization_subscription, **kwargs)


class ZendeskDeleteOrganizationSubscription(Tool):
    name: str = "zendesk_delete_organization_subscription"
    description: str | None = "Delete a Zendesk organization subscription."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_organization_subscription(
            organization_subscription_id: int = Field(..., description="Organization subscription ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organization_subscriptions/{organization_subscription_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_organization_subscription, **kwargs)


# --- Session ---


class ZendeskListSessions(Tool):
    name: str = "zendesk_list_sessions"
    description: str | None = "List Zendesk sessions (all for admin, own for agent/end-user)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sessions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/sessions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_sessions, **kwargs)


class ZendeskShowSession(Tool):
    name: str = "zendesk_show_session"
    description: str | None = "Get a Zendesk session by user ID and session ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_session(
            user_id: int = Field(..., description="User ID"),
            session_id: int = Field(..., description="Session ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_id}/sessions/{session_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_session, **kwargs)


class ZendeskShowCurrentSession(Tool):
    name: str = "zendesk_show_current_session"
    description: str | None = "Show the currently authenticated session."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_current_session() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/me/session.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_current_session, **kwargs)


class ZendeskRenewCurrentSession(Tool):
    name: str = "zendesk_renew_current_session"
    description: str | None = "Renew the current session (returns new authenticity token)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _renew_current_session() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/me/session/renew.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_renew_current_session, **kwargs)


class ZendeskDeleteSession(Tool):
    name: str = "zendesk_delete_session"
    description: str | None = "Delete a Zendesk session (sign out user)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_session(
            user_id: int = Field(..., description="User ID"),
            session_id: int = Field(..., description="Session ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/users/{user_id}/sessions/{session_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_session, **kwargs)


class ZendeskDeleteCurrentSession(Tool):
    name: str = "zendesk_delete_current_session"
    description: str | None = "Delete the authenticated session (logout). Works with session auth only."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_current_session() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/users/me/logout.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_current_session, **kwargs)


# --- Sharing Agreement ---


class ZendeskListSharingAgreements(Tool):
    name: str = "zendesk_list_sharing_agreements"
    description: str | None = "List Zendesk sharing agreements."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sharing_agreements() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/sharing_agreements.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_sharing_agreements, **kwargs)


class ZendeskShowSharingAgreement(Tool):
    name: str = "zendesk_show_sharing_agreement"
    description: str | None = "Get a Zendesk sharing agreement by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_sharing_agreement(
            sharing_agreement_id: int = Field(..., description="Sharing agreement ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/sharing_agreements/{sharing_agreement_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_sharing_agreement, **kwargs)


class ZendeskCreateSharingAgreement(Tool):
    name: str = "zendesk_create_sharing_agreement"
    description: str | None = "Create a Zendesk sharing agreement."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_sharing_agreement(
            sharing_agreement: dict[str, Any] = Field(..., description="Sharing agreement (e.g. remote_subdomain)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/sharing_agreements.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sharing_agreement": sharing_agreement},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_sharing_agreement, **kwargs)


class ZendeskUpdateSharingAgreement(Tool):
    name: str = "zendesk_update_sharing_agreement"
    description: str | None = "Update a Zendesk sharing agreement (status only)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_sharing_agreement(
            sharing_agreement_id: int = Field(..., description="Sharing agreement ID"),
            sharing_agreement: dict[str, Any] = Field(..., description="Fields to update (e.g. status)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/sharing_agreements/{sharing_agreement_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sharing_agreement": sharing_agreement},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_sharing_agreement, **kwargs)


class ZendeskDeleteSharingAgreement(Tool):
    name: str = "zendesk_delete_sharing_agreement"
    description: str | None = "Delete a Zendesk sharing agreement."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_sharing_agreement(
            sharing_agreement_id: int = Field(..., description="Sharing agreement ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/sharing_agreements/{sharing_agreement_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_sharing_agreement, **kwargs)


# --- SLA Policy ---


class ZendeskListSlaPolicies(Tool):
    name: str = "zendesk_list_sla_policies"
    description: str | None = "List Zendesk SLA policies (Support Professional+)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sla_policies() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/slas/policies.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_sla_policies, **kwargs)


class ZendeskShowSlaPolicy(Tool):
    name: str = "zendesk_show_sla_policy"
    description: str | None = "Get a Zendesk SLA policy by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_sla_policy(
            sla_policy_id: int = Field(..., description="SLA policy ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/slas/policies/{sla_policy_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_sla_policy, **kwargs)


class ZendeskCreateSlaPolicy(Tool):
    name: str = "zendesk_create_sla_policy"
    description: str | None = "Create a Zendesk SLA policy."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_sla_policy(
            sla_policy: dict[str, Any] = Field(..., description="SLA policy: title, filter, policy_metrics, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/slas/policies.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sla_policy": sla_policy},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_sla_policy, **kwargs)


class ZendeskUpdateSlaPolicy(Tool):
    name: str = "zendesk_update_sla_policy"
    description: str | None = "Update a Zendesk SLA policy."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_sla_policy(
            sla_policy_id: int = Field(..., description="SLA policy ID"),
            sla_policy: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/slas/policies/{sla_policy_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sla_policy": sla_policy},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_sla_policy, **kwargs)


class ZendeskDeleteSlaPolicy(Tool):
    name: str = "zendesk_delete_sla_policy"
    description: str | None = "Delete a Zendesk SLA policy."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_sla_policy(
            sla_policy_id: int = Field(..., description="SLA policy ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/slas/policies/{sla_policy_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_sla_policy, **kwargs)


class ZendeskReorderSlaPolicies(Tool):
    name: str = "zendesk_reorder_sla_policies"
    description: str | None = "Reorder Zendesk SLA policies."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_sla_policies(
            sla_policy_ids: list[int] = Field(..., description="SLA policy IDs in desired order"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/slas/policies/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sla_policy_ids": sla_policy_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_sla_policies, **kwargs)


class ZendeskRetrieveSlaPolicyFilterDefinitions(Tool):
    name: str = "zendesk_retrieve_sla_policy_filter_definitions"
    description: str | None = "Retrieve SLA policy filter definition items (supported fields, operators, values)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_definitions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/slas/policies/definitions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_definitions, **kwargs)


# --- Group SLA Policy ---


class ZendeskListGroupSlaPolicies(Tool):
    name: str = "zendesk_list_group_sla_policies"
    description: str | None = "List Zendesk SLA policies for a group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_group_sla_policies(
            group_id: int = Field(..., description="Group ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/{group_id}/slas/policies.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_group_sla_policies, **kwargs)


class ZendeskShowGroupSlaPolicy(Tool):
    name: str = "zendesk_show_group_sla_policy"
    description: str | None = "Get a Zendesk group SLA policy by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_group_sla_policy(
            group_id: int = Field(..., description="Group ID"),
            sla_policy_id: int = Field(..., description="SLA policy ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/{group_id}/slas/policies/{sla_policy_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_group_sla_policy, **kwargs)


class ZendeskCreateGroupSlaPolicy(Tool):
    name: str = "zendesk_create_group_sla_policy"
    description: str | None = "Create a Zendesk SLA policy for a group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_group_sla_policy(
            group_id: int = Field(..., description="Group ID"),
            sla_policy: dict[str, Any] = Field(..., description="SLA policy object"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/groups/{group_id}/slas/policies.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sla_policy": sla_policy},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_group_sla_policy, **kwargs)


class ZendeskUpdateGroupSlaPolicy(Tool):
    name: str = "zendesk_update_group_sla_policy"
    description: str | None = "Update a Zendesk group SLA policy."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_group_sla_policy(
            group_id: int = Field(..., description="Group ID"),
            sla_policy_id: int = Field(..., description="SLA policy ID"),
            sla_policy: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/groups/{group_id}/slas/policies/{sla_policy_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sla_policy": sla_policy},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_group_sla_policy, **kwargs)


class ZendeskDeleteGroupSlaPolicy(Tool):
    name: str = "zendesk_delete_group_sla_policy"
    description: str | None = "Delete a Zendesk group SLA policy."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_group_sla_policy(
            group_id: int = Field(..., description="Group ID"),
            sla_policy_id: int = Field(..., description="SLA policy ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/groups/{group_id}/slas/policies/{sla_policy_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_group_sla_policy, **kwargs)


# --- Deletion Schedule ---


class ZendeskListDeletionSchedules(Tool):
    name: str = "zendesk_list_deletion_schedules"
    description: str | None = "List all Zendesk deletion schedules for the account."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deletion_schedules() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/deletion_schedules.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_deletion_schedules, **kwargs)


class ZendeskCreateDeletionSchedule(Tool):
    name: str = "zendesk_create_deletion_schedule"
    description: str | None = "Create a Zendesk deletion schedule."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_deletion_schedule(
            deletion_schedule: dict[str, Any] = Field(..., description="Deletion schedule object (title, description, active, conditions, object)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/deletion_schedules.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"deletion_schedule": deletion_schedule},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_deletion_schedule, **kwargs)


class ZendeskGetDeletionSchedule(Tool):
    name: str = "zendesk_get_deletion_schedule"
    description: str | None = "Get a Zendesk deletion schedule by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_deletion_schedule(
            deletion_schedule_id: int = Field(..., description="Deletion schedule ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/deletion_schedules/{deletion_schedule_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_deletion_schedule, **kwargs)


class ZendeskUpdateDeletionSchedule(Tool):
    name: str = "zendesk_update_deletion_schedule"
    description: str | None = "Update a Zendesk deletion schedule."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_deletion_schedule(
            deletion_schedule_id: int = Field(..., description="Deletion schedule ID"),
            deletion_schedule: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/deletion_schedules/{deletion_schedule_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"deletion_schedule": deletion_schedule},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_deletion_schedule, **kwargs)


class ZendeskDeleteDeletionSchedule(Tool):
    name: str = "zendesk_delete_deletion_schedule"
    description: str | None = "Delete a Zendesk deletion schedule."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_deletion_schedule(
            deletion_schedule_id: int = Field(..., description="Deletion schedule ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/deletion_schedules/{deletion_schedule_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_deletion_schedule, **kwargs)


# --- Search ---


class ZendeskListSearchResults(Tool):
    name: str = "zendesk_list_search_results"
    description: str | None = "List Zendesk search results (tickets, users, organizations)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_search_results(
            query: str = Field(..., description="Search query (e.g. type:ticket status:open)"),
            sort_by: str = Field("updated_at", description="Sort field: updated_at, created_at, priority, status, ticket_type"),
            sort_order: str = Field("desc", description="asc or desc"),
            include: str | None = Field(None, description="Sideloads to include (e.g. users,organizations)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query, "sort_by": sort_by, "sort_order": sort_order}
            if include:
                params["include"] = include

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_search_results, **kwargs)


class ZendeskShowSearchResultsCount(Tool):
    name: str = "zendesk_show_search_results_count"
    description: str | None = "Get the count of items matching a Zendesk search query."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_search_results_count(
            query: str = Field(..., description="Search query"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/search/count.json",
                    headers={"Authorization": auth},
                    params={"query": query},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_search_results_count, **kwargs)


class ZendeskExportSearchResults(Tool):
    name: str = "zendesk_export_search_results"
    description: str | None = "Export Zendesk search results (for queries returning >1000 results)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _export_search_results(
            query: str = Field(..., description="Search query"),
            filter_type: str = Field(..., description="Object type: ticket, organization, user, or group"),
            page_size: int = Field(100, description="Results per page (max 1000)"),
            page_after: str | None = Field(None, description="Cursor for next page"),
            include: str | None = Field(None, description="Sideloads to include"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query, "filter[type]": filter_type, "page[size]": page_size}
            if page_after:
                params["page[after]"] = page_after
            if include:
                params["include"] = include

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/search/export.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_export_search_results, **kwargs)


# --- Custom Roles ---


class ZendeskListCustomRoles(Tool):
    name: str = "zendesk_list_custom_roles"
    description: str | None = "List Zendesk custom agent roles (Enterprise plan)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_custom_roles() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_roles.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_custom_roles, **kwargs)


class ZendeskShowCustomRole(Tool):
    name: str = "zendesk_show_custom_role"
    description: str | None = "Show a Zendesk custom role by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_custom_role(
            custom_role_id: int = Field(..., description="Custom role ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_roles/{custom_role_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_custom_role, **kwargs)


class ZendeskCreateCustomRole(Tool):
    name: str = "zendesk_create_custom_role"
    description: str | None = "Create a Zendesk custom agent role."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_custom_role(
            custom_role: dict[str, Any] = Field(..., description="Custom role object (name, role_type required; description, configuration optional)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/custom_roles.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_role": custom_role},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_custom_role, **kwargs)


class ZendeskUpdateCustomRole(Tool):
    name: str = "zendesk_update_custom_role"
    description: str | None = "Update a Zendesk custom role."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_custom_role(
            custom_role_id: int = Field(..., description="Custom role ID"),
            custom_role: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/custom_roles/{custom_role_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_role": custom_role},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_custom_role, **kwargs)


class ZendeskDeleteCustomRole(Tool):
    name: str = "zendesk_delete_custom_role"
    description: str | None = "Delete a Zendesk custom role (only if no agents assigned)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_custom_role(
            custom_role_id: int = Field(..., description="Custom role ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/custom_roles/{custom_role_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_custom_role, **kwargs)


# --- Routing (Skill-based) ---


class ZendeskListAccountAttributes(Tool):
    name: str = "zendesk_list_account_attributes"
    description: str | None = "List Zendesk routing attributes (skill types)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_account_attributes(
            include: str | None = Field(None, description="Sideload attribute_values"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if include:
                params["include"] = include

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/attributes.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_account_attributes, **kwargs)


class ZendeskListRoutingAttributeDefinitions(Tool):
    name: str = "zendesk_list_routing_attribute_definitions"
    description: str | None = "List condition definitions for applying attributes to tickets."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_routing_attribute_definitions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/attributes/definitions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_routing_attribute_definitions, **kwargs)


class ZendeskShowRoutingAttribute(Tool):
    name: str = "zendesk_show_routing_attribute"
    description: str | None = "Show a Zendesk routing attribute by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_routing_attribute(
            attribute_id: str = Field(..., description="Attribute ID (UUID)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/attributes/{attribute_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_routing_attribute, **kwargs)


class ZendeskCreateRoutingAttribute(Tool):
    name: str = "zendesk_create_routing_attribute"
    description: str | None = "Create a Zendesk routing attribute (skill type)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_routing_attribute(
            name: str = Field(..., description="Attribute name"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/routing/attributes.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attribute": {"name": name}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_routing_attribute, **kwargs)


class ZendeskUpdateRoutingAttribute(Tool):
    name: str = "zendesk_update_routing_attribute"
    description: str | None = "Update a Zendesk routing attribute."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_routing_attribute(
            attribute_id: str = Field(..., description="Attribute ID"),
            attribute: dict[str, Any] = Field(..., description="Fields to update (e.g. name)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/routing/attributes/{attribute_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attribute": attribute},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_routing_attribute, **kwargs)


class ZendeskDeleteRoutingAttribute(Tool):
    name: str = "zendesk_delete_routing_attribute"
    description: str | None = "Delete a Zendesk routing attribute."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_routing_attribute(
            attribute_id: str = Field(..., description="Attribute ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/routing/attributes/{attribute_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_routing_attribute, **kwargs)


class ZendeskListAttributeValuesForAttribute(Tool):
    name: str = "zendesk_list_attribute_values_for_attribute"
    description: str | None = "List attribute values (skills) for a routing attribute."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_attribute_values_for_attribute(
            attribute_id: str = Field(..., description="Attribute ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/attributes/{attribute_id}/values.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_attribute_values_for_attribute, **kwargs)


class ZendeskShowRoutingAttributeValue(Tool):
    name: str = "zendesk_show_routing_attribute_value"
    description: str | None = "Show a Zendesk routing attribute value (skill) by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_routing_attribute_value(
            attribute_id: str = Field(..., description="Attribute ID"),
            attribute_value_id: str = Field(..., description="Attribute value ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/attributes/{attribute_id}/values/{attribute_value_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_routing_attribute_value, **kwargs)


class ZendeskCreateRoutingAttributeValue(Tool):
    name: str = "zendesk_create_routing_attribute_value"
    description: str | None = "Create a Zendesk routing attribute value (skill)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_routing_attribute_value(
            attribute_id: str = Field(..., description="Attribute ID"),
            attribute_value: dict[str, Any] = Field(..., description="Attribute value object (name required)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/routing/attributes/{attribute_id}/values.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attribute_value": attribute_value},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_routing_attribute_value, **kwargs)


class ZendeskUpdateRoutingAttributeValue(Tool):
    name: str = "zendesk_update_routing_attribute_value"
    description: str | None = "Update a Zendesk routing attribute value."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_routing_attribute_value(
            attribute_id: str = Field(..., description="Attribute ID"),
            attribute_value_id: str = Field(..., description="Attribute value ID"),
            attribute_value: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{base_url}/routing/attributes/{attribute_id}/values/{attribute_value_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attribute_value": attribute_value},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_routing_attribute_value, **kwargs)


class ZendeskDeleteRoutingAttributeValue(Tool):
    name: str = "zendesk_delete_routing_attribute_value"
    description: str | None = "Delete a Zendesk routing attribute value."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_routing_attribute_value(
            attribute_id: str = Field(..., description="Attribute ID"),
            attribute_value_id: str = Field(..., description="Attribute value ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/routing/attributes/{attribute_id}/values/{attribute_value_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_routing_attribute_value, **kwargs)


class ZendeskListAgentAttributeValues(Tool):
    name: str = "zendesk_list_agent_attribute_values"
    description: str | None = "List attribute values (skills) for an agent."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_agent_attribute_values(
            user_id: int = Field(..., description="User (agent) ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/agents/{user_id}/instance_values.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_agent_attribute_values, **kwargs)


class ZendeskListAttributeValuesForManyAgents(Tool):
    name: str = "zendesk_list_attribute_values_for_many_agents"
    description: str | None = "List attribute values for up to 100 agents."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_attribute_values_for_many_agents(
            agent_ids: str = Field(..., description="Comma-separated list of agent IDs (max 100)"),
            page_size: int | None = Field(None, description="Results per page"),
            page_after: str | None = Field(None, description="Cursor for next page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"filter[agent_ids]": agent_ids}
            if page_size is not None:
                params["page[size]"] = page_size
            if page_after:
                params["page[after]"] = page_after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/agents/instance_values.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_attribute_values_for_many_agents, **kwargs)


class ZendeskSetAgentAttributeValues(Tool):
    name: str = "zendesk_set_agent_attribute_values"
    description: str | None = "Set or replace attribute values (skills) for an agent."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_agent_attribute_values(
            user_id: int = Field(..., description="User (agent) ID"),
            attribute_value_ids: list[str] = Field(..., description="List of attribute value IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/routing/agents/{user_id}/instance_values.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attribute_value_ids": attribute_value_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_agent_attribute_values, **kwargs)


class ZendeskSetTicketAttributeValues(Tool):
    name: str = "zendesk_set_ticket_attribute_values"
    description: str | None = "Set attribute values (skills) for a ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_ticket_attribute_values(
            ticket_id: int = Field(..., description="Ticket ID"),
            attribute_value_ids: list[str] = Field(..., description="List of attribute value IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/routing/tickets/{ticket_id}/instance_values.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attribute_value_ids": attribute_value_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_ticket_attribute_values, **kwargs)


class ZendeskBulkSetAgentAttributeValuesJob(Tool):
    name: str = "zendesk_bulk_set_agent_attribute_values_job"
    description: str | None = "Bulk add, replace, or remove attribute values for up to 100 agents (async job)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_set_agent_attribute_values_job(
            job: dict[str, Any] = Field(
                ...,
                description="Job object: action (upsert|update|delete), attributes.attribute_values, items (agent ids)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/routing/agents/instance_values/jobs.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"job": job},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_set_agent_attribute_values_job, **kwargs)


class ZendeskListTicketsFulfilledByUser(Tool):
    name: str = "zendesk_list_tickets_fulfilled_by_user"
    description: str | None = "List ticket IDs that match the current user's attributes."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tickets_fulfilled_by_user(
            ticket_ids: str = Field(..., description="Comma-separated ticket IDs to check"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            ids = [x.strip() for x in ticket_ids.split(",")]
            params = [("ticket_ids[]", tid) for tid in ids]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/requirements/fulfilled.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_tickets_fulfilled_by_user, **kwargs)


class ZendeskListTicketAttributeValues(Tool):
    name: str = "zendesk_list_ticket_attribute_values"
    description: str | None = "List attribute values (skills) for a ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_attribute_values(
            ticket_id: int = Field(..., description="Ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/tickets/{ticket_id}/instance_values.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_attribute_values, **kwargs)


# --- Reseller (uses zendeskaccounts.zendesk.com) ---


class ZendeskCreateTrialAccount(Tool):
    name: str = "zendesk_create_trial_account"
    description: str | None = "Create a Zendesk trial account via Reseller API. Requires reseller token."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None
    reseller_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
                "reseller_token": self.reseller_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_trial_account(
            account: dict[str, Any] = Field(..., description="Account object (name, subdomain, help_desk_size)"),
            owner: dict[str, Any] = Field(..., description="Owner object (name, email, password)"),
            address: dict[str, Any] = Field(..., description="Address object (phone)"),
            partner: dict[str, Any] | None = Field(None, description="Partner object (name, url)"),
            language: str | None = Field(None, description="Language code (e.g. en-us)"),
            utc_offset: int | None = Field(None, description="UTC offset (e.g. -8)"),
        ) -> Any:
            token = (
                self.reseller_token.get_secret_value()
                if self.reseller_token
                else os.getenv("ZENDESK_RESELLER_TOKEN")
            )
            if not token:
                raise ValueError("Zendesk reseller token required. Set ZENDESK_RESELLER_TOKEN or pass reseller_token.")
            import httpx

            payload: dict[str, Any] = {"account": account, "owner": owner, "address": address}
            if partner:
                payload["partner"] = partner
            if language:
                payload["language"] = language
            if utc_offset is not None:
                payload["utc_offset"] = utc_offset

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://zendeskaccounts.zendesk.com/api/v2/accounts",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_trial_account, **kwargs)


class ZendeskVerifySubdomainAvailability(Tool):
    name: str = "zendesk_verify_subdomain_availability"
    description: str | None = "Verify if a Zendesk subdomain is available. No auth required."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _verify_subdomain_availability(
            subdomain: str = Field(..., description="Subdomain to check (min 3 chars, no underscores/hyphens/spaces)"),
        ) -> Any:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://zendeskaccounts.zendesk.com/api/v2/accounts/available",
                    params={"subdomain": subdomain},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_verify_subdomain_availability, **kwargs)


# --- Resource Collections ---


class ZendeskListResourceCollections(Tool):
    name: str = "zendesk_list_resource_collections"
    description: str | None = "List Zendesk resource collections."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_resource_collections(
            per_page: int = Field(50, description="Results per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/resource_collections.json",
                    headers={"Authorization": auth},
                    params={"per_page": per_page},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_resource_collections, **kwargs)


class ZendeskShowResourceCollection(Tool):
    name: str = "zendesk_show_resource_collection"
    description: str | None = "Show a Zendesk resource collection by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_resource_collection(
            resource_collection_id: int = Field(..., description="Resource collection ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/resource_collections/{resource_collection_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_resource_collection, **kwargs)


class ZendeskCreateResourceCollection(Tool):
    name: str = "zendesk_create_resource_collection"
    description: str | None = "Create a Zendesk resource collection from a payload (like app requirements.json)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_resource_collection(
            payload: dict[str, Any] = Field(..., description="Payload object (ticket_fields, triggers, targets, etc.)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/resource_collections.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"payload": payload},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_resource_collection, **kwargs)


class ZendeskUpdateResourceCollection(Tool):
    name: str = "zendesk_update_resource_collection"
    description: str | None = "Update a Zendesk resource collection."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_resource_collection(
            resource_collection_id: int = Field(..., description="Resource collection ID"),
            payload: dict[str, Any] = Field(..., description="Payload object with resources to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/resource_collections/{resource_collection_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"payload": payload},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_resource_collection, **kwargs)


class ZendeskDeleteResourceCollection(Tool):
    name: str = "zendesk_delete_resource_collection"
    description: str | None = "Delete a Zendesk resource collection."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_resource_collection(
            resource_collection_id: int = Field(..., description="Resource collection ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/resource_collections/{resource_collection_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_resource_collection, **kwargs)


# --- Requests (end-user ticket view) ---


class ZendeskListRequests(Tool):
    name: str = "zendesk_list_requests"
    description: str | None = "List Zendesk requests (end-user view of tickets)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_requests(
            per_page: int = Field(50, description="Results per page"),
            page: int = Field(1, description="Page number"),
            sort_by: str = Field("updated_at", description="updated_at or created_at"),
            sort_order: str = Field("asc", description="asc or desc"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page, "sort_by": sort_by, "sort_order": sort_order}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/requests.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_requests, **kwargs)


class ZendeskSearchRequests(Tool):
    name: str = "zendesk_search_requests"
    description: str | None = "Search Zendesk requests by query."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_requests(
            query: str = Field(..., description="Search query"),
            status: str | None = Field(None, description="Filter by status (e.g. hold,open)"),
            cc_id: bool | None = Field(None, description="Filter by CC"),
            organization_id: int | None = Field(None, description="Filter by organization"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query}
            if status:
                params["status"] = status
            if cc_id is not None:
                params["cc_id"] = str(cc_id).lower()
            if organization_id is not None:
                params["organization_id"] = organization_id

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/requests/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_requests, **kwargs)


class ZendeskShowRequest(Tool):
    name: str = "zendesk_show_request"
    description: str | None = "Show a Zendesk request by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_request(
            request_id: int = Field(..., description="Request ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/requests/{request_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_request, **kwargs)


class ZendeskCreateRequest(Tool):
    name: str = "zendesk_create_request"
    description: str | None = "Create a Zendesk request (end-user ticket)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_request(
            request: dict[str, Any] = Field(..., description="Request object (subject, comment required; requester for anonymous)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/requests.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"request": request},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_request, **kwargs)


class ZendeskUpdateRequest(Tool):
    name: str = "zendesk_update_request"
    description: str | None = "Update a Zendesk request (add comment, mark solved, add collaborators)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_request(
            request_id: int = Field(..., description="Request ID"),
            request: dict[str, Any] = Field(..., description="Fields to update (comment, solved, additional_collaborators)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/requests/{request_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"request": request},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_request, **kwargs)


class ZendeskListRequestComments(Tool):
    name: str = "zendesk_list_request_comments"
    description: str | None = "List comments for a Zendesk request."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_request_comments(
            request_id: int = Field(..., description="Request ID"),
            role: str | None = Field(None, description="Filter: agent or end_user"),
            since: str | None = Field(None, description="Filter comments from datetime"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if role:
                params["role"] = role
            if since:
                params["since"] = since

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/requests/{request_id}/comments.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_request_comments, **kwargs)


class ZendeskShowRequestComment(Tool):
    name: str = "zendesk_show_request_comment"
    description: str | None = "Show a specific comment on a Zendesk request."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_request_comment(
            request_id: int = Field(..., description="Request ID"),
            ticket_comment_id: int = Field(..., description="Comment ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/requests/{request_id}/comments/{ticket_comment_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_request_comment, **kwargs)


# --- Satisfaction Ratings ---


class ZendeskListSatisfactionRatings(Tool):
    name: str = "zendesk_list_satisfaction_ratings"
    description: str | None = "List Zendesk satisfaction ratings (legacy CSAT)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_satisfaction_ratings(
            score: str | None = Field(None, description="Filter: good, bad, offered, unoffered, etc."),
            start_time: int | None = Field(None, description="Unix epoch - oldest rating"),
            end_time: int | None = Field(None, description="Unix epoch - most recent rating"),
            per_page: int = Field(50, description="Results per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page}
            if score:
                params["score"] = score
            if start_time is not None:
                params["start_time"] = start_time
            if end_time is not None:
                params["end_time"] = end_time

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/satisfaction_ratings.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_satisfaction_ratings, **kwargs)


class ZendeskCountSatisfactionRatings(Tool):
    name: str = "zendesk_count_satisfaction_ratings"
    description: str | None = "Count Zendesk satisfaction ratings."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_satisfaction_ratings() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/satisfaction_ratings/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_satisfaction_ratings, **kwargs)


class ZendeskShowSatisfactionRating(Tool):
    name: str = "zendesk_show_satisfaction_rating"
    description: str | None = "Show a Zendesk satisfaction rating by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_satisfaction_rating(
            satisfaction_rating_id: int = Field(..., description="Satisfaction rating ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/satisfaction_ratings/{satisfaction_rating_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_satisfaction_rating, **kwargs)


class ZendeskCreateSatisfactionRating(Tool):
    name: str = "zendesk_create_satisfaction_rating"
    description: str | None = "Create a satisfaction rating for a solved ticket (requester only)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_satisfaction_rating(
            ticket_id: int = Field(..., description="Ticket ID"),
            satisfaction_rating: dict[str, Any] = Field(..., description="Rating object (score: good/bad, comment, reason_code for bad)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/tickets/{ticket_id}/satisfaction_rating.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"satisfaction_rating": satisfaction_rating},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_satisfaction_rating, **kwargs)


class ZendeskListSatisfactionReasons(Tool):
    name: str = "zendesk_list_satisfaction_reasons"
    description: str | None = "List reasons for negative satisfaction ratings."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_satisfaction_reasons() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/satisfaction_reasons.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_satisfaction_reasons, **kwargs)


class ZendeskShowSatisfactionReason(Tool):
    name: str = "zendesk_show_satisfaction_reason"
    description: str | None = "Show a satisfaction reason by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_satisfaction_reason(
            satisfaction_reason_id: int = Field(..., description="Satisfaction reason ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/satisfaction_reasons/{satisfaction_reason_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_satisfaction_reason, **kwargs)


# --- Password ---


class ZendeskChangePassword(Tool):
    name: str = "zendesk_change_password"
    description: str | None = "Change your own Zendesk password."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _change_password(
            user_id: int = Field(..., description="Your user ID"),
            previous_password: str = Field(..., description="Current password"),
            password: str = Field(..., description="New password"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/password.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"previous_password": previous_password, "password": password},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_change_password, **kwargs)


class ZendeskSetUserPassword(Tool):
    name: str = "zendesk_set_user_password"
    description: str | None = "Set a user's password (admin only, must be enabled in settings)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_user_password(
            user_id: int = Field(..., description="User ID"),
            password: str = Field(..., description="New password"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/users/{user_id}/password.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"password": password},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_user_password, **kwargs)


class ZendeskListPasswordRequirements(Tool):
    name: str = "zendesk_list_password_requirements"
    description: str | None = "List password requirements for a user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_password_requirements(
            user_id: int = Field(..., description="User ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_id}/password/requirements.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_password_requirements, **kwargs)


# --- Queues (Omnichannel Routing) ---


class ZendeskListQueues(Tool):
    name: str = "zendesk_list_queues"
    description: str | None = "List Zendesk omnichannel routing queues."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_queues() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/queues.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_queues, **kwargs)


class ZendeskShowQueue(Tool):
    name: str = "zendesk_show_queue"
    description: str | None = "Show a Zendesk omnichannel routing queue by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_queue(
            queue_id: str = Field(..., description="Queue ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/queues/{queue_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_queue, **kwargs)


class ZendeskCreateQueue(Tool):
    name: str = "zendesk_create_queue"
    description: str | None = "Create a Zendesk omnichannel routing queue."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_queue(
            queue: dict[str, Any] = Field(..., description="Queue object (name, description, definition, primary_groups_id, etc.)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/queues.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"queue": queue},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_queue, **kwargs)


class ZendeskUpdateQueue(Tool):
    name: str = "zendesk_update_queue"
    description: str | None = "Update a Zendesk omnichannel routing queue."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_queue(
            queue_id: str = Field(..., description="Queue ID"),
            queue: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/queues/{queue_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"queue": queue},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_queue, **kwargs)


class ZendeskDeleteQueue(Tool):
    name: str = "zendesk_delete_queue"
    description: str | None = "Delete a Zendesk omnichannel routing queue."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_queue(
            queue_id: str = Field(..., description="Queue ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/queues/{queue_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_queue, **kwargs)


class ZendeskListQueueDefinitions(Tool):
    name: str = "zendesk_list_queue_definitions"
    description: str | None = "List queue definitions and condition definitions."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_queue_definitions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/queues/definitions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_queue_definitions, **kwargs)


class ZendeskReorderQueues(Tool):
    name: str = "zendesk_reorder_queues"
    description: str | None = "Reorder Zendesk omnichannel routing queues."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_queues(
            queue_ids: list[str] = Field(..., description="Queue IDs in desired order (must include every queue id)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{base_url}/queues/order",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"queue_ids": queue_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json() if response.content else {"status": "ok"}

        super().__init__(handler=_reorder_queues, **kwargs)


# --- Relationship (Lookup) ---


class ZendeskGetSourcesByTarget(Tool):
    name: str = "zendesk_get_sources_by_target"
    description: str | None = (
        "Get source objects whose lookup relationship field references a target object. "
        "E.g., tickets where a user is the 'Success Manager'."
    )
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sources_by_target(
            target_type: str = Field(
                ...,
                description="Target type: zen:user, zen:ticket, zen:organization, or zen:custom_object:CUSTOM_OBJECT_KEY",
            ),
            target_id: int = Field(..., description="Target object ID"),
            field_id: int = Field(..., description="Lookup relationship field ID"),
            source_type: str = Field(
                ...,
                description="Source type: zen:user, zen:ticket, zen:organization, or zen:custom_object:CUSTOM_OBJECT_KEY",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/{target_type}/{target_id}/relationship_fields/{field_id}/{source_type}",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_sources_by_target, **kwargs)


class ZendeskListRelationshipFilterDefinitions(Tool):
    name: str = "zendesk_list_relationship_filter_definitions"
    description: str | None = (
        "List filter definitions for relationship filters by target type. "
        "Used to build relationship_filter for custom/ticket fields."
    )
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_filter_definitions(
            target_type: str = Field(
                ...,
                description="Target type: zen:user, zen:ticket, zen:organization, or zen:custom_object:CUSTOM_OBJECT_KEY",
            ),
            source_type: str | None = Field(
                None,
                description="Optional source type: zen:user, zen:ticket, zen:organization",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if source_type is not None:
                params["source_type"] = source_type

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/relationships/definitions/{target_type}",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_filter_definitions, **kwargs)


# --- Custom Object Records ---


class ZendeskListCustomObjectRecords(Tool):
    name: str = "zendesk_list_custom_object_records"
    description: str | None = "List custom object records for a custom object."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_custom_object_records(
            custom_object_key: str = Field(..., description="Custom object key"),
            filter_ids: str | None = Field(None, description="Comma-separated record IDs"),
            filter_external_ids: str | None = Field(None, description="Comma-separated external IDs"),
            page_size: int = Field(100, description="Records per page"),
            page_after: str | None = Field(None, description="Cursor for next page"),
            sort: str = Field("id", description="id, updated_at, -id, -updated_at"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"page[size]": page_size, "sort": sort}
            if filter_ids:
                params["filter[ids]"] = filter_ids
            if filter_external_ids:
                params["filter[external_ids]"] = filter_external_ids
            if page_after:
                params["page[after]"] = page_after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/{custom_object_key}/records.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_custom_object_records, **kwargs)


class ZendeskShowCustomObjectRecord(Tool):
    name: str = "zendesk_show_custom_object_record"
    description: str | None = "Show a custom object record by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_custom_object_record(
            custom_object_key: str = Field(..., description="Custom object key"),
            custom_object_record_id: str = Field(..., description="Record ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/{custom_object_key}/records/{custom_object_record_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_custom_object_record, **kwargs)


class ZendeskCreateCustomObjectRecord(Tool):
    name: str = "zendesk_create_custom_object_record"
    description: str | None = "Create a custom object record."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_custom_object_record(
            custom_object_key: str = Field(..., description="Custom object key"),
            custom_object_record: dict[str, Any] = Field(..., description="Record (name, custom_object_fields)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/custom_objects/{custom_object_key}/records.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_object_record": custom_object_record},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_custom_object_record, **kwargs)


class ZendeskUpdateCustomObjectRecord(Tool):
    name: str = "zendesk_update_custom_object_record"
    description: str | None = "Update a custom object record."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_custom_object_record(
            custom_object_key: str = Field(..., description="Custom object key"),
            custom_object_record_id: str = Field(..., description="Record ID"),
            custom_object_record: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{base_url}/custom_objects/{custom_object_key}/records/{custom_object_record_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_object_record": custom_object_record},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_custom_object_record, **kwargs)


class ZendeskDeleteCustomObjectRecord(Tool):
    name: str = "zendesk_delete_custom_object_record"
    description: str | None = "Delete a custom object record by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_custom_object_record(
            custom_object_key: str = Field(..., description="Custom object key"),
            custom_object_record_id: str = Field(..., description="Record ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/custom_objects/{custom_object_key}/records/{custom_object_record_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_custom_object_record, **kwargs)


class ZendeskSetCustomObjectRecordByExternalId(Tool):
    name: str = "zendesk_set_custom_object_record_by_external_id"
    description: str | None = "Create or update a custom object record by external ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_custom_object_record_by_external_id(
            custom_object_key: str = Field(..., description="Custom object key"),
            external_id: str = Field(..., description="External ID"),
            custom_object_record: dict[str, Any] = Field(..., description="Record data"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{base_url}/custom_objects/{custom_object_key}/records.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params={"external_id": external_id},
                    json={"custom_object_record": custom_object_record},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_custom_object_record_by_external_id, **kwargs)


class ZendeskDeleteCustomObjectRecordByExternalId(Tool):
    name: str = "zendesk_delete_custom_object_record_by_external_id"
    description: str | None = "Delete a custom object record by external ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_custom_object_record_by_external_id(
            custom_object_key: str = Field(..., description="Custom object key"),
            external_id: str = Field(..., description="External ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/custom_objects/{custom_object_key}/records.json",
                    headers={"Authorization": auth},
                    params={"external_id": external_id},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_custom_object_record_by_external_id, **kwargs)


class ZendeskSearchCustomObjectRecords(Tool):
    name: str = "zendesk_search_custom_object_records"
    description: str | None = "Search custom object records by query (text fields)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_custom_object_records(
            custom_object_key: str = Field(..., description="Custom object key"),
            query: str = Field(..., description="Search query"),
            page_size: int = Field(100, description="Records per page"),
            page_after: str | None = Field(None, description="Cursor"),
            sort: str | None = Field(None, description="name, created_at, updated_at, -name, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query, "page[size]": page_size}
            if page_after:
                params["page[after]"] = page_after
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/{custom_object_key}/records/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_custom_object_records, **kwargs)


class ZendeskCountCustomObjectRecords(Tool):
    name: str = "zendesk_count_custom_object_records"
    description: str | None = "Count custom object records for a custom object."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_custom_object_records(
            custom_object_key: str = Field(..., description="Custom object key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/{custom_object_key}/records/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_custom_object_records, **kwargs)


class ZendeskAutocompleteCustomObjectRecords(Tool):
    name: str = "zendesk_autocomplete_custom_object_records"
    description: str | None = "Autocomplete custom object records by name."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _autocomplete_custom_object_records(
            custom_object_key: str = Field(..., description="Custom object key"),
            name: str = Field(..., description="Part of record name to match"),
            page_size: int = Field(100, description="Records per page"),
            page_after: str | None = Field(None, description="Cursor"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"name": name, "page[size]": page_size}
            if page_after:
                params["page[after]"] = page_after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/{custom_object_key}/records/autocomplete.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_autocomplete_custom_object_records, **kwargs)


# --- Brand ---


class ZendeskListBrands(Tool):
    name: str = "zendesk_list_brands"
    description: str | None = "List Zendesk brands."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_brands(
            sort: str | None = Field(None, description="Sort field e.g. name, -created_at"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, str] = {}
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/brands.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_brands, **kwargs)


class ZendeskShowBrand(Tool):
    name: str = "zendesk_show_brand"
    description: str | None = "Show a Zendesk brand by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_brand(
            brand_id: int = Field(..., description="Brand ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/brands/{brand_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_brand, **kwargs)


class ZendeskCreateBrand(Tool):
    name: str = "zendesk_create_brand"
    description: str | None = "Create a Zendesk brand."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_brand(
            brand: dict[str, Any] = Field(
                ...,
                description="Brand object: name, subdomain (required)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/brands.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"brand": brand},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_brand, **kwargs)


class ZendeskUpdateBrand(Tool):
    name: str = "zendesk_update_brand"
    description: str | None = "Update a Zendesk brand."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_brand(
            brand_id: int = Field(..., description="Brand ID"),
            brand: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/brands/{brand_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"brand": brand},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_brand, **kwargs)


class ZendeskDeleteBrand(Tool):
    name: str = "zendesk_delete_brand"
    description: str | None = "Delete a Zendesk brand."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_brand(
            brand_id: int = Field(..., description="Brand ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/brands/{brand_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_brand, **kwargs)


class ZendeskCheckHostMappingValidity(Tool):
    name: str = "zendesk_check_host_mapping_validity"
    description: str | None = "Check if a host mapping is valid for a subdomain."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check(
            host_mapping: str = Field(..., description="Host mapping to check"),
            subdomain_param: str = Field(..., description="Subdomain for the account"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/brands/check_host_mapping.json",
                    headers={"Authorization": auth},
                    params={"host_mapping": host_mapping, "subdomain": subdomain_param},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_check, **kwargs)


class ZendeskCheckHostMappingValidityForExistingBrand(Tool):
    name: str = "zendesk_check_host_mapping_validity_for_existing_brand"
    description: str | None = "Check if host mapping is valid for an existing brand."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check(
            brand_id: int = Field(..., description="Brand ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/brands/{brand_id}/check_host_mapping.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_check, **kwargs)


# --- Organizations ---


class ZendeskListOrganizations(Tool):
    name: str = "zendesk_list_organizations"
    description: str | None = "List Zendesk organizations."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_organizations(
            per_page: int | None = Field(None, description="Records per page"),
            sort: str | None = Field(None, description="Sort field (prefix with - for descending)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if per_page is not None:
                params["per_page"] = per_page
            if sort is not None:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_organizations, **kwargs)


class ZendeskShowOrganization(Tool):
    name: str = "zendesk_show_organization"
    description: str | None = "Show a Zendesk organization by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_organization(
            organization_id: int = Field(..., description="Organization ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/{organization_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_organization, **kwargs)


class ZendeskShowManyOrganizations(Tool):
    name: str = "zendesk_show_many_organizations"
    description: str | None = "Show many Zendesk organizations by IDs or external IDs."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many_organizations(
            ids: str | None = Field(None, description="Comma-separated organization IDs"),
            external_ids: str | None = Field(None, description="Comma-separated external IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if ids is not None:
                params["ids"] = ids
            if external_ids is not None:
                params["external_ids"] = external_ids

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/show_many.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many_organizations, **kwargs)


class ZendeskSearchOrganizations(Tool):
    name: str = "zendesk_search_organizations"
    description: str | None = "Search Zendesk organizations by name or external_id."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_organizations(
            name: str | None = Field(None, description="Exact organization name"),
            external_id: str | None = Field(None, description="Exact external ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if name is not None:
                params["name"] = name
            if external_id is not None:
                params["external_id"] = external_id

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_organizations, **kwargs)


class ZendeskShowOrganizationRelatedInformation(Tool):
    name: str = "zendesk_show_organization_related_information"
    description: str | None = "Show related information for a Zendesk organization (tickets count, users count)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_organization_related(
            organization_id: int = Field(..., description="Organization ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/{organization_id}/related.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_organization_related, **kwargs)


class ZendeskCreateOrganization(Tool):
    name: str = "zendesk_create_organization"
    description: str | None = "Create a Zendesk organization."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_organization(
            organization: dict[str, Any] = Field(..., description="Organization object (name required)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organizations.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization": organization},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_organization, **kwargs)


class ZendeskCreateManyOrganizations(Tool):
    name: str = "zendesk_create_many_organizations"
    description: str | None = "Create many Zendesk organizations (up to 100). Returns job_status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_many_organizations(
            organizations: list[dict[str, Any]] = Field(..., description="List of organization objects"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organizations/create_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organizations": organizations},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_many_organizations, **kwargs)


class ZendeskCreateOrUpdateOrganization(Tool):
    name: str = "zendesk_create_or_update_organization"
    description: str | None = "Create or update a Zendesk organization by id or external_id."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_or_update_organization(
            organization: dict[str, Any] = Field(
                ...,
                description="Organization object (name required; id or external_id for update)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organizations/create_or_update.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization": organization},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_or_update_organization, **kwargs)


class ZendeskUpdateOrganization(Tool):
    name: str = "zendesk_update_organization"
    description: str | None = "Update a Zendesk organization."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_organization(
            organization_id: int = Field(..., description="Organization ID"),
            organization: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/organizations/{organization_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization": organization},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_organization, **kwargs)


class ZendeskUpdateManyOrganizations(Tool):
    name: str = "zendesk_update_many_organizations"
    description: str | None = "Bulk or batch update Zendesk organizations (up to 100). Returns job_status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_many_organizations(
            organization: dict[str, Any] | None = Field(None, description="Same update for all (use with ids)"),
            organizations: list[dict[str, Any]] | None = Field(None, description="Different updates per org"),
            ids: str | None = Field(None, description="Comma-separated org IDs (for bulk)"),
            external_ids: str | None = Field(None, description="Comma-separated external IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if ids is not None:
                params["ids"] = ids
            if external_ids is not None:
                params["external_ids"] = external_ids

            payload: dict[str, Any] = {}
            if organization is not None:
                payload["organization"] = organization
            if organizations is not None:
                payload["organizations"] = organizations

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/organizations/update_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params or None,
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_many_organizations, **kwargs)


class ZendeskDeleteOrganization(Tool):
    name: str = "zendesk_delete_organization"
    description: str | None = "Delete a Zendesk organization."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_organization(
            organization_id: int = Field(..., description="Organization ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organizations/{organization_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_organization, **kwargs)


class ZendeskBulkDeleteOrganizations(Tool):
    name: str = "zendesk_bulk_delete_organizations"
    description: str | None = "Bulk delete Zendesk organizations (up to 100). Returns job_status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_organizations(
            ids: str | None = Field(None, description="Comma-separated organization IDs"),
            external_ids: str | None = Field(None, description="Comma-separated external IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if ids is not None:
                params["ids"] = ids
            if external_ids is not None:
                params["external_ids"] = external_ids

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organizations/destroy_many.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_delete_organizations, **kwargs)


class ZendeskCountOrganizations(Tool):
    name: str = "zendesk_count_organizations"
    description: str | None = "Count Zendesk organizations."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_organizations() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_organizations, **kwargs)


class ZendeskAutocompleteOrganizations(Tool):
    name: str = "zendesk_autocomplete_organizations"
    description: str | None = "Autocomplete Zendesk organizations by name prefix."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _autocomplete_organizations(
            name: str = Field(..., description="Name prefix to search"),
            field_id: str | None = Field(None, description="Lookup relationship field ID"),
            source: str | None = Field(None, description="Field source type (e.g. zen:user)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"name": name}
            if field_id is not None:
                params["field_id"] = field_id
            if source is not None:
                params["source"] = source

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organizations/autocomplete.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_autocomplete_organizations, **kwargs)


class ZendeskMergeOrganizationWithAnotherOrganization(Tool):
    name: str = "zendesk_merge_organization_with_another_organization"
    description: str | None = "Merge two Zendesk organizations. Source org is deleted; users/tickets move to winner."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _merge_organization(
            organization_id: int = Field(..., description="Organization ID to merge (loser)"),
            winner_id: int = Field(..., description="Organization ID that will remain (winner)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organizations/{organization_id}/merge.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_merge": {"winner_id": winner_id}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_merge_organization, **kwargs)


class ZendeskShowOrganizationMerge(Tool):
    name: str = "zendesk_show_organization_merge"
    description: str | None = "Show status of a Zendesk organization merge."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_organization_merge(
            organization_merge_id: str = Field(..., description="Organization merge ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_merges/{organization_merge_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_organization_merge, **kwargs)


class ZendeskListOrganizationMerges(Tool):
    name: str = "zendesk_list_organization_merges"
    description: str | None = "List Zendesk organization merges."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_organization_merges() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_merges.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_organization_merges, **kwargs)


# --- Custom Objects (schema) ---


class ZendeskListCustomObjects(Tool):
    name: str = "zendesk_list_custom_objects"
    description: str | None = "List Zendesk custom object definitions (schemas)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_custom_objects() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_custom_objects, **kwargs)


class ZendeskShowCustomObject(Tool):
    name: str = "zendesk_show_custom_object"
    description: str | None = "Show a Zendesk custom object definition by key."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_custom_object(
            custom_object_key: str = Field(..., description="Custom object key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/{custom_object_key}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_custom_object, **kwargs)


class ZendeskCreateCustomObject(Tool):
    name: str = "zendesk_create_custom_object"
    description: str | None = "Create a Zendesk custom object definition."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_custom_object(
            custom_object: dict[str, Any] = Field(..., description="Custom object schema (key, title, custom_object_fields)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/custom_objects.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_object": custom_object},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_custom_object, **kwargs)


class ZendeskUpdateCustomObject(Tool):
    name: str = "zendesk_update_custom_object"
    description: str | None = "Update a Zendesk custom object definition."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_custom_object(
            custom_object_key: str = Field(..., description="Custom object key"),
            custom_object: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/custom_objects/{custom_object_key}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"custom_object": custom_object},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_custom_object, **kwargs)


class ZendeskDeleteCustomObject(Tool):
    name: str = "zendesk_delete_custom_object"
    description: str | None = "Delete a Zendesk custom object definition."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_custom_object(
            custom_object_key: str = Field(..., description="Custom object key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/custom_objects/{custom_object_key}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_custom_object, **kwargs)


class ZendeskCustomObjectsLimit(Tool):
    name: str = "zendesk_custom_objects_limit"
    description: str | None = "Get Zendesk custom objects limit for the account."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _custom_objects_limit() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/custom_objects/limit.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_custom_objects_limit, **kwargs)


# --- OAuth ---


class ZendeskListOauthClients(Tool):
    name: str = "zendesk_list_oauth_clients"
    description: str | None = "List Zendesk OAuth clients."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_oauth_clients() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/oauth/clients.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_oauth_clients, **kwargs)


class ZendeskListGlobalOauthClients(Tool):
    name: str = "zendesk_list_global_oauth_clients"
    description: str | None = "List global OAuth clients."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_global_oauth_clients() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/oauth/global_clients.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_global_oauth_clients, **kwargs)


class ZendeskShowOauthClient(Tool):
    name: str = "zendesk_show_oauth_client"
    description: str | None = "Show a Zendesk OAuth client by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_oauth_client(
            oauth_client_id: int = Field(..., description="OAuth client ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/oauth/clients/{oauth_client_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_oauth_client, **kwargs)


class ZendeskCreateOauthClient(Tool):
    name: str = "zendesk_create_oauth_client"
    description: str | None = "Create a Zendesk OAuth client."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_oauth_client(
            client_data: dict[str, Any] = Field(
                ...,
                description="Client object (name, identifier, kind; user_id, redirect_uri, etc.)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/oauth/clients.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"client": client_data},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_oauth_client, **kwargs)


class ZendeskUpdateOauthClient(Tool):
    name: str = "zendesk_update_oauth_client"
    description: str | None = "Update a Zendesk OAuth client."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_oauth_client(
            oauth_client_id: int = Field(..., description="OAuth client ID"),
            client: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/oauth/clients/{oauth_client_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"client": client},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_oauth_client, **kwargs)


class ZendeskDeleteOauthClient(Tool):
    name: str = "zendesk_delete_oauth_client"
    description: str | None = "Delete a Zendesk OAuth client."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_oauth_client(
            oauth_client_id: int = Field(..., description="OAuth client ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/oauth/clients/{oauth_client_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_oauth_client, **kwargs)


class ZendeskGenerateOauthClientSecret(Tool):
    name: str = "zendesk_generate_oauth_client_secret"
    description: str | None = "Generate a new secret for a Zendesk OAuth client."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _generate_secret(
            oauth_client_id: int = Field(..., description="OAuth client ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/oauth/clients/{oauth_client_id}/generate_secret.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_generate_secret, **kwargs)


class ZendeskListOauthTokens(Tool):
    name: str = "zendesk_list_oauth_tokens"
    description: str | None = "List Zendesk OAuth tokens."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_oauth_tokens(
            all: bool = Field(False, description="Return all tokens in account (admin only)"),
            client_id: int | None = Field(None, description="Filter by OAuth client ID"),
            global_client_id: int | None = Field(None, description="Filter by global OAuth client ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if all:
                params["all"] = "true"
            if client_id is not None:
                params["client_id"] = client_id
            if global_client_id is not None:
                params["global_client_id"] = global_client_id

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/oauth/tokens.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_oauth_tokens, **kwargs)


class ZendeskShowOauthToken(Tool):
    name: str = "zendesk_show_oauth_token"
    description: str | None = "Show a Zendesk OAuth token by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_oauth_token(
            oauth_token_id: int = Field(..., description="OAuth token ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/oauth/tokens/{oauth_token_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_oauth_token, **kwargs)


class ZendeskCreateOauthToken(Tool):
    name: str = "zendesk_create_oauth_token"
    description: str | None = "Create a Zendesk OAuth token (client credentials)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_oauth_token(
            token: dict[str, Any] = Field(
                ...,
                description="Token object (client_id, scopes e.g. ['read','write'])",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/oauth/tokens.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"token": token},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_oauth_token, **kwargs)


class ZendeskRevokeOauthToken(Tool):
    name: str = "zendesk_revoke_oauth_token"
    description: str | None = "Revoke a Zendesk OAuth token."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _revoke_oauth_token(
            oauth_token_id: int = Field(..., description="OAuth token ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/oauth/tokens/{oauth_token_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_revoke_oauth_token, **kwargs)


class ZendeskCreateTokenForGrantType(Tool):
    name: str = "zendesk_create_token_for_grant_type"
    description: str | None = (
        "Create OAuth token for grant type (authorization_code, refresh_token, or client_credentials). "
        "Uses /oauth/tokens (not /api/v2)."
    )
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_token_for_grant_type(
            grant_type: str = Field(
                ...,
                description="authorization_code, refresh_token, or client_credentials",
            ),
            client_id: str = Field(..., description="OAuth client unique identifier"),
            client_secret: str | None = Field(None, description="Client secret (for confidential clients)"),
            code: str | None = Field(None, description="Authorization code (for authorization_code grant)"),
            redirect_uri: str | None = Field(None, description="Redirect URI (for authorization_code grant)"),
            refresh_token: str | None = Field(None, description="Refresh token (for refresh_token grant)"),
            scope: str | None = Field(None, description="Space-separated scopes (e.g. 'read write')"),
            code_verifier: str | None = Field(None, description="PKCE code verifier (for authorization_code)"),
            expires_in: int | None = Field(None, description="Access token expiry in seconds"),
            refresh_token_expires_in: int | None = Field(None, description="Refresh token expiry in seconds"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            oauth_base = base_url.replace("/api/v2", "")
            payload: dict[str, Any] = {"grant_type": grant_type, "client_id": client_id}
            if client_secret is not None:
                payload["client_secret"] = client_secret
            if code is not None:
                payload["code"] = code
            if redirect_uri is not None:
                payload["redirect_uri"] = redirect_uri
            if refresh_token is not None:
                payload["refresh_token"] = refresh_token
            if scope is not None:
                payload["scope"] = scope
            if code_verifier is not None:
                payload["code_verifier"] = code_verifier
            if expires_in is not None:
                payload["expires_in"] = expires_in
            if refresh_token_expires_in is not None:
                payload["refresh_token_expires_in"] = refresh_token_expires_in

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{oauth_base}/oauth/tokens",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_token_for_grant_type, **kwargs)


# --- Ticket Metrics ---


class ZendeskShowTicketMetrics(Tool):
    name: str = "zendesk_show_ticket_metrics"
    description: str | None = "Show ticket metrics by metric ID or by ticket ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_ticket_metrics(
            ticket_id: int | None = Field(None, description="Ticket ID (use this or ticket_metric_id)"),
            ticket_metric_id: int | None = Field(None, description="Ticket metric ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            if ticket_id is not None:
                url = f"{base_url}/tickets/{ticket_id}/metrics.json"
            elif ticket_metric_id is not None:
                url = f"{base_url}/ticket_metrics/{ticket_metric_id}.json"
            else:
                raise ValueError("Provide ticket_id or ticket_metric_id")

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth})
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_ticket_metrics, **kwargs)


class ZendeskListTicketMetrics(Tool):
    name: str = "zendesk_list_ticket_metrics"
    description: str | None = "List ticket metrics."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_metrics() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_metrics.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_metrics, **kwargs)


# --- Notifications ---


class ZendeskListEmailNotifications(Tool):
    name: str = "zendesk_list_email_notifications"
    description: str | None = "List email notification delivery status (filter by ticket_id, comment_id, or notification_id)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_email_notifications(
            ticket_id: int | None = Field(None, description="Filter by ticket ID"),
            comment_id: int | None = Field(None, description="Filter by comment ID"),
            notification_id: int | None = Field(None, description="Filter by notification ID"),
            per_page: int | None = Field(None, description="Records per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, str | int] = {}
            if ticket_id is not None:
                params["filter[ticket_id]"] = ticket_id
            elif comment_id is not None:
                params["filter[comment_id]"] = comment_id
            elif notification_id is not None:
                params["filter[notification_id]"] = notification_id
            else:
                raise ValueError("Provide ticket_id, comment_id, or notification_id")
            if per_page is not None:
                params["per_page"] = per_page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/email_notifications.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_email_notifications, **kwargs)


class ZendeskShowEmailNotification(Tool):
    name: str = "zendesk_show_email_notification"
    description: str | None = "Show email notification delivery status by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_email_notification(
            email_notification_id: int = Field(..., description="Email notification ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/email_notifications/{email_notification_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_email_notification, **kwargs)


class ZendeskShowManyEmailNotifications(Tool):
    name: str = "zendesk_show_many_email_notifications"
    description: str | None = "Show many email notifications by IDs."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many_email_notifications(
            ids: str = Field(..., description="Comma-separated email notification IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/email_notifications/show_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many_email_notifications, **kwargs)


# --- Macros ---


class ZendeskListMacros(Tool):
    name: str = "zendesk_list_macros"
    description: str | None = "List Zendesk macros."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_macros(
            access: str | None = Field(None, description="personal, agents, shared, or account"),
            active: bool | None = Field(None, description="Filter by active status"),
            category: int | None = Field(None, description="Filter by category"),
            group_id: int | None = Field(None, description="Filter by group"),
            per_page: int | None = Field(None, description="Records per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {}
            if access is not None:
                params["access"] = access
            if active is not None:
                params["active"] = str(active).lower()
            if category is not None:
                params["category"] = category
            if group_id is not None:
                params["group_id"] = group_id
            if per_page is not None:
                params["per_page"] = per_page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_macros, **kwargs)


class ZendeskListActiveMacros(Tool):
    name: str = "zendesk_list_active_macros"
    description: str | None = "List active Zendesk macros."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_active_macros() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/active.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_active_macros, **kwargs)


class ZendeskShowMacro(Tool):
    name: str = "zendesk_show_macro"
    description: str | None = "Show a Zendesk macro by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_macro(
            macro_id: int = Field(..., description="Macro ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/{macro_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_macro, **kwargs)


class ZendeskCreateMacro(Tool):
    name: str = "zendesk_create_macro"
    description: str | None = "Create a Zendesk macro."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_macro(
            macro: dict[str, Any] = Field(..., description="Macro object (title, actions required)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/macros.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"macro": macro},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_macro, **kwargs)


class ZendeskUpdateMacro(Tool):
    name: str = "zendesk_update_macro"
    description: str | None = "Update a Zendesk macro."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_macro(
            macro_id: int = Field(..., description="Macro ID"),
            macro: dict[str, Any] = Field(..., description="Fields to update (include all actions)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/macros/{macro_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"macro": macro},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_macro, **kwargs)


class ZendeskUpdateManyMacros(Tool):
    name: str = "zendesk_update_many_macros"
    description: str | None = "Update many Zendesk macros (position, active status)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_many_macros(
            macros: list[dict[str, Any]] = Field(..., description="List of {id, position?, active?}"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/macros/update_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"macros": macros},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_many_macros, **kwargs)


class ZendeskDeleteMacro(Tool):
    name: str = "zendesk_delete_macro"
    description: str | None = "Delete a Zendesk macro."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_macro(
            macro_id: int = Field(..., description="Macro ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/macros/{macro_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_macro, **kwargs)


class ZendeskBulkDeleteMacros(Tool):
    name: str = "zendesk_bulk_delete_macros"
    description: str | None = "Bulk delete Zendesk macros."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_macros(
            ids: str = Field(..., description="Comma-separated macro IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/macros/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_bulk_delete_macros, **kwargs)


class ZendeskSearchMacros(Tool):
    name: str = "zendesk_search_macros"
    description: str | None = "Search Zendesk macros by title."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_macros(
            query: str = Field(..., description="Search query for macro titles"),
            active: bool | None = Field(None, description="Filter by active"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"query": query}
            if active is not None:
                params["active"] = str(active).lower()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_macros, **kwargs)


class ZendeskListMacroCategories(Tool):
    name: str = "zendesk_list_macro_categories"
    description: str | None = "List Zendesk macro categories."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_macro_categories() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/categories.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_macro_categories, **kwargs)


class ZendeskListMacroActionDefinitions(Tool):
    name: str = "zendesk_list_macro_action_definitions"
    description: str | None = "List macro action definitions (what actions a macro can perform)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_macro_action_definitions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/definitions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_macro_action_definitions, **kwargs)


class ZendeskListSupportedActionsForMacros(Tool):
    name: str = "zendesk_list_supported_actions_for_macros"
    description: str | None = "List supported actions for macros."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_supported_actions() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/actions.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_supported_actions, **kwargs)


class ZendeskListMacroAttachments(Tool):
    name: str = "zendesk_list_macro_attachments"
    description: str | None = "List attachments for a Zendesk macro."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_macro_attachments(
            macro_id: int = Field(..., description="Macro ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/{macro_id}/attachments.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_macro_attachments, **kwargs)


class ZendeskShowMacroAttachment(Tool):
    name: str = "zendesk_show_macro_attachment"
    description: str | None = "Show a Zendesk macro attachment by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_macro_attachment(
            attachment_id: int = Field(..., description="Macro attachment ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/attachments/{attachment_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_macro_attachment, **kwargs)


class ZendeskShowMacroReplica(Tool):
    name: str = "zendesk_show_macro_replica"
    description: str | None = "Show unpersisted macro replica from macro or ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_macro_replica(
            macro_id: int | None = Field(None, description="Macro ID to replicate"),
            ticket_id: int | None = Field(None, description="Ticket ID to build replica from"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            if macro_id is None and ticket_id is None:
                raise ValueError("Provide macro_id or ticket_id")
            params = {}
            if macro_id is not None:
                params["macro_id"] = macro_id
            if ticket_id is not None:
                params["ticket_id"] = ticket_id

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/new.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_macro_replica, **kwargs)


class ZendeskShowChangesToTicket(Tool):
    name: str = "zendesk_show_changes_to_ticket"
    description: str | None = "Show changes a macro would make to a ticket (without applying)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_changes_to_ticket(
            macro_id: int = Field(..., description="Macro ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/macros/{macro_id}/apply.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_changes_to_ticket, **kwargs)


class ZendeskShowTicketAfterChanges(Tool):
    name: str = "zendesk_show_ticket_after_changes"
    description: str | None = "Show full ticket as it would be after applying a macro (without applying)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_ticket_after_changes(
            ticket_id: int = Field(..., description="Ticket ID"),
            macro_id: int = Field(..., description="Macro ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/macros/{macro_id}/apply.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_ticket_after_changes, **kwargs)


class ZendeskListTicketMetricEvents(Tool):
    name: str = "zendesk_list_ticket_metric_events"
    description: str | None = "List ticket metric events (reply time, agent work time, etc.)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_metric_events(
            start_time: int = Field(..., description="Unix UTC epoch time of oldest event"),
            include_changes: bool = Field(False, description="Include changes for incremental retrieval"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"start_time": start_time}
            if include_changes:
                params["include_changes"] = "true"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/ticket_metric_events.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_metric_events, **kwargs)


# --- Group Memberships ---


class ZendeskListGroupMemberships(Tool):
    name: str = "zendesk_list_group_memberships"
    description: str | None = "List group memberships (optionally by group_id)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_group_memberships(
            group_id: int | None = Field(None, description="Filter by group ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            if group_id is not None:
                url = f"{base_url}/groups/{group_id}/memberships.json"
            else:
                url = f"{base_url}/group_memberships.json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth})
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_group_memberships, **kwargs)


class ZendeskShowGroupMembership(Tool):
    name: str = "zendesk_show_group_membership"
    description: str | None = "Show a group membership by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_group_membership(
            group_membership_id: int = Field(..., description="Group membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/group_memberships/{group_membership_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_group_membership, **kwargs)


class ZendeskCreateGroupMembership(Tool):
    name: str = "zendesk_create_group_membership"
    description: str | None = "Create a group membership (assign agent to group)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_group_membership(
            user_id: int = Field(..., description="Agent user ID"),
            group_id: int = Field(..., description="Group ID"),
            default: bool = Field(False, description="Set as default group for agent"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/group_memberships.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"group_membership": {"user_id": user_id, "group_id": group_id, "default": default}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_group_membership, **kwargs)


class ZendeskDeleteGroupMembership(Tool):
    name: str = "zendesk_delete_group_membership"
    description: str | None = "Delete a group membership."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_group_membership(
            group_membership_id: int = Field(..., description="Group membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/group_memberships/{group_membership_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_group_membership, **kwargs)


class ZendeskListAssignableMemberships(Tool):
    name: str = "zendesk_list_assignable_memberships"
    description: str | None = "List assignable group memberships."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_assignable_memberships() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/group_memberships/assignable.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_assignable_memberships, **kwargs)


class ZendeskSetGroupMembershipAsDefault(Tool):
    name: str = "zendesk_set_group_membership_as_default"
    description: str | None = "Set a group membership as default for an agent."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_group_membership_as_default(
            user_id: int = Field(..., description="User ID (agent)"),
            group_membership_id: int = Field(..., description="Group membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/group_memberships/{group_membership_id}/make_default.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_group_membership_as_default, **kwargs)


# --- Organization Memberships ---


class ZendeskListOrganizationMemberships(Tool):
    name: str = "zendesk_list_organization_memberships"
    description: str | None = "List organization memberships (optionally by user_id or organization_id)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_organization_memberships(
            user_id: int | None = Field(None, description="Filter by user ID"),
            organization_id: int | None = Field(None, description="Filter by organization ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            if user_id is not None:
                url = f"{base_url}/users/{user_id}/organization_memberships.json"
            elif organization_id is not None:
                url = f"{base_url}/organizations/{organization_id}/organization_memberships.json"
            else:
                url = f"{base_url}/organization_memberships.json"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth})
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_organization_memberships, **kwargs)


class ZendeskShowOrganizationMembership(Tool):
    name: str = "zendesk_show_organization_membership"
    description: str | None = "Show an organization membership by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_organization_membership(
            organization_membership_id: int = Field(..., description="Organization membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_memberships/{organization_membership_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_organization_membership, **kwargs)


class ZendeskCreateOrganizationMembership(Tool):
    name: str = "zendesk_create_organization_membership"
    description: str | None = "Create an organization membership (assign user to organization)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_organization_membership(
            user_id: int = Field(..., description="User ID"),
            organization_id: int = Field(..., description="Organization ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organization_memberships.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_membership": {"user_id": user_id, "organization_id": organization_id}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_organization_membership, **kwargs)


class ZendeskCreateManyMemberships(Tool):
    name: str = "zendesk_create_many_memberships"
    description: str | None = "Create many organization memberships (up to 100). Returns job_status."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_many_memberships(
            organization_memberships: list[dict[str, Any]] = Field(
                ...,
                description="List of {user_id, organization_id}",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organization_memberships/create_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_memberships": organization_memberships},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_many_memberships, **kwargs)


class ZendeskDeleteOrganizationMembership(Tool):
    name: str = "zendesk_delete_organization_membership"
    description: str | None = "Delete an organization membership."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_organization_membership(
            organization_membership_id: int = Field(..., description="Organization membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organization_memberships/{organization_membership_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_organization_membership, **kwargs)


class ZendeskSetOrganizationMembershipAsDefault(Tool):
    name: str = "zendesk_set_organization_membership_as_default"
    description: str | None = "Set an organization membership as default for a user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_organization_membership_as_default(
            user_id: int = Field(..., description="User ID"),
            organization_membership_id: int = Field(..., description="Organization membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/organization_memberships/{organization_membership_id}/make_default.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_organization_membership_as_default, **kwargs)


class ZendeskSetOrganizationAsDefault(Tool):
    name: str = "zendesk_set_organization_as_default"
    description: str | None = "Set a user's default organization (updates user.organization_id)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _set_organization_as_default(
            user_id: int = Field(..., description="User ID"),
            organization_id: int = Field(..., description="Organization ID to set as default"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/organizations/{organization_id}/make_default.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_set_organization_as_default, **kwargs)


class ZendeskBulkDeleteGroupMemberships(Tool):
    name: str = "zendesk_bulk_delete_group_memberships"
    description: str | None = "Bulk delete group memberships."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_group_memberships(
            ids: str = Field(..., description="Comma-separated group membership IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/group_memberships/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_delete_group_memberships, **kwargs)


class ZendeskBulkCreateMemberships(Tool):
    name: str = "zendesk_bulk_create_memberships"
    description: str | None = "Bulk create group memberships (up to 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_create_memberships(
            group_memberships: list[dict[str, Any]] = Field(
                ...,
                description="List of {user_id, group_id}",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/group_memberships/create_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"group_memberships": group_memberships},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_create_memberships, **kwargs)


class ZendeskBulkDeleteOrganizationMemberships(Tool):
    name: str = "zendesk_bulk_delete_organization_memberships"
    description: str | None = "Bulk delete organization memberships."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_delete_organization_memberships(
            ids: str = Field(..., description="Comma-separated organization membership IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organization_memberships/destroy_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_delete_organization_memberships, **kwargs)


class ZendeskUnassignOrganization(Tool):
    name: str = "zendesk_unassign_organization"
    description: str | None = "Unassign a user from an organization (delete org membership)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _unassign_organization(
            organization_membership_id: int = Field(..., description="Organization membership ID to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organization_memberships/{organization_membership_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_unassign_organization, **kwargs)


# --- Identity ---


class ZendeskListIdentities(Tool):
    name: str = "zendesk_list_identities"
    description: str | None = "List user identities (email, phone, etc.) for a user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_identities(
            user_id: int = Field(..., description="User ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_id}/identities.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_identities, **kwargs)


class ZendeskShowIdentity(Tool):
    name: str = "zendesk_show_identity"
    description: str | None = "Show a single user identity by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_identity(
            user_id: int = Field(..., description="User ID"),
            identity_id: int = Field(..., description="Identity ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_id}/identities/{identity_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_identity, **kwargs)


class ZendeskCreateIdentity(Tool):
    name: str = "zendesk_create_identity"
    description: str | None = "Create a user identity (email, phone, twitter, etc.)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_identity(
            user_id: int = Field(..., description="User ID"),
            identity_type: str = Field(..., description="Identity type: email, twitter, facebook, google, phone_number, agent_forwarding"),
            value: str = Field(..., description="Identity value (e.g. email address, phone number)"),
            primary: bool = Field(False, description="Set as primary identity"),
            skip_verify_email: bool = Field(False, description="Skip sending verification email"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            identity: dict[str, Any] = {"type": identity_type, "value": value}
            if primary:
                identity["primary"] = True
            if skip_verify_email:
                identity["skip_verify_email"] = True

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/users/{user_id}/identities.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"identity": identity},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_identity, **kwargs)


class ZendeskUpdateIdentity(Tool):
    name: str = "zendesk_update_identity"
    description: str | None = "Update a user identity (value, verification_method)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_identity(
            user_id: int = Field(..., description="User ID"),
            identity_id: int = Field(..., description="Identity ID"),
            value: str | None = Field(None, description="New identity value"),
            verification_method: str | None = Field(None, description="Verification: none, low, sso, embed, full"),
            verified: bool | None = Field(None, description="Mark as verified (deprecated, use verification_method)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            identity: dict[str, Any] = {}
            if value is not None:
                identity["value"] = value
            if verification_method is not None:
                identity["verification_method"] = verification_method
            if verified is not None:
                identity["verified"] = verified
            if not identity:
                raise ValueError("At least one of value, verification_method, or verified must be provided")

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/identities/{identity_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"identity": identity},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_identity, **kwargs)


class ZendeskMakeIdentityPrimary(Tool):
    name: str = "zendesk_make_identity_primary"
    description: str | None = "Set an identity as the user's primary identity."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _make_identity_primary(
            user_id: int = Field(..., description="User ID"),
            identity_id: int = Field(..., description="Identity ID to set as primary"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/identities/{identity_id}/make_primary",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_make_identity_primary, **kwargs)


class ZendeskRequestUserVerification(Tool):
    name: str = "zendesk_request_user_verification"
    description: str | None = "Send verification email to user for an identity."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _request_user_verification(
            user_id: int = Field(..., description="User ID"),
            identity_id: int = Field(..., description="Identity ID to verify"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/identities/{identity_id}/request_verification",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_request_user_verification, **kwargs)


class ZendeskVerifyIdentity(Tool):
    name: str = "zendesk_verify_identity"
    description: str | None = "Mark an identity as verified (agent-only)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _verify_identity(
            user_id: int = Field(..., description="User ID"),
            identity_id: int = Field(..., description="Identity ID to verify"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/users/{user_id}/identities/{identity_id}/verify",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_verify_identity, **kwargs)


class ZendeskDeleteIdentity(Tool):
    name: str = "zendesk_delete_identity"
    description: str | None = "Delete a user identity."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_identity(
            user_id: int = Field(..., description="User ID"),
            identity_id: int = Field(..., description="Identity ID to delete"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/users/{user_id}/identities/{identity_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_identity, **kwargs)


# --- Import ---


class ZendeskTicketImport(Tool):
    name: str = "zendesk_ticket_import"
    description: str | None = "Import a single ticket from a legacy system (admin only)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _ticket_import(
            ticket: dict[str, Any] = Field(..., description="Ticket object with subject, comments, requester_id, etc."),
            archive_immediately: bool = Field(False, description="Archive closed tickets immediately"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if archive_immediately:
                params["archive_immediately"] = "true"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/imports/tickets.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params,
                    json={"ticket": ticket},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_ticket_import, **kwargs)


class ZendeskTicketBulkImport(Tool):
    name: str = "zendesk_ticket_bulk_import"
    description: str | None = "Bulk import up to 100 tickets (admin only, returns job_status)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _ticket_bulk_import(
            tickets: list[dict[str, Any]] = Field(..., description="List of ticket objects (max 100)"),
            archive_immediately: bool = Field(False, description="Archive closed tickets immediately"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if archive_immediately:
                params["archive_immediately"] = "true"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/imports/tickets/create_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params,
                    json={"tickets": tickets},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_ticket_bulk_import, **kwargs)


# --- Locale ---


class ZendeskListLocales(Tool):
    name: str = "zendesk_list_locales"
    description: str | None = "List locales available for the account."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_locales() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/locales.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_locales, **kwargs)


class ZendeskListLocalesForAgent(Tool):
    name: str = "zendesk_list_locales_for_agent"
    description: str | None = "List locales localized for agents."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_locales_for_agent() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/locales/agent.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_locales_for_agent, **kwargs)


class ZendeskListAvailablePublicLocales(Tool):
    name: str = "zendesk_list_available_public_locales"
    description: str | None = "List public locales available to all accounts."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_available_public_locales() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/locales/public.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_available_public_locales, **kwargs)


class ZendeskShowCurrentLocale(Tool):
    name: str = "zendesk_show_current_locale"
    description: str | None = "Show the locale of the current user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_current_locale() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/locales/current.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_current_locale, **kwargs)


class ZendeskShowLocale(Tool):
    name: str = "zendesk_show_locale"
    description: str | None = "Show a locale by ID or BCP-47 code."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_locale(
            locale_id: str = Field(..., description="Locale ID or BCP-47 code (e.g. en-US, es-419)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/locales/{locale_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_locale, **kwargs)


class ZendeskDetectBestLanguageForUser(Tool):
    name: str = "zendesk_detect_best_language_for_user"
    description: str | None = "Detect best locale from available_locales for the user."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _detect_best_language_for_user(
            available_locales: list[str] = Field(..., description="List of locale codes (e.g. es, ja, en-uk)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/locales/detect_best_locale.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"available_locales": available_locales},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_detect_best_language_for_user, **kwargs)


# --- Export ---


class ZendeskIncrementalTicketExportCursor(Tool):
    name: str = "zendesk_incremental_ticket_export_cursor"
    description: str | None = "Incremental ticket export (cursor-based). Use start_time for initial request, cursor for subsequent."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_ticket_export_cursor(
            start_time: int | None = Field(None, description="Unix epoch for initial export (required on first request)"),
            cursor: str | None = Field(None, description="Cursor for subsequent pages"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            elif start_time is not None:
                params["start_time"] = start_time
            else:
                raise ValueError("Either start_time or cursor must be provided")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/tickets/cursor.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_ticket_export_cursor, **kwargs)


class ZendeskIncrementalTicketExportTimeBased(Tool):
    name: str = "zendesk_incremental_ticket_export_time_based"
    description: str | None = "Incremental ticket export (time-based)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_ticket_export_time_based(
            start_time: int = Field(..., description="Unix epoch to start export from"),
            per_page: int | None = Field(None, description="Results per page (max 1000)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"start_time": start_time}
            if per_page is not None:
                params["per_page"] = per_page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/tickets.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_ticket_export_time_based, **kwargs)


class ZendeskIncrementalTicketEventExport(Tool):
    name: str = "zendesk_incremental_ticket_event_export"
    description: str | None = "Incremental ticket event export (changes on tickets)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_ticket_event_export(
            start_time: int = Field(..., description="Unix epoch to start export from"),
            per_page: int | None = Field(None, description="Results per page"),
            include: str | None = Field(None, description="Sideloads, e.g. comment_events"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"start_time": start_time}
            if per_page is not None:
                params["per_page"] = per_page
            if include:
                params["include"] = include

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/ticket_events.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_ticket_event_export, **kwargs)


class ZendeskIncrementalUserExportCursor(Tool):
    name: str = "zendesk_incremental_user_export_cursor"
    description: str | None = "Incremental user export (cursor-based)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_user_export_cursor(
            start_time: int | None = Field(None, description="Unix epoch for initial export"),
            cursor: str | None = Field(None, description="Cursor for subsequent pages"),
            per_page: int | None = Field(None, description="Results per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            elif start_time is not None:
                params["start_time"] = start_time
            else:
                raise ValueError("Either start_time or cursor must be provided")
            if per_page is not None:
                params["per_page"] = per_page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/users/cursor.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_user_export_cursor, **kwargs)


class ZendeskIncrementalUserExportTimeBased(Tool):
    name: str = "zendesk_incremental_user_export_time_based"
    description: str | None = "Incremental user export (time-based)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_user_export_time_based(
            start_time: int = Field(..., description="Unix epoch to start export from"),
            per_page: int | None = Field(None, description="Results per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"start_time": start_time}
            if per_page is not None:
                params["per_page"] = per_page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/users.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_user_export_time_based, **kwargs)


class ZendeskIncrementalOrganizationExport(Tool):
    name: str = "zendesk_incremental_organization_export"
    description: str | None = "Incremental organization export."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_organization_export(
            start_time: int = Field(..., description="Unix epoch to start export from"),
            per_page: int | None = Field(None, description="Results per page"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"start_time": start_time}
            if per_page is not None:
                params["per_page"] = per_page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/organizations.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_organization_export, **kwargs)


class ZendeskIncrementalSampleExport(Tool):
    name: str = "zendesk_incremental_sample_export"
    description: str | None = "Sample incremental export (max 50 results, for testing format)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _incremental_sample_export(
            resource: str = Field(..., description="Resource: tickets, ticket_events, users, or organizations"),
            start_time: int = Field(..., description="Unix epoch to start export from"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/incremental/{resource}/sample.json",
                    headers={"Authorization": auth},
                    params={"start_time": start_time},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_incremental_sample_export, **kwargs)


# --- Ticket Field ---


class ZendeskCountTicketFields(Tool):
    name: str = "zendesk_count_ticket_fields"
    description: str | None = "Count ticket fields in the account."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_ticket_fields() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_fields/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_ticket_fields, **kwargs)


class ZendeskListTicketFields(Tool):
    name: str = "zendesk_list_ticket_fields"
    description: str | None = "List all ticket fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_fields() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_fields.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_fields, **kwargs)


class ZendeskShowTicketField(Tool):
    name: str = "zendesk_show_ticket_field"
    description: str | None = "Show a ticket field by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_ticket_field(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_fields/{ticket_field_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_ticket_field, **kwargs)


class ZendeskCreateTicketField(Tool):
    name: str = "zendesk_create_ticket_field"
    description: str | None = "Create a custom ticket field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_ticket_field(
            ticket_field: dict[str, Any] = Field(..., description="Ticket field object with type, title, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/ticket_fields.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_field": ticket_field},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_ticket_field, **kwargs)


class ZendeskUpdateTicketField(Tool):
    name: str = "zendesk_update_ticket_field"
    description: str | None = "Update a ticket field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_ticket_field(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
            ticket_field: dict[str, Any] = Field(..., description="Ticket field updates"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/ticket_fields/{ticket_field_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_field": ticket_field},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_ticket_field, **kwargs)


class ZendeskDeleteTicketField(Tool):
    name: str = "zendesk_delete_ticket_field"
    description: str | None = "Delete a ticket field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_ticket_field(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/ticket_fields/{ticket_field_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_ticket_field, **kwargs)


class ZendeskReorderTicketFields(Tool):
    name: str = "zendesk_reorder_ticket_fields"
    description: str | None = "Reorder ticket fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_ticket_fields(
            ticket_field_ids: list[int] = Field(..., description="Ticket field IDs in desired order"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/ticket_fields/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_field_ids": ticket_field_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_ticket_fields, **kwargs)


class ZendeskListTicketFieldOptions(Tool):
    name: str = "zendesk_list_ticket_field_options"
    description: str | None = "List options for a ticket field (dropdown/tagger)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_field_options(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_fields/{ticket_field_id}/options.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_field_options, **kwargs)


class ZendeskShowTicketFieldOption(Tool):
    name: str = "zendesk_show_ticket_field_option"
    description: str | None = "Show a ticket field option by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_ticket_field_option(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
            option_id: int = Field(..., description="Option ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_fields/{ticket_field_id}/options/{option_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_ticket_field_option, **kwargs)


class ZendeskCreateOrUpdateTicketFieldOption(Tool):
    name: str = "zendesk_create_or_update_ticket_field_option"
    description: str | None = "Create or update a ticket field option."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_or_update_ticket_field_option(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
            option: dict[str, Any] = Field(..., description="Option object with name/value, id for update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/ticket_fields/{ticket_field_id}/options.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_field_option": option},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_or_update_ticket_field_option, **kwargs)


class ZendeskDeleteTicketFieldOption(Tool):
    name: str = "zendesk_delete_ticket_field_option"
    description: str | None = "Delete a ticket field option."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_ticket_field_option(
            ticket_field_id: int = Field(..., description="Ticket field ID"),
            option_id: int = Field(..., description="Option ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/ticket_fields/{ticket_field_id}/options/{option_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_ticket_field_option, **kwargs)


# --- Ticket Form ---


class ZendeskListTicketForms(Tool):
    name: str = "zendesk_list_ticket_forms"
    description: str | None = "List ticket forms."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_forms() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_forms.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_forms, **kwargs)


class ZendeskShowTicketForm(Tool):
    name: str = "zendesk_show_ticket_form"
    description: str | None = "Show a ticket form by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_ticket_form(
            ticket_form_id: int = Field(..., description="Ticket form ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_forms/{ticket_form_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_ticket_form, **kwargs)


class ZendeskShowManyTicketForms(Tool):
    name: str = "zendesk_show_many_ticket_forms"
    description: str | None = "Show multiple ticket forms by IDs (comma-separated, max 100)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many_ticket_forms(
            ids: str = Field(..., description="Comma-separated ticket form IDs"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_forms/show_many.json",
                    headers={"Authorization": auth},
                    params={"ids": ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many_ticket_forms, **kwargs)


class ZendeskCreateTicketForm(Tool):
    name: str = "zendesk_create_ticket_form"
    description: str | None = "Create a ticket form."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_ticket_form(
            ticket_form: dict[str, Any] = Field(..., description="Ticket form object with name, ticket_field_ids, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/ticket_forms.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_form": ticket_form},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_ticket_form, **kwargs)


class ZendeskUpdateTicketForm(Tool):
    name: str = "zendesk_update_ticket_form"
    description: str | None = "Update a ticket form."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_ticket_form(
            ticket_form_id: int = Field(..., description="Ticket form ID"),
            ticket_form: dict[str, Any] = Field(..., description="Ticket form updates"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/ticket_forms/{ticket_form_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_form": ticket_form},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_ticket_form, **kwargs)


class ZendeskDeleteTicketForm(Tool):
    name: str = "zendesk_delete_ticket_form"
    description: str | None = "Delete a ticket form."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_ticket_form(
            ticket_form_id: int = Field(..., description="Ticket form ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/ticket_forms/{ticket_form_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_ticket_form, **kwargs)


class ZendeskCloneTicketForm(Tool):
    name: str = "zendesk_clone_ticket_form"
    description: str | None = "Clone an existing ticket form."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _clone_ticket_form(
            ticket_form_id: int = Field(..., description="Ticket form ID to clone"),
            prepend_clone_title: bool = Field(True, description="Prepend 'Clone of' to the form name"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/ticket_forms/{ticket_form_id}/clone.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"prepend_clone_title": prepend_clone_title},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_clone_ticket_form, **kwargs)


class ZendeskReorderTicketForms(Tool):
    name: str = "zendesk_reorder_ticket_forms"
    description: str | None = "Reorder ticket forms."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_ticket_forms(
            ticket_form_ids: list[int] = Field(..., description="Ticket form IDs in desired order"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/ticket_forms/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_form_ids": ticket_form_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_ticket_forms, **kwargs)


class ZendeskListTicketFormStatuses(Tool):
    name: str = "zendesk_list_ticket_form_statuses"
    description: str | None = "List ticket form statuses of a ticket form."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ticket_form_statuses(
            ticket_form_id: int = Field(..., description="Ticket form ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_forms/{ticket_form_id}/ticket_form_statuses.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_ticket_form_statuses, **kwargs)


class ZendeskCreateTicketFormStatuses(Tool):
    name: str = "zendesk_create_ticket_form_statuses"
    description: str | None = "Create ticket form status associations."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_ticket_form_statuses(
            ticket_form_id: int = Field(..., description="Ticket form ID"),
            ticket_form_status: list[dict[str, Any]] = Field(..., description="List of {custom_status_id} objects"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/ticket_forms/{ticket_form_id}/ticket_form_statuses.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_form_status": ticket_form_status},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_ticket_form_statuses, **kwargs)


class ZendeskBulkUpdateTicketFormStatuses(Tool):
    name: str = "zendesk_bulk_update_ticket_form_statuses"
    description: str | None = "Bulk update ticket form statuses (add/remove associations)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_update_ticket_form_statuses(
            ticket_form_id: int = Field(..., description="Ticket form ID"),
            ticket_form_status: list[dict[str, Any]] = Field(
                ...,
                description="List of {custom_status_id} to add or {_destroy: '1', id} to remove",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/ticket_forms/{ticket_form_id}/ticket_form_statuses.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"ticket_form_status": ticket_form_status},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_bulk_update_ticket_form_statuses, **kwargs)


# --- User Field ---


class ZendeskListUserFields(Tool):
    name: str = "zendesk_list_user_fields"
    description: str | None = "List custom user fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_user_fields() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/user_fields.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_user_fields, **kwargs)


class ZendeskShowUserField(Tool):
    name: str = "zendesk_show_user_field"
    description: str | None = "Show a user field by ID or key."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_user_field(
            user_field_id: str | int = Field(..., description="User field ID or key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/user_fields/{user_field_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_user_field, **kwargs)


class ZendeskCreateUserField(Tool):
    name: str = "zendesk_create_user_field"
    description: str | None = "Create a custom user field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_user_field(
            user_field: dict[str, Any] = Field(..., description="User field object with key, title, type, etc."),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/user_fields.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"user_field": user_field},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_user_field, **kwargs)


class ZendeskUpdateUserField(Tool):
    name: str = "zendesk_update_user_field"
    description: str | None = "Update a user field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_user_field(
            user_field_id: str | int = Field(..., description="User field ID or key"),
            user_field: dict[str, Any] = Field(..., description="User field updates"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/user_fields/{user_field_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"user_field": user_field},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_user_field, **kwargs)


class ZendeskDeleteUserField(Tool):
    name: str = "zendesk_delete_user_field"
    description: str | None = "Delete a user field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_user_field(
            user_field_id: str | int = Field(..., description="User field ID or key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/user_fields/{user_field_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_user_field, **kwargs)


class ZendeskReorderUserField(Tool):
    name: str = "zendesk_reorder_user_field"
    description: str | None = "Reorder user fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_user_field(
            user_field_ids: list[int | str] = Field(..., description="User field IDs in desired order"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/user_fields/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"user_field_ids": user_field_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_user_field, **kwargs)


class ZendeskListUserFieldOptions(Tool):
    name: str = "zendesk_list_user_field_options"
    description: str | None = "List options for a dropdown/multiselect user field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_user_field_options(
            user_field_id: str | int = Field(..., description="User field ID or key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/user_fields/{user_field_id}/options.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_user_field_options, **kwargs)


class ZendeskShowUserFieldOption(Tool):
    name: str = "zendesk_show_user_field_option"
    description: str | None = "Show a user field option by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_user_field_option(
            user_field_id: str | int = Field(..., description="User field ID or key"),
            option_id: int = Field(..., description="Option ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/user_fields/{user_field_id}/options/{option_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_user_field_option, **kwargs)


class ZendeskCreateOrUpdateUserFieldOption(Tool):
    name: str = "zendesk_create_or_update_user_field_option"
    description: str | None = "Create or update a user field option."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_or_update_user_field_option(
            user_field_id: str | int = Field(..., description="User field ID or key"),
            option: dict[str, Any] = Field(..., description="Option object with name, value; id for update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/user_fields/{user_field_id}/options.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"user_field_option": option},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_or_update_user_field_option, **kwargs)


class ZendeskDeleteUserFieldOption(Tool):
    name: str = "zendesk_delete_user_field_option"
    description: str | None = "Delete a user field option."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_user_field_option(
            user_field_id: str | int = Field(..., description="User field ID or key"),
            option_id: int = Field(..., description="Option ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/user_fields/{user_field_id}/options/{option_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_user_field_option, **kwargs)


# --- Organization Field ---


class ZendeskListOrganizationFields(Tool):
    name: str = "zendesk_list_organization_fields"
    description: str | None = "List custom organization fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_organization_fields() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_fields.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_organization_fields, **kwargs)


class ZendeskShowOrganizationField(Tool):
    name: str = "zendesk_show_organization_field"
    description: str | None = "Show an organization field by ID or key."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_organization_field(
            organization_field_id: str | int = Field(..., description="Organization field ID or key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/organization_fields/{organization_field_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_organization_field, **kwargs)


class ZendeskCreateOrganizationField(Tool):
    name: str = "zendesk_create_organization_field"
    description: str | None = "Create a custom organization field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_organization_field(
            organization_field: dict[str, Any] = Field(..., description="Organization field object with key, title, type"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/organization_fields.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_field": organization_field},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_organization_field, **kwargs)


class ZendeskUpdateOrganizationField(Tool):
    name: str = "zendesk_update_organization_field"
    description: str | None = "Update an organization field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_organization_field(
            organization_field_id: str | int = Field(..., description="Organization field ID or key"),
            organization_field: dict[str, Any] = Field(..., description="Organization field updates"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/organization_fields/{organization_field_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_field": organization_field},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_organization_field, **kwargs)


class ZendeskDeleteOrganizationField(Tool):
    name: str = "zendesk_delete_organization_field"
    description: str | None = "Delete an organization field."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_organization_field(
            organization_field_id: str | int = Field(..., description="Organization field ID or key"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/organization_fields/{organization_field_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_organization_field, **kwargs)


class ZendeskReorderOrganizationField(Tool):
    name: str = "zendesk_reorder_organization_field"
    description: str | None = "Reorder organization fields."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_organization_field(
            organization_field_ids: list[int | str] = Field(..., description="Organization field IDs in desired order"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/organization_fields/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"organization_field_ids": organization_field_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_organization_field, **kwargs)


# --- Content (Dynamic Content) ---


class ZendeskListDynamicContentItems(Tool):
    name: str = "zendesk_list_dynamic_content_items"
    description: str | None = "List Zendesk dynamic content items."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_items(
            per_page: int = Field(50, description="Items per page"),
            sort: str | None = Field(None, description="Sort field (e.g. name, -created_at)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page}
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/dynamic_content/items.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_items, **kwargs)


class ZendeskShowDynamicContentItem(Tool):
    name: str = "zendesk_show_dynamic_content_item"
    description: str | None = "Show a Zendesk dynamic content item by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_item(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_item, **kwargs)


class ZendeskShowManyDynamicContentItems(Tool):
    name: str = "zendesk_show_many_dynamic_content_items"
    description: str | None = "Show many dynamic content items by identifiers."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_many(
            identifiers: str = Field(..., description="Comma-separated identifiers"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/dynamic_content/items/show_many.json",
                    headers={"Authorization": auth},
                    params={"identifiers": identifiers},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_many, **kwargs)


class ZendeskCreateDynamicContentItem(Tool):
    name: str = "zendesk_create_dynamic_content_item"
    description: str | None = "Create a Zendesk dynamic content item."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_item(
            item: dict[str, Any] = Field(
                ...,
                description="Item object: name, default_locale_id, variants (or content)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/dynamic_content/items.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"item": item},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_item, **kwargs)


class ZendeskUpdateDynamicContentItem(Tool):
    name: str = "zendesk_update_dynamic_content_item"
    description: str | None = "Update a Zendesk dynamic content item (name only)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_item(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            item: dict[str, Any] = Field(..., description="Fields to update (e.g. name)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"item": item},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_item, **kwargs)


class ZendeskDeleteDynamicContentItem(Tool):
    name: str = "zendesk_delete_dynamic_content_item"
    description: str | None = "Delete a Zendesk dynamic content item."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_item, **kwargs)


class ZendeskListDynamicContentVariants(Tool):
    name: str = "zendesk_list_dynamic_content_variants"
    description: str | None = "List variants of a Zendesk dynamic content item."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_variants(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            per_page: int = Field(50, description="Variants per page"),
            sort: str | None = Field(None, description="Sort field"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page}
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_variants, **kwargs)


class ZendeskShowDynamicContentVariant(Tool):
    name: str = "zendesk_show_dynamic_content_variant"
    description: str | None = "Show a Zendesk dynamic content variant."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_variant(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            dynamic_content_variant_id: int = Field(..., description="Variant ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants/{dynamic_content_variant_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_variant, **kwargs)


class ZendeskCreateDynamicContentVariant(Tool):
    name: str = "zendesk_create_dynamic_content_variant"
    description: str | None = "Create a Zendesk dynamic content variant."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_variant(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            variant: dict[str, Any] = Field(
                ...,
                description="Variant object: locale_id, content, active, default",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"variant": variant},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_variant, **kwargs)


class ZendeskCreateManyDynamicContentVariants(Tool):
    name: str = "zendesk_create_many_dynamic_content_variants"
    description: str | None = "Create many Zendesk dynamic content variants."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_many(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            variants: list[dict[str, Any]] = Field(
                ...,
                description="List of variant objects (locale_id, content, active, default)",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants/create_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"variants": variants},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_many, **kwargs)


class ZendeskUpdateDynamicContentVariant(Tool):
    name: str = "zendesk_update_dynamic_content_variant"
    description: str | None = "Update a Zendesk dynamic content variant."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_variant(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            dynamic_content_variant_id: int = Field(..., description="Variant ID"),
            variant: dict[str, Any] = Field(..., description="Fields to update"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants/{dynamic_content_variant_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"variant": variant},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_variant, **kwargs)


class ZendeskUpdateManyDynamicContentVariants(Tool):
    name: str = "zendesk_update_many_dynamic_content_variants"
    description: str | None = "Update many Zendesk dynamic content variants."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_many(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            variants: list[dict[str, Any]] = Field(
                ...,
                description="List of variant objects with id and fields to update",
            ),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants/update_many.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"variants": variants},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_many, **kwargs)


class ZendeskDeleteDynamicContentVariant(Tool):
    name: str = "zendesk_delete_dynamic_content_variant"
    description: str | None = "Delete a Zendesk dynamic content variant."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_variant(
            dynamic_content_item_id: int = Field(..., description="Dynamic content item ID"),
            dynamic_content_variant_id: int = Field(..., description="Variant ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/dynamic_content/items/{dynamic_content_item_id}/variants/{dynamic_content_variant_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_delete_variant, **kwargs)


# --- Basic (Voice / Talk Partner Edition) ---


class ZendeskCreateTicketOrVoicemailTicket(Tool):
    name: str = "zendesk_create_ticket_or_voicemail_ticket"
    description: str | None = "Create ticket or voicemail ticket via Voice channel (via_id: 44 voicemail, 45 inbound, 46 outbound)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            ticket: dict[str, Any] = Field(
                ...,
                description="Ticket object: via_id (44/45/46), subject/description, comment, voice_comment for voicemail",
            ),
            display_to_agent: int | None = Field(None, description="Agent ID to display ticket to"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {"ticket": ticket}
            if display_to_agent is not None:
                body["display_to_agent"] = display_to_agent

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/channels/voice/tickets.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create, **kwargs)


class ZendeskOpenUserProfileInAgentWorkspace(Tool):
    name: str = "zendesk_open_user_profile_in_agent_workspace"
    description: str | None = "Open a user's profile in an agent's browser (Voice/Talk)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _open(
            agent_id: int = Field(..., description="Agent ID"),
            user_id: int = Field(..., description="User ID to display"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/channels/voice/agents/{agent_id}/users/{user_id}/display.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_open, **kwargs)


class ZendeskOpenTicketInAgentBrowser(Tool):
    name: str = "zendesk_open_ticket_in_agent_browser"
    description: str | None = "Open a ticket in an agent's browser (Voice/Talk)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _open(
            agent_id: int = Field(..., description="Agent ID"),
            ticket_id: int = Field(..., description="Ticket ID to display"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/channels/voice/agents/{agent_id}/tickets/{ticket_id}/display.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_open, **kwargs)


# --- Channel ---


class ZendeskListMonitoredXHandles(Tool):
    name: str = "zendesk_list_monitored_x_handles"
    description: str | None = "List monitored X (Twitter) handles."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_handles() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/channels/twitter/monitored_twitter_handles.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_handles, **kwargs)


class ZendeskShowMonitoredXHandle(Tool):
    name: str = "zendesk_show_monitored_x_handle"
    description: str | None = "Show a monitored X (Twitter) handle by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_handle(
            monitored_twitter_handle_id: int = Field(..., description="Monitored X handle ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/channels/twitter/monitored_twitter_handles/{monitored_twitter_handle_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_handle, **kwargs)


class ZendeskCreateTicketFromTweet(Tool):
    name: str = "zendesk_create_ticket_from_tweet"
    description: str | None = "Create a Zendesk ticket from a tweet."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_from_tweet(
            twitter_status_message_id: int = Field(..., description="Tweet ID"),
            monitored_twitter_handle_id: int = Field(..., description="Monitored X handle ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/channels/twitter/tickets.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={
                        "ticket": {
                            "twitter_status_message_id": twitter_status_message_id,
                            "monitored_twitter_handle_id": monitored_twitter_handle_id,
                        }
                    },
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json() if response.content else {"status": "created"}

        super().__init__(handler=_create_from_tweet, **kwargs)


class ZendeskListTicketStatuses(Tool):
    name: str = "zendesk_list_ticket_statuses"
    description: str | None = "List ticket statuses for a Twitter channel comment."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_statuses(
            comment_id: int = Field(..., description="Comment ID"),
            ids: str | None = Field(None, description="Optional comment ids filter"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, str] = {}
            if ids:
                params["ids"] = ids

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/channels/twitter/tickets/{comment_id}/statuses.json",
                    headers={"Authorization": auth},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_statuses, **kwargs)


class ZendeskPushContentToSupport(Tool):
    name: str = "zendesk_push_content_to_support"
    description: str | None = "Push Channel framework content to Zendesk Support."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _push_content(
            instance_push_id: str = Field(..., description="Account ID for data pushes"),
            external_resources: list[dict[str, Any]] = Field(
                ...,
                description="Resources to push (external_id, message, created_at, author, etc.)",
            ),
            request_id: str | None = Field(None, description="Unique request identifier"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {
                "instance_push_id": instance_push_id,
                "external_resources": external_resources,
            }
            if request_id:
                body["request_id"] = request_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/any_channel/push.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_push_content, **kwargs)


class ZendeskValidateToken(Tool):
    name: str = "zendesk_validate_token"
    description: str | None = "Validate Channel framework token."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _validate_token(
            instance_push_id: str = Field(..., description="Account ID for push"),
            request_id: str | None = Field(None, description="Unique request identifier"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {"instance_push_id": instance_push_id}
            if request_id:
                body["request_id"] = request_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/any_channel/validate_token.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_validate_token, **kwargs)


class ZendeskReportChannelbackErrorToZendesk(Tool):
    name: str = "zendesk_report_channelback_error_to_zendesk"
    description: str | None = "Report a channelback error to Zendesk."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _report_error(
            instance_push_id: str = Field(..., description="Account ID"),
            external_id: str = Field(..., description="External resource ID from channelback"),
            description: str | None = Field(None, description="Human-readable error description"),
            request_id: str | None = Field(None, description="Unique request identifier"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {
                "instance_push_id": instance_push_id,
                "external_id": external_id,
            }
            if description:
                body["description"] = description
            if request_id:
                body["request_id"] = request_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/any_channel/channelback/report_error.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "ok"} if not response.content else response.json()

        super().__init__(handler=_report_error, **kwargs)


class ZendeskReorderGroupSlaPolicies(Tool):
    name: str = "zendesk_reorder_group_sla_policies"
    description: str | None = "Reorder Zendesk SLA policies for a group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reorder_group_sla_policies(
            group_id: int = Field(..., description="Group ID"),
            sla_policy_ids: list[int] = Field(..., description="SLA policy IDs in desired order"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/groups/{group_id}/slas/policies/reorder.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"sla_policy_ids": sla_policy_ids},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reorder_group_sla_policies, **kwargs)


# --- Audit ---


class ZendeskChangeCommentFromPublicToPrivate(Tool):
    """Change a comment from public to private via ticket audit."""

    name: str = "zendesk_change_comment_from_public_to_private"
    description: str | None = "Change a ticket audit comment from public to private."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _change_comment(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            ticket_audit_id: int = Field(..., description="Ticket audit ID containing the comment"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}/audits/{ticket_audit_id}/make_private",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json() if response.content else {"status": "ok"}

        super().__init__(handler=_change_comment, **kwargs)


class ZendeskCountAuditsForTicket(Tool):
    name: str = "zendesk_count_audits_for_ticket"
    description: str | None = "Count audits for a Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_audits(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/audits/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_audits, **kwargs)


class ZendeskExportAuditLogs(Tool):
    name: str = "zendesk_export_audit_logs"
    description: str | None = "Export account audit logs (Enterprise plan). Rate limit: 1 req/min."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _export_audit_logs(
            filter_action: str | None = Field(None, description="Filter by action (create, update, destroy, etc.)"),
            filter_actor_id: int | None = Field(None, description="Filter by actor ID"),
            filter_source_type: str | None = Field(None, description="Filter by source type (user, rule, etc.)"),
            filter_source_id: int | None = Field(None, description="Filter by source ID (requires filter_source_type)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if filter_action:
                params["filter[action]"] = filter_action
            if filter_actor_id is not None:
                params["filter[actor_id]"] = filter_actor_id
            if filter_source_type:
                params["filter[source_type]"] = filter_source_type
            if filter_source_id is not None:
                params["filter[source_id]"] = filter_source_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/audit_logs/export",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "accepted", "message": "Export request submitted"}

        super().__init__(handler=_export_audit_logs, **kwargs)


class ZendeskListAllTicketAudits(Tool):
    name: str = "zendesk_list_all_ticket_audits"
    description: str | None = "List all ticket audits across the account (cursor pagination)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_audits(
            page_size: int = Field(100, description="Records per page (max 100)"),
            page_after: str | None = Field(None, description="Cursor for next page (meta.after_cursor)"),
            page_before: str | None = Field(None, description="Cursor for previous page (meta.before_cursor)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"page[size]": page_size}
            if page_after:
                params["page[after]"] = page_after
            elif page_before:
                params["page[before]"] = page_before

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/ticket_audits",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_audits, **kwargs)


class ZendeskListAuditLogs(Tool):
    name: str = "zendesk_list_audit_logs"
    description: str | None = "List account audit logs (Enterprise plan)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_audit_logs(
            per_page: int = Field(50, description="Records per page"),
            page: int = Field(1, description="Page number"),
            sort_by: str = Field("created_at", description="Sort field"),
            sort_order: str = Field("desc", description="asc or desc"),
            filter_action: str | None = Field(None, description="Filter by action"),
            filter_actor_id: int | None = Field(None, description="Filter by actor ID"),
            filter_source_type: str | None = Field(None, description="Filter by source type"),
            filter_source_id: int | None = Field(None, description="Filter by source ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page, "sort_by": sort_by, "sort_order": sort_order}
            if filter_action:
                params["filter[action]"] = filter_action
            if filter_actor_id is not None:
                params["filter[actor_id]"] = filter_actor_id
            if filter_source_type:
                params["filter[source_type]"] = filter_source_type
            if filter_source_id is not None:
                params["filter[source_id]"] = filter_source_id

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/audit_logs",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_audit_logs, **kwargs)


class ZendeskListAuditsForTicket(Tool):
    name: str = "zendesk_list_audits_for_ticket"
    description: str | None = "List audits for a specific Zendesk ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_audits(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            per_page: int = Field(25, description="Records per page"),
            page: int = Field(1, description="Page number"),
            sort_order: str = Field("asc", description="asc or desc"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page, "sort_order": sort_order}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/audits.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_audits, **kwargs)


class ZendeskShowAudit(Tool):
    name: str = "zendesk_show_audit"
    description: str | None = "Show a single ticket audit by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_audit(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            ticket_audit_id: int = Field(..., description="Ticket audit ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/tickets/{ticket_id}/audits/{ticket_audit_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_audit, **kwargs)


class ZendeskShowAuditLog(Tool):
    name: str = "zendesk_show_audit_log"
    description: str | None = "Show a single account audit log by ID (Enterprise plan)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_audit_log(
            audit_log_id: int = Field(..., description="Audit log ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/audit_logs/{audit_log_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_audit_log, **kwargs)


# --- Article ---


class ZendeskSearchArticles(Tool):
    name: str = "zendesk_search_articles"
    description: str | None = "Search Help Center articles."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_articles(
            query: str = Field(..., description="Search query for articles"),
            locale: str | None = Field(None, description="Locale (e.g. en-us)"),
            per_page: int = Field(25, description="Results per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"query": query, "per_page": per_page, "page": page}
            if locale:
                params["locale"] = locale

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/help_center/articles/search.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_articles, **kwargs)


# --- Attachment ---


class ZendeskDeleteUpload(Tool):
    name: str = "zendesk_delete_upload"
    description: str | None = "Delete an uploaded file by token."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_upload(
            token: str = Field(..., description="Upload token from upload response"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/uploads/{token}",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_upload, **kwargs)


class ZendeskRedactCommentAttachment(Tool):
    name: str = "zendesk_redact_comment_attachment"
    description: str | None = "Redact attachment URLs from a ticket comment."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _redact_attachment(
            ticket_id: int = Field(..., description="Zendesk ticket ID"),
            ticket_comment_id: int = Field(..., description="Ticket comment ID"),
            external_attachment_urls: list[str] = Field(..., description="Attachment URLs to redact"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/tickets/{ticket_id}/comments/{ticket_comment_id}/redact.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"external_attachment_urls": external_attachment_urls},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json() if response.content else {"status": "ok"}

        super().__init__(handler=_redact_attachment, **kwargs)


class ZendeskShowAttachment(Tool):
    name: str = "zendesk_show_attachment"
    description: str | None = "Show attachment details by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_attachment(
            attachment_id: int = Field(..., description="Attachment ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/attachments/{attachment_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_attachment, **kwargs)


class ZendeskUpdateAttachmentForMalware(Tool):
    name: str = "zendesk_update_attachment_for_malware"
    description: str | None = "Update attachment status after malware scan (safe or malware)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_attachment(
            attachment_id: int = Field(..., description="Attachment ID"),
            safe: bool = Field(..., description="True if safe, False if malware detected"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/attachments/{attachment_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"attachment": {"safe": safe}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_attachment, **kwargs)


class ZendeskUploadFiles(Tool):
    name: str = "zendesk_upload_files"
    description: str | None = "Upload files to Zendesk. Returns token for use in ticket/comment creation."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_files(
            filename: str = Field(..., description="Filename for the upload"),
            file_content_base64: str = Field(..., description="Base64-encoded file content"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import base64 as b64

            import httpx

            content = b64.b64decode(file_content_base64)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/uploads.json",
                    headers={
                        "Authorization": auth,
                        "Content-Type": "application/octet-stream",
                    },
                    params={"filename": filename},
                    content=content,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_upload_files, **kwargs)


# --- Approval ---


class ZendeskCreateApprovalRequest(Tool):
    name: str = "zendesk_create_approval_request"
    description: str | None = "Create an approval request for a ticket."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_approval(
            ticket_id: int = Field(..., description="Ticket ID"),
            assignee_user_id: int | None = Field(None, description="User ID of the approver"),
            assignee_group_id: int | None = Field(None, description="Group ID for group approval"),
            subject: str = Field(..., description="Approval subject"),
            message: str | None = Field(None, description="Approval message body"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            body_data: dict[str, Any] = {"ticket_id": ticket_id, "subject": subject}
            if assignee_user_id is not None:
                body_data["assignee_user_id"] = assignee_user_id
            elif assignee_group_id is not None:
                body_data["assignee_group_id"] = assignee_group_id
            else:
                raise ValueError("Either assignee_user_id or assignee_group_id is required")
            if message:
                body_data["message"] = message

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/approval_requests",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=body_data,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_approval, **kwargs)


class ZendeskCreateApprovalWorkflowInstance(Tool):
    name: str = "zendesk_create_approval_workflow_instance"
    description: str | None = "Create an approval workflow instance."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_workflow(
            ticket_id: int = Field(..., description="Ticket ID"),
            approval_workflow_id: int = Field(..., description="Approval workflow ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/approval_workflow_instances.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={
                        "approval_workflow_instance": {
                            "ticket_id": ticket_id,
                            "approval_workflow_id": approval_workflow_id,
                        }
                    },
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_workflow, **kwargs)


class ZendeskGetApprovalsByApprovalWorkflowId(Tool):
    name: str = "zendesk_get_approvals_by_approval_workflow_id"
    description: str | None = "Get approvals for an approval workflow."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_approvals(
            approval_workflow_id: int = Field(..., description="Approval workflow ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/approval_workflows/{approval_workflow_id}/approvals.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_approvals, **kwargs)


class ZendeskShowApprovalRequest(Tool):
    name: str = "zendesk_show_approval_request"
    description: str | None = "Show an approval request by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_approval(
            approval_request_id: int = Field(..., description="Approval request ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/approval_requests/{approval_request_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_approval, **kwargs)


class ZendeskUpdateApprovalRequestStatus(Tool):
    name: str = "zendesk_update_approval_request_status"
    description: str | None = "Update approval request status (approved or denied)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_status(
            approval_request_id: int = Field(..., description="Approval request ID"),
            status: str = Field(..., description="approved or denied"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/approval_requests/{approval_request_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"approval_request": {"status": status}},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_status, **kwargs)


# --- Agent ---


class ZendeskListAssignableAgentsFromGroup(Tool):
    name: str = "zendesk_list_assignable_agents_from_group"
    description: str | None = "List assignable agents (users) in a group."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_agents(
            group_id: int = Field(..., description="Group ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/{group_id}.json",
                    headers={"Authorization": auth},
                    params={"include": "users"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_agents, **kwargs)


class ZendeskListAssignableGroupsAndAgentsBasedOnSkill(Tool):
    name: str = "zendesk_list_assignable_groups_and_agents_based_on_skill"
    description: str | None = "List assignable groups and agents filtered by skill (skill-based routing)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_by_skill(
            skill_id: int = Field(..., description="Skill ID for filtering"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/routing/attributes/skills/{skill_id}/assignable_groups.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_by_skill, **kwargs)


class ZendeskListAssignableGroupsOnAssigneeField(Tool):
    name: str = "zendesk_list_assignable_groups_on_assignee_field"
    description: str | None = "List assignable groups for the assignee field (same as assignable groups)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_groups() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/groups/assignable.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_groups, **kwargs)


class ZendeskListBrandAgentMemberships(Tool):
    name: str = "zendesk_list_brand_agent_memberships"
    description: str | None = "List all brand agent memberships."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_memberships(
            per_page: int = Field(50, description="Records per page"),
            page: int = Field(1, description="Page number"),
            sort: str = Field("name", description="Sort field"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page, "sort": sort}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/brand_agents.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_memberships, **kwargs)


class ZendeskShowBrandAgentMembership(Tool):
    name: str = "zendesk_show_brand_agent_membership"
    description: str | None = "Show a brand agent membership by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_membership(
            brand_agent_id: str = Field(..., description="Brand agent membership ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/brand_agents/{brand_agent_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_membership, **kwargs)


# --- Address ---


class ZendeskCreateSupportAddress(Tool):
    name: str = "zendesk_create_support_address"
    description: str | None = "Create a support address."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_address(
            support_email: str = Field(..., description="Support email address"),
            name: str = Field(..., description="Name for the address"),
            brand_id: int | None = Field(None, description="Brand ID"),
            default: bool = Field(False, description="Set as default support address"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            address_data: dict[str, Any] = {"email": support_email, "name": name, "default": default}
            if brand_id is not None:
                address_data["brand_id"] = brand_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/recipient_addresses.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"recipient_address": address_data},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_address, **kwargs)


class ZendeskDeleteSupportAddress(Tool):
    name: str = "zendesk_delete_support_address"
    description: str | None = "Delete a support address."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_address(
            support_address_id: int = Field(..., description="Support address ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/recipient_addresses/{support_address_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted"}

        super().__init__(handler=_delete_address, **kwargs)


class ZendeskListSupportAddresses(Tool):
    name: str = "zendesk_list_support_addresses"
    description: str | None = "List all support addresses."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_addresses(
            per_page: int = Field(50, description="Records per page"),
            page: int = Field(1, description="Page number"),
            sort: str = Field("name", description="Sort field"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params = {"per_page": per_page, "page": page, "sort": sort}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/recipient_addresses.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_addresses, **kwargs)


class ZendeskShowSupportAddress(Tool):
    name: str = "zendesk_show_support_address"
    description: str | None = "Show a support address by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_address(
            support_address_id: int = Field(..., description="Support address ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/recipient_addresses/{support_address_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_address, **kwargs)


class ZendeskUpdateSupportAddress(Tool):
    name: str = "zendesk_update_support_address"
    description: str | None = "Update a support address."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_address(
            support_address_id: int = Field(..., description="Support address ID"),
            name: str | None = Field(None, description="Name for the address"),
            default: bool | None = Field(None, description="Set as default"),
            brand_id: int | None = Field(None, description="Brand ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            address_data: dict[str, Any] = {}
            if name is not None:
                address_data["name"] = name
            if default is not None:
                address_data["default"] = default
            if brand_id is not None:
                address_data["brand_id"] = brand_id

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/recipient_addresses/{support_address_id}.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"recipient_address": address_data},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_address, **kwargs)


class ZendeskVerifySupportAddressForwarding(Tool):
    name: str = "zendesk_verify_support_address_forwarding"
    description: str | None = "Verify email forwarding for a support address."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _verify_forwarding(
            support_address_id: int = Field(..., description="Support address ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/recipient_addresses/{support_address_id}/verify_forwarding.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_verify_forwarding, **kwargs)


# --- Account ---


class ZendeskShowSettings(Tool):
    name: str = "zendesk_show_settings"
    description: str | None = "Show account settings."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_settings() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/account/settings.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_settings, **kwargs)


class ZendeskUpdateAccountSettings(Tool):
    name: str = "zendesk_update_account_settings"
    description: str | None = "Update account settings."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_settings(
            settings: dict[str, Any] = Field(..., description="Settings to update (e.g. branding, tickets)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/account/settings.json",
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json={"settings": settings},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_settings, **kwargs)


# --- Activity ---


class ZendeskCountActivities(Tool):
    name: str = "zendesk_count_activities"
    description: str | None = "Count ticket activities affecting the requesting agent (last 30 days)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _count_activities() -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/activities/count.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_count_activities, **kwargs)


class ZendeskListActivities(Tool):
    name: str = "zendesk_list_activities"
    description: str | None = "List ticket activities affecting the requesting agent (last 30 days)."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_activities(
            per_page: int = Field(50, description="Records per page"),
            page: int = Field(1, description="Page number"),
            since: str | None = Field(None, description="ISO 8601 UTC time to return activities since"),
            sort: str = Field("-created_at", description="Sort field (prefix - for desc)"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page, "sort": sort}
            if since:
                params["since"] = since

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/activities.json",
                    headers={"Authorization": auth},
                    params=params,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_activities, **kwargs)


class ZendeskShowActivity(Tool):
    name: str = "zendesk_show_activity"
    description: str | None = "Show a single ticket activity by ID."
    integration: Annotated[str, Integration("zendesk")] | None = None
    subdomain: str | None = None
    email: SecretStr | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "subdomain": self.subdomain,
                "email": self.email,
                "api_token": self.api_token,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _show_activity(
            activity_id: int = Field(..., description="Activity ID"),
        ) -> Any:
            base_url, auth = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/activities/{activity_id}.json",
                    headers={"Authorization": auth},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_show_activity, **kwargs)
